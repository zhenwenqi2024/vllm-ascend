# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed direct SFA must match the preserved materializing bridge."""

import sys
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401


@pytest.fixture(autouse=True)
def normal_native_registration():
    """Exercise the production registration, never a second private binding."""
    if hasattr(torch.ops.hisparse_direct, "sparse_flash_attention") and "vllm_ascend.vllm_ascend_C" not in sys.modules:
        raise RuntimeError("A private binding is already loaded; run normal-build tests in a fresh process")
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env()
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    from vllm_ascend.ops.hisparse_direct import ensure_direct_operator

    ensure_direct_operator()


@pytest.mark.parametrize("rows", [1, 32, 8192])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("capture", [False, True])
def test_direct_sfa_matches_bridge_and_cpu_reference(rows, strided, capture):
    if not torch.npu.is_available() or not torch.npu.get_device_name(0).startswith("Ascend950"):
        pytest.skip("Packed direct SFA initially targets Ascend 950")
    torch.npu.set_device(0)
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env()
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    from vllm_ascend.attention.sfa_hisparse import AscendHiSparseImpl
    from vllm_ascend.ops.hisparse_direct import packed_sparse_flash_attention

    torch.manual_seed(71)
    device, topk, heads, block, pages = "npu:0", 2048, 8, 128, 17
    variants = min(rows, 4)
    q_cpu = torch.randn(variants, heads, 512).to(torch.bfloat16)
    qr_cpu = torch.randn(variants, heads, 64).to(torch.bfloat16)
    kv_cpu = torch.randn(pages, block, 576).to(torch.bfloat16)
    ids_cpu = torch.randint(0, pages * block, (variants, topk), dtype=torch.int32)
    ids_cpu[:, :6] = torch.tensor([2175, 0, 129, 128, 129, 129], dtype=torch.int32)
    if variants > 1:
        ids_cpu[1] = -1
        ids_cpu[1, :9] = torch.tensor([2175, -1, 2, 2175, 0, -1, 128, 7, 3])
    if variants > 2:
        ids_cpu[2] = -1
    repeat = (rows + variants - 1) // variants
    query = q_cpu.repeat(repeat, 1, 1)[:rows].to(device)
    query_rope = qr_cpu.repeat(repeat, 1, 1)[:rows].to(device)
    physical = ids_cpu.repeat(repeat, 1)[:rows].to(device)
    if strided:
        backing = torch.full((pages, 3, block, 576), 31, dtype=torch.bfloat16, device=device)
        cache = backing[:, 1]
        cache.copy_(kv_cpu)
    else:
        cache = kv_cpu.to(device)
    positions = torch.arange(topk, dtype=torch.float32, device=device).view(1, -1)
    ends = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
    oracle_workspace = (
        torch.empty((32, topk, 1, 512), dtype=torch.bfloat16, device=device),
        torch.empty((32, topk, 1, 64), dtype=torch.bfloat16, device=device),
        torch.arange(32 * topk // block, dtype=torch.int32, device=device).view(32, -1),
        torch.arange(topk, dtype=torch.int32, device=device).view(1, 1, -1),
        torch.arange(1, 33, dtype=torch.int32, device=device),
        torch.full((32,), topk, dtype=torch.int32, device=device),
    )
    oracle = SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64, scale=0.0625, _workspace=lambda _: oracle_workspace)

    def forward():
        return packed_sparse_flash_attention(query, query_rope, cache, physical, 0.0625, positions, ends)

    graph = None
    try:
        if capture:
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                forward()
                forward()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                actual = forward()
            # Replay with a different valid-count pattern and source payload.
            if variants > 1:
                ids_cpu[1, :9] = torch.tensor([-1, 3, 3, 128, -1, 2175, 9, 1, 0])
                physical.copy_(ids_cpu.repeat(repeat, 1)[:rows])
            q_cpu.neg_()
            query.copy_(q_cpu.repeat(repeat, 1, 1)[:rows])
            for _ in range(3):
                graph.replay()
        else:
            actual = forward()
        reference = AscendHiSparseImpl._attention_physical_bridge(
            oracle, query[:variants], query_rope[:variants], cache, physical[:variants]
        )
        torch.testing.assert_close(actual, reference.repeat(repeat, 1, 1)[:rows], rtol=0, atol=0)

        selected = kv_cpu.flatten(0, 1)[ids_cpu.clamp_min(0).long()].float()
        scores = torch.einsum("qhd,qkd->qhk", q_cpu.float(), selected[..., :512])
        scores += torch.einsum("qhd,qkd->qhk", qr_cpu.float(), selected[..., 512:])
        scores = (scores * 0.0625).masked_fill((ids_cpu < 0)[:, None, :], -torch.inf)
        expected = torch.einsum("qhk,qkd->qhd", scores.softmax(-1).nan_to_num(0), selected[..., :512])
        torch.testing.assert_close(actual[:variants].float().cpu(), expected, atol=0.005, rtol=0.025)
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()


@pytest.mark.parametrize("rows", [1, 32])
def test_direct_sfa_addresses_beyond_eight_gib(rows):
    """Exercise uint64 page products without initializing an 8-GiB tensor."""
    if not torch.npu.is_available() or not torch.npu.get_device_name(0).startswith("Ascend950"):
        pytest.skip("Packed direct SFA initially targets Ascend 950")
    torch.npu.set_device(0)
    from vllm_ascend.ops.hisparse_direct import packed_sparse_flash_attention

    torch.manual_seed(79)
    block, width, heads, topk = 128, 576, 8, 2048
    stride = (1 << 32) + block * width  # BF16 elements: second page is above 8 GiB.
    offset = 3 * width
    span = offset + stride + block * width
    backing = cache = None
    try:
        backing = torch.empty(span, dtype=torch.bfloat16, device="npu:0")
        cache = torch.as_strided(backing, (2, block, width), (stride, width, 1), storage_offset=offset)
        source = torch.randn(2, block, width).to(torch.bfloat16)
        # Write exactly the two semantic pages. Gaps are deliberately unread.
        cache[0].copy_(source[0])
        cache[1].copy_(source[1])
        small = source.to("npu:0")
        query_cpu = torch.randn(rows, heads, 512).to(torch.bfloat16)
        rope_cpu = torch.randn(rows, heads, 64).to(torch.bfloat16)
        query, rope = query_cpu.to("npu:0"), rope_cpu.to("npu:0")
        indices = torch.full((rows, topk), -1, dtype=torch.int32, device="npu:0")
        # Preserve descending pairs and duplicate entries on both physical pages.
        chosen = torch.tensor([255, 0, 128, 127, 128, 255, 42], dtype=torch.int32)
        indices[:, : chosen.numel()] = chosen.to("npu:0")
        positions = torch.arange(topk, dtype=torch.float32, device="npu:0").view(1, -1)
        ends = torch.arange(1, rows + 1, dtype=torch.int32, device="npu:0")
        actual = packed_sparse_flash_attention(query, rope, cache, indices, 0.0625, positions, ends)
        compact = packed_sparse_flash_attention(query, rope, small, indices, 0.0625, positions, ends)
        torch.testing.assert_close(actual, compact, rtol=0, atol=0)
        selected = source.flatten(0, 1)[chosen.long()].float()
        logits = torch.einsum("qhd,kd->qhk", query_cpu.float(), selected[:, :512])
        logits += torch.einsum("qhd,kd->qhk", rope_cpu.float(), selected[:, 512:])
        expected = torch.einsum("qhk,kd->qhd", (logits * 0.0625).softmax(-1), selected[:, :512])
        torch.testing.assert_close(actual.float().cpu(), expected, atol=0.005, rtol=0.025)
    finally:
        torch.npu.synchronize()
        cache = None
        backing = None
        torch.npu.empty_cache()


@pytest.mark.parametrize("counts", [(0,), (0,) * 32, (0, 1, 127, 128, 129, 2048)])
@pytest.mark.parametrize("capture", [False, True])
def test_direct_sfa_empty_and_tile_boundary_counts(counts, capture):
    if not torch.npu.is_available() or not torch.npu.get_device_name(0).startswith("Ascend950"):
        pytest.skip("Packed direct SFA initially targets Ascend 950")
    torch.npu.set_device(0)
    from vllm_ascend.ops.hisparse_direct import packed_sparse_flash_attention

    torch.manual_seed(83)
    rows, topk, heads = len(counts), 2048, 8
    query_cpu = torch.randn(rows, heads, 512).to(torch.bfloat16)
    rope_cpu = torch.randn(rows, heads, 64).to(torch.bfloat16)
    source = torch.randn(16, 128, 576).to(torch.bfloat16)
    query, rope, cache = query_cpu.to("npu:0"), rope_cpu.to("npu:0"), source.to("npu:0")
    chosen = torch.arange(topk - 1, -1, -1, dtype=torch.int32)
    indices_cpu = torch.full((rows, topk), -1, dtype=torch.int32)
    for row, count in enumerate(counts):
        indices_cpu[row, :count] = chosen[:count]
    indices = indices_cpu.to("npu:0")
    positions = torch.arange(topk, dtype=torch.float32, device="npu:0").view(1, -1)
    ends = torch.arange(1, rows + 1, dtype=torch.int32, device="npu:0")

    def forward():
        return packed_sparse_flash_attention(query, rope, cache, indices, 0.0625, positions, ends)

    graph = None
    try:
        if capture:
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                forward()
                forward()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                actual = forward()
            # Zero-only cases remain zero; the mixed case changes per-row lengths.
            if any(counts):
                counts = tuple(reversed(counts))
                indices_cpu.fill_(-1)
                for row, count in enumerate(counts):
                    indices_cpu[row, :count] = chosen[:count]
                indices.copy_(indices_cpu)
            for _ in range(3):
                graph.replay()
        else:
            actual = forward()
        expected = torch.zeros_like(query_cpu, dtype=torch.float32)
        for row, count in enumerate(counts):
            if count:
                selected = source.flatten(0, 1)[chosen[:count].long()].float()
                score = query_cpu[row].float() @ selected[:, :512].T
                score += rope_cpu[row].float() @ selected[:, 512:].T
                expected[row] = (score * 0.0625).softmax(-1) @ selected[:, :512]
        torch.testing.assert_close(actual.float().cpu(), expected, atol=0.005, rtol=0.025)
        for row, count in enumerate(counts):
            if count == 0:
                assert torch.count_nonzero(actual[row].cpu()) == 0
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
