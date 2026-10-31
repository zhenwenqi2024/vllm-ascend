# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for alias boundaries and the direct SFA launch contract."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/hisparse_direct.py"
_SPEC = importlib.util.spec_from_file_location("hisparse_direct_under_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
direct = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(direct)


@pytest.mark.parametrize("layer", [0, 1, 3])
def test_raw_alias_preserves_layer_offset_and_page_gaps(layer):
    backing = torch.arange(3 * 4 * 128 * 576, dtype=torch.float32).to(torch.bfloat16).view(3, 4, 128, 576)
    cache = backing[:, layer]
    raw, block, stride, capacity = direct.packed_cache_storage(cache)
    assert raw.data_ptr() == cache.data_ptr()
    assert raw.storage_offset() == cache.storage_offset()
    assert raw.numel() == 2 * backing.stride(0) + 128 * 576
    for row in (0, 127, 128, 255, capacity - 1):
        address = row // block * stride + row % block * 576
        torch.testing.assert_close(raw[address : address + 576], cache[row // block, row % block], rtol=0, atol=0)
    assert raw.storage_offset() + raw.numel() <= backing.numel()


def test_expanded_attention_view_alias_stays_within_tail_storage():
    backing = torch.zeros((4, 6, 128, 576), dtype=torch.bfloat16)
    cache = backing.view(-1, 128, 576)[3:]
    raw, block, stride, capacity = direct.packed_cache_storage(cache)
    assert block == 128 and stride == 128 * 576
    assert raw.numel() == capacity * 576
    assert raw.storage_offset() + raw.numel() == backing.numel()


@pytest.mark.parametrize("bad", ["dtype", "row_stride", "overlap", "empty"])
def test_reject_unsupported_cache_layouts(bad):
    cache = torch.zeros((2, 128, 576), dtype=torch.bfloat16)
    if bad == "dtype":
        cache = cache.float()
    elif bad == "row_stride":
        cache = cache[:, ::2]
    elif bad == "overlap":
        cache = cache.as_strided(cache.shape, (576, 576, 1))
    else:
        cache = cache[:0]
    with pytest.raises(ValueError):
        direct.packed_cache_storage(cache)


def test_compaction_preserves_order_duplicates_and_high_physical_ids():
    indices = torch.full((2, 128), -1, dtype=torch.int32)
    values = [2097151, -1, 0, 262144, 262144, 9, 3, 2097152, -7]
    indices[0, : len(values)] = torch.tensor(values, dtype=torch.int32)
    positions = torch.arange(128, dtype=torch.float32).view(1, -1)
    compact, counts = direct.compact_physical_indices(indices, 2097152, positions)
    assert counts.tolist() == [6, 0]
    assert compact[0, :6].tolist() == [2097151, 0, 262144, 262144, 9, 3]
    assert (compact[0, 6:] == -1).all() and (compact[1] == -1).all()


def test_full_prefill_issues_one_launch_without_materializing_kv(monkeypatch):
    rows, heads, topk = 8192, 8, 128
    query = torch.zeros((rows, heads, 512), dtype=torch.bfloat16)
    rope = torch.zeros((rows, heads, 64), dtype=torch.bfloat16)
    backing = torch.ones((2, 3, 128, 576), dtype=torch.bfloat16)
    cache = backing[:, 1]
    indices = torch.full((rows, topk), -1, dtype=torch.int32)
    indices[:, :2] = torch.tensor([255, 0], dtype=torch.int32)
    indices[-1] = -1
    positions = torch.arange(topk, dtype=torch.float32).view(1, -1)
    ends = torch.arange(1, rows + 1, dtype=torch.int32)
    calls = []

    def launch(q, raw, physical, query_ends, counts, qr, scale, block, stride, capacity):
        calls.append(q.shape[0])
        assert raw.data_ptr() == cache.data_ptr()
        assert raw.numel() == (cache.size(0) - 1) * cache.stride(0) + 128 * 576
        assert physical.shape == (rows, 1, topk)
        assert query_ends is ends and counts[-1] == 0 and counts[0] == 2
        assert (block, stride, capacity) == (128, cache.stride(0), 256)
        return torch.ones_like(q)

    monkeypatch.setattr(torch.ops.hisparse_direct, "sparse_flash_attention", launch, raising=False)
    monkeypatch.setattr(direct, "ensure_direct_operator", lambda: None)
    out = direct.packed_sparse_flash_attention(query, rope, cache, indices, 0.0625, positions, ends)
    assert calls == [8192]
    assert out[:-1].eq(1).all() and out[-1].eq(0).all()


def test_private_library_loads_once_and_rejects_different_registration(monkeypatch, tmp_path):
    library = tmp_path / "private.so"
    library.write_bytes(b"test fixture; never dlopened")
    other = tmp_path / "other.so"
    other.write_bytes(b"other fixture")
    namespace = SimpleNamespace()
    loaded = set()
    calls = []
    monkeypatch.setattr(torch.ops, "hisparse_direct", namespace)
    monkeypatch.setattr(torch.ops, "loaded_libraries", loaded)

    def fake_load(path):
        calls.append(path)
        loaded.add(path)
        namespace.sparse_flash_attention = object()

    monkeypatch.setattr(torch.ops, "load_library", fake_load)
    direct.load_direct_library(str(library))
    direct.load_direct_library(str(library))
    assert calls == [str(library.resolve())]
    with pytest.raises(RuntimeError, match="different library"):
        direct.load_direct_library(str(other))


def test_private_library_requires_existing_absolute_shared_object(tmp_path):
    with pytest.raises(ValueError):
        direct.load_direct_library("relative.so")
    with pytest.raises(ValueError):
        direct.load_direct_library(str(tmp_path / "missing.so"))
