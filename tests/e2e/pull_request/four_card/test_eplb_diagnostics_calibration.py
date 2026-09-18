# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scratch calibration on two independent EP-like pairs, using four NPUs."""

import io
import logging
import math
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("torch_npu")

from vllm_ascend.eplb.diagnostics.scratch import measure_overheads

_SOURCE = ((0, 1), (2, 3))
_SCRATCH_LIMIT = 1024 * 1024


def _worker(rank, rendezvous):
    torch.npu.set_device(rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=90))
    try:
        # Noncontiguous ranks ensure P2P peers are global, not local group IDs.
        members = ([0, 2], [1, 3])
        cpu_groups = [dist.new_group(pair, backend="gloo") for pair in members]
        device_groups = [dist.new_group(pair, backend="hccl") for pair in members]
        stage, local = rank % 2, rank // 2
        group = SimpleNamespace(
            ranks=members[stage],
            rank_in_group=local,
            cpu_group=cpu_groups[stage],
            device_group=device_groups[stage],
            device=torch.device(f"npu:{rank}"),
        )
        candidate = ((0, 2), (1, 3)) if stage == 0 else ((2, 3), (0, 1))
        result = measure_overheads(group, _SOURCE, candidate, 4096, repeats=2, max_scratch_bytes=_SCRATCH_LIMIT)
        assert result["reason"] is None, result
        assert result["moved_experts"] == (2 if stage == 0 else 4)
        assert result["scratch_bytes"] <= _SCRATCH_LIMIT
        for field in ("collection_ms", "transfer_ms", "apply_ms"):
            timing = result[field]
            assert timing["samples"] == 2
            assert 0 <= timing["min_ms"] <= timing["median_ms"] <= timing["max_ms"]
            assert math.isfinite(timing["max_ms"])

        # A refusal on one participant must be observed by its peer before HCCL.
        result = measure_overheads(
            group,
            _SOURCE,
            candidate,
            4096,
            repeats=1,
            max_scratch_bytes=1 if local == 0 else _SCRATCH_LIMIT,
        )
        assert result["reason"] == "scratch_memory_cap_exceeded"
        result = measure_overheads(
            group,
            _SOURCE,
            candidate if local == 0 else _SOURCE,
            4096,
            repeats=1,
            max_scratch_bytes=_SCRATCH_LIMIT,
        )
        assert result["reason"] == "inconsistent_calibration_parameters"
        # Both groups can still enter a valid device collective after refusals.
        result = measure_overheads(group, _SOURCE, _SOURCE, 4096, repeats=1, max_scratch_bytes=_SCRATCH_LIMIT)
        assert result["reason"] is None and result["moved_experts"] == 0
        assert result["transfer_ms"]["max_ms"] == 0
        _check_real_assessment(group, stage)
    finally:
        dist.destroy_process_group()


def _check_real_assessment(group, stage):
    # Real model hooks + real collectives, using tiny weights and synthetic
    # route traces. This is functional coverage, not full-model performance.
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from vllm_ascend.eplb.diagnostics.assessment import calibrate
    from vllm_ascend.eplb.diagnostics.model_calibration import capture_template
    from vllm_ascend.eplb.diagnostics.placement import build_plan
    from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights
    from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
    from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
    from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod

    layer = torch.nn.Module()
    layer.w13_weight = torch.full((2, 128, 256), 0.01, dtype=torch.bfloat16, device=group.device)
    layer.w2_weight = torch.full((2, 128, 128), 0.01, dtype=torch.bfloat16, device=group.device)
    method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
    torch.nn.Module.__init__(method)
    method.moe = SimpleNamespace(has_bias=False)
    method._lora_routing = method.lora_context = None
    method.dynamic_eplb = False
    probe = SimpleNamespace(calibration_templates={})
    payload = MoEMlpComputeInput(
        hidden_states=torch.ones((64, 128), dtype=torch.bfloat16, device=group.device),
        group_list=torch.tensor([32, 32], dtype=torch.int64, device=group.device),
        group_list_type=1,
        dynamic_scale=None,
        topk_scales=None,
        weights=MoEWeights(w1=layer.w13_weight, w2=layer.w2_weight),
        quant=MoEQuantParams(),
        fusion=False,
        layer=layer,
        activation=MoEActivation.SILU,
    )
    assert capture_template(probe, payload, method, "MC2CommImpl") is None
    plan, reason = build_plan(
        [{0: 100, 1: 100}, {2: 1, 3: 1}],
        [[0, 1, -1, -1], [-1, -1, 0, 1]],
        1,
        "decode",
    )
    assert reason is None
    job = {
        "plan": plan,
        "window": 2,
        "loads": [[48, 8, 4, 4], [24, 4, 2, 2]],
        "metadata": [[{"phase": "decode", "graph_mode": "FULL", "padded_tokens": 64}] * 2 for _ in group.ranks],
        "comm_codes": [[1, 1] for _ in group.ranks],
    }
    recorder = SimpleNamespace(
        group=group,
        summary=SimpleNamespace(stage=stage, layers={"layer.0": SimpleNamespace(calibration_job=job)}),
        layers=[("layer.0", SimpleNamespace(eplb_diagnostic_probe=probe))],
        finished=False,
        step=4,
    )
    recorder.finish = lambda: setattr(recorder, "finished", True)
    output = io.StringIO()
    logger = logging.getLogger("vllm.eplb.diagnostics")
    handler = logging.StreamHandler(output)
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        calibrate(recorder, update_interval=8, samples=2, repeats=2, max_scratch_bytes=_SCRATCH_LIMIT)
        assert recorder.calibrated, output.getvalue()
        assert recorder.finished
        if group.rank_in_group == 0:
            text = output.getvalue()
            assert "calibrated_layers=1 available_layers=1" in text
            assert "baseline_mlp_ms=" in text and "transfer_per_update_ms=" in text
            assert "conclusion=insufficient_benefit_evidence net_saving_ms=unknown" in text
            print(text, flush=True)
        calibrate(recorder, update_interval=8, samples=2, repeats=2, max_scratch_bytes=_SCRATCH_LIMIT)
        if group.rank_in_group == 0:
            assert "reason=already_calibrated" in output.getvalue()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


@pytest.mark.skipif(not torch.npu.is_available() or torch.npu.device_count() < 4, reason="requires four NPUs")
def test_scratch_calibration_in_independent_ep_groups(tmp_path):
    rendezvous = (tmp_path / "calibration-gloo-init").as_uri()
    mp.spawn(_worker, args=(rendezvous,), nprocs=4, join=True)
