# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-NPU graph/buffer regression. Run on actual Ascend hardware before review."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")

from vllm_ascend.eplb_diagnostics.config import EplbDiagnosticsConfig
from vllm_ascend.eplb_diagnostics.probe import ExpertLoadProbe
from vllm_ascend.eplb_diagnostics.report import load_records
from vllm_ascend.eplb_diagnostics.runtime import DiagnosticsRecorder, NpuSnapshotBackend


@pytest.mark.parametrize("group_type", [0, 1])
def test_graph_replay_updates_probe_without_python_reentry(group_type, tmp_path):
    torch.npu.set_device(0)
    probe = ExpertLoadProbe(2, "npu", 2)
    probe.source_token_count = torch.tensor(2, device="npu")
    probe.source_positions = torch.arange(3, device="npu")
    source = torch.tensor([2, 5] if group_type == 0 else [2, 3], dtype=torch.int64, device="npu")
    route_ids = torch.tensor([[0], [1], [1]], device="npu")
    # Communication masks can include graph padding. Scheduler validity wins.
    route_mask = torch.tensor([True, True, True], device="npu")
    probe.record(source, group_type, "MC2CommImpl")
    probe.record_routes(route_ids, route_mask)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        probe.record(source, group_type, "MC2CommImpl")
        probe.record_routes(route_ids, route_mask)
    graph.replay()
    torch.npu.synchronize()
    pointer = probe.totals.data_ptr()
    route_pointer = probe.route_totals.data_ptr()
    layer = SimpleNamespace(
        eplb_diagnostic_probe=probe,
        _use_v2_model_runner=False,
        dynamic_eplb=False,
        moe_config=SimpleNamespace(num_logical_experts=2, ep_size=1, ep_rank=0),
        n_shared_experts=0,
        mix_placement=False,
        log2phy=None,
        router=None,
        ascend_expert_map=torch.tensor([0, 1]),
    )
    config = EplbDiagnosticsConfig(
        mode="observe",
        run_id="graph-replay",
        output_dir=str(tmp_path),
        sample_interval=1,
        warmup_steps=0,
        max_samples=2,
        max_pending=4,
    )
    recorder = DiagnosticsRecorder(config, [("layer", layer)], {"rank": 0}, NpuSnapshotBackend(torch.device("npu:0")))
    try:
        # All host copies/events are outside the captured graph.
        assert recorder.begin({"graph_mode": "FULL"})
        graph.replay()
        recorder.end()
        # A captured graph still runs for DP participation, but its synthetic
        # work must not be copied, exported, or consume the two-real-sample budget.
        assert not recorder.begin({"dummy_run": True, "scheduled_tokens": 0, "graph_mode": "FULL"})
        source.copy_(torch.tensor([900, 1000] if group_type == 0 else [900, 100], device="npu"))
        graph.replay()
        recorder.end()
        assert recorder.collected == 1
        source.copy_(torch.tensor([7, 8] if group_type == 0 else [7, 1], device="npu"))
        route_ids.copy_(torch.tensor([[0], [0], [1]], device="npu"))
        probe.source_token_count.fill_(3)
        assert recorder.begin({"graph_mode": "FULL"})
        graph.replay()
        recorder.end()
    finally:
        recorder.close()
    rows = list(load_records([tmp_path]))
    assert [r["layers"][0]["expert_assignments"] for r in rows] == [[2, 3], [7, 1]]
    assert all(r["layers"][0]["calls"] == 1 for r in rows)
    assert [r["layers"][0]["routing"]["physical_assignments"] for r in rows] == [[1, 1], [2, 1]]
    assert all(r["layers"][0]["routing"]["semantics"] == "valid_scheduled_mc2_source_destinations" for r in rows)
    assert all(r["execute_stream_ms"] >= 0 for r in rows)
    assert probe.totals.data_ptr() == pointer
    assert probe.route_totals.data_ptr() == route_pointer
