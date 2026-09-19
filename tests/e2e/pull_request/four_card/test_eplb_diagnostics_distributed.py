# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual graph replay with independent EP groups, idle sources and live maps."""

import io
import logging
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("torch_npu")

from vllm_ascend.eplb.diagnostics.config import EplbDiagnosticsConfig
from vllm_ascend.eplb.diagnostics.probe import ExpertLoadProbe
from vllm_ascend.eplb.diagnostics.runtime import DiagnosticsRecorder


@torch.inference_mode()
def _worker(rank, rendezvous):
    torch.npu.set_device(rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=90))
    try:
        members = ([0, 2], [1, 3])
        groups = [dist.new_group(ranks, backend="gloo") for ranks in members]
        stage = rank % 2
        local = members[stage].index(rank)
        group = SimpleNamespace(ranks=members[stage], rank_in_group=local, cpu_group=groups[stage])
        source = torch.tensor(0, device="npu")
        slot = torch.zeros(1, dtype=torch.int64, device="npu")
        layers = []
        for name in ("layer.0", "layer.1"):
            probe = ExpertLoadProbe(4, "npu")
            probe.eplb_enabled = True
            probe.source_token_count, probe.source_positions = source, torch.arange(4, device="npu")
            probe.logical_to_physical = torch.arange(4, device="npu")
            probe.owner_totals = torch.zeros((2, 4), dtype=torch.int64, device="npu")
            probe.history = torch.zeros((2, 6), dtype=torch.int64, device="npu")
            probe.history_slot = slot
            probe.comm_history = torch.zeros(2, dtype=torch.int64, device="npu")
            layers.append(
                (
                    name,
                    SimpleNamespace(
                        eplb_diagnostic_probe=probe,
                        log2phy=probe.logical_to_physical,
                        moe_config=SimpleNamespace(experts_per_token=1),
                    ),
                )
            )
        ids = torch.tensor([[0], [0], [1], [99]], device="npu")
        graph = torch.npu.NPUGraph()
        for _, layer in layers:
            layer.eplb_diagnostic_probe.record_routes(ids)
        torch.npu.synchronize()
        with torch.npu.graph(graph):
            for _, layer in layers:
                layer.eplb_diagnostic_probe.record_routes(ids)
                layer.eplb_diagnostic_probe.record_comm(1)
        recorder = DiagnosticsRecorder(
            EplbDiagnosticsConfig(mode="benefit", window_size=2, warmup_steps=0),
            layers,
            group,
            [rank],
            stage,
            source,
        )
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        logger = logging.getLogger("vllm.eplb.diagnostics")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            for step in range(5):
                recorder.begin(3, dummy=local == 1)
                if step == 1:
                    for _, layer in layers:
                        layer.log2phy.copy_(torch.tensor([2, 0, 1, 3], device="npu"))
                        recorder.committed(layer)
                    ids[:3].copy_(torch.tensor([[2], [2], [0]], device="npu"))
                if local == 0:
                    recorder.metadata[-1] = {"phase": "decode", "graph_mode": "FULL"}
                graph.replay()
                recorder.end()
            recorder.finish()
            recorder.finish()
            if local == 0:
                assert output.getvalue().count("placement_changed_during_window") == 2
                assert output.getvalue().count("[EPLB experts]") == 12
                assert output.getvalue().count("[EPLB summary]") == 1
                assert recorder.jobs["layer.0"]["loads"] == [[2, 1, 0, 0]]
                assert recorder.jobs["layer.0"]["current_placement"] == ((1, 2), (0, 3))
                assert recorder.jobs["layer.0"]["initial_placement"] == ((0, 1), (2, 3))
                assert recorder.jobs["layer.0"]["generation"] == 1
                assert recorder.logical_totals["layer.0"] == {0: 10, 1: 5, 2: 0, 3: 0}
                assert recorder.expert_totals["layer.0"][0] == {0: 2, 1: 5, 2: 0, 3: 0}
                assert recorder.expert_totals["layer.0"][1] == {0: 8, 1: 0, 2: 0, 3: 0}
            else:
                assert not output.getvalue()
        finally:
            logger.removeHandler(handler)
    finally:
        dist.destroy_process_group()


def test_graph_live_placement_with_idle_sources_and_independent_groups(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "gloo-init").as_uri(),), nprocs=4, join=True)
