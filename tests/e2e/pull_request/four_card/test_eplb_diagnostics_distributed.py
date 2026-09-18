# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Four NPU processes: independent PP-like groups and idle DP participation."""

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
        groups = [dist.new_group(members, backend="gloo") for members in ([0, 1], [2, 3])]
        stage, local = divmod(rank, 2)
        group = SimpleNamespace(ranks=[stage * 2, stage * 2 + 1], rank_in_group=local, cpu_group=groups[stage])
        probe = ExpertLoadProbe(4, "npu")
        opposite = ExpertLoadProbe(4, "npu")
        source = torch.tensor(3, device="npu")
        probe.source_token_count, probe.source_positions = source, torch.arange(4, device="npu")
        opposite.source_token_count, opposite.source_positions = source, probe.source_positions
        ids = torch.zeros((4, 1), dtype=torch.int64, device="npu")
        other_ids = torch.full_like(ids, 2)
        mask = torch.ones(4, dtype=torch.bool, device="npu")
        probe.record_routes(ids, mask)
        opposite.record_routes(other_ids, mask)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            probe.record_routes(ids, mask)
            opposite.record_routes(other_ids, mask)
        graph.replay()
        torch.npu.synchronize()
        layer = SimpleNamespace(
            eplb_diagnostic_probe=probe,
            local_num_experts=2,
            ascend_expert_map=torch.tensor([0, 1, -1, -1] if local == 0 else [-1, -1, 0, 1]),
            moe_config=SimpleNamespace(experts_per_token=1),
        )
        config = EplbDiagnosticsConfig(mode="observe", window_size=2, warmup_steps=0, max_windows=0)
        other_layer = SimpleNamespace(**dict(vars(layer), eplb_diagnostic_probe=opposite))
        recorder = DiagnosticsRecorder(
            config, [("layer.0", layer), ("layer.1", other_layer)], group, [rank], stage, source
        )
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        logger = logging.getLogger("vllm.eplb.diagnostics")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            for _ in range(2):
                for tokens, dummy in ((3, False), (2, local == 1)):
                    recorder.begin(tokens, dummy)
                    if not dummy:
                        recorder.phases.add(("decode", "FULL"))
                    graph.replay()
                    recorder.end()
            recorder.begin(1, dummy=local == 1)
            graph.replay()
            recorder.end()
            recorder.finish()
            recorder.finish()
            if local == 0:
                assert output.getvalue().count("[EPLB diagnostic]") == 6
                assert output.getvalue().count("[EPLB experts]") == 12
                assert "skewed_layers=['layer.0', 'layer.1']" in output.getvalue()
                assert "cumulative_work=17" in output.getvalue()
                assert output.getvalue().count("final=True") == 1
                assert "valid_assignments=8 rank_work=[8, 0]" in output.getvalue()
                assert "hint=persistent_load_skew" in output.getvalue()
                assert f"stage={stage}" in output.getvalue()
            else:
                assert not output.getvalue()
        finally:
            logger.removeHandler(handler)
    finally:
        dist.destroy_process_group()


def test_graph_window_gather_with_idle_dp_and_separate_stages(tmp_path):
    rendezvous = (tmp_path / "gloo-init").as_uri()
    mp.spawn(_worker, args=(rendezvous,), nprocs=4, join=True)
