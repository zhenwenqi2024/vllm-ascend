# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable checks for live EPLB placement generations and valid source work."""

import importlib
import logging
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def api():
    stubbed = "vllm_ascend" not in sys.modules and importlib.util.find_spec("vllm") is None
    original = set(sys.modules)
    if stubbed:
        package = ModuleType("vllm_ascend")
        package.__path__ = [str(Path(__file__).resolve().parents[3] / "vllm_ascend")]
        sys.modules["vllm_ascend"] = package
    yield SimpleNamespace(
        **{
            name: importlib.import_module(f"vllm_ascend.eplb.diagnostics.{name}")
            for name in ("config", "probe", "runtime")
        }
    )
    if stubbed:
        for name in set(sys.modules) - original:
            if name == "vllm_ascend" or name.startswith("vllm_ascend."):
                del sys.modules[name]


def rows():
    return [
        {
            "rank": rank,
            "tp_ranks": [rank],
            "step": 2,
            "token_steps": [4, 2],
            "metadata": [{"phase": "decode", "graph_mode": "FULL"}] * 2,
            "layers": {
                "layer": {
                    "initial": [0, 1, 2, 3],
                    "current": [0, 2, 1, 3],
                    "generation_start": 1,
                    "generation_end": 1,
                    "top_k": 1,
                    "history": [[3, 1, 0, 0, 1, 0], [0, 2, 0, 0, 1, 0]],
                    "comm_codes": [1, 1],
                }
            },
        }
        for rank in (2, 5)
    ]


def test_actual_layouts_use_same_logical_work(api):
    job, reason = api.runtime.validate_window(rows(), "layer", 2)
    assert reason is None
    assert job["initial_placement"] == ((0, 1), (2, 3))
    assert job["current_placement"] == ((0, 2), (1, 3))
    assert job["loads"] == [[6, 2, 0, 0], [0, 4, 0, 0]]


@pytest.mark.parametrize(
    "case,reason",
    [
        ("commit", "placement_changed_during_window"),
        ("generation", "unaligned_placement_generations"),
        ("mapping", "placement_mismatch"),
        ("replica", "unsupported_placement"),
        ("calls", "invalid_route_history"),
        ("invalid_id", "invalid_route_history"),
        ("negative", "invalid_route_history"),
        ("tokens", "source_conservation_failure"),
        ("missing_source", "source_tp_mismatch"),
        ("step", "unaligned_ranks"),
        ("missing_layer", "missing_layer"),
    ],
)
def test_incomparable_or_invalid_work_never_becomes_benefit(api, case, reason):
    data = rows()
    layer = data[0]["layers"]["layer"]
    if case == "commit":
        layer["generation_end"] = 2
    elif case == "generation":
        layer["generation_start"] = layer["generation_end"] = 2
    elif case == "mapping":
        layer["current"] = [0, 1, 2, 3]
    elif case == "replica":
        for row in data:
            row["layers"]["layer"]["current"] = [0, 0, 2, 3]
    elif case == "calls":
        layer["history"][0][-2] = 2
    elif case == "invalid_id":
        layer["history"][0][-1] = 1
    elif case == "negative":
        layer["history"][0][0] = -1
    elif case == "tokens":
        data[0]["token_steps"][0] = 3
    elif case == "missing_source":
        data[0]["tp_ranks"] = [2, 7]
    elif case == "step":
        data[0]["step"] = 3
    elif case == "missing_layer":
        data[0]["layers"] = {}
    assert api.runtime.validate_window(data, "layer", 2) == (None, reason)


def test_idle_source_participates_without_fake_routes(api):
    data = rows()
    data[1]["token_steps"] = [0, 0]
    data[1]["layers"]["layer"]["history"] = [[0, 0, 0, 0, 1, 0]] * 2
    assert api.runtime.validate_window(data, "layer", 2)[0]["loads"][0] == [3, 1, 0, 0]


def test_off_observation_mode_removed(api):
    with pytest.raises(ValueError):
        api.config.EplbDiagnosticsConfig(mode="observe")
    assert api.config.EplbDiagnosticsConfig(mode="benefit").mode == "benefit"


def test_live_mapping_changes_keep_logical_expert_counters(api):
    probe = api.probe.ExpertLoadProbe(4, "cpu")
    probe.source_token_count = torch.tensor(2)
    probe.source_positions = torch.arange(3)
    probe.logical_to_physical = torch.arange(4)
    probe.owner_totals = torch.zeros((2, 4), dtype=torch.int64)
    ids = torch.tensor([[0], [1], [99]])
    probe.record_routes(ids)
    probe.logical_to_physical.copy_(torch.tensor([2, 0, 1, 3]))
    ids[:2] = torch.tensor([[2], [0]])
    probe.record_routes(ids)
    probe.source_token_count.zero_()
    probe.record_routes(ids)
    assert probe.totals.tolist() == [2, 2, 0, 0, 3, 0]
    assert probe.owner_totals.tolist() == [[1, 2, 0, 0], [1, 0, 0, 0]]


def make_recorder(api, monkeypatch, *, warmup=0, limit=0):
    probe = api.probe.ExpertLoadProbe(2, "cpu")
    source = torch.tensor(0)
    probe.source_token_count, probe.source_positions = source, torch.arange(3)
    probe.logical_to_physical = torch.arange(2)
    probe.owner_totals = torch.zeros((1, 2), dtype=torch.int64)
    probe.history = torch.zeros((2, 4), dtype=torch.int64)
    probe.history_slot = torch.zeros(1, dtype=torch.int64)
    probe.comm_history = torch.zeros(2, dtype=torch.int64)
    layer = SimpleNamespace(
        log2phy=probe.logical_to_physical,
        eplb_diagnostic_probe=probe,
        moe_config=SimpleNamespace(experts_per_token=1),
    )
    group = SimpleNamespace(ranks=[0], rank_in_group=0, cpu_group=None)
    monkeypatch.setattr(api.runtime, "gather", lambda group, row: [deepcopy(row)])
    config = api.config.EplbDiagnosticsConfig(mode="benefit", window_size=2, warmup_steps=warmup, max_windows=limit)
    recorder = api.runtime.DiagnosticsRecorder(config, [("layer", layer)], group, [0], 0, source)
    return recorder, probe, layer


def run_step(recorder, probe, tokens=2, dummy=False):
    recorder.begin(tokens, dummy)
    probe.record_routes(torch.tensor([[0], [1], [99]]))
    probe.record_comm(1)
    if recorder.collecting and recorder.real:
        recorder.metadata[-1] = {"phase": "decode", "graph_mode": "FULL"}
    recorder.end()


def test_warmup_dummy_cap_and_finalization(api, monkeypatch):
    recorder, probe, _ = make_recorder(api, monkeypatch, warmup=1, limit=1)
    run_step(recorder, probe)
    run_step(recorder, probe)
    run_step(recorder, probe, dummy=True)
    run_step(recorder, probe)
    recorder.finish()
    assert recorder.observed_steps == 1
    assert recorder.jobs["layer"]["loads"] == [[1, 1], [0, 0]]
    assert not recorder.monitor.active and probe.source_token_count == 0
    saved = deepcopy(recorder.jobs)
    recorder.finish()
    assert recorder.jobs == saved


def test_cross_commit_invalidates_job_then_new_stable_window_recovers(api, monkeypatch):
    recorder, probe, layer = make_recorder(api, monkeypatch)
    run_step(recorder, probe)
    recorder.committed(layer)
    probe.logical_to_physical.copy_(torch.tensor([1, 0]))
    run_step(recorder, probe)
    assert not recorder.jobs
    assert recorder.invalid_windows["layer"] == 1
    assert recorder.logical_totals["layer"] == {0: 2, 1: 2}
    assert recorder.expert_totals["layer"][0] == {0: 2, 1: 2}
    run_step(recorder, probe)
    run_step(recorder, probe)
    assert recorder.jobs["layer"]["generation"] == 1
    assert recorder.jobs["layer"]["initial_placement"] == ((0, 1),)
    assert recorder.jobs["layer"]["current_placement"] == ((1, 0),)


def test_partial_window_and_worker_logging(api, monkeypatch, caplog):
    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)
    caplog.set_level(logging.INFO, logger="vllm.eplb.diagnostics")
    recorder, probe, _ = make_recorder(api, monkeypatch)
    run_step(recorder, probe)
    recorder.finish()
    assert recorder.jobs["layer"]["loads"] == [[1, 1]]
    assert "[EPLB experts]" in caplog.text
    assert "estimated_net_saving_ms=unknown" in caplog.text
    assert "[EPLB overhead]" in caplog.text


@pytest.mark.parametrize("v2", [False, True])
def test_enabled_validation_rejects_off_configuration(api, v2):
    runner = SimpleNamespace(
        ascend_config=SimpleNamespace(
            eplb_diagnostics=api.config.EplbDiagnosticsConfig(mode="benefit"),
            eplb_config=SimpleNamespace(dynamic_eplb=False),
        ),
        vllm_config=SimpleNamespace(
            use_v2_model_runner=v2,
            parallel_config=SimpleNamespace(enable_eplb=False, enable_expert_parallel=True),
        ),
    )
    with pytest.raises(ValueError, match="EP and EPLB enabled"):
        api.runtime.initialize_diagnostics(runner)


def test_mrv2_router_mapping_and_wrapper_commit_keep_graph_view(api, monkeypatch):
    recorder, probe, layer = make_recorder(api, monkeypatch)
    mapping = torch.tensor([[1], [0]])
    layer._use_v2_model_runner = True
    layer.router = SimpleNamespace(eplb_state=SimpleNamespace(logical_to_physical_map=mapping))
    view = api.runtime.logical_to_physical(layer)
    assert view.tolist() == [1, 0]
    mapping.copy_(torch.tensor([[0], [1]]))
    assert view.tolist() == [0, 1]
    recorder.committed(SimpleNamespace(routed_experts=layer))
    assert probe.generation == 1


def test_owner_count_corruption_rejected_before_accumulation(api, monkeypatch):
    recorder, probe, _ = make_recorder(api, monkeypatch)
    recorder.begin(2)
    probe.record_routes(torch.tensor([[0], [1], [99]]))
    probe.owner_totals[0, 0] += 1
    recorder.finish()
    assert recorder.invalid_windows["layer"] == 1
    assert not recorder.jobs and not recorder.logical_totals


def test_post_forward_step_keeps_one_pending_real_qualification(api, monkeypatch):
    recorder, probe, _ = make_recorder(api, monkeypatch)
    run_step(recorder, probe)
    assert not recorder.monitor.active and recorder.monitor.pending_real
    recorder.begin(0, dummy=True)
    assert not recorder.monitor.pending_real


@pytest.mark.parametrize("busy_kind", ["forward", "sample", "planner", "transfer", "v1_cycle", "v1_copy"])
def test_calibration_defers_for_any_rank_with_pending_work(api, monkeypatch, busy_kind):
    recorder, _, _ = make_recorder(api, monkeypatch)
    model = SimpleNamespace(rebalanced=False, pending_result=None)
    runner = SimpleNamespace(
        _eplb_diagnostics_recorder=recorder,
        eplb=SimpleNamespace(state=SimpleNamespace(model_states={"model": model})),
    )
    if busy_kind == "forward":
        runner._eplb_diagnostics_in_execution = True
    elif busy_kind == "sample":
        recorder.monitor.pending_real = True
    elif busy_kind == "planner":
        model.rebalanced = True
    elif busy_kind == "transfer":
        model.pending_result = object()
    else:
        runner.eplb_updator = SimpleNamespace(
            cur_iterations=10 if busy_kind == "v1_cycle" else 0,
            expert_heat_collection_interval=10,
            eplb_loader=SimpleNamespace(
                state=SimpleNamespace(name="TRANSFERRING" if busy_kind == "v1_copy" else "WAITING")
            ),
        )
    assert not api.runtime.diagnostics_quiescent(runner, calibration=True)
    assert not recorder.finished


def test_quiescent_check_does_not_treat_old_waited_requests_as_busy(api, monkeypatch):
    recorder, _, _ = make_recorder(api, monkeypatch)
    runner = SimpleNamespace(
        _eplb_diagnostics_recorder=recorder,
        eplb_updator=SimpleNamespace(
            reqs=[object()],
            cur_iterations=0,
            expert_heat_collection_interval=10,
            eplb_loader=SimpleNamespace(state=SimpleNamespace(name="WAITING")),
        ),
    )
    assert api.runtime.diagnostics_quiescent(runner, calibration=True)
