# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual-layout comparison, cost scope, and coordinated calibration tests."""

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
    yield importlib.import_module("vllm_ascend.eplb.diagnostics.assessment")
    if stubbed:
        for name in set(sys.modules) - original:
            if name == "vllm_ascend" or name.startswith("vllm_ascend."):
                del sys.modules[name]


@pytest.fixture(autouse=True)
def capture_worker_logs(api, monkeypatch):
    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)
    monkeypatch.setattr(logging.getLogger("vllm.eplb.diagnostics"), "propagate", True)


def metadata(phase="decode", graph="FULL", padded=8):
    return {"phase": phase, "graph_mode": graph, "padded_tokens": padded}


def job():
    return {
        "initial_placement": ((0, 1), (2, 3)),
        "current_placement": ((0, 2), (1, 3)),
        "generation": 3,
        "window_end_step": 160,
        "loads": [[9, 3, 1, 0], [0, 1, 2, 7]],
        "metadata": [[metadata(), metadata()], [metadata(), metadata()]],
        "comm_codes": [[1, 1], [1, 1]],
    }


def measurements():
    return [
        {"baseline": [(8, 10), (1, 2)], "candidate": [(3, 4), (2, 3)]},
        {"baseline": [(1, 3), (9, 12)], "candidate": [(2, 3), (4, 5)]},
    ]


def costs():
    return [
        {
            "pending": 0,
            "dropped": 0,
            "totals": [
                {"component": "eplb_step_before", "kind": "device", "sum_ms": 4},
                {"component": "eplb_step", "kind": "host", "sum_ms": 99},
                {"component": "transfer_launch", "kind": "device", "sum_ms": 90},
                {"component": "planner_wait", "kind": "host", "sum_ms": 80},
            ],
        },
        {
            "pending": 0,
            "dropped": 0,
            "totals": [
                {"component": "eplb_step_before", "kind": "device", "sum_ms": 2},
                {"component": "eplb_step_after", "kind": "device", "sum_ms": 5},
                {"component": "commit_apply", "kind": "device", "sum_ms": 50},
            ],
        },
    ]


def test_samples_keep_phase_graph_shape_and_communication_comparable(api):
    data = job()
    data["loads"] = [[1, 0, 0, 0] for _ in range(8)]
    data["loads"][1] = [0, 0, 0, 0]
    data["metadata"] = [[metadata() for _ in range(8)] for _ in range(2)]
    data["metadata"][0][2] = data["metadata"][1][2] = metadata("prefill", "NONE", 64)
    data["metadata"][0][6] = data["metadata"][1][6] = metadata(padded=16)
    data["metadata"][1][7] = metadata(graph="NONE")
    data["comm_codes"] = [[1] * 8, [1] * 8]
    data["comm_codes"][1][4] = 2
    assert api.select_samples(data, 2) == ((1, "decode", "FULL", (8, 8)), [0, 3])
    assert api.select_samples(data, 32)[1] == [0, 3, 5]


def test_idle_source_is_allowed_but_missing_rank_or_comm_is_not(api):
    data = job()
    data["metadata"][1] = [None, None]
    assert api.select_samples(data, 2) == ((1, "decode", "FULL", (8, None)), [0, 1])
    for key in ("metadata", "comm_codes"):
        incomplete = deepcopy(data)
        incomplete[key] = incomplete[key][:1]
        assert api.select_samples(incomplete, 2) == (None, [])
    data["comm_codes"][1] = None
    assert api.select_samples(data, 2) == (None, [])


def test_rank_histories_must_match_the_complete_workload_window(api):
    for key in ("metadata", "comm_codes"):
        for delta in (-1, 1):
            data = job()
            history = data[key][1]
            data[key][1] = history[:-1] if delta < 0 else history + [history[-1]]
            assert api.select_samples(data, 2) == (None, [])


@pytest.mark.parametrize("problem", ["dummy", "unknown_comm", "phase_mismatch"])
def test_invalid_sample_scope_is_skipped(api, problem):
    data = job()
    if problem == "dummy":
        data["metadata"] = [[None, None], [None, None]]
    elif problem == "unknown_comm":
        data["comm_codes"] = [[0, 0], [0, 0]]
    else:
        data["metadata"][1] = [metadata("prefill"), metadata("prefill")]
    assert api.select_samples(data, 2) == (None, [])


def test_critical_path_takes_ep_max_for_each_step_before_averaging(api):
    assert api.critical_path_samples(measurements(), "baseline") == [(8, 10), (9, 12)]
    assert api.critical_path_samples(measurements(), "candidate") == [(3, 4), (4, 5)]
    estimate = api.estimate_adjustment_benefit(
        previous_step_ms=api.critical_path_samples(measurements(), "baseline"),
        current_step_ms=api.critical_path_samples(measurements(), "candidate"),
        missing=("exposed_overhead",),
    )
    assert estimate["adjustment_saving_ms"] == (4, 7.5)
    assert estimate["estimated_net_saving_ms"] is None
    for broken in ([], [{"baseline": []}], [{"baseline": [1]}, {"baseline": [1, 2]}]):
        with pytest.raises(ValueError):
            api.critical_path_samples(broken, "baseline")


def test_observed_cost_excludes_host_and_nested_spans(api):
    data = costs()
    assert api.observed_overhead(data) == (4, 7)
    # Raw component sizes do not change the covered outer device span.
    data[0]["totals"][1]["sum_ms"] = 9999
    data[1]["totals"][2]["sum_ms"] = 9999
    assert api.observed_overhead(data) == (4, 7)
    for row, total in zip(data, (4, 7)):
        row["totals"] = [{"component": "eplb_step", "kind": "device", "sum_ms": total}]
    assert api.observed_overhead(data) == (4, 7)


@pytest.mark.parametrize("problem", ["pending", "dropped", "missing_outer"])
def test_incomplete_observed_cost_is_unknown(api, problem):
    data = costs()
    if problem == "missing_outer":
        data[1]["totals"] = [{"component": "commit_apply", "kind": "device", "sum_ms": 5}]
    else:
        data[1][problem] = 1
    assert api.observed_overhead(data) is None
    assert api.observed_overhead([]) is None


def test_invalid_or_overlapping_outer_costs_are_unknown(api):
    for invalid in (-1, float("nan"), float("inf"), True, None, (1, 2)):
        data = costs()
        data[0]["totals"][0]["sum_ms"] = invalid
        assert api.observed_overhead(data) is None
    data = costs()
    data[0]["totals"][0]["component"] = "eplb_step"
    assert api.observed_overhead(data) is None  # Different families across ranks.
    data = costs()
    data[1]["totals"].append({"component": "eplb_step", "kind": "device", "sum_ms": 1})
    assert api.observed_overhead(data) is None  # Nested families on one rank.


def rpc_fixture(api, monkeypatch, *, rank=0, jobs=None):
    jobs = {"layer.0": job()} if jobs is None else jobs
    actions = []
    state = SimpleNamespace(failed_rank=None, reject_cap_below=0, mismatch=None, peer_success=None, peer_inventory=None)
    group = SimpleNamespace(ranks=[2, 5], rank_in_group=rank, cpu_group=object())
    probe_class = importlib.import_module("vllm_ascend.eplb.diagnostics.probe").ExpertLoadProbe
    probe = probe_class(4, "cpu")
    probe.eplb_enabled = True
    probe.logical_to_physical = torch.tensor([0, 2, 1, 3])
    recorder = SimpleNamespace(
        group=group,
        stage=1,
        jobs=jobs,
        layers=[("layer.0", SimpleNamespace(eplb_diagnostic_probe=probe))],
        finished=False,
        step=160,
        observed_steps=150,
        costs=costs(),
    )

    def finish():
        if not recorder.finished:
            actions.append("finish")
            recorder.finished = True

    recorder.finish = finish

    def gather(output, value, *, group):
        assert group is recorder.group.cpu_group
        assert not torch.is_inference_mode_enabled()
        if isinstance(value, tuple):
            output[:] = [value, value]
            if len(value) == 3 and state.mismatch:
                arguments, finished, step = value
                output[1] = (
                    (1, *arguments[1:]) if state.mismatch == "arguments" else arguments,
                    not finished if state.mismatch == "finished" else finished,
                    step + 1 if state.mismatch == "step" else step,
                )
            elif len(value) == 2 and state.peer_success is not None:
                output[1] = (value[0], state.peer_success)
        elif isinstance(value, list):
            output[:] = [value, value if state.peer_inventory is None else state.peer_inventory]
        elif value is None or "result" not in value:
            output[:] = [deepcopy(jobs), None]
        else:
            output[:] = [{"result": result, "reason": None} for result in measurements()]
            output[rank] = deepcopy(value)
            if state.failed_rank is not None:
                output[state.failed_rank] = {"result": None, "reason": "missing_mlp_template"}

    def measure(probe_arg, baseline, current, comm, repeats, *, max_scratch_bytes):
        assert recorder.finished and probe_arg is probe and probe.eplb_enabled
        actions.append(("mlp", baseline, current, comm, repeats, max_scratch_bytes))
        if max_scratch_bytes < state.reject_cap_below:
            return None, "calibration_scratch_estimate_exceeds_limit"
        return measurements()[rank], None

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    monkeypatch.setattr(api, "calibrate_layer", measure)
    return recorder, actions, state


@pytest.mark.parametrize("rank", [0, 1])
def test_rpc_uses_actual_layouts_keeps_live_state_and_reports_three_fields(api, monkeypatch, caplog, rank):
    caplog.set_level(logging.INFO)
    recorder, actions, _ = rpc_fixture(api, monkeypatch, rank=rank)
    original = deepcopy(recorder.jobs)
    mapping = recorder.layers[0][1].eplb_diagnostic_probe.logical_to_physical.clone()
    with torch.inference_mode():
        api.calibrate(recorder, samples=2, repeats=2, max_scratch_bytes=4096)
    expected = ([[9, 3], [0, 1]], [[9, 1], [0, 2]]) if rank == 0 else ([[1, 0], [2, 7]], [[3, 0], [1, 7]])
    assert actions == ["finish", ("mlp", *expected, "MC2CommImpl", 2, 4096)]
    assert recorder.jobs == original
    assert torch.equal(mapping, recorder.layers[0][1].eplb_diagnostic_probe.logical_to_physical)
    assert recorder.calibrated
    if rank == 0:
        assert "reference=initial_live_placement" in caplog.text
        assert "adjustment_saving_ms=(4.0, 7.5) adjustment_status=positive" in caplog.text
        assert "eplb_overhead_ms=shared_run_cost" in caplog.text
        assert "covered_update_span_total_ms=(4.0, 7.0) observed_real_steps=150 eplb_overhead_ms=unknown" in caplog.text
        assert "estimated_net_saving_ms=unknown" in caplog.text
        assert "calibrated_layers=1 available_layers=1" in caplog.text
    else:
        assert "[EPLB benefit]" not in caplog.text
    before = list(actions)
    api.calibrate(recorder, samples=2, repeats=2, max_scratch_bytes=4096)
    assert actions == before
    api.calibrate(recorder, samples=2, repeats=3, max_scratch_bytes=4096)
    assert len(actions) == len(before) + 1
    assert actions[-1][-2] == 3


@pytest.mark.parametrize("field", ["arguments", "finished", "step"])
def test_peer_argument_or_recorder_state_mismatch_precedes_device_work(api, monkeypatch, field):
    recorder, actions, state = rpc_fixture(api, monkeypatch)
    state.mismatch = field
    with pytest.raises(ValueError, match="All EP workers"):
        api.calibrate(recorder, samples=2)
    assert not actions


def test_rank_failure_leaves_calibration_retryable_and_unknown_net(api, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    recorder, actions, state = rpc_fixture(api, monkeypatch)
    state.failed_rank = 1
    api.calibrate(recorder, samples=2)
    assert not recorder.calibrated and "missing_mlp_template" in caplog.text
    state.failed_rank = None
    api.calibrate(recorder, samples=2)
    assert recorder.calibrated and len(actions) == 3
    assert "estimated_net_saving_ms=unknown" in caplog.text


def test_local_measurement_exception_is_coordinated(api, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    recorder, actions, _ = rpc_fixture(api, monkeypatch)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic timing failure")

    monkeypatch.setattr(api, "calibrate_layer", fail)
    api.calibrate(recorder)
    assert actions == ["finish"] and not recorder.calibrated
    assert "calibration_failed:RuntimeError" in caplog.text


def test_local_workload_preparation_exception_is_coordinated(api, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    data = job()
    data["initial_placement"] = ((0, 99), (2, 3))
    recorder, actions, _ = rpc_fixture(api, monkeypatch, jobs={"layer.0": data})
    api.calibrate(recorder)
    assert actions == ["finish"] and not recorder.calibrated
    assert "calibration_failed:IndexError" in caplog.text


def test_unchanged_actual_placement_has_exact_zero_saving_without_timing_noise(api, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    data = job()
    data["current_placement"] = data["initial_placement"]
    recorder, actions, _ = rpc_fixture(api, monkeypatch, jobs={"layer.0": data})
    api.calibrate(recorder)
    assert actions == ["finish"] and recorder.calibrated
    assert "adjustment_saving_ms=(0.0, 0.0) adjustment_status=no_adjustment" in caplog.text
    assert "calibration_execution=not_required" in caplog.text
    assert "ranges=exact_zero" in caplog.text
    assert "estimated_net_saving_ms=unknown" in caplog.text


def test_cap_failure_retries_after_larger_cap_without_changing_placement(api, monkeypatch):
    recorder, actions, state = rpc_fixture(api, monkeypatch)
    state.reject_cap_below = 4096
    api.calibrate(recorder, samples=2, max_scratch_bytes=1024)
    assert not recorder.calibrated
    api.calibrate(recorder, samples=2, max_scratch_bytes=4096)
    assert recorder.calibrated
    assert [action[-1] for action in actions if isinstance(action, tuple)] == [1024, 4096]


def test_partial_success_retries_same_arguments_and_requires_all_ranks_for_cache(api, monkeypatch):
    recorder, actions, state = rpc_fixture(api, monkeypatch, jobs={"layer.0": job(), "layer.1": job()})
    api.calibrate(recorder, samples=2)
    assert not recorder.calibrated
    recorder.layers.append(("layer.1", recorder.layers[0][1]))
    api.calibrate(recorder, samples=2)
    assert recorder.calibrated
    before = len(actions)
    state.peer_success = False
    api.calibrate(recorder, samples=2)
    assert len(actions) == before + 2


def test_no_observed_comparison_never_runs_calibration(api, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    recorder, actions, _ = rpc_fixture(api, monkeypatch, jobs={})
    api.calibrate(recorder)
    assert actions == ["finish"] and not recorder.calibrated
    assert "calibrated_layers=0 available_layers=1 comparable_layers=0" in caplog.text
    assert "layer=layer.0 reason=no_comparable_window" in caplog.text
    assert "estimated_net_saving_ms=unknown" in caplog.text


def test_peer_layers_without_windows_remain_visible_and_retryable(api, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    recorder, actions, state = rpc_fixture(api, monkeypatch)
    state.peer_inventory = ["layer.0", "layer.1"]
    api.calibrate(recorder)
    assert not recorder.calibrated
    assert "calibrated_layers=1 available_layers=2 comparable_layers=1" in caplog.text
    assert "layer=layer.1 reason=no_comparable_window" in caplog.text
    api.calibrate(recorder)
    assert len(actions) == 3
