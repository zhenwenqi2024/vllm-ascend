# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable checks for aligned calibration evidence and collective RPC flow."""

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


def metadata(phase="decode", graph="FULL", padded=8):
    return {"phase": phase, "graph_mode": graph, "padded_tokens": padded}


def job():
    return {
        "plan": SimpleNamespace(
            source_placement=((0, 1), (2, 3)),
            candidate_placement=((0, 2), (1, 3)),
            planner_ms=5,
            source_window=4,
            moved_experts=2,
        ),
        "window": 5,
        "loads": [[9, 3, 1, 0], [0, 1, 2, 7]],
        "metadata": [[metadata(), metadata()], [metadata(), metadata()]],
        "comm_codes": [[1, 1], [1, 1]],
    }


def measurements():
    # The busiest EP rank changes between steps. Averaging each rank first
    # would hide the critical-path work and produce the wrong estimate.
    return [
        {"baseline": [(8, 10), (1, 2)], "candidate": [(3, 4), (2, 3)]},
        {"baseline": [(1, 3), (9, 12)], "candidate": [(2, 3), (4, 5)]},
    ]


def costs():
    return [
        {"collection_ms": (1, 2), "transfer_ms": (20, 30), "apply_ms": (1, 2), "reason": None},
        {"collection_ms": (3, 4), "transfer_ms": (10, 25), "apply_ms": (2, 3), "reason": None},
    ]


def test_samples_keep_phase_graph_and_shape_comparable(api):
    data = job()
    data["loads"] = [[1, 0, 0, 0] for _ in range(7)]
    data["loads"][1] = [0, 0, 0, 0]
    data["metadata"] = [[metadata() for _ in range(7)] for _ in range(2)]
    data["metadata"][0][2] = data["metadata"][1][2] = metadata("prefill", "NONE", 64)
    data["metadata"][0][6] = data["metadata"][1][6] = metadata(padded=16)
    data["comm_codes"] = [[1] * 7, [1] * 7]
    data["comm_codes"][1][4] = 2
    key, selected = api.select_samples(data, 2)
    assert key == (1, "decode", "FULL", (8, 8))
    assert selected == [0, 3]
    assert api.select_samples(data, 32)[1] == [0, 3, 5]


def test_idle_source_still_has_an_expert_owner_in_sample(api):
    data = job()
    data["metadata"][1] = [None, None]
    key, selected = api.select_samples(data, 2)
    assert key == (1, "decode", "FULL", (8, None))
    assert selected == [0, 1]
    # An idle source can be present, but missing communication evidence cannot.
    data["comm_codes"][1] = None
    assert api.select_samples(data, 2) == (None, [])


@pytest.mark.parametrize("problem", ["dummy", "unknown_comm", "mixed_phase", "missing_comm"])
def test_incomplete_samples_are_not_calibrated(api, problem):
    data = job()
    if problem == "dummy":
        data["metadata"] = [[None, None], [None, None]]
    elif problem == "unknown_comm":
        data["comm_codes"] = [[0, 0], [0, 0]]
    elif problem == "mixed_phase":
        data["metadata"][1] = [metadata("prefill"), metadata("prefill")]
    else:
        data["comm_codes"][1] = []
    assert api.select_samples(data, 2) == (None, [])


def test_critical_path_uses_aligned_step_maxima(api):
    assert api.critical_path_interval(measurements(), "baseline") == (8.5, 11.0)
    assert api.critical_path_interval(measurements(), "candidate") == (3.5, 4.5)


def test_budget_amortizes_all_update_costs_by_production_interval(api):
    baseline, candidate, saving, overhead, budget = api.calibrated_budget(measurements(), costs(), 5, 10)
    assert baseline == (8.5, 11.0)
    assert candidate == (3.5, 4.5)
    assert saving == (4.0, 7.5)
    assert overhead == pytest.approx((3.0, 4.2))
    assert budget == pytest.approx((-0.2, 4.5))
    slower_updates = api.calibrated_budget(measurements(), costs(), 5, 20)
    assert slower_updates[:3] == (baseline, candidate, saving)
    assert slower_updates[3] == pytest.approx((1.5, 2.1))
    assert slower_updates[4] == pytest.approx((1.9, 6.0))


def rpc_fixture(api, monkeypatch, *, rank=0, jobs=None, failed_rank=None):
    jobs = {"layer.0": job()} if jobs is None else jobs
    actions = []
    group = SimpleNamespace(ranks=[2, 5], rank_in_group=rank, cpu_group=object())
    probe = object()
    recorder = SimpleNamespace(
        group=group,
        summary=SimpleNamespace(
            stage=1, layers={name: SimpleNamespace(calibration_job=value) for name, value in jobs.items()}
        ),
        layers=[("layer.0", SimpleNamespace(eplb_diagnostic_probe=probe))],
        finished=False,
        step=160,
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
        elif value is None or (isinstance(value, dict) and "result" not in value and "reason" not in value):
            output[:] = [deepcopy(jobs), None]
        elif "result" in value:
            output[:] = [{"result": result, "reason": None, "bytes": 1024} for result in measurements()]
            output[rank] = deepcopy(value)
            if failed_rank is not None:
                output[failed_rank] = {"result": None, "reason": "missing_mlp_template", "bytes": None}
        else:
            output[:] = costs()
            output[rank] = deepcopy(value)

    def measure(probe_arg, baseline, candidate, comm, repeats):
        assert recorder.finished, "Live request collection must finish before calibration"
        assert probe_arg is probe
        actions.append(("mlp", baseline, candidate, comm, repeats))
        if rank == failed_rank:
            return None, "missing_mlp_template"
        return measurements()[rank], None

    def overheads(group_arg, source, candidate, payload, **kwargs):
        assert group_arg is group
        actions.append(("scratch", source, candidate, payload, kwargs))
        return costs()[rank]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    monkeypatch.setattr(api, "calibrate_layer", measure)
    monkeypatch.setattr(api, "expert_payload_bytes", lambda *_: (1024, None))
    monkeypatch.setattr(api, "measure_overheads", overheads)
    return recorder, actions


@pytest.mark.parametrize("rank", [0, 1])
def test_rpc_calibrates_candidate_without_changing_placement_and_is_idempotent(api, monkeypatch, caplog, rank):
    caplog.set_level(logging.INFO)
    recorder, actions = rpc_fixture(api, monkeypatch, rank=rank)
    original = deepcopy(recorder.summary.layers["layer.0"].calibration_job)
    with torch.inference_mode():
        api.calibrate(recorder, update_interval=10, samples=2, repeats=2, max_scratch_bytes=4096)
    expected = ([[9, 3], [0, 1]], [[9, 1], [0, 2]]) if rank == 0 else ([[1, 0], [2, 7]], [[3, 0], [1, 7]])
    assert actions[0] == "finish"
    assert actions[1] == ("mlp", *expected, "MC2CommImpl", 2)
    assert actions[2][0] == "scratch"
    assert actions[2][-1] == {"repeats": 2, "max_scratch_bytes": 4096}
    assert recorder.summary.layers["layer.0"].calibration_job == original
    assert recorder.calibrated
    assert ("calibrated_layers=1" in caplog.text) == (rank == 0)
    if rank == 0:
        assert "source_window=4 evaluation_window=5" in caplog.text
        assert "baseline_mlp_ms=(8.5, 11.0)" in caplog.text
        assert "production_update_interval=10" in caplog.text
        assert "collection_per_update_ms=(3.0, 4.0)" in caplog.text
        assert "transfer_per_update_ms=(20.0, 30.0)" in caplog.text
        assert "apply_per_update_ms=(2.0, 3.0)" in caplog.text
        assert "compute_budget_status=uncertain" in caplog.text
        assert (
            "uncalibrated_layers=0 compute_budget_layer_counts={'positive': 0, 'nonpositive': 0, 'uncertain': 1}"
            in caplog.text
        )
        assert "conclusion=insufficient_benefit_evidence net_saving_ms=unknown" in caplog.text
    previous_actions = list(actions)
    api.calibrate(recorder, update_interval=10, samples=2, repeats=2, max_scratch_bytes=4096)
    assert actions == previous_actions
    assert ("reason=already_calibrated" in caplog.text) == (rank == 0)
    # New settings intentionally request a new isolated calibration.
    api.calibrate(recorder, update_interval=10, samples=2, repeats=3, max_scratch_bytes=4096)
    assert len(actions) == len(previous_actions) + 2
    assert actions[-2][-1] == 3
    assert recorder.calibrated
    assert recorder.calibration_arguments == (10, 2, 3, 4096)


@pytest.mark.parametrize("failed_rank", [0, 1])
def test_missing_template_on_any_rank_skips_collective_scratch(api, monkeypatch, caplog, failed_rank):
    caplog.set_level(logging.INFO)
    recorder, actions = rpc_fixture(api, monkeypatch, failed_rank=failed_rank)
    api.calibrate(recorder, update_interval=10, samples=2)
    assert not recorder.calibrated
    assert not any(isinstance(action, tuple) and action[0] == "scratch" for action in actions)
    assert "missing_mlp_template" in caplog.text
    assert "calibrated_layers=0 available_layers=1 uncalibrated_layers=1" in caplog.text
    assert "compute_budget_layer_counts={'positive': 0, 'nonpositive': 0, 'uncertain': 0}" in caplog.text


def test_rpc_without_previous_window_evidence_never_runs_measurements(api, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    recorder, actions = rpc_fixture(api, monkeypatch, jobs={})
    api.calibrate(recorder, update_interval=10)
    assert actions == ["finish"]
    assert not recorder.calibrated
    assert "calibrated_layers=0 available_layers=0 uncalibrated_layers=0" in caplog.text
    assert "compute_budget_layer_counts={'positive': 0, 'nonpositive': 0, 'uncertain': 0}" in caplog.text
    assert "speedup=not_estimated" in caplog.text


@pytest.mark.parametrize("results", [[], [{"baseline": []}], [{"baseline": [1]}, {"baseline": [1, 2]}]])
def test_missing_or_unaligned_timings_never_become_a_budget(api, results):
    with pytest.raises(ValueError):
        api.critical_path_interval(results, "baseline")


def test_rank_local_measurement_error_is_shared_before_scratch(api, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    recorder, actions = rpc_fixture(api, monkeypatch)

    def failure(*args, **kwargs):
        raise RuntimeError("synthetic device failure")

    monkeypatch.setattr(api, "calibrate_layer", failure)
    api.calibrate(recorder, update_interval=10)
    assert actions == ["finish"]
    assert not recorder.calibrated
    assert "calibration_failed:RuntimeError" in caplog.text
    assert "calibrated_layers=0 available_layers=1 uncalibrated_layers=1" in caplog.text
    assert "compute_budget_layer_counts={'positive': 0, 'nonpositive': 0, 'uncertain': 0}" in caplog.text


@pytest.mark.parametrize("field", ["arguments", "finished", "calibrated", "calibration_arguments", "step"])
def test_peer_state_disagreement_is_rejected_before_finish_or_device_work(api, monkeypatch, field):
    recorder, actions = rpc_fixture(api, monkeypatch)

    def mismatch(output, value, *, group):
        arguments, finished, calibrated, calibration_arguments, step = value
        peer = (
            (20, *arguments[1:]) if field == "arguments" else arguments,
            not finished if field == "finished" else finished,
            not calibrated if field == "calibrated" else calibrated,
            arguments if field == "calibration_arguments" else calibration_arguments,
            step + 1 if field == "step" else step,
        )
        output[:] = [value, peer]

    monkeypatch.setattr(torch.distributed, "all_gather_object", mismatch)
    with pytest.raises(ValueError, match="All EP workers"):
        api.calibrate(recorder, update_interval=10)
    assert not actions


def test_failed_scratch_budget_can_retry_with_larger_cap(api, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    recorder, actions = rpc_fixture(api, monkeypatch)
    caps = []

    def overheads(*args, max_scratch_bytes, **kwargs):
        caps.append(max_scratch_bytes)
        return costs()[0] | {"reason": "scratch_memory_cap_exceeded" if max_scratch_bytes < 4096 else None}

    monkeypatch.setattr(api, "measure_overheads", overheads)
    api.calibrate(recorder, update_interval=10, max_scratch_bytes=1024)
    assert not recorder.calibrated
    assert "scratch_memory_cap_exceeded" in caplog.text
    api.calibrate(recorder, update_interval=10, max_scratch_bytes=4096)
    assert caps == [1024, 4096]
    assert recorder.calibrated
    assert "calibrated_layers=1 available_layers=1" in caplog.text
    assert sum(isinstance(action, tuple) and action[0] == "mlp" for action in actions) == 2


def test_partial_calibration_is_retryable_with_the_same_arguments(api, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    recorder, actions = rpc_fixture(api, monkeypatch, jobs={"layer.0": job(), "layer.1": job()})
    # The second local layer is initially unavailable. First-layer success
    # must not prevent a later retry from filling in the missing evidence.
    api.calibrate(recorder, update_interval=10)
    assert not recorder.calibrated
    assert "calibrated_layers=1 available_layers=2 uncalibrated_layers=1" in caplog.text
    assert "compute_budget_layer_counts={'positive': 0, 'nonpositive': 0, 'uncertain': 1}" in caplog.text
    recorder.layers.append(("layer.1", recorder.layers[0][1]))
    api.calibrate(recorder, update_interval=10)
    assert recorder.calibrated
    assert "calibrated_layers=2 available_layers=2 uncalibrated_layers=0" in caplog.text
    assert "compute_budget_layer_counts={'positive': 0, 'nonpositive': 0, 'uncertain': 2}" in caplog.text
    assert sum(isinstance(action, tuple) and action[0] == "mlp" for action in actions) == 3


@pytest.mark.parametrize(("interval", "status"), [(100, "positive"), (1, "nonpositive")])
def test_isolated_budget_sign_does_not_become_an_enable_recommendation(api, monkeypatch, caplog, interval, status):
    caplog.set_level(logging.INFO)
    recorder, _ = rpc_fixture(api, monkeypatch)
    api.calibrate(recorder, update_interval=interval)
    assert f"compute_budget_status={status}" in caplog.text
    expected_counts = dict.fromkeys(("positive", "nonpositive", "uncertain"), 0)
    expected_counts[status] = 1
    assert f"uncalibrated_layers=0 compute_budget_layer_counts={expected_counts}" in caplog.text
    assert "conclusion=insufficient_benefit_evidence net_saving_ms=unknown" in caplog.text
