# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable benefit/cost and isolated device timing orchestration tests."""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def api():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/eplb/diagnostics/calibration.py"
    spec = importlib.util.spec_from_file_location("eplb_calibration_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def costs(**overrides):
    return (
        dict(
            baseline_ms=(9, 11),
            candidate_ms=(5, 6),
            collection_ms=(0.1, 0.2),
            planner_ms=(1, 2),
            transfer_ms=(4, 8),
            apply_ms=(1, 2),
            update_interval=10,
            communication_delta_ms=(0, 0.1),
        )
        | overrides
    )


def test_conservative_net_interval_and_production_amortization(api):
    result = api.estimate_net_benefit(**costs())
    assert result["gross_saving_ms"] == (3, 6)
    assert result["amortized_cost_ms"] == pytest.approx((0.7, 1.5))
    assert result["net_saving_ms"] == pytest.approx((1.5, 5.3))
    assert result["conclusion"] == "positive_margin"
    # Same measurements, more frequent updates: all saving can be lost.
    result = api.estimate_net_benefit(**costs(update_interval=1))
    assert result["net_saving_ms"][1] < 0
    assert result["conclusion"] == "cost_exceeds_benefit"


def test_uncertain_bounds_do_not_recommend_enabling(api):
    result = api.estimate_net_benefit(**costs(candidate_ms=(8, 10)))
    assert result["net_saving_ms"][0] < 0 < result["net_saving_ms"][1]
    assert result["conclusion"] == "uncertain"


def test_candidate_can_be_slower_and_zero_margin_is_not_positive(api):
    assert api.estimate_net_benefit(**costs(candidate_ms=12))["conclusion"] == "cost_exceeds_benefit"
    result = api.estimate_net_benefit(
        **costs(
            baseline_ms=1,
            candidate_ms=1,
            collection_ms=0,
            planner_ms=0,
            transfer_ms=0,
            apply_ms=0,
            communication_delta_ms=0,
        )
    )
    assert result["net_saving_ms"] == (0, 0)
    assert result["conclusion"] == "cost_exceeds_benefit"


@pytest.mark.parametrize("missing", list(costs()))
def test_unknown_costs_and_timing_are_never_free(api, missing):
    result = api.estimate_net_benefit(**costs(**{missing: None}))
    assert result["conclusion"] == "insufficient_evidence"
    assert result["missing"] == [missing]
    assert result["net_saving_ms"] is None


def test_signed_communication_change_is_accounted_for(api):
    regular = api.estimate_net_benefit(**costs(communication_delta_ms=0))
    improved = api.estimate_net_benefit(**costs(communication_delta_ms=(-0.5, -0.2)))
    assert improved["net_saving_ms"][0] == pytest.approx(regular["net_saving_ms"][0] + 0.2)
    assert improved["net_saving_ms"][1] == pytest.approx(regular["net_saving_ms"][1] + 0.5)


def test_latency_normalization(api):
    assert api.latency_interval(2) == (2.0, 2.0)
    assert api.latency_interval([1, 3]) == (1.0, 3.0)
    assert api.latency_interval({"min_ms": 1, "median_ms": 2, "max_ms": 3}) == (1.0, 3.0)


@pytest.mark.parametrize("value", [(2, 1), (-1, 2), math.nan, math.inf, True, (1,), "12"])
def test_invalid_latency_rejected(api, value):
    with pytest.raises(ValueError):
        api.latency_interval(value)


@pytest.mark.parametrize("interval", [0, -1, 1.5, True])
def test_invalid_update_interval_rejected(api, interval):
    with pytest.raises(ValueError):
        api.estimate_net_benefit(**costs(update_interval=interval))


def test_timing_prepares_fresh_inputs_outside_events_and_retains_outputs(api, monkeypatch):
    actions = []
    times = iter([3, 1, 2])

    class Event:
        def __init__(self, **kwargs):
            assert kwargs == {"enable_timing": True}

        def record(self):
            actions.append("record")

        def synchronize(self):
            actions.append("event_sync")

        def elapsed_time(self, other):
            actions.append("elapsed")
            return next(times)

    class Scratch:
        def __del__(self):
            actions.append("release")

    def factory():
        actions.append("prepare")
        scratch = Scratch()

        def operation():
            actions.append("operation")
            return scratch

        return operation

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            Event=Event, synchronize=lambda: actions.append("sync"), is_current_stream_capturing=lambda: False
        ),
        raising=False,
    )
    result = api.measure_callable(factory, repeats=3, warmup=1)
    assert result == {"min_ms": 1, "median_ms": 2, "max_ms": 3, "samples": 3}
    assert (
        actions
        == ["prepare", "sync", "operation", "sync", "release"]
        + ["prepare", "sync", "record", "operation", "record", "event_sync", "elapsed", "release"] * 3
    )


def test_timing_cannot_run_inside_capture(api, monkeypatch):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: True), raising=False)
    with pytest.raises(RuntimeError, match="outside graph capture"):
        api.measure_callable(lambda: pytest.fail("must not prepare inside capture"))


@pytest.mark.parametrize("kwargs", [{"repeats": 0}, {"repeats": True}, {"warmup": -1}, {"warmup": 1.5}])
def test_bad_sample_counts_rejected(api, kwargs):
    with pytest.raises(ValueError):
        api.measure_callable(lambda: None, **kwargs)
