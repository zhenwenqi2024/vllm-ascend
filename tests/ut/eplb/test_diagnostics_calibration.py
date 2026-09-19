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


def measurements(**overrides):
    return (
        dict(
            previous_step_ms=[(9, 11), (5, 7)],
            current_step_ms=[(5, 6), (4, 5)],
            overhead_total_ms=(2, 4),
            observed_steps=10,
            missing=(),
        )
        | overrides
    )


def test_actual_adjustment_cost_and_net_are_separate(api):
    result = api.estimate_adjustment_benefit(**measurements())
    assert result["adjustment_saving_ms"] == (1.5, 4.5)
    assert result["eplb_overhead_ms"] == (0.2, 0.4)
    assert result["measured_budget_ms"] == result["estimated_net_saving_ms"] == (1.1, 4.3)
    assert result["conclusion"] == "positive_margin"
    # Actual observed calls, not an assumed future update interval, amortize cost.
    result = api.estimate_adjustment_benefit(**measurements(observed_steps=1, overhead_total_ms=(5, 6)))
    assert result["adjustment_saving_ms"] == (1.5, 4.5)
    assert result["eplb_overhead_ms"] == (5, 6)
    assert result["conclusion"] == "cost_exceeds_benefit"


def test_paired_step_maxima_keep_alternating_busiest_ranks(api):
    # Rank latencies alternate [10, 0] and [0, 10]. Per-step maxima are 10, 10;
    # max of rank averages would incorrectly use 5. Current layout takes 6, 6.
    result = api.estimate_adjustment_benefit(
        **measurements(previous_step_ms=[10, 10], current_step_ms=[6, 6], overhead_total_ms=0)
    )
    assert result["adjustment_saving_ms"] == (4, 4)


@pytest.mark.parametrize("scope", ["fused_route_collection", "graph_delta", "communication_delta", "overlap"])
def test_missing_scope_preserves_components_but_never_claims_net(api, scope):
    result = api.estimate_adjustment_benefit(**measurements(missing=(scope,)))
    assert result["adjustment_saving_ms"] == (1.5, 4.5)
    assert result["eplb_overhead_ms"] == (0.2, 0.4)
    assert result["measured_budget_ms"] == (1.1, 4.3)
    assert result["estimated_net_saving_ms"] is None
    assert result["conclusion"] == "insufficient_evidence"
    assert result["missing"] == [scope]


def test_unknown_exposed_total_is_not_zero_or_sum_of_components(api):
    result = api.estimate_adjustment_benefit(**measurements(overhead_total_ms=None))
    assert result["adjustment_saving_ms"] == (1.5, 4.5)
    assert result["eplb_overhead_ms"] is None
    assert result["measured_budget_ms"] is None
    assert result["estimated_net_saving_ms"] is None
    assert result["missing"] == ["total_exposed_overhead"]


def test_adjustment_can_be_regression_and_bounds_can_be_uncertain(api):
    result = api.estimate_adjustment_benefit(
        **measurements(previous_step_ms=[5], current_step_ms=[8], overhead_total_ms=1)
    )
    assert result["adjustment_saving_ms"] == (-3, -3)
    assert result["conclusion"] == "cost_exceeds_benefit"
    result = api.estimate_adjustment_benefit(**measurements(previous_step_ms=[(4, 6)], current_step_ms=[(4, 6)]))
    assert result["estimated_net_saving_ms"][0] < 0 < result["estimated_net_saving_ms"][1]
    assert result["conclusion"] == "uncertain"


def test_zero_adjustment_does_not_turn_into_positive_gain(api):
    result = api.estimate_adjustment_benefit(
        **measurements(previous_step_ms=[1], current_step_ms=[1], overhead_total_ms=0)
    )
    assert result["adjustment_saving_ms"] == result["estimated_net_saving_ms"] == (0, 0)
    assert result["conclusion"] == "cost_exceeds_benefit"


@pytest.mark.parametrize("values", [(None, [1]), ([1], None), ([], [])])
def test_missing_comparison_keeps_measured_overhead(api, values):
    result = api.estimate_adjustment_benefit(**measurements(previous_step_ms=values[0], current_step_ms=values[1]))
    assert result["adjustment_saving_ms"] is None
    assert result["eplb_overhead_ms"] == (0.2, 0.4)
    assert result["estimated_net_saving_ms"] is None
    assert result["missing"] == ["matched_layout_timing"]


def test_matched_sample_lengths_required(api):
    with pytest.raises(ValueError, match="paired"):
        api.estimate_adjustment_benefit(**measurements(previous_step_ms=[1], current_step_ms=[1, 2]))


def test_missing_observed_steps_does_not_use_sample_count_as_cost_denominator(api):
    result = api.estimate_adjustment_benefit(**measurements(observed_steps=None))
    assert result["adjustment_saving_ms"] == (1.5, 4.5)
    assert result["eplb_overhead_ms"] is None
    assert result["missing"] == ["observed_steps"]


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_invalid_observed_steps_rejected(api, steps):
    with pytest.raises(ValueError):
        api.estimate_adjustment_benefit(**measurements(observed_steps=steps))


def test_missing_scope_must_be_explicit(api):
    with pytest.raises(TypeError):
        api.estimate_adjustment_benefit()
    with pytest.raises(ValueError):
        api.estimate_adjustment_benefit(**measurements(missing="graph_delta"))


def test_latency_normalization(api):
    assert api.latency_interval(2) == (2.0, 2.0)
    assert api.latency_interval([1, 3]) == (1.0, 3.0)
    assert api.latency_interval({"min_ms": 1, "median_ms": 2, "max_ms": 3}) == (1.0, 3.0)


@pytest.mark.parametrize("value", [(2, 1), (-1, 2), math.nan, math.inf, True, (1,), "12"])
def test_invalid_latency_rejected(api, value):
    with pytest.raises(ValueError):
        api.latency_interval(value)


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


def test_missing_generator_is_not_consumed_before_net_gate(api):
    result = api.estimate_adjustment_benefit(**measurements(missing=(x for x in ("graph_delta",))))
    assert result["missing"] == ["graph_delta"]
    assert result["estimated_net_saving_ms"] is None
