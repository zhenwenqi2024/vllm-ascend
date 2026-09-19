# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable tests of opt-in actual-work timing and deferred NPU events."""

import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def api():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/eplb/diagnostics/overhead.py"
    spec = importlib.util.spec_from_file_location("live_eplb_overhead_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def events(monkeypatch):
    calls, instances = [], []

    class Event:
        ready = False

        def __init__(self, **kwargs):
            assert kwargs == {"enable_timing": True}
            instances.append(self)

        def record(self, stream=None):
            calls.append(("record", stream))

        def query(self):
            calls.append(("query", self))
            return self.ready

        def synchronize(self):
            calls.append(("synchronize", self))
            self.ready = True

        def elapsed_time(self, other):
            calls.append(("elapsed", other))
            assert other.ready
            return 2.5

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(Event=Event, is_current_stream_capturing=lambda: False),
        raising=False,
    )
    return calls, instances


def test_inactive_monitor_never_reads_clock_or_creates_events(api, monkeypatch, events):
    monkeypatch.setattr(api, "perf_counter", lambda: pytest.fail("inactive diagnostic read the clock"))
    monitor = api.LiveEplbOverhead(lambda layer: None)
    with monitor.cpu_span("planner_compute"), monitor.device_span("eplb_step"):
        pass
    assert events == ([], [])
    assert monitor.drain() == {"samples": [], "totals": [], "pending": 0, "dropped": 0}


def test_live_drain_queries_without_synchronizing_and_keeps_run_totals(api, events):
    monitor = api.LiveEplbOverhead(lambda layer: None)
    monitor.active = True
    stream = object()
    with monitor.device_span("migration_pipeline", "layer.3", stream=stream):
        pass
    calls, instances = events
    assert calls == [("record", stream), ("record", stream)]
    first = monitor.drain()
    assert first["pending"] == 1 and not first["samples"]
    assert not any(name == "synchronize" for name, _ in calls)
    instances[-1].ready = True
    second = monitor.drain()
    assert second["pending"] == 0
    assert second["samples"] == [dict(component="migration_pipeline", kind="device", layer="layer.3", duration_ms=2.5)]
    total = dict(component="migration_pipeline", kind="device", layer="layer.3", count=1, sum_ms=2.5)
    assert second["totals"] == [total]
    third = monitor.drain()
    assert third["samples"] == [] and third["totals"] == [total]


def test_bounded_events_report_incomplete_coverage_and_final_rpc_may_wait(api, events):
    monitor = api.LiveEplbOverhead(lambda layer: None, max_samples=1)
    monitor.active = True
    with monitor.device_span("eplb_step"):
        pass
    with monitor.device_span("eplb_step"):
        pass
    assert len(events[1]) == 2
    summary = monitor.drain(synchronize=True)
    assert summary["pending"] == 0 and summary["dropped"] == 1
    assert summary["totals"][0]["count"] == 1
    assert sum(name == "synchronize" for name, _ in events[0]) == 1


def test_host_totals_survive_bounded_recent_details_and_nested_spans(api, monkeypatch):
    ticks = iter([0.0, 0.001, 0.003, 0.008, 0.01, 0.02])
    monkeypatch.setattr(api, "perf_counter", lambda: next(ticks))
    monitor = api.LiveEplbOverhead(lambda layer: None, max_samples=1)
    monitor.active = True
    with monitor.cpu_span("eplb_step"), monitor.cpu_span("planner_wait"):
        pass
    with monitor.cpu_span("eplb_step"):
        pass
    summary = monitor.drain()
    assert len(summary["samples"]) == 1
    totals = {row["component"]: row for row in summary["totals"]}
    assert totals["planner_wait"]["sum_ms"] == pytest.approx(2)
    assert totals["eplb_step"]["count"] == 2
    assert totals["eplb_step"]["sum_ms"] == pytest.approx(18)
    # No combined total is exposed: the nested wait overlaps the enclosing step.
    assert "total_overhead_ms" not in summary


def test_drain_keeps_capacity_reserved_during_background_event_submission(api, events):
    monitor = api.LiveEplbOverhead(lambda layer: None, max_samples=1)
    monitor.active = True
    with monitor.device_span("eplb_step"):
        pass

    def query_while_background_submits():
        with monitor.device_span("migration_pipeline"):
            pass
        return False

    events[1][-1].query = query_while_background_submits
    summary = monitor.drain()
    assert summary["pending"] == 1 and summary["dropped"] == 1
    assert len(events[1]) == 2


def test_commit_notifications_are_cpu_only_even_while_dummy_is_inactive(api, events):
    committed = []
    layer = object()
    monitor = api.LiveEplbOverhead(committed.append)
    monitor.committed(layer)
    assert committed == [layer]
    assert events == ([], [])


def test_instrumentation_preserves_returns_and_errors(api, monkeypatch):
    ticks = iter([0, 1, 2, 3])
    monkeypatch.setattr(api, "perf_counter", lambda: next(ticks))
    monitor = api.LiveEplbOverhead(lambda layer: None)
    monitor.active = True

    class Worker:
        @api.monitor_call("planner_wait")
        def work(self, fail=False):
            if fail:
                raise ValueError("original failure")
            return 42

    worker = Worker()
    worker._eplb_diagnostic_monitor = monitor
    assert worker.work() == 42
    with pytest.raises(ValueError, match="original failure"):
        worker.work(fail=True)
    assert monitor.drain()["totals"][0]["count"] == 2


def test_unattached_hook_is_transparent_and_repeated_policy_attach_does_not_stack(api, monkeypatch):
    monitor = api.LiveEplbOverhead(lambda layer: None)
    ticks = iter([0, 1])
    monkeypatch.setattr(api, "perf_counter", lambda: next(ticks))

    class Policy:
        def rebalance_experts(self, value):
            return value + 1

    class Worker:
        @api.monitor_call("eplb_step", device=True)
        def work(self):
            return 10

    assert Worker().work() == 10
    policy = Policy()
    monitor.wrap_policy(policy)
    monitor.wrap_policy(policy)
    monitor.active = True
    assert policy.rebalance_experts(4) == 5
    assert monitor.drain()["totals"][0]["count"] == 1


def test_graph_capture_never_inserts_new_dynamic_events(api, monkeypatch, events):
    monitor = api.LiveEplbOverhead(lambda layer: None)
    monitor.active = True
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: True)
    with monitor.device_span("eplb_step"):
        pass
    assert events == ([], [])


def test_event_record_failure_does_not_change_model_execution(api, monkeypatch, events):
    monitor = api.LiveEplbOverhead(lambda layer: None, max_samples=1)
    monitor.active = True

    def unavailable(**kwargs):
        raise RuntimeError("timing unavailable")

    monkeypatch.setattr(torch.npu, "Event", unavailable)
    ran = []
    with monitor.device_span("eplb_step"):
        ran.append(True)
    assert ran == [True]
    summary = monitor.drain()
    assert summary["pending"] == 0 and summary["dropped"] == 1


@pytest.fixture
def loader_api(api, monkeypatch):
    # Load the real loader with only its external logger/group imports stubbed.
    # This exercises commit ordering without requiring an installed NPU runtime.
    dependencies = {
        "vllm.logger": {"logger": SimpleNamespace(debug=lambda *args: None, info=lambda *args: None)},
        "vllm.v1.utils": {"record_function_or_nullcontext": lambda *args: nullcontext()},
        "vllm_ascend.distributed.parallel_state": {"get_dynamic_eplb_group": lambda: None},
        "vllm_ascend.eplb.diagnostics.overhead": {"monitor_call": api.monitor_call},
    }
    for name, attributes in dependencies.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/eplb/core/eplb_device_transfer_loader.py"
    spec = importlib.util.spec_from_file_location("live_eplb_loader_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_loader_notifies_after_wait_maps_and_weight_copy(api, loader_api, events):
    actions = []
    layer = SimpleNamespace(layer_name="layer.2")
    loader = loader_api.D2DExpertWeightLoader()
    loader.eplb_adaptor = SimpleNamespace(
        moe_layers=[layer],
        num_moe_layers=1,
        do_update_expert_map=lambda *args: actions.append("expert_map"),
        do_update_log2phy_map=lambda *args: actions.append("routing_map"),
        do_update_expert_weight=lambda *args: actions.append("weight"),
    )
    loader.layer_id = 0
    loader.state = loader_api.ExpertWeightUpdateState.TRANSFERRING
    loader.recv_expert_list = [(0, 0)]
    monitor = api.LiveEplbOverhead(lambda actual: actions.append(("committed", actual)))
    monitor.active = True
    loader._eplb_diagnostic_monitor = monitor
    request = SimpleNamespace(wait=lambda: actions.append("wait"))
    loader.update_expert_map_and_weight([request])
    assert actions == ["wait", "expert_map", "routing_map", "weight", ("committed", layer)]
    assert loader.state == loader_api.ExpertWeightUpdateState.WAITING
    assert not any(name == "synchronize" for name, _ in events[0])
    records = monitor.drain(synchronize=True)
    assert {sample["component"] for sample in records["samples"]} == {"transfer_wait", "commit_apply"}
    assert {sample["layer"] for sample in records["samples"]} == {"layer.2"}


def test_uninstrumented_real_loader_does_not_require_npu_timing(loader_api, monkeypatch):
    monkeypatch.delattr(torch, "npu", raising=False)
    loader = loader_api.D2DExpertWeightLoader()
    actions = []
    loader.eplb_adaptor = SimpleNamespace(
        num_moe_layers=1,
        do_update_expert_map=lambda *args: actions.append("map"),
        do_update_log2phy_map=lambda *args: actions.append("route"),
        do_update_expert_weight=lambda *args: actions.append("weight"),
    )
    loader.layer_id = 0
    loader.state = loader_api.ExpertWeightUpdateState.TRANSFERRING
    loader.update_expert_map_and_weight([])
    assert actions == ["map", "route"]
    assert loader.state == loader_api.ExpertWeightUpdateState.WAITING


@pytest.fixture
def state_api(monkeypatch):
    # Exercise the real Ascend override. The upstream stub exposes whether
    # nested production work sees the correct post-sampling qualification.
    class UpstreamState:
        def step(self, is_dummy=False, is_profile=False, log_stats=False):
            monitor = getattr(self, "_eplb_diagnostic_monitor", None)
            self.calls.append((is_dummy, is_profile, log_stats, monitor is not None and monitor.active))
            if self.fail:
                raise RuntimeError("upstream step failed")
            return "upstream result"

    upstream = SimpleNamespace(EplbState=UpstreamState, EplbLayerState=type("UpstreamLayerState", (), {}))
    dependencies = {
        "vllm.distributed": {"get_ep_group": lambda: None},
        "vllm.distributed.eplb": {"eplb_state": upstream},
        "vllm_ascend.ops.fused_moe": {"eplb": SimpleNamespace()},
    }
    for name, attributes in dependencies.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/distributed/eplb/state.py"
    spec = importlib.util.spec_from_file_location("live_eplb_state_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = module.AscendEplbState.__new__(module.AscendEplbState)
    state.calls = []
    state.fail = False
    return state


def test_mrv2_sampling_step_consumes_real_forward_once(api, state_api, events):
    monitor = api.LiveEplbOverhead(lambda layer: None)
    state_api._eplb_diagnostic_monitor = monitor
    # execute_model has returned and disabled timing before sample_tokens.
    monitor.active = False
    monitor.pending_real = True
    assert state_api.step(log_stats=True) == "upstream result"
    assert state_api.calls == [(False, False, True, True)]
    assert not monitor.active and not monitor.pending_real
    assert len(events[1]) == 2
    assert not any(name == "synchronize" for name, _ in events[0])

    # A second controller call must not reuse the same forward's permission.
    assert state_api.step() == "upstream result"
    assert state_api.calls[-1] == (False, False, False, False)
    assert len(events[1]) == 2
    summary = monitor.drain(synchronize=True)
    assert {(row["component"], row["kind"], row["count"]) for row in summary["totals"]} == {
        ("eplb_step", "host", 1),
        ("eplb_step", "device", 1),
    }


@pytest.mark.parametrize("excluded", ["is_dummy", "is_profile"])
def test_mrv2_dummy_and_profile_discard_stale_real_qualification(api, state_api, events, excluded):
    monitor = api.LiveEplbOverhead(lambda layer: None)
    state_api._eplb_diagnostic_monitor = monitor
    monitor.active = True
    monitor.pending_real = True
    assert state_api.step(**{excluded: True}) == "upstream result"
    assert state_api.calls == [(excluded == "is_dummy", excluded == "is_profile", False, False)]
    assert not monitor.active and not monitor.pending_real
    state_api.step()
    assert state_api.calls[-1] == (False, False, False, False)
    assert events == ([], [])
    assert monitor.drain()["totals"] == []


def test_mrv2_failed_step_always_consumes_qualification(api, state_api, events):
    monitor = api.LiveEplbOverhead(lambda layer: None)
    state_api._eplb_diagnostic_monitor = monitor
    monitor.pending_real = True
    state_api.fail = True
    with pytest.raises(RuntimeError, match="upstream step failed"):
        state_api.step()
    assert not monitor.active and not monitor.pending_real
    state_api.fail = False
    state_api.step()
    assert state_api.calls[-1][-1] is False
    assert len(events[1]) == 2


def test_mrv2_unattached_step_is_transparent(state_api, monkeypatch):
    monkeypatch.delattr(torch, "npu", raising=False)
    assert state_api.step(is_dummy=True, log_stats=True) == "upstream result"
    assert state_api.calls == [(True, False, True, False)]
