# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable tests: pytest --noconftest tests/ut/eplb/test_diagnostics.py."""

import importlib
import json
import subprocess
import sys
from concurrent.futures import Future
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def api():
    # The parent package initializes vLLM logging. In the portable CPU test only,
    # bypass that initializer; import all diagnostic implementation files unchanged.
    # A normal installed vLLM test session uses the real parent package.
    stubbed = "vllm_ascend" not in sys.modules and importlib.util.find_spec("vllm") is None
    original_modules = set(sys.modules)
    if stubbed:
        package = ModuleType("vllm_ascend")
        package.__path__ = [str(Path(__file__).resolve().parents[3] / "vllm_ascend")]
        sys.modules["vllm_ascend"] = package
    modules = {
        name: importlib.import_module(f"vllm_ascend.eplb_diagnostics.{name}")
        for name in ("config", "probe", "schema", "runtime", "report")
    }
    yield SimpleNamespace(**modules)
    if stubbed:
        for name in set(sys.modules) - original_modules:
            if name == "vllm_ascend" or name.startswith("vllm_ascend."):
                del sys.modules[name]


class CpuBackend:
    def copy(self, source):
        return source.clone()

    def event(self, timing=False):
        return object()

    def finish(self, ready, start, end):
        return 2.5  # Explicit fake timing; these tests make no NPU timing claim.


def make_layer(api):
    return SimpleNamespace(
        eplb_diagnostic_probe=api.probe.ExpertLoadProbe(2, "cpu"),
        _use_v2_model_runner=False,
        dynamic_eplb=False,
        moe_config=SimpleNamespace(num_logical_experts=4, ep_size=2, ep_rank=0),
        n_shared_experts=0,
        mix_placement=False,
        ascend_expert_map=torch.tensor([1, -1, 0, -1]),
        router=None,
        log2phy=None,
    )


@pytest.mark.parametrize("group_type,values", [(1, [2, 3]), (0, [2, 5])])
def test_normalization_and_stable_storage(api, group_type, values):
    probe = api.probe.ExpertLoadProbe(2, "cpu")
    ptr = probe.totals.data_ptr()
    values = torch.tensor(values)
    original = values.clone()
    before = probe.totals.tolist()
    for _ in range(3):
        probe.record(values, group_type, "MC2CommImpl")
    result = api.schema.decode_sample(before, probe.totals.tolist(), 2)
    assert result["expert_assignments"] == [6, 9]
    assert result["calls"] == result["valid_calls"] == 3
    assert result["comm_calls"] == [0, 0, 3, 0, 0]
    assert torch.equal(values, original)
    assert probe.totals.data_ptr() == ptr


@pytest.mark.parametrize("values,group_type", [(None, 1), ([1], 1), ([1, 2], 2)])
def test_missing_counts_are_not_zero_load(api, values, group_type):
    probe = api.probe.ExpertLoadProbe(2, "cpu")
    before = probe.totals.tolist()
    probe.record(None if values is None else torch.tensor(values), group_type, "unknown")
    result = api.schema.decode_sample(before, probe.totals.tolist(), 2)
    assert result["quality"] == "backend_counts_unavailable"
    assert result["expert_assignments"] is None


def test_no_execution_and_counter_reset(api):
    probe = api.probe.ExpertLoadProbe(2, "cpu")
    before = probe.totals.tolist()
    assert api.schema.decode_sample(before, before, 2)["quality"] == "not_executed"
    probe.record(torch.tensor([3, 2]), 1, "MC2CommImpl")
    result = api.schema.decode_sample(probe.totals.tolist(), before, 2)
    assert result["quality"] == "counter_reset_or_invalid_counts"


def test_actual_noncontiguous_layout_and_replicas(api):
    assert api.schema.local_logical_ids(
        {"map_semantics": "logical_to_local", "global_to_local": [1, -1, 0, -1]}, 2
    ) == [2, 0]
    assert api.schema.local_logical_ids(
        {
            "map_semantics": "physical_to_local",
            "global_to_local": [-1, 1, -1, -1, 0],
            "logical_to_physical": [[0, 4], [1, -1], [2, 3]],
        },
        2,
    ) == [0, 1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "observe"},
        {"mode": "unknown"},
        {"sample_interval": 0},
        {"max_pending": 0},
        {"warmup_steps": -1},
        {"run_id": "../bad"},
        {"max_samples": 0},
        {"unsupported": 1},
    ],
)
def test_config_rejects_invalid_input(api, kwargs):
    with pytest.raises(ValueError):
        api.config.EplbDiagnosticsConfig(**kwargs)


def test_snapshot_is_immutable_and_layout_transition_detected(api, tmp_path):
    config = api.config.EplbDiagnosticsConfig(
        mode="observe",
        run_id="test",
        output_dir=str(tmp_path),
        sample_interval=1,
        warmup_steps=0,
    )
    layer = make_layer(api)
    recorder = api.runtime.DiagnosticsRecorder(config, [("layers.2", layer)], {"rank": 0}, CpuBackend())
    assert recorder.begin({"phase": "decode"})
    layer.eplb_diagnostic_probe.record(torch.tensor([7, 1]), 1, "MC2CommImpl")
    layer.ascend_expert_map.copy_(torch.tensor([0, -1, 1, -1]))
    recorder.end()
    layer.eplb_diagnostic_probe.totals.zero_()
    layer.ascend_expert_map.fill_(-1)
    recorder.close()
    row = json.loads(recorder.path.read_text())
    assert row["layers"][0]["expert_assignments"] == [7, 1]
    assert row["layers"][0]["layout_changed"] is True
    assert row["layers"][0]["local_logical_ids"] == [0, 2]
    assert row["execute_stream_ms"] == 2.5


def test_sampling_backpressure_and_sample_budget(api, tmp_path):
    config = api.config.EplbDiagnosticsConfig(
        mode="observe",
        run_id="test",
        output_dir=str(tmp_path),
        sample_interval=2,
        warmup_steps=1,
        max_pending=1,
        max_samples=1,
    )
    recorder = api.runtime.DiagnosticsRecorder(config, [("layer", make_layer(api))], {"rank": 0}, CpuBackend())
    assert not recorder.begin({})
    pending = Future()
    recorder.pending.append(pending)
    assert not recorder.begin({})
    assert recorder.drop_reasons == {"writer_backpressure": 1}
    pending.set_result(None)
    assert not recorder.begin({})
    assert recorder.begin({})
    recorder.end()
    recorder.close()
    assert not recorder.begin({})


def test_disabled_wrapper_does_not_touch_device_or_recorder(api):
    runner = SimpleNamespace(ascend_config=SimpleNamespace(eplb_diagnostics=api.config.EplbDiagnosticsConfig()))
    called = []

    @api.runtime.record_diagnostics
    def execute(runner, output):
        called.append(output)
        return output

    token = object()
    assert execute(runner, token) is token
    assert called == [token]
    assert not hasattr(runner, "_eplb_diagnostics_recorder")


def row(rank, step, loads, *, group=(0, 1), phase="decode", call=None):
    return {
        "schema_version": 1,
        "run_id": "test",
        "host": "host",
        "pid": rank + 100,
        "rank": rank,
        "local_step": step,
        "execute_stream_ms": 2.0,
        "groups": {"ep": {"ranks": list(group)}},
        "batch": {"phase": phase, "graph_mode": "FULL"},
        "layers": [
            {
                "layer": "layers.2",
                "quality": "ok",
                "expert_assignments": loads,
                "layout_changed": False,
                "layout_end": {"ep_size": len(group)},
                "calls": 1,
                "call_start": call if call is not None else step,
                "call_end": (call if call is not None else step) + 1,
                "comm_calls": [1, 0, 0, 0, 0],
                "comm_methods": ["AllGatherCommImpl", "AlltoAllCommImpl", "MC2CommImpl", "FusedMC2CommImpl", "unknown"],
            }
        ],
    }


def test_zero_rank_and_alternating_hotspots_survive_temporal_aggregation(api):
    records = [row(0, 1, [10, 0]), row(1, 1, [0, 0]), row(0, 2, [0, 0]), row(1, 2, [10, 0])]
    report = api.report.summarize(records, aligned_collective_ordinals=True)
    skew = report["rank_work_skew"][0]
    assert skew["samples"] == 2
    assert skew["rank_max_mean_p50"] == 2.0
    assert skew["ideal_work_reduction_proxy_p50"] == 0.5


def test_dp_steps_are_not_alignment_keys_and_default_does_not_claim_gain(api):
    records = [row(0, 3, [8, 0], call=7), row(1, 99, [2, 0], call=7, phase="prefill")]
    unverified = api.report.summarize(records)
    assert not unverified["rank_work_skew"]
    assert unverified["quality_counts"]["complete_group_alignment_unverified"] == 1
    verified = api.report.summarize(records, aligned_collective_ordinals=True)
    assert verified["rank_work_skew"][0]["rank_max_mean_p50"] == 1.6
    assert verified["recommendation"] == "insufficient_evidence_to_claim_EPLB_speedup"


def test_missing_rank_restart_and_layout_switch_excluded(api):
    records = [row(0, 1, [8, 0]), row(0, 2, [8, 0]), row(1, 2, [2, 0])]
    records[-1]["layers"][0]["layout_changed"] = True
    result = api.report.summarize(records, aligned_collective_ordinals=True)
    assert not result["rank_work_skew"]
    assert result["quality_counts"]["layout_changed_within_sample"] == 1
    records = [row(0, 1, [8, 0]), row(1, 1, [2, 0]), row(0, 2, [8, 0])]
    records[-1]["pid"] = 999
    assert not api.report.summarize(records, aligned_collective_ordinals=True)["rank_work_skew"]


def test_distinct_pp_groups_and_duplicate_files(api):
    records = [row(0, 1, [8, 0]), row(1, 1, [2, 0])]
    records += [row(2, 1, [5, 0], group=(2, 3)), row(3, 1, [5, 0], group=(2, 3))]
    records.append(deepcopy(records[0]))
    result = api.report.summarize(records, aligned_collective_ordinals=True)
    assert len(result["rank_work_skew"]) == 2
    assert result["quality_counts"]["duplicate_step"] == 1


def test_multiple_invocations_not_collapsed_into_rank_skew(api):
    records = [row(0, 1, [8, 0]), row(1, 1, [2, 0])]
    for record in records:
        record["layers"][0]["calls"] = 2
    result = api.report.summarize(records, aligned_collective_ordinals=True)
    assert not result["rank_work_skew"]
    assert result["quality_counts"]["multiple_invocations_not_joined"] == 2


def test_malformed_trace_reports_location(api, tmp_path):
    file = tmp_path / "broken.jsonl"
    file.write_text("{broken\n")
    with pytest.raises(ValueError, match="broken.jsonl:1"):
        list(api.report.load_records([file]))


def test_start_waits_for_warmup_and_resets_capture_counters(api, tmp_path, monkeypatch):
    config = api.config.EplbDiagnosticsConfig(
        mode="observe",
        run_id="test",
        output_dir=str(tmp_path),
        warmup_steps=0,
    )
    layer = make_layer(api)
    # Model loading may create the buffers under inference_mode; arming must be
    # allowed to reset those inference tensors after capture.
    with torch.inference_mode():
        layer.eplb_diagnostic_probe = api.probe.ExpertLoadProbe(2, "cpu")
        layer.eplb_diagnostic_probe.record(torch.tensor([9, 9]), 1, "MC2CommImpl")
    recorder = api.runtime.DiagnosticsRecorder(config, [("layer", layer)], {"rank": 0}, CpuBackend())
    runner = SimpleNamespace(ascend_config=SimpleNamespace(eplb_diagnostics=config))
    monkeypatch.setattr(api.runtime, "_create_recorder", lambda runner: recorder)

    @api.runtime.record_diagnostics
    def warmup(runner, output):
        return "warmup"

    assert warmup(runner, None) == "warmup"  # No NPU API or metadata access before arming.
    assert not hasattr(runner, "_eplb_diagnostics_recorder")
    api.runtime.start_diagnostics(runner)
    assert runner._eplb_diagnostics_ready
    assert layer.eplb_diagnostic_probe.totals.tolist() == [0] * 9
    recorder.close()


def test_idle_dp_dummy_is_not_sampled(api, tmp_path, monkeypatch):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: False), raising=False)
    config = api.config.EplbDiagnosticsConfig(
        mode="observe",
        run_id="test",
        output_dir=str(tmp_path),
        warmup_steps=0,
    )
    layer = make_layer(api)
    recorder = api.runtime.DiagnosticsRecorder(config, [("layer", layer)], {"rank": 0}, CpuBackend())
    monkeypatch.setattr(recorder, "_snapshots", lambda _: pytest.fail("dummy must not snapshot"))
    monkeypatch.setattr(recorder.backend, "event", lambda **_: pytest.fail("dummy must not record timing"))
    runner = SimpleNamespace(
        ascend_config=SimpleNamespace(eplb_diagnostics=config),
        _eplb_diagnostics_ready=True,
        _eplb_diagnostics_recorder=recorder,
    )

    def dummy(num_tokens, uniform_decode):
        assert num_tokens == 1 and uniform_decode
        layer.eplb_diagnostic_probe.record(torch.tensor([8, 0]), 1, "MC2CommImpl")

    runner._dummy_run = dummy
    api.runtime.run_dummy_batch(runner, 1)
    recorder.close()
    assert recorder.path.read_text() == ""
    assert recorder.collected == recorder.dropped == 0
    assert recorder.model_step == 1


def test_diagnostics_abort_error_does_not_mask_inference_error(api, monkeypatch):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: False), raising=False)

    def abort():
        raise RuntimeError("diagnostics failed")

    recorder = SimpleNamespace(active=None, begin=lambda batch: None, abort=abort)
    runner = SimpleNamespace(
        ascend_config=SimpleNamespace(eplb_diagnostics=api.config.EplbDiagnosticsConfig(mode="observe", run_id="test")),
        _eplb_diagnostics_ready=True,
        _eplb_diagnostics_recorder=recorder,
    )

    @api.runtime.record_diagnostics
    def execute(runner, scheduler_output):
        raise ValueError("original inference failure")

    with pytest.raises(ValueError, match="original inference failure"):
        execute(runner, SimpleNamespace(total_num_scheduled_tokens=1, num_scheduled_tokens={"r": 1}))


def test_mrv2_internal_dummy_does_not_invent_user_work(api, tmp_path, monkeypatch):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: False), raising=False)
    config = api.config.EplbDiagnosticsConfig(mode="observe", run_id="dummy", output_dir=str(tmp_path), warmup_steps=0)
    layer = make_layer(api)
    recorder = api.runtime.DiagnosticsRecorder(config, [("layer", layer)], {"rank": 0}, CpuBackend())
    runner = SimpleNamespace(
        ascend_config=SimpleNamespace(eplb_diagnostics=config),
        _eplb_diagnostics_ready=True,
        _eplb_diagnostics_recorder=recorder,
    )

    @api.runtime.record_diagnostics
    def execute(runner, scheduler_output, *, dummy_run):
        assert dummy_run
        layer.eplb_diagnostic_probe.record(torch.tensor([4, 0]), 1, "MC2CommImpl")

    # MRv2's idle path constructs a synthetic SchedulerOutput for graph dispatch.
    execute(runner, SimpleNamespace(total_num_scheduled_tokens=1, num_scheduled_tokens={"dummy": 1}), dummy_run=True)
    recorder.close()
    assert recorder.path.read_text() == ""
    assert recorder.collected == 0
    assert recorder.model_step == 1


def test_memory_budget_is_checked_before_copy(api, tmp_path):
    config = api.config.EplbDiagnosticsConfig(
        mode="observe",
        run_id="test",
        output_dir=str(tmp_path),
        warmup_steps=0,
        max_snapshot_mb=1,
    )
    layer = make_layer(api)
    layer.ascend_expert_map = torch.zeros(140_000, dtype=torch.int32)
    recorder = api.runtime.DiagnosticsRecorder(config, [("layer", layer)], {"rank": 0}, CpuBackend())
    assert not recorder.begin({})
    assert recorder.drop_reasons == {"snapshot_memory_budget": 1}
    recorder.close()


def test_expanded_replica_table_does_not_exhaust_snapshot_budget(api, tmp_path):
    config = api.config.EplbDiagnosticsConfig(
        mode="observe", run_id="replicas", output_dir=str(tmp_path), warmup_steps=0, max_snapshot_mb=1
    )
    layer = make_layer(api)
    layer._use_v2_model_runner = True
    layer.ascend_expert_map = torch.full((129,), -1)
    layer.ascend_expert_map[0] = 0
    layer.ascend_expert_map[128] = 1
    layer.moe_config.num_logical_experts = 128
    layer.moe_config.num_experts = 129
    mapping = torch.full((128, 1024), -1)
    mapping[:, 0] = torch.arange(128)
    mapping[0, 1] = 128
    replicas = torch.ones(128, dtype=torch.int64)
    replicas[0] = 2
    layer.router = SimpleNamespace(
        eplb_state=SimpleNamespace(
            logical_to_physical_map=mapping,
            logical_replica_count=replicas,
            expert_replica_routing_table=torch.zeros(1024, 128, dtype=torch.int32),
        )
    )
    recorder = api.runtime.DiagnosticsRecorder(config, [("layer", layer)], {"rank": 0}, CpuBackend())
    assert recorder.begin({})
    layer.eplb_diagnostic_probe.record(torch.tensor([3, 1]), 1, "MC2CommImpl")
    recorder.end()
    recorder.close()
    sample = json.loads(recorder.path.read_text())["layers"][0]
    assert sample["quality"] == "ok"
    assert sample["local_logical_ids"] == [0, 0]
    assert sample["layout_end"]["logical_replica_count"] == replicas.tolist()
    assert sample["layout_end"]["replica_routing_table_shape"] == [1024, 128]
    assert sample["layout_end"]["logical_to_physical_storage_shape"] == [128, 1024]
    assert sample["layout_end"]["logical_to_physical"] == mapping[:, :2].tolist()
    assert "expert_replica_routing_table" not in sample["layout_end"]
    assert not recorder.drop_reasons


def test_cli_runs_without_vllm_or_torch_imports(tmp_path):
    trace = tmp_path / "sample.jsonl"
    trace.write_text(json.dumps(row(0, 1, [8, 0])) + "\n", encoding="utf-8")
    output = tmp_path / "summary.json"
    script = Path(__file__).resolve().parents[3] / "vllm_ascend/eplb_diagnostics/report.py"
    subprocess.run(
        [sys.executable, "-I", str(script), str(trace), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["quality_counts"]["incomplete_group_sample"] == 1
    assert result["recommendation"] == "insufficient_evidence_to_claim_EPLB_speedup"


def routed_rows(step, loads=(8, 0, 2, 0)):
    rows = [row(0, step, list(loads[:2]), call=step - 1), row(1, step, list(loads[2:]), call=step - 1)]
    for rank, record in enumerate(rows):
        record["groups"]["mc2"] = {"ranks": [0, 1]}
        record["groups"]["tp"] = {"ranks": [rank]}
        record["batch"]["scheduled_tokens"] = sum(loads) if rank == 0 else 0
        record["alignment_contract"] = "post_warmup_executed_collective_v2"
        record["expected_model_calls"] = step
        record["batch"]["padded_tokens"] = 16
        layer = record["layers"][0]
        layer["comm_calls"] = [0, 0, 1, 0, 0]
        layer["alignment_eligible"] = True
        layer["layout_fingerprint"] = f"static-{rank}"
        layer["layout_end"] = {
            "ep_size": 2,
            "num_logical_experts": 4,
            "experts_per_token": 1,
            "mixed_shared_placement": False,
            "map_semantics": "physical_to_local",
            "dynamic_eplb": False,
            "global_to_local": [0, 1, -1, -1] if rank == 0 else [-1, -1, 0, 1],
        }
        layer["routing"] = {
            "semantics": "valid_scheduled_mc2_source_destinations",
            "physical_assignments": list(loads) if rank == 0 else [0, 0, 0, 0],
        }
    return rows


def test_masked_routes_exclude_padding_and_invalid_ids(api):
    probe = api.probe.ExpertLoadProbe(2, "cpu", 4)
    ids = torch.tensor([[0, 3], [1, 2], [-1, 9]])
    mask = torch.tensor([True, False, False])
    pointer = probe.route_totals.data_ptr()
    probe.record_routes(ids, mask)
    assert probe.route_totals.tolist() == [1, 0, 0, 1, 1, 0]
    mask.copy_(torch.tensor([False, False, True]))
    probe.record_routes(ids, mask)
    assert probe.route_totals.tolist() == [1, 0, 0, 1, 2, 2]
    assert probe.route_totals.data_ptr() == pointer
    probe.record_routes(ids, None)
    assert probe.route_totals[-2].item() == 2


def test_persistent_imbalance_uses_valid_routes_not_padding(api):
    rows = [r for step in range(1, 17) for r in routed_rows(step)]
    # Backend padding reverses which rank appears busiest; valid routes must win.
    for r in rows:
        if r["rank"] == 1:
            r["layers"][0]["expert_assignments"][0] += 30
    result = api.report.summarize(rows)
    assessment = result["persistent_imbalance"][0]
    assert assessment["status"] == "persistent_rank_imbalance_observed"
    assert assessment["hot_expert_adjacent_window_jaccard"] == 1
    assert assessment["hotspot_status"] == "stable_hot_experts_observed"
    assert assessment["unattributed_backend_work"] == 16 * 30
    assert assessment["windows"][0]["busiest_rank_counts"] == {0: 8}
    assert assessment["windows"][0]["rank_skew_p50"] == 1.6
    assert result["recommendation"] == "insufficient_evidence_to_claim_EPLB_speedup"


def test_alternating_rank_and_expert_hotspots_not_hidden_by_average(api):
    rows = [r for step in range(1, 17) for r in routed_rows(step, (10, 0, 0, 0) if step <= 8 else (0, 0, 10, 0))]
    assessment = api.report.summarize(rows)["persistent_imbalance"][0]
    assert assessment["status"] == "persistent_rank_imbalance_observed"
    assert assessment["hot_expert_adjacent_window_jaccard"] == 0
    assert assessment["hotspot_status"] == "hot_experts_shift_observed"
    assert assessment["windows"][0]["busiest_rank_counts"] == {0: 8}
    assert assessment["windows"][1]["busiest_rank_counts"] == {1: 8}


def test_balanced_uniform_work_has_no_hot_experts(api):
    rows = [r for step in range(1, 17) for r in routed_rows(step, (2, 2, 2, 2))]
    assessment = api.report.summarize(rows)["persistent_imbalance"][0]
    assert assessment["status"] == "no_persistent_imbalance_observed"
    assert assessment["hot_expert_adjacent_window_jaccard"] is None
    assert assessment["hotspot_status"] == "no_hot_experts_observed"
    assert all(not w["hot_logical_experts"] for w in assessment["windows"])


def test_expert_imbalance_is_independent_of_rank_balance(api):
    rows = [r for step in range(1, 17) for r in routed_rows(step, (5, 0, 5, 0))]
    assessment = api.report.summarize(rows)["workload_persistence"][0]
    assert assessment["status"] == "no_persistent_imbalance_observed"
    diagnosis = assessment["expert_diagnosis"]
    assert diagnosis["load_status"] == "expert_imbalance_observed"
    assert diagnosis["persistent_hot_experts"] == [0, 2]
    assert diagnosis["expert_max_mean_p50"] == 2
    assert diagnosis["total_valid_assignments"] == 160
    first, cold = diagnosis["experts"][:2]
    assert first["load_share"] == 0.5 and first["load_over_mean"] == 2
    assert first["hot_window_fraction"] == 1
    assert first["longest_consecutive_hot_windows"] == 2
    assert first["calls_in_longest_hot_window_run"] == 16
    assert cold["valid_assignments"] == cold["hot_windows"] == 0


def test_expert_diagnosis_distinguishes_transient_and_persistent_hotspots(api):
    rows = [r for step in range(1, 17) for r in routed_rows(step, (10, 0, 0, 0) if step <= 8 else (0, 0, 10, 0))]
    diagnosis = api.report.summarize(rows)["workload_persistence"][0]["expert_diagnosis"]
    assert diagnosis["load_status"] == "expert_imbalance_observed"
    assert diagnosis["hotspot_status"] == "no_persistent_hot_experts_observed"
    assert diagnosis["persistent_hot_experts"] == []
    for expert in (0, 2):
        assert diagnosis["experts"][expert]["hot_window_fraction"] == 0.5
        assert diagnosis["experts"][expert]["longest_consecutive_hot_windows"] == 1


def test_one_expert_can_persist_while_other_hotspots_change(api):
    rows = [r for step in range(1, 17) for r in routed_rows(step, (5, 5, 0, 0) if step <= 8 else (5, 0, 5, 0))]
    assessment = api.report.summarize(rows)["workload_persistence"][0]
    assert assessment["hotspot_status"] == "hot_experts_shift_observed"
    assert assessment["expert_diagnosis"]["persistent_hot_experts"] == [0]
    assert assessment["expert_diagnosis"]["hotspot_status"] == "persistent_hot_experts_observed"


def test_enablement_screening_rejects_hot_experts_on_balanced_ranks(api):
    rows = [r for step in range(1, 25) for r in routed_rows(step, (5, 0, 5, 0))]
    assessment = api.report.summarize(rows)["workload_persistence"][0]
    assert assessment["expert_diagnosis"]["persistent_hot_experts"] == [0, 2]
    assert assessment["enablement_assessment"]["decision"] == "not_recommended_from_observed_load"
    assert assessment["enablement_assessment"]["estimated_speedup"] is None
    assert assessment["expert_diagnosis"]["experts"][2]["observed_owner_ranks"] == [1]


def test_held_out_placement_reduces_concentrated_hot_experts(api):
    rows = [r for step in range(1, 25) for r in routed_rows(step, (8, 8, 0, 0))]
    assessment = api.report.summarize(rows)["workload_persistence"][0]
    placement = assessment["placement_diagnosis"]
    assert placement["evaluated_windows"] == 2
    assert placement["held_out_peak_work_reduction"] == 0.5
    for pair in placement["pairs"]:
        assert pair["train_call_end"] == pair["evaluation_call_start"]
        assert pair["candidate_expert_ranks"].count(0) == pair["candidate_expert_ranks"].count(1) == 2
        assert pair["current_peak_assignments_sum"] == 128
        assert pair["candidate_peak_assignments_sum"] == 64
    assert assessment["enablement_assessment"]["decision"] == "enable_candidate"
    strict = api.report.summarize(rows, min_work_reduction=0.6)["workload_persistence"][0]
    assert strict["enablement_assessment"]["decision"] == "insufficient_evidence"


def test_single_hot_expert_cannot_be_split_by_relocation(api):
    rows = [r for step in range(1, 25) for r in routed_rows(step, (10, 0, 0, 0))]
    assessment = api.report.summarize(rows)["workload_persistence"][0]
    assert assessment["expert_diagnosis"]["persistent_hot_experts"] == [0]
    assert assessment["placement_diagnosis"]["held_out_peak_work_reduction"] == 0
    # This candidate cannot model replicas: it cannot rule out every EPLB policy.
    assert assessment["enablement_assessment"]["decision"] == "insufficient_evidence"


def test_placement_candidate_does_not_look_ahead(api):
    rows = [r for step in range(1, 17) for r in routed_rows(step, (8, 8, 0, 0) if step <= 8 else (8, 0, 0, 8))]
    placement = api.report.summarize(rows)["workload_persistence"][0]["placement_diagnosis"]
    pair = placement["pairs"][0]
    assert pair["current_peak_assignments_sum"] == 64
    assert pair["candidate_peak_assignments_sum"] == 128
    assert pair["held_out_peak_work_reduction"] == -1
    assert pair["regressed_call_fraction"] == 1
    assert placement["status"] == "insufficient_adjacent_windows"


def test_placement_never_evaluates_across_sampling_gaps(api):
    rows = [r for step in [*range(1, 9), *range(17, 25)] for r in routed_rows(step)]
    placement = api.report.summarize(rows)["workload_persistence"][0]["placement_diagnosis"]
    assert placement["pairs"] == []
    assert placement["skipped_pairs"] == {"discontinuous_windows": 1}


def test_replica_placement_is_not_silently_treated_as_unique_ownership(api):
    rows = [r for step in range(1, 25) for r in routed_rows(step, (3, 0, 7, 0))]
    for r in rows:
        r["layers"][0]["layout_end"].update(
            dynamic_eplb=True, num_logical_experts=3, logical_to_physical=[[0, 2], [1], [3]]
        )
    placement = api.report.summarize(rows)["workload_persistence"][0]["placement_diagnosis"]
    assert placement["pairs"] == []
    assert placement["skipped_pairs"] == {"replicas_or_changing_ownership_unsupported": 2}


def test_slot_candidate_preserves_unequal_noncontiguous_capacity(api):
    owners = [2, 0, 2, 2, 0]
    candidate = api.report.slot_balanced_candidate([10, 9, 8, 1, 0], owners, 3)
    assert [candidate.count(rank) for rank in range(3)] == [2, 0, 3]


def test_current_placement_must_replay_before_counterfactual_is_accepted(api):
    first, reason = api.report.routed_group_sample([(r, r["layers"][0]) for r in routed_rows(1)])
    assert reason is None
    # A broken owner mapping must not produce a plausible-looking recommendation.
    first["logical_owners"] = [1, 1, 0, 0]
    samples = [dict(first, ordinal=i) for i in range(24)]
    assessment = api.report.persistence_summary(
        samples, [0, 1], window_size=8, min_windows=2, skew_threshold=1.2, persistent_fraction=0.8
    )
    assert assessment["placement_diagnosis"]["skipped_pairs"] == {"current_placement_replay_mismatch": 2}
    assert assessment["enablement_assessment"]["decision"] == "insufficient_evidence"


@pytest.mark.parametrize("minimum", [0, -0.1, 1.1, float("nan")])
def test_enablement_threshold_validation(api, minimum):
    with pytest.raises(ValueError, match="min_work_reduction"):
        api.report.summarize([], min_work_reduction=minimum)


def test_expert_diagnosis_does_not_join_hot_streaks_across_gaps(api):
    rows = [r for step in [*range(1, 9), *range(17, 25)] for r in routed_rows(step)]
    diagnosis = api.report.summarize(rows)["workload_persistence"][0]["expert_diagnosis"]
    assert diagnosis["experts"][0]["hot_window_fraction"] == 1
    assert diagnosis["experts"][0]["longest_consecutive_hot_windows"] == 1
    assert diagnosis["hotspot_status"] == "insufficient_contiguous_samples"
    assert diagnosis["persistent_hot_experts"] == []


def test_expert_diagnosis_streak_breaks_when_expert_cools(api):
    rows = [r for step in range(1, 25) for r in routed_rows(step, (0, 0, 10, 0) if 9 <= step <= 16 else (10, 0, 0, 0))]
    diagnosis = api.report.summarize(rows, persistent_fraction=0.6)["workload_persistence"][0]["expert_diagnosis"]
    first = diagnosis["experts"][0]
    assert first["hot_window_fraction"] == pytest.approx(2 / 3)
    assert first["longest_consecutive_hot_windows"] == 1
    assert not first["persistent_hot"]


@pytest.mark.parametrize("calls", [4, 16])
def test_expert_diagnosis_distinguishes_uniform_and_insufficient(api, calls):
    rows = [r for step in range(1, calls + 1) for r in routed_rows(step, (2, 2, 2, 2))]
    diagnosis = api.report.summarize(rows)["workload_persistence"][0]["expert_diagnosis"]
    if calls == 4:
        assert diagnosis["load_status"] == diagnosis["hotspot_status"] == "insufficient_contiguous_samples"
        assert not diagnosis["experts"]
    else:
        assert diagnosis["load_status"] == "no_expert_imbalance_observed"
        assert diagnosis["hotspot_status"] == "no_hot_experts_observed"
        assert all(e["load_over_mean"] == 1 and not e["persistent_hot"] for e in diagnosis["experts"])


def test_gaps_and_layout_changes_do_not_form_contiguous_windows(api):
    rows = [r for step in [1, 2, 4, 5, 7, 8, 10, 11] for r in routed_rows(step)]
    assessment = api.report.summarize(rows, window_size=4)["persistent_imbalance"][0]
    assert assessment["status"] == "insufficient_contiguous_samples"
    assert assessment["max_observed_contiguous_calls"] == 2
    assert assessment["hotspot_status"] == "insufficient_adjacent_windows"
    rows = [r for step in range(1, 9) for r in routed_rows(step)]
    for r in rows:
        r["layers"][0]["layout_fingerprint"] = str(r["local_step"])
    assert api.report.summarize(rows, window_size=4)["persistent_imbalance"][0]["max_observed_contiguous_calls"] == 1


@pytest.mark.parametrize(
    "fault", ["origin", "mask", "owner", "conservation", "restart", "shared", "missing_rank", "source", "tp"]
)
def test_routing_assessment_fails_closed(api, fault):
    rows = [r for step in range(1, 17) for r in routed_rows(step)]
    for r in rows:
        layer = r["layers"][0]
        if fault == "origin":
            layer["alignment_eligible"] = False
        elif fault == "mask":
            layer["routing"]["semantics"] = "unavailable"
        elif fault == "owner":
            layer["layout_end"]["global_to_local"] = [0, 1, -1, -1]
        elif fault == "conservation":
            layer["expert_assignments"] = [0, 0]
        elif fault == "restart" and r["local_step"] > 8:
            r["pid"] += 1000
        elif fault == "shared":
            layer["layout_end"]["mixed_shared_placement"] = True
        elif fault == "source":
            r["batch"]["scheduled_tokens"] += 1
        elif fault == "tp":
            r["groups"]["tp"]["ranks"] = [0, 1, 2]
    if fault == "missing_rank":
        rows = [r for r in rows if r["rank"] == 0]
    assert not api.report.summarize(rows)["persistent_imbalance"]


def test_replica_counts_aggregate_to_logical_hotspot(api):
    rows = [r for step in range(1, 17) for r in routed_rows(step, (3, 0, 7, 0))]
    for r in rows:
        layout = r["layers"][0]["layout_end"]
        layout.update(dynamic_eplb=True, num_logical_experts=3, logical_to_physical=[[0, 2], [1], [3]])
    assessment = api.report.summarize(rows)["persistent_imbalance"][0]
    assert assessment["windows"][0]["hot_logical_experts"] == [0]


def test_burst_sampling_and_validation(api, tmp_path):
    with pytest.raises(ValueError):
        api.config.EplbDiagnosticsConfig(sample_interval=4, burst_size=5)
    config = api.config.EplbDiagnosticsConfig(
        mode="observe",
        run_id="burst",
        output_dir=str(tmp_path),
        warmup_steps=0,
        sample_interval=8,
        burst_size=3,
        max_samples=20,
        max_pending=16,
    )
    recorder = api.runtime.DiagnosticsRecorder(config, [("layer", make_layer(api))], {"rank": 0}, CpuBackend())
    sampled = []
    for step in range(1, 13):
        if recorder.begin({}):
            sampled.append(step)
            recorder.end()
    recorder.close()
    assert sampled == [1, 2, 3, 9, 10, 11]


def test_runtime_routes_dummy_and_alignment_contract(api, tmp_path):
    config = api.config.EplbDiagnosticsConfig(
        mode="observe", run_id="route", output_dir=str(tmp_path), warmup_steps=0, sample_interval=1, max_pending=4
    )
    layer = make_layer(api)
    layer.eplb_diagnostic_probe = api.probe.ExpertLoadProbe(2, "cpu", 2)
    probe = layer.eplb_diagnostic_probe
    metadata = {"rank": 0, "alignment_contract": "post_warmup_executed_collective_v2"}
    recorder = api.runtime.DiagnosticsRecorder(config, [("layer", layer)], metadata, CpuBackend())

    def execute():
        probe.record_routes(torch.tensor([[0], [1]]), torch.tensor([True, False]))
        probe.record(torch.tensor([1, 1]), 1, "MC2CommImpl")

    assert recorder.begin({})
    execute()
    recorder.end()
    assert not recorder.begin({"scheduled_tokens": 0})
    assert recorder.step == 2 and recorder.model_step == 1
    assert not recorder.begin({"dummy_run": True})
    execute()
    recorder.end()
    # An extra graph replay outside the expected boundary invalidates alignment.
    execute()
    assert recorder.begin({})
    execute()
    recorder.end()
    recorder.close()
    layers = [r["layers"][0] for r in api.report.load_records([tmp_path])]
    assert [r["routing"]["physical_assignments"] for r in layers] == [[1, 0], [1, 0]]
    assert [r["alignment_eligible"] for r in layers] == [True, False]


def test_nested_dummy_boundaries_count_once_even_when_not_sampled(api, tmp_path, monkeypatch):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: False), raising=False)
    config = api.config.EplbDiagnosticsConfig(
        mode="observe",
        run_id="nested",
        output_dir=str(tmp_path),
        warmup_steps=3,
        sample_interval=4,
        burst_size=2,
        max_pending=16,
    )
    layer = make_layer(api)
    metadata = {"rank": 0, "alignment_contract": "post_warmup_executed_collective_v2"}
    recorder = api.runtime.DiagnosticsRecorder(config, [("layer", layer)], metadata, CpuBackend())
    runner = SimpleNamespace(
        ascend_config=SimpleNamespace(eplb_diagnostics=config),
        _eplb_diagnostics_ready=True,
        _eplb_diagnostics_recorder=recorder,
    )

    @api.runtime.record_diagnostics
    def execute(runner, scheduler_output, *, dummy_run=False):
        layer.eplb_diagnostic_probe.record(torch.tensor([1, 1]), 1, "MC2CommImpl")

    def dummy(num_tokens, uniform_decode):
        execute(
            runner, SimpleNamespace(total_num_scheduled_tokens=1, num_scheduled_tokens={"dummy": 1}), dummy_run=True
        )

    runner._dummy_run = dummy
    with torch.inference_mode():
        runner._eplb_diagnostics_source_tokens = torch.tensor(8)
    for _ in range(12):
        api.runtime.run_dummy_batch(runner, 1)
    assert recorder.step == recorder.model_step == 12
    assert not runner._eplb_diagnostics_in_execution
    assert recorder.collected == 0
    # Real work after twelve nested dummy calls remains aligned and has budget.
    execute(runner, SimpleNamespace(total_num_scheduled_tokens=1, num_scheduled_tokens={"real": 1}))
    recorder.close()
    records = list(api.report.load_records([tmp_path]))
    assert [r["expected_model_calls"] for r in records] == [13]
    assert all(r["layers"][0]["alignment_eligible"] for r in records)


def test_historical_dummy_rows_cannot_bridge_windows(api):
    rows = [r for step in range(1, 33) for r in routed_rows(step, (8, 0, 2, 0) if step % 2 else (0, 0, 0, 0))]
    for r in rows:
        if r["local_step"] % 2 == 0:
            r["batch"].update(phase="dummy", graph_mode="unknown", dummy_run=True)
    report = api.report.summarize(rows)
    assessment = report["persistent_imbalance"][0]
    assert assessment["status"] == "insufficient_contiguous_samples"
    assert assessment["hotspot_status"] == "insufficient_adjacent_windows"
    assert assessment["verified_idle_calls_skipped"] == 0
    assert report["quality_counts"]["dummy_run_excluded"] == 32
    # A missing rank cannot prove that the entire group was idle.
    missing = [r for r in rows if r["local_step"] % 2 or r["rank"] == 0]
    assert api.report.summarize(missing)["persistent_imbalance"][0]["status"] == "insufficient_contiguous_samples"
    # A layout change during idle must also break the observed window.
    for r in rows:
        if r["local_step"] % 2 == 0:
            r["layers"][0]["layout_fingerprint"] = "changed-during-idle"
    assert api.report.summarize(rows)["persistent_imbalance"][0]["status"] == "insufficient_contiguous_samples"


def test_scheduler_validity_overrides_padded_communication_mask(api):
    probe = api.probe.ExpertLoadProbe(2, "cpu", 2)
    probe.source_token_count = torch.tensor(7)
    probe.source_positions = torch.arange(8)
    ids = torch.tensor([[0], [1], [0], [1]])
    mask = torch.ones(4, dtype=torch.bool)
    probe.record_routes(ids, mask, source_offset=4)
    assert probe.route_totals.tolist() == [2, 1, 1, 0]
    probe.source_token_count.fill_(5)
    probe.record_routes(ids, mask, source_offset=4)
    assert probe.route_totals.tolist() == [3, 1, 2, 0]
    assert mask.all()  # The actual communication mask remains unchanged.


def test_source_validity_buffers_are_shared_before_capture(api):
    model = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    for layer in model:
        layer.eplb_diagnostic_probe = api.probe.ExpertLoadProbe(2, "cpu", 2)
    runner = SimpleNamespace(
        model=model,
        device="cpu",
        ascend_config=SimpleNamespace(
            eplb_diagnostics=api.config.EplbDiagnosticsConfig(mode="observe", run_id="shared")
        ),
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=6),
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 8]),
        ),
    )
    api.runtime.initialize_diagnostics(runner)
    first, second = [layer.eplb_diagnostic_probe for layer in model]
    assert first.source_token_count is second.source_token_count is runner._eplb_diagnostics_source_tokens
    assert first.source_positions is second.source_positions
    assert first.source_positions.numel() == 8


@pytest.mark.parametrize("change", [None, "phase", "bucket"])
def test_excluded_dummy_participants_do_not_become_zero_load_ranks(api, change):
    rows = []
    for step in range(1, 17):
        pair = routed_rows(step)
        source = step % 2
        for rank, row in enumerate(pair):
            active = rank == source
            row["batch"]["scheduled_tokens"] = 10 if active else 0
            row["batch"]["phase"] = "decode" if active else "dummy"
            row["batch"]["graph_mode"] = "FULL" if active else "unknown"
            row["layers"][0]["routing"]["physical_assignments"] = [8, 0, 2, 0] if active else [0, 0, 0, 0]
            if active and source == 1 and change == "phase":
                row["batch"]["phase"] = "prefill"
            if active and source == 1 and change == "bucket":
                row["batch"]["padded_tokens"] = 32
        rows.extend(pair)
    report = api.report.summarize(rows)
    assert report["persistent_imbalance"] == report["workload_persistence"] == []
    assert report["quality_counts"]["dummy_run_excluded"] == 16
    assert report["quality_counts"]["incomplete_group_sample"] == 16
    assert report["enablement_summary"]["candidate_layers"] == []


def test_dummy_work_does_not_contaminate_next_real_sample_or_budget(api, tmp_path):
    config = api.config.EplbDiagnosticsConfig(
        mode="observe",
        run_id="exclude",
        output_dir=str(tmp_path),
        warmup_steps=0,
        sample_interval=1,
        max_samples=1,
    )
    layer = make_layer(api)
    metadata = {"rank": 0, "alignment_contract": "post_warmup_executed_collective_v2"}
    recorder = api.runtime.DiagnosticsRecorder(config, [("layer", layer)], metadata, CpuBackend())
    for _ in range(3):
        assert not recorder.begin({"dummy_run": True, "scheduled_tokens": 0})
        layer.eplb_diagnostic_probe.record(torch.tensor([1000, 9000]), 1, "MC2CommImpl")
        recorder.end()
    assert recorder.collected == 0
    assert recorder.begin({"scheduled_tokens": 5})
    layer.eplb_diagnostic_probe.record(torch.tensor([2, 3]), 1, "MC2CommImpl")
    recorder.end()
    recorder.close()
    records = list(api.report.load_records([tmp_path]))
    assert len(records) == 1
    assert records[0]["expected_model_calls"] == 4
    assert records[0]["layers"][0]["expert_assignments"] == [2, 3]
    assert records[0]["layers"][0]["alignment_eligible"]


def test_late_dummy_annotation_discards_snapshot_and_refunds_budget(api, tmp_path):
    config = api.config.EplbDiagnosticsConfig(
        mode="observe", run_id="late", output_dir=str(tmp_path), warmup_steps=0, sample_interval=1, max_samples=1
    )
    layer = make_layer(api)
    recorder = api.runtime.DiagnosticsRecorder(config, [("layer", layer)], {"rank": 0}, CpuBackend())
    assert recorder.begin({"scheduled_tokens": 1})
    recorder.annotate(dummy_run=True, phase="dummy")
    layer.eplb_diagnostic_probe.record(torch.tensor([100, 0]), 1, "MC2CommImpl")
    recorder.end()
    recorder.close()
    assert recorder.path.read_text() == ""
    assert recorder.collected == 0 and recorder.active is None


def test_dummy_only_trace_produces_no_load_timing_or_recommendation(api):
    rows = routed_rows(1)
    for r in rows:
        r["batch"]["dummy_run"] = True
    report = api.report.summarize(rows)
    assert report["execution_timing"] == report["local_expert_skew"] == report["workload_persistence"] == []
    assert report["quality_counts"] == {"dummy_run_excluded": 2}
    assert report["enablement_summary"]["candidate_layers"] == []
