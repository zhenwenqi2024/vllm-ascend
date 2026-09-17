# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable diagnostics checks: pytest --noconftest tests/ut/eplb/test_diagnostics.py."""

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
            name: importlib.import_module(f"vllm_ascend.eplb_diagnostics.{name}")
            for name in ("config", "probe", "runtime")
        }
    )
    if stubbed:
        for name in set(sys.modules) - original:
            if name == "vllm_ascend" or name.startswith("vllm_ascend."):
                del sys.modules[name]


def rows(two_layers=False):
    # Two layers on noncontiguous ranks. EP owners deliberately differ from
    # expert-ID modulo rank count. Each rank supplies four top-1 assignments.
    return [
        {
            "rank": rank,
            "tp_ranks": [rank],
            "tokens": 4,
            "step": 8,
            "phases": [("decode", "FULL")],
            "layers": [
                {
                    "name": name,
                    "top_k": 1,
                    "local_experts": 2,
                    "mapping": [0, 1, -1, -1] if i == 0 else [-1, -1, 0, 1],
                    "counts": [3, 1, 0, 0, 8, 0],
                }
                for name in (("layer.0", "layer.1") if two_layers else ("layer.0",))
            ],
        }
        for i, rank in enumerate((2, 5))
    ]


def test_layer_work_and_real_owners(api):
    result, reason = api.runtime.layer_work(rows(), 8)
    assert reason is None
    assert result["total"] == 8
    assert result["rank_work"] == [8, 0]
    assert result["skew"] == 2
    assert result["hot"] == {("layer.0", 0): 6}


def test_window_reports_prefill_and_decode_coverage(api, caplog):
    caplog.set_level(logging.INFO)
    data = rows()
    data[0]["phases"].append(("prefill", "NONE"))
    api.runtime.WorkloadLogger([2, 5], 0).log(data, 1, 8)
    assert "phases=['decode', 'prefill']" in caplog.text


def test_routes_without_comm_mask_and_empty_replica(api):
    probe = api.probe.ExpertLoadProbe(2, "cpu")
    probe.source_positions = torch.arange(4)
    probe.source_token_count = torch.tensor(3)
    ids = torch.tensor([[0], [1], [0], [1]])
    probe.record_routes(ids, source_offset=2)
    probe.record_routes(ids[:0])
    probe.source_token_count.zero_()
    probe.record_routes(ids)
    assert probe.totals.tolist() == [1, 0, 3, 0]


@pytest.mark.parametrize(
    "problem,reason",
    [
        ("step", "unaligned_windows"),
        ("layer", "incomplete_layer_set"),
        ("rank", "duplicate_rank"),
        ("tp", "source_tp_mismatch"),
        ("counts", "unsupported_routing_or_call_count"),
        ("invalid_id", "unsupported_routing_or_call_count"),
        ("tokens", "source_conservation_failure"),
        ("mapping", "invalid_expert_mapping"),
        ("owner", "duplicate_expert_owner"),
        ("top_k", "expert_space_mismatch"),
    ],
)
def test_incomplete_data_never_becomes_balance(api, problem, reason):
    data = rows()
    row, layer = data[1], data[1]["layers"][0]
    if problem == "step":
        row["step"] += 1
    elif problem == "layer":
        row["layers"].pop()
    elif problem == "rank":
        row["rank"] = 2
    elif problem == "tp":
        row["tp_ranks"] = [2, 5]
    elif problem == "counts":
        layer["counts"][-2] -= 1
    elif problem == "invalid_id":
        layer["counts"][-1] = 1
    elif problem == "tokens":
        row["tokens"] += 1
    elif problem == "mapping":
        layer["mapping"] = [-1, -1, 0, 0]
    elif problem == "owner":
        layer["mapping"] = [0, 1, -1, -1]
    else:
        layer["top_k"] = 2
    assert api.runtime.layer_work(data, 8) == (None, reason)


def test_tp_shards_and_idle_dp_sources(api):
    data = rows()
    for row in data:
        row["tp_ranks"] = [2, 5]
        row["tokens"] = 8
    assert api.runtime.layer_work(data, 8)[0]["total"] == 8
    data = rows()
    data[1]["tokens"] = 0
    for layer in data[1]["layers"]:
        layer["counts"][:4] = [0] * 4
    assert api.runtime.layer_work(data, 8)[0]["rank_work"] == [4, 0]
    for row in data:
        row["tokens"] = 0
        for layer in row["layers"]:
            layer["counts"][:4] = [0] * 4
    assert api.runtime.layer_work(data, 8) == (None, "no_real_work")


def diagnostics(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("[EPLB diagnostic]")]


def test_rank_expert_totals_and_layer_persistence(api, caplog):
    caplog.set_level(logging.INFO)
    summary = api.runtime.WorkloadLogger([2, 5], 1)
    summary.log(rows(), 1, 8)
    summary.log(rows(), 2, 8)
    assert len(diagnostics(caplog)) == 2
    assert "valid_assignments=8 rank_work=[8, 0]" in caplog.text
    assert "persistent_hot_experts=['0(2/2)']" in caplog.text
    assert "cumulative_work=16 expert_work=['0:12', '1:4']" in caplog.text
    assert "expert_work=['2:0', '3:0']" in caplog.text
    assert "conclusion=consider_eplb_for_observed_layers" in caplog.text
    data = rows()
    data[1]["step"] = 9
    summary.log(data, 3, 8)
    layer = summary.layers["layer.0"]
    assert layer.invalid_windows == 1 and not layer.streaks
    assert layer.valid_windows == 2
    assert layer.hint == "insufficient_evidence"
    summary.log(rows(), 4, 8)
    assert layer.valid_windows == 3
    data = rows()
    data[0]["phases"] = [("prefill", "NONE")]
    summary.log(data, 5, 8)
    assert layer.streaks[("layer.0", 0)] == 1


def test_opposite_layer_skew_does_not_cancel(api, caplog):
    caplog.set_level(logging.INFO)
    data = rows(two_layers=True)
    for row in data:
        row["layers"][1]["counts"][:4] = [0, 0, 3, 1]
    summary = api.runtime.WorkloadLogger([2, 5], 0)
    summary.log(data, 1, 8)
    caplog.clear()
    summary.log(data, 2, 8)
    first, second = diagnostics(caplog)
    assert "layer=layer.0" in first and "rank_work=[8, 0]" in first
    assert "layer=layer.1" in second and "rank_work=[0, 8]" in second
    assert all("hint=consider_eplb" in line for line in (first, second))
    assert "candidate_layers=['layer.0', 'layer.1']" in caplog.text
    data[1]["layers"][1]["counts"][-1] = 1
    caplog.clear()
    summary.log(data, 3, 8)
    assert "insufficient_evidence=" in diagnostics(caplog)[1]
    assert summary.layers["layer.0"].valid_windows == 3
    assert summary.layers["layer.1"].valid_windows == 2
    assert summary.layers["layer.1"].invalid_windows == 1


def test_layer_hotspot_histories_are_independent(api, caplog):
    caplog.set_level(logging.INFO)
    summary = api.runtime.WorkloadLogger([2, 5], 0)
    data = rows(two_layers=True)
    summary.log(data, 1, 8)
    for row in data:
        row["layers"][1]["counts"][:4] = [1, 3, 0, 0]
    caplog.clear()
    summary.log(data, 2, 8)
    assert "hint=consider_eplb" in diagnostics(caplog)[0]
    assert "hint=insufficient_hotspot_evidence" in diagnostics(caplog)[1]


def test_run_history_is_cumulative_not_last_four_windows(api, caplog):
    caplog.set_level(logging.INFO)
    summary = api.runtime.WorkloadLogger([2, 5], 0)
    summary.log(rows(), 1, 8)
    data = rows()
    for row in data:
        data_layer = row["layers"][0]
        data_layer["counts"][:4] = [1, 3, 0, 0]
    summary.log(data, 2, 8)
    assert "hint=insufficient_hotspot_evidence" in caplog.text
    for window in range(3, 9):
        summary.log(data, window, 8)
    layer = summary.layers["layer.0"]
    assert layer.valid_windows == 8
    assert layer.hot_windows[("layer.0", 1)] == 7
    assert sum(sum(counts.values()) for counts in layer.expert_totals) == 64


def test_partial_window_counts_work_but_not_persistence(api, caplog):
    caplog.set_level(logging.INFO)
    summary = api.runtime.WorkloadLogger([2, 5], 0)
    summary.log(rows(), 1, 8)
    data = rows()
    for row in data:
        row["layers"][0]["counts"][-2] = 1
    summary.log(data, 2, 1, complete=False)
    layer = summary.layers["layer.0"]
    assert layer.valid_windows == 1 and layer.observed_calls == 9
    assert not layer.persistent
    assert sum(sum(counts.values()) for counts in layer.expert_totals) == 16
    summary.summarize(2, final=True)
    assert "final=True" in caplog.text
    assert "conclusion=insufficient_evidence" in caplog.text


def test_probe_padding_source_offset_dummy_and_stable_storage(api):
    probe = api.probe.ExpertLoadProbe(2, "cpu")
    probe.source_token_count = torch.tensor(3)
    probe.source_positions = torch.arange(4)
    ptr = probe.totals.data_ptr()
    ids = torch.tensor([[0], [1], [99], [1]])
    probe.record_routes(ids, torch.tensor([True, False, True, True]), source_offset=2)
    assert probe.totals.tolist() == [1, 0, 1, 0]
    probe.source_token_count.fill_(0)
    probe.record_routes(ids, torch.ones(4, dtype=torch.bool))
    assert probe.totals.tolist() == [1, 0, 2, 0]
    probe.source_token_count.fill_(4)
    probe.record_routes(ids, torch.ones(4, dtype=torch.bool))
    assert probe.totals.tolist() == [2, 2, 3, 1]
    assert probe.totals.data_ptr() == ptr


def test_window_collection_limits_and_dummy(api, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    config = api.config.EplbDiagnosticsConfig(mode="observe", warmup_steps=1, window_size=2, max_windows=1)
    probe = api.probe.ExpertLoadProbe(2, "cpu")
    source = torch.tensor(0)
    probe.source_token_count, probe.source_positions = source, torch.arange(1)
    layer = SimpleNamespace(
        eplb_diagnostic_probe=probe,
        ascend_expert_map=torch.tensor([0, 1]),
        moe_config=SimpleNamespace(experts_per_token=1),
        local_num_experts=2,
    )
    group = SimpleNamespace(ranks=[0], rank_in_group=0, cpu_group=None)
    calls = []

    def gather(output, row, group):
        calls.append(deepcopy(row))
        output[0] = row

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    recorder = api.runtime.DiagnosticsRecorder(config, [("layer", layer)], group, [0], 0, source)
    for tokens, dummy in ((1, False), (1, False), (99, True), (1, False)):
        recorder.begin(tokens, dummy)
        probe.record_routes(torch.tensor([[0]]), torch.tensor([True]))
        recorder.end()
    assert len(calls) == 1
    assert calls[0]["tokens"] == 1
    assert calls[0]["layers"][0]["counts"] == [1, 0, 2, 0]
    assert "valid_assignments=1" in caplog.text
    assert source.item() == 0


@pytest.mark.parametrize("steps", [1, 3, 6])
def test_finish_collects_tail_once_and_stops(api, monkeypatch, caplog, steps):
    caplog.set_level(logging.INFO)
    config = api.config.EplbDiagnosticsConfig(mode="observe", warmup_steps=0, window_size=2, max_windows=0)
    probe = api.probe.ExpertLoadProbe(2, "cpu")
    source = torch.tensor(0)
    probe.source_token_count, probe.source_positions = source, torch.arange(1)
    layer = SimpleNamespace(
        eplb_diagnostic_probe=probe,
        ascend_expert_map=torch.tensor([0, 1]),
        moe_config=SimpleNamespace(experts_per_token=1),
        local_num_experts=2,
    )
    group = SimpleNamespace(ranks=[0], rank_in_group=0, cpu_group=None)
    monkeypatch.setattr(torch.distributed, "all_gather_object", lambda out, row, group: out.__setitem__(0, row))
    recorder = api.runtime.DiagnosticsRecorder(config, [("layer", layer)], group, [0], 0, source)
    for _ in range(steps):
        recorder.begin(1)
        probe.record_routes(torch.tensor([[0]]), torch.tensor([True]))
        recorder.end()
    recorder.finish()
    recorder.finish()
    summary = recorder.summary.layers["layer"]
    assert summary.observed_calls == steps
    assert summary.valid_windows == steps // 2
    assert summary.expert_totals[0][0] == steps
    assert caplog.text.count("final=True") == 1
    recorder.begin(1)
    assert not recorder.collecting and source.item() == 0


def test_wrapper_nested_dummy_and_failure(api, monkeypatch):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: False), raising=False)
    calls = []
    recorder = SimpleNamespace(begin=lambda *x: calls.append(x), end=lambda: calls.append("end"))
    runner = SimpleNamespace(_eplb_diagnostics_recorder=recorder)
    scheduler = SimpleNamespace(total_num_scheduled_tokens=0)

    @api.runtime.record_diagnostics
    def execute(runner, scheduler, dummy_run=False, nested=False):
        if not nested:
            return execute(runner, scheduler, dummy_run=True, nested=True)
        return 42

    assert execute(runner, scheduler, dummy_run=True) == 42
    assert calls == [(0, True), "end"]
    calls.clear()
    assert execute(runner, scheduler) == 42
    assert calls == [(0, True), "end"]  # Only the actual nested dummy executes.

    @api.runtime.record_diagnostics
    def fail(*args, **kwargs):
        raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        fail(runner, scheduler, dummy_run=True)
    assert runner._eplb_diagnostics_in_execution is False


@pytest.mark.parametrize("field", ["window_size", "warmup_steps", "max_windows"])
def test_config_bounds(api, field):
    with pytest.raises(ValueError):
        api.config.EplbDiagnosticsConfig(**{field: -1})
