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


def rows():
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
                for name in ("layer.0", "layer.1")
            ],
        }
        for i, rank in enumerate((2, 5))
    ]


def test_aggregate_all_layers_and_real_owners(api):
    result, reason = api.runtime.aggregate_work(rows(), 8)
    assert reason is None
    assert result["total"] == 16
    assert result["rank_work"] == [16, 0]
    assert result["skew"] == 2
    assert result["hot"] == {("layer.0", 0): 6, ("layer.1", 0): 6}


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
    assert api.runtime.aggregate_work(data, 8) == (None, reason)


def test_tp_shards_and_idle_dp_sources(api):
    data = rows()
    for row in data:
        row["tp_ranks"] = [2, 5]
        row["tokens"] = 8
    assert api.runtime.aggregate_work(data, 8)[0]["total"] == 16
    data = rows()
    data[1]["tokens"] = 0
    for layer in data[1]["layers"]:
        layer["counts"][:4] = [0] * 4
    assert api.runtime.aggregate_work(data, 8)[0]["rank_work"] == [8, 0]
    for row in data:
        row["tokens"] = 0
        for layer in row["layers"]:
            layer["counts"][:4] = [0] * 4
    assert api.runtime.aggregate_work(data, 8) == (None, "no_real_work")


def test_print_one_overall_line_and_persistence(api, caplog):
    caplog.set_level(logging.INFO)
    summary = api.runtime.WorkloadLogger([2, 5], 1)
    summary.log(rows(), 1, 8)
    assert "hint=collecting" in caplog.text
    summary.log(rows(), 2, 8)
    assert len(caplog.records) == 2
    assert "valid_assignments=16 rank_work=[16, 0]" in caplog.text
    assert "persistent_hot_count=2" in caplog.text
    assert "hint=consider_eplb" in caplog.text
    # A missing window cannot extend a hotspot's persistence.
    data = rows()
    data[1]["step"] = 9
    summary.log(data, 3, 8)
    caplog.clear()
    summary.log(rows(), 4, 8)
    assert "hint=collecting" in caplog.text
    data = rows()
    data[0]["phases"] = [("prefill", "NONE")]
    summary.log(data, 5, 8)
    assert len(summary.history) == 1


def test_aggregate_balance_can_hide_layer_skew(api, caplog):
    caplog.set_level(logging.INFO)
    data = rows()
    for row in data:
        row["layers"][1]["counts"][:4] = [0, 0, 3, 1]
    summary = api.runtime.WorkloadLogger([2, 5], 0)
    summary.log(data, 1, 8)
    summary.log(data, 2, 8)
    assert "rank_work=[8, 8]" in caplog.text
    assert "hint=no_persistent_rank_skew" in caplog.text


def test_hot_expert_rotation_is_not_persistence(api, caplog):
    caplog.set_level(logging.INFO)
    summary = api.runtime.WorkloadLogger([2, 5], 0)
    summary.log(rows(), 1, 8)
    data = rows()
    for row in data:
        for layer in row["layers"]:
            layer["counts"][:4] = [1, 3, 0, 0]
    summary.log(data, 2, 8)
    assert "hint=insufficient_hotspot_evidence" in caplog.text
    for window in range(3, 9):
        summary.log(data, window, 8)
    assert len(summary.history) == 4


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
