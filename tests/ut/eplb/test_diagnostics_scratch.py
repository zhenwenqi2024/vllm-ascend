# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable scratch calibration orchestration, ownership and refusal tests."""

import importlib.util
import logging
from itertools import count
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def api():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/eplb/diagnostics/scratch.py"
    spec = importlib.util.spec_from_file_location("eplb_scratch_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake(api, monkeypatch):
    state = SimpleNamespace(
        agreements=0,
        agreement_values=[],
        round_peer_reason=None,
        peer_reason=None,
        mismatch=False,
        allocation_peer_reason=None,
        batches=[],
        barriers=0,
        gathers=0,
        copies=[],
    )
    group = SimpleNamespace(
        ranks=[2, 5, 9], rank_in_group=0, cpu_group="cpu_group", device_group="device_group", device="cpu"
    )

    def agree(rows, value, *, group):
        assert group == "cpu_group"
        state.agreements += 1
        state.agreement_values.append(value)
        rows[:] = [value] * 3
        if state.peer_reason and state.agreements == 1:
            rows[1] = (value[0], state.peer_reason)
        if state.allocation_peer_reason and state.agreements == 2:
            rows[1] = (value[0], state.allocation_peer_reason)
        if state.round_peer_reason and value[0] == ("apply", 0, "after"):
            rows[1] = (value[0], state.round_peer_reason)
        if state.mismatch:
            rows[1] = (None, None)

    def barrier(*, group):
        assert group == "cpu_group"
        state.barriers += 1

    def gather(output, value, *, group):
        assert state.agreements >= 2
        assert group == "device_group"
        assert value.shape == (1, 2) and value.dtype == torch.int64
        output.copy_(value.expand_as(output))
        state.gathers += 1

    def batch(operations):
        assert state.agreements >= 2 and state.gathers > 0
        state.batches.append(operations)
        for operation in operations:
            if operation.op is api.dist.irecv:
                operation.tensor.fill_(7)
        return [SimpleNamespace(wait=lambda: None)] * len(operations)

    empty_like = torch.empty_like

    def track_target(tensor):
        target = empty_like(tensor)
        state.copies.append((tensor, target))
        return target

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(is_available=lambda: True, is_current_stream_capturing=lambda: False, synchronize=lambda: None),
        raising=False,
    )
    monkeypatch.setattr(api.dist, "all_gather_object", agree)
    monkeypatch.setattr(api.dist, "all_gather_into_tensor", gather)
    monkeypatch.setattr(api.dist, "barrier", barrier)
    monkeypatch.setattr(
        api.dist, "P2POp", lambda op, tensor, peer, group: SimpleNamespace(op=op, tensor=tensor, peer=peer, group=group)
    )
    monkeypatch.setattr(api.dist, "batch_isend_irecv", batch)
    monkeypatch.setattr(torch, "empty_like", track_target)
    ticks = count()
    monkeypatch.setattr(api, "perf_counter", lambda: next(ticks) / 1000)
    return group, state


SOURCE = ((0, 1), (2, 3), (4, 5))
CANDIDATE = ((2, 4), (0, 3), (1, 5))


def test_collect_transfer_and_apply_use_only_exact_scratch_payloads(api, fake):
    group, state = fake
    result = api.measure_overheads(group, SOURCE, CANDIDATE, 4, repeats=2)
    assert result["reason"] is None
    assert result["scope"] == "isolated_serial_packed_scratch"
    assert result["incoming_experts"] == result["outgoing_experts"] == 2
    assert result["moved_experts"] == 4
    assert result["scratch_bytes"] == 228
    assert state.agreements == 20
    assert state.barriers == 0 and len(state.batches) == state.gathers == 3
    operations = state.batches[0]
    sends = [op for op in operations if op.op is api.dist.isend]
    receives = [op for op in operations if op.op is api.dist.irecv]
    assert [op.peer for op in sends] == [5, 9]
    assert [op.peer for op in receives] == [5, 9]
    assert sends[0].tensor is sends[1].tensor
    assert receives[0].tensor is not receives[1].tensor
    assert all(op.tensor.numel() == 4 and op.tensor.dtype == torch.uint8 for op in operations)
    assert all(
        torch.equal(source, target) and source.data_ptr() != target.data_ptr() for source, target in state.copies
    )
    for field in ("collection_ms", "transfer_ms", "apply_ms"):
        assert result[field]["median_ms"] == pytest.approx(1)
        assert result[field]["samples"] == 2


def test_local_cap_failure_is_agreed_before_allocating(api, fake, monkeypatch):
    group, state = fake
    monkeypatch.setattr(torch, "zeros", lambda *args, **kwargs: pytest.fail("must not allocate"))
    result = api.measure_overheads(group, SOURCE, CANDIDATE, 4, max_scratch_bytes=227)
    assert result["reason"] == "scratch_memory_cap_exceeded"
    assert state.agreements == 1 and state.gathers == 0


@pytest.mark.parametrize("reason", ["scratch_memory_cap_exceeded", "npu_unavailable"])
def test_peer_rejection_stops_all_before_device_communication(api, fake, reason):
    group, state = fake
    state.peer_reason = reason
    result = api.measure_overheads(group, SOURCE, CANDIDATE, 4)
    assert result["reason"] == reason
    assert state.agreements == 1 and state.gathers == 0


def test_parameter_mismatch_stops_before_allocation(api, fake):
    group, state = fake
    state.mismatch = True
    result = api.measure_overheads(group, SOURCE, CANDIDATE, 4)
    assert result["reason"] == "inconsistent_calibration_parameters"
    assert state.agreements == 1 and state.gathers == 0


def test_peer_allocation_failure_is_agreed_before_hccl(api, fake):
    group, state = fake
    state.allocation_peer_reason = "scratch_allocation_failed"
    result = api.measure_overheads(group, SOURCE, CANDIDATE, 4)
    assert result["reason"] == "scratch_allocation_failed"
    assert state.agreements == 2 and state.gathers == 0


def test_local_allocation_failure_is_agreed_before_hccl(api, fake, monkeypatch):
    group, state = fake

    def fail(*args, **kwargs):
        raise MemoryError("out of scratch memory")

    monkeypatch.setattr(torch, "zeros", fail)
    result = api.measure_overheads(group, SOURCE, CANDIDATE, 4)
    assert result["reason"] == "scratch_allocation_failed"
    assert state.agreements == 2 and state.gathers == 0


def test_no_migration_never_launches_p2p(api, fake):
    group, state = fake
    result = api.measure_overheads(group, SOURCE, SOURCE, 4, repeats=1)
    assert result["reason"] is None and result["moved_experts"] == 0
    assert not state.batches
    assert result["transfer_ms"]["max_ms"] == 0


@pytest.mark.parametrize("candidate", [((0, 1), (2, 3)), ((0, 1), (2, 3), (4, 4)), ((0,), (1, 2), (3, 4, 5))])
def test_invalid_placement_is_collectively_rejected(api, fake, candidate):
    group, state = fake
    result = api.measure_overheads(group, SOURCE, candidate, 4)
    assert result["reason"] is not None
    assert state.agreements == 1 and state.gathers == 0


def test_group_all_gather_path_matches_one_layer_payload(api, fake):
    group, state = fake

    def gather(value, dim):
        assert dim == 0 and value.shape == (1, 2)
        state.gathers += 1
        return value.repeat(3, 1)

    group.all_gather = gather
    result = api.measure_overheads(group, SOURCE, CANDIDATE, 4, repeats=1)
    assert result["reason"] is None and state.gathers == 2


def test_local_apply_failure_is_agreed_and_discards_all_timing(api, fake, monkeypatch, caplog):
    group, state = fake

    def fail_copy(*args, **kwargs):
        raise RuntimeError("recoverable scratch copy error")

    monkeypatch.setattr(logging.getLogger("vllm.eplb.diagnostics"), "propagate", True)
    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(torch, "empty_like", lambda tensor: SimpleNamespace(copy_=fail_copy))
    result = api.measure_overheads(group, SOURCE, CANDIDATE, 4, repeats=2)
    assert result == {"scope": "isolated_serial_packed_scratch", "reason": "scratch_apply_failed"}
    assert state.agreement_values[-1] == (("apply", 0, "after"), "scratch_apply_failed")
    assert "rank=2 phase=apply error=RuntimeError: recoverable scratch copy error" in caplog.text
    assert state.agreements == 16 and state.barriers == 0
    assert len(state.batches) == state.gathers == 3


def test_peer_apply_failure_stops_before_next_round_and_discards_timing(api, fake):
    group, state = fake
    state.round_peer_reason = "scratch_apply_failed"
    result = api.measure_overheads(group, SOURCE, CANDIDATE, 4, repeats=2)
    assert result == {"scope": "isolated_serial_packed_scratch", "reason": "scratch_apply_failed"}
    assert state.agreement_values[-1][0] == ("apply", 0, "after")
    assert state.agreements == 16 and state.barriers == 0


def test_local_pre_round_sync_failure_is_agreed_before_operation(api, fake, monkeypatch):
    group, state = fake

    def fail_sync():
        raise RuntimeError("synchronize failed before launch")

    monkeypatch.setattr(torch.npu, "synchronize", fail_sync)
    result, reason = api._measure(lambda: pytest.fail("must not launch"), group, 2, "apply")
    assert result is None and reason == "scratch_apply_synchronize_failed"
    assert state.agreement_values == [(("apply", 0, "before"), reason)]
    assert state.barriers == 0
