# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collective isolated overhead proxies using scratch buffers, never model weights."""

import logging
from numbers import Integral
from statistics import median
from time import perf_counter

import torch
import torch.distributed as dist


def _moves(source, candidate, size):
    source, candidate = tuple(map(tuple, source)), tuple(map(tuple, candidate))
    if len(source) != size or len(candidate) != size or size < 2:
        raise ValueError("invalid_rank_count")
    slots = len(source[0])
    if slots < 1 or any(len(row) != slots for rows in (source, candidate) for row in rows):
        raise ValueError("unequal_expert_capacity")
    for rows in (source, candidate):
        flat = [expert for row in rows for expert in row]
        if any(isinstance(e, bool) or not isinstance(e, Integral) for e in flat):
            raise ValueError("invalid_expert_id")
        if sorted(flat) != list(range(size * slots)):
            raise ValueError("invalid_expert_placement")
    old = {expert: rank for rank, row in enumerate(source) for expert in row}
    new = {expert: rank for rank, row in enumerate(candidate) for expert in row}
    return source, candidate, [(e, old[e], new[e]) for e in sorted(old) if old[e] != new[e]]


def _agree(group, metadata, reason):
    rows = [None] * len(group.ranks)
    # Gloo writes receive tensors from a background thread.
    with torch.inference_mode(False):
        dist.all_gather_object(rows, (metadata, reason), group=group.cpu_group)
    if any(error for _, error in rows):
        return next(error for _, error in rows if error)
    return "inconsistent_calibration_parameters" if any(row[0] != rows[0][0] for row in rows) else None


def _warn_failure(group, phase, error):
    logging.getLogger("vllm.eplb.diagnostics").warning(
        "EPLB scratch calibration failed: rank=%s phase=%s error=%s: %s",
        group.ranks[group.rank_in_group],
        phase,
        type(error).__name__,
        error,
    )


def _measure(operation, group, repeats, phase):
    samples = []
    for index in range(repeats + 1):
        reason = None
        try:
            torch.npu.synchronize()
        except Exception as exc:
            _warn_failure(group, f"{phase}.synchronize", exc)
            reason = f"scratch_{phase}_synchronize_failed"
        reason = _agree(group, (phase, index, "before"), reason)
        if reason:
            return None, reason
        result = None
        start = perf_counter()
        try:
            result = operation()
            torch.npu.synchronize()
            elapsed = (perf_counter() - start) * 1000
        except Exception as exc:
            _warn_failure(group, phase, exc)
            reason = f"scratch_{phase}_failed"
        # Coordinate recoverable local failures before any next device round.
        # A fatal device/HCCL error may strand peers inside operation itself;
        # this best-effort CPU agreement cannot recover a broken communicator.
        reason = _agree(group, (phase, index, "after"), reason)
        if reason:
            return None, reason
        if index:
            samples.append(elapsed)
        del result
    return {
        "min_ms": min(samples),
        "median_ms": median(samples),
        "max_ms": max(samples),
        "samples": repeats,
    }, None


def measure_overheads(
    group, source_placement, candidate_placement, expert_bytes, repeats=3, max_scratch_bytes=134217728
):
    """Measure local serialized collection, packed transfer and apply proxies.

    All EP ranks must call collectively while normal inference is stopped. CPU
    agreement precedes allocation and again precedes device communication. No
    participant enters HCCL if another rejects the plan or scratch budget.
    Recoverable per-round failures are agreed before continuing; failed rounds
    return no timing data. Fatal device/communication failures require normal
    distributed failure handling and are not made recoverable by this helper.

    A packed byte payload per moved expert preserves bytes and peer mapping,
    but not production parameter-message counts. One-layer load collection is
    not the packed all-layer production gather. Apply copies incoming buffers
    plus two expert-map proxies. These are isolated serial *proxies*, not live
    overlap costs, upper bounds, or automatic evidence of a positive speedup.
    """
    result = {"scope": "isolated_serial_packed_scratch", "reason": None}
    if getattr(group, "cpu_group", None) is None or getattr(group, "device_group", None) is None:
        return result | {"reason": "missing_collective_group"}
    metadata, reason = None, None
    try:
        if (
            any(
                isinstance(x, bool) or not isinstance(x, Integral) or x < 1
                for x in (expert_bytes, repeats, max_scratch_bytes)
            )
            or repeats > 32
        ):
            raise ValueError("invalid_calibration_limits")
        ranks, local = tuple(group.ranks), group.rank_in_group
        if len(set(ranks)) != len(ranks) or local not in range(len(ranks)):
            raise ValueError("invalid_group_ranks")
        source, candidate, moves = _moves(source_placement, candidate_placement, len(ranks))
        outgoing = [(e, dst) for e, src, dst in moves if src == local]
        incoming = [(e, src) for e, src, dst in moves if dst == local]
        experts, slots = sum(map(len, source)), len(source[0])
        # Two receive-sized banks, one shared read-only send buffer; reserve
        # input, gathered/intermediate load tensors and two device map tensors.
        required = (2 * len(incoming) + bool(outgoing)) * expert_bytes + (slots + 4 * experts) * 8
        metadata = (source, candidate, expert_bytes, repeats, max_scratch_bytes, ranks)
        if required > max_scratch_bytes:
            raise ValueError("scratch_memory_cap_exceeded")
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise ValueError("npu_unavailable")
        if torch.npu.is_current_stream_capturing():
            raise ValueError("calibration_inside_graph_capture")
    except (TypeError, ValueError) as exc:
        reason = str(exc)
    reason = _agree(group, metadata, reason)
    if reason:
        return result | {"reason": reason}
    try:
        device = group.device
        load = torch.zeros((1, slots), dtype=torch.int64, device=device)
        gathered = (
            None if hasattr(group, "all_gather") else torch.empty((len(ranks), slots), dtype=load.dtype, device=device)
        )
        send = torch.zeros(expert_bytes, dtype=torch.uint8, device=device) if outgoing else None
        receive = {e: torch.empty(expert_bytes, dtype=torch.uint8, device=device) for e, _ in incoming}
        targets = {e: torch.empty_like(buffer) for e, buffer in receive.items()}
        maps = [torch.arange(experts, dtype=torch.int64) for _ in range(2)]
        map_targets = [torch.empty(experts, dtype=torch.int64, device=device) for _ in maps]
        # Identical expert order on every rank also fixes ordering for same peers.
        operations = []
        for expert, src, dst in moves:
            if src == local:
                operations.append(dist.P2POp(dist.isend, send, ranks[dst], group=group.device_group))
            if dst == local:
                operations.append(dist.P2POp(dist.irecv, receive[expert], ranks[src], group=group.device_group))
        torch.npu.synchronize()
    except (RuntimeError, MemoryError, AttributeError, TypeError, ValueError) as exc:
        _warn_failure(group, "allocation", exc)
        reason = "scratch_allocation_failed"
    reason = _agree(group, metadata, reason)
    if reason:
        return result | {"reason": reason}

    def collect():
        if gathered is None:
            return group.all_gather(load, dim=0).cpu()
        dist.all_gather_into_tensor(gathered, load, group=group.device_group)
        return gathered.cpu()

    def transfer():
        for request in dist.batch_isend_irecv(operations) if operations else []:
            request.wait()

    def apply():
        for expert, buffer in receive.items():
            targets[expert].copy_(buffer)
        for target, mapping in zip(map_targets, maps):
            target.copy_(mapping)

    timings = {}
    # Collection also initializes every HCCL group participant before P2P.
    for phase, operation in (("collection", collect), ("transfer", transfer), ("apply", apply)):
        if phase == "transfer" and not moves:
            timings["transfer_ms"] = {"min_ms": 0, "median_ms": 0, "max_ms": 0, "samples": 0}
            continue
        timing, reason = _measure(operation, group, repeats, phase)
        if reason:
            return result | {"reason": reason}
        timings[f"{phase}_ms"] = timing
    return result | {
        **timings,
        "scratch_bytes": required,
        "moved_experts": len(moves),
        "incoming_experts": len(incoming),
        "outgoing_experts": len(outgoing),
    }
