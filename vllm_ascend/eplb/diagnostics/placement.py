# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only, one-window-ahead evaluation of the real default EPLB policy."""

from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from time import perf_counter_ns

from vllm_ascend.eplb.core.policy.policy_default_eplb import DefaultEplb


@dataclass(frozen=True)
class PlacementPlan:
    """One hypothetical placement; no expert weights or device state are changed."""

    source_window: int
    signature: object
    source_placement: tuple[tuple[int, ...], ...]
    candidate_placement: tuple[tuple[int, ...], ...]
    moved_experts: int
    incoming_experts_per_rank: tuple[int, ...]
    planner_ms: float
    policy_change: bool


def _snapshot(expert_work, mappings):
    if len(mappings) < 2 or len(expert_work) != len(mappings):
        return None, "invalid_rank_count"
    experts = len(mappings[0])
    placement, loads = [], [0] * experts
    for work, mapping in zip(expert_work, mappings):
        if len(mapping) != experts or any(not isinstance(slot, Integral) or slot < -1 for slot in mapping):
            return None, "invalid_expert_mapping"
        owned = {expert: slot for expert, slot in enumerate(mapping) if slot >= 0}
        if set(owned.values()) != set(range(len(owned))):
            return None, "invalid_expert_mapping"
        if set(work) != set(owned) or any(not isinstance(value, Integral) or value < 0 for value in work.values()):
            return None, "invalid_expert_work"
        placement.append(tuple(sorted(owned, key=owned.get)))
        for expert, value in work.items():
            loads[expert] += int(value)
    if len({len(rank) for rank in placement}) != 1:
        return None, "unequal_expert_capacity"
    if sorted(expert for rank in placement for expert in rank) != list(range(experts)) or not experts:
        return None, "invalid_expert_mapping"
    if not sum(loads):
        return None, "no_real_work"
    return (tuple(placement), tuple(loads)), None


def build_plan(expert_work, mappings, window, signature):
    """Plan from a complete valid window using current slots, with zero redundancy.

    ``expert_work`` is ``layer_work()['expert_work']`` and mappings are ordered
    by the same EP ranks. The caller retains only the most recent plan, and
    clears it on invalid or partial windows. Signature includes phase/graph
    mode and rank identity. Planner CPU time is diagnostic cost, not a measured
    runtime EPLB overhead or a performance prediction.
    """
    snapshot, reason = _snapshot(expert_work, mappings)
    if reason:
        return None, reason
    source, loads = snapshot
    workload = [[[loads[expert] for expert in rank] for rank in source]]
    start = perf_counter_ns()
    try:
        change, _, candidate = DefaultEplb().rebalance_experts([source], workload)
    except (ArithmeticError, AssertionError, IndexError, TypeError, ValueError):
        return None, "planner_failed"
    planner_ms = (perf_counter_ns() - start) / 1_000_000
    candidate = tuple(tuple(int(expert) for expert in rank) for rank in candidate[0])
    if (
        len(candidate) != len(source)
        or any(len(new) != len(old) for new, old in zip(candidate, source))
        or sorted(expert for rank in candidate for expert in rank) != list(range(len(loads)))
    ):
        return None, "invalid_candidate_placement"
    incoming = tuple(len(set(new) - set(old)) for old, new in zip(source, candidate))
    # The real worker consumes the returned placement even when change == 0.
    return PlacementPlan(
        source_window=window,
        signature=deepcopy(signature),
        source_placement=source,
        candidate_placement=candidate,
        moved_experts=sum(incoming),
        incoming_experts_per_rank=incoming,
        planner_ms=planner_ms,
        policy_change=bool(change),
    ), None


def evaluate_plan(plan, expert_work, mappings, window, signature):
    """Evaluate a preceding plan on new complete-window work, never training work.

    A signed peak-work reduction compares token assignments, not elapsed time.
    Rank loads remain in the caller's rank order. Stale or incompatible plans
    must be discarded; neither these functions nor the policy retain history.
    """
    if plan is None or window != plan.source_window + 1:
        return None, "no_previous_window_plan"
    if signature != plan.signature:
        return None, "incomparable_window"
    snapshot, reason = _snapshot(expert_work, mappings)
    if reason:
        return None, reason
    source, loads = snapshot
    if source != plan.source_placement:
        return None, "expert_layout_changed"
    current = [sum(loads[expert] for expert in rank) for rank in source]
    candidate = [sum(loads[expert] for expert in rank) for rank in plan.candidate_placement]
    return {
        "current_rank_work": current,
        "candidate_rank_work": candidate,
        "peak_work_reduction": 1 - max(candidate) / max(current),
        "moved_experts": plan.moved_experts,
        "incoming_experts_per_rank": list(plan.incoming_experts_per_rank),
        "planner_ms": plan.planner_ms,
        "policy_change": plan.policy_change,
        "source_window": plan.source_window,
    }, None
