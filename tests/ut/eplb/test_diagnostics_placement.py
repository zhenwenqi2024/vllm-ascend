# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable checks for shadow planning with the actual default EPLB policy."""

import importlib
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def api():
    stubbed = "vllm_ascend" not in sys.modules and importlib.util.find_spec("vllm") is None
    original = set(sys.modules)
    if stubbed:
        package = ModuleType("vllm_ascend")
        package.__path__ = [str(Path(__file__).resolve().parents[3] / "vllm_ascend")]
        sys.modules["vllm_ascend"] = package
    yield importlib.import_module("vllm_ascend.eplb.diagnostics.placement")
    if stubbed:
        for name in set(sys.modules) - original:
            if name == "vllm_ascend" or name.startswith("vllm_ascend."):
                del sys.modules[name]


def sample(loads=(100, 100, 0, 0)):
    # Deliberately non-sorted physical slots; preserve the real slot ordering.
    return [{0: loads[0], 1: loads[1]}, {2: loads[2], 3: loads[3]}], [[1, 0, -1, -1], [-1, -1, 0, 1]]


def test_next_window_uses_real_policy_and_preserves_input(api, monkeypatch):
    work, mappings = sample()
    original = deepcopy((work, mappings))
    called = []
    real_policy = api.DefaultEplb.rebalance_experts

    def track(self, placement, workload):
        called.append((deepcopy(placement), deepcopy(workload)))
        return real_policy(self, placement, workload)

    monkeypatch.setattr(api.DefaultEplb, "rebalance_experts", track)
    plan, reason = api.build_plan(work, mappings, 7, [("decode", "FULL")])
    assert reason is None
    assert len(called) == 1
    assert called[0][0] == [((1, 0), (2, 3))]
    assert (work, mappings) == original
    assert sorted(expert for rank in plan.candidate_placement for expert in rank) == [0, 1, 2, 3]
    assert all(len(rank) == 2 for rank in plan.candidate_placement)
    assert plan.moved_experts == sum(plan.incoming_experts_per_rank) == 2
    assert plan.planner_ms >= 0
    result, reason = api.evaluate_plan(plan, work, mappings, 8, [("decode", "FULL")])
    assert reason is None
    assert result["current_rank_work"] == [200, 0]
    assert result["candidate_rank_work"] == [100, 100]
    assert result["peak_work_reduction"] == 0.5
    assert result["source_window"] == 7
    assert result["policy_change"] is True


def test_single_unsplittable_hot_expert_does_not_gain(api):
    work, mappings = sample((200, 0, 0, 0))
    plan, reason = api.build_plan(work, mappings, 1, "decode")
    assert reason is None
    result, reason = api.evaluate_plan(plan, work, mappings, 2, "decode")
    assert reason is None
    assert max(result["candidate_rank_work"]) == 200
    assert result["peak_work_reduction"] == 0
    assert result["policy_change"] is False


def test_new_adversarial_hotspots_can_make_previous_plan_worse(api):
    work, mappings = sample()
    plan, reason = api.build_plan(work, mappings, 1, "decode")
    assert reason is None
    # New hot experts share one candidate rank but were on separate source ranks.
    hot_experts = set(plan.candidate_placement[0])
    loads = [100 if expert in hot_experts else 0 for expert in range(4)]
    next_work, _ = sample(loads)
    result, reason = api.evaluate_plan(plan, next_work, mappings, 2, "decode")
    assert reason is None
    assert result["current_rank_work"] == [100, 100]
    assert sorted(result["candidate_rank_work"]) == [0, 200]
    assert result["peak_work_reduction"] == -1


@pytest.mark.parametrize("window", [1, 3, 0])
def test_never_evaluate_training_window_or_stale_plan(api, window):
    work, mappings = sample()
    plan, _ = api.build_plan(work, mappings, 1, "decode")
    result, reason = api.evaluate_plan(plan, work, mappings, window, "decode")
    assert result is None
    assert reason == "no_previous_window_plan"


def test_signature_and_layout_changes_reject_plan(api):
    work, mappings = sample()
    signature = [("decode", "FULL")]
    plan, _ = api.build_plan(work, mappings, 1, signature)
    signature.append(("prefill", "NONE"))
    result, reason = api.evaluate_plan(plan, work, mappings, 2, signature)
    assert result is None
    assert reason == "incomparable_window"
    result, reason = api.evaluate_plan(plan, work[::-1], mappings[::-1], 2, [("decode", "FULL")])
    assert result is None
    assert reason == "expert_layout_changed"


def test_unequal_capacities_are_rejected(api):
    work = [{0: 10}, {1: 10, 2: 0, 3: 0}]
    mappings = [[0, -1, -1, -1], [-1, 0, 1, 2]]
    plan, reason = api.build_plan(work, mappings, 1, "decode")
    assert plan is None
    assert reason == "unequal_expert_capacity"


@pytest.mark.parametrize(
    ("work", "mappings", "reason"),
    [
        ([{0: 1, 1: 0}, {0: 0, 2: 0}], [[0, 1, -1, -1], [0, -1, 1, -1]], "invalid_expert_mapping"),
        ([{0: 1, 1: 0}, {2: 0, 3: 0}], [[0, 0, -1, -1], [-1, -1, 0, 1]], "invalid_expert_mapping"),
        ([{0: 1, 1: 0}, {2: -1, 3: 0}], [[0, 1, -1, -1], [-1, -1, 0, 1]], "invalid_expert_work"),
        ([{0: 1, 1: 0}, {2: 0}], [[0, 1, -1, -1], [-1, -1, 0, 1]], "invalid_expert_work"),
        ([{0: 0, 1: 0}, {2: 0, 3: 0}], [[0, 1, -1, -1], [-1, -1, 0, 1]], "no_real_work"),
    ],
)
def test_invalid_work_or_ownership_never_creates_a_plan(api, work, mappings, reason):
    plan, actual = api.build_plan(work, mappings, 1, "decode")
    assert plan is None
    assert actual == reason
