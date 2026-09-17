# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize local JSONL traces without live distributed communication.

Usage: python -m vllm_ascend.eplb_diagnostics.report /path/to/logs
"""

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def routed_group_sample(members_rows):
    """Attribute masked source destinations to actual owners, not padded slots."""
    rows = sorted(members_rows, key=lambda pair: pair[0]["rank"])
    sources = [layer.get("routing", {}) for _, layer in rows]
    if any(s.get("semantics") != "valid_scheduled_mc2_source_destinations" for s in sources):
        return None, "valid_routing_unavailable"
    assignments = [s.get("physical_assignments") for s in sources]
    if any(not a or any(n < 0 for n in a) for a in assignments):
        return None, "invalid_routing_counts"
    n = len(assignments[0])
    if any(len(a) != n for a in assignments):
        return None, "physical_space_mismatch"
    top_ks = {layer["layout_end"].get("experts_per_token") for _, layer in rows}
    if len(top_ks) != 1 or next(iter(top_ks)) is None or next(iter(top_ks)) <= 0:
        return None, "source_top_k_unknown"
    top_k = next(iter(top_ks))
    tp_sources = defaultdict(list)
    members = {row["rank"] for row, _ in rows}
    for (row, _), assignment in zip(rows, assignments):
        tp = tuple(sorted(row["groups"].get("tp", {}).get("ranks", [])))
        if row["rank"] not in tp or not set(tp).issubset(members):
            return None, "source_tp_ownership_unknown"
        tp_sources[tp].append((row, assignment))
    for tp, source_rows in tp_sources.items():
        scheduled = {row["batch"].get("scheduled_tokens") for row, _ in source_rows}
        if {row["rank"] for row, _ in source_rows} != set(tp) or len(scheduled) != 1 or None in scheduled:
            return None, "source_tp_ownership_unknown"
        if sum(sum(a) for _, a in source_rows) != next(iter(scheduled)) * top_k:
            return None, "source_schedule_conservation_failure"
    owners, logical = {}, None
    backends = [0] * n
    for index, (_, layer) in enumerate(rows):
        layout = layer["layout_end"]
        if layout.get("mixed_shared_placement"):
            return None, "shared_placement_unresolved"
        candidate = list(range(n))
        if layout.get("map_semantics") == "physical_to_local" and layout.get("dynamic_eplb"):
            candidate = [None] * n
            for expert, physical_ids in enumerate(layout.get("logical_to_physical", [])):
                for physical in physical_ids:
                    if physical < 0:
                        continue
                    if physical >= n or candidate[physical] is not None:
                        return None, "invalid_replica_layout"
                    candidate[physical] = expert
        if any(e is None or e >= layout.get("num_logical_experts", 0) for e in candidate):
            return None, "logical_ownership_unknown"
        if logical is not None and logical != candidate:
            return None, "inconsistent_replica_layout"
        logical = candidate
        mapping = layout.get("global_to_local", [])
        if len(mapping) != n:
            return None, "physical_ownership_unknown"
        local_slots = set()
        for physical, local in enumerate(mapping):
            if local < 0:
                continue
            if physical in owners or local in local_slots or local >= len(layer["expert_assignments"]):
                return None, "duplicate_or_invalid_owner"
            local_slots.add(local)
            owners[physical] = index
            backends[physical] = layer["expert_assignments"][local]
    if len(owners) != n:
        return None, "missing_physical_owner"
    physical_loads = [sum(a[p] for a in assignments) for p in range(n)]
    if any(actual < valid for actual, valid in zip(backends, physical_loads)):
        return None, "routing_backend_conservation_failure"
    loads = [0] * len(rows)
    logical_loads = [0] * (max(logical) + 1)
    for p, count in enumerate(physical_loads):
        loads[owners[p]] += count
        logical_loads[logical[p]] += count
    return {
        "ordinal": rows[0][1]["call_start"],
        "loads": loads,
        "logical_loads": logical_loads,
        "logical_owners": (
            [owners[logical.index(e)] for e in range(len(logical_loads))] if sorted(logical) == list(range(n)) else None
        ),
        "unattributed_backend_work": sum(backends) - sum(physical_loads),
        "layout": tuple(layer.get("layout_fingerprint") for _, layer in rows),
    }, None


def expert_diagnosis(windows, *, window_size, min_windows, skew_threshold, persistent_fraction):
    """Describe expert skew and each selected expert's persistence in full windows."""
    if not windows:
        return {
            "load_status": "insufficient_contiguous_samples",
            "hotspot_status": "insufficient_contiguous_samples",
            "complete_windows": 0,
            "persistent_hot_experts": [],
            "experts": [],
        }
    num_experts = len(windows[0]["logical_expert_assignments"])
    total = sum(sum(w["logical_expert_assignments"]) for w in windows)
    totals = [sum(w["logical_expert_assignments"][e] for w in windows) for e in range(num_experts)]
    hot_windows, streaks, longest = Counter(), Counter(), Counter()
    previous_segment = None
    chain = longest_chain = 0
    for window in windows:
        if window["segment_index"] != previous_segment:
            streaks.clear()
            chain = 0
        chain += 1
        longest_chain = max(longest_chain, chain)
        previous_segment = window["segment_index"]
        hot = set(window["hot_logical_experts"])
        for expert in list(streaks):
            if expert not in hot:
                del streaks[expert]
        for expert in hot:
            hot_windows[expert] += 1
            streaks[expert] += 1
            longest[expert] = max(longest[expert], streaks[expert])
    persistent = [
        e
        for e in range(num_experts)
        if hot_windows[e] / len(windows) >= persistent_fraction and longest[e] >= min_windows
    ]
    hotspot_status = "insufficient_contiguous_samples"
    if longest_chain >= min_windows:
        if persistent:
            hotspot_status = "persistent_hot_experts_observed"
        elif hot_windows:
            hotspot_status = "no_persistent_hot_experts_observed"
        else:
            hotspot_status = "no_hot_experts_observed"
    imbalanced = sum(w["expert_max_mean"] >= skew_threshold for w in windows)
    return {
        "load_status": "expert_imbalance_observed" if imbalanced else "no_expert_imbalance_observed",
        "hotspot_status": hotspot_status,
        "basis": "logical_expert_work_in_complete_windows",
        "complete_windows": len(windows),
        "observed_calls": len(windows) * window_size,
        "total_valid_assignments": total,
        "num_logical_experts": num_experts,
        "imbalanced_window_fraction": imbalanced / len(windows),
        "expert_max_mean_p50": percentile([w["expert_max_mean"] for w in windows], 0.5),
        "hot_selection": "top_4_including_ties_and_at_least_mean_times_skew_threshold",
        "hot_windows_present": sum(bool(w["hot_logical_experts"]) for w in windows),
        "persistent_hot_experts": persistent,
        "persistence_rule": {
            "min_hot_window_fraction": persistent_fraction,
            "min_consecutive_hot_windows": min_windows,
            "window_size_calls": window_size,
            "scope": "window_aggregate_not_every_call_or_wall_clock_lifetime",
        },
        "experts": [
            {
                "logical_expert": e,
                "valid_assignments": count,
                "load_share": count / total,
                "load_over_mean": count * num_experts / total,
                "hot_windows": hot_windows[e],
                "hot_window_fraction": hot_windows[e] / len(windows),
                "longest_consecutive_hot_windows": longest[e],
                "calls_in_longest_hot_window_run": longest[e] * window_size,
                "persistent_hot": e in persistent,
                "observed_owner_ranks": sorted(
                    {w["logical_expert_ranks"][e] for w in windows if w.get("logical_expert_ranks") is not None}
                ),
            }
            for e, count in enumerate(totals)
        ],
    }


def slot_balanced_candidate(expert_loads, owners, rank_count):
    """Diagnostic greedy placement under observed slot counts, not an EPLB policy."""
    slots = [owners.count(rank) for rank in range(rank_count)]
    loads = [0] * rank_count
    placement = [None] * len(owners)
    for expert in sorted(range(len(expert_loads)), key=lambda e: (-expert_loads[e], e)):
        rank = min(
            (r for r in range(rank_count) if slots[r]),
            key=lambda r: (loads[r], r != owners[expert], r),
        )
        placement[expert] = rank
        slots[rank] -= 1
        loads[rank] += expert_loads[expert]
    return placement


def placement_diagnosis(window_blocks, ranks, *, min_windows, persistent_fraction):
    """Plan on one window and evaluate each call of the next contiguous window."""
    pairs, skipped = [], Counter()
    for (previous, train), (current, evaluate) in zip(window_blocks, window_blocks[1:]):
        if previous["segment_index"] != current["segment_index"]:
            skipped["discontinuous_windows"] += 1
            continue
        owners = train[0].get("logical_owners")
        if owners is None or any(s.get("logical_owners") != owners for s in train + evaluate):
            skipped["replicas_or_changing_ownership_unsupported"] += 1
            continue

        # An exact replay of the observed placement must succeed before a
        # counterfactual mapping is trusted. No future counts enter planning.
        def replay(sample, placement):
            loads = [0] * len(ranks)
            for expert, rank in enumerate(placement):
                loads[rank] += sample["logical_loads"][expert]
            return loads

        if any(replay(s, owners) != s["loads"] for s in train + evaluate):
            skipped["current_placement_replay_mismatch"] += 1
            continue
        candidate = slot_balanced_candidate(previous["logical_expert_assignments"], owners, len(ranks))
        current_peaks = [max(s["loads"]) for s in evaluate]
        proposed_peaks = [max(replay(s, candidate)) for s in evaluate]
        baseline, proposed = sum(current_peaks), sum(proposed_peaks)
        pairs.append(
            {
                "train_call_start": previous["call_start"],
                "train_call_end": previous["call_end"],
                "evaluation_call_start": current["call_start"],
                "evaluation_call_end": current["call_end"],
                "current_peak_assignments_sum": baseline,
                "candidate_peak_assignments_sum": proposed,
                "held_out_peak_work_reduction": 1 - proposed / baseline,
                "improved_call_fraction": sum(a < b for a, b in zip(proposed_peaks, current_peaks)) / len(evaluate),
                "regressed_call_fraction": sum(a > b for a, b in zip(proposed_peaks, current_peaks)) / len(evaluate),
                "moved_experts": sum(a != b for a, b in zip(candidate, owners)),
                "candidate_expert_ranks": [ranks[index] for index in candidate],
            }
        )
    reduction = positive_fraction = None
    status = "insufficient_adjacent_windows"
    if pairs:
        reduction = 1 - sum(p["candidate_peak_assignments_sum"] for p in pairs) / sum(
            p["current_peak_assignments_sum"] for p in pairs
        )
        positive_fraction = sum(p["held_out_peak_work_reduction"] > 0 for p in pairs) / len(pairs)
        if len(pairs) >= min_windows:
            status = (
                "held_out_work_reduction_observed"
                if reduction > 0 and positive_fraction >= persistent_fraction
                else "candidate_did_not_show_consistent_work_reduction"
            )
    return {
        "status": status,
        "candidate_kind": "slot_only_greedy_not_configured_eplb_policy",
        "constraints": "observed_expert_slots_per_rank_no_new_replicas",
        "interpretation": "per_call_max_assignments_not_latency_or_throughput",
        "evaluation": "previous_window_plan_next_contiguous_window_test",
        "min_evaluation_windows": min_windows,
        "evaluated_windows": len(pairs),
        "positive_window_fraction": positive_fraction,
        "held_out_peak_work_reduction": reduction,
        "skipped_pairs": dict(skipped),
        "limits": [
            "not_the_configured_EPLB_policy_or_a_deployable_plan",
            "node_topology_communication_and_migration_cost_not_modeled",
            "no_update_delay_or_redundant_expert_simulation",
            "overlapping_train_test_pairs_are_not_independent_trials",
        ],
        "pairs": pairs,
    }


def enablement_assessment(experts, rank_status, placement, *, min_work_reduction):
    """Screen the need to enable EPLB from off-state work evidence alone."""
    decision = "insufficient_evidence"
    if rank_status == "insufficient_contiguous_samples":
        signal = "insufficient_workload_evidence"
    elif rank_status != "persistent_rank_imbalance_observed":
        signal = "no_persistent_rank_imbalance_observed"
        decision = "not_recommended_from_observed_load"
    elif not experts["persistent_hot_experts"]:
        signal = "rank_imbalance_without_persistent_hot_experts"
    elif (
        placement["status"] == "held_out_work_reduction_observed"
        and placement["held_out_peak_work_reduction"] >= min_work_reduction
    ):
        signal = "persistent_hotspots_rank_imbalance_and_held_out_work_reduction"
        decision = "enable_candidate"
    else:
        signal = "persistent_hotspots_and_rank_imbalance_without_consistent_candidate_evidence"
    return {
        "decision": decision,
        "scope": "off_state_workload_screening_not_measured_speedup",
        "workload_signal": signal,
        "expert_load_status": experts["load_status"],
        "rank_load_status": rank_status,
        "persistent_hot_experts": experts["persistent_hot_experts"],
        "candidate_work_status": placement["status"],
        "held_out_peak_work_reduction": placement["held_out_peak_work_reduction"],
        "min_work_reduction": min_work_reduction,
        "threshold_basis": "configurable_heuristic_not_performance_calibrated",
        "estimated_speedup": None,
        "unresolved_for_performance_prediction": [
            "configured_EPLB_policy_and_deployment_constraints",
            "MoE_critical_path_time_attributable_to_rank_imbalance",
            "exposed_update_cost_and_layout_effective_lifetime",
        ],
        "timing_limit": "execute_model_stream_time_is_not_per_layer_MoE_time",
    }


def persistence_summary(
    samples,
    ranks,
    *,
    window_size,
    min_windows,
    skew_threshold,
    persistent_fraction,
    idle_calls=None,
    min_work_reduction=0.05,
):
    """Only bridge gaps proven to contain whole-group idle calls with the same layout."""
    samples = sorted(samples, key=lambda s: s["ordinal"])
    idle_calls = idle_calls or {}
    segments = []
    skipped_idle_calls = 0
    for sample in samples:
        previous = segments[-1][-1] if segments else None
        gap = range(previous["ordinal"] + 1, sample["ordinal"]) if previous else range(0)
        idle_gap = previous is not None and all(idle_calls.get(i) == sample["layout"] for i in gap)
        if not segments or not idle_gap or sample["layout"] != segments[-1][-1]["layout"]:
            segments.append([])
        else:
            skipped_idle_calls += len(gap)
        segments[-1].append(sample)
    windows, retentions, window_blocks = [], [], []
    longest_window_chain = 0
    for segment_index, segment in enumerate(segments):
        previous_hot = None
        window_chain = 0
        for start in range(0, len(segment) - window_size + 1, window_size):
            block = segment[start : start + window_size]
            nonempty = [s for s in block if sum(s["loads"]) > 0]
            if len(nonempty) != window_size:
                previous_hot = None
                window_chain = 0
                continue
            window_chain += 1
            longest_window_chain = max(longest_window_chain, window_chain)
            ratios = [max(s["loads"]) / statistics.mean(s["loads"]) for s in block]
            experts = [sum(s["logical_loads"][e] for s in block) for e in range(len(block[0]["logical_loads"]))]
            # Include cutoff ties so expert IDs do not invent persistence.
            cutoff = sorted(experts, reverse=True)[min(4, len(experts)) - 1]
            hot = {
                e
                for e, count in enumerate(experts)
                if count > 0 and count >= cutoff and count >= statistics.mean(experts) * skew_threshold
            }
            if previous_hot is not None and hot | previous_hot:
                retentions.append(len(hot & previous_hot) / len(hot | previous_hot))
            previous_hot = hot
            busiest = Counter()
            for s in block:
                peak = max(s["loads"])
                if s["loads"].count(peak) == 1:
                    busiest[ranks[s["loads"].index(peak)]] += 1
            windows.append(
                {
                    "segment_index": segment_index,
                    "call_start": block[0]["ordinal"],
                    "call_end": block[-1]["ordinal"] + 1,
                    "rank_skew_p50": percentile(ratios, 0.5),
                    "rank_valid_assignments": {rank: sum(s["loads"][i] for s in block) for i, rank in enumerate(ranks)},
                    "imbalanced_fraction": sum(r >= skew_threshold for r in ratios) / window_size,
                    "busiest_rank_counts": dict(busiest),
                    "hot_logical_experts": sorted(hot),
                    "logical_expert_assignments": experts,
                    "logical_expert_ranks": (
                        [ranks[index] for index in block[0]["logical_owners"]]
                        if block[0].get("logical_owners") is not None
                        else None
                    ),
                    "expert_max_mean": max(experts) / statistics.mean(experts),
                }
            )
            window_blocks.append((windows[-1], block))
    fraction = (
        (sum(w["imbalanced_fraction"] >= persistent_fraction for w in windows) / len(windows)) if windows else None
    )
    status = "insufficient_contiguous_samples"
    if len(windows) >= min_windows:
        status = (
            "persistent_rank_imbalance_observed"
            if fraction >= persistent_fraction
            else "no_persistent_imbalance_observed"
        )
    hotspot_status = "insufficient_adjacent_windows"
    if longest_window_chain >= min_windows:
        if not any(w["hot_logical_experts"] for w in windows):
            hotspot_status = "no_hot_experts_observed"
        elif retentions:
            hotspot_status = (
                "stable_hot_experts_observed"
                if statistics.mean(retentions) >= persistent_fraction
                else "hot_experts_shift_observed"
            )
    experts = expert_diagnosis(
        windows,
        window_size=window_size,
        min_windows=min_windows,
        skew_threshold=skew_threshold,
        persistent_fraction=persistent_fraction,
    )
    placement = placement_diagnosis(
        window_blocks, ranks, min_windows=min_windows, persistent_fraction=persistent_fraction
    )
    return {
        "status": status,
        "basis": "masked_routed_assignments_not_latency_or_EPLB_gain",
        "samples": len(samples),
        "contiguous_segments": [len(s) for s in segments],
        "max_observed_contiguous_calls": max(map(len, segments), default=0),
        "continuity_basis": "nonempty_calls_with_only_verified_whole_group_idle_gaps",
        "verified_idle_calls_skipped": skipped_idle_calls,
        "window_size": window_size,
        "min_windows": min_windows,
        "skew_threshold": skew_threshold,
        "persistent_fraction_threshold": persistent_fraction,
        "persistent_window_fraction": fraction,
        "hot_expert_adjacent_window_jaccard": statistics.mean(retentions) if retentions else None,
        "hotspot_status": hotspot_status,
        "hotspot_adjacent_comparisons": len(retentions),
        "max_contiguous_windows": longest_window_chain,
        "hotspot_lifetime": "not_extrapolated_across_gaps",
        "unattributed_backend_work": sum(s["unattributed_backend_work"] for s in samples),
        "expert_diagnosis": experts,
        "placement_diagnosis": placement,
        "enablement_assessment": enablement_assessment(
            experts, status, placement, min_work_reduction=min_work_reduction
        ),
        "windows": windows,
    }


def summarize(
    records,
    *,
    aligned_collective_ordinals: bool = False,
    window_size: int = 8,
    min_windows: int = 2,
    skew_threshold: float = 1.2,
    persistent_fraction: float = 0.8,
    min_work_reduction: float = 0.05,
) -> dict:
    """Compute skew per invocation before aggregating in time.

    New traces may establish alignment through the post-warmup counter contract.
    Legacy traces require explicit verification. Independent DP schedulers have
    different local step IDs; only proven collective ordinals may be joined.
    """
    if window_size < 2 or min_windows < 2 or not math.isfinite(skew_threshold) or skew_threshold <= 1:
        raise ValueError("Need window_size/min_windows >= 2 and finite skew_threshold > 1")
    if not 0 < persistent_fraction <= 1:
        raise ValueError("persistent_fraction must be in (0, 1]")
    if not 0 < min_work_reduction <= 1:
        raise ValueError("min_work_reduction must be in (0, 1]")
    quality = Counter()
    local = defaultdict(list)
    groups = defaultdict(list)
    executions = defaultdict(list)
    seen = set()
    rank_processes = defaultdict(set)
    for row in records:
        if row.get("schema_version") != 1:
            quality["unsupported_schema"] += 1
            continue
        # Historical traces may contain synthetic DP work. Never use it for
        # expert counts, timing, alignment joins, or proof of idle continuity.
        if row["batch"].get("dummy_run") or row["batch"].get("phase") == "dummy":
            quality["dummy_run_excluded"] += 1
            continue
        identity = (row["run_id"], row["host"], row["pid"], row["rank"], row["local_step"])
        if identity in seen:
            quality["duplicate_step"] += 1
            continue
        seen.add(identity)
        rank_processes[(row["run_id"], row["rank"])].add((row["host"], row["pid"]))
        phase = row["batch"].get("phase", "unknown")
        graph = row["batch"].get("graph_mode", "unknown")
        executions[(row["run_id"], row["rank"], phase, graph)].append(row["execute_stream_ms"])
        for layer in row["layers"]:
            quality[layer["quality"]] += 1
            if layer["quality"] != "ok":
                continue
            counts = layer["expert_assignments"]
            if not counts or sum(counts) == 0:
                quality["empty_work"] += 1
            graph = row["batch"].get("graph_mode", "unknown")
            phase = row["batch"].get("phase", "unknown")
            key = (row["run_id"], row["rank"], layer["layer"], phase, graph)
            if counts and sum(counts) > 0:
                local[key].append(max(counts) / statistics.mean(counts))
            if layer["layout_changed"]:
                quality["layout_changed_within_sample"] += 1
                continue
            if layer["calls"] != 1:
                # Summing multiple invocations can hide alternating hotspots.
                quality["multiple_invocations_not_joined"] += 1
                continue
            comm_indices = [i for i, calls in enumerate(layer["comm_calls"]) if calls]
            if len(comm_indices) != 1:
                quality["ambiguous_communication"] += 1
                continue
            comm = layer["comm_methods"][comm_indices[0]]
            group_name = "mc2" if comm in ("MC2CommImpl", "FusedMC2CommImpl") else "ep"
            group = row["groups"].get(group_name)
            if group is None or comm == "unknown":
                quality["missing_group"] += 1
                continue
            members = tuple(group["ranks"])
            if len(members) != layer["layout_end"]["ep_size"]:
                quality["expert_and_communication_group_mismatch"] += 1
                continue
            # PP stages and distinct EP groups must never be merged.
            group_key = (row["run_id"], members, layer["layer"], layer["call_start"], layer["call_end"], comm)
            groups[group_key].append((row, layer))

    rank_summary = []
    for key, ratios in sorted(local.items()):
        run, rank, layer, phase, graph = key
        rank_summary.append(
            {
                "run_id": run,
                "rank": rank,
                "layer": layer,
                "phase": phase,
                "graph_mode": graph,
                "samples": len(ratios),
                "local_expert_max_mean_p50": percentile(ratios, 0.50),
                "local_expert_max_mean_p95": percentile(ratios, 0.95),
                "meaning": "within_rank_expert_skew_does_not_establish_EP_rank_skew",
            }
        )

    aligned = defaultdict(list)
    routed = defaultdict(list)
    workload_routed = defaultdict(list)
    idle_calls = defaultdict(dict)
    routing_quality = Counter()
    for key, members_rows in groups.items():
        run, members, layer_name, _, _, comm = key
        ranks = [row["rank"] for row, _ in members_rows]
        if len(ranks) != len(set(ranks)):
            quality["ambiguous_rank_sample"] += 1
            continue
        if set(ranks) != set(members):
            quality["incomplete_group_sample"] += 1
            continue
        if any(len(rank_processes[(run, rank)]) != 1 for rank in members):
            quality["worker_restart_or_rank_collision"] += 1
            continue
        automatic_alignment = all(
            row.get("alignment_contract") == "post_warmup_executed_collective_v2"
            and layer.get("alignment_eligible") is True
            and layer["call_start"] == row.get("expected_model_calls", -1) - 1
            and layer["call_end"] == row.get("expected_model_calls", -1)
            for row, layer in members_rows
        )
        if not aligned_collective_ordinals and not automatic_alignment:
            quality["complete_group_alignment_unverified"] += 1
            continue
        # DP ranks may legitimately have different phases and padded token counts.
        # Preserve that vector rather than pretending all ranks used one batch.
        batch_mix = tuple(
            sorted((row["rank"], row["batch"].get("phase"), row["batch"].get("graph_mode")) for row, _ in members_rows)
        )
        sample, reason = routed_group_sample(members_rows)
        if reason:
            routing_quality[reason] += 1
        else:
            routed_mix = tuple(
                sorted(
                    (
                        row["rank"],
                        row["batch"].get("phase"),
                        row["batch"].get("graph_mode"),
                        str(row["batch"].get("padded_tokens", "unknown")),
                    )
                    for row, _ in members_rows
                )
            )
            if sum(sample["loads"]) == 0:
                idle_calls[(run, members, layer_name, comm)][sample["ordinal"]] = sample["layout"]
                routing_quality["whole_group_idle_samples"] += 1
            else:
                # DP schedulers may alternate which source rank is active. This
                # does not interrupt the observed group workload. Preserve the
                # full source pattern, but stratify work-only analysis by the
                # phases/graph buckets of sources that actually have user work.
                active_mix = tuple(
                    sorted(
                        {
                            (
                                row["batch"].get("phase"),
                                row["batch"].get("graph_mode"),
                                str(row["batch"].get("padded_tokens", "unknown")),
                            )
                            for row, _ in members_rows
                            if row["batch"].get("scheduled_tokens", 0) > 0
                        }
                    )
                )
                sample["source_batch_pattern"] = routed_mix
                routed[(run, members, layer_name, comm, active_mix)].append(sample)
                phase_graph = tuple(sorted({(phase, graph) for phase, graph, _ in active_mix}))
                workload_routed[(run, members, layer_name, comm, phase_graph)].append(sample)
            routing_quality["valid_aligned_routing_samples"] += 1
        loads = [sum(layer["expert_assignments"]) for _, layer in members_rows]
        peak = max(loads)
        if peak == 0:
            continue
        avg = statistics.mean(loads)
        item = {
            "rank_max_mean": peak / avg,
            "ideal_work_reduction_proxy": 1.0 - avg / peak,
        }
        aligned[(run, members, layer_name, comm, batch_mix)].append(item)

    comparisons = []
    for (run, members, layer, comm, batch_mix), samples in sorted(aligned.items()):
        comparisons.append(
            {
                "run_id": run,
                "ranks": members,
                "layer": layer,
                "comm_method": comm,
                "rank_phase_graph": batch_mix,
                "samples": len(samples),
                "rank_max_mean_p50": percentile([s["rank_max_mean"] for s in samples], 0.50),
                "rank_max_mean_p95": percentile([s["rank_max_mean"] for s in samples], 0.95),
                "ideal_work_reduction_proxy_p50": percentile([s["ideal_work_reduction_proxy"] for s in samples], 0.50),
                "conclusion": "work_skew_only_requires_critical_path_timing_and_EPLB_cost_measurement",
            }
        )
    timing = [
        {
            "run_id": run,
            "rank": rank,
            "phase": phase,
            "graph_mode": graph,
            "samples": len(values),
            "execute_stream_ms_p50": percentile(values, 0.50),
            "execute_stream_ms_p95": percentile(values, 0.95),
            "scope": "local_execute_model_stream_not_request_TPOT_or_service_throughput",
        }
        for (run, rank, phase, graph), values in sorted(executions.items())
    ]
    result = {
        "schema_version": 1,
        "quality_counts": dict(quality),
        "collective_ordinal_alignment_user_asserted": aligned_collective_ordinals,
        "local_expert_skew": rank_summary,
        "rank_work_skew": comparisons,
        "routing_quality": dict(routing_quality),
        "persistent_imbalance": [
            {
                "run_id": run,
                "ranks": members,
                "layer": layer,
                "comm_method": comm,
                "active_source_phase_graph_buckets": batch_mix,
                "observed_source_batch_patterns": [
                    {"rank_phase_graph_bucket": pattern, "samples": count}
                    for pattern, count in Counter(s["source_batch_pattern"] for s in samples).items()
                ],
                **persistence_summary(
                    samples,
                    tuple(sorted(members)),
                    window_size=window_size,
                    min_windows=min_windows,
                    skew_threshold=skew_threshold,
                    persistent_fraction=persistent_fraction,
                    idle_calls=idle_calls[(run, members, layer, comm)],
                    min_work_reduction=min_work_reduction,
                ),
            }
            for (run, members, layer, comm, batch_mix), samples in sorted(routed.items())
        ],
        "execution_timing": timing,
        "workload_persistence": [
            {
                "run_id": run,
                "ranks": members,
                "layer": layer,
                "comm_method": comm,
                "active_source_phase_graph": phase_graph,
                "scope": "observed_workload_including_bucket_changes",
                "comparison_limit": "work_counts_only_not_same_shape_latency",
                "observed_source_batch_patterns": [
                    {"rank_phase_graph_bucket": pattern, "samples": count}
                    for pattern, count in Counter(s["source_batch_pattern"] for s in samples).items()
                ],
                **persistence_summary(
                    samples,
                    tuple(sorted(members)),
                    window_size=window_size,
                    min_windows=min_windows,
                    skew_threshold=skew_threshold,
                    persistent_fraction=persistent_fraction,
                    idle_calls=idle_calls[(run, members, layer, comm)],
                    min_work_reduction=min_work_reduction,
                ),
            }
            for (run, members, layer, comm, phase_graph), samples in sorted(workload_routed.items())
        ],
        "recommendation": "insufficient_evidence_to_claim_EPLB_speedup",
        "recommendation_scope": "legacy_performance_claim_only_use_enablement_summary_for_off_state_screening",
        "next_measurements_scope": "optional_performance_prediction_not_required_for_enablement_screening",
        "next_measurements": [
            "MoE critical-path time with a graph-aware device profiler",
            "EPLB migration/wait/steady-state overhead over complete rebalance cycles",
            "matched EPLB off/on workload benchmark, including diagnostic overhead control",
        ],
    }
    result["enablement_summary"] = {
        "scope": "off_state_workload_screening_not_measured_speedup",
        "decisions": dict(Counter(a["enablement_assessment"]["decision"] for a in result["workload_persistence"])),
        "candidate_layers": [
            {"run_id": a["run_id"], "layer": a["layer"], "ranks": a["ranks"]}
            for a in result["workload_persistence"]
            if a["enablement_assessment"]["decision"] == "enable_candidate"
        ],
        "no_assessments_means": "insufficient_evidence_check_routing_quality_and_group_completeness",
        "aggregation_limit": "layer_counts_are_not_service_speedup_and_work_reductions_must_not_be_summed",
    }
    return result


def load_records(paths):
    for path in paths:
        files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
        for file in files:
            with file.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON at {file}:{line_number}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--min-windows", type=int, default=2)
    parser.add_argument("--skew-threshold", type=float, default=1.2)
    parser.add_argument("--persistent-fraction", type=float, default=0.8)
    parser.add_argument("--min-work-reduction", type=float, default=0.05)
    parser.add_argument(
        "--aligned-collective-ordinals",
        action="store_true",
        help="Assert that ranks have identical collective/capture counter origins and stable topology.",
    )
    args = parser.parse_args()
    result = summarize(
        load_records(args.paths),
        aligned_collective_ordinals=args.aligned_collective_ordinals,
        window_size=args.window_size,
        min_windows=args.min_windows,
        skew_threshold=args.skew_threshold,
        persistent_fraction=args.persistent_fraction,
        min_work_reduction=args.min_work_reduction,
    )
    output = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is None:
        print(output, end="")
    else:
        args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
