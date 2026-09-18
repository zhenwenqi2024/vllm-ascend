# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collective, post-generation calibration; never an inference routing change."""

import logging

import torch

from vllm_ascend.eplb.diagnostics.calibration import estimate_net_benefit, latency_interval
from vllm_ascend.eplb.diagnostics.model_calibration import calibrate_layer, expert_payload_bytes
from vllm_ascend.eplb.diagnostics.scratch import measure_overheads

_COMM_NAMES = {1: "MC2CommImpl", 2: "AlltoAllCommImpl", 3: "AllGatherCommImpl"}
_LOGGER = logging.getLogger("vllm.eplb.diagnostics")


def gather(group, value):
    rows = [None] * len(group.ranks)
    with torch.inference_mode(False):
        torch.distributed.all_gather_object(rows, value, group=group.cpu_group)
    return rows


def select_samples(job, samples):
    """Use one real phase/communication/graph bucket, including idle owners."""
    buckets = {}
    for step, loads in enumerate(job["loads"]):
        if not sum(loads):
            continue
        metadata = [rank[step] for rank in job["metadata"] if len(rank) > step and rank[step] is not None]
        if not metadata:
            continue
        phases = {(value["phase"], value["graph_mode"]) for value in metadata}
        codes = {rank[step] for rank in job["comm_codes"] if rank is not None and len(rank) > step}
        if (
            len(phases) != 1
            or len(codes) != 1
            or len(job["comm_codes"]) != sum(rank is not None and len(rank) > step for rank in job["comm_codes"])
        ):
            continue
        code = next(iter(codes))
        if code not in _COMM_NAMES:
            continue
        phase, graph = next(iter(phases))
        # Keep padded source shapes comparable as well as phase/graph mode.
        shape = tuple(
            None if len(rank) <= step or rank[step] is None else rank[step].get("padded_tokens")
            for rank in job["metadata"]
        )
        buckets.setdefault((code, phase, graph, shape), []).append(step)
    if not buckets:
        return None, []
    key, indices = max(buckets.items(), key=lambda item: len(item[1]))
    count = min(samples, len(indices))
    chosen = [indices[index * len(indices) // count] for index in range(count)]
    return key, chosen


def critical_path_interval(results, name):
    """Average aligned per-step EP maxima; never max of per-rank totals."""
    if not results or any(not rank.get(name) for rank in results):
        raise ValueError("Missing aligned timing samples")
    samples = len(results[0][name])
    if any(len(rank[name]) != samples for rank in results):
        raise ValueError("Unaligned timing samples")
    return tuple(
        sum(max(latency_interval(rank[name][step])[bound] for rank in results) for step in range(samples)) / samples
        for bound in (0, 1)
    )


def maximum_interval(values):
    return tuple(max(latency_interval(value)[bound] for value in values) for bound in (0, 1))


def calibrated_budget(results, costs, planner_ms, update_interval):
    if any(len(row.get("baseline", [])) != len(row.get("candidate", [])) for row in results):
        raise ValueError("Baseline and candidate samples must be paired")
    baseline = critical_path_interval(results, "baseline")
    candidate = critical_path_interval(results, "candidate")
    collection = maximum_interval([row["collection_ms"] for row in costs])
    # This arithmetic budget covers ONLY isolated MLP minus scratch proxies.
    # There is no dispatch/combine inside those MLP callbacks (explicit zero).
    # The caller must keep actual EPLB net gain UNKNOWN for omitted live effects.
    budget = estimate_net_benefit(
        baseline_ms=baseline,
        candidate_ms=candidate,
        collection_ms=tuple(value / update_interval for value in collection),
        planner_ms=planner_ms,
        transfer_ms=maximum_interval([row["transfer_ms"] for row in costs]),
        apply_ms=maximum_interval([row["apply_ms"] for row in costs]),
        update_interval=update_interval,
        communication_delta_ms=0,
    )
    return baseline, candidate, budget["gross_saving_ms"], budget["amortized_cost_ms"], budget["net_saving_ms"]


def calibrate(recorder, *, update_interval, samples=4, repeats=3, max_scratch_bytes=134217728):
    """Explicit collective RPC after requests finish; prints evidence, no files.

    Model-backed MLP tests use fresh synthetic inputs and immutable OFF weights.
    Scratch costs are packed serial proxies. Missing ON-kernel, graph, routing
    collection, communication and overlap effects prevent an enable decision.
    """
    group = recorder.group
    arguments = (update_interval, samples, repeats, max_scratch_bytes)
    state = (
        arguments,
        recorder.finished,
        getattr(recorder, "calibrated", False),
        getattr(recorder, "calibration_arguments", None),
        recorder.step,
    )
    peers = gather(group, state)
    if any(peer != state for peer in peers):
        raise ValueError("All EP workers must use identical calibration arguments and recorder state")
    if (
        isinstance(update_interval, bool)
        or not isinstance(update_interval, int)
        or update_interval < 1
        or not isinstance(samples, int)
        or isinstance(samples, bool)
        or not 1 <= samples <= 32
        or not isinstance(repeats, int)
        or isinstance(repeats, bool)
        or not 1 <= repeats <= 20
        or not isinstance(max_scratch_bytes, int)
        or isinstance(max_scratch_bytes, bool)
        or max_scratch_bytes < 1
    ):
        raise ValueError("Calibration needs a positive production update_interval, samples 1..32 and repeats 1..20")
    recorder.finish()
    if getattr(recorder, "calibrated", False) and getattr(recorder, "calibration_arguments", None) == arguments:
        if group.rank_in_group == 0:
            _LOGGER.info(
                "[EPLB calibration summary] stage=%s reason=already_calibrated "
                "production_update_interval=%s samples=%s repeats=%s max_scratch_bytes=%s",
                recorder.summary.stage,
                *arguments,
            )
        return
    jobs = None
    if group.rank_in_group == 0:
        jobs = {name: layer.calibration_job for name, layer in recorder.summary.layers.items() if layer.calibration_job}
    jobs = gather(group, jobs)[0]
    local_layers = dict(recorder.layers)
    completed = 0
    budget_counts = dict.fromkeys(("positive", "nonpositive", "uncertain"), 0)
    for name, job in sorted(jobs.items()):
        key, indices = select_samples(job, samples)
        if not indices:
            if group.rank_in_group == 0:
                _LOGGER.info(
                    "[EPLB calibration] layer=%s evaluation_window=%s reason=no_comparable_real_samples",
                    name,
                    job["window"],
                )
            continue
        comm_name = _COMM_NAMES[key[0]]
        plan = job["plan"]
        rank = group.rank_in_group
        source, target = plan.source_placement[rank], plan.candidate_placement[rank]
        baseline_counts = [[job["loads"][step][expert] for expert in source] for step in indices]
        candidate_counts = [[job["loads"][step][expert] for expert in target] for step in indices]
        result = payload = None
        reason = payload_reason = None
        try:
            probe = local_layers[name].eplb_diagnostic_probe
            result, reason = calibrate_layer(probe, baseline_counts, candidate_counts, comm_name, repeats)
            payload, payload_reason = expert_payload_bytes(probe, comm_name)
        except Exception as error:
            # A rank-local failure must reach peers before their next collective.
            reason = f"calibration_failed:{type(error).__name__}"
        evidence = gather(group, {"result": result, "reason": reason or payload_reason, "bytes": payload})
        failures = [row["reason"] for row in evidence if row["reason"]]
        if failures or len({row["bytes"] for row in evidence}) != 1:
            if group.rank_in_group == 0:
                _LOGGER.info(
                    "[EPLB calibration] layer=%s evaluation_window=%s reason=%s "
                    "conclusion=insufficient_benefit_evidence",
                    name,
                    job["window"],
                    failures or "payload_mismatch",
                )
            continue
        cost = measure_overheads(
            group,
            plan.source_placement,
            plan.candidate_placement,
            payload,
            repeats=repeats,
            max_scratch_bytes=max_scratch_bytes,
        )
        costs = gather(group, cost)
        if any(row["reason"] for row in costs):
            if group.rank_in_group == 0:
                _LOGGER.info(
                    "[EPLB calibration] layer=%s evaluation_window=%s reason=%s "
                    "conclusion=insufficient_benefit_evidence",
                    name,
                    job["window"],
                    [row["reason"] for row in costs],
                )
            continue
        try:
            baseline, candidate, saving, overhead, budget = calibrated_budget(
                [row["result"] for row in evidence], costs, plan.planner_ms, update_interval
            )
        except (ValueError, TypeError, KeyError, IndexError):
            if group.rank_in_group == 0:
                _LOGGER.info("[EPLB calibration] layer=%s reason=invalid_measurement_evidence", name)
            continue
        completed += 1
        budget_status = "positive" if budget[0] > 0 else "nonpositive" if budget[1] <= 0 else "uncertain"
        budget_counts[budget_status] += 1
        if group.rank_in_group == 0:
            _LOGGER.info(
                "[EPLB calibration] stage=%s layer=%s source_window=%s evaluation_window=%s sampled_steps=%s "
                "phase=%s observed_graph=%s calibration_execution=eager comm=%s "
                "baseline_mlp_ms=%s candidate_mlp_ms=%s compute_saving_ms=%s "
                "collection_per_update_ms=%s transfer_per_update_ms=%s apply_per_update_ms=%s "
                "serial_scratch_cost_per_step_ms=%s compute_budget_after_proxy_cost_ms=%s compute_budget_status=%s "
                "production_update_interval=%s planner_ms=%.3f moved_experts=%s logical_bytes_per_expert=%s "
                "conclusion=insufficient_benefit_evidence net_saving_ms=unknown "
                "missing=communication_delta,on_kernel_delta,graph_delta,overlap,production_collection "
                "scope=sampled_isolated_mlp_and_serial_scratch ranges=observed_min_max",
                recorder.summary.stage,
                name,
                plan.source_window,
                job["window"],
                indices,
                key[1],
                key[2],
                comm_name,
                baseline,
                candidate,
                saving,
                maximum_interval([row["collection_ms"] for row in costs]),
                maximum_interval([row["transfer_ms"] for row in costs]),
                maximum_interval([row["apply_ms"] for row in costs]),
                overhead,
                budget,
                budget_status,
                update_interval,
                plan.planner_ms,
                plan.moved_experts,
                payload,
            )
    # Incomplete attempts remain retryable. A successful result is idempotent
    # only for the same settings; a larger scratch cap can recover failed runs.
    recorder.calibrated = completed > 0 and completed == len(jobs)
    recorder.calibration_arguments = arguments
    if group.rank_in_group == 0:
        _LOGGER.info(
            "[EPLB calibration summary] stage=%s calibrated_layers=%s available_layers=%s "
            "uncalibrated_layers=%s compute_budget_layer_counts=%s "
            "conclusion=insufficient_benefit_evidence speedup=not_estimated "
            "reason=isolated_measurements_do_not_cover_live_critical_path",
            recorder.summary.stage,
            completed,
            len(jobs),
            len(jobs) - completed,
            budget_counts,
        )
