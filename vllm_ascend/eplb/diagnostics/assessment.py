# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Post-run comparison of real EPLB-on placements, without changing live routing."""

import logging
import math
from numbers import Real

from vllm_ascend.eplb.diagnostics.calibration import estimate_adjustment_benefit, latency_interval
from vllm_ascend.eplb.diagnostics.model_calibration import calibrate_layer
from vllm_ascend.eplb.diagnostics.runtime import gather

_COMM_NAMES = {1: "MC2CommImpl", 2: "AlltoAllCommImpl", 3: "AllGatherCommImpl"}
_LOGGER = logging.getLogger("vllm.eplb.diagnostics")


def select_samples(job, samples):
    """Choose identical phase, graph shape and communication across EP ranks."""
    ranks = len(job["initial_placement"])
    if not ranks or any(len(job[key]) != ranks for key in ("current_placement", "metadata", "comm_codes")):
        return None, []
    steps = len(job["loads"])
    if any(history is None or len(history) != steps for key in ("metadata", "comm_codes") for history in job[key]):
        return None, []
    buckets = {}
    for step, loads in enumerate(job["loads"]):
        if not sum(loads):
            continue
        metadata = [rank[step] for rank in job["metadata"] if rank[step] is not None]
        if not metadata:
            continue
        phases = {(value["phase"], value["graph_mode"]) for value in metadata}
        codes = {rank[step] for rank in job["comm_codes"]}
        if len(phases) != 1 or len(codes) != 1:
            continue
        code = next(iter(codes))
        if code not in _COMM_NAMES:
            continue
        shape = tuple(None if rank[step] is None else rank[step].get("padded_tokens") for rank in job["metadata"])
        buckets.setdefault((code, *next(iter(phases)), shape), []).append(step)
    if not buckets:
        return None, []
    key, indices = max(buckets.items(), key=lambda item: len(item[1]))
    count = min(samples, len(indices))
    return key, [indices[index * len(indices) // count] for index in range(count)]


def critical_path_samples(results, name):
    """EP maximum per aligned step; do not average ranks before the maximum."""
    if not results or any(not rank.get(name) for rank in results):
        raise ValueError("Missing aligned timing samples")
    samples = len(results[0][name])
    if any(len(rank[name]) != samples for rank in results):
        raise ValueError("Unaligned timing samples")
    return [
        tuple(max(latency_interval(rank[name][step])[bound] for rank in results) for bound in (0, 1))
        for step in range(samples)
    ]


def observed_overhead(costs):
    """Print raw components and only one set of outer spans as a covered total.

    Device spans include idle gaps and async interference; they are not a
    causal end-to-end slowdown. Host and nested spans must not be added.
    """
    outer = {"eplb_step", "eplb_step_before", "eplb_step_after"}
    if not costs or any(row["pending"] or row["dropped"] for row in costs):
        return None
    totals = []
    families = set()
    for row in costs:
        spans = [record for record in row["totals"] if record["kind"] == "device" and record["component"] in outer]
        if not spans:
            return None
        families.update("v2" if record["component"] == "eplb_step" else "v1" for record in spans)
        try:
            if any(not isinstance(record["sum_ms"], Real) for record in spans):
                return None
            total = sum(latency_interval(record["sum_ms"])[0] for record in spans)
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(total) or len(families) != 1:
            return None
        totals.append(total)
    # Per-rank total maximum is descriptive only, not a per-step EP critical path.
    return (min(totals), max(totals))


def _validate_arguments(recorder, samples, repeats, max_scratch_bytes):
    arguments = (samples, repeats, max_scratch_bytes)
    state = (arguments, recorder.finished, recorder.step)
    if any(peer != state for peer in gather(recorder.group, state)):
        raise ValueError("All EP workers must use identical calibration arguments and recorder state")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in arguments):
        raise ValueError("Calibration arguments must be integers")
    if not 1 <= samples <= 32 or not 1 <= repeats <= 20 or max_scratch_bytes < 1:
        raise ValueError("Calibration needs samples 1..32, repeats 1..20 and a positive scratch cap")
    return arguments


def _measure_job(recorder, name, job, samples, repeats, max_scratch_bytes):
    key, indices = select_samples(job, samples)
    if not indices:
        return None, "no_comparable_real_samples"
    missing = ("per_layer_exposed_overhead", "live_communication_delta", "graph_and_overlap")
    if job["initial_placement"] == job["current_placement"]:
        # Repeating an identical kernel workload measures noise, not a layout
        # adjustment. The exact placement contribution is zero on every rank.
        estimate = estimate_adjustment_benefit(previous_step_ms=[0], current_step_ms=[0], missing=missing)
        return {"estimate": estimate, "key": key, "indices": indices, "unchanged": True}, None
    rank = recorder.group.rank_in_group
    result, reason = None, None
    try:
        previous, current = job["initial_placement"][rank], job["current_placement"][rank]
        previous_counts = [[job["loads"][step][expert] for expert in previous] for step in indices]
        current_counts = [[job["loads"][step][expert] for expert in current] for step in indices]
        probe = dict(recorder.layers)[name].eplb_diagnostic_probe
        result, reason = calibrate_layer(
            probe,
            previous_counts,
            current_counts,
            _COMM_NAMES[key[0]],
            repeats,
            max_scratch_bytes=max_scratch_bytes,
        )
    except Exception as error:
        _LOGGER.warning("EPLB adjustment calibration failed on rank %s: %s", rank, error)
        reason = f"calibration_failed:{type(error).__name__}"
    evidence = gather(recorder.group, {"result": result, "reason": reason})
    failures = [row["reason"] or "missing_result" for row in evidence if row["reason"] or row["result"] is None]
    if failures:
        return None, failures
    results = [row["result"] for row in evidence]
    try:
        estimate = estimate_adjustment_benefit(
            previous_step_ms=critical_path_samples(results, "baseline"),
            current_step_ms=critical_path_samples(results, "candidate"),
            # Actual update costs belong to the entire model/run. Never charge
            # them once per sampled layer or add incomparable layer timings.
            missing=missing,
        )
    except (ValueError, KeyError, TypeError):
        return None, "invalid_timing_evidence"
    return {"estimate": estimate, "key": key, "indices": indices}, None


def calibrate(recorder, *, samples=4, repeats=3, max_scratch_bytes=134217728):
    """Both placements use ON kernels. The live EPLB service must be quiescent."""
    arguments = _validate_arguments(recorder, samples, repeats, max_scratch_bytes)
    recorder.finish()
    previous = getattr(recorder, "calibration_arguments", None)
    success = getattr(recorder, "calibrated", False)
    statuses = gather(recorder.group, (previous, success))
    if all(status == (arguments, True) for status in statuses):
        if recorder.group.rank_in_group == 0:
            _LOGGER.info("[EPLB benefit summary] reason=already_calibrated")
        return
    jobs = gather(recorder.group, recorder.jobs if recorder.group.rank_in_group == 0 else None)[0]
    inventories = gather(recorder.group, [name for name, _ in recorder.layers])
    available = set(jobs).union(*(set(names) for names in inventories))
    completed = 0
    counts = dict.fromkeys(("positive", "nonpositive", "uncertain", "no_adjustment"), 0)
    if recorder.group.rank_in_group == 0:
        for name in sorted(available - jobs.keys()):
            _LOGGER.info("[EPLB benefit] stage=%s layer=%s reason=no_comparable_window", recorder.stage, name)
    for name, job in sorted(jobs.items()):
        result, reason = _measure_job(recorder, name, job, samples, repeats, max_scratch_bytes)
        if reason:
            if recorder.group.rank_in_group == 0:
                _LOGGER.info("[EPLB benefit] stage=%s layer=%s reason=%s", recorder.stage, name, reason)
            continue
        completed += 1
        saving = result["estimate"]["adjustment_saving_ms"]
        unchanged = result.get("unchanged", False)
        if unchanged:
            status = "no_adjustment"
        else:
            status = "positive" if saving[0] > 0 else "nonpositive" if saving[1] <= 0 else "uncertain"
        counts[status] += 1
        if recorder.group.rank_in_group == 0:
            key = result["key"]
            _LOGGER.info(
                "[EPLB benefit] stage=%s layer=%s generation=%s window_end_step=%s reference=initial_live_placement "
                "sampled_steps=%s phase=%s observed_graph=%s comm=%s calibration_execution=%s "
                "adjustment_saving_ms=%s adjustment_status=%s eplb_overhead_ms=shared_run_cost "
                "estimated_net_saving_ms=unknown missing=%s "
                "scope=same_workload_isolated_eplb_on_mlp ranges=%s",
                recorder.stage,
                name,
                job["generation"],
                job["window_end_step"],
                result["indices"],
                key[1],
                key[2],
                _COMM_NAMES[key[0]],
                "not_required" if unchanged else "eager",
                saving,
                status,
                result["estimate"]["missing"],
                "exact_zero" if unchanged else "observed_min_max",
            )
    recorder.calibration_arguments = arguments
    recorder.calibrated = completed > 0 and completed == len(available)
    if recorder.group.rank_in_group == 0:
        _LOGGER.info(
            "[EPLB benefit summary] stage=%s calibrated_layers=%s available_layers=%s comparable_layers=%s "
            "adjustment_layer_counts=%s "
            "covered_update_span_total_ms=%s observed_real_steps=%s eplb_overhead_ms=unknown "
            "estimated_net_saving_ms=unknown missing=fused_route_collection,exposed_overhead,aligned_model_gain "
            "scope=per_layer_estimate_and_actual_update_spans",
            recorder.stage,
            completed,
            len(available),
            len(jobs),
            counts,
            observed_overhead(recorder.costs),
            recorder.observed_steps,
        )
