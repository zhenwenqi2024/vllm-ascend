# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated device measurements and explicit EPLB benefit/cost bounds.

These helpers never change live expert placement or derive milliseconds from
routing counts. Calibration must run collectively outside normal inference.
"""

import math
from collections.abc import Callable, Mapping
from numbers import Real
from statistics import median


def measure_callable(factory: Callable, repeats: int = 5, warmup: int = 2) -> dict:
    """Time independent scratch operations, including an NPUGraph replay callable.

    ``factory()`` runs before timing each repetition and returns a zero-argument
    callable retaining its scratch inputs. It may capture a graph and return its
    replay callable. The operation must join auxiliary streams to the current
    stream and must not mutate live model parameters, buffers, or route maps.

    Synchronization is intentional for this isolated calibration, never the
    serving hot path. Device events include device execution and any host launch
    gaps, but exclude allocation and input preparation performed by the factory.
    The reported range is the observed sample range, not a confidence interval.
    """
    import torch

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    if torch.npu.is_current_stream_capturing():
        raise RuntimeError("EPLB calibration must run outside graph capture")
    samples = []
    with torch.inference_mode():
        for index in range(warmup + repeats):
            operation = factory()
            if not callable(operation):
                raise TypeError("calibration factory must return a callable")
            torch.npu.synchronize()
            if index < warmup:
                result = operation()
                torch.npu.synchronize()
            else:
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                result = operation()
                end.record()
                end.synchronize()
                elapsed = float(start.elapsed_time(end))
                if not math.isfinite(elapsed) or elapsed < 0:
                    raise ValueError("device timing returned an invalid latency")
                samples.append(elapsed)
            # Inputs and outputs must outlive asynchronous device execution.
            del result, operation
    return {"min_ms": min(samples), "median_ms": median(samples), "max_ms": max(samples), "samples": len(samples)}


def latency_interval(value, *, signed: bool = False) -> tuple[float, float]:
    """Normalize a scalar, (lower, upper), or measure_callable result to ms bounds."""
    if isinstance(value, Mapping):
        endpoints = value["min_ms"], value["max_ms"]
    elif isinstance(value, Real) and not isinstance(value, bool):
        endpoints = value, value
    else:
        try:
            endpoints = tuple(value)
        except TypeError as exc:
            raise ValueError("latency must be a scalar or an interval") from exc
        if len(endpoints) != 2:
            raise ValueError("latency interval requires two endpoints")
    if any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) for x in endpoints):
        raise ValueError("latency endpoints must be finite numbers")
    lower, upper = map(float, endpoints)
    if lower > upper or (not signed and lower < 0):
        raise ValueError("latency bounds must be ordered and nonnegative")
    return lower, upper


def estimate_net_benefit(
    *,
    baseline_ms=None,
    candidate_ms=None,
    collection_ms=None,
    planner_ms=None,
    transfer_ms=None,
    apply_ms=None,
    update_interval=None,
    communication_delta_ms=None,
) -> dict:
    """Compare per-step savings with explicitly supplied cost bounds.

    Baseline/candidate must have the same per-step measurement scope, model,
    shapes, quantization, communication method, and graph mode. For GMM-only
    measurements, communication_delta_ms bounds the omitted communication
    change (candidate minus baseline); use explicit zero only when justified.
    Collection is per step. Planner, transfer and apply are per update, divided
    by the intended *production* update interval, not the diagnostic window.

    Costs represent critical-path exposure, or explicitly conservative upper
    bounds. Raw asynchronous stage durations must not be called exposed costs.
    None means unknown; unknown components cannot silently become free.
    """
    components = {
        "baseline_ms": baseline_ms,
        "candidate_ms": candidate_ms,
        "collection_ms": collection_ms,
        "planner_ms": planner_ms,
        "transfer_ms": transfer_ms,
        "apply_ms": apply_ms,
        "communication_delta_ms": communication_delta_ms,
    }
    missing = [name for name, value in components.items() if value is None]
    if update_interval is None:
        missing.append("update_interval")
    elif isinstance(update_interval, bool) or not isinstance(update_interval, int) or update_interval < 1:
        raise ValueError("update_interval must be a positive integer")
    intervals = {
        name: latency_interval(value, signed=name == "communication_delta_ms")
        for name, value in components.items()
        if value is not None
    }
    if missing:
        return {
            "conclusion": "insufficient_evidence",
            "missing": missing,
            "gross_saving_ms": None,
            "amortized_cost_ms": None,
            "net_saving_ms": None,
        }
    baseline, candidate = intervals["baseline_ms"], intervals["candidate_ms"]
    gross = baseline[0] - candidate[1], baseline[1] - candidate[0]
    cost = tuple(
        intervals["collection_ms"][i]
        + intervals["communication_delta_ms"][i]
        + sum(intervals[key][i] for key in ("planner_ms", "transfer_ms", "apply_ms")) / update_interval
        for i in (0, 1)
    )
    net = gross[0] - cost[1], gross[1] - cost[0]
    if net[0] > 0:
        conclusion = "positive_margin"
    elif net[1] <= 0:
        conclusion = "cost_exceeds_benefit"
    else:
        conclusion = "uncertain"
    return {
        "conclusion": conclusion,
        "missing": [],
        "gross_saving_ms": gross,
        "amortized_cost_ms": cost,
        "net_saving_ms": net,
    }
