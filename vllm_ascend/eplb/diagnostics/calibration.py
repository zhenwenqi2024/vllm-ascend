# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched actual EPLB-layout timing and explicitly scoped live cost arithmetic.

The arithmetic separates layout adjustment saving from measured live overhead.
It never infers milliseconds from route counts or treats missing costs as free.
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


def estimate_adjustment_benefit(
    *,
    previous_step_ms=None,
    current_step_ms=None,
    overhead_total_ms=None,
    observed_steps=None,
    missing,
) -> dict:
    """Separate actual layout saving, covered live overhead and estimated net.

    previous_step_ms/current_step_ms are aligned lists of per-step EP maximum
    MLP latency bounds, using the same logical workload and EPLB-on kernels for
    the previous and current *observed* layouts. Maximum over ranks precedes
    mean over steps; do not pass maximum of rank means or mix different layers.

    overhead_total_ms is a non-overlapping, measured exposed span over the
    matching observed_steps. It is not a sum of nested or overlapping raw host,
    device, planner and transfer durations. No intended update interval is used.

    missing is required: the caller names unmeasured collection, graph,
    communication, overlap/interference, or other omitted scope. The reported
    eplb_overhead_ms covers only the supplied observed spans. measured_budget_ms
    subtracts that covered overhead, but estimated_net_saving_ms stays unknown
    until missing is empty. Sample min/max bounds are not confidence intervals.
    """
    if isinstance(missing, str):
        raise ValueError("missing must explicitly list nonempty scope names")
    omissions = list(dict.fromkeys(missing))
    if any(not isinstance(name, str) or not name for name in omissions):
        raise ValueError("missing must explicitly list nonempty scope names")
    saving = overhead = budget = None
    if previous_step_ms is None or current_step_ms is None:
        omissions.append("matched_layout_timing")
    else:
        previous, current = list(previous_step_ms), list(current_step_ms)
        if len(previous) != len(current):
            raise ValueError("previous and current layout samples must be paired")
        if not previous:
            omissions.append("matched_layout_timing")
        else:
            previous = [latency_interval(value) for value in previous]
            current = [latency_interval(value) for value in current]
            saving = (
                sum(old[0] - new[1] for old, new in zip(previous, current)) / len(previous),
                sum(old[1] - new[0] for old, new in zip(previous, current)) / len(previous),
            )
    if observed_steps is None:
        omissions.append("observed_steps")
    elif isinstance(observed_steps, bool) or not isinstance(observed_steps, int) or observed_steps < 1:
        raise ValueError("observed_steps must be a positive integer")
    if overhead_total_ms is None:
        omissions.append("total_exposed_overhead")
    else:
        total = latency_interval(overhead_total_ms)
        if observed_steps is not None:
            overhead = tuple(value / observed_steps for value in total)
    if saving is not None and overhead is not None:
        budget = saving[0] - overhead[1], saving[1] - overhead[0]
    omissions = list(dict.fromkeys(omissions))
    net = budget if not omissions else None
    if net is None:
        conclusion = "insufficient_evidence"
    elif net[0] > 0:
        conclusion = "positive_margin"
    elif net[1] <= 0:
        conclusion = "cost_exceeds_benefit"
    else:
        conclusion = "uncertain"
    return {
        "adjustment_saving_ms": saving,
        "eplb_overhead_ms": overhead,
        "measured_budget_ms": budget,
        "estimated_net_saving_ms": net,
        "missing": omissions,
        "conclusion": conclusion,
    }
