# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in timing of real EPLB work; raw overlapping spans are never summed."""

import math
from collections import deque
from contextlib import contextmanager, nullcontext
from functools import wraps
from threading import Lock
from time import perf_counter

import torch


def _layer_name(layer):
    if layer is None or isinstance(layer, (str, int)):
        return layer
    return getattr(layer, "layer_name", None)


class LiveEplbOverhead:
    """Bounded deferred events and cumulative counts, attached to one runner.

    ``active`` is controlled by the recorder for real observed forwards.
    Host and device spans overlap; nested phases also overlap their enclosing
    eplb_step/forward spans. None is an independently additive net overhead.
    Only the enclosing device spans are non-overlapping update intervals;
    fused routing/collection and background resource contention remain outside
    that scope. No host/device synchronization is added to normal execution.
    """

    def __init__(self, on_commit, max_samples=256):
        if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 1:
            raise ValueError("max_samples must be a positive integer")
        self.active = False
        # MRv2 advances EPLB after sampling, after execute_model has returned.
        # Consume this qualification exactly once in AscendEplbState.step.
        self.pending_real = False
        self.on_commit = on_commit
        self.max_samples = max_samples
        self._samples = deque()
        self._pending = []
        self._inflight = 0
        self._totals = {}
        self._dropped = 0
        self._lock = Lock()

    def _record(self, component, kind, layer, duration_ms):
        if not math.isfinite(duration_ms) or duration_ms < 0:
            with self._lock:
                self._dropped += 1
            return
        sample = dict(component=component, kind=kind, layer=_layer_name(layer), duration_ms=duration_ms)
        key = component, kind, sample["layer"]
        with self._lock:
            total = self._totals.setdefault(
                key, dict(component=component, kind=kind, layer=sample["layer"], count=0, sum_ms=0.0)
            )
            total["count"] += 1
            total["sum_ms"] += duration_ms
            if len(self._samples) == self.max_samples:
                self._samples.popleft()
            self._samples.append(sample)

    @contextmanager
    def cpu_span(self, component, layer=None):
        if not self.active:
            yield
            return
        start = perf_counter()
        try:
            yield
        finally:
            self._record(component, "host", layer, (perf_counter() - start) * 1000)

    @contextmanager
    def device_span(self, component, layer=None, stream=None):
        if not self.active or torch.npu.is_current_stream_capturing():
            yield
            return
        with self._lock:
            accepted = len(self._pending) + self._inflight < self.max_samples
            if accepted:
                self._inflight += 1
            else:
                self._dropped += 1
        if not accepted:
            yield
            return
        try:
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record(stream)
        except Exception:
            with self._lock:
                self._inflight -= 1
                self._dropped += 1
            yield
            return
        try:
            yield
        finally:
            try:
                end.record(stream)
            except Exception:
                with self._lock:
                    self._inflight -= 1
                    self._dropped += 1
            else:
                with self._lock:
                    self._inflight -= 1
                    self._pending.append((component, _layer_name(layer), start, end))

    def committed(self, layer):
        # Commits during an idle/dummy participant still change the placement
        # used by its next real forward. Notification must remain CPU-only.
        self.on_commit(layer)

    def wrap_policy(self, policy):
        """Attach host planner timing to an in-process policy instance (MRv2)."""
        original = getattr(policy, "_eplb_diagnostic_original_rebalance", policy.rebalance_experts)

        @wraps(original)
        def rebalance(*args, **kwargs):
            with self.cpu_span("planner_compute"):
                return original(*args, **kwargs)

        policy._eplb_diagnostic_original_rebalance = original
        policy.rebalance_experts = rebalance

    def drain(self, *, synchronize=False):
        """Resolve completed events; synchronize=True is only for an explicit final RPC.

        ``samples`` is bounded recent detail. ``totals`` persists across drains
        and includes every finalized measurement. ``dropped`` counts timing
        events that could not be retained/recorded, making coverage incomplete.
        """
        with self._lock:
            pending, self._pending = self._pending, []
            # Background transfer spans can finish while events are queried.
            # Keep these slots reserved until their resolution is complete.
            self._inflight += len(pending)
        unresolved = []
        for component, layer, start, end in pending:
            try:
                if synchronize:
                    end.synchronize()
                elif not end.query():
                    unresolved.append((component, layer, start, end))
                    continue
                self._record(component, "device", layer, float(start.elapsed_time(end)))
            except Exception:
                with self._lock:
                    self._dropped += 1
        with self._lock:
            self._inflight -= len(pending)
            self._pending = unresolved + self._pending
            samples = list(self._samples)
            self._samples.clear()
            return dict(
                samples=samples,
                totals=[dict(value) for value in self._totals.values()],
                pending=len(self._pending) + self._inflight,
                dropped=self._dropped,
            )


def monitor_call(component, *, device=False):
    """Time an attached instance only; an uninstrumented call creates no timers."""

    def decorate(function):
        @wraps(function)
        def call(instance, *args, **kwargs):
            monitor = getattr(instance, "_eplb_diagnostic_monitor", None)
            if monitor is None or not monitor.active:
                return function(instance, *args, **kwargs)
            with (
                monitor.cpu_span(component),
                monitor.device_span(component) if device else nullcontext(),
            ):
                return function(instance, *args, **kwargs)

        return call

    return decorate
