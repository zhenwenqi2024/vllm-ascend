# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, window-level logging of aggregate workload across one EP group."""

import functools
import logging
from collections import Counter, deque
from types import SimpleNamespace

import torch

from vllm_ascend.eplb_diagnostics.probe import ExpertLoadProbe


def aggregate_work(rows, window_size):
    """Validate source conservation and map valid routes to actual expert owners."""
    if len({r["step"] for r in rows}) != 1:
        return None, "unaligned_windows"
    names = [x["name"] for x in rows[0]["layers"]]
    if not names or any([x["name"] for x in r["layers"]] != names for r in rows):
        return None, "incomplete_layer_set"
    ranks = [r["rank"] for r in rows]
    if len(set(ranks)) != len(ranks):
        return None, "duplicate_rank"
    sources = {}
    for row in rows:
        sources.setdefault(tuple(row["tp_ranks"]), []).append(row)
    for tp, members in sources.items():
        if set(tp) != {r["rank"] for r in members} or len({r["tokens"] for r in members}) != 1:
            return None, "source_tp_mismatch"
    rank_work, hot = [0] * len(rows), {}
    for index, name in enumerate(names):
        layers = [r["layers"][index] for r in rows]
        n = len(layers[0]["counts"]) - 2
        top_k = layers[0]["top_k"]
        if n <= 0 or top_k <= 0 or any(len(x["counts"]) != n + 2 or x["top_k"] != top_k for x in layers):
            return None, "expert_space_mismatch"
        if any(x["counts"][-2] != window_size or x["counts"][-1] or min(x["counts"]) < 0 for x in layers):
            return None, "unsupported_routing_or_call_count"
        for members in sources.values():
            actual = sum(sum(r["layers"][index]["counts"][:-2]) for r in members)
            if actual != members[0]["tokens"] * top_k:
                return None, "source_conservation_failure"
        owners = {}
        for owner, layer in enumerate(layers):
            mapping = layer["mapping"]
            local_slots = [slot for slot in mapping if slot >= 0]
            if (
                len(mapping) != n
                or len(local_slots) != layer["local_experts"]
                or set(local_slots) != set(range(layer["local_experts"]))
            ):
                return None, "invalid_expert_mapping"
            for expert, slot in enumerate(mapping):
                if slot >= 0:
                    if expert in owners:
                        return None, "duplicate_expert_owner"
                    owners[expert] = owner
        if len(owners) != n:
            return None, "missing_expert_owner"
        loads = [sum(x["counts"][e] for x in layers) for e in range(n)]
        for expert, count in enumerate(loads):
            rank_work[owners[expert]] += count
        mean = sum(loads) / n
        cutoff = sorted(loads, reverse=True)[min(4, n) - 1]
        hot.update(
            {(name, e): count for e, count in enumerate(loads) if count > 0 and count >= max(cutoff, 1.2 * mean)}
        )
    total = sum(rank_work)
    if not total:
        return None, "no_real_work"
    return {"rank_work": rank_work, "total": total, "skew": max(rank_work) * len(rows) / total, "hot": hot}, None


class WorkloadLogger:
    """Print one aggregate line per window, with at most four windows of history."""

    def __init__(self, ranks, stage):
        self.ranks, self.stage = ranks, stage
        self.history = deque(maxlen=4)
        self.signature = None

    def log(self, rows, window, window_size):
        result, reason = aggregate_work(rows, window_size)
        logger = logging.getLogger(__name__)
        if reason:
            self.history.clear()
            logger.info(
                "[EPLB diagnostic] stage=%s window=%s ranks=%s insufficient_evidence=%s",
                self.stage,
                window,
                self.ranks,
                reason,
            )
            return
        # Reset persistence when source participation, phase, graph mode or
        # placement changes. Windows contain aggregate work, not per-call peaks.
        signature = [(r["phases"], [x["mapping"] for x in r["layers"]]) for r in rows]
        if signature != self.signature:
            self.history.clear()
        self.signature = signature
        self.history.append((set(result["hot"]), result["skew"] >= 1.2))
        hits = Counter(expert for hot, _ in self.history for expert in hot)
        previous_hot = self.history[-2][0] if len(self.history) >= 2 else set()
        persistent = {e for e in result["hot"] if e in previous_hot and hits[e] / len(self.history) >= 0.8}
        imbalanced = sum(skew for _, skew in self.history)
        hint = "collecting"
        if len(self.history) >= 2:
            hint = (
                ("consider_eplb" if persistent else "insufficient_hotspot_evidence")
                if (imbalanced / len(self.history) >= 0.8)
                else "no_persistent_rank_skew"
            )
        hottest = sorted(result["hot"], key=lambda e: (-result["hot"][e], e))
        hot_text = [f"{name}:{expert}({result['hot'][(name, expert)]})" for name, expert in hottest[:4]]
        stable_text = [
            f"{name}:{expert}({hits[(name, expert)]}/{len(self.history)})"
            for name, expert in hottest
            if (name, expert) in persistent
        ][:4]
        logger.info(
            "[EPLB diagnostic] stage=%s window=%s calls=%s ranks=%s valid_assignments=%s rank_work=%s "
            "window_rank_max_mean=%.3f imbalanced_windows=%s/%s hot_experts=%s "
            "persistent_hot_count=%s persistent_hot_experts=%s hint=%s",
            self.stage,
            window,
            window_size,
            self.ranks,
            result["total"],
            result["rank_work"],
            result["skew"],
            imbalanced,
            len(self.history),
            hot_text,
            len(persistent),
            stable_text,
            hint,
        )


class DiagnosticsRecorder:
    def __init__(self, config, layers, group, tp_ranks, stage, source_tokens):
        self.config, self.layers, self.group = config, layers, group
        self.tp_ranks, self.source_tokens = list(tp_ranks), source_tokens
        self.summary = WorkloadLogger(list(group.ranks), stage)
        self.step = self.tokens = 0
        self.phases = set()
        self.collecting = self.real = False
        self.layout = [
            {
                "name": name,
                "mapping": layer.ascend_expert_map.cpu().tolist(),
                "top_k": layer.moe_config.experts_per_token,
                "local_experts": layer.local_num_experts,
            }
            for name, layer in layers
        ]

    def begin(self, tokens, dummy=False):
        self.step += 1
        offset = self.step - self.config.warmup_steps - 1
        self.collecting = 0 <= offset < self.config.window_size * self.config.max_windows
        self.real = not dummy
        if self.collecting and offset % self.config.window_size == 0:
            self.tokens = 0
            self.phases.clear()
            for _, layer in self.layers:
                layer.eplb_diagnostic_probe.totals.zero_()
        self.source_tokens.fill_(tokens if self.collecting and self.real else 0)
        if self.collecting and self.real:
            self.tokens += tokens

    def end(self):
        if not self.collecting or (self.step - self.config.warmup_steps) % self.config.window_size:
            return
        # Deliberately synchronous only once per window. All group members,
        # including idle DP participants, must join in model-collective order.
        # Dummy requests contribute no routes/tokens or timing; their participation
        # is control bookkeeping. No world/PP collective is added here.
        counters = [layer.eplb_diagnostic_probe.totals for _, layer in self.layers]
        snapshots = torch.cat(counters).cpu().split([counter.numel() for counter in counters])
        row = {
            "step": self.step,
            "rank": self.group.ranks[self.group.rank_in_group],
            "tp_ranks": self.tp_ranks,
            "tokens": self.tokens,
            "phases": sorted(self.phases),
            "layers": [dict(layout, counts=counts.tolist()) for layout, counts in zip(self.layout, snapshots)],
        }
        rows = [None] * len(self.group.ranks)
        # Gloo writes its temporary receive tensors from a background thread.
        with torch.inference_mode(False):
            torch.distributed.all_gather_object(rows, row, group=self.group.cpu_group)
        if self.group.rank_in_group == 0:
            self.summary.log(
                rows, (self.step - self.config.warmup_steps) // self.config.window_size, self.config.window_size
            )


def annotate_batch(runner, **metadata):
    recorder = getattr(runner, "_eplb_diagnostics_recorder", None)
    if recorder is not None and recorder.collecting and recorder.real:
        recorder.phases.add((metadata["phase"], metadata["graph_mode"]))


@torch.inference_mode()
def initialize_diagnostics(runner):
    if runner.ascend_config.eplb_diagnostics.mode == "off":
        return
    parallel = runner.vllm_config.parallel_config
    if not parallel.enable_expert_parallel or parallel.enable_eplb or runner.ascend_config.eplb_config.dynamic_eplb:
        raise ValueError("EPLB diagnostics requires EP enabled and EPLB disabled.")
    config = runner.vllm_config
    capacity = max(
        config.scheduler_config.max_num_batched_tokens, max(config.compilation_config.cudagraph_capture_sizes or [0])
    )
    source_tokens = torch.zeros((), dtype=torch.int64, device=runner.device)
    positions = torch.arange(capacity, device=runner.device)
    for layer in runner.model.modules():
        probe = getattr(layer, "eplb_diagnostic_probe", None)
        if isinstance(probe, ExpertLoadProbe):
            probe.source_token_count, probe.source_positions = source_tokens, positions
    runner._eplb_diagnostics_source_tokens = source_tokens


def start_diagnostics(runner):
    if runner.ascend_config.eplb_diagnostics.mode == "off":
        return
    # Import after worker initialization; portable tests do not initialize vLLM.
    from vllm.distributed.parallel_state import get_pp_group, get_tp_group

    from vllm_ascend.distributed.parallel_state import get_mc2_group

    layers = [
        (getattr(layer, "layer_name", name), layer)
        for name, layer in runner.model.named_modules()
        if isinstance(getattr(layer, "eplb_diagnostic_probe", None), ExpertLoadProbe)
    ]
    if not layers or any(
        layer.ascend_expert_map is None or layer.moe_config.num_experts != layer.moe_config.num_logical_experts
        for _, layer in layers
    ):
        raise ValueError("EPLB diagnostics requires explicit expert ownership without redundant experts.")
    runner._eplb_diagnostics_recorder = DiagnosticsRecorder(
        runner.ascend_config.eplb_diagnostics,
        layers,
        get_mc2_group(),
        get_tp_group().ranks,
        get_pp_group().rank_in_group,
        runner._eplb_diagnostics_source_tokens,
    )
    logging.getLogger(__name__).info(
        "EPLB diagnostics: aggregate valid MoE assignments per EP group/stage; synchronous window summaries; "
        "dummy/padding excluded. consider_eplb is a load signal, not a speedup prediction."
    )


def record_diagnostics(func):
    @functools.wraps(func)
    def wrapped(runner, scheduler_output, *args, **kwargs):
        recorder = getattr(runner, "_eplb_diagnostics_recorder", None)
        dummy = kwargs.get("dummy_run", False)
        if (
            recorder is None
            or getattr(runner, "_eplb_diagnostics_in_execution", False)
            or kwargs.get("is_profile", False)
            or torch.npu.is_current_stream_capturing()
            or (scheduler_output.total_num_scheduled_tokens == 0 and not dummy)
        ):
            return func(runner, scheduler_output, *args, **kwargs)
        runner._eplb_diagnostics_in_execution = True
        try:
            with torch.inference_mode():
                recorder.begin(scheduler_output.total_num_scheduled_tokens, dummy)
            result = func(runner, scheduler_output, *args, **kwargs)
            recorder.end()
            return result
        finally:
            runner._eplb_diagnostics_in_execution = False

    return wrapped


@record_diagnostics
def _record_dummy_batch(runner, scheduler_output, num_tokens, *, dummy_run):
    return runner._dummy_run(num_tokens, uniform_decode=True)


def run_dummy_batch(runner, num_tokens):
    metadata = SimpleNamespace(total_num_scheduled_tokens=0)
    return _record_dummy_batch(runner, metadata, num_tokens, dummy_run=True)
