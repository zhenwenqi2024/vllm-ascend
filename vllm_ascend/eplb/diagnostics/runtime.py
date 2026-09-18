# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, per-layer workload logging across one EP group."""

import functools
import logging
from collections import Counter
from types import SimpleNamespace

import torch

from vllm_ascend.eplb.diagnostics.placement import build_plan, evaluate_plan
from vllm_ascend.eplb.diagnostics.probe import ExpertLoadProbe

_LOG_NAME = "vllm.eplb.diagnostics"


def layer_work(rows, window_size):
    """Validate source conservation and map valid routes to actual expert owners."""
    if len({r["step"] for r in rows}) != 1:
        return None, "unaligned_windows"
    names = [x["name"] for x in rows[0]["layers"]]
    if len(names) != 1 or any([x["name"] for x in r["layers"]] != names for r in rows):
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
    return {
        "rank_work": rank_work,
        "total": total,
        "skew": max(rank_work) * len(rows) / total,
        "hot": hot,
        "expert_work": [{e: loads[e] for e in range(n) if owners[e] == rank} for rank in range(len(rows))],
    }, None


def validate_history_sources(rows, histories, calls):
    sources = {}
    for index, row in enumerate(rows):
        tokens = row.get("token_steps", [])
        if len(tokens) != calls or any(not isinstance(count, int) or count < 0 for count in tokens):
            return False
        sources.setdefault(tuple(row["tp_ranks"]), []).append(index)
    for members in sources.values():
        expected = rows[members[0]]["token_steps"]
        if any(rows[index]["token_steps"] != expected for index in members):
            return False
        top_k = rows[members[0]]["layers"][0]["top_k"]
        if any(
            sum(sum(histories[index][step][:-2]) for index in members) != expected[step] * top_k
            for step in range(calls)
        ):
            return False
    return True


class LayerLogger:
    """Keep cumulative evidence for one layer, with no retained sample history."""

    def __init__(self, ranks, stage, layer):
        self.ranks, self.stage, self.layer = ranks, stage, layer
        self.signature = None
        self.valid_windows = self.invalid_windows = self.observed_calls = self.imbalanced_windows = 0
        self.hot_windows, self.streaks, self.longest = Counter(), Counter(), Counter()
        self.expert_totals = [Counter() for _ in ranks]
        self.hint = "collecting"
        self.persistent = set()
        self.pending_plan = self.calibration_job = None
        self.projection = None

    def log(self, rows, window, window_size, complete=True):
        result, reason = layer_work(rows, window_size)
        # Spawned workers configure the vLLM logging namespace.
        logger = logging.getLogger(_LOG_NAME)
        if reason:
            self.streaks.clear()
            self.pending_plan = self.calibration_job = self.projection = None
            self.invalid_windows += 1
            self.hint = "insufficient_evidence"
            logger.info(
                "[EPLB diagnostic] stage=%s layer=%s window=%s ranks=%s insufficient_evidence=%s",
                self.stage,
                self.layer,
                window,
                self.ranks,
                reason,
            )
            return
        signature = [(r["phases"], r["layers"][0]["mapping"]) for r in rows]
        if signature != self.signature:
            self.streaks.clear()
        self.signature = signature
        self.observed_calls += window_size
        hot = set(result["hot"])
        if complete:
            self.valid_windows += 1
            self.imbalanced_windows += result["skew"] >= 1.2
            self.hot_windows.update(hot)
            self.streaks = Counter({e: self.streaks[e] + 1 for e in hot})
            for expert, streak in self.streaks.items():
                self.longest[expert] = max(self.longest[expert], streak)
            self.persistent = {
                e for e, hits in self.hot_windows.items() if hits / self.valid_windows >= 0.8 and self.longest[e] >= 2
            }
            self.hint = "collecting"
            if self.valid_windows >= 2:
                self.hint = (
                    ("persistent_load_skew" if self.persistent else "insufficient_hotspot_evidence")
                    if (self.imbalanced_windows / self.valid_windows >= 0.8)
                    else "no_persistent_rank_skew"
                )
        self.project(rows, result, window, window_size, signature, complete)
        for rank, counts, total, work in zip(
            self.ranks, result["expert_work"], self.expert_totals, result["rank_work"]
        ):
            total.update(counts)
            # Include owned cold experts. Expert IDs belong to this layer only.
            logger.info(
                "[EPLB experts] stage=%s layer=%s rank=%s window=%s window_work=%s cumulative_work=%s expert_work=%s",
                self.stage,
                self.layer,
                rank,
                window,
                work,
                sum(total.values()),
                [f"{e}:{count}" for e, count in sorted(total.items())],
            )
        cumulative_rank_work = [sum(counts.values()) for counts in self.expert_totals]
        cumulative_peak = max(cumulative_rank_work)
        hottest = sorted(hot, key=lambda e: (-result["hot"][e], e))
        hot_text = [f"{e}({result['hot'][(name, e)]})" for name, e in hottest[:4]]
        stable_text = [
            f"{e}({self.hot_windows[(name, e)]}/{self.valid_windows})" for name, e in sorted(self.persistent)
        ]
        logger.info(
            "[EPLB diagnostic] stage=%s layer=%s window=%s calls=%s ranks=%s valid_assignments=%s rank_work=%s "
            "window_rank_max_mean=%.3f cumulative_rank_max_mean=%.3f busiest_rank=%s "
            "imbalanced_windows=%s/%s invalid_windows=%s observed_calls=%s "
            "hot_experts=%s persistent_hot_experts=%s hint=%s phases=%s",
            self.stage,
            self.layer,
            window,
            window_size,
            self.ranks,
            result["total"],
            result["rank_work"],
            result["skew"],
            cumulative_peak * len(self.ranks) / sum(cumulative_rank_work),
            self.ranks[cumulative_rank_work.index(cumulative_peak)],
            self.imbalanced_windows,
            self.valid_windows,
            self.invalid_windows,
            self.observed_calls,
            hot_text,
            stable_text,
            self.hint,
            sorted({phase for row in rows for phase, _ in row["phases"]}),
        )

    def project(self, rows, result, window, calls, signature, complete):
        if not complete:
            self.pending_plan = None
            return
        mappings = [row["layers"][0]["mapping"] for row in rows]
        previous = self.pending_plan
        self.projection, reason = evaluate_plan(previous, result["expert_work"], mappings, window, signature)
        self.calibration_job = None
        if self.projection is not None:
            histories = [row["layers"][0].get("history") for row in rows]
            if all(history is not None for history in histories):
                experts = len(mappings[0])
                if any(
                    len(h) != calls or any(len(step) != experts + 2 or step[-2] != 1 or step[-1] for step in h)
                    for h in histories
                ):
                    reason = "invalid_per_step_history"
                elif any(
                    [sum(step[e] for step in h) for e in range(experts + 2)] != row["layers"][0]["counts"]
                    for h, row in zip(histories, rows)
                ):
                    reason = "history_conservation_failure"
                elif not validate_history_sources(rows, histories, calls):
                    reason = "per_step_source_conservation_failure"
                else:
                    loads = [
                        [sum(history[t][e] for history in histories) for e in range(experts)] for t in range(calls)
                    ]
                    before = [[sum(step[e] for e in rank) for rank in previous.source_placement] for step in loads]
                    after = [[sum(step[e] for e in rank) for rank in previous.candidate_placement] for step in loads]
                    baseline = sum(max(step) for step in before)
                    candidate = sum(max(step) for step in after)
                    self.projection.update(
                        step_peak_work=baseline,
                        candidate_step_peak_work=candidate,
                        step_peak_reduction=1 - candidate / baseline,
                    )
                    self.calibration_job = {
                        "plan": previous,
                        "loads": loads,
                        "window": window,
                        "metadata": [row.get("step_metadata", []) for row in rows],
                        "comm_codes": [row["layers"][0].get("comm_codes") for row in rows],
                    }
            else:
                reason = "missing_per_step_history"
            logging.getLogger(_LOG_NAME).info(
                "[EPLB projection] stage=%s layer=%s window=%s policy=default_eplb source_window=%s "
                "candidate_rank_work=%s moved_experts=%s step_peak_work=%s candidate_step_peak_work=%s "
                "work_reduction=%s planner_ms=%.3f timing=not_estimated reason=%s",
                self.stage,
                self.layer,
                window,
                previous.source_window,
                self.projection["candidate_rank_work"],
                previous.moved_experts,
                self.projection.get("step_peak_work", "unknown"),
                self.projection.get("candidate_step_peak_work", "unknown"),
                self.projection.get("step_peak_reduction", "unknown"),
                previous.planner_ms,
                reason or "none",
            )
        self.pending_plan, plan_reason = build_plan(result["expert_work"], mappings, window, signature)
        if self.projection is None or plan_reason:
            logging.getLogger(_LOG_NAME).info(
                "[EPLB projection] stage=%s layer=%s window=%s reason=%s timing=not_estimated",
                self.stage,
                self.layer,
                window,
                plan_reason or reason,
            )


class WorkloadLogger:
    """Print each rank/layer, then an overall assessment based on layer evidence."""

    def __init__(self, ranks, stage):
        self.ranks, self.stage = ranks, stage
        self.layers = {}

    def log(self, rows, window, window_size, complete=True):
        names = sorted({layer["name"] for row in rows for layer in row["layers"]} | self.layers.keys())
        for name in names:
            if name not in self.layers:
                self.layers[name] = LayerLogger(self.ranks, self.stage, name)
            scoped = [dict(row, layers=[layer for layer in row["layers"] if layer["name"] == name]) for row in rows]
            self.layers[name].log(scoped, window, window_size, complete)
        self.summarize(window)

    def summarize(self, window, final=False):
        names = list(self.layers)
        counts = Counter(layer.hint for layer in self.layers.values())
        candidates = [name for name, layer in self.layers.items() if layer.hint == "persistent_load_skew"]
        logging.getLogger(_LOG_NAME).info(
            "[EPLB summary] stage=%s window=%s final=%s ranks=%s layers=%s layer_assessments=%s skewed_layers=%s "
            "invalid_layer_windows=%s conclusion=insufficient_benefit_evidence "
            "missing=calibrated_compute_and_eplb_cost scope=observed_valid_windows speedup=not_estimated",
            self.stage,
            window,
            final,
            self.ranks,
            len(names),
            dict(counts),
            candidates,
            sum(layer.invalid_windows for layer in self.layers.values()),
        )


class DiagnosticsRecorder:
    def __init__(self, config, layers, group, tp_ranks, stage, source_tokens):
        self.config, self.layers, self.group = config, layers, group
        self.tp_ranks, self.source_tokens = list(tp_ranks), source_tokens
        self.summary = WorkloadLogger(list(group.ranks), stage)
        self.step = self.tokens = 0
        self.phases = set()
        self.step_metadata = []
        self.token_steps = []
        self.history_slot = layers[0][1].eplb_diagnostic_probe.history_slot if layers else None
        self.collecting = self.real = self.finished = False
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
        self.collecting = (
            not self.finished
            and offset >= 0
            and (self.config.max_windows == 0 or offset < self.config.window_size * self.config.max_windows)
        )
        self.real = not dummy
        if self.collecting and offset % self.config.window_size == 0:
            self.tokens = 0
            self.phases.clear()
            self.step_metadata.clear()
            self.token_steps.clear()
            for _, layer in self.layers:
                probe = layer.eplb_diagnostic_probe
                probe.totals.zero_()
                if probe.history is not None:
                    probe.history.zero_()
                    probe.comm_history.zero_()
        if self.history_slot is not None:
            self.history_slot.fill_(max(0, offset) % self.config.window_size)
        if self.collecting:
            self.step_metadata.append(None)
            self.token_steps.append(tokens if self.real else 0)
        self.source_tokens.fill_(tokens if self.collecting and self.real else 0)
        if self.collecting and self.real:
            self.tokens += tokens

    def end(self):
        if not self.collecting or (self.step - self.config.warmup_steps) % self.config.window_size:
            return
        self._collect(self.config.window_size)

    def _collect(self, calls, complete=True):
        # Deliberately synchronous only once per window. All group members,
        # including idle DP participants, must join in model-collective order.
        # Dummy requests contribute no routes/tokens or timing; their participation
        # is control bookkeeping. No world/PP collective is added here.
        counters = [layer.eplb_diagnostic_probe.totals for _, layer in self.layers]
        snapshots = torch.cat(counters).cpu().split([counter.numel() for counter in counters])
        histories = [layer.eplb_diagnostic_probe.history for _, layer in self.layers]
        if all(history is not None for history in histories):
            flat = torch.cat([history[:calls].flatten() for history in histories]).cpu()
            history_rows = [
                part.reshape(calls, -1).tolist() for part in flat.split([calls * h.shape[1] for h in histories])
            ]
            comm_rows = (
                torch.stack([layer.eplb_diagnostic_probe.comm_history[:calls] for _, layer in self.layers])
                .cpu()
                .tolist()
            )
        else:
            history_rows, comm_rows = [None] * len(histories), [None] * len(histories)
        row = {
            "step": self.step,
            "rank": self.group.ranks[self.group.rank_in_group],
            "tp_ranks": self.tp_ranks,
            "tokens": self.tokens,
            "phases": sorted(self.phases),
            "step_metadata": self.step_metadata,
            "token_steps": self.token_steps,
            "layers": [
                dict(layout, counts=counts.tolist(), history=history, comm_codes=comm)
                for layout, counts, history, comm in zip(self.layout, snapshots, history_rows, comm_rows)
            ],
        }
        rows = [None] * len(self.group.ranks)
        # Gloo writes its temporary receive tensors from a background thread.
        with torch.inference_mode(False):
            torch.distributed.all_gather_object(rows, row, group=self.group.cpu_group)
        if self.group.rank_in_group == 0:
            self.summary.log(
                rows, (self.step - self.config.warmup_steps - 1) // self.config.window_size + 1, calls, complete
            )

    @torch.inference_mode()
    def finish(self):
        """Called collectively after generation, never from asynchronous teardown."""
        if self.finished:
            return
        pending = (self.step - self.config.warmup_steps) % self.config.window_size
        if self.collecting and pending:
            self._collect(pending, complete=False)
        self.finished = True
        self.source_tokens.zero_()
        if self.group.rank_in_group == 0:
            self.summary.summarize(
                max(0, (self.step - self.config.warmup_steps - 1) // self.config.window_size + 1), final=True
            )


def annotate_batch(runner, **metadata):
    recorder = getattr(runner, "_eplb_diagnostics_recorder", None)
    if recorder is not None and recorder.collecting and recorder.real:
        recorder.phases.add((metadata["phase"], metadata["graph_mode"]))
        if recorder.step_metadata:
            recorder.step_metadata[-1] = metadata


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
    history_slot = torch.zeros(1, dtype=torch.int64, device=runner.device)
    for layer in runner.model.modules():
        probe = getattr(layer, "eplb_diagnostic_probe", None)
        if isinstance(probe, ExpertLoadProbe):
            probe.source_token_count, probe.source_positions = source_tokens, positions
            probe.history = torch.zeros(
                (runner.ascend_config.eplb_diagnostics.window_size, probe.num_experts + 2),
                dtype=torch.int64,
                device=runner.device,
            )
            probe.history_slot = history_slot
            probe.comm_history = torch.zeros(
                runner.ascend_config.eplb_diagnostics.window_size, dtype=torch.int64, device=runner.device
            )
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
    logging.getLogger(_LOG_NAME).info(
        "EPLB diagnostics: per-layer valid MoE assignments across each EP group/stage; synchronous window summaries; "
        "dummy/padding excluded. Load skew alone never recommends EPLB; isolated calibration is required."
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
