# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Live EPLB diagnostics. Initial and current placements are both real layouts."""

import functools
import logging
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm_ascend.eplb.diagnostics.overhead import LiveEplbOverhead
from vllm_ascend.eplb.diagnostics.probe import ExpertLoadProbe

_LOGGER = logging.getLogger("vllm.eplb.diagnostics")


def gather(group, value):
    rows = [None] * len(group.ranks)
    with torch.inference_mode(False):
        torch.distributed.all_gather_object(rows, value, group=group.cpu_group)
    return rows


def local_node_id():
    """Linux boot identity is shared by containers on one host; never log it."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip() or None
    except OSError:
        return None


def logical_to_physical(layer):
    if getattr(layer, "_use_v2_model_runner", False):
        state = getattr(getattr(layer, "router", None), "eplb_state", None)
        mapping = getattr(state, "logical_to_physical_map", None)
        return None if mapping is None else mapping[:, 0]
    return getattr(layer, "log2phy", None)


def placement(mapping, ranks):
    """Zero-redundancy permutation, ordered by physical rank and local slot."""
    if not mapping or len(mapping) % ranks or sorted(mapping) != list(range(len(mapping))):
        raise ValueError("EPLB benefit diagnostics requires a complete nonredundant expert permutation")
    inverse = sorted(range(len(mapping)), key=mapping.__getitem__)
    slots = len(mapping) // ranks
    return tuple(tuple(inverse[start : start + slots]) for start in range(0, len(mapping), slots))


def logical_work(rows, name, calls):
    """Validate logical and owner counters even when placement changes."""
    if not rows or calls < 1:
        return None, "empty_window"
    if len({row["step"] for row in rows}) != 1 or len({row["rank"] for row in rows}) != len(rows):
        return None, "unaligned_ranks"
    layers = [row["layers"].get(name) for row in rows]
    if any(layer is None for layer in layers):
        return None, "missing_layer"
    try:
        for layer in layers:
            placement(layer["initial"], len(rows))
            placement(layer["current"], len(rows))
    except ValueError:
        return None, "unsupported_placement"
    experts = len(layers[0]["initial"])
    histories = [layer["history"] for layer in layers]
    if any(
        len(history) != calls
        or any(len(step) != experts + 2 or step[-2] != 1 or step[-1] or min(step) < 0 for step in history)
        for history in histories
    ):
        return None, "invalid_route_history"
    sources = {}
    for index, row in enumerate(rows):
        sources.setdefault(tuple(row["tp_ranks"]), []).append(index)
    top_k = layers[0]["top_k"]
    if top_k < 1 or any(layer["top_k"] != top_k for layer in layers):
        return None, "top_k_mismatch"
    for ranks, members in sources.items():
        expected = rows[members[0]]["token_steps"]
        if set(ranks) != {rows[i]["rank"] for i in members} or len(expected) != calls:
            return None, "source_tp_mismatch"
        if any(rows[i]["token_steps"] != expected for i in members):
            return None, "source_token_mismatch"
        if any(
            count < 0 or sum(sum(histories[i][step][:-2]) for i in members) != count * top_k
            for step, count in enumerate(expected)
        ):
            return None, "source_conservation_failure"
    loads = [
        [sum(history[step][expert] for history in histories) for expert in range(experts)] for step in range(calls)
    ]
    if not any(sum(step) for step in loads):
        return None, "no_real_work"
    return loads, None


def validate_window(rows, name, calls, loads=None):
    """Only stable, aligned placement generations may become timing evidence."""
    if loads is None:
        loads, reason = logical_work(rows, name, calls)
        if reason:
            return None, reason
    layers = [row["layers"][name] for row in rows]
    if any(layer["generation_start"] != layer["generation_end"] for layer in layers):
        return None, "placement_changed_during_window"
    if len({layer["generation_end"] for layer in layers}) != 1:
        return None, "unaligned_placement_generations"
    if any(layer["initial"] != layers[0]["initial"] or layer["current"] != layers[0]["current"] for layer in layers):
        return None, "placement_mismatch"
    try:
        old = placement(layers[0]["initial"], len(rows))
        current = placement(layers[0]["current"], len(rows))
    except ValueError:
        return None, "unsupported_placement"
    return {
        "initial_placement": old,
        "current_placement": current,
        "loads": loads,
        "metadata": [row["metadata"] for row in rows],
        "comm_codes": [layer["comm_codes"] for layer in layers],
        "generation": layers[0]["generation_end"],
        "window_end_step": rows[0]["step"],
    }, None


class DiagnosticsRecorder:
    def __init__(self, config, layers, group, tp_ranks, stage, source_tokens):
        self.config, self.layers, self.group = config, layers, group
        self.tp_ranks, self.stage, self.source_tokens = list(tp_ranks), stage, source_tokens
        self.step = self.observed_steps = self.window_calls = 0
        self.collecting = self.real = self.finished = False
        self.metadata, self.token_steps, self.generations = [], [], {}
        self.initial = {name: logical_to_physical(layer).cpu().tolist() for name, layer in layers}
        for mapping in self.initial.values():
            placement(mapping, len(group.ranks))
        self.jobs, self.expert_totals, self.invalid_windows = {}, {}, Counter()
        self.logical_totals = {}
        self.history_slot = layers[0][1].eplb_diagnostic_probe.history_slot
        self.monitor = LiveEplbOverhead(on_commit=self.committed)
        self.costs = None

    def committed(self, layer):
        layer = getattr(layer, "routed_experts", layer)
        probe = getattr(layer, "eplb_diagnostic_probe", None)
        if probe is not None:
            probe.generation += 1

    def begin(self, tokens, dummy=False):
        self.step += 1
        offset = self.step - self.config.warmup_steps - 1
        self.collecting = (
            not self.finished
            and offset >= 0
            and (not self.config.max_windows or offset < self.config.max_windows * self.config.window_size)
        )
        self.real = not dummy and tokens > 0
        self.monitor.observing = self.collecting
        self.monitor.active = self.collecting and self.real
        self.monitor.pending_real = self.monitor.active
        self.source_tokens.fill_(tokens if self.monitor.active else 0)
        if not self.collecting:
            return
        if not self.window_calls:
            self.metadata, self.token_steps = [], []
            for name, layer in self.layers:
                probe = layer.eplb_diagnostic_probe
                probe.history.zero_()
                probe.comm_history.zero_()
                probe.owner_totals.zero_()
                self.generations[name] = probe.generation
        self.history_slot.fill_(self.window_calls)
        self.window_calls += 1
        self.observed_steps += int(self.real)
        self.metadata.append(None)
        self.token_steps.append(tokens if self.real else 0)

    def end(self):
        self.monitor.active = False
        if self.collecting and self.window_calls == self.config.window_size:
            self._collect()
        self.monitor.drain()

    def _collect(self):
        calls = self.window_calls
        probes = [layer.eplb_diagnostic_probe for _, layer in self.layers]
        flat = torch.cat([probe.history[:calls].flatten() for probe in probes]).cpu()
        histories = [
            part.reshape(calls, -1).tolist() for part in flat.split([calls * p.history.shape[1] for p in probes])
        ]
        comms = torch.stack([probe.comm_history[:calls] for probe in probes]).cpu().tolist()
        owner_flat = torch.cat([probe.owner_totals.flatten() for probe in probes]).cpu()
        ownership = [
            part.reshape(len(self.group.ranks), -1).tolist()
            for part in owner_flat.split([probe.owner_totals.numel() for probe in probes])
        ]
        mappings = torch.cat([logical_to_physical(layer) for _, layer in self.layers]).cpu()
        current = [part.tolist() for part in mappings.split([probe.num_experts for probe in probes])]
        row = {
            "step": self.step,
            "rank": self.group.ranks[self.group.rank_in_group],
            "tp_ranks": self.tp_ranks,
            "token_steps": self.token_steps,
            "metadata": self.metadata,
            "layers": {
                name: {
                    "history": history,
                    "comm_codes": comm,
                    "owner_counts": owned,
                    "initial": self.initial[name],
                    "current": mapping,
                    "generation_start": self.generations[name],
                    "generation_end": layer.eplb_diagnostic_probe.generation,
                    "top_k": layer.moe_config.experts_per_token,
                }
                for (name, layer), history, comm, owned, mapping in zip(
                    self.layers, histories, comms, ownership, current
                )
            },
        }
        rows = gather(self.group, row)
        if self.group.rank_in_group == 0:
            for name, _ in self.layers:
                self._log_layer(rows, name, calls)
        self.window_calls = 0

    def _log_layer(self, rows, name, calls):
        loads, reason = logical_work(rows, name, calls)
        if loads is not None:
            reason = self._log_work(rows, name, loads)
        job = None
        if reason is None:
            job, reason = validate_window(rows, name, calls, loads)
        if reason:
            self.jobs.pop(name, None)
            self.invalid_windows[name] += 1
            _LOGGER.info("[EPLB diagnostic] stage=%s layer=%s reason=%s", self.stage, name, reason)
            return
        self.jobs[name] = job
        initial = job["initial_placement"]
        current = job["current_placement"]
        rank_work = [sum(sum(step[e] for e in rank) for step in loads) for rank in current]
        old_peak = sum(max(sum(step[e] for e in rank) for rank in initial) for step in job["loads"])
        new_peak = sum(max(sum(step[e] for e in rank) for rank in current) for step in job["loads"])
        _LOGGER.info(
            "[EPLB adjustment] stage=%s layer=%s generation=%s calls=%s reference=initial_live_placement "
            "rank_max_mean=%.3f initial_step_peak_work=%s current_step_peak_work=%s "
            "moved_experts=%s timing=requires_calibration",
            self.stage,
            name,
            job["generation"],
            calls,
            max(rank_work) * len(rows) / sum(rank_work),
            old_peak,
            new_peak,
            sum(len(set(new) - set(old)) for old, new in zip(initial, current)),
        )

    def _log_work(self, rows, name, loads):
        experts = len(loads[0])
        counts = [row["layers"][name]["owner_counts"] for row in rows]
        if any(
            len(rank) != len(rows) or any(len(owner) != experts or min(owner) < 0 for owner in rank) for rank in counts
        ):
            return "invalid_owner_counts"
        owned = [[sum(rank[owner][e] for rank in counts) for e in range(experts)] for owner in range(len(rows))]
        totals = [sum(step[e] for step in loads) for e in range(experts)]
        if any(sum(owner[e] for owner in owned) != totals[e] for e in range(experts)):
            return "owner_conservation_failure"
        self.logical_totals.setdefault(name, Counter()).update(dict(enumerate(totals)))
        cumulative = self.expert_totals.setdefault(name, [Counter() for _ in rows])
        current = placement(rows[0]["layers"][name]["current"], len(rows))
        for rank, work, history, hosted in zip(self.group.ranks, owned, cumulative, current):
            history.update(dict(enumerate(work)))
            visible = set(hosted) | {expert for expert, count in history.items() if count}
            _LOGGER.info(
                "[EPLB experts] stage=%s layer=%s rank=%s window_work=%s "
                "expert_work=%s cumulative_valid_work=%s cumulative_expert_work=%s",
                self.stage,
                name,
                rank,
                sum(work),
                {expert: work[expert] for expert in sorted(visible)},
                sum(history.values()),
                {expert: history[expert] for expert in sorted(visible)},
            )
        return None

    @torch.inference_mode()
    def finish(self):
        if self.finished:
            return
        self.monitor.active = False
        self.monitor.pending_real = False
        self.monitor.observing = False
        self.source_tokens.zero_()
        if self.window_calls:
            self._collect()
        self.costs = gather(self.group, self.monitor.drain(synchronize=True))
        self.finished = True
        if self.group.rank_in_group == 0:
            for rank, costs in zip(self.group.ranks, self.costs):
                _LOGGER.info(
                    "[EPLB migration] stage=%s rank=%s records=%s "
                    "scope=observed_submissions accounting=send_recv_describe_same_payload",
                    self.stage,
                    rank,
                    costs["transfers"],
                )
            _LOGGER.info(
                "[EPLB overhead] stage=%s observed_real_steps=%s rank_records=%s "
                "accounting=overlapping_components_do_not_sum",
                self.stage,
                self.observed_steps,
                self.costs,
            )
            _LOGGER.info(
                "[EPLB summary] stage=%s layers=%s comparable_layers=%s invalid_windows=%s "
                "estimated_net_saving_ms=unknown reason=calibration_required",
                self.stage,
                len(self.layers),
                len(self.jobs),
                dict(self.invalid_windows),
            )


def attach_monitor(runner, monitor):
    updater = getattr(runner, "eplb_updator", None)
    if updater is not None:
        updater._eplb_diagnostic_monitor = monitor
        updater.eplb_loader._eplb_diagnostic_monitor = monitor
    state = getattr(getattr(runner, "eplb", None), "state", None)
    if state is not None:
        state._eplb_diagnostic_monitor = monitor
        monitor.wrap_policy(state.policy)
        for model_state in state.model_states.values():
            model_state._eplb_diagnostic_monitor = monitor
            model_state.communicator._eplb_diagnostic_monitor = monitor
            model_state.communicator._eplb_diagnostic_layers = model_state.model.moe_layers


def diagnostics_quiescent(runner, *, calibration=False):
    """Collectively defer RPCs while model execution or weight updates are pending."""
    recorder = getattr(runner, "_eplb_diagnostics_recorder", None)
    if recorder is None:
        return False
    state = getattr(getattr(runner, "eplb", None), "state", None)
    busy = bool(
        getattr(runner, "_eplb_diagnostics_in_execution", False)
        or getattr(runner, "execute_model_state", None) is not None
        or (state is not None and recorder.monitor.pending_real)
    )
    if calibration:
        if state is not None:
            busy |= any(model.rebalanced or model.pending_result is not None for model in state.model_states.values())
        updater = getattr(runner, "eplb_updator", None)
        if updater is not None:
            busy |= (
                updater.eplb_loader.state.name != "WAITING"
                or updater.cur_iterations >= updater.expert_heat_collection_interval
            )
    peers = gather(recorder.group, busy)
    if any(peers):
        if recorder.group.rank_in_group == 0:
            _LOGGER.info(
                "[EPLB diagnostic] reason=execution_or_update_pending busy_ranks=%s retry_after_completed_cycle=True",
                [rank for rank, pending in zip(recorder.group.ranks, peers) if pending],
            )
        return False
    return True


@torch.inference_mode()
def initialize_diagnostics(runner):
    config = runner.ascend_config.eplb_diagnostics
    if config.mode == "off":
        return
    parallel = runner.vllm_config.parallel_config
    enabled = (
        parallel.enable_eplb
        if runner.vllm_config.use_v2_model_runner
        else runner.ascend_config.eplb_config.dynamic_eplb
    )
    if not parallel.enable_expert_parallel or not enabled:
        raise ValueError("EPLB benefit diagnostics requires EP and EPLB enabled")
    if getattr(parallel, "enable_elastic_ep", False):
        raise ValueError("EPLB benefit diagnostics requires a fixed EP group")
    vconfig = runner.vllm_config
    capacity = max(
        vconfig.scheduler_config.max_num_batched_tokens, max(vconfig.compilation_config.cudagraph_capture_sizes or [0])
    )
    source = torch.zeros((), dtype=torch.int64, device=runner.device)
    positions = torch.arange(capacity, device=runner.device)
    history_slot = torch.zeros(1, dtype=torch.int64, device=runner.device)
    for layer in runner.model.modules():
        probe = getattr(layer, "eplb_diagnostic_probe", None)
        if not isinstance(probe, ExpertLoadProbe):
            continue
        mapping = logical_to_physical(layer)
        if mapping is None or layer.moe_config.num_experts != layer.moe_config.num_logical_experts:
            raise ValueError("EPLB benefit diagnostics requires explicit mappings and zero redundant experts")
        if not layer._diagnostic_route_ownership_supported or layer.mix_placement:
            raise ValueError("EPLB benefit diagnostics does not support context parallel or mixed expert placement")
        probe.eplb_enabled = True
        probe.logical_to_physical = mapping
        ep_size = layer.moe_config.ep_size
        probe.owner_totals = torch.zeros((ep_size, probe.num_experts), dtype=torch.int64, device=runner.device)
        probe.source_token_count, probe.source_positions = source, positions
        probe.history = torch.zeros(
            (config.window_size, probe.num_experts + 2), dtype=torch.int64, device=runner.device
        )
        probe.comm_history = torch.zeros(config.window_size, dtype=torch.int64, device=runner.device)
        probe.history_slot = history_slot
    runner._eplb_diagnostics_source_tokens = source


def start_diagnostics(runner):
    if runner.ascend_config.eplb_diagnostics.mode == "off":
        return
    from vllm.distributed.parallel_state import get_pp_group, get_tp_group

    from vllm_ascend.distributed.parallel_state import get_mc2_group

    layers = [
        (getattr(layer, "layer_name", name), layer)
        for name, layer in runner.model.named_modules()
        if isinstance(getattr(layer, "eplb_diagnostic_probe", None), ExpertLoadProbe)
    ]
    if not layers:
        raise ValueError("No supported MoE layers for EPLB benefit diagnostics")
    recorder = DiagnosticsRecorder(
        runner.ascend_config.eplb_diagnostics,
        layers,
        get_mc2_group(),
        get_tp_group().ranks,
        get_pp_group().rank_in_group,
        runner._eplb_diagnostics_source_tokens,
    )
    runner._eplb_diagnostics_recorder = recorder
    node = local_node_id()
    nodes = gather(recorder.group, node)
    recorder.monitor.peer_locality = {
        rank: "unknown" if node is None or peer is None else "same_node" if node == peer else "cross_node"
        for rank, peer in zip(recorder.group.ranks, nodes)
    }
    attach_monitor(runner, recorder.monitor)
    _LOGGER.info("EPLB benefit diagnostics enabled; compare actual initial/current layouts; exclude dummy and padding")


def annotate_batch(runner, **metadata):
    recorder = getattr(runner, "_eplb_diagnostics_recorder", None)
    if recorder is not None and recorder.collecting and recorder.real and recorder.metadata:
        recorder.metadata[-1] = metadata


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
        except BaseException:
            recorder.monitor.pending_real = False
            raise
        finally:
            recorder.monitor.active = False
            runner._eplb_diagnostics_in_execution = False

    return wrapped


@record_diagnostics
def _record_dummy_batch(runner, scheduler_output, num_tokens, *, dummy_run):
    return runner._dummy_run(num_tokens, uniform_decode=True)


def run_dummy_batch(runner, num_tokens):
    return _record_dummy_batch(runner, SimpleNamespace(total_num_scheduled_tokens=0), num_tokens, dummy_run=True)
