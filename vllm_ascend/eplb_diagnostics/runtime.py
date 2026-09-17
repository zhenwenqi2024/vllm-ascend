# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local, bounded asynchronous snapshots outside eager/ACL graph execution."""

import functools
import json
import logging
import os
import socket
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm_ascend.eplb_diagnostics.probe import ExpertLoadProbe
from vllm_ascend.eplb_diagnostics.schema import decode_sample, fingerprint, local_logical_ids


def _layout_tensors(layer) -> tuple[dict, dict[str, torch.Tensor]]:
    v2 = layer._use_v2_model_runner
    state = getattr(getattr(layer, "router", None), "eplb_state", None)
    logical_map = getattr(state, "logical_to_physical_map", None) if v2 else None
    metadata = {
        "map_semantics": "physical_to_local" if v2 else "logical_to_local",
        "dynamic_eplb": bool(logical_map is not None) if v2 else layer.dynamic_eplb,
        "num_logical_experts": layer.moe_config.num_logical_experts,
        "experts_per_token": getattr(layer.moe_config, "experts_per_token", None),
        "num_shared_experts": layer.n_shared_experts,
        "mixed_shared_placement": layer.mix_placement,
        "ep_size": layer.moe_config.ep_size,
        "ep_rank": layer.moe_config.ep_rank,
    }
    tensors = {}
    if layer.ascend_expert_map is not None:
        tensors["global_to_local"] = layer.ascend_expert_map
    if logical_map is not None:
        # Upstream reserves 1024 replica slots per logical expert. Only
        # 1 + total redundant experts can be populated for any one expert.
        # Use the host configuration bound; reading the device max would sync.
        physical_experts = getattr(layer.moe_config, "num_experts", None)
        if physical_experts is not None and physical_experts >= layer.moe_config.num_logical_experts:
            max_replicas = physical_experts - layer.moe_config.num_logical_experts + 1
            metadata["logical_to_physical_storage_shape"] = list(logical_map.shape)
            logical_map = logical_map[:, :max_replicas]
        tensors["logical_to_physical"] = logical_map
    replica_count = getattr(state, "logical_replica_count", None)
    if v2 and replica_count is not None:
        tensors["logical_replica_count"] = replica_count
    replica_table = getattr(state, "expert_replica_routing_table", None)
    if v2 and replica_table is not None:
        # This expanded lookup is derived from the compact map/count and EP rank.
        # Copying it for every layer can exceed the entire snapshot budget.
        metadata["replica_routing_table_shape"] = list(replica_table.shape)
        metadata["replica_routing_table_snapshot"] = "omitted_derived_lookup"
    if not v2 and layer.log2phy is not None:
        tensors["selected_physical_map"] = layer.log2phy
    return metadata, tensors


class NpuSnapshotBackend:
    """D2H is enqueued on the inference stream; only the writer waits for its event."""

    def __init__(self, device):
        self.device = device

    def copy(self, source: torch.Tensor) -> torch.Tensor:
        if source.device.type == "cpu":
            return source.clone()
        target = torch.empty(source.shape, dtype=source.dtype, device="cpu", pin_memory=True)
        target.copy_(source, non_blocking=True)
        return target

    def event(self, timing: bool = False):
        event = torch.npu.Event(enable_timing=timing)
        event.record()
        return event

    def finish(self, ready, start, end) -> float:
        # This method only runs on the writer thread. This is event-specific, not a
        # device-wide synchronize, and adds no barrier between inference ranks.
        with torch.npu.device(self.device):
            ready.synchronize()
            return start.elapsed_time(end)


class DiagnosticsRecorder:
    def __init__(self, config, layers, metadata: dict, backend):
        self.config = config
        self.layers = layers
        self.metadata = metadata
        self.backend = backend
        self.step = 0
        self.model_step = 0
        self.dropped = 0
        self.collected = 0
        self.drop_reasons: dict[str, int] = {}
        self.pending: list[Future] = []
        self.active: dict | None = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eplb-diagnostics")
        directory = Path(config.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        # An exclusive per-process file prevents duplicate ranks/restarts overwriting logs.
        name = f"{config.run_id}-rank{metadata['rank']}-{uuid.uuid4().hex[:12]}.jsonl"
        self.path = directory / name
        self.path.touch(exist_ok=False)

    def _snapshots(self, descriptions):
        return [
            {
                "layer": name,
                "num_experts": layer.eplb_diagnostic_probe.num_experts,
                "counts": self.backend.copy(layer.eplb_diagnostic_probe.totals),
                "routes": self.backend.copy(layer.eplb_diagnostic_probe.route_totals),
                "layout_metadata": metadata.copy(),
                "layout_tensors": {key: self.backend.copy(value) for key, value in tensors.items()},
            }
            for (name, layer), (metadata, tensors) in zip(self.layers, descriptions)
        ]

    def begin(self, batch: dict) -> bool:
        self.step += 1
        # Propagate writer failures at the next boundary, where the wrapper can
        # disable collection and report the error without masking inference errors.
        running = []
        for future in self.pending:
            if future.done():
                future.result()
            else:
                running.append(future)
        self.pending = running
        # Empty scheduler boundaries do not execute the model. Idle DP dummy
        # work does execute collectives and must advance the expected ordinal.
        if batch.get("scheduled_tokens", 1) == 0 and not batch.get("dummy_run"):
            return False
        self.model_step += 1
        # Dummy collectives still advance the device call counters. Track only
        # that ordinal on the host: no snapshots, timing, rows, or sample budget.
        if batch.get("dummy_run") or batch.get("phase") == "dummy":
            return False
        if self.collected >= self.config.max_samples:
            return False
        if self.model_step <= self.config.warmup_steps:
            return False
        if (self.model_step - self.config.warmup_steps - 1) % self.config.sample_interval >= self.config.burst_size:
            return False
        if len(self.pending) >= self.config.max_pending:
            self._drop("writer_backpressure")
            return False
        descriptions = [_layout_tensors(layer) for _, layer in self.layers]
        size = sum(
            (layer.eplb_diagnostic_probe.totals.numel() + layer.eplb_diagnostic_probe.route_totals.numel()) * 8
            + sum(t.numel() * t.element_size() for t in tensors.values())
            for (_, layer), (_, tensors) in zip(self.layers, descriptions)
        )
        if 2 * size > self.config.max_snapshot_mb * 1024 * 1024:
            self._drop("snapshot_memory_budget")
            return False
        before = self._snapshots(descriptions)
        self.collected += 1
        self.active = {
            "step": self.step,
            "model_step": self.model_step,
            "batch": batch,
            "before": before,
            "start": self.backend.event(timing=True),
            "dropped_samples": self.dropped,
            "drop_reasons": self.drop_reasons.copy(),
        }
        return True

    def _drop(self, reason: str) -> None:
        self.dropped += 1
        self.drop_reasons[reason] = self.drop_reasons.get(reason, 0) + 1
        if self.drop_reasons[reason] == 1:
            logging.getLogger(__name__).warning("EPLB diagnostic samples dropped: %s", reason)

    def annotate(self, **metadata) -> None:
        if self.active is not None:
            self.active["batch"].update(metadata)

    def end(self) -> None:
        sample = self.active
        if sample is not None and (sample["batch"].get("dummy_run") or sample["batch"].get("phase") == "dummy"):
            # Defensive fallback if the runner identifies synthetic work late.
            # Drain an already queued before-copy, but never export this sample.
            self.collected -= 1
            self.abort()
            return
        self.active = None
        if sample is None:
            return
        sample["end"] = self.backend.event(timing=True)
        sample["after"] = self._snapshots([_layout_tensors(layer) for _, layer in self.layers])
        sample["ready"] = self.backend.event()
        self.pending.append(self.executor.submit(self._write, sample))

    def abort(self) -> None:
        # Inference errors still propagate. Drain outstanding copies in a background
        # task before releasing their pinned destinations.
        sample = self.active
        self.active = None
        if sample is not None:
            end = self.backend.event(timing=True)
            ready = self.backend.event()
            self.pending.append(self.executor.submit(self._discard, sample, ready, end))

    def _discard(self, sample, ready, end) -> None:
        self.backend.finish(ready, sample["start"], end)

    @staticmethod
    def _decode_layout(snapshot: dict) -> dict:
        layout = snapshot["layout_metadata"].copy()
        layout.update({key: value.tolist() for key, value in snapshot["layout_tensors"].items()})
        return layout

    def _write(self, sample) -> None:
        elapsed_ms = self.backend.finish(sample["ready"], sample["start"], sample["end"])
        record = {
            "schema_version": 1,
            "run_id": self.config.run_id,
            **self.metadata,
            "local_step": sample["step"],
            "expected_model_calls": sample["model_step"],
            "batch": sample["batch"],
            "execute_stream_ms": elapsed_ms,
            "timing_scope": "execute_model_current_stream_including_instrumentation",
            "dropped_samples": sample["dropped_samples"],
            "drop_reasons": sample["drop_reasons"],
            "layers": [],
        }
        for before, after in zip(sample["before"], sample["after"]):
            info = decode_sample(before["counts"].tolist(), after["counts"].tolist(), before["num_experts"])
            routes = (after["routes"] - before["routes"]).tolist()
            route_valid = (
                len(routes) > 2
                and routes[-2] == info["calls"] == 1
                and routes[-1] == 0
                and all(value >= 0 for value in routes)
            )
            info["routing"] = {
                "semantics": "valid_scheduled_mc2_source_destinations" if route_valid else "unavailable",
                "physical_assignments": routes[:-2] if route_valid else None,
                "recorded_calls": routes[-2],
                "invalid_ids": routes[-1],
            }
            info["alignment_eligible"] = (
                self.metadata.get("alignment_contract") == "post_warmup_executed_collective_v2"
                and info["call_start"] == sample["model_step"] - 1
                and info["call_end"] == sample["model_step"]
            )
            layout_start = self._decode_layout(before)
            layout_end = self._decode_layout(after)
            info.update(
                layer=before["layer"],
                layout_start=layout_start,
                layout_end=layout_end,
                layout_changed=layout_start != layout_end,
                layout_fingerprint=fingerprint(layout_end),
                local_logical_ids=local_logical_ids(layout_end, before["num_experts"]),
                moe_device_ms=None,
                moe_timing_missing_reason="requires_graph_aware_device_profiler",
                comm_methods=list(ExpertLoadProbe.COMM_METHODS),
            )
            record["layers"].append(info)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")

    def close(self) -> None:
        """Drain after inference stops; never call this from the inference hot path."""
        self.executor.shutdown(wait=True)
        for future in self.pending:
            future.result()


def _create_recorder(runner):
    # Lazy imports isolate the portable probe/schema tests from vLLM initialization.
    from vllm.distributed.parallel_state import (
        get_dcp_group,
        get_dp_group,
        get_ep_group,
        get_etp_group,
        get_pcp_group,
        get_pp_group,
        get_tp_group,
        get_world_group,
    )

    from vllm_ascend.distributed.parallel_state import get_mc2_group

    layers = [
        (getattr(layer, "layer_name", name), layer)
        for name, layer in runner.model.named_modules()
        if isinstance(getattr(layer, "eplb_diagnostic_probe", None), ExpertLoadProbe)
    ]
    if not layers:
        raise ValueError("No AscendRoutedExperts diagnostic probes found in the target model.")
    groups = {}
    for name, getter in (
        ("world", get_world_group),
        ("ep", get_ep_group),
        ("tp", get_tp_group),
        ("etp", get_etp_group),
        ("dp", get_dp_group),
        ("pp", get_pp_group),
        ("pcp", get_pcp_group),
        ("dcp", get_dcp_group),
        ("mc2", get_mc2_group),
    ):
        try:
            group = getter()
        except AssertionError:
            continue
        groups[name] = {"ranks": list(group.ranks), "rank_in_group": group.rank_in_group}
    parallel = runner.vllm_config.parallel_config
    metadata = {
        "rank": get_world_group().rank,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "groups": groups,
        "runner": type(runner).__module__,
        "configured_graph_mode": str(runner.vllm_config.compilation_config.cudagraph_mode),
        "enforce_eager": runner.vllm_config.model_config.enforce_eager,
        "eplb_enabled": bool(parallel.enable_eplb or runner.ascend_config.eplb_config.dynamic_eplb),
        "static_placement": runner.ascend_config.eplb_config.expert_map_path is not None,
        "data_parallel_size": parallel.data_parallel_size,
        "speculative_decoding": runner.vllm_config.speculative_config is not None,
        "alignment": "rank_local_call_spans_require_collective_order_validation",
        "counter_origin": "after_worker_warmup",
        "alignment_contract": "post_warmup_executed_collective_v2",
    }
    return DiagnosticsRecorder(
        runner.ascend_config.eplb_diagnostics, layers, metadata, NpuSnapshotBackend(runner.device)
    )


def annotate_batch(runner, **metadata) -> None:
    recorder = getattr(runner, "_eplb_diagnostics_recorder", None)
    if recorder is not None:
        recorder.annotate(**metadata)


@torch.inference_mode()
def initialize_diagnostics(runner) -> None:
    """Bind shared source validity before capture; never modify inference masks."""
    if runner.ascend_config.eplb_diagnostics.mode == "off":
        return
    config = runner.vllm_config
    capacity = max(
        config.scheduler_config.max_num_batched_tokens, max(config.compilation_config.cudagraph_capture_sizes or [0])
    )
    source_tokens = torch.full((), capacity, dtype=torch.int64, device=runner.device)
    positions = torch.arange(capacity, device=runner.device)
    for layer in runner.model.modules():
        probe = getattr(layer, "eplb_diagnostic_probe", None)
        if isinstance(probe, ExpertLoadProbe):
            probe.source_token_count = source_tokens
            probe.source_positions = positions
    runner._eplb_diagnostics_source_tokens = source_tokens


@torch.inference_mode()
def start_diagnostics(runner) -> None:
    """Arm after warmup, so capture/profile work cannot exhaust the sample budget."""
    if runner.ascend_config.eplb_diagnostics.mode == "off":
        return
    if getattr(runner, "_eplb_diagnostics_recorder", None) is not None:
        return
    try:
        recorder = _create_recorder(runner)
        runner._eplb_diagnostics_recorder = recorder
        for _, layer in recorder.layers:
            layer.eplb_diagnostic_probe.totals.zero_()
            layer.eplb_diagnostic_probe.route_totals.zero_()
        runner._eplb_diagnostics_ready = True
        logging.getLogger(__name__).info("EPLB diagnostics output: %s", recorder.path)
    except Exception:
        runner._eplb_diagnostics_failed = True
        logging.getLogger(__name__).exception("EPLB diagnostics initialization disabled")


def _abort_safely(recorder) -> None:
    if recorder is not None:
        try:
            recorder.abort()
        except Exception:
            logging.getLogger(__name__).exception("Failed to discard an EPLB diagnostic snapshot")


def record_diagnostics(func):
    """Wrap both runners; graph replay itself remains untouched."""

    @functools.wraps(func)
    def wrapped(runner, scheduler_output, *args, **kwargs):
        if runner.ascend_config.eplb_diagnostics.mode == "off" or getattr(
            runner, "_eplb_diagnostics_in_execution", False
        ):
            return func(runner, scheduler_output, *args, **kwargs)
        # MRv2 worker dummy execution re-enters the decorated execute_model.
        # Guard every boundary, including warmup/skipped/dropped samples: active
        # snapshots alone cannot prevent double-counting unsampled boundaries.
        runner._eplb_diagnostics_in_execution = True
        try:
            return execute_with_diagnostics(runner, scheduler_output, *args, **kwargs)
        finally:
            runner._eplb_diagnostics_in_execution = False

    def execute_with_diagnostics(runner, scheduler_output, *args, **kwargs):
        config = runner.ascend_config.eplb_diagnostics
        if config.mode == "off" or getattr(runner, "_eplb_diagnostics_failed", False):
            return func(runner, scheduler_output, *args, **kwargs)
        if not getattr(runner, "_eplb_diagnostics_ready", False):
            return func(runner, scheduler_output, *args, **kwargs)
        if kwargs.get("is_profile", False) or torch.npu.is_current_stream_capturing():
            return func(runner, scheduler_output, *args, **kwargs)
        recorder = getattr(runner, "_eplb_diagnostics_recorder", None)
        if recorder is not None and recorder.active is not None:
            return func(runner, scheduler_output, *args, **kwargs)
        try:
            if recorder is None:
                recorder = _create_recorder(runner)
                runner._eplb_diagnostics_recorder = recorder
                logging.getLogger(__name__).info("EPLB diagnostics output: %s", recorder.path)
            dummy_run = kwargs.get("dummy_run", False)
            source_tokens = getattr(runner, "_eplb_diagnostics_source_tokens", None)
            if source_tokens is not None:
                # Worker dummy entry is outside the runner's inference-mode decorator.
                with torch.inference_mode():
                    source_tokens.fill_(0 if dummy_run else scheduler_output.total_num_scheduled_tokens)
            recorder.begin(
                {
                    "scheduled_tokens": 0 if dummy_run else scheduler_output.total_num_scheduled_tokens,
                    "scheduled_requests": 0 if dummy_run else len(scheduler_output.num_scheduled_tokens),
                    "dummy_tokens": scheduler_output.total_num_scheduled_tokens if dummy_run else 0,
                    "dummy_run": dummy_run,
                    "phase": "dummy" if dummy_run else "unknown",
                    "graph_mode": "unknown",
                }
            )
        except Exception:
            runner._eplb_diagnostics_failed = True
            logging.getLogger(__name__).exception("EPLB diagnostics collection disabled")
            # If collection failed after enqueuing a snapshot, keep its buffers alive.
            _abort_safely(recorder)
        try:
            result = func(runner, scheduler_output, *args, **kwargs)
        except BaseException:
            _abort_safely(recorder)
            raise
        if recorder is not None and not getattr(runner, "_eplb_diagnostics_failed", False):
            try:
                recorder.end()
            except Exception:
                runner._eplb_diagnostics_failed = True
                logging.getLogger(__name__).exception("EPLB diagnostics collection disabled")
        return result

    return wrapped


@record_diagnostics
def _record_dummy_batch(runner, scheduler_output, num_tokens, *, dummy_run):
    # DP dispatch may pad this synthetic input to another rank's batch size.
    # Its actual padded size is not exposed here; do not label the input as it.
    annotate_batch(runner, dummy_run=True, phase="dummy", dummy_tokens=num_tokens)
    return runner._dummy_run(num_tokens, uniform_decode=True)


def run_dummy_batch(runner, num_tokens):
    """Advance collective bookkeeping for idle DP without collecting a sample."""
    if runner.ascend_config.eplb_diagnostics.mode == "off":
        return runner._dummy_run(num_tokens, uniform_decode=True)
    metadata = SimpleNamespace(total_num_scheduled_tokens=0, num_scheduled_tokens={})
    return _record_dummy_batch(runner, metadata, num_tokens, dummy_run=True)
