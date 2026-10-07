# SPDX-License-Identifier: Apache-2.0
"""Bounded hand-off of DSpark prompt context from a reader to its model worker.

This does not perform network reads or target forward recomputation. Readers
submit lossless, ordered auxiliary tensors; only the owning model thread may
write the draft KV and join that completion with target KV readiness.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import torch

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.spec_decode.dspark_utils import get_dspark_aux_layer_ids

if TYPE_CHECKING:
    from vllm.config import VllmConfig

MAX_DSPARK_CONTEXT_CHUNK_TOKENS = 64
BF16_BYTES = 2


def get_pd_dspark_aux_layer_ids(vllm_config: VllmConfig) -> tuple[int, ...]:
    """Resolve and validate opt-in P-side context capture before model loading.

    Keep transport/schema constraints in this backend. Unconfigured runners
    return immediately without applying DSpark-specific restrictions.
    """
    transfer = getattr(vllm_config, "kv_transfer_config", None)
    extra = getattr(transfer, "kv_connector_extra_config", None) or {}
    layer_ids = extra.get("dspark_aux_hidden_state_layer_ids")
    if layer_ids is None:
        return ()
    if transfer.is_kv_consumer or not transfer.is_kv_producer or vllm_config.speculative_config is not None:
        raise ValueError("DSpark auxiliary capture requires a P-only producer without speculative decoding.")
    parallel = vllm_config.parallel_config
    if parallel.prefill_context_parallel_size * parallel.decode_context_parallel_size != 1:
        raise ValueError("P-side DSpark auxiliary capture does not support context parallelism.")
    model_config = vllm_config.model_config
    if not model_config.enforce_eager:
        raise ValueError("P-side DSpark auxiliary capture currently requires eager prefill.")
    if vllm_config.cache_config.enable_prefix_caching:
        raise ValueError(
            "P-side DSpark auxiliary capture requires prefix caching disabled until auxiliary caching exists."
        )
    return get_dspark_aux_layer_ids(vllm_config)


def resident_mla_context_group_ids(groups: Sequence[Any]) -> tuple[int, ...]:
    """Resolve residency before scheduler conversion drops per-layer specs.

    A uniform scheduler block group may contain host target and resident draft
    tensors. Sharing block IDs does not mean sharing physical KV storage.
    """
    result = []
    for group_id, group in enumerate(groups):
        spec = group.kv_cache_spec
        layer_specs = getattr(spec, "kv_cache_specs", None)
        specs = layer_specs.values() if layer_specs is not None else (spec,)
        if any(isinstance(item, AscendMLAAttentionSpec) and not item.store_on_host for item in specs):
            result.append(group_id)
    return tuple(result)


def find_dspark_context_connector(connector: Any, metadata: Any) -> tuple[Any, Any]:
    """Select the unique PD child and its metadata without patching MultiConnector.

    Upstream exposes child connectors but does not forward model-specific
    methods. Keep metadata paired with its child, including nested wrappers.
    """
    pending = [(connector, metadata)]
    matches = []
    while pending:
        current, current_meta = pending.pop()
        children = getattr(current, "_connectors", ())
        if children:
            child_metadata = getattr(current_meta, "metadata", ())
            if len(children) != len(child_metadata):
                raise RuntimeError("DSpark MultiConnector children and metadata are not aligned")
            pending.extend(zip(children, child_metadata))
        elif callable(getattr(current, "get_dspark_context_descriptor", None)) and callable(
            getattr(current, "send_dspark_context_chunk", None)
        ):
            matches.append((current, current_meta))
    if len(matches) != 1:
        raise RuntimeError("DSpark prefill requires exactly one PD context connector")
    return matches[0]


@dataclass(frozen=True)
class DSparkContextDescriptor:
    request_id: str
    generation: str
    prompt_tokens: int
    aux_layer_ids: tuple[int, ...]
    hidden_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("DSpark context requires a nonempty external request ID")
        if not isinstance(self.generation, str) or not self.generation:
            raise ValueError("DSpark context requires request identity and allocation generation")
        if type(self.prompt_tokens) is not int or self.prompt_tokens <= 0:
            raise ValueError("DSpark prompt_tokens must be a positive integer")
        if type(self.hidden_size) is not int or self.hidden_size <= 0:
            raise ValueError("DSpark hidden_size must be a positive integer")
        if (
            not isinstance(self.aux_layer_ids, tuple)
            or not self.aux_layer_ids
            or any(type(layer) is not int or layer < 0 for layer in self.aux_layer_ids)
            or tuple(sorted(set(self.aux_layer_ids))) != self.aux_layer_ids
        ):
            raise ValueError("DSpark auxiliary boundary IDs must be a nonempty ordered unique tuple")

    @property
    def feature_width(self) -> int:
        return self.hidden_size * len(self.aux_layer_ids)


@dataclass(frozen=True)
class DSparkContextChunk:
    descriptor: DSparkContextDescriptor
    token_offset: int
    num_tokens: int


class DSparkContextSubmission(Enum):
    ACCEPTED = auto()
    BACKPRESSURE = auto()
    STALE = auto()


def build_dspark_context_inputs(
    chunk: DSparkContextChunk,
    block_ids_by_group: Mapping[int, Sequence[int]],
    layer_group_ids: Sequence[int],
    *,
    block_size: int,
    num_blocks: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Build absolute positions and real resident-draft slots, never target slots.

    ``layer_group_ids`` are the actual global KV group IDs discovered by the
    draft loader. Copy only the block-table slice touched by this chunk, and
    reuse its mapping for draft layers sharing one physical cache group.
    """
    if type(block_size) is not int or block_size <= 0 or type(num_blocks) is not int or num_blocks <= 0:
        raise ValueError("DSpark context requires positive integer cache bounds")
    if block_size * num_blocks - 1 > torch.iinfo(torch.int32).max:
        raise ValueError("DSpark context slot IDs exceed the NPU int32 mapping range")
    if not layer_group_ids or any(type(gid) is not int or gid < 0 for gid in layer_group_ids):
        raise ValueError("DSpark context requires loader-discovered draft cache group IDs")
    start = chunk.token_offset
    end = start + chunk.num_tokens
    if (
        type(start) is not int
        or type(chunk.num_tokens) is not int
        or not 0 <= start < end <= chunk.descriptor.prompt_tokens
    ):
        raise ValueError("DSpark context chunk is outside the allocated prompt")
    required_blocks = (chunk.descriptor.prompt_tokens + block_size - 1) // block_size
    first_block, last_block = start // block_size, (end - 1) // block_size + 1
    selected: dict[int, Sequence[int]] = {}
    for gid in set(layer_group_ids):
        ids = block_ids_by_group.get(gid)
        if ids is None or len(ids) < required_blocks:
            raise ValueError(f"DSpark draft group {gid} does not cover the full prompt")
        prompt_ids = ids[:required_blocks]
        if any(type(block) is not int or not 0 <= block < num_blocks for block in prompt_ids):
            raise ValueError(f"DSpark draft group {gid} contains an invalid physical block")
        if len(set(prompt_ids)) != len(prompt_ids):
            raise ValueError(f"DSpark draft group {gid} aliases distinct prompt blocks")
        selected[gid] = prompt_ids[first_block:last_block]
    positions = torch.arange(start, end, dtype=torch.int64, device=device)
    logical_blocks = positions // block_size - first_block
    mappings = {
        gid: (
            torch.tensor(ids, dtype=torch.int64, device=device)[logical_blocks] * block_size + positions % block_size
        ).to(torch.int32)
        for gid, ids in selected.items()
    }
    return positions, [mappings[gid] for gid in layer_group_ids]


@dataclass
class _ContextProgress:
    descriptor: DSparkContextDescriptor
    received_tokens: int = 0
    initialized_tokens: int = 0
    target_kv_done: bool = False
    in_flight: bool = False
    failed: bool = False


@dataclass
class _QueuedContext:
    chunk: DSparkContextChunk
    tensor: torch.Tensor
    release: Callable[[], None]

    @property
    def size_bytes(self) -> int:
        return self.tensor.numel() * self.tensor.element_size()


class DSparkContextReceiver:
    """Queue registered staging views until synchronous main-thread KV writes.

    A successful submit transfers ownership of the tensor and release callback.
    STALE/BACKPRESSURE leave ownership with the reader. An initialization failure
    quarantines accepted buffers until explicit discard after device writes have
    been synchronized; it never reports readiness or silently recomputes prompt.
    """

    def __init__(self, *, max_pending_bytes: int, max_requests: int) -> None:
        if type(max_pending_bytes) is not int or max_pending_bytes <= 0:
            raise ValueError("DSpark staging byte budget must be a positive integer")
        if type(max_requests) is not int or max_requests <= 0:
            raise ValueError("DSpark request capacity must be a positive integer")
        self.max_pending_bytes = max_pending_bytes
        self.max_requests = max_requests
        self._owner_thread = threading.get_ident()
        self._lock = threading.Lock()
        self._contexts: dict[str, _ContextProgress] = {}
        self._queue: deque[_QueuedContext] = deque()
        self._pending_bytes = 0

    def _require_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("DSpark context lifecycle and draft KV writes require the model worker thread")

    @property
    def pending_bytes(self) -> int:
        with self._lock:
            return self._pending_bytes

    def register_request(self, descriptor: DSparkContextDescriptor) -> None:
        self._require_owner()
        with self._lock:
            if descriptor.request_id in self._contexts:
                raise ValueError("Discard the previous DSpark allocation before registering a reused request ID")
            if len(self._contexts) >= self.max_requests:
                raise RuntimeError("DSpark context request capacity exhausted")
            self._contexts[descriptor.request_id] = _ContextProgress(descriptor)

    def get_descriptor(self, request_id: str) -> DSparkContextDescriptor | None:
        # The network reader may inspect request admission, but it never owns
        # lifecycle transitions or draft-model execution.
        with self._lock:
            state = self._contexts.get(request_id)
            return state.descriptor if state is not None else None

    def can_accept(self, chunk: DSparkContextChunk) -> DSparkContextSubmission:
        """Check reader admission before allocating/reading a pinned chunk.

        The reader is the sole submitter for one D TP rank. The model thread can
        only drain entries (which frees capacity), so a positive result cannot
        be invalidated by a competing submit before ``submit``.
        """
        with self._lock:
            state = self._contexts.get(chunk.descriptor.request_id)
            if state is None:
                # D may not have reached start_load_kv yet. Ask P to retry.
                return DSparkContextSubmission.BACKPRESSURE
            if state.descriptor.generation != chunk.descriptor.generation:
                return DSparkContextSubmission.STALE
            if state.failed:
                raise RuntimeError("DSpark context allocation is quarantined after an initialization failure")
            if state.descriptor != chunk.descriptor:
                raise ValueError("DSpark context schema/prompt length changed within one allocation")
            if type(chunk.token_offset) is not int or chunk.token_offset != state.received_tokens:
                raise ValueError("DSpark context chunks must cover the prompt in order without gaps or duplicates")
            if (
                type(chunk.num_tokens) is not int
                or chunk.num_tokens <= 0
                or chunk.token_offset + chunk.num_tokens > state.descriptor.prompt_tokens
            ):
                raise ValueError("DSpark context chunk is outside the allocated prompt")
            size_bytes = chunk.num_tokens * state.descriptor.feature_width * BF16_BYTES
            if size_bytes > self.max_pending_bytes:
                raise ValueError("DSpark context chunk exceeds the whole staging budget; split it before reading")
            if self._pending_bytes + size_bytes > self.max_pending_bytes:
                return DSparkContextSubmission.BACKPRESSURE
            return DSparkContextSubmission.ACCEPTED

    def submit(
        self, chunk: DSparkContextChunk, tensor: torch.Tensor, release: Callable[[], None]
    ) -> DSparkContextSubmission:
        """Called by the reader only after its MemFabric read has completed."""
        admission = self.can_accept(chunk)
        if admission is not DSparkContextSubmission.ACCEPTED:
            return admission
        with self._lock:
            state = self._contexts.get(chunk.descriptor.request_id)
            if state is None or state.descriptor.generation != chunk.descriptor.generation:
                return DSparkContextSubmission.STALE
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.bfloat16
                or tuple(tensor.shape)
                != (
                    chunk.num_tokens,
                    state.descriptor.feature_width,
                )
            ):
                raise ValueError("DSpark context tensor must contain all ordered BF16 auxiliary features")
            if not tensor.is_contiguous() or not callable(release):
                raise ValueError("DSpark context requires contiguous staging storage and its release callback")
            item = _QueuedContext(chunk, tensor, release)
            if item.size_bytes > self.max_pending_bytes:
                raise ValueError("DSpark context chunk exceeds the whole staging budget; split it before reading")
            if self._pending_bytes + item.size_bytes > self.max_pending_bytes:
                return DSparkContextSubmission.BACKPRESSURE
            self._queue.append(item)
            self._pending_bytes += item.size_bytes
            state.received_tokens += chunk.num_tokens
            return DSparkContextSubmission.ACCEPTED

    def drain(self, initialize: Callable[[DSparkContextChunk, torch.Tensor], None]) -> int:
        """Project context and write real draft slots, waiting for device completion.

        The callback owns the model-specific projection, RoPE, block-group slot
        mapping and device event wait. Returning before the writes finish is not
        permitted: these buffers may be recycled immediately after return.
        """
        self._require_owner()
        initialized = 0
        while True:
            with self._lock:
                if not self._queue:
                    return initialized
                item = self._queue[0]
                state = self._contexts[item.chunk.descriptor.request_id]
                if state.failed:
                    raise RuntimeError("DSpark context remains quarantined after an initialization failure")
                if state.in_flight:
                    raise RuntimeError("Reentrant DSpark context drain is not permitted")
                state.in_flight = True
            try:
                result = initialize(item.chunk, item.tensor)
                if result is not None:
                    raise RuntimeError("DSpark initializer must wait synchronously for draft KV writes")
            except BaseException:
                with self._lock:
                    state.in_flight = False
                    state.failed = True
                raise
            with self._lock:
                state.in_flight = False
                state.initialized_tokens += item.chunk.num_tokens
                self._queue.popleft()
                self._pending_bytes -= item.size_bytes
            try:
                item.release()
            except BaseException:
                with self._lock:
                    state.failed = True
                raise
            initialized += item.chunk.num_tokens

    def mark_target_kv_done(self, request_id: str, generation: str) -> None:
        self._require_owner()
        with self._lock:
            state = self._contexts.get(request_id)
            if state is not None and state.descriptor.generation == generation:
                state.target_kv_done = True

    def ready_requests(self) -> set[str]:
        """Local readiness only; the connector must still join all D TP ranks."""
        self._require_owner()
        with self._lock:
            return {
                req_id
                for req_id, state in self._contexts.items()
                if state.target_kv_done
                and not state.failed
                and not state.in_flight
                and state.initialized_tokens == state.descriptor.prompt_tokens
            }

    def retire_ready_request(self, request_id: str, generation: str) -> None:
        """Release ingress bookkeeping after every TP rank joined readiness.

        Resident draft KV and its block tables are owned by the runner, not
        this receiver. Keeping completed ingress allocations until decoding
        finishes can overlap the next scheduler admission: start_load_kv runs
        before finished-request cleanup in the same connector-only step.
        """
        self._require_owner()
        with self._lock:
            state = self._contexts.get(request_id)
            if state is None or state.descriptor.generation != generation:
                return
            if (
                not state.target_kv_done
                or state.failed
                or state.in_flight
                or state.initialized_tokens != state.descriptor.prompt_tokens
            ):
                raise RuntimeError("Cannot retire DSpark ingress before synchronized context and target KV readiness")
            del self._contexts[request_id]

    def discard_request(self, request_id: str, generation: str) -> None:
        """Cancel/retire an allocation after any device writes are synchronized."""
        self._require_owner()
        discarded: list[_QueuedContext] = []
        with self._lock:
            state = self._contexts.get(request_id)
            if state is None or state.descriptor.generation != generation:
                return
            if state.in_flight:
                raise RuntimeError("Cannot recycle DSpark blocks while a context initializer is running")
            kept: deque[_QueuedContext] = deque()
            for item in self._queue:
                if item.chunk.descriptor == state.descriptor:
                    discarded.append(item)
                    self._pending_bytes -= item.size_bytes
                else:
                    kept.append(item)
            self._queue = kept
            del self._contexts[request_id]
        for item in discarded:
            item.release()


def build_draft_context_slot_mappings(
    chunk: DSparkContextChunk,
    *,
    draft_group_ids: tuple[int, ...],
    draft_block_ids_by_group: dict[int, tuple[int, ...]],
    block_sizes_by_group: dict[int, int],
    layer_group_ids: tuple[int, ...],
    device: torch.device,
) -> list[torch.Tensor]:
    """Map absolute prompt rows into the loader-discovered DSpark KV groups.

    ``layer_group_ids`` is in the loaded DSpark model's attention-layer order;
    callers must derive it from the actual speculative model and KV cache
    groups. No target, indexer, or inferred layer-index block IDs are accepted.
    """
    if not draft_group_ids or len(set(draft_group_ids)) != len(draft_group_ids):
        raise ValueError("DSpark context needs unique loaded draft KV cache-group IDs")
    if not layer_group_ids or any(group_id not in draft_group_ids for group_id in layer_group_ids):
        raise ValueError("Every DSpark attention layer must map to an actual resident draft cache group")
    if set(draft_group_ids) != set(draft_block_ids_by_group) or set(draft_group_ids) != set(block_sizes_by_group):
        raise ValueError("DSpark draft block tables must cover every and only resident draft cache group")

    positions = range(chunk.token_offset, chunk.token_offset + chunk.num_tokens)
    per_group: dict[int, torch.Tensor] = {}
    for group_id in draft_group_ids:
        block_ids = draft_block_ids_by_group[group_id]
        block_size = block_sizes_by_group[group_id]
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("DSpark draft KV block sizes must be positive integers")
        if len(block_ids) * block_size < chunk.descriptor.prompt_tokens:
            raise ValueError("DSpark draft block table does not cover the full prompt")
        if any(type(block_id) is not int or block_id < 0 for block_id in block_ids):
            raise ValueError("DSpark draft block table contains an invalid block ID")
        slots = [block_ids[position // block_size] * block_size + position % block_size for position in positions]
        per_group[group_id] = torch.tensor(slots, dtype=torch.int32, device=device)
    return [per_group[group_id] for group_id in layer_group_ids]


def initialize_draft_context_chunk(
    model: torch.nn.Module,
    chunk: DSparkContextChunk,
    aux_features: torch.Tensor,
    *,
    draft_group_ids: tuple[int, ...],
    draft_block_ids_by_group: dict[int, tuple[int, ...]],
    block_sizes_by_group: dict[int, int],
    layer_group_ids: tuple[int, ...],
    device: torch.device,
) -> None:
    """Project one received chunk and wait for its own MLA KV writes to finish."""
    if aux_features.device != device:
        raise ValueError("DSpark auxiliary staging tensor must be placed on the draft runner device")
    if aux_features.dtype != torch.bfloat16:
        raise ValueError("GLM MLA DSpark auxiliary context must retain BF16 checkpoint inputs")
    context_states = model.combine_hidden_states(aux_features)
    context_positions = torch.arange(
        chunk.token_offset,
        chunk.token_offset + chunk.num_tokens,
        dtype=torch.long,
        device=device,
    )
    slots = build_draft_context_slot_mappings(
        chunk,
        draft_group_ids=draft_group_ids,
        draft_block_ids_by_group=draft_block_ids_by_group,
        block_sizes_by_group=block_sizes_by_group,
        layer_group_ids=layer_group_ids,
        device=device,
    )
    model.precompute_and_store_context_kv(context_states, context_positions, slots)
    torch.npu.current_stream(device).synchronize()
