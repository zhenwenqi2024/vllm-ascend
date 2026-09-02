# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step metadata and cache ownership helpers for KDA state parallelism.

KDA state-parallel ranks are carved out of the data-parallel axis.  Every
rank therefore has an independent scheduler and an independent physical block
namespace.  This module federates the small control-plane metadata once per
model step and maps a logical ``(owner, block_id)`` to the physical state cache
layout ``[block_id, owner, ...local_heads]``.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.distributed.parallel_state import get_kda_tp_group


@dataclasses.dataclass(frozen=True)
class KDAStateCopyPlan:
    """One logical prefix-cache state move in an owner's block namespace."""

    mamba_group_id: int
    conv_src_block_id: int
    temporal_src_block_id: int
    dest_block_id: int
    accepted_token_bias: int


def make_kda_copy_plan_callback(
    plans: list[KDAStateCopyPlan],
) -> Callable[[list[int], int, int, int, Any], None]:
    """Create the callback consumed by vLLM's ``preprocess_mamba`` hook."""

    def collect(
        mamba_group_ids: list[int],
        src_block_idx: int,
        dest_block_idx: int,
        accepted_token_bias: int,
        req_state: Any,
    ) -> None:
        for group_id in mamba_group_ids:
            block_ids = req_state.block_ids[group_id]
            plans.append(
                KDAStateCopyPlan(
                    mamba_group_id=group_id,
                    conv_src_block_id=block_ids[src_block_idx],
                    temporal_src_block_id=block_ids[src_block_idx + accepted_token_bias],
                    dest_block_id=block_ids[dest_block_idx],
                    accepted_token_bias=accepted_token_bias,
                )
            )

    return collect


def _map_tensors(value: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    """Copy a metadata tree while applying ``fn`` to every tensor leaf."""
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {key: _map_tensors(item, fn) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_tensors(item, fn) for item in value]
    if isinstance(value, tuple):
        return (
            type(value)(*(_map_tensors(item, fn) for item in value))
            if hasattr(value, "_fields")
            else tuple(_map_tensors(item, fn) for item in value)
        )
    if dataclasses.is_dataclass(value):
        result = copy.copy(value)
        names = {field.name for field in dataclasses.fields(value)}
        names.update(getattr(value, "__dict__", {}))
        for name in names:
            object.__setattr__(
                result,
                name,
                _map_tensors(getattr(value, name), fn),
            )
        return result
    return value


def _metadata_for_cpu(metadata: GDNAttentionMetadata) -> GDNAttentionMetadata:
    return _map_tensors(metadata, lambda tensor: tensor.detach().cpu())


def _metadata_for_device(
    metadata: GDNAttentionMetadata,
    device: torch.device,
) -> GDNAttentionMetadata:
    return _map_tensors(
        metadata,
        lambda tensor: tensor.to(device=device, non_blocking=True),
    )


def gather_kda_step_metadata(
    attn_metadata: dict[str, Any] | None,
    local_copy_plans: list[KDAStateCopyPlan],
    device: torch.device,
) -> tuple[
    dict[str, tuple[GDNAttentionMetadata, ...]],
    tuple[tuple[KDAStateCopyPlan, ...], ...],
]:
    """Federate KDA metadata and prefix-copy plans once per model step.

    This intentionally uses the group's CPU process group.  The data is built
    on the host by the scheduler and the collective stays outside ACL graph
    capture.  A packed device-side control plane can replace this correctness
    path without changing the KDA layer contract.
    """
    metadata_indices: dict[str, int] = {}
    unique_metadata: list[GDNAttentionMetadata] = []
    object_indices: dict[int, int] = {}
    for prefix, metadata in (attn_metadata or {}).items():
        if not isinstance(metadata, GDNAttentionMetadata):
            continue
        object_id = id(metadata)
        metadata_index = object_indices.get(object_id)
        if metadata_index is None:
            metadata_index = len(unique_metadata)
            object_indices[object_id] = metadata_index
            unique_metadata.append(_metadata_for_cpu(metadata))
        metadata_indices[prefix] = metadata_index

    local_packet = (
        metadata_indices,
        tuple(unique_metadata),
        tuple(local_copy_plans),
    )
    group = get_kda_tp_group()
    packets: list[Any] = [None] * group.world_size
    dist.all_gather_object(packets, local_packet, group=group.cpu_group)

    metadata_by_prefix: dict[str, list[GDNAttentionMetadata]] = {}
    copy_plans_by_owner: list[tuple[KDAStateCopyPlan, ...]] = []
    for owner_indices, owner_unique_metadata, owner_copy_plans in packets:
        copy_plans_by_owner.append(owner_copy_plans)
        device_metadata = tuple(_metadata_for_device(metadata, device) for metadata in owner_unique_metadata)
        for prefix, metadata_index in owner_indices.items():
            metadata_by_prefix.setdefault(prefix, []).append(device_metadata[metadata_index])

    expected_owners = group.world_size
    for prefix, owner_metadata in metadata_by_prefix.items():
        if len(owner_metadata) != expected_owners:
            raise RuntimeError(
                f"KDA metadata for {prefix!r} is missing on one or more "
                f"state-parallel ranks: {len(owner_metadata)}/{expected_owners}."
            )
    return (
        {prefix: tuple(items) for prefix, items in metadata_by_prefix.items()},
        tuple(copy_plans_by_owner),
    )


def _copy_owner_state(
    state: torch.Tensor,
    owner_rank: int,
    plan: KDAStateCopyPlan,
    *,
    is_conv_state: bool,
) -> None:
    if is_conv_state:
        bias = plan.accepted_token_bias
        if is_conv_state_dim_first():
            if bias != 0:
                raise RuntimeError(
                    "KDA state parallelism requires SD convolution-state "
                    "layout when more than one speculative token is accepted."
                )
            state[plan.dest_block_id, owner_rank].copy_(state[plan.conv_src_block_id, owner_rank])
        else:
            destination = state[plan.dest_block_id, owner_rank]
            source = state[plan.conv_src_block_id, owner_rank]
            destination[: source.shape[0] - bias].copy_(source[bias:])
        return

    state[plan.dest_block_id, owner_rank].copy_(state[plan.temporal_src_block_id, owner_rank])


def execute_kda_copy_plans(
    copy_plans_by_owner: tuple[tuple[KDAStateCopyPlan, ...], ...],
    kv_cache_config: Any,
    static_forward_context: dict[str, Any],
) -> None:
    """Execute every owner's prefix-cache moves on this rank's head shard."""
    for owner_rank, plans in enumerate(copy_plans_by_owner):
        for plan in plans:
            layer_names = kv_cache_config.kv_cache_groups[plan.mamba_group_id].layer_names
            for layer_name in layer_names:
                attention = static_forward_context[layer_name]
                if getattr(attention, "kda_state_parallel_size", 1) <= 1:
                    continue
                conv_state, recurrent_state = attention.kv_cache[:2]
                _copy_owner_state(
                    conv_state,
                    owner_rank,
                    plan,
                    is_conv_state=True,
                )
                _copy_owner_state(
                    recurrent_state,
                    owner_rank,
                    plan,
                    is_conv_state=False,
                )


def _remap_state_indices(
    indices: torch.Tensor | None,
    owner_rank: int,
    parallel_size: int,
) -> torch.Tensor | None:
    if indices is None:
        return None
    mapped = indices * parallel_size + owner_rank
    return torch.where(indices == PAD_SLOT_ID, indices, mapped)


def remap_kda_metadata_state_indices(
    metadata: GDNAttentionMetadata,
    owner_rank: int,
    parallel_size: int,
) -> GDNAttentionMetadata:
    """Map one owner's logical block ids to flattened physical state slots."""
    result = copy.copy(metadata)
    for name in (
        "spec_state_indices_tensor",
        "non_spec_state_indices_tensor",
        "prefill_state_indices",
    ):
        setattr(
            result,
            name,
            _remap_state_indices(
                getattr(metadata, name),
                owner_rank,
                parallel_size,
            ),
        )

    prefill = getattr(metadata, "non_spec_prefill_metadata", None)
    if prefill is not None:
        prefill = copy.copy(prefill)
        prefill.causal_conv1d = copy.copy(prefill.causal_conv1d)
        prefill.causal_conv1d.cache_indices = _remap_state_indices(
            prefill.causal_conv1d.cache_indices,
            owner_rank,
            parallel_size,
        )
    result.non_spec_prefill_metadata = prefill

    decode = getattr(metadata, "non_spec_decode_metadata", None)
    if decode is not None:
        decode = copy.copy(decode)
        decode.causal_conv1d = copy.copy(decode.causal_conv1d)
        decode.causal_conv1d.cache_indices = _remap_state_indices(
            decode.causal_conv1d.cache_indices,
            owner_rank,
            parallel_size,
        )
    result.non_spec_decode_metadata = decode

    spec = getattr(metadata, "spec_decode_metadata", None)
    if spec is not None:
        spec = copy.copy(spec)
        spec.spec_causal_conv1d = copy.copy(spec.spec_causal_conv1d)
        spec.spec_causal_conv1d.cache_indices = _remap_state_indices(
            spec.spec_causal_conv1d.cache_indices,
            owner_rank,
            parallel_size,
        )
    result.spec_decode_metadata = spec
    return result


def flatten_kda_state_cache(state: torch.Tensor, parallel_size: int) -> torch.Tensor:
    """Flatten ``[block, owner, ...]`` without copying storage."""
    if state.shape[1] != parallel_size:
        raise RuntimeError(
            "KDA state cache owner dimension does not match the configured "
            f"parallel size: {state.shape[1]} != {parallel_size}."
        )
    return state.flatten(0, 1)
