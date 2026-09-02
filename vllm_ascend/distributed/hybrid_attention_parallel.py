# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Layout helpers shared by DP-local MLA and fine-grained TP projections."""

from dataclasses import dataclass

import torch
from vllm.distributed.parallel_state import GroupCoordinator


@dataclass(frozen=True)
class HybridAttentionLayout:
    """The fixed-capacity token layout used by one fine-grained TP group."""

    local_num_tokens: int
    exchange_num_tokens: int
    group_size: int


def make_global_state_slot(owner_rank: int, local_slot: int, slots_per_rank: int) -> int:
    """Map a DP-local state slot to a subgroup-wide KDA slot."""
    if owner_rank < 0 or local_slot < 0 or slots_per_rank <= 0:
        raise ValueError("owner_rank/local_slot must be non-negative and slots_per_rank must be positive")
    if local_slot >= slots_per_rank:
        raise ValueError(f"local_slot={local_slot} must be smaller than slots_per_rank={slots_per_rank}")
    return owner_rank * slots_per_rank + local_slot


def gather_dp_tokens(
    hidden_states: torch.Tensor,
    group: GroupCoordinator,
    exchange_num_tokens: int,
) -> tuple[torch.Tensor, HybridAttentionLayout]:
    """Pad and gather DP-local tokens in deterministic subgroup-rank order."""
    local_num_tokens = hidden_states.shape[0]
    if local_num_tokens > exchange_num_tokens:
        raise ValueError(
            f"local_num_tokens={local_num_tokens} exceeds exchange_num_tokens={exchange_num_tokens}"
        )
    padded = hidden_states.new_zeros((exchange_num_tokens, *hidden_states.shape[1:]))
    padded[:local_num_tokens].copy_(hidden_states)
    gathered = group.all_gather(padded, dim=0)
    return gathered, HybridAttentionLayout(local_num_tokens, exchange_num_tokens, group.world_size)


def reduce_scatter_dp_tokens(
    partial_output: torch.Tensor,
    group: GroupCoordinator,
    layout: HybridAttentionLayout,
) -> torch.Tensor:
    """Sum sharded outputs and restore the caller's DP-local token batch."""
    expected_tokens = layout.exchange_num_tokens * layout.group_size
    if partial_output.shape[0] != expected_tokens:
        raise ValueError(f"partial_output has {partial_output.shape[0]} tokens, expected {expected_tokens}")
    output = group.reduce_scatter(partial_output, dim=0)
    return output[: layout.local_num_tokens]
