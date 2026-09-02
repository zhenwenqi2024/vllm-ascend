# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Token-layout helpers for DP-local attention with sharded weights."""

import torch
from vllm.distributed.parallel_state import GroupCoordinator


def gather_token_counts(
    input_: torch.Tensor,
    group: GroupCoordinator,
) -> tuple[list[int], int]:
    """Collect DP-local token counts and return the largest batch size."""
    local_count = torch.tensor(
        [input_.shape[0]], dtype=torch.int64, device=input_.device
    )
    counts = group.all_gather(local_count, dim=0).cpu().tolist()
    return counts, max(counts)


def pad_tokens(input_: torch.Tensor, token_capacity: int) -> torch.Tensor:
    """Right-pad a token-major tensor to a common collective capacity."""
    if input_.shape[0] > token_capacity:
        raise ValueError(
            f"input has {input_.shape[0]} tokens, capacity is {token_capacity}"
        )
    if input_.shape[0] == token_capacity:
        return input_
    padded = input_.new_zeros((token_capacity, *input_.shape[1:]))
    padded[: input_.shape[0]].copy_(input_)
    return padded
