# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed device counters for valid source routes, including ACL graph replay."""

import torch


class ExpertLoadProbe(torch.nn.Module):
    def __init__(self, num_experts: int, device: torch.device | str):
        super().__init__()
        self.num_experts = num_experts
        self.register_buffer("source_token_count", None, persistent=False)
        self.register_buffer("source_positions", None, persistent=False)
        # Expert assignments, supported calls, invalid IDs. The call count is
        # alignment bookkeeping, including dummy collective participation.
        self.register_buffer("totals", torch.zeros(num_experts + 2, dtype=torch.int64, device=device), persistent=False)

    def record_routes(self, physical_ids, valid_mask, source_offset=0):
        if (
            valid_mask is None
            or valid_mask.dtype != torch.bool
            or physical_ids.ndim != 2
            or valid_mask.shape != physical_ids.shape[:1]
            or self.source_positions is None
            or physical_ids.shape[0] > self.source_positions.numel()
        ):
            return
        # The shared scalar is zero for dummy, warmup and stopped collection.
        active = valid_mask[:, None] & (
            self.source_positions[: physical_ids.shape[0], None] + source_offset < self.source_token_count
        )
        valid = (physical_ids >= 0) & (physical_ids < self.num_experts)
        self.totals[: self.num_experts].scatter_add_(
            0,
            physical_ids.clamp(0, self.num_experts - 1).reshape(-1).to(torch.int64),
            (active & valid).reshape(-1).to(torch.int64),
        )
        self.totals[-2].add_(1)
        self.totals[-1].add_((active & ~valid).sum())
