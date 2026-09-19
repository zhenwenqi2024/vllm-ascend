# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed device counters for valid source routes, including ACL graph replay."""

import torch


class ExpertLoadProbe(torch.nn.Module):
    def __init__(self, num_experts: int, device: torch.device | str):
        super().__init__()
        self.num_experts = num_experts
        self.eplb_enabled = False
        self.generation = 0
        self.register_buffer("logical_to_physical", None, persistent=False)
        self.register_buffer("owner_totals", None, persistent=False)
        self.calibration_templates = {}
        self.register_buffer("history", None, persistent=False)
        self.register_buffer("history_slot", None, persistent=False)
        self.register_buffer("comm_history", None, persistent=False)
        self.register_buffer("source_token_count", None, persistent=False)
        self.register_buffer("source_positions", None, persistent=False)
        # Standalone counters / history-row template; history mode does not
        # maintain a duplicate running total. The fields are expert assignments,
        # supported calls and invalid IDs. The call count is
        # alignment bookkeeping, including dummy collective participation.
        self.register_buffer("totals", torch.zeros(num_experts + 2, dtype=torch.int64, device=device), persistent=False)

    def record_routes(self, physical_ids, valid_mask=None, source_offset=0):
        if (
            physical_ids.ndim != 2
            or (
                valid_mask is not None
                and (valid_mask.dtype != torch.bool or valid_mask.shape != physical_ids.shape[:1])
            )
            or self.source_positions is None
            or physical_ids.shape[0] > self.source_positions.numel()
        ):
            return
        counts = self.totals if self.history is None else torch.zeros_like(self.totals)
        # AllGather replicas without source ownership still count the call.
        if physical_ids.shape[0] == 0:
            counts[-2].add_(1)
            self._append_history(counts)
            return
        # The shared scalar is zero for dummy, warmup and stopped collection.
        active = self.source_positions[: physical_ids.shape[0], None] + source_offset < self.source_token_count
        if valid_mask is not None:
            active = active & valid_mask[:, None]
        valid = (physical_ids >= 0) & (physical_ids < self.num_experts)
        physical_slots = physical_ids.clamp(0, self.num_experts - 1).to(torch.int64)
        logical_ids = physical_slots
        valid_counts = (active & valid).reshape(-1).to(torch.int64)
        if self.logical_to_physical is not None:
            # Zero redundancy: the live permutation changes in place, including
            # between graph replays. Keep counters in logical expert space.
            inverse = torch.argsort(self.logical_to_physical)
            logical_ids = inverse[logical_ids]
        if self.owner_totals is not None:
            slots = self.num_experts // self.owner_totals.shape[0]
            owners = physical_slots // slots
            self.owner_totals.view(-1).scatter_add_(
                0,
                (owners * self.num_experts + logical_ids).reshape(-1),
                valid_counts,
            )
        counts[: self.num_experts].scatter_add_(
            0,
            logical_ids.reshape(-1),
            valid_counts,
        )
        counts[-2].add_(1)
        counts[-1].add_((active & ~valid).sum())
        self._append_history(counts)

    def _append_history(self, counts):
        if self.history is not None:
            self.history.index_add_(0, self.history_slot, counts.unsqueeze(0))

    def record_comm(self, code):
        if self.comm_history is not None:
            self.comm_history.index_fill_(0, self.history_slot, code)
