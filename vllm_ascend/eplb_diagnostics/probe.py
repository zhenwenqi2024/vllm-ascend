# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-only counters, captured and replayed together with the MoE operation."""

import torch


class ExpertLoadProbe(torch.nn.Module):
    # Stable schema order, including an explicit bucket for an unrecognized backend.
    COMM_METHODS = ("AllGatherCommImpl", "AlltoAllCommImpl", "MC2CommImpl", "FusedMC2CommImpl", "unknown")

    def __init__(self, num_experts: int, device: torch.device | str, num_physical_experts: int = 0):
        super().__init__()
        self.num_experts = num_experts
        self.num_physical_experts = num_physical_experts
        # Bound to shared model buffers after loading and before graph capture.
        self.register_buffer("source_token_count", None, persistent=False)
        self.register_buffer("source_positions", None, persistent=False)
        # Source-side destinations, valid-mask calls, invalid IDs. Never assume
        # backend group_list excludes graph padding or shared expert work.
        self.register_buffer(
            "route_totals", torch.zeros(num_physical_experts + 2, dtype=torch.int64, device=device), persistent=False
        )
        # Expert counts, total calls, valid calls, then calls per communication method.
        # A named buffer participates in sleep/wake restoration. Never replace its storage.
        self.register_buffer(
            "totals",
            torch.zeros(num_experts + 2 + len(self.COMM_METHODS), dtype=torch.int64, device=device),
            persistent=False,
        )
        # Reuse device constants instead of launching separate scalar increments
        # for the total, valid-call, and communication counters on every layer.
        headers = [
            [[1, valid] + [int(index == comm) for index in range(len(self.COMM_METHODS))] for valid in (0, 1)]
            for comm in range(len(self.COMM_METHODS))
        ]
        self.register_buffer("call_headers", torch.tensor(headers, dtype=torch.int64, device=device), persistent=False)
        self.register_buffer(
            "zero_counts", torch.zeros(num_experts, dtype=torch.int64, device=device), persistent=False
        )

    def record_routes(
        self, physical_ids: torch.Tensor, valid_mask: torch.Tensor | None, source_offset: int = 0
    ) -> None:
        """Count actual routed destinations using the TP-sharded MC2 mask.

        Caller must exclude unsupported ownership/shared/synthetic paths. All
        operations are device operations and are replayed in a captured graph.
        """
        n = self.num_physical_experts
        if (
            n == 0
            or valid_mask is None
            or physical_ids.ndim != 2
            or valid_mask.ndim != 1
            or valid_mask.numel() != physical_ids.shape[0]
            or valid_mask.dtype != torch.bool
            or valid_mask.device != self.route_totals.device
            or physical_ids.device != self.route_totals.device
        ):
            return
        if self.source_token_count is not None:
            if self.source_positions is None or physical_ids.shape[0] > self.source_positions.numel():
                return
            valid_mask = valid_mask & (
                self.source_positions[: physical_ids.shape[0]] + source_offset < self.source_token_count
            )
        valid_id = (physical_ids >= 0) & (physical_ids < n)
        active = valid_mask[:, None]
        self.route_totals[:n].scatter_add_(
            0,
            physical_ids.clamp(0, n - 1).reshape(-1).to(torch.int64),
            (active & valid_id).reshape(-1).to(torch.int64),
        )
        self.route_totals[n].add_(1)
        self.route_totals[n + 1].add_((active & ~valid_id).sum())

    def record(self, expert_tokens: torch.Tensor | None, group_list_type: int, comm_method: str) -> None:
        """Record backend-reported local work, not source-rank routing decisions.

        This runs for every invocation, including graph capture/replay. Sampling is
        performed outside the graph by subtracting two cumulative snapshots.
        No Python sampling flag, host read, synchronization, or collective belongs here.
        """
        n = self.num_experts
        comm_idx = self.COMM_METHODS.index(comm_method) if comm_method in self.COMM_METHODS else 4
        valid = (
            expert_tokens is not None
            and expert_tokens.ndim == 1
            and expert_tokens.numel() == n
            and expert_tokens.device == self.totals.device
            and group_list_type in (0, 1)
        )
        counts = self.zero_counts
        if valid:
            assert expert_tokens is not None
            counts = expert_tokens.to(torch.int64)
            if group_list_type == 0:
                counts = torch.cat((counts[:1], counts[1:] - counts[:-1]))
        self.totals.add_(torch.cat((counts, self.call_headers[comm_idx, int(valid)])))
