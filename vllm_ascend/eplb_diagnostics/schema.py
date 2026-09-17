# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only interpretation shared by the writer and offline report."""

import hashlib
import json
from typing import Any


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def decode_sample(before: list[int], after: list[int], num_experts: int) -> dict:
    delta = [b - a for a, b in zip(before, after)]
    calls = delta[num_experts]
    valid_calls = delta[num_experts + 1]
    counts = delta[:num_experts]
    quality = "ok"
    if any(value < 0 for value in delta):
        quality = "counter_reset_or_invalid_counts"
    elif calls == 0:
        quality = "not_executed"
    elif valid_calls != calls:
        quality = "backend_counts_unavailable"
    return {
        "quality": quality,
        "call_start": before[num_experts],
        "call_end": after[num_experts],
        "calls": calls,
        "valid_calls": valid_calls,
        "expert_assignments": counts if quality == "ok" else None,
        "comm_calls": delta[num_experts + 2 :],
        "count_semantics": "backend_reported_including_backend_padding_and_shared_experts",
    }


def local_logical_ids(layout: dict, num_experts: int) -> list[int | None]:
    """Invert the actual local map; never assume contiguous rank ownership."""
    result: list[int | None] = [None] * num_experts
    mapping = layout.get("global_to_local")
    if mapping is None:
        return result
    logical_to_physical = layout.get("logical_to_physical")
    physical_to_logical = {}
    if logical_to_physical is not None:
        for logical, physical_ids in enumerate(logical_to_physical):
            for physical in physical_ids:
                if physical >= 0:
                    physical_to_logical[physical] = logical
    for global_id, local in enumerate(mapping):
        if 0 <= local < num_experts:
            if layout["map_semantics"] == "logical_to_local":
                result[local] = global_id
            elif logical_to_physical is not None:
                result[local] = physical_to_logical.get(global_id)
            elif not layout.get("dynamic_eplb", False):
                # Without EPLB, the physical and logical IDs coincide.
                result[local] = global_id
    return result
