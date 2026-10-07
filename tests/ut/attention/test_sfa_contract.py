# SPDX-License-Identifier: Apache-2.0
"""Keep Python's lightweight SFA contract aligned with the kernel sources."""

from pathlib import Path

import pytest
import regex as re

from vllm_ascend.attention.sfa_contract import (
    COPY_SFA_TAIL_BLOCKS,
    COPY_SFA_TAIL_TOKENS,
    LIM_CACHE_BLOCK_SIZE,
    LIM_MAX_HOT_TOKENS,
    LIM_MAX_QUERY_ROWS,
    LIM_TOPK,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("kernel", ["fused_lightning_indexer_manage", "fused_quant_lightning_indexer_manage"])
@pytest.mark.parametrize(
    "cpp_name,python_value",
    [
        ("TOPK", LIM_TOPK),
        ("MAX_ROUTES", LIM_MAX_QUERY_ROWS),
        ("MAX_CACHE_TOKENS", LIM_MAX_HOT_TOKENS),
        ("CACHE_BLOCK_SIZE", LIM_CACHE_BLOCK_SIZE),
    ],
)
def test_lim_contract_matches_cpp_header(kernel, cpp_name, python_value):
    header = REPO_ROOT / "csrc" / "attention" / kernel / "op_kernel" / f"{kernel}_constants.h"
    # A missing/changed definition fails rather than silently skipping parity.
    matches = re.findall(rf"\bconstexpr\s+uint32_t\s+{cpp_name}\s*=\s*(\d+)U?\s*;", header.read_text(encoding="utf-8"))
    assert len(matches) == 1, f"Missing or ambiguous {cpp_name} in {header}"
    assert int(matches[0]) == python_value, f"Update the shared Python contract for {header}:{cpp_name}"


def test_copy_sfa_alignment_is_derived_from_circular_tail_layout():
    assert COPY_SFA_TAIL_BLOCKS == 2
    assert COPY_SFA_TAIL_TOKENS == COPY_SFA_TAIL_BLOCKS * LIM_CACHE_BLOCK_SIZE
    assert COPY_SFA_TAIL_TOKENS > LIM_CACHE_BLOCK_SIZE
    # Kernel capacity and serving alignment are separate: the largest valid
    # serving budget rounds down to a whole circular-tail period.
    largest_aligned = LIM_MAX_HOT_TOKENS // COPY_SFA_TAIL_TOKENS * COPY_SFA_TAIL_TOKENS
    assert largest_aligned <= LIM_MAX_HOT_TOKENS < largest_aligned + COPY_SFA_TAIL_TOKENS
    assert largest_aligned >= LIM_MAX_QUERY_ROWS * LIM_TOPK
