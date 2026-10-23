# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Map and stably pack one fresh C8 selection without cross-layer caching."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _map_compact(
    source,
    output,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, TILE)
    token = tl.load(source + row.to(tl.int64) * ROW_STRIDE + column * COL_STRIDE, mask=column < WIDTH, other=-1)
    valid = (column < WIDTH) & (token >= 0) & (token < CAPACITY)
    # Scan counts valid entries in input order; duplicates remain independent.
    destination = tl.cumsum(valid.to(tl.int32), 0) - 1
    count = tl.sum(valid.to(tl.int32), 0)
    safe = tl.maximum(token, 0).to(tl.int64)
    mapped = (safe // BLOCK * PAGE_STRIDE + safe % BLOCK).to(tl.int32)
    base = output + row.to(tl.int64) * WIDTH
    tl.store(base + destination, mapped, mask=valid)
    # Disjoint from [0,count): no racing initialization or inter-program writes.
    tl.store(base + column, -1, mask=(column < WIDTH) & (column >= count))


def map_compact_c8_indices(indices: torch.Tensor, block: int, stride_rows: int, capacity: int) -> torch.Tensor:
    """Preserve original map+stable-compaction values, including all duplicates.

    Storage belongs to this invocation. Dynamic values are read on every call,
    including graph replay; no group, request, pointer or shape cache is used.
    """
    if indices.ndim != 2 or indices.dtype != torch.int32 or block <= 0 or stride_rows < block:
        raise ValueError("C8 mapping requires int32 [queries,top_k] and valid page geometry")
    if capacity <= 0 or capacity % block or (capacity // block - 1) * stride_rows + block > 2**31 - 1:
        raise ValueError("C8 physical row capacity is invalid")
    width = indices.shape[1]
    if indices.device.type != "npu" or not 0 < width <= 4096:
        from vllm_ascend.ops.hisparse_c8 import compact_c8_indices, map_c8_physical_indices

        return compact_c8_indices(map_c8_physical_indices(indices, block, stride_rows, capacity))
    result = torch.empty(indices.shape, device=indices.device, dtype=torch.int32)
    if indices.shape[0]:
        _map_compact[(indices.shape[0],)](
            indices,
            result,
            indices.stride(0),
            indices.stride(1),
            width,
            block,
            stride_rows,
            capacity,
            triton.next_power_of_2(width),
        )
    return result
