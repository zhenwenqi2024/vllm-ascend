# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MLA over shared packed KV, without per-query KV materialization."""

import math
from pathlib import Path
from threading import Lock

import torch

_LATENT_DIM = 512
_ROPE_DIM = 64
_ROW_WIDTH = _LATENT_DIM + _ROPE_DIM
_MAX_TOPK = 32768
_MAX_QUERIES = 1048576
_INT64_MAX = (1 << 63) - 1
_INT32_MAX = (1 << 31) - 1

# This lock only protects process-local extension registration, never inference
# execution. It prevents simultaneous first callers from registering twice.
_LIBRARY_LOAD_LOCK = Lock()


def load_direct_library(path: str) -> None:
    """Load only the private extension; keep the existing Ascend extension."""
    library = Path(path).expanduser()
    if not library.is_absolute() or library.suffix != ".so" or not library.is_file():
        raise ValueError("HiSparse direct library must be an existing absolute .so path")
    canonical = str(library.resolve(strict=True))
    with _LIBRARY_LOAD_LOCK:
        if canonical in torch.ops.loaded_libraries:
            if not hasattr(torch.ops.hisparse_direct, "sparse_flash_attention"):
                raise RuntimeError("Loaded private library does not register HiSparse direct SFA")
            return
        if hasattr(torch.ops.hisparse_direct, "sparse_flash_attention"):
            raise RuntimeError("HiSparse direct SFA was already registered by a different library")
        torch.ops.load_library(canonical)
        if not hasattr(torch.ops.hisparse_direct, "sparse_flash_attention"):
            raise RuntimeError("Private library did not register HiSparse direct SFA")


def ensure_direct_operator() -> None:
    """Require explicit extension registration before model construction."""
    if not hasattr(torch.ops.hisparse_direct, "sparse_flash_attention"):
        raise RuntimeError(
            "HiSparse direct SFA is unavailable; rebuild the Ascend 950 native extension and custom operator vendor, "
            "then import vllm_ascend.vllm_ascend_C before model construction. "
            "A private binding is only supported in a separate probe process without the normal extension."
        )


def packed_cache_storage(cache: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
    """Return a bounded contiguous alias starting at this layer's first row.

    The alias includes inter-page gaps without copying them. Its pointer keeps
    the layer's storage offset; explicit page strides decide which rows the
    kernel may read. Both ordinary strided views and expanded attention views
    use the same addressing contract.
    """
    if cache.dtype != torch.bfloat16 or cache.ndim != 3 or cache.shape[-1] != _ROW_WIDTH:
        raise ValueError("Packed HiSparse KV must be BF16 [pages, block_size, 576]")
    pages, block_size, _ = cache.shape
    if pages <= 0 or block_size <= 0 or block_size > 4095:
        raise ValueError("Packed HiSparse KV requires nonempty pages with block_size <= 4095")
    block_stride = cache.stride(0)
    if cache.stride(1) != _ROW_WIDTH or cache.stride(2) != 1 or block_stride < block_size * _ROW_WIDTH:
        raise ValueError("Packed HiSparse KV requires contiguous rows and nonoverlapping pages")
    capacity = pages * block_size
    span = (pages - 1) * block_stride + block_size * _ROW_WIDTH
    offset = cache.storage_offset()
    if span > _INT64_MAX or capacity > _INT32_MAX or offset < 0:
        raise ValueError("Packed HiSparse KV exceeds int64 addressing or int32 physical-row capacity")
    if (offset + span) * cache.element_size() > cache.untyped_storage().nbytes():
        raise ValueError("Packed HiSparse KV view exceeds its backing storage")
    raw = torch.as_strided(cache, (span,), (1,), storage_offset=offset)
    return raw, block_size, block_stride, capacity


def compact_physical_indices(
    indices: torch.Tensor, capacity: int, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep valid indices in their original order, including duplicates.

    Bounds are checked on device. Out-of-range entries are excluded like -1,
    so no malformed physical index can address another allocation. Unique
    FP32 position keys preserve order without integer ArgSort on AiCPU.
    """
    if indices.ndim != 2 or indices.dtype != torch.int32 or indices.stride(1) != 1:
        raise ValueError("HiSparse physical indices must be int32 [queries, top_k] with contiguous columns")
    topk = indices.shape[1]
    if topk <= 0 or topk > _MAX_TOPK or topk % 128:
        raise ValueError("HiSparse direct attention requires top_k in [128, 32768], divisible by 128")
    if positions.shape != (1, topk) or positions.dtype != torch.float32 or positions.device != indices.device:
        raise ValueError("HiSparse position keys must be FP32 [1, top_k] on the indices device")
    valid = (indices >= 0) & (indices.to(torch.int64) < capacity)
    keys = positions + (~valid).to(torch.float32) * topk
    order = keys.sort(dim=-1).indices
    compacted = indices.masked_fill(~valid, -1).gather(1, order)
    counts = valid.sum(dim=-1, dtype=torch.int32)
    return compacted, counts


def packed_sparse_flash_attention(
    query: torch.Tensor,
    query_rope: torch.Tensor,
    packed_cache: torch.Tensor,
    physical_topk: torch.Tensor,
    scale: float,
    positions: torch.Tensor,
    query_ends: torch.Tensor,
    *,
    use_eager_compaction: bool = False,
) -> torch.Tensor:
    """Run one SFA invocation for all queries over a shared packed KV view.

    Each query is a one-token sequence with an explicit selected-row count.
    Physical addresses carry no causal position; the caller supplies the
    already causally selected rows from the indexer/residency resolver.
    There is no fallback to the materializing bridge.
    """
    if query.ndim != 3 or query.dtype != torch.bfloat16 or query.shape[-1] != _LATENT_DIM:
        raise ValueError("HiSparse direct query must be BF16 [queries, heads, 512]")
    rows, heads, _ = query.shape
    if rows <= 0 or rows > _MAX_QUERIES or heads < 8 or heads > 64 or heads % 8:
        raise ValueError("HiSparse direct attention requires 1..1048576 queries and 8..64 heads divisible by 8")
    if query_rope is None or query_rope.shape != (rows, heads, _ROPE_DIM) or query_rope.dtype != query.dtype:
        raise ValueError("HiSparse direct query RoPE must be BF16 [queries, heads, 64]")
    if any(t.device != query.device for t in (query_rope, packed_cache, physical_topk, positions, query_ends)):
        raise ValueError("HiSparse direct attention inputs must share a device")
    if physical_topk.ndim != 2 or physical_topk.shape[0] != rows:
        raise ValueError("HiSparse physical indices must have one row per query")
    if query_ends.dtype != torch.int32 or query_ends.shape != (rows,) or not query_ends.is_contiguous():
        raise ValueError("HiSparse query ends must be contiguous int32 [queries]")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("HiSparse attention scale must be positive and finite")
    raw, block_size, block_stride, capacity = packed_cache_storage(packed_cache)
    if use_eager_compaction and rows <= 32:
        from vllm_ascend.ops.hisparse_compact import compact_physical_indices_eager

        selected, valid_counts = compact_physical_indices_eager(physical_topk, capacity, positions)
    else:
        selected, valid_counts = compact_physical_indices(physical_topk, capacity, positions)
    ensure_direct_operator()
    output = torch.ops.hisparse_direct.sparse_flash_attention(
        query.contiguous(),
        raw,
        selected.unsqueeze(1),
        query_ends,
        valid_counts,
        query_rope.contiguous(),
        scale,
        block_size,
        block_stride,
        capacity,
    )
    # Keep the public contract explicit even when the kernel skips empty rows.
    return output.masked_fill((valid_counts == 0)[:, None, None], 0)
