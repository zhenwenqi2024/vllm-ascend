# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Bind Ascend indexer views while retaining upstream HiSparse allocation."""

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.v1.hisparse.binding import init_hisparse_kv_cache as upstream_init_hisparse_kv_cache
from vllm.v1.hisparse.runtime import drop_hisparse_host_pool, get_hisparse_host_pool, release_pinned_state
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.block_table import BlockTables


def align_hisparse_indexer_specs(specs: dict[str, KVCacheSpec]) -> dict[str, KVCacheSpec]:
    """Declare dense indexer-page addressing to Core's shared-slab planner.

    Core includes this alignment when sizing each physical device block, before
    choosing the block count. Source MLA specs and their host layout stay intact.
    """
    return {
        name: replace(spec, block_stride_alignment=math.lcm(spec.get_block_stride_alignment(), spec.page_size_bytes))
        if isinstance(spec, AscendSFAIndexerCacheSpec)
        else spec
        for name, spec in specs.items()
    }


def init_hisparse_kv_cache(
    kv_cache_config: KVCacheConfig,
    device: torch.device,
    kernel_block_sizes: list[int],
    vllm_config: VllmConfig,
    forward_context: dict[str, Any],
    block_tables: "BlockTables",
) -> dict[str, Any]:
    indexer_specs = {}
    for group in kv_cache_config.kv_cache_groups:
        for name in group.layer_names:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            if isinstance(spec, AscendSFAIndexerCacheSpec):
                if spec.scale_dim or spec.cache_sparse_li_c8 or spec.cache_sparse_li_c4:
                    raise ValueError("Ascend HiSparse currently requires an unquantized indexer cache.")
                indexer_specs[name] = spec

    kv_caches = upstream_init_hisparse_kv_cache(
        kv_cache_config, device, kernel_block_sizes, vllm_config, forward_context, block_tables
    )
    try:
        _bind_indexer_views(kv_caches, indexer_specs, forward_context)
    except Exception:
        pool = get_hisparse_host_pool(vllm_config, kv_cache_config)
        runtimes = [
            handle.runtime
            for layer in forward_context.values()
            if (handle := getattr(layer, "hisparse_cache", None)) is not None
        ]
        registered = [] if pool.shared_region is not None or pool.registered is None else [pool.registered]
        release_pinned_state(runtimes, registered, pool.shared_region)
        drop_hisparse_host_pool(vllm_config)
        raise
    return kv_caches


def _bind_indexer_views(
    kv_caches: dict[str, Any], indexer_specs: dict[str, AscendSFAIndexerCacheSpec], forward_context: dict[str, Any]
) -> None:
    for name, spec in indexer_specs.items():
        cache = kv_caches[name]
        # The indexer kernels address dense pages. Expand the view over the
        # shared slab and let the layer scale its block IDs and write slots.
        if cache.ndim != 4 or cache.shape[1] != spec.num_kv_heads:
            raise ValueError(f"Unexpected HiSparse indexer cache shape for {name}: {tuple(cache.shape)}")
        block_size = cache.shape[2]
        page_elements = block_size * spec.num_kv_heads * spec.head_size
        if cache.stride(0) % page_elements:
            raise ValueError("HiSparse indexer slab stride must be a whole number of indexer pages.")
        block_stride = cache.stride(0) // page_elements
        physical_blocks = (cache.shape[0] - 1) * block_stride + 1
        indexer_cache = torch.as_strided(
            cache,
            size=(physical_blocks, block_size, spec.num_kv_heads, spec.head_size),
            stride=(page_elements, spec.num_kv_heads * spec.head_size, spec.head_size, 1),
        )
        forward_context[name].hisparse_block_stride = block_stride
        kv_caches[name] = (indexer_cache,)
