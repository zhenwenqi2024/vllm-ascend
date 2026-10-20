# SPDX-License-Identifier: Apache-2.0
"""Translate existing HiSparse SOURCE/INDEXER views to byte descriptors."""

from __future__ import annotations

import torch

from .protocol import LayerLayout, validate_layouts


def build_layouts(kv_caches, kv_cache_config, forward_context, *, producer: bool, allow_cpu_indexer=False):
    """Return descriptors, deduplicated registration ranges and strong owners.

    SOURCE uses a mapped producer alias or a raw consumer host address. The
    reader registers producer ranges and consumer INDEXER allocations only.
    RESIDENT/HOT are local replicas, absent from the transfer protocol.
    """
    if not isinstance(producer, bool):
        raise TypeError("HiSparse P/D layout requires an explicit producer boolean")
    tensors = {name: config for config in kv_cache_config.kv_cache_tensors for name in config.layers}
    layouts = []
    regions = {}
    owners = []
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        role = str(getattr(group.role, "value", group.role))
        if not group.enable_kv_transfer:
            continue
        if role not in ("hisparse_source", "hisparse_indexer") or group.is_eagle_group:
            raise ValueError("HiSparse P/D supports only target SOURCE and INDEXER groups")
        block_size = group.kv_cache_spec.block_size
        for name in group.layer_names:
            cache = kv_caches[name]
            if isinstance(cache, tuple):
                if len(cache) != 1:
                    raise ValueError("HiSparse P/D requires unquantized BF16 indexer KV")
                cache = cache[0]
            if cache.dtype != torch.bfloat16:
                raise ValueError("HiSparse P/D requires BF16 SOURCE and INDEXER caches")
            if role == "hisparse_source":
                if cache.device.type != "cpu" or not cache.is_contiguous():
                    raise ValueError("SOURCE must be a contiguous packed host layer")
                handle = forward_context[name].hisparse_cache
                mapping = handle.runtime.cache_ops.host_mapping
                # An existing ACL mapping is borrowed, never re-registered or closed here.
                alias = mapping.address_of(cache)  # Also validates the mapped allocation bounds and liveness.
                base = alias if producer else cache.data_ptr()
                row_bytes = cache.shape[-1] * cache.element_size()
                num_blocks = kv_cache_config.hisparse_host_num_blocks
                block_stride = block_size * row_bytes
                if cache.numel() * cache.element_size() != num_blocks * block_stride:
                    raise ValueError("SOURCE shape disagrees with its host manager capacity")
                key = (mapping.device_ptr if producer else mapping.host_address, mapping.nbytes)
                regions[key] = role
                owners.extend((cache, mapping))
            else:
                if cache.device.type != "npu" and not (allow_cpu_indexer and cache.device.type == "cpu"):
                    raise ValueError("INDEXER must be on the Ascend device")
                if cache.ndim != 4 or cache.shape[1] != block_size:
                    raise ValueError("INDEXER must have the Ascend [physical blocks, tokens, heads, dim] view")
                row_bytes = cache.shape[2] * cache.shape[3] * cache.element_size()
                if cache.stride(3) != 1 or cache.stride(2) != cache.shape[3] or cache.stride(1) * 2 != row_bytes:
                    raise ValueError("INDEXER token rows must be dense")
                base = cache.data_ptr()
                block_stride = tensors[name].block_stride
                num_blocks = kv_cache_config.num_blocks
                factor = forward_context[name].hisparse_block_stride
                if block_stride != factor * block_size * row_bytes:
                    raise ValueError("Expanded INDEXER view disagrees with manager block stride")
                storage = cache.untyped_storage()
                extent = (num_blocks - 1) * block_stride + block_size * row_bytes
                if base + extent > storage.data_ptr() + storage.nbytes():
                    raise ValueError("INDEXER manager view extends past its shared storage")
                key = (storage.data_ptr(), storage.nbytes())
                regions[key] = role
                owners.append(cache)
            layouts.append(LayerLayout(name, group_id, role, block_size, row_bytes, base, block_stride, num_blocks))
    validate_layouts(layouts)
    # Shared slabs are registered once; actual per-layer offsets stay in layouts.
    return layouts, regions, owners
