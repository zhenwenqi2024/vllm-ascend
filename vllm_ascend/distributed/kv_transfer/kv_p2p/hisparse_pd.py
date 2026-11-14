# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Push P KV into HiSparse's Host Source and NPU Indexer groups."""

import ctypes
import math
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import MooncakeBaseConnector, MooncakePushConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeConnectorMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.push_scheduler import MooncakePushConnectorScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.push_worker import MooncakePushConnectorWorker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import as_kv_cache_tensors

_HOST_ALIGNMENT = 2 * 1024 * 1024


class _HiSparsePDWorker(MooncakePushConnectorWorker):
    def _is_transfer_group(self, group_index: int) -> bool:
        return self.kv_cache_config.kv_cache_groups[group_index].enable_kv_transfer

    def _logical_num_blocks(self, group_index: int) -> int:
        group = self.kv_cache_config.kv_cache_groups[group_index]
        if group.host_resident:
            assert self.kv_cache_config.hisparse_host_num_blocks is not None
            return self.kv_cache_config.hisparse_host_num_blocks
        return self.num_blocks

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor | list[torch.Tensor]]) -> None:
        if self.kv_transfer_config.is_kv_consumer:
            context = self.vllm_config.compilation_config.static_forward_context
            caches: dict[str, Any] = dict(kv_caches)
            source_names = [
                name
                for group in self.kv_cache_config.kv_cache_groups
                if group.host_resident and group.enable_kv_transfer
                for name in group.layer_names
            ]
            source_parts = [cache for name in source_names for cache in as_kv_cache_tensors(kv_caches[name])]
            if not source_parts or any(cache.device.type != "cpu" for cache in source_parts):
                raise ValueError("HiSparse PD Source must consist of Host tensors")
            source_start = min(cache.data_ptr() for cache in source_parts)
            source_end = max(
                cache.data_ptr()
                + cache.element_size()
                + sum((size - 1) * stride * cache.element_size() for size, stride in zip(cache.shape, cache.stride()))
                for cache in source_parts
            )
            host_blocks = self.kv_cache_config.hisparse_host_num_blocks
            host_stride = self.kv_cache_config.hisparse_host_block_stride
            assert host_blocks is not None and host_stride is not None
            staging_size = (
                (max(source_end - source_start, host_blocks * host_stride) + _HOST_ALIGNMENT - 1) // _HOST_ALIGNMENT
            ) * _HOST_ALIGNMENT
            owner = torch.empty(staging_size + _HOST_ALIGNMENT - 1, dtype=torch.uint8, pin_memory=True)
            alignment_offset = (-owner.data_ptr()) % _HOST_ALIGNMENT
            staging = owner.narrow(0, alignment_offset, staging_size)
            self._source_staging_owner = owner
            self._host_stage_range = (staging.data_ptr(), staging.data_ptr() + staging_size, source_start, source_end)
            for name in source_names:
                staged_parts = []
                for cache in as_kv_cache_tensors(kv_caches[name]):
                    offset = alignment_offset + cache.data_ptr() - source_start
                    if offset % cache.element_size():
                        raise ValueError(f"Misaligned HiSparse Source layer {name!r}")
                    staged_parts.append(
                        staging.view(cache.dtype).as_strided(
                            cache.shape, cache.stride(), storage_offset=offset // cache.element_size()
                        )
                    )
                caches[name] = tuple(staged_parts)
            for group in self.kv_cache_config.kv_cache_groups:
                if group.role != "hisparse_indexer":
                    continue
                for name in group.layer_names:
                    stride = getattr(context[name], "hisparse_block_stride", 1)
                    if stride > 1:
                        caches[name] = tuple(cache[::stride] for cache in as_kv_cache_tensors(kv_caches[name]))
            kv_caches = caches
        super().register_kv_caches(kv_caches)

    def _append_packed_mla_addresses(
        self, batch, local_index, remote_index, local, pp, worker, src_list, dst_list, length_list
    ) -> bool:
        source_addrs = local.kv_caches_base_addr[local_index]
        target_addrs = worker.kv_caches_base_addr[remote_index]
        if len(source_addrs) != 2 or len(target_addrs) != 1:
            return False
        tokens = self.kv_cache_specs[self.layer_name_to_spec_index[batch.layer_name]].block_size
        component_bytes = local.block_lens[local_index]
        target_bytes = pp.block_lens[remote_index][0]
        if any(size % tokens for size in component_bytes) or sum(component_bytes) != target_bytes:
            raise ValueError("HiSparse PD MLA component sizes differ from Host Source")
        row_bytes = [size // tokens for size in component_bytes]
        for request in batch.requests:
            for source_block, target_block in zip(request.local_block_ids, request.remote_block_ids, strict=True):
                source_pages = [
                    addr + source_block * local.block_strides[local_index][component]
                    for component, addr in enumerate(source_addrs)
                ]
                target_page = target_addrs[0] + target_block * pp.block_strides[remote_index][0]
                for token in range(tokens):
                    for component in range(2):
                        src_list.append(source_pages[component] + token * row_bytes[component])
                        dst_list.append(target_page + token * sum(row_bytes) + (row_bytes[0] if component else 0))
                        length_list.append(row_bytes[component])
        return True

    def start_load_kv(self, metadata: MooncakeConnectorMetadata) -> None:
        if self.kv_transfer_config.is_kv_consumer:
            self._source_imports = getattr(self, "_source_imports", {})
            self._source_imports.update((req.request_id, req) for req in metadata.push_requests.values())
        super().start_load_kv(metadata)

    def get_transfer_results(self):
        results = super().get_transfer_results()
        if not self.kv_transfer_config.is_kv_consumer:
            return results
        for request_id in results.finished_recving:
            request = self._source_imports.pop(request_id, None)
            if request_id in results.failed_recving:
                continue
            if request is None:
                logger.error("HiSparse PD missing Host Source import metadata for %s", request_id)
                results.failed_recving.add(request_id)
                continue
            try:
                self._copy_imported_source(request)
            except Exception:
                logger.exception("HiSparse PD failed to import Host Source for %s", request_id)
                results.failed_recving.add(request_id)
        return results

    def _copy_imported_source(self, request) -> None:
        stage_start, stage_end, source_start, source_end = self._host_stage_range
        for layer_index, name in enumerate(self.transfer_metadata.layer_names):
            group_index = self.transfer_metadata.group_indices[layer_index]
            group = self.kv_cache_config.kv_cache_groups[group_index]
            if not group.host_resident:
                continue
            if self.transfer_metadata.block_size_scales[layer_index][0] != 1:
                raise ValueError("HiSparse PD Host Source must have one physical page per block")
            blocks = request.local_block_ids[group_index]
            block_size = group.kv_cache_spec.block_size
            first = request.num_computed_tokens // block_size
            last = math.ceil((request.num_computed_tokens + request.num_external_tokens) / block_size)
            if last > len(blocks):
                raise ValueError("HiSparse PD Host Source block table is shorter than the import")
            base = self.transfer_metadata.kv_caches_base_addr[layer_index][0]
            stride = self.transfer_metadata.block_strides[layer_index][0]
            length = self.transfer_metadata.block_lens[layer_index][0]
            for block in blocks[first:last]:
                page = base + block * stride
                target = source_start + page - stage_start
                if block == 0 or page < stage_start or page + length > stage_end or target + length > source_end:
                    raise ValueError(f"HiSparse PD Host Source page out of range for {name}")
                ctypes.memmove(target, page, length)


class HiSparsePDConnector(MooncakePushConnector):
    """Use Ascend Mooncake transport with Core HiSparse import ownership."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        MooncakeBaseConnector.__init__(self, vllm_config, role, kv_cache_config)
        if self.kv_transfer_config.is_kv_consumer:
            if kv_cache_config.hisparse_host_num_blocks is None:
                raise ValueError("HiSparsePDConnector consumer requires HiSparse Host Source")
            transferable = [group for group in kv_cache_config.kv_cache_groups if group.enable_kv_transfer]
            if (
                len(transferable) != 2
                or not transferable[0].host_resident
                or transferable[1].role != "hisparse_indexer"
            ):
                raise ValueError("HiSparsePDConnector requires Host Source and NPU Indexer transfer groups")
        elif kv_cache_config.hisparse_host_num_blocks is not None:
            raise ValueError("HiSparsePDConnector producer must use ordinary Prefill KV")

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = MooncakePushConnectorScheduler(vllm_config, str(self.engine_id), kv_cache_config)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = _HiSparsePDWorker(vllm_config, str(self.engine_id), kv_cache_config)
        else:
            raise ValueError(f"Unsupported KVConnectorRole: {role}")
