# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""NPU streams, mapped host memory and DMA for the common HiSparse runtime."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.hisparse.backend import HiSparseBackend

if TYPE_CHECKING:
    from vllm.v1.hisparse.runtime import HiSparseRuntime


class AscendHiSparseBackend(HiSparseBackend):
    # The common shared pool currently exchanges CUDA IPC events. Private
    # per-rank pools preserve the same scheduler and transfer completion rules.
    supports_shared_host_pool = False
    stats_dtype = torch.int64

    def check_available(self) -> None:
        from vllm_ascend.ops.hisparse import HiSparseNPUOps
        from vllm_ascend.utils import bootstrap_custom_op_env

        if not hasattr(torch.ops._C_ascend, "swap_blocks_batch"):
            # A5 disables the global custom-op selection policy. HiSparse still
            # needs its explicit DMA dependency; load the extension lazily here
            # without changing that policy for unrelated model operators.
            bootstrap_custom_op_env()
            try:
                importlib.import_module("vllm_ascend.vllm_ascend_C")
            except ImportError as error:
                raise RuntimeError("Ascend HiSparse requires the compiled Ascend DMA extension.") from error
            if not hasattr(torch.ops._C_ascend, "swap_blocks_batch"):
                raise RuntimeError("Ascend HiSparse requires the Ascend swap_blocks_batch operator.")
        if not callable(HiSparseNPUOps):
            raise RuntimeError("Ascend HiSparse device kernels are unavailable.")

    def pin_tensor(self, tensor: torch.Tensor) -> None:
        from vllm_ascend.ops.hisparse import MappedHostTensor

        tensor._hisparse_registration = MappedHostTensor(tensor)

    def unpin_tensor(self, tensor: torch.Tensor) -> None:
        registration = tensor._hisparse_registration
        registration.close()
        del tensor._hisparse_registration

    def create_cache_ops(self, runtime: HiSparseRuntime) -> Any:
        from vllm_ascend.ops.hisparse import HiSparseNPUOps, ResolverWorkspace

        group = runtime.index_group
        if group.backend_state is None:
            group.backend_state = ResolverWorkspace(
                max_rows=runtime.max_swap_rows,
                hot_size=runtime.region_stride,
                max_union_rows=runtime.max_union_rows,
                device=runtime.device,
            )
        mapping = runtime.registered_host_pool._hisparse_registration
        return HiSparseNPUOps(mapping, group.backend_state)

    def create_stream(self, device: torch.device, *, generic: bool = False) -> Any:
        return torch.npu.Stream(device=device)

    def create_lru_init(self, size: int, device: torch.device) -> torch.Tensor:
        # CANN Range does not support int16 outputs. This runs only at allocation.
        return torch.arange(size, dtype=torch.int32, device=device).to(torch.int16)

    def create_event(self) -> Any:
        return torch.npu.Event()

    def current_stream(self) -> Any:
        # HiSparse switches between model and copy streams. Query the active
        # stream here: the generic Ascend helper caches its first result.
        return torch.npu.current_stream()

    def stream_context(self, stream: Any, *, generic: bool = False) -> Any:
        return torch.npu.stream(stream)

    def is_current_stream_capturing(self) -> bool:
        return torch.npu.is_current_stream_capturing()

    def synchronize(self) -> None:
        torch.npu.synchronize()

    def swap_blocks_batch(self, src: torch.Tensor, dst: torch.Tensor, sizes: torch.Tensor) -> None:
        torch.ops._C_ascend.swap_blocks_batch(src, dst, sizes, 1)
