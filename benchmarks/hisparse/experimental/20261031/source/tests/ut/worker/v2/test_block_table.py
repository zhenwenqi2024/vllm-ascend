# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import ModuleType

import torch

from vllm_ascend.patch.worker.patch_v2 import patch_uva
from vllm_ascend.worker.v2.block_table import AscendBlockTables


def _allow_cpu_buffers(monkeypatch) -> None:
    original_zeros = torch.zeros

    def zeros_without_pinning(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_zeros(*args, **kwargs)

    # CPU CI cannot allocate Ascend pinned memory. Only substitute that device
    # facility; the production AscendBlockTables constructor remains real.
    monkeypatch.setattr(torch, "zeros", zeros_without_pinning)
    monkeypatch.setattr(patch_uva, "is_uva_available", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.vllm_ascend_C",
        ModuleType("vllm_ascend.vllm_ascend_C"),
    )
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "get_npu_view_from_cpu_tensor",
        lambda tensor: tensor,
        raising=False,
    )


def test_encoder_only_zero_cache_groups_initialize_and_compute(monkeypatch) -> None:
    """Exercise the real block-table construction used by an encoder producer."""
    _allow_cpu_buffers(monkeypatch)
    block_tables = AscendBlockTables(
        block_sizes=[],
        max_num_reqs=2,
        max_num_batched_tokens=8,
        max_num_blocks_per_group=[],
        device=torch.device("cpu"),
        kernel_block_sizes=[],
    )

    slot_mappings = block_tables.compute_slot_mappings(
        idx_mapping=torch.tensor([0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        positions=torch.tensor([0], dtype=torch.int64),
        num_tokens_padded=1,
    )

    assert block_tables.num_kv_cache_groups == 0
    assert block_tables._block_table_window_size == 1
    assert slot_mappings.shape == (0, 1)


def test_current_core_keeps_manager_ids_with_smaller_kernel_pages(monkeypatch) -> None:
    """The real Core parent must receive CP flags and retain unexpanded IDs."""
    _allow_cpu_buffers(monkeypatch)
    tables = AscendBlockTables(
        block_sizes=[128],
        max_num_reqs=2,
        max_num_batched_tokens=8,
        max_num_blocks_per_group=[4],
        device=torch.device("cpu"),
        kernel_block_sizes=[64],
        cp_size=2,
        cp_rank=1,
        cp_interleave=8,
        slot_mapping_enabled=[True],
        dcp_sharded=[False],
    )
    tables.append_block_ids(0, ([3, 5],), overwrite=True)

    assert (tables.cp_size, tables.cp_rank, tables.cp_interleave) == (2, 1, 8)
    assert tables.dcp_sharded.tolist() == [False]
    assert tables.block_tables[0].gpu.shape == (2, 4)
    assert tables.block_tables[0]._staged_write_contents == [3, 5]
    assert tables.num_blocks.np[0, 0] == 2
    assert tables.kernel_block_sizes == [64]
    assert tables.kernel_block_sizes_tensor.tolist() == [128]
