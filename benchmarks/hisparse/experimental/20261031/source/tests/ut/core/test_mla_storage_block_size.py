# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch
from vllm.v1.kv_cache_interface import MLAAttentionSpec, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
    AscendSlidingWindowMLASpec,
    get_storage_block_size,
)


@pytest.mark.parametrize("spec_type", [MLAAttentionSpec, AscendMLAAttentionSpec])
@pytest.mark.parametrize("compression", [1, 4])
def test_mla_storage_uses_num_states(spec_type, compression):
    spec = spec_type(block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16, tokens_per_state=compression)
    assert get_storage_block_size(spec) == 128 // compression
    assert (
        get_storage_block_size(UniformTypeKVCacheSpecs(block_size=128, kv_cache_specs={"layer": spec}))
        == 128 // compression
    )


def test_sfa_indexer_storage_rows_match_page_size():
    spec = AscendSFAIndexerCacheSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    rows = get_storage_block_size(spec)
    assert rows == 128
    assert get_storage_block_size(UniformTypeKVCacheSpecs(block_size=128, kv_cache_specs={"indexer": spec})) == rows
    assert spec.real_page_size_bytes == rows * spec.num_kv_heads * spec.head_size * 2


def test_sliding_window_mla_preserves_custom_storage_compression():
    spec = AscendSlidingWindowMLASpec(
        block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16, sliding_window=512, compress_ratio=4
    )
    assert get_storage_block_size(spec) == 32
    assert spec.real_page_size_bytes == 32 * 128 * 2
