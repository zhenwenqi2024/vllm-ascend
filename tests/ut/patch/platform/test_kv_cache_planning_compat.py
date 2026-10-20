# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exercise the current Core builder signatures through Ascend's hooks."""

from types import SimpleNamespace

import torch
from vllm.v1.core import kv_cache_planning
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheLayout, MLAAttentionSpec

from vllm_ascend.patch.platform import patch_kv_cache_utils  # noqa: F401


def _config_and_groups():
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC,
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=None,
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
        model_config=SimpleNamespace(max_model_len=17),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        speculative_config=None,
        kv_transfer_config=None,
    )
    spec = MLAAttentionSpec(block_size=4, num_kv_heads=1, head_size=8, dtype=torch.bfloat16)
    return config, [KVCacheGroupSpec(["layer"], spec)]


def test_default_builder_materializes_exact_requested_block_count():
    config, groups = _config_and_groups()
    builder = kv_cache_planning.DefaultKVCacheConfigBuilder()

    cache = builder.get_kv_cache_config_from_groups(config, groups, num_blocks=7)

    assert cache.num_blocks == 7
    assert len(cache.kv_cache_tensors) == 1
    assert cache.kv_cache_tensors[0].size == 7 * groups[0].kv_cache_spec.page_size_bytes


def test_memory_accounting_preserves_selected_builder():
    config, groups = _config_and_groups()

    class SplitAllocationBuilder(kv_cache_planning.DefaultKVCacheConfigBuilder):
        def _get_pool_bytes_per_block(self, vllm_config, kv_cache_groups):
            return 3 * kv_cache_groups[0].kv_cache_spec.page_size_bytes

    # 17 tokens occupy five four-token pages. Core must ask this builder for
    # its three-allocation footprint rather than assume the default layout.
    required = kv_cache_planning._max_memory_usage_bytes_from_groups(SplitAllocationBuilder(), config, groups)

    assert required == 5 * 3 * groups[0].kv_cache_spec.page_size_bytes
