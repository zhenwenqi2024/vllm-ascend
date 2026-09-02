# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm_ascend.distributed.kda_state_parallel import (
    KDAStateCopyPlan,
    _copy_owner_state,
    flatten_kda_state_cache,
    make_kda_copy_plan_callback,
    remap_kda_metadata_state_indices,
)


def _metadata(indices: torch.Tensor):
    conv = SimpleNamespace(cache_indices=indices)
    return SimpleNamespace(
        spec_state_indices_tensor=indices,
        non_spec_state_indices_tensor=indices,
        prefill_state_indices=indices,
        non_spec_prefill_metadata=SimpleNamespace(causal_conv1d=conv),
        non_spec_decode_metadata=SimpleNamespace(causal_conv1d=conv),
        spec_decode_metadata=SimpleNamespace(spec_causal_conv1d=SimpleNamespace(cache_indices=indices)),
    )


def test_remap_kda_metadata_preserves_pad_slot_and_owner_namespace():
    metadata = _metadata(torch.tensor([0, 7, -1], dtype=torch.int32))

    remapped = remap_kda_metadata_state_indices(
        metadata,
        owner_rank=2,
        parallel_size=4,
    )

    expected = torch.tensor([2, 30, -1], dtype=torch.int32)
    torch.testing.assert_close(remapped.non_spec_state_indices_tensor, expected)
    torch.testing.assert_close(
        remapped.non_spec_prefill_metadata.causal_conv1d.cache_indices,
        expected,
    )
    torch.testing.assert_close(
        remapped.spec_decode_metadata.spec_causal_conv1d.cache_indices,
        expected,
    )


def test_kda_copy_plan_uses_distinct_conv_and_temporal_sources():
    plans = []
    callback = make_kda_copy_plan_callback(plans)
    req_state = SimpleNamespace(block_ids={3: [4, 8, 12, 16]})

    callback([3], 0, 3, 2, req_state)

    assert plans == [
        KDAStateCopyPlan(
            mamba_group_id=3,
            conv_src_block_id=4,
            temporal_src_block_id=12,
            dest_block_id=16,
            accepted_token_bias=2,
        )
    ]


def test_owner_state_copy_does_not_touch_other_owner_slots():
    state = torch.arange(4 * 2 * 3 * 2).reshape(4, 2, 3, 2).clone()
    untouched = state[:, 0].clone()
    plan = KDAStateCopyPlan(
        mamba_group_id=0,
        conv_src_block_id=1,
        temporal_src_block_id=1,
        dest_block_id=3,
        accepted_token_bias=1,
    )

    _copy_owner_state(state, 1, plan, is_conv_state=True)

    torch.testing.assert_close(state[:, 0], untouched)
    torch.testing.assert_close(state[3, 1, :2], state[1, 1, 1:])


def test_flatten_kda_state_cache_keeps_block_owner_order():
    state = torch.arange(3 * 2 * 4).reshape(3, 2, 4)

    flattened = flatten_kda_state_cache(state, parallel_size=2)

    assert flattened.shape == (6, 4)
    torch.testing.assert_close(flattened[5], state[2, 1])
    assert flattened.data_ptr() == state.data_ptr()
