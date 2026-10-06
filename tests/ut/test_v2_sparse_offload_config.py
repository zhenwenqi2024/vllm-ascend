# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm_ascend.ascend_config import SparseKVOffloadConfig


def make_config(v2, mtp):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_topk=2048)),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        kv_transfer_config=SimpleNamespace(is_kv_consumer=True),
        use_v2_model_runner=v2,
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=mtp) if mtp else None,
    )


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("mtp", [0, 1, 2, 3])
def test_both_runners_accept_sparse_fused_mtp(v2, mtp):
    config = SparseKVOffloadConfig.from_additional_config(
        make_config(v2, mtp),
        {"enabled": True, "fused_op_type": "fused_copy_sfa", "topk_buffer_size": 8192},
    )
    assert config.enabled
    assert config.use_fused_copy_sfa


@pytest.mark.parametrize("v2", [False, True])
def test_decode_pp_remains_rejected(v2):
    config = make_config(v2, 2)
    config.parallel_config.pipeline_parallel_size = 2
    with pytest.raises(ValueError):
        SparseKVOffloadConfig.from_additional_config(config, {"enabled": True})


@pytest.mark.parametrize("v2", [False, True])
def test_producer_offload_remains_rejected(v2):
    config = make_config(v2, 2)
    config.kv_transfer_config.is_kv_consumer = False
    with pytest.raises(AssertionError):
        SparseKVOffloadConfig.from_additional_config(config, {"enabled": True})
