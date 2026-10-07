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


def make_dspark_config():
    config = make_config(True, 0)
    config.speculative_config = SimpleNamespace(
        method="dspark",
        num_speculative_tokens=8,
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["Glm5DSparkForCausalLM"],
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                block_size=8,
                sample_from_anchor=True,
            )
        ),
    )
    return config


@pytest.mark.parametrize("hot_tokens", [18432, 18688, 32512])
def test_v2_glm_mla_dspark_accepts_verified_nine_row_budget(hot_tokens):
    config = SparseKVOffloadConfig.from_additional_config(
        make_dspark_config(),
        {"enabled": True, "fused_op_type": "fused_copy_sfa", "topk_buffer_size": hot_tokens},
    )
    assert config.use_fused_copy_sfa


@pytest.mark.parametrize("hot_tokens", [8192, 16384, 18433, 32768])
def test_dspark_rejects_undersized_unaligned_or_kernel_overflow_budget(hot_tokens):
    with pytest.raises(ValueError, match="hot budget"):
        SparseKVOffloadConfig.from_additional_config(
            make_dspark_config(),
            {"enabled": True, "fused_op_type": "fused_copy_sfa", "topk_buffer_size": hot_tokens},
        )


@pytest.mark.parametrize("field,value", [("block_size", 7), ("sample_from_anchor", False), ("kv_lora_rank", None)])
def test_dspark_rejects_unverified_or_gqa_checkpoint(field, value):
    config = make_dspark_config()
    setattr(config.speculative_config.draft_model_config.hf_config, field, value)
    with pytest.raises(ValueError, match="outside V2 GLM MLA"):
        SparseKVOffloadConfig.from_additional_config(
            config, {"enabled": True, "fused_op_type": "fused_copy_sfa", "topk_buffer_size": 18432}
        )


@pytest.mark.parametrize("v2,draft_tokens", [(False, 8), (True, 7), (True, 9)])
def test_dspark_rejects_unverified_runner_or_width(v2, draft_tokens):
    config = make_dspark_config()
    config.use_v2_model_runner = v2
    config.speculative_config.num_speculative_tokens = draft_tokens
    with pytest.raises(ValueError, match="outside V2 GLM MLA"):
        SparseKVOffloadConfig.from_additional_config(
            config, {"enabled": True, "fused_op_type": "fused_copy_sfa", "topk_buffer_size": 18432}
        )
