# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.quantization.methods.w4a8 import w4a8


@pytest.mark.parametrize("a5", [False, True])
def test_w4a8_fused_activation_hardware_scope(a5):
    method = object.__new__(w4a8.AscendW4A8DynamicFusedMoEMethod)
    method.use_a5_split_gmm = a5
    assert method.supports_fused_activation(MoEActivation.SILU) is not a5
    assert not method.supports_fused_activation(MoEActivation.GELU)


@pytest.mark.parametrize(
    ("quant_method", "eplb", "message"),
    [(w4a8.COMPRESSED_TENSORS_METHOD, False, "msModelSlim"), (w4a8.ASCEND_QUANTIZATION_METHOD, True, "EPLB")],
)
def test_a5_unverified_weight_layouts_fail_before_loading(quant_method, eplb, message):
    config = SimpleNamespace(
        quant_config=SimpleNamespace(quant_description={}, get_name=lambda: quant_method),
        parallel_config=SimpleNamespace(enable_expert_parallel=True, enable_eplb=eplb),
        use_v2_model_runner=True,
    )
    with (
        patch.object(w4a8, "get_current_vllm_config", return_value=config),
        patch.object(w4a8, "get_ascend_device_type", return_value=w4a8.AscendDeviceType.A5),
        pytest.raises(ValueError, match=message),
    ):
        w4a8.AscendW4A8DynamicFusedMoEMethod()
