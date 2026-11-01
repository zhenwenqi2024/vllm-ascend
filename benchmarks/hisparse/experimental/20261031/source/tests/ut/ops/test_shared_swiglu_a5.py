# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.ops.fused_moe import shared_experts as module
from vllm_ascend.ops.fused_moe.dataclass.shared_experts import RoutedMoEMilestones
from vllm_ascend.utils import AscendDeviceType


@pytest.mark.parametrize(
    "device,limit,alpha,beta,use_standard",
    [
        (AscendDeviceType.A5, 0.0, 1.0, 0.0, True),
        (AscendDeviceType.A2, 0.0, 1.0, 0.0, False),
        (AscendDeviceType.A5, 7.0, 1.0, 0.0, False),
        (AscendDeviceType.A5, 0.0, 1.702, 0.0, False),
        (AscendDeviceType.A5, 0.0, 1.0, 1.0, False),
    ],
)
def test_standard_swiglu_keeps_other_contracts(monkeypatch, device, limit, alpha, beta, use_standard):
    executor = module.AscendSharedExperts.__new__(module.AscendSharedExperts)
    executor.multistream_overlap = False
    executor.situ_activation = None
    executor.swiglu_limit, executor.swiglu_alpha, executor.swiglu_beta = limit, alpha, beta
    executor.layer = SimpleNamespace(
        gate_up_proj=SimpleNamespace(weight_scale_fp32=torch.ones(4)),
        down_proj=SimpleNamespace(weight=torch.ones(2, 4), weight_scale=torch.ones(4)),
    )
    q, scale = torch.ones(1, 2, dtype=torch.int8), torch.ones(1)
    standard, variant = MagicMock(return_value=(q, scale)), MagicMock(return_value=(q, scale))
    output = torch.ones(1, 4)
    down = MagicMock(return_value=output)
    monkeypatch.setattr(module, "get_ascend_device_type", lambda: device)
    monkeypatch.setattr(module, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: True))
    monkeypatch.setattr(module.torch_npu, "npu_dequant_swiglu_quant", standard, raising=False)
    monkeypatch.setattr(module.torch.ops._C_ascend, "npu_dequant_swiglu_quant", variant, raising=False)
    monkeypatch.setattr(module.torch_npu, "npu_quant_matmul", down)
    submitted = module.SubmittedSharedExpertGateUp(
        torch.ones(1, 4),
        torch.ones(1, 4, dtype=torch.int32),
        torch.ones(1),
        None,
        module.SharedExpertMLPPath.A8_INT_FUSED,
    )
    assert executor._run_a8_int_mlp(submitted, RoutedMoEMilestones()) is output
    if use_standard:
        standard.assert_called_once()
        variant.assert_not_called()
        assert standard.call_args.kwargs["swiglu_mode"] == 0
        assert "clamp_limit" not in standard.call_args.kwargs
    else:
        variant.assert_called_once()
        standard.assert_not_called()
        assert variant.call_args.kwargs["clamp_limit"] == limit
        assert variant.call_args.kwargs["glu_alpha"] == alpha
        assert variant.call_args.kwargs["glu_bias"] == beta
    assert down.call_args.args[0] is q
    assert down.call_args.kwargs["pertoken_scale"] is scale
