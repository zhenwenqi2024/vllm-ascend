# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from vllm_ascend.ops.fused_moe.prepare_finalize import (
    PrepareAndFinalize,
    PrepareAndFinalizeWithAllGather,
)


def test_quant_stream_all_gather_records_input_and_output_lifetimes():
    moe_config = SimpleNamespace(
        dp_size=1,
        tp_size=1,
        pcp_size=1,
        ep_size=1,
        dp_group=MagicMock(),
        tp_group=MagicMock(),
        original_num_experts=8,
    )
    with patch(
        "vllm_ascend.ops.fused_moe.prepare_finalize.get_ascend_config",
        return_value=SimpleNamespace(multistream_overlap_gate=False),
    ):
        layer = PrepareAndFinalizeWithAllGather(moe_config)

    layer.multistream_overlap_gate = True
    hidden_states = torch.randn(3, 8)
    router_logits = torch.randn(3, 2)
    gathered_hidden_states = hidden_states + 1
    main_stream = MagicMock()
    quant_stream = MagicMock()

    with (
        patch.object(PrepareAndFinalize, "quant_stream", quant_stream),
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize.npu_stream_switch",
            side_effect=lambda *_args, **_kwargs: nullcontext(),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize.torch.npu.current_stream",
            return_value=main_stream,
        ),
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize.fc3_all_gather_and_maybe_unpad_impl",
            return_value=gathered_hidden_states,
        ),
        patch.object(torch.Tensor, "record_stream") as record_stream,
    ):
        result = layer._prepare_with_ep_group(hidden_states, router_logits)

    assert result.hidden_states is gathered_hidden_states
    assert record_stream.call_args_list == [call(quant_stream), call(main_stream)]
    main_stream.wait_stream.assert_called_once_with(quant_stream)
