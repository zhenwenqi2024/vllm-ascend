# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import ast
import inspect
import textwrap
from collections import Counter
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.attention.dsa_v1 import AscendDSAImpl
from vllm_ascend.device.device_op import DeviceOperator


def _record_stream_targets(owner: type) -> Counter[tuple[str, str]]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(owner)))
    targets = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "record_stream"
            and isinstance(node.func.value, ast.Name)
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
        ):
            continue
        targets.append((node.func.value.id, node.args[0].id))
    return Counter(targets)


def test_dsa_declares_all_cross_stream_lifetime_registrations():
    assert _record_stream_targets(AscendDSAImpl) == Counter(
        {
            ("hidden_states", "aux_stream"): 1,
            ("weights_proj_output", "main_stream"): 2,
            ("kv", "aux_stream"): 1,
        }
    )


def test_mla_prolog_records_hidden_states_on_aux_stream():
    impl = AscendDSAImpl.__new__(AscendDSAImpl)
    hidden_states = torch.ones((2, 2))
    q_quant = torch.ones((2, 2))
    q_scale = torch.ones((2,))
    kv_quant = torch.ones((2, 2))
    kv_scale = torch.ones((2,))
    qr = torch.ones((2, 2))
    kv = torch.ones((2, 2))
    q = torch.ones((2, 2))
    impl.cv_wq_a = SimpleNamespace(
        quantize=MagicMock(return_value=(q_quant, q_scale)),
        matmul=MagicMock(return_value=qr),
    )
    impl.cv_wkv = SimpleNamespace(
        quantize=MagicMock(return_value=(kv_quant, kv_scale)),
        matmul=MagicMock(return_value=kv),
    )
    impl.cv_wq_b = SimpleNamespace(matmul=MagicMock(return_value=q))
    impl.wq_b = MagicMock()
    impl.q_norm = MagicMock(return_value=qr)
    impl.kv_norm = MagicMock(return_value=kv)
    impl.q_norm_without_weight = MagicMock()
    impl.rope_head_dim = 1
    impl.nope_head_dim = 1
    impl.head_dim = 2
    impl.n_local_heads = 1
    impl.eps = 1e-6
    main_stream = MagicMock()
    aux_stream = MagicMock()

    with (
        patch("vllm_ascend.attention.dsa_v1._is_w8a8_dynamic", return_value=False),
        patch("vllm_ascend.attention.dsa_v1.dsv4_dsa_overlap_stream", return_value=aux_stream),
        patch(
            "vllm_ascend.attention.dsa_v1.npu_stream_switch",
            side_effect=lambda *_args, **_kwargs: nullcontext(),
        ),
        patch("vllm_ascend.attention.dsa_v1.torch.npu.current_stream", return_value=main_stream),
        patch.object(torch.Tensor, "record_stream") as record_stream,
        patch.object(
            torch.ops._C_ascend,
            "inplace_partial_rotary_mul",
            create=True,
        ),
        patch.object(DeviceOperator, "dsa_kv_compress_scatter"),
        patch.object(DeviceOperator, "apply_dsa_q_rms", side_effect=lambda value, *_args: value),
    ):
        impl._mla_prolog_multistream(
            hidden_states,
            cos=torch.ones((2, 1, 1, 1)),
            sin=torch.zeros((2, 1, 1, 1)),
            swa_kv_cache=torch.empty(0),
            slot_mapping=torch.zeros((2, 2), dtype=torch.int32),
        )

    record_stream.assert_called_once_with(aux_stream)
