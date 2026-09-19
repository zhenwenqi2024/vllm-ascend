# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small real-NPU calibration checks; no model download or timing speedup claim."""

import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")

import torch_npu
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.eplb.diagnostics.calibration import measure_callable
from vllm_ascend.eplb.diagnostics.model_calibration import (
    calibrate_layer,
    capture_template,
)
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import enable_custom_op


def _small_mlp(quant_name, group_type, runner):
    experts, hidden, intermediate, capacity = 4, 128, 128, 64
    layer = torch.nn.Module()
    quantized = quant_name == "W8A8"
    dtype = torch.int8 if quantized else torch.bfloat16
    layer.w13_weight = torch.full((experts, hidden, 2 * intermediate), 1, dtype=dtype, device="npu")
    layer.w2_weight = torch.full((experts, intermediate, hidden), 1, dtype=dtype, device="npu")
    if quantized:
        torch.npu.config.allow_internal_format = True
        assert enable_custom_op(), "W8A8 fused calibration requires compiled Ascend custom ops"
        layer.w13_weight = torch_npu.npu_format_cast(layer.w13_weight, 29)  # FRACTAL_NZ
        layer.w2_weight = torch_npu.npu_format_cast(layer.w2_weight, 29)
        layer.w13_weight_scale_fp32 = torch.full((experts, 2 * intermediate), 1 / 128, device="npu")
        layer.w2_weight_scale = torch.full((experts, hidden), 1 / 128, dtype=torch.bfloat16, device="npu")
        # Exercise real compute hooks without initializing serving/communication.
        method = AscendW8A8DynamicFusedMoEMethod.__new__(AscendW8A8DynamicFusedMoEMethod)
        method.use_expert_weight_list = True
        for source, destination in (
            ("w13_weight", "w13_weight_list"),
            ("w2_weight", "w2_weight_list"),
            ("w13_weight_scale_fp32", "w13_weight_scale_fp32_list"),
            ("w2_weight_scale", "w2_weight_scale_list"),
        ):
            setattr(layer, destination, [weight.clone() for weight in getattr(layer, source).unbind(0)])
            delattr(layer, source)
    else:
        method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
        torch.nn.Module.__init__(method)
        method.moe = SimpleNamespace(has_bias=False)
        method._lora_routing = None
        method.lora_context = None
        if runner == "mrv1_list":
            layer.w13_weight_list = [weight.clone() for weight in layer.w13_weight.unbind(0)]
            layer.w2_weight_list = [weight.clone() for weight in layer.w2_weight.unbind(0)]
            del layer.w13_weight, layer.w2_weight
    method.dynamic_eplb = runner != "mrv2"
    weights = method.get_mlp_weights(layer)
    if not quantized:
        weights = MoEWeights(w1=weights[0], w2=weights[1])
    groups = [48, 8, 4, 4] if group_type == 1 else [48, 56, 60, 64]
    payload = MoEMlpComputeInput(
        hidden_states=torch.ones((capacity, hidden), dtype=dtype, device="npu"),
        group_list=torch.tensor(groups, dtype=torch.int64, device="npu"),
        group_list_type=group_type,
        dynamic_scale=torch.full((capacity,), 1 / 128, device="npu") if quantized else None,
        topk_scales=None,
        weights=weights,
        quant=MoEQuantParams(quant_type=getattr(QuantType, quant_name)),
        fusion=quantized,
        activation=MoEActivation.SILU,
        layer=layer,
        dynamic_eplb=quantized or method.dynamic_eplb,
        topk_ids=torch.zeros((8, 2), dtype=torch.int32, device="npu"),
    )
    return layer, method, payload


def _assert_interval(interval, repeats):
    assert interval["samples"] == repeats
    assert all(math.isfinite(interval[key]) for key in ("min_ms", "median_ms", "max_ms"))
    assert 0 <= interval["min_ms"] <= interval["median_ms"] <= interval["max_ms"]
    assert interval["max_ms"] > 0


@pytest.mark.parametrize(
    "quant_name,runner",
    [("NONE", "mrv1"), ("NONE", "mrv1_list"), ("NONE", "mrv2"), ("W8A8", "mrv1"), ("W8A8", "mrv2")],
)
@pytest.mark.parametrize("group_type", [0, 1])
def test_model_calibration_real_mlp_preserves_live_state(quant_name, runner, group_type):
    torch.npu.set_device(0)
    layer, method, payload = _small_mlp(quant_name, group_type, runner)
    weight_views = [
        tensor
        for view in method.get_eplb_weight_views(layer)
        for tensor in (view if isinstance(view, list) else [view])
    ]
    originals = [weight.cpu().clone() for weight in weight_views]
    source_hidden = payload.hidden_states.cpu().clone()
    source_routes = payload.topk_ids.cpu().clone()
    source_pointer = payload.hidden_states.data_ptr()
    old_context = get_forward_context() if is_forward_context_available() else None
    probe = SimpleNamespace(calibration_templates={}, eplb_enabled=True)
    assert capture_template(probe, payload, method, "MC2CommImpl") is None
    # Repeat real metadata capture to exercise stable quant dataclass equality.
    assert capture_template(probe, payload, method, "MC2CommImpl") is None
    # The forward hook also runs during graph construction. It must capture
    # metadata without device synchronization, counters, or timing events.
    graph_input = torch.ones(16, device="npu")
    graph_output = torch.empty_like(graph_input)
    torch.add(graph_input, 1, out=graph_output)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        capture_reason = capture_template(probe, payload, method, "MC2CommImpl")
        torch.add(graph_input, 1, out=graph_output)
    assert capture_reason is None
    graph.replay()
    torch.npu.synchronize()
    result, reason = calibrate_layer(
        probe,
        baseline_counts=[[96, 16, 8, 8], [24, 4, 2, 2]],
        candidate_counts=[[32, 32, 32, 32], [8, 8, 8, 8]],
        comm_name="MC2CommImpl",
        repeats=2,
    )
    assert reason is None, reason
    assert result["scope"] == "isolated_eplb_on_mlp"
    assert result["dynamic_eplb_kernel_flag"] is (quant_name == "W8A8" or runner != "mrv2")
    assert result["communication_delta"] == result["graph_delta"] == "unknown"
    assert 0 < result["estimated_scratch_bytes"] < 128 * 1024 * 1024
    for interval in result["baseline"] + result["candidate"]:
        _assert_interval(interval, 2)
    assert (get_forward_context() if is_forward_context_available() else None) is old_context
    for actual, original in zip(weight_views, originals):
        torch.testing.assert_close(actual.cpu(), original, rtol=0, atol=0)
    # W8A8 disposes synthetic activation storage every repetition. Source
    # activations and route IDs must remain intact after all real kernel calls.
    assert payload.hidden_states.data_ptr() == source_pointer
    torch.testing.assert_close(payload.hidden_states.cpu(), source_hidden, rtol=0, atol=0)
    torch.testing.assert_close(payload.topk_ids.cpu(), source_routes, rtol=0, atol=0)
    if quant_name == "NONE":
        assert method._lora_routing is None


@pytest.mark.parametrize("quant_name", ["NONE", "W8A8"])
def test_model_calibration_empty_local_rank(quant_name):
    torch.npu.set_device(0)
    layer, method, payload = _small_mlp(quant_name, 1, "mrv2")
    probe = SimpleNamespace(calibration_templates={}, eplb_enabled=True)
    assert capture_template(probe, payload, method, "MC2CommImpl") is None
    result, reason = calibrate_layer(probe, [[0, 0, 0, 0]], [[1, 0, 0, 0]], "MC2CommImpl", repeats=2)
    assert reason is None, reason
    for interval in result["baseline"] + result["candidate"]:
        _assert_interval(interval, 2)
    assert layer is payload.layer  # Keep the weakly referenced model alive.


def test_measure_callable_times_real_graph_replay():
    torch.npu.set_device(0)
    completed = []

    def factory():
        x = torch.ones(4096, device="npu")
        output = torch.empty_like(x)
        torch.add(x, 2, out=output)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            torch.add(x, 2, out=output)
        output.zero_()

        def replay():
            graph.replay()
            completed.append((x, output))  # Retain each tiny result until checked.
            return output

        return replay

    interval = measure_callable(factory, repeats=3, warmup=1)
    _assert_interval(interval, 3)
    assert len(completed) == 4
    for original, output in completed:
        torch.testing.assert_close(original.cpu(), torch.ones(4096), rtol=0, atol=0)
        torch.testing.assert_close(output.cpu(), torch.full((4096,), 3.0), rtol=0, atol=0)
