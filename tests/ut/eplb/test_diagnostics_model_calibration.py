# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract checks; synthetic timings here are never NPU validation evidence."""

import gc
import importlib
import sys
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from weakref import ref

import pytest
import torch


@pytest.fixture(scope="module")
def api():
    stubbed = "vllm_ascend" not in sys.modules and importlib.util.find_spec("vllm") is None
    original = set(sys.modules)
    if stubbed:
        package = ModuleType("vllm_ascend")
        package.__path__ = [str(Path(__file__).resolve().parents[3] / "vllm_ascend")]
        sys.modules["vllm_ascend"] = package
    yield importlib.import_module("vllm_ascend.eplb.diagnostics.model_calibration")
    if stubbed:
        for name in set(sys.modules) - original:
            if name == "vllm_ascend" or name.startswith("vllm_ascend."):
                del sys.modules[name]


@dataclass
class Weights:
    w1: object
    w2: object


@dataclass
class Payload:
    hidden_states: object
    group_list: object
    layer: object
    weights: object
    quant: object
    dynamic_scale: object = None
    group_list_type: int = 1
    topk_scales: object = None
    fusion: bool = False
    activation: str = "silu"
    need_trans: bool = False
    dynamic_eplb: bool = False
    activation_situ_beta: float | None = None
    activation_situ_linear_beta: float | None = None
    swiglu_limit: float = 0
    swiglu_alpha: float = 1
    swiglu_beta: float = 0
    expanded_row_idx: object = None
    topk_ids: object = None
    lora_context: object = None


class AscendUnquantizedFusedMoEMethod:
    _lora_routing = "unchanged"

    def get_mlp_weights(self, layer):
        return layer.w1, layer.w2


class AscendW8A8DynamicFusedMoEMethod:
    def get_mlp_weights(self, layer):
        return Weights(layer.w1, layer.w2)


def sample(quant="NONE"):
    layer = torch.nn.Module()
    layer.w1, layer.w2 = torch.ones(2, 4, 8), torch.ones(2, 8, 4)
    method = AscendUnquantizedFusedMoEMethod() if quant == "NONE" else AscendW8A8DynamicFusedMoEMethod()
    payload = Payload(
        torch.ones(8, 4, dtype=torch.float16 if quant == "NONE" else torch.int8),
        torch.tensor([4, 4], dtype=torch.int64),
        layer,
        Weights(layer.w1, layer.w2),
        SimpleNamespace(quant_type=SimpleNamespace(name=quant)),
        dynamic_scale=None if quant == "NONE" else torch.ones(8, 1),
    )
    return SimpleNamespace(calibration_templates={}), payload, method


@pytest.mark.parametrize("quant", ["NONE", "W8A8"])
def test_capture_retains_metadata_and_weak_model_refs_only(api, quant):
    probe, payload, method = sample(quant)
    hidden_ref, group_ref = ref(payload.hidden_states), ref(payload.group_list)
    layer = payload.layer
    assert api.capture_template(probe, payload, method, "MC2CommImpl") is None
    template = probe.calibration_templates["MC2CommImpl"]
    assert template.layer_ref() is layer
    assert template.method_ref() is method
    assert not any(isinstance(value, torch.Tensor) for value in vars(template).values())
    del payload
    gc.collect()
    assert hidden_ref() is None
    assert group_ref() is None
    del layer
    gc.collect()
    assert template.layer_ref() is None


@pytest.mark.parametrize("quant", ["NONE", "W8A8"])
@pytest.mark.parametrize("group_type", [0, 1])
def test_fresh_inputs_preserve_group_mode_and_weight_identity(api, quant, group_type):
    probe, payload, method = sample(quant)
    payload.group_list_type = group_type
    assert api.capture_template(probe, payload, method, "AllGatherCommImpl") is None
    template = probe.calibration_templates["AllGatherCommImpl"]
    first = api._new_input(template, [2, 3], payload.layer, method)
    second = api._new_input(template, [2, 3], payload.layer, method)
    assert first.hidden_states.shape == (5, 4)
    assert first.hidden_states.data_ptr() != second.hidden_states.data_ptr()
    assert first.weights.w1 is payload.layer.w1
    assert first.weights.w2 is payload.layer.w2
    assert first.group_list.tolist() == ([2, 5] if group_type == 0 else [2, 3])
    assert first.dynamic_eplb is False
    assert first.topk_ids is first.topk_scales is first.expanded_row_idx is None
    if quant == "W8A8":
        assert first.dynamic_scale.shape == (5, 1)
        assert first.dynamic_scale.data_ptr() != second.dynamic_scale.data_ptr()
    first.hidden_states.resize_(0)  # Real W8A8 kernels may dispose their input.
    assert second.hidden_states.shape == (5, 4)
    assert payload.hidden_states.shape == (8, 4)


def test_context_is_restored_and_method_state_is_not_mutated(api, monkeypatch):
    probe, payload, method = sample()
    api.capture_template(probe, payload, method, "MC2CommImpl")
    context = SimpleNamespace(moe_comm_type="previous")
    captured = []

    def apply_mlp(item, copied_method):
        assert context.moe_comm_type == "mc2"
        assert copied_method is not method
        copied_method._lora_routing = "temporary"
        captured.append(item)
        return item.hidden_states + 1, None

    def measure(factory, repeats, warmup):
        for _ in range(repeats + warmup):
            factory()()
        return {"min_ms": 1.0, "median_ms": 2.0, "max_ms": 3.0, "samples": repeats}

    monkeypatch.setattr(
        api, "_runtime", lambda: (apply_mlp, context, SimpleNamespace(MC2="mc2"), measure, nullcontext())
    )
    result, reason = api.calibrate_layer(probe, [[7, 1], [0, 0]], [[4, 4], [0, 0]], "MC2CommImpl", repeats=2)
    assert reason is None
    assert len(result["baseline"]) == len(result["candidate"]) == 2
    assert result["scope"] == "isolated_eplb_off_mlp"
    assert result["on_kernel_delta"] == result["communication_delta"] == result["graph_delta"] == "unknown"
    assert len(captured) == 12
    assert len({id(item.hidden_states) for item in captured}) == 12
    assert context.moe_comm_type == "previous"
    assert method._lora_routing == "unchanged"
    assert torch.equal(payload.layer.w1, torch.ones_like(payload.layer.w1))

    def failing_measure(*args, **kwargs):
        raise RuntimeError("kernel unsupported")

    monkeypatch.setattr(
        api, "_runtime", lambda: (apply_mlp, context, SimpleNamespace(MC2="mc2"), failing_measure, nullcontext())
    )
    result, reason = api.calibrate_layer(probe, [[7, 1]], [[4, 4]], "MC2CommImpl")
    assert result is None
    assert reason == "mlp_calibration_failed:RuntimeError"
    assert context.moe_comm_type == "previous"


@pytest.mark.parametrize("field_name", ["lora_context", "topk_scales"])
def test_routing_dependent_path_fails_closed(api, field_name):
    probe, payload, method = sample()
    setattr(payload, field_name, object())
    assert api.capture_template(probe, payload, method, "MC2CommImpl") == "routing_dependent_mlp"
    result, reason = api.calibrate_layer(probe, [[7, 1]], [[4, 4]], "MC2CommImpl")
    assert result is None
    assert reason == "routing_dependent_mlp"


def test_variant_templates_and_excessive_shapes_fail_closed(api):
    probe, payload, method = sample()
    api.capture_template(probe, payload, method, "MC2CommImpl")
    result, reason = api.calibrate_layer(probe, [[7, 1]], [[9, 0]], "MC2CommImpl")
    assert result is None
    assert reason == "calibration_shape_exceeds_observed_capacity"
    larger = replace(payload, hidden_states=torch.ones(16, 4, dtype=torch.float16))
    assert api.capture_template(probe, larger, method, "MC2CommImpl") is None
    assert probe.calibration_templates["MC2CommImpl"].row_capacity == 16
    assert api.capture_template(probe, replace(payload, fusion=True), method, "MC2CommImpl") == "ambiguous_mlp_template"


def test_payload_bytes_include_scales_and_deduplicate_aliases(api, monkeypatch):
    probe, payload, method = sample("W8A8")
    scale = torch.ones(2, 8, dtype=torch.float32)
    api.capture_template(probe, payload, method, "MC2CommImpl")
    monkeypatch.setattr(
        method,
        "get_eplb_weight_views",
        lambda layer: [layer.w1, layer.w2, scale, layer.w1.view_as(layer.w1), list(scale.unbind(0))],
        raising=False,
    )
    size, reason = api.expert_payload_bytes(probe, "MC2CommImpl")
    assert reason is None
    assert size == (4 * 8 + 8 * 4 + 8) * 4
    monkeypatch.setattr(method, "get_eplb_weight_views", lambda layer: [[torch.ones(2), torch.ones(3)]])
    size, reason = api.expert_payload_bytes(probe, "MC2CommImpl")
    assert size is None
    assert reason == "nonuniform_expert_payload"
