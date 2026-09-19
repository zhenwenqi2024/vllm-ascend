# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated synthetic MLP calibration with immutable, already-loaded weights."""

from copy import copy
from dataclasses import dataclass
from itertools import accumulate
from numbers import Integral
from weakref import ref

import torch

_SUPPORTED_METHODS = {
    "NONE": "AscendUnquantizedFusedMoEMethod",
    "W8A8": "AscendW8A8DynamicFusedMoEMethod",
}
_COMM_TYPES = {"MC2CommImpl": "MC2", "AlltoAllCommImpl": "ALLTOALL", "AllGatherCommImpl": "ALLGATHER"}
_STATIC_FIELDS = (
    "group_list_type",
    "quant",
    "fusion",
    "activation",
    "need_trans",
    "dynamic_eplb",
    "activation_situ_beta",
    "activation_situ_linear_beta",
    "swiglu_limit",
    "swiglu_alpha",
    "swiglu_beta",
)


@dataclass(frozen=True)
class MlpTemplate:
    layer_ref: object
    method_ref: object
    input_class: type
    weights_class: type
    metadata: dict
    hidden_width: int
    hidden_dtype: torch.dtype
    device: torch.device
    scale_tail: tuple | None
    scale_dtype: torch.dtype | None
    group_dtype: torch.dtype
    slots: int
    quant_name: str


def capture_template(probe, mlp_input, quant_method, comm_name):
    """Capture shapes/flags only, including during graph construction.

    No tensor contents, routed activations, device counters, or timing events
    are retained. Weak references identify existing model weights/methods.
    Mixed compute layouts for one comm path fail closed.
    """
    if probe is None:
        return "missing_probe"
    templates = probe.calibration_templates
    previous = templates.get(comm_name)
    if isinstance(previous, str):
        return previous
    quant_name = getattr(mlp_input.quant.quant_type, "name", "unknown")
    reason = None
    if comm_name not in _COMM_TYPES or _SUPPORTED_METHODS.get(quant_name) != type(quant_method).__name__:
        reason = "unsupported_mlp_method"
    elif mlp_input.lora_context is not None or mlp_input.topk_scales is not None:
        reason = "routing_dependent_mlp"
    elif getattr(probe, "eplb_enabled", False) is not True:
        reason = "requires_eplb_enabled"
    elif mlp_input.layer is None:
        reason = "model_weights_unavailable"
    elif mlp_input.hidden_states.ndim != 2 or mlp_input.group_list.ndim != 1 or mlp_input.group_list_type not in (0, 1):
        reason = "unsupported_mlp_shape"
    scale = mlp_input.dynamic_scale
    if scale is not None and (scale.ndim not in (1, 2) or (scale.ndim == 2 and scale.shape[1] != 1)):
        reason = "unsupported_dynamic_scale_shape"
    if reason:
        templates[comm_name] = reason
        return reason
    template = MlpTemplate(
        ref(mlp_input.layer),
        ref(quant_method),
        type(mlp_input),
        type(mlp_input.weights),
        {key: getattr(mlp_input, key) for key in _STATIC_FIELDS},
        mlp_input.hidden_states.shape[1],
        mlp_input.hidden_states.dtype,
        mlp_input.hidden_states.device,
        None if scale is None else tuple(scale.shape[1:]),
        None if scale is None else scale.dtype,
        mlp_input.group_list.dtype,
        mlp_input.group_list.numel(),
        quant_name,
    )
    if previous is not None and previous != template:
        templates[comm_name] = "ambiguous_mlp_template"
        return "ambiguous_mlp_template"
    templates[comm_name] = template
    return None


def _new_input(template, counts, layer, method):
    rows = sum(counts)
    hidden = torch.full(
        (rows, template.hidden_width),
        1 if template.hidden_dtype == torch.int8 else 0.125,
        dtype=template.hidden_dtype,
        device=template.device,
    )
    groups = list(accumulate(counts)) if template.metadata["group_list_type"] == 0 else counts
    group_list = torch.tensor(groups, dtype=template.group_dtype, device=template.device)
    scale = None
    if template.scale_tail is not None:
        scale = torch.full((rows, *template.scale_tail), 1 / 128, dtype=template.scale_dtype, device=template.device)
    weights = method.get_mlp_weights(layer)
    if template.quant_name == "NONE":
        weights = template.weights_class(w1=weights[0], w2=weights[1])
    return template.input_class(
        hidden_states=hidden,
        group_list=group_list,
        dynamic_scale=scale,
        topk_scales=None,
        weights=weights,
        layer=layer,
        expanded_row_idx=None,
        topk_ids=None,
        lora_context=None,
        **template.metadata,
    )


def _runtime():
    # Keep imports inert on CPU and delay worker-only dependencies until RPC.
    from vllm.forward_context import ForwardContext, override_forward_context

    from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
    from vllm_ascend.eplb.diagnostics.calibration import measure_callable
    from vllm_ascend.ops.fused_moe.moe_mlp import apply_moe_mlp

    context = ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})
    return apply_moe_mlp, _EXTRA_CTX, MoECommType, measure_callable, override_forward_context(context)


def _matrix_shape(weight, slots, need_trans):
    """Read logical dimensions without copying dense or split expert weights."""
    if isinstance(weight, (list, tuple)):
        if len(weight) == 1 and isinstance(weight[0], torch.Tensor) and weight[0].ndim == 3:
            return _matrix_shape(weight[0], slots, need_trans)
        if len(weight) != slots or not weight or need_trans:
            raise ValueError("unsupported split expert weights")
        shape = tuple(weight[0].shape)
        if len(shape) != 2 or any(not isinstance(w, torch.Tensor) or tuple(w.shape) != shape for w in weight):
            raise ValueError("nonuniform split expert weights")
    elif isinstance(weight, torch.Tensor) and weight.ndim == 3 and weight.shape[0] == slots:
        shape = tuple(weight.shape[1:])
        if need_trans:
            shape = shape[::-1]
    else:
        raise ValueError("unsupported dense expert weights")
    return shape


def _estimated_scratch_bytes(template, rows, layer, method):
    weights = method.get_mlp_weights(layer)
    w1, w2 = weights if template.quant_name == "NONE" else (weights.w1, weights.w2)
    need_trans = template.metadata["need_trans"]
    width, up = _matrix_shape(w1, template.slots, need_trans)
    middle, output = _matrix_shape(w2, template.slots, need_trans)
    if width != template.hidden_width or min(width, up, middle, output) <= 0:
        raise ValueError("incompatible MLP dimensions")
    # Allow FP32 intermediates and simultaneous input/output buffers, including
    # non-fused activation/quantization branches. Internal CANN workspaces are
    # opaque, so this is an explicit estimate, not an allocator memory limit.
    return rows * (8 * (width + output) + 16 * (up + middle) + 32) + template.slots * 16


@torch.inference_mode()
def calibrate_layer(
    probe, baseline_counts, candidate_counts, comm_name, repeats=3, *, max_scratch_bytes=128 * 1024 * 1024
):
    """Time the previous/current actual layouts with the same EPLB-on kernels.

    Each repetition uses disposable synthetic activations and existing local
    weights without modifying them. The preserved runtime flags select the real
    ON MLP path, including MRv1/MRv2 W8A8 expert-weight-list kernels. Local counts
    can exceed the captured row count, subject to the explicit scratch estimate.
    This estimates the compute effect of shape changes, not output accuracy or
    dispatch/combine cost. It runs eagerly outside graph capture and excludes
    graph-launch differences. The caller must quiesce serving AND complete any
    outstanding EPLB weight updates before calling this collective RPC.
    """
    if getattr(probe, "eplb_enabled", False) is not True:
        return None, "requires_eplb_enabled"
    template = probe.calibration_templates.get(comm_name)
    if template is None or isinstance(template, str):
        return None, template or "missing_mlp_template"
    if (
        not baseline_counts
        or len(baseline_counts) != len(candidate_counts)
        or isinstance(repeats, bool)
        or not isinstance(repeats, Integral)
        or not 1 <= repeats <= 20
        or isinstance(max_scratch_bytes, bool)
        or not isinstance(max_scratch_bytes, Integral)
        or max_scratch_bytes <= 0
    ):
        return None, "invalid_calibration_samples"
    largest_rows = 0
    for counts in (*baseline_counts, *candidate_counts):
        if len(counts) != template.slots or any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in counts
        ):
            return None, "invalid_slot_counts"
        largest_rows = max(largest_rows, sum(counts))
    layer, source_method = template.layer_ref(), template.method_ref()
    if layer is None or source_method is None:
        return None, "model_weights_unavailable"
    try:
        scratch_bytes = _estimated_scratch_bytes(template, largest_rows, layer, source_method)
    except (AttributeError, RuntimeError, ValueError, TypeError):
        return None, "unsupported_weight_layout"
    if scratch_bytes > max_scratch_bytes:
        return None, "calibration_scratch_estimate_exceeds_limit"
    # The unquantized method writes temporary LoRA routing state. Keep even that
    # write off the live method; the shallow copy shares only immutable weights.
    method = copy(source_method)
    apply_mlp, context, comm_types, measure, isolated_context = _runtime()
    baseline, candidate = [], []
    try:
        with isolated_context:
            old_comm_type = context.moe_comm_type
            try:
                context.moe_comm_type = getattr(comm_types, _COMM_TYPES[comm_name])
                for old, new in zip(baseline_counts, candidate_counts):
                    for counts, destination in ((old, baseline), (new, candidate)):

                        def factory(counts=counts):
                            payload = _new_input(template, counts, layer, method)
                            return lambda: apply_mlp(payload, method)

                        destination.append(measure(factory, repeats=int(repeats), warmup=1))
            finally:
                context.moe_comm_type = old_comm_type
    except (RuntimeError, ValueError, TypeError) as error:
        return None, f"mlp_calibration_failed:{type(error).__name__}"
    return {
        "baseline": baseline,
        "candidate": candidate,
        "comm_method": comm_name,
        "scope": "isolated_eplb_on_mlp",
        "execution": "eager",
        "input": "synthetic_same_local_weights",
        "dynamic_eplb_kernel_flag": template.metadata["dynamic_eplb"],
        "estimated_scratch_bytes": scratch_bytes,
        "internal_workspace_bytes": "unknown",
        "communication_delta": "unknown",
        "graph_delta": "unknown",
    }, None
