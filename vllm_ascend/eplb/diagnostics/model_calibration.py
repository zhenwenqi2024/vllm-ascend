# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated synthetic MLP calibration with immutable, already-loaded weights."""

from copy import copy
from dataclasses import dataclass, replace
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
    row_capacity: int
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
    elif mlp_input.dynamic_eplb or mlp_input.layer is None:
        reason = "requires_eplb_off_weights"
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
        mlp_input.hidden_states.shape[0],
        quant_name,
    )
    if previous is not None:
        if replace(previous, row_capacity=template.row_capacity) != template:
            templates[comm_name] = "ambiguous_mlp_template"
            return "ambiguous_mlp_template"
        template = replace(template, row_capacity=max(previous.row_capacity, template.row_capacity))
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


@torch.inference_mode()
def calibrate_layer(probe, baseline_counts, candidate_counts, comm_name, repeats=3):
    """Measure per-step local MLP intervals after inference, outside graph capture.

    Each repetition allocates disposable synthetic activations; W8A8 may free
    its input storage. Only immutable existing local weights are reused, so
    candidate counts emulate shape changes without moving expert weights.
    Both cases use EPLB-off kernels in eager mode. These timings exclude
    dispatch/combine, graph-launch differences, and EPLB-on kernel differences.
    The caller must keep serving quiescent throughout this collective RPC.
    """
    template = probe.calibration_templates.get(comm_name)
    if template is None or isinstance(template, str):
        return None, template or "missing_mlp_template"
    if not baseline_counts or len(baseline_counts) != len(candidate_counts) or not 1 <= repeats <= 20:
        return None, "invalid_calibration_samples"
    for counts in (*baseline_counts, *candidate_counts):
        if len(counts) != template.slots or any(not isinstance(value, Integral) or value < 0 for value in counts):
            return None, "invalid_slot_counts"
        if sum(counts) > template.row_capacity:
            return None, "calibration_shape_exceeds_observed_capacity"
    layer, source_method = template.layer_ref(), template.method_ref()
    if layer is None or source_method is None:
        return None, "model_weights_unavailable"
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

                        destination.append(measure(factory, repeats=repeats, warmup=1))
            finally:
                context.moe_comm_type = old_comm_type
    except (RuntimeError, ValueError, TypeError) as error:
        return None, f"mlp_calibration_failed:{type(error).__name__}"
    return {
        "baseline": baseline,
        "candidate": candidate,
        "comm_method": comm_name,
        "scope": "isolated_eplb_off_mlp",
        "execution": "eager",
        "input": "synthetic_same_weight_shapes",
        "on_kernel_delta": "unknown",
        "communication_delta": "unknown",
        "graph_delta": "unknown",
    }, None


def expert_payload_bytes(probe, comm_name):
    """Logical bytes per expert from actual OFF weight views, for a scratch proxy.

    Exact storage aliases are counted once. This does not measure the physical
    NZ padding, an EPLB-on layout conversion, or actual migration traffic.
    """
    template = probe.calibration_templates.get(comm_name)
    if template is None or isinstance(template, str):
        return None, template or "missing_mlp_template"
    layer, method = template.layer_ref(), template.method_ref()
    if layer is None or method is None:
        return None, "model_weights_unavailable"
    get_views = getattr(method, "get_eplb_weight_views", None)
    if get_views is None:
        return None, "expert_weight_views_unavailable"
    seen = [set() for _ in range(template.slots)]
    sizes = [0] * template.slots
    try:
        views = get_views(layer)
        for view in views:
            if isinstance(view, torch.Tensor):
                if view.ndim == 0 or view.shape[0] != template.slots:
                    return None, "unsupported_expert_weight_view"
                experts = view.unbind(0)
            elif isinstance(view, (list, tuple)) and len(view) == template.slots:
                experts = view
            else:
                return None, "unsupported_expert_weight_view"
            for slot, tensor in enumerate(experts):
                if not isinstance(tensor, torch.Tensor):
                    return None, "unsupported_expert_weight_view"
                size = tensor.numel() * tensor.element_size()
                identity = (tensor.data_ptr(), size)
                if identity not in seen[slot]:
                    seen[slot].add(identity)
                    sizes[slot] += size
    except (RuntimeError, ValueError, TypeError):
        return None, "expert_payload_unavailable"
    if not sizes or not sizes[0] or len(set(sizes)) != 1:
        return None, "nonuniform_expert_payload"
    return sizes[0], None
