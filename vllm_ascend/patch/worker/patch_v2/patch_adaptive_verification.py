# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Adapt only the legacy Torch adaptive-budget allocator's index_fill_.

The current vLLM allocator uses a Triton kernel and no longer exposes the
private Python/compiled pair. Keep that implementation unchanged. Remove
this compatibility branch when the legacy allocator is no longer supported.
"""

import torch
from torch.overrides import TorchFunctionMode
from vllm.v1.worker.gpu.spec_decode import adaptive_verification


def _index_fill(tensor, dim, index, value):
    # Import lazily to avoid circular imports during plugin startup.
    from vllm_ascend.device.device_op import DeviceOperator

    return DeviceOperator.index_fill(tensor, dim, index, value)


class _IndexFillMode(TorchFunctionMode):
    """Route only the old allocator's index_fill_ through the device adaptor."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func is torch.Tensor.index_fill_:
            return _index_fill(*args, **kwargs)
        return func(*args, **kwargs)


_original_assign_draft_token_budget = getattr(adaptive_verification, "_assign_draft_token_budget", None)


def _assign_draft_token_budget_ascend(*args, **kwargs):
    if _original_assign_draft_token_budget is None:
        raise RuntimeError("The legacy adaptive-verification allocator is unavailable.")
    with _IndexFillMode():
        return _original_assign_draft_token_budget(*args, **kwargs)


if _original_assign_draft_token_budget is not None:
    adaptive_verification._assign_draft_token_budget_compiled = _assign_draft_token_budget_ascend
elif not (
    callable(getattr(adaptive_verification, "assign_draft_token_budget", None))
    and hasattr(adaptive_verification, "_assign_draft_token_budget_kernel")
):
    raise RuntimeError("Unsupported vLLM adaptive-verification allocator API.")
