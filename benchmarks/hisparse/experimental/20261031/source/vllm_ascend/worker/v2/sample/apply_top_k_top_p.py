import torch

from vllm_ascend.sample.sampler import _apply_top_k_top_p_pytorch


def apply_top_k_top_p_npu(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    has_topp_only_rows: bool = True,
) -> torch.Tensor:
    # Upstream uses this CPU-derived hint to schedule its CUDA split top-p path.
    # Ascend's existing sort-based implementation handles all rows together, so
    # the hint does not change masking: False must not disable top-p itself.
    # NOTE: During the warmup stage, if both k and p are None, the NPU op is skipped.
    # As a result, its workspace is not reserved during warmup, which may lead to an OOM
    # when the op is executed later.
    # TODO: Fix the workspace reservation issue in the future.
    if k is None and p is None:
        return logits
    # use pytorch ops
    return _apply_top_k_top_p_pytorch(logits, k, p)
