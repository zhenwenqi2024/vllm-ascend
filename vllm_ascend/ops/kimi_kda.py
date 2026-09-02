# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend backend for the vLLM 0.27 Kimi K3 delta-attention layer.

The default path retains upstream projection and cache ownership. Optional
KDA state parallelism rebuilds the KDA head-local parameters and cache shape
over an Ascend fine-grained group while preserving DP execution for MLA.
"""

from functools import wraps

import torch
from einops import rearrange
from torch import nn
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.models.kimi_k3.nvidia.kda import (
    KimiK3DeltaAttention,
    _KimiGDNMergedColumnParallelLinear,
)
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.distributed.kda_state_parallel import (
    flatten_kda_state_cache,
    remap_kda_metadata_state_indices,
)
from vllm_ascend.distributed.parallel_state import get_kda_tp_group
from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionBackend
from vllm_ascend.ops.triton.fla.utils import clear_ssm_states
from vllm_ascend.utils import kda_tp_enable

_KDA_CHUNK_SIZE = 64
_PACKED_CONV_WEIGHT_NAME = "ascend_conv1d_weight"


def _kda_sharded_weight_loader(shard_axis: int, *, legacy_a_log: bool = False):
    """Load a tensor shard using the KDA-specific group rank."""

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        if legacy_a_log and loaded_weight.dim() == 4:
            if loaded_weight.shape[:2] != (1, 1) or loaded_weight.shape[-1] != 1:
                raise ValueError(f"Unexpected legacy KDA A_log shape: {loaded_weight.shape}.")
            loaded_weight = loaded_weight.reshape(loaded_weight.shape[2])
        shard_size = param.shape[shard_axis]
        start = get_kda_tp_group().rank_in_group * shard_size
        param.data.copy_(loaded_weight.narrow(shard_axis, start, shard_size))

    return loader


def _kda_conv_weight_loader(local_projection_size: int):
    """Load one Q/K/V convolution shard from an unfused checkpoint tensor."""

    def loader(
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
    ) -> None:
        if loaded_weight.dim() == 2:
            loaded_weight = loaded_weight.unsqueeze(1)
        rank = get_kda_tp_group().rank_in_group
        source_start = rank * local_projection_size
        target_start = loaded_shard_id * local_projection_size
        param.data[target_start : target_start + local_projection_size].copy_(
            loaded_weight[source_start : source_start + local_projection_size]
        )

    return loader


def _zero_padded_output(
    output: torch.Tensor,
    num_live_tokens: torch.Tensor,
) -> torch.Tensor:
    """Clear graph-padding rows using a device-side live-token count."""
    token_indices = torch.arange(
        output.shape[1],
        dtype=num_live_tokens.dtype,
        device=output.device,
    )
    valid_tokens = token_indices < num_live_tokens
    return torch.where(valid_tokens.view(1, -1, 1, 1), output, 0.0)


def _zero_padded_recurrent_output(
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    """Clear graph-padding rows skipped by recurrent KDA."""
    return _zero_padded_output(output, query_start_loc[-1])


def _prepare_beta(
    raw_beta: torch.Tensor,
    num_actual_tokens: int,
) -> torch.Tensor:
    """Convert vLLM 0.27's packed raw beta to the AscendC contract."""
    return raw_beta[:, :num_actual_tokens].float().sigmoid()


class AscendKimiK3DeltaAttention(KimiK3DeltaAttention):
    """Kimi K3 KDA using AscendC prefill and recurrent kernels."""

    def __init__(self, config, vllm_config, prefix: str = "") -> None:
        quant_config = getattr(vllm_config, "quant_config", None)
        uses_mixed_projection = bool(
            quant_config is not None
            and getattr(
                quant_config,
                "uses_kimi_k3_mixed_kda_projection",
                lambda _prefix: False,
            )(f"{prefix}.in_proj_qkvgfab")
        )
        super().__init__(config, vllm_config, prefix)
        self.uses_mixed_projection = uses_mixed_projection
        if kda_tp_enable():
            self._init_state_parallel_components(prefix, quant_config)
        else:
            self.kda_state_parallel_size = 1
            self.kda_state_parallel_rank = 0
        if uses_mixed_projection:
            # vLLM 0.27 packs all KDA input projections into one linear.  A
            # QuaRot checkpoint instead stores q/k/v as W8A8 and keeps the
            # three gates in floating point, so form one fused GEMM per
            # precision group instead of falling back to four projections.
            self.in_proj_qkvgfab = MergedColumnParallelLinear(
                self.hidden_size,
                [self.projection_size] * 3,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_qkv",
            )
            gate_output_sizes = [
                self.projection_size,
                self.head_dim,
                self.num_heads,
            ]
            if self.in_proj_padding:
                gate_output_sizes.append(
                    self.in_proj_padding * (get_kda_tp_group().world_size if kda_tp_enable() else self.tp_size)
                )
            self.in_proj_gfab = _KimiGDNMergedColumnParallelLinear(
                self.hidden_size,
                gate_output_sizes,
                replicated_shard_id=1,
                tp_size=(get_kda_tp_group().world_size if kda_tp_enable() else self.tp_size),
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_gfab",
            )
            if self.in_proj_padding:
                self.in_proj_gfab.weight_only_replicated_output_shards = {len(gate_output_sizes) - 1}
            if self.in_proj_padding:
                self.in_proj_gfab.weight.data[-self.in_proj_padding :].zero_()
        # Upstream's FusedRMSNormGated constructor defaults to 1e-5, while
        # Kimi K3 checkpoints use the model-configured RMS epsilon (1e-6 for
        # the production checkpoint). Preserve the checkpoint contract used
        # by the validated v0.26 implementation.
        self.o_norm.eps = config.rms_norm_eps
        # vLLM keeps the checkpoint-compatible FP32 [3C, 1, W] weight, while
        # npu_causal_conv1d_custom consumes an activation-dtype [W, 3C]
        # tensor. Materialize that kernel layout once after weight loading.
        self.register_parameter(
            _PACKED_CONV_WEIGHT_NAME,
            nn.Parameter(
                torch.empty(
                    self.conv_size,
                    3 * self.local_projection_size,
                    dtype=self.model_config.dtype,
                    device=self.conv1d.weight.device,
                ),
                requires_grad=False,
            ),
        )
        original_process_weights = self.conv1d.quant_method.process_weights_after_loading

        @wraps(original_process_weights)
        def process_weights_and_pack(*args, **kwargs):
            result = original_process_weights(*args, **kwargs)
            self._pack_conv_weights()
            return result

        self.conv1d.quant_method.process_weights_after_loading = process_weights_and_pack

    def _init_state_parallel_components(self, prefix, quant_config) -> None:
        """Rebuild KDA parameters over its DP-axis state-parallel group."""
        group = get_kda_tp_group()
        weight_tp_size = group.world_size
        self.kda_state_parallel_size = weight_tp_size
        self.kda_state_parallel_rank = group.rank_in_group
        self.tp_size = weight_tp_size
        self.tp_rank = group.rank_in_group
        self.local_num_heads = self.num_heads // weight_tp_size
        self.local_projection_size = self.projection_size // weight_tp_size

        qkvg_output_sizes = [self.projection_size] * 4
        output_sizes = qkvg_output_sizes + [self.head_dim, self.num_heads]
        local_output_size = 4 * self.local_projection_size + self.head_dim + self.local_num_heads
        weight_padding = -local_output_size % 16
        if weight_padding:
            output_sizes.append(weight_padding * weight_tp_size)
        self.in_proj_padding = weight_padding
        self.in_proj_qkvgfab = _KimiGDNMergedColumnParallelLinear(
            self.hidden_size,
            output_sizes,
            replicated_shard_id=4,
            tp_size=weight_tp_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_qkvgfab",
        )
        if weight_padding:
            self.in_proj_qkvgfab.weight_only_replicated_output_shards = {len(output_sizes) - 1}
        if weight_padding:
            self.in_proj_qkvgfab.weight.data[-weight_padding:].zero_()
        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            self.projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.dt_bias = nn.Parameter(torch.empty(self.local_projection_size, dtype=torch.float32))
        set_weight_attrs(
            self.dt_bias,
            {"weight_loader": _kda_sharded_weight_loader(0)},
        )
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=3 * self.projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        if hasattr(self.conv1d.weight, "weight_loader"):
            delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {"weight_loader": _kda_conv_weight_loader(self.local_projection_size)},
        )
        self.A_log = nn.Parameter(torch.empty(self.local_num_heads, dtype=torch.float32))
        set_weight_attrs(
            self.A_log,
            {
                "weight_loader": _kda_sharded_weight_loader(
                    0,
                    legacy_a_log=True,
                )
            },
        )
        self.o_proj = RowParallelLinear(
            self.projection_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.gemm_rs_ar = None
        self.decode_conv1d_weight = None
        self.decode_norm_weight = None

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        shapes = super().get_state_shape()
        parallel_size = getattr(self, "kda_state_parallel_size", 1)
        if parallel_size == 1:
            return shapes
        return tuple((parallel_size, *shape) for shape in shapes)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AscendGDNAttentionBackend

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if not kda_tp_enable() and not self.uses_mixed_projection:
            return super().forward(hidden_states, positions)

        local_token_capacity = hidden_states.size(0)
        if kda_tp_enable():
            hidden_states = get_kda_tp_group().all_gather(hidden_states, dim=0)
        num_tokens = hidden_states.size(0)

        if self.uses_mixed_projection:
            mixed_qkv = self.in_proj_qkvgfab(hidden_states)[0]
            projected_gfab = self.in_proj_gfab(hidden_states)[0]
            split_sizes = [
                self.local_projection_size,
                self.head_dim,
                self.local_num_heads,
            ]
            if self.in_proj_padding:
                split_sizes.append(self.in_proj_padding)
            g_proj_states, f_a, beta = projected_gfab.split(split_sizes, dim=-1)[:3]
        else:
            projected_qkvgfab = self.in_proj_qkvgfab(hidden_states)[0]
            split_sizes = [
                3 * self.local_projection_size,
                self.local_projection_size,
                self.head_dim,
                self.local_num_heads,
            ]
            if self.in_proj_padding:
                split_sizes.append(self.in_proj_padding)
            mixed_qkv, g_proj_states, f_a, beta = projected_qkvgfab.split(
                split_sizes,
                dim=-1,
            )[:4]

        beta = beta.unsqueeze(0)
        g1 = self.f_b_proj(f_a)[0]
        g1 = rearrange(g1, "n (h d) -> 1 n h d", d=self.head_dim)
        g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)
        core_attn_out = torch.empty(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        if kda_tp_enable():
            forward_context = get_forward_context()
            metadata_by_prefix = getattr(
                forward_context,
                "kda_parallel_metadata",
                None,
            )
            if metadata_by_prefix is None:
                if forward_context.attn_metadata is not None:
                    raise RuntimeError("KDA state-parallel metadata was not prepared by the model runner.")
                core_attn_out.zero_()
            else:
                owner_metadata = metadata_by_prefix[self.prefix]
                if len(owner_metadata) != self.kda_state_parallel_size:
                    raise RuntimeError(
                        f"Expected {self.kda_state_parallel_size} KDA metadata owners, got {len(owner_metadata)}."
                    )
                flat_kv_cache = (
                    flatten_kda_state_cache(
                        self.kv_cache[0],
                        self.kda_state_parallel_size,
                    ),
                    flatten_kda_state_cache(
                        self.kv_cache[1],
                        self.kda_state_parallel_size,
                    ),
                )
                for owner_rank, metadata in enumerate(owner_metadata):
                    token_slice = slice(
                        owner_rank * local_token_capacity,
                        (owner_rank + 1) * local_token_capacity,
                    )
                    self._forward(
                        mixed_qkv=mixed_qkv[token_slice],
                        g1=g1[:, token_slice],
                        g2=g2[token_slice],
                        beta=beta[:, token_slice],
                        core_attn_out=core_attn_out[:, token_slice],
                        attn_metadata=remap_kda_metadata_state_indices(
                            metadata,
                            owner_rank,
                            self.kda_state_parallel_size,
                        ),
                        kv_cache=flat_kv_cache,
                    )
        else:
            self._forward(
                mixed_qkv=mixed_qkv,
                g1=g1,
                g2=g2,
                beta=beta,
                core_attn_out=core_attn_out,
            )

        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        output_parallel = self.o_proj(core_attn_out)[0]
        if kda_tp_enable():
            return get_kda_tp_group().reduce_scatter(output_parallel, dim=0)
        return output_parallel

    @staticmethod
    def _run_causal_conv1d(
        mixed_qkv: torch.Tensor,
        conv_weights_t: torch.Tensor,
        conv_state: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor,
        initial_state_mode: torch.Tensor | None,
        *,
        run_mode: int,
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = torch.empty_like(mixed_qkv)
        # Consume the operator's declared output alias. Returning ``output``
        # independently would let graph functionalization treat the custom-op
        # result as dead and expose the uninitialized allocation instead.
        return torch.ops._C_ascend.npu_causal_conv1d_custom(
            output,
            mixed_qkv,
            conv_weights_t,
            conv_state=conv_state,
            bias_opt=None,
            query_start_loc_opt=query_start_loc,
            cache_indices_opt=cache_indices,
            initial_state_mode_opt=initial_state_mode,
            num_accepted_tokens_opt=num_accepted_tokens,
            activation_mode=1,
            pad_slot_id=PAD_SLOT_ID,
            run_mode=run_mode,
        )

    @torch.no_grad()
    def _pack_conv_weights(self) -> None:
        if self.conv1d.weight.is_meta:
            return
        packed_param = self.get_parameter(_PACKED_CONV_WEIGHT_NAME)
        packed_weight = (
            self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            .transpose(0, 1)
            .to(device=packed_param.device, dtype=packed_param.dtype)
            .contiguous()
        )
        replace_parameter(
            self,
            _PACKED_CONV_WEIGHT_NAME,
            packed_weight,
            prefer_copy=True,
        )

    def _run_recurrent(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        state_indices: torch.Tensor,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.ops._C_ascend.recurrent_kda(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            raw_gate.contiguous(),
            beta.contiguous(),
            recurrent_state,
            cu_seqlens,
            state_indices,
            self.A_log.reshape(-1).contiguous(),
            self.dt_bias.contiguous(),
            num_accepted_tokens=num_accepted_tokens,
            scale=self.head_dim**-0.5,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=False,
            allow_neg_eigval=False,
            safe_gate=self.gate_lower_bound is not None,
            lower_bound=(self.gate_lower_bound if self.gate_lower_bound is not None else -5.0),
        )

    def _run_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        prebuilt_metadata,
    ) -> torch.Tensor:
        cu_seqlens = (
            prebuilt_metadata.cu_seqlens_host
            if prebuilt_metadata.cu_seqlens_kern is None
            else prebuilt_metadata.cu_seqlens_kern
        )
        keep = prebuilt_metadata.keep_meta
        if keep is not None:
            state_indices = state_indices[keep]
            has_initial_state = has_initial_state[keep]

        # The recurrent cache uses [H,V,K]. The fused prefill operator accepts
        # that state layout directly through state_v_first.
        initial_state_vk = recurrent_state[state_indices].contiguous()
        clear_ssm_states(initial_state_vk, has_initial_state)

        q = l2norm_fwd(q.contiguous())
        k = l2norm_fwd(k.contiguous())
        result = torch.ops._C_ascend.chunk_kda_fwd(
            q,
            k,
            v.contiguous(),
            raw_gate.contiguous(),
            beta.contiguous(),
            self.head_dim**-0.5,
            _KDA_CHUNK_SIZE,
            layout="BSND",
            initial_state=initial_state_vk,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            chunk_indices=prebuilt_metadata.chunk_indices_chunk64_host,
            safe_gate=self.gate_lower_bound is not None,
            lower_bound=self.gate_lower_bound if self.gate_lower_bound is not None else -5.0,
            use_gate_in_kernel=True,
            A_log=self.A_log.reshape(-1).contiguous(),
            dt_bias=self.dt_bias.contiguous(),
            disable_recompute=False,
            return_intermediate_states=False,
            state_v_first=True,
        )
        recurrent_state[state_indices] = result[1].to(recurrent_state.dtype)
        return result[0]

    @eager_break_during_capture
    def _forward(
        self,
        mixed_qkv: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Dispatch speculative, prefill, and decode tokens through KDA kernels."""
        forward_context = get_forward_context()
        if attn_metadata is None:
            attn_metadata_raw = forward_context.attn_metadata
            if attn_metadata_raw is None:
                core_attn_out.zero_()
                return

            assert isinstance(attn_metadata_raw, dict)
            attn_metadata = attn_metadata_raw[self.prefix]
            assert isinstance(attn_metadata, GDNAttentionMetadata)

        num_actual_tokens = attn_metadata.num_actual_tokens
        mixed_qkv = mixed_qkv[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        g2 = g2[:num_actual_tokens]
        beta = _prepare_beta(beta, num_actual_tokens)

        conv_state, recurrent_state = self.kv_cache if kv_cache is None else kv_cache
        conv_weights_t = self.get_parameter(_PACKED_CONV_WEIGHT_NAME)
        spec_masks = attn_metadata.spec_sequence_masks
        spec_token_indices = attn_metadata.spec_token_indx
        non_spec_token_indices = attn_metadata.non_spec_token_indx

        if spec_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_spec = mixed_qkv
                raw_gate_spec = g1
                beta_spec = beta
                mixed_non_spec = raw_gate_non_spec = beta_non_spec = None
            else:
                assert spec_token_indices is not None
                assert non_spec_token_indices is not None
                mixed_spec = mixed_qkv.index_select(0, spec_token_indices)
                raw_gate_spec = g1.index_select(1, spec_token_indices)
                beta_spec = beta.index_select(1, spec_token_indices)
                mixed_non_spec = mixed_qkv.index_select(0, non_spec_token_indices)
                raw_gate_non_spec = g1.index_select(1, non_spec_token_indices)
                beta_non_spec = beta.index_select(1, non_spec_token_indices)
        else:
            mixed_spec = raw_gate_spec = beta_spec = None
            mixed_non_spec = mixed_qkv
            raw_gate_non_spec = g1
            beta_non_spec = beta

        core_spec = None
        if mixed_spec is not None:
            spec_meta = attn_metadata.spec_decode_metadata
            assert spec_meta is not None
            spec_conv_meta = spec_meta.spec_causal_conv1d
            mixed_spec = self._run_causal_conv1d(
                mixed_spec,
                conv_weights_t,
                conv_state,
                spec_conv_meta.query_start_loc,
                spec_conv_meta.cache_indices,
                None,
                run_mode=1,
                num_accepted_tokens=spec_conv_meta.num_accepted_tokens,
            )
            q_spec, k_spec, v_spec = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim) for x in mixed_spec.chunk(3, dim=-1)
            )
            assert raw_gate_spec is not None and beta_spec is not None
            assert attn_metadata.spec_query_start_loc is not None
            assert attn_metadata.spec_state_indices_tensor is not None
            core_spec = self._run_recurrent(
                q_spec,
                k_spec,
                v_spec,
                raw_gate_spec,
                beta_spec,
                recurrent_state,
                attn_metadata.spec_query_start_loc,
                attn_metadata.spec_state_indices_tensor,
                num_accepted_tokens=spec_conv_meta.num_accepted_tokens,
            )
            core_spec = _zero_padded_recurrent_output(
                core_spec,
                attn_metadata.spec_query_start_loc,
            )

        core_non_spec = None
        if mixed_non_spec is not None and mixed_non_spec.shape[0] > 0:
            if attn_metadata.num_prefills > 0:
                prefill_meta = attn_metadata.non_spec_prefill_metadata
                assert prefill_meta is not None
                mixed_non_spec = self._run_causal_conv1d(
                    mixed_non_spec,
                    conv_weights_t,
                    conv_state,
                    prefill_meta.causal_conv1d.query_start_loc,
                    prefill_meta.causal_conv1d.cache_indices,
                    prefill_meta.causal_conv1d.initial_state_mode,
                    run_mode=0,
                )
            elif attn_metadata.num_decodes > 0:
                decode_meta = attn_metadata.non_spec_decode_metadata
                assert decode_meta is not None
                mixed_non_spec = self._run_causal_conv1d(
                    mixed_non_spec,
                    conv_weights_t,
                    conv_state,
                    decode_meta.causal_conv1d.query_start_loc,
                    decode_meta.causal_conv1d.cache_indices,
                    None,
                    run_mode=1,
                )

            q_non_spec, k_non_spec, v_non_spec = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim) for x in mixed_non_spec.chunk(3, dim=-1)
            )
            assert raw_gate_non_spec is not None
            assert beta_non_spec is not None

            split_non_spec = spec_masks is None and attn_metadata.num_prefills > 0 and attn_metadata.num_decodes > 0
            num_decode_tokens = attn_metadata.num_decode_tokens
            core_decode = None
            if split_non_spec:
                assert attn_metadata.non_spec_query_start_loc is not None
                assert attn_metadata.non_spec_state_indices_tensor is not None
                core_decode = self._run_recurrent(
                    q_non_spec[:, :num_decode_tokens],
                    k_non_spec[:, :num_decode_tokens],
                    v_non_spec[:, :num_decode_tokens],
                    raw_gate_non_spec[:, :num_decode_tokens],
                    beta_non_spec[:, :num_decode_tokens],
                    recurrent_state,
                    attn_metadata.non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                    attn_metadata.non_spec_state_indices_tensor[: attn_metadata.num_decodes],
                )

            if attn_metadata.num_prefills > 0:
                if split_non_spec:
                    q_non_spec = q_non_spec[:, num_decode_tokens:]
                    k_non_spec = k_non_spec[:, num_decode_tokens:]
                    v_non_spec = v_non_spec[:, num_decode_tokens:]
                    raw_gate_non_spec = raw_gate_non_spec[:, num_decode_tokens:]
                    beta_non_spec = beta_non_spec[:, num_decode_tokens:]

                assert attn_metadata.prefill_state_indices is not None
                assert attn_metadata.prefill_has_initial_state is not None
                prefill_meta = attn_metadata.non_spec_prefill_metadata
                assert prefill_meta is not None
                core_prefill = self._run_prefill(
                    q_non_spec,
                    k_non_spec,
                    v_non_spec,
                    raw_gate_non_spec,
                    beta_non_spec,
                    recurrent_state,
                    attn_metadata.prefill_state_indices,
                    attn_metadata.prefill_has_initial_state,
                    prefill_meta.chunk,
                )
                core_non_spec = (
                    torch.cat((core_decode, core_prefill), dim=1) if core_decode is not None else core_prefill
                )
            elif attn_metadata.num_decodes > 0:
                assert attn_metadata.non_spec_query_start_loc is not None
                assert attn_metadata.non_spec_state_indices_tensor is not None
                core_non_spec = self._run_recurrent(
                    q_non_spec,
                    k_non_spec,
                    v_non_spec,
                    raw_gate_non_spec,
                    beta_non_spec,
                    recurrent_state,
                    attn_metadata.non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                    attn_metadata.non_spec_state_indices_tensor,
                )

        if core_non_spec is not None:
            assert attn_metadata.non_spec_query_start_loc is not None
            core_non_spec = _zero_padded_recurrent_output(
                core_non_spec,
                attn_metadata.non_spec_query_start_loc,
            )

        if core_spec is None and core_non_spec is None:
            # Idle DP dummy runs carry graph-shaped metadata with no live work.
            # Do not feed a previous replay's output through the norm gate.
            core_attn_out.zero_()
            return

        num_live_tokens = None
        if core_spec is not None:
            assert attn_metadata.spec_query_start_loc is not None
            num_live_tokens = attn_metadata.spec_query_start_loc[-1]
        if core_non_spec is not None:
            assert attn_metadata.non_spec_query_start_loc is not None
            num_non_spec_tokens = attn_metadata.non_spec_query_start_loc[-1]
            num_live_tokens = num_non_spec_tokens if num_live_tokens is None else num_live_tokens + num_non_spec_tokens
        assert num_live_tokens is not None

        # Reuse the caller-owned result buffer. FULL graphs can leave rows
        # outside the live spec/non-spec index sets, so define them before the
        # two index copies rather than allocating a temporary merged tensor.
        core_attn_out[:, :num_actual_tokens].zero_()
        if core_spec is not None and core_non_spec is not None:
            assert spec_token_indices is not None
            assert non_spec_token_indices is not None
            assert spec_token_indices.numel() + non_spec_token_indices.numel() <= num_actual_tokens
            core_attn_out[:, :num_actual_tokens].index_copy_(1, spec_token_indices, core_spec)
            core_attn_out[:, :num_actual_tokens].index_copy_(1, non_spec_token_indices, core_non_spec)
        elif core_spec is not None:
            core_attn_out[:, :num_actual_tokens] = core_spec
        elif core_non_spec is not None:
            core_attn_out[:, :num_actual_tokens] = core_non_spec

        # The registered Ascend FusedRMSNormGated uses the fused norm-gate
        # kernel while preserving the upstream parameter/loading contract.
        normalized = self.o_norm(core_attn_out[:, :num_actual_tokens], g2)
        # Mask again after the norm gate: zero * sigmoid(NaN) is still NaN in
        # static padding rows whose captured gate values are not live.
        core_attn_out[:, :num_actual_tokens].copy_(_zero_padded_output(normalized, num_live_tokens))
        core_attn_out[:, num_actual_tokens:].zero_()
