from collections.abc import Iterable
from copy import copy

import torch
from torch import nn
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention, DeepseekV2MLP
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3DecoderLayer, _get_dflash_fc_input_size
from vllm.model_executor.models.qwen3_dspark import DSparkConfidenceHead, DSparkMarkovHead, Qwen3DSparkForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper, maybe_prefix

from vllm_ascend.models.llama_eagle3 import load_quarot_target_layer
from vllm_ascend.utils import (
    get_rotation_matrix,
    get_rotation_path,
)

TARGET_EMBED_WEIGHT_NAMES = (
    "language_model.model.embed_tokens.weight",
    "model.embed_tokens.weight",
)
TARGET_LM_HEAD_WEIGHT_NAMES = (
    "language_model.lm_head.weight",
    "lm_head.weight",
)


# Process the first linear weight with rotation matrix, if the target model uses rotary quantization
def process_weight(linear_weight: torch.Tensor, rotation_weight: torch.Tensor):
    assert linear_weight.shape[1] % rotation_weight.shape[0] == 0, (
        f"Linear weight shape[1] must be a multiple of rotation weight shape[0],"
        f" but get {linear_weight.shape[1]=} and {rotation_weight.shape[0]=}"
    )
    rotation_weight = rotation_weight.to(device=linear_weight.device, dtype=torch.float32)
    hidden_size = rotation_weight.shape[0]
    ori_dtype = linear_weight.dtype
    processed_weight = torch.empty(linear_weight.shape, dtype=torch.float32, device=linear_weight.device)
    for start_pos in range(0, linear_weight.shape[1], hidden_size):
        linear_weight_chunked = linear_weight[:, start_pos : start_pos + hidden_size].to(torch.float32)
        processed_weight[:, start_pos : start_pos + hidden_size].copy_(
            torch.matmul(linear_weight_chunked, rotation_weight)
        )
    return processed_weight.to(ori_dtype)


@torch.no_grad()
def align_draft_weights(model, projection, vllm_config):
    """Align draft inputs with the rotated target without modifying shared weights."""
    rotation_path = get_rotation_path(vllm_config)
    if rotation_path is not None:
        rotation = get_rotation_matrix(rotation_path).cpu()
        weight = projection.weight
        weight.copy_(process_weight(weight.cpu(), rotation).to(weight.device))
    else:
        if get_pp_group().world_size == 1:
            return
        # PP stages past the first cannot alias the target's stage-0
        # embedding; load this rank's vocab shard from the target
        # checkpoint unrotated so the draft owns its copy.
        rotation = None
        weight = projection.weight
    target_config = vllm_config.model_config.hf_text_config
    for owner, name, layer_cls, weight_names, own_flag in (
        (model.model, "embed_tokens", VocabParallelEmbedding, TARGET_EMBED_WEIGHT_NAMES, "has_own_embed_tokens"),
        (model, "lm_head", ParallelLMHead, TARGET_LM_HEAD_WEIGHT_NAMES, "has_own_lm_head"),
    ):
        if getattr(model, own_flag, False):
            continue
        if rotation is None and name != "embed_tokens":
            # The target lm_head lives on the last PP stage and can still
            # be aliased; only the embedding needs its own copy.
            continue
        with torch.device(weight.device):
            layer = layer_cls(target_config.vocab_size, target_config.hidden_size, params_dtype=weight.dtype)
        load_quarot_target_layer(layer, vllm_config.model_config.model, weight_names, rotation, f"draft {name}.weight")
        layer.quant_method.process_weights_after_loading(layer)
        setattr(owner, name, layer)
        setattr(model, own_flag, True)


class AscendQwen3DSparkForCausalLM(Qwen3DSparkForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = self.config
        self.enable_confidence_head = bool(getattr(config, "enable_confidence_head", False))

    def compute_confidence(self, head_hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        """Per-position acceptance probability for each drafted token."""
        if not self.enable_confidence_head:
            raise RuntimeError("The DSpark confidence head is disabled.")
        assert self.model.confidence_head is not None
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    def configure_target_aux_hidden_capture(self, target_model: torch.nn.Module) -> None:
        """Select draft auxiliary inputs, without changing target Eager/Graph mode."""
        set_capture_mode = getattr(target_model, "set_dspark_aux_capture_materialized", None)
        if set_capture_mode is None:
            get_language_model = getattr(target_model, "get_language_model", None)
            if callable(get_language_model):
                set_capture_mode = getattr(get_language_model(), "set_dspark_aux_capture_materialized", None)
        if set_capture_mode is not None:
            set_capture_mode(True)

    def post_process(self, vllm_config: VllmConfig) -> None:
        align_draft_weights(self, self.model.fc, vllm_config)


class Glm5DSparkMLAAttention(DeepseekV2MLAAttention):
    """Reuse the patched dense MLA attention, not the Qwen GQA backbone."""

    def __init__(self, *, vllm_config: VllmConfig, config, prefix: str) -> None:
        super().__init__(
            vllm_config=vllm_config,
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=config.max_position_embeddings,
            cache_config=vllm_config.cache_config,
            quant_config=None,
            prefix=prefix,
            non_causal_multi_token_decode=True,
        )
        self.attn = self.mla_attn.mla_attn

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        return super().forward(positions, hidden_states, llama_4_scaling=None)


class Glm5DSparkDecoderLayer(DFlashQwen3DecoderLayer):
    """Keep the upstream residual/normalization flow with dense MLA components."""

    def __init__(self, *, vllm_config: VllmConfig, config, prefix: str) -> None:
        nn.Module.__init__(self)
        self.self_attn = Glm5DSparkMLAAttention(
            vllm_config=vllm_config, config=config, prefix=maybe_prefix(prefix, "self_attn")
        )
        self.mlp = DeepseekV2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=None,
            prefix=maybe_prefix(prefix, "mlp"),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Glm5DSparkModel(nn.Module):
    """Checkpoint-specific MLA components composed with upstream DSpark heads.

    This compatibility extension can be replaced by the upstream GLM MLA draft
    implementation once its loader and context-cache interface are available.
    It intentionally does not inherit target W4A8/SFA quantization or offloading.
    """

    def __init__(self, *, vllm_config: VllmConfig, start_layer_id: int, prefix: str) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        draft = vllm_config.speculative_config.draft_model_config
        if draft is None:
            raise ValueError("GLM MLA DSpark requires a draft model config.")
        self.config = config = draft.hf_config
        mla_fields = ("q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim")
        if any(not isinstance(getattr(config, name, None), int) or getattr(config, name) <= 0 for name in mla_fields):
            raise ValueError("GLM MLA DSpark requires positive dense MLA projection dimensions; GQA is unsupported.")
        if config.num_hidden_layers <= 0:
            raise ValueError("GLM MLA DSpark requires at least one draft layer.")
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, quant_config=None, prefix=maybe_prefix(prefix, "embed_tokens")
        )
        self.context_proj = ReplicatedLinear(
            _get_dflash_fc_input_size(vllm_config),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=maybe_prefix(prefix, "context_proj"),
        )
        self.context_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            Glm5DSparkDecoderLayer(
                vllm_config=vllm_config, config=config, prefix=maybe_prefix(prefix, f"layers.{start_layer_id + index}")
            )
            for index in range(config.num_hidden_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            getattr(config, "draft_vocab_size", None) or config.vocab_size,
            config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
            quant_config=None,
        )
        self.confidence_head = None
        if getattr(config, "enable_confidence_head", False):
            with_markov = bool(getattr(config, "confidence_head_with_markov", False))
            self.confidence_head = DSparkConfidenceHead(
                config.hidden_size + (config.markov_rank if with_markov else 0),
                prefix=maybe_prefix(prefix, "confidence_head"),
                bias=True,
                with_markov=with_markov,
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        expected = self.context_proj.input_size
        if hidden_states.shape[-1] != expected:
            raise ValueError(
                f"GLM MLA DSpark expects {expected} ordered auxiliary features, got {hidden_states.shape[-1]}."
            )
        return self.context_norm(self.context_proj(hidden_states))

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | tuple[torch.Tensor | None, ...] | None = None,
    ) -> None:
        if context_states.numel() == 0 or context_slot_mapping is None:
            return
        per_layer = isinstance(context_slot_mapping, (list, tuple))
        if per_layer and len(context_slot_mapping) != len(self.layers):
            raise ValueError("context_slot_mapping must contain one entry per GLM MLA draft layer.")
        if context_positions.numel() != context_states.shape[0]:
            raise ValueError("GLM MLA draft context positions must match context rows.")
        # Use the draft's own RoPE cache, not the target's global cache: the
        # target and this checkpoint need not have identical RoPE parameters.
        rotary = self.layers[0].self_attn.rotary_emb
        cos, sin = rotary.cos_sin_cache.chunk(2, dim=-1)
        cos = cos.repeat(1, 2)[context_positions].unsqueeze(1).unsqueeze(2)
        sin = sin.repeat(1, 2)[context_positions].unsqueeze(1).unsqueeze(2)
        for index, layer in enumerate(self.layers):
            slots = context_slot_mapping[index] if per_layer else context_slot_mapping
            if slots is None or slots.numel() == 0:
                continue
            if slots.numel() != context_states.shape[0]:
                raise ValueError("GLM MLA draft slot mapping must match context rows.")
            attn = layer.self_attn
            qkv = attn.fused_qkv_a_proj(context_states)[0]
            attn.attn.impl.exec_kv_prefill(
                qkv[..., attn.q_lora_rank :].contiguous(), cos, sin, attn.attn.kv_cache, slots
            )

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, inputs_embeds=None) -> torch.Tensor:
        hidden_states = self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.final_norm(hidden_states, residual)
        return hidden_states


class Glm5DSparkForCausalLM(AscendQwen3DSparkForCausalLM):
    """Inherit sampling/Markov contracts while replacing only the MLA backbone."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
            ".q_a_proj": (".fused_qkv_a_proj", 0),
            ".kv_a_proj_with_mqa": (".fused_qkv_a_proj", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        if self.draft_model_config is None:
            raise ValueError("GLM MLA DSpark requires a draft model config.")
        self.config = self.draft_model_config.hf_config
        draft_config = copy(vllm_config)
        draft_config.quant_config = None
        draft_vocab_size = getattr(self.config, "draft_vocab_size", None) or self.config.vocab_size
        self.target_vocab_size = vllm_config.model_config.get_vocab_size()
        with set_current_vllm_config(draft_config):
            self.model = Glm5DSparkModel(
                vllm_config=draft_config,
                start_layer_id=vllm_config.model_config.get_total_num_hidden_layers(),
                prefix=maybe_prefix(prefix, "model"),
            )
            self.lm_head = ParallelLMHead(
                draft_vocab_size, self.config.hidden_size, prefix=maybe_prefix(prefix, "lm_head")
            )
            self.logits_processor = LogitsProcessor(draft_vocab_size, scale=getattr(self.config, "logit_scale", 1.0))
        self.enable_confidence_head = self.model.confidence_head is not None
        self.has_own_embed_tokens = self.has_own_lm_head = True
        self.draft_id_to_target_id = (
            nn.Parameter(torch.zeros(draft_vocab_size, dtype=torch.long), requires_grad=False)
            if draft_vocab_size < self.target_vocab_size
            else None
        )

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(hidden_states)

    def get_draft_attn_causal(self) -> list[bool]:
        # GLM MLA DFlash training exposes the whole current draft block to each
        # query. sliding_window_non_causal only controls sliding-window layers;
        # it does not make this checkpoint's full-attention layers causal.
        return [False] * len(self.model.layers)

    def post_process(self, vllm_config: VllmConfig) -> None:
        align_draft_weights(self, self.model.context_proj, vllm_config)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        normalized = []
        included = set()
        for name, weight in weights:
            if name == "t2d":
                continue
            if name == "d2t":
                name = "draft_id_to_target_id"
            if name == "norm.weight":
                name = "final_norm.weight"
            if name != "draft_id_to_target_id" and not name.startswith("lm_head."):
                name = f"model.{name}"
            included.add(name)
            normalized.append((name, weight))
        self.has_own_embed_tokens = "model.embed_tokens.weight" in included
        self.has_own_lm_head = "lm_head.weight" in included
        if self.draft_id_to_target_id is not None and not {"draft_id_to_target_id", "lm_head.weight"} <= included:
            raise ValueError("Reduced-vocabulary GLM MLA DSpark requires d2t and draft lm_head weights.")
        if self.config.vocab_size > self.target_vocab_size and not self.has_own_embed_tokens:
            raise ValueError("Expanded-vocabulary GLM MLA DSpark requires draft embedding weights.")
        self.enable_confidence_head = any(name.startswith("model.confidence_head.") for name in included)
        skipped = {"draft_id_to_target_id": None} if "draft_id_to_target_id" not in included else {}
        if not self.enable_confidence_head:
            skipped["confidence_head"] = None
        if not self.has_own_embed_tokens:
            skipped["embed_tokens"] = None
        if not self.has_own_lm_head:
            skipped["lm_head"] = None
        return AutoWeightsLoader(self).load_weights(
            normalized, mapper=self.hf_to_vllm_mapper | WeightsMapper(orig_to_new_substr=skipped)
        )
