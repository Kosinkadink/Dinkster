"""Native causal Qwen3 sparse-MoE model math.

The operations follow ``transformers/models/qwen3_moe/modeling_qwen3_moe.py``
at Hugging Face Transformers commit ``0720e206`` (v4.51.0).

The router follows Qwen3-30B-A3B exactly: full-expert float32 softmax, sorted
top-k selection, selected-score normalization, then expert-index-ordered
SwiGLU accumulation. Each expert declares one residency unit containing its
three projections so placement policy treats the expert as one unit.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
import torch.nn.functional as F
from dinkster_inference import Qwen3MoeConfig

from .attention import AttentionKernel
from .model_prefetch import close_prefetch_queue, make_prefetch_queue, prefetch_queue_pop
from .module_residency import declare_residency_unit
from .operations import INITLESS, Operations, materialized_rms_norm_weight
from .qwen_text import QwenAttention, _rope  # pyright: ignore[reportPrivateUsage]


def _qwen3_moe_rms_norm(module: torch.nn.RMSNorm, hidden: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden.dtype
    normalized = hidden.float()
    variance = normalized.pow(2).mean(-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + cast(float, module.eps))
    with materialized_rms_norm_weight(module) as weight:
        return weight * normalized.to(input_dtype)


def _qwen3_moe_apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    frequencies: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine, sine, negative_sine = frequencies
    cosine = cosine.to(query.dtype)
    sine = torch.cat((sine, -negative_sine), dim=-1).to(query.dtype)

    def rotate_half(value: torch.Tensor) -> torch.Tensor:
        half = value.shape[-1] // 2
        return torch.cat((-value[..., half:], value[..., :half]), dim=-1)

    return (
        (query * cosine) + (rotate_half(query) * sine),
        (key * cosine) + (rotate_half(key) * sine),
    )


class _Qwen3MoeAttention(QwenAttention):
    def _normalize(self, module: torch.nn.RMSNorm, hidden: torch.Tensor) -> torch.Tensor:
        return _qwen3_moe_rms_norm(module, hidden)

    def _apply_rotary(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        frequencies: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _qwen3_moe_apply_rope(query, key, frequencies)

    def _attention_is_causal(self, mask: torch.Tensor | None, length: int) -> bool:
        return mask is None and length > 1


class Qwen3MoeExpert(torch.nn.Module):
    """One independently placeable Qwen3 SwiGLU expert."""

    def __init__(self, config: Qwen3MoeConfig, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        hidden = config.hidden_size
        intermediate = config.moe_intermediate_size
        self.gate_proj = operations.linear(hidden, intermediate, bias=False)
        self.up_proj = operations.linear(hidden, intermediate, bias=False)
        self.down_proj = operations.linear(intermediate, hidden, bias=False)
        declare_residency_unit(self, expert=True)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class Qwen3MoeSparseMoeBlock(torch.nn.Module):
    """Token router and expert dispatch for one decoder layer."""

    def __init__(self, config: Qwen3MoeConfig, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = operations.linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = torch.nn.ModuleList(
            Qwen3MoeExpert(config, operations=operations) for _ in range(config.num_experts)
        )

    def forward(self, hidden: torch.Tensor, *, prefetch: bool = True) -> torch.Tensor:
        original_shape = hidden.shape
        flat = hidden.reshape(-1, hidden.shape[-1])
        router_logits = self.gate(flat)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(
            routing_weights,
            self.top_k,
            dim=-1,
        )
        if self.norm_topk_prob:
            routing_weights.div_(routing_weights.sum(dim=-1, keepdim=True))
        routing_weights = routing_weights.to(hidden.dtype)

        output = torch.zeros_like(flat)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        active_experts = torch.where(expert_mask.sum(dim=(-1, -2)) > 0)[0].tolist()
        prefetch_queue = (
            make_prefetch_queue(self.experts[index] for index in active_experts)
            if prefetch
            else None
        )
        try:
            for expert_index in active_experts:
                expert = self.experts[expert_index]
                prefetch_queue_pop(prefetch_queue, expert)
                top_index, token_index = torch.where(expert_mask[expert_index])
                current = expert(flat[token_index])
                current.mul_(routing_weights[token_index, top_index, None])
                output.index_add_(0, token_index, current)
            prefetch_queue_pop(prefetch_queue, None)
        finally:
            close_prefetch_queue(prefetch_queue)
        return output.reshape(original_shape)


class Qwen3MoeBlock(torch.nn.Module):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = (
            _Qwen3MoeAttention(config, operations=operations)
            if attention_kernel is None
            else _Qwen3MoeAttention(
                config,
                operations=operations,
                attention_kernel=attention_kernel,
            )
        )
        self.mlp = Qwen3MoeSparseMoeBlock(config, operations=operations)
        self.input_layernorm = operations.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = operations.rms_norm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward_causal(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None,
        frequencies: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        cache_key: torch.Tensor | None,
        cache_value: torch.Tensor | None,
        cache_position: int,
        *,
        prefetch: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attention, key, value = self.self_attn.forward_causal(
            _qwen3_moe_rms_norm(self.input_layernorm, hidden),
            mask,
            frequencies,
            None if cache_key is None or cache_value is None else (cache_key, cache_value),
            cache_position,
        )
        hidden = hidden + attention
        return (
            hidden
            + self.mlp(
                _qwen3_moe_rms_norm(self.post_attention_layernorm, hidden),
                prefetch=prefetch,
            ),
            key,
            value,
        )


class Qwen3MoeModel(torch.nn.Module):
    """Qwen3 sparse decoder with caller-owned causal KV storage."""

    def __init__(
        self,
        config: Qwen3MoeConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = operations.embedding(config.vocab_size, config.hidden_size)
        self.layers = torch.nn.ModuleList(
            Qwen3MoeBlock(
                config,
                operations=operations,
                attention_kernel=attention_kernel,
            )
            for _ in range(config.num_hidden_layers)
        )
        self.norm = operations.rms_norm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden, _ = self.forward_causal(ids)
        return hidden

    def forward_causal(
        self,
        ids: torch.Tensor,
        cache_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]] = (),
        *,
        cache_position: int = 0,
        frequencies: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        prefetch: bool = True,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError(
                f"Qwen3 MoE causal ids must be [batch x nonempty tokens], got {tuple(ids.shape)}"
            )
        if type(cache_position) is not int or cache_position < 0:
            raise ValueError("Qwen3 MoE cache position must be a non-negative exact integer")
        if len(cache_key_values) not in (0, len(self.layers)):
            raise ValueError("Qwen3 MoE KV must be empty or contain one pair per decoder layer")
        batch = ids.shape[0]
        if cache_key_values:
            first_key, _ = cache_key_values[0]
            if first_key.ndim != 4:
                raise ValueError("Qwen3 MoE KV tensors must be rank four")
            if first_key.shape[2] < cache_position + ids.shape[1]:
                raise ValueError("Qwen3 MoE KV capacity is smaller than the requested cache span")
            expected = (
                batch,
                self.config.num_key_value_heads,
                first_key.shape[2],
                self.config.head_dim,
            )
            if any(
                key.shape != expected or value.shape != expected for key, value in cache_key_values
            ):
                raise ValueError(
                    "Qwen3 MoE KV tensors must match batch, capacity, KV heads, and head width"
                )
        elif cache_position:
            raise ValueError("Qwen3 MoE cache position requires KV storage")
        total_length = cache_position + ids.shape[1]
        if total_length > self.config.max_position_embeddings:
            raise ValueError(
                f"{self.config.architecture} received {ids.shape[1]} new tokens after "
                f"{cache_position} cached tokens; maximum is "
                f"{self.config.max_position_embeddings}"
            )

        hidden = self.embed_tokens(ids)
        if cache_key_values and any(
            key.device != hidden.device or value.device != hidden.device
            for key, value in cache_key_values
        ):
            raise ValueError("Qwen3 MoE KV tensors must be on the model execution device")
        if cache_key_values and any(
            key.dtype != hidden.dtype or value.dtype != hidden.dtype
            for key, value in cache_key_values
        ):
            raise ValueError("Qwen3 MoE KV tensors must use the model execution dtype")
        length = ids.shape[1]
        mask = None
        if length > 1 and cache_position:
            mask = torch.full(
                (length, total_length),
                torch.finfo(hidden.dtype).min / 4,
                dtype=hidden.dtype,
                device=hidden.device,
            ).triu_(cache_position + 1)
        if frequencies is None:
            frequencies = self.causal_frequencies(
                length,
                device=hidden.device,
                start=cache_position,
            )
        else:
            expected_shapes = (
                (1, 1, length, self.config.head_dim),
                (1, 1, length, self.config.head_dim // 2),
                (1, 1, length, self.config.head_dim // 2),
            )
            if len(frequencies) != len(expected_shapes) or any(
                tensor.shape != shape
                or tensor.device != hidden.device
                or tensor.dtype != torch.float32
                for tensor, shape in zip(frequencies, expected_shapes, strict=True)
            ):
                raise ValueError(
                    "Qwen3 MoE frequencies must match token count, head width, device, and "
                    "float32 dtype"
                )

        new_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, layer in enumerate(self.layers):
            layer = cast(Qwen3MoeBlock, layer)
            cache_key, cache_value = cache_key_values[index] if cache_key_values else (None, None)
            hidden, key, value = layer.forward_causal(
                hidden,
                mask,
                frequencies,
                cache_key,
                cache_value,
                cache_position,
                prefetch=prefetch,
            )
            new_key_values.append((key, value))
        return _qwen3_moe_rms_norm(self.norm, hidden), tuple(new_key_values)

    def causal_frequencies(
        self,
        length: int,
        *,
        device: torch.device,
        start: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if type(length) is not int or length < 1 or type(start) is not int or start < 0:
            raise ValueError("Qwen3 MoE frequency span must use positive exact integer bounds")
        if start + length > self.config.max_position_embeddings:
            raise ValueError(
                f"{self.config.architecture} frequency span ends at {start + length}; "
                f"maximum is {self.config.max_position_embeddings}"
            )
        return _rope(
            self.config.head_dim,
            length,
            self.config.rope_theta,
            device=device,
            start=start,
        )


class Qwen3MoeForCausalLM(torch.nn.Module):
    """Original Qwen3-MoE checkpoint structure with an untied language head."""

    def __init__(
        self,
        config: Qwen3MoeConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3MoeModel(
            config,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.lm_head = operations.linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.model(ids))

    def forward_causal(
        self,
        ids: torch.Tensor,
        cache_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]] = (),
        *,
        cache_position: int = 0,
        frequencies: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        prefetch: bool = True,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        hidden, new_key_values = self.model.forward_causal(
            ids,
            cache_key_values,
            cache_position=cache_position,
            frequencies=frequencies,
            prefetch=prefetch,
        )
        return self.lm_head(hidden), new_key_values


__all__ = [
    "Qwen3MoeBlock",
    "Qwen3MoeExpert",
    "Qwen3MoeForCausalLM",
    "Qwen3MoeModel",
    "Qwen3MoeSparseMoeBlock",
]
