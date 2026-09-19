"""Native Qwen text tower and the exact Ovis conditioning policy.

This is the text-only ``Llama2_`` path used by ComfyUI's
``Ovis25_2B`` at b78cec87: grouped-query causal attention, half-split
RoPE, per-head q/k RMS normalization, pre-norm residual blocks, and a
SiLU-gated MLP. State keys match the ``model.*`` subtree after the
assembly planner strips that source prefix.

Multi-capture profiles (Flux2 dev Mistral3-Small and Klein Qwen3)
return the hidden states after each configured layer stacked on a new
dimension, never final-normed, exactly as the reference collects
intermediate layers for ``Flux2TEModel``.

The encoder classes own the behavior outside transformer math: fixed
templates, tokenizer policy, attention masks, and each family's output
transform (Ovis right-pads to 284, zeroes padded outputs, and applies
the reference's post-template slice; the Flux2 encoders fold the three
layer captures into the width). Prompt weighting is disabled
everywhere, and each encoder returns sequence-only ``Conditioning``.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import torch
import torch.nn.functional as F
from dinkster_inference import Conditioning
from dinkster_inference.qwen_bpe import (
    tokenize_flux2_klein_prompt,
    tokenize_ovis_prompt,
    tokenize_z_image_prompt,
)
from dinkster_inference.qwen_text import QwenTextConfig
from dinkster_inference.tekken_bpe import TekkenBpe, tokenize_flux2_dev_prompt

from .attention import AttentionKernel, select_attention
from .model_prefetch import close_prefetch_queue, make_prefetch_queue, prefetch_queue_pop
from .operations import INITLESS, Operations, bound_compute_device, materialized_embedding_weight
from .quant_linear import linear_input_act

_DEFAULT_QWEN_ATTENTION = select_attention("qwen").kernel
_QwenFrequencies = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_QwenLayerFrequencies = _QwenFrequencies | Sequence[_QwenFrequencies]


@dataclass
class _QwenFixedKV:
    key: torch.Tensor
    value: torch.Tensor
    position: torch.Tensor
    seqlen: torch.Tensor


_QwenCausalCache = tuple[torch.Tensor, torch.Tensor] | _QwenFixedKV


def _execution_device(module: torch.nn.Module) -> torch.device:
    bound = bound_compute_device(module)
    if bound is not None:
        return bound
    state = next(module.parameters(recurse=False), None)
    if state is None:
        state = next(module.buffers(recurse=False), None)
    if state is None:
        raise ValueError("Qwen execution module has no direct state")
    return state.device


def _frequencies_for_layer(
    frequencies: _QwenLayerFrequencies,
    layer: int,
) -> _QwenFrequencies:
    if isinstance(frequencies[0], torch.Tensor):
        return cast(_QwenFrequencies, frequencies)
    return cast(Sequence[_QwenFrequencies], frequencies)[layer]


class QwenAttentionConfig(Protocol):
    @property
    def hidden_size(self) -> int: ...

    @property
    def num_attention_heads(self) -> int: ...

    @property
    def num_key_value_heads(self) -> int: ...

    @property
    def head_dim(self) -> int: ...

    @property
    def qkv_bias(self) -> bool: ...

    @property
    def qk_norm(self) -> bool: ...

    @property
    def rms_norm_eps(self) -> float: ...


def _rope(
    head_dim: int,
    length: int,
    theta: float,
    *,
    device: torch.device,
    start: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    numerator = torch.arange(0, head_dim, 2, device=device).float()
    inverse = 1.0 / (theta ** (numerator / head_dim))
    positions = torch.arange(start, start + length, device=device).float()
    frequencies = inverse[:, None] @ positions[None, :]
    frequencies = frequencies.transpose(0, 1)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cosine = embedding.cos().unsqueeze(0).unsqueeze(0)
    sine = embedding.sin().unsqueeze(0).unsqueeze(0)
    half = head_dim // 2
    return cosine, sine[..., :half], -sine[..., half:]


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    frequencies: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine, sine, negative_sine = frequencies
    query_dtype = query.dtype
    key_dtype = key.dtype
    q = query * cosine
    q_half = q.shape[-1] // 2
    q[..., :q_half].addcmul_(query[..., q_half:], negative_sine)
    q[..., q_half:].addcmul_(query[..., :q_half], sine)
    k = key * cosine
    k_half = k.shape[-1] // 2
    k[..., :k_half].addcmul_(key[..., k_half:], negative_sine)
    k[..., k_half:].addcmul_(key[..., :k_half], sine)
    return q.to(query_dtype), k.to(key_dtype)


class QwenAttention(torch.nn.Module):
    _attention_kernel: AttentionKernel

    def __init__(
        self,
        config: QwenAttentionConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_QWEN_ATTENTION,
    ) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        object.__setattr__(self, "_attention_kernel", attention_kernel)
        hidden = config.hidden_size
        query = self.num_heads * self.head_dim
        kv = self.num_kv_heads * self.head_dim
        self.qkv_widths = (query, kv, kv)
        if getattr(config, "merged_qkv", False):
            self.qkv_proj = operations.linear(hidden, query + 2 * kv, bias=config.qkv_bias)
            self.q_proj = self.k_proj = self.v_proj = None
        else:
            self.qkv_proj = None
            self.q_proj = operations.linear(hidden, query, bias=config.qkv_bias)
            self.k_proj = operations.linear(hidden, kv, bias=config.qkv_bias)
            self.v_proj = operations.linear(hidden, kv, bias=config.qkv_bias)
        self.o_proj = operations.linear(query, hidden, bias=False)
        self.q_norm = (
            operations.rms_norm(self.head_dim, eps=config.rms_norm_eps) if config.qk_norm else None
        )
        self.k_norm = (
            operations.rms_norm(self.head_dim, eps=config.rms_norm_eps) if config.qk_norm else None
        )

    def _normalize(self, module: torch.nn.RMSNorm, hidden: torch.Tensor) -> torch.Tensor:
        return module(hidden)

    def _apply_rotary(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        frequencies: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _apply_rope(query, key, frequencies)

    def _attention_is_causal(self, mask: torch.Tensor | None, length: int) -> bool:
        return False

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None,
        frequencies: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch, length, _ = hidden.shape
        if self.qkv_proj is not None:
            q, k, v = self.qkv_proj(hidden).split(self.qkv_widths, dim=-1)
        else:
            assert self.q_proj is not None and self.k_proj is not None and self.v_proj is not None
            q, k, v = self.q_proj(hidden), self.k_proj(hidden), self.v_proj(hidden)
        query = q.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        key = k.view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = v.view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            query = self._normalize(self.q_norm, query)
        if self.k_norm is not None:
            key = self._normalize(self.k_norm, key)
        query, key = self._apply_rotary(query, key, frequencies)
        output = self._attention_kernel(
            query,
            key,
            value,
            mask=mask,
            causal=False,
            enable_gqa=self.num_heads != self.num_kv_heads,
        )
        return self.o_proj(output.transpose(1, 2).reshape(batch, length, -1))

    def forward_causal(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None,
        frequencies: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        cache: _QwenCausalCache | None,
        cache_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Attend over a caller-owned cache and return only the new KV rows."""
        batch, length, _ = hidden.shape
        if self.qkv_proj is not None:
            q, k, v = self.qkv_proj(hidden).split(self.qkv_widths, dim=-1)
        else:
            assert self.q_proj is not None and self.k_proj is not None and self.v_proj is not None
            q, k, v = self.q_proj(hidden), self.k_proj(hidden), self.v_proj(hidden)
        query = q.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        key = k.view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = v.view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            query = self._normalize(self.q_norm, query)
        if self.k_norm is not None:
            key = self._normalize(self.k_norm, key)
        query, key = self._apply_rotary(query, key, frequencies)
        if cache is None:
            attention_key = key
            attention_value = value
        elif isinstance(cache, _QwenFixedKV):
            end = cache_position + length
            cache.position.copy_(cache.seqlen)
            cache.seqlen.add_(length)
            transposed_key = key.transpose(1, 2)
            transposed_value = value.transpose(1, 2)
            if length == 1 and cache_position > 0 and mask is None:
                dinkster_kitchen = importlib.import_module("dinkster_kitchen")
                position = cache.position.view(batch, 1, 1, 1).expand_as(transposed_key)
                cache.key.scatter_(1, position, transposed_key)
                cache.value.scatter_(1, position, transposed_value)
                output = dinkster_kitchen.flash_attention_decode(
                    query.transpose(1, 2), cache.key, cache.value, cache.seqlen
                )
                projected = self.o_proj(output.reshape(batch, length, -1))
                return projected, key, value
            cache.key[:, cache_position:end].copy_(transposed_key.detach())
            cache.value[:, cache_position:end].copy_(transposed_value.detach())
            attention_key = cache.key[:, :end].transpose(1, 2)
            attention_value = cache.value[:, :end].transpose(1, 2)
        else:
            cache_key, cache_value = cache
            end = cache_position + length
            cache_key[:, :, cache_position:end].copy_(key.detach())
            cache_value[:, :, cache_position:end].copy_(value.detach())
            attention_key = cache_key[:, :, :end]
            attention_value = cache_value[:, :, :end]
        output = self._attention_kernel(
            query,
            attention_key,
            attention_value,
            mask=mask,
            causal=self._attention_is_causal(mask, length),
            enable_gqa=self.num_heads != self.num_kv_heads,
        )
        projected = self.o_proj(output.transpose(1, 2).reshape(batch, length, -1))
        return projected, key, value


class QwenMlp(torch.nn.Module):
    def __init__(self, config: QwenTextConfig, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        if config.merged_mlp:
            self.gate_up_proj = operations.linear(
                config.hidden_size, 2 * config.intermediate_size, bias=False
            )
            self.gate_proj = self.up_proj = None
        else:
            self.gate_up_proj = None
            self.gate_proj = operations.linear(
                config.hidden_size, config.intermediate_size, bias=False
            )
            self.up_proj = operations.linear(
                config.hidden_size, config.intermediate_size, bias=False
            )
        self.down_proj = operations.linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.gate_up_proj is not None:
            return linear_input_act(
                self.down_proj,
                self.gate_up_proj(hidden),
                "swiglu",
            )
        assert self.gate_proj is not None and self.up_proj is not None
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class QwenBlock(torch.nn.Module):
    def __init__(
        self,
        config: QwenTextConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_QWEN_ATTENTION,
    ) -> None:
        super().__init__()
        self.self_attn = QwenAttention(
            config, operations=operations, attention_kernel=attention_kernel
        )
        self.mlp = QwenMlp(config, operations=operations)
        self.input_layernorm = operations.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = operations.rms_norm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None,
        frequencies: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), mask, frequencies)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))

    def forward_causal(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None,
        frequencies: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        cache: _QwenCausalCache | None,
        cache_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attention, key, value = self.self_attn.forward_causal(
            self.input_layernorm(hidden),
            mask,
            frequencies,
            cache,
            cache_position,
        )
        hidden = hidden + attention
        return hidden + self.mlp(self.post_attention_layernorm(hidden)), key, value


class QwenTextModel(torch.nn.Module):
    """The exact model-scoped state layout shared with the detector."""

    def __init__(
        self,
        config: QwenTextConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_QWEN_ATTENTION,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = operations.embedding(config.vocab_size, config.hidden_size)
        self.layers = torch.nn.ModuleList(
            QwenBlock(config, operations=operations, attention_kernel=attention_kernel)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = operations.rms_norm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        *,
        embeds: torch.Tensor | None = None,
        hidden_layer: int | None = None,
    ) -> torch.Tensor:
        if (ids is None) == (embeds is None):
            raise ValueError("Qwen execution requires exactly one of ids or embeds")
        if ids is not None and ids.ndim != 2:
            raise ValueError(f"Qwen ids must be [batch x tokens], got {tuple(ids.shape)}")
        if embeds is not None and (embeds.ndim != 3 or embeds.shape[2] != self.config.hidden_size):
            raise ValueError("Qwen embeds must be [batch x tokens x hidden]")
        input_shape = ids.shape if ids is not None else cast(torch.Tensor, embeds).shape[:2]
        if attention_mask is not None and attention_mask.shape != input_shape:
            raise ValueError(
                "Qwen attention mask must match input, got"
                f" {tuple(attention_mask.shape)} vs {tuple(input_shape)}"
            )
        if input_shape[1] > self.config.max_position_embeddings:
            raise ValueError(
                f"{self.config.architecture} received {input_shape[1]} tokens; "
                f"maximum is {self.config.max_position_embeddings}"
            )
        if ids is not None:
            embedding_device = _execution_device(self.embed_tokens)
            if ids.device != embedding_device:
                ids = ids.to(embedding_device)
            hidden = self.embed_tokens(ids)
        else:
            assert embeds is not None
            hidden = embeds
        layer_devices = tuple(
            _execution_device(cast(QwenBlock, layer).input_layernorm) for layer in self.layers
        )
        if embeds is not None:
            hidden = hidden.to(layer_devices[0])
        length = input_shape[1]
        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(device=hidden.device, dtype=hidden.dtype).reshape(
                attention_mask.shape[0], 1, 1, length
            ).expand(attention_mask.shape[0], 1, length, length)
            mask = mask.masked_fill(mask.to(torch.bool), torch.finfo(hidden.dtype).min / 4)
        if length > 1:
            causal = torch.full(
                (length, length),
                torch.finfo(hidden.dtype).min / 4,
                dtype=hidden.dtype,
                device=hidden.device,
            ).triu_(1)
            mask = causal if mask is None else mask + causal
        masks_by_device = (
            {}
            if mask is None
            else {device: mask.to(device) for device in dict.fromkeys(layer_devices)}
        )
        frequencies_by_device = {
            device: _rope(
                self.config.head_dim,
                length,
                self.config.rope_theta,
                device=device,
            )
            for device in dict.fromkeys(layer_devices)
        }
        if hidden_layer is not None and type(hidden_layer) is not int:
            raise ValueError("hidden_layer must be an integer or None")
        capture = self.config.output_hidden_layer if hidden_layer is None else hidden_layer
        if hidden_layer is not None and not -len(self.layers) <= hidden_layer < len(self.layers):
            raise ValueError("hidden_layer must address a Qwen model layer")
        if hidden_layer is not None and self.config.output_hidden_layers is not None:
            raise ValueError("hidden_layer cannot override a multi-capture Qwen profile")
        if capture is not None and capture < 0:
            capture += len(self.layers)
        captures = self.config.output_hidden_layers
        collected: dict[int, torch.Tensor] = {}
        output: torch.Tensor | None = None
        prefetch = make_prefetch_queue(self.layers)
        try:
            for index, layer in enumerate(self.layers):
                prefetch_queue_pop(prefetch, layer)
                device = layer_devices[index]
                if hidden.device != device:
                    hidden = hidden.to(device)
                hidden = layer(
                    hidden,
                    None if mask is None else masks_by_device[device],
                    frequencies_by_device[device],
                )
                if index == capture:
                    output = hidden
                if captures is not None and index in captures:
                    collected[index] = hidden
            prefetch_queue_pop(prefetch, None)
        finally:
            close_prefetch_queue(prefetch)
        norm_captures = self.config.layer_norm_hidden_state and self.config.final_norm
        if captures is not None:
            norm_device = _execution_device(self.norm)
            output_device = norm_device if norm_captures else collected[captures[-1]].device
            stacked = torch.stack(
                [collected[index].to(output_device) for index in captures],
                dim=1,
            )
            return self.norm(stacked) if norm_captures else stacked
        if output is None:
            if not self.config.final_norm:
                return hidden
            norm_device = _execution_device(self.norm)
            return self.norm(hidden.to(norm_device))
        if not norm_captures:
            return output
        norm_device = _execution_device(self.norm)
        return self.norm(output.to(norm_device))

    def forward_causal(
        self,
        ids: torch.Tensor | None,
        cache_key_values: Sequence[_QwenCausalCache] = (),
        *,
        cache_position: int = 0,
        frequencies: _QwenLayerFrequencies | None = None,
        prefetch: bool = True,
        embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        """Return final hidden states and new KV rows for incremental generation."""
        if (ids is None) == (embeds is None):
            raise ValueError("Qwen causal execution requires exactly one of ids or embeds")
        if ids is not None and (ids.ndim != 2 or ids.shape[1] == 0):
            raise ValueError(
                f"Qwen causal ids must be [batch x nonempty tokens], got {tuple(ids.shape)}"
            )
        if embeds is not None and (
            embeds.ndim != 3 or embeds.shape[1] == 0 or embeds.shape[2] != self.config.hidden_size
        ):
            raise ValueError("Qwen causal embeds must be [batch x nonempty tokens x hidden]")
        if type(cache_position) is not int or cache_position < 0:
            raise ValueError("Qwen causal cache position must be a non-negative exact integer")
        if len(cache_key_values) not in (0, len(self.layers)):
            raise ValueError("Qwen causal KV must be empty or contain one pair per model layer")
        if ids is not None:
            batch, length = ids.shape
        else:
            assert embeds is not None
            batch, length = embeds.shape[:2]
        pairs: tuple[tuple[torch.Tensor, torch.Tensor], ...] = ()
        if cache_key_values:
            fixed = isinstance(cache_key_values[0], _QwenFixedKV)
            if any(isinstance(cache, _QwenFixedKV) != fixed for cache in cache_key_values):
                raise ValueError("Qwen causal KV cache layouts must match across model layers")
            pairs = tuple(
                (cache.key, cache.value) if isinstance(cache, _QwenFixedKV) else cache
                for cache in cache_key_values
            )
            if any(key.ndim != 4 or value.ndim != 4 for key, value in pairs):
                raise ValueError("Qwen causal KV tensors must be rank four")
            capacity_axis = 1 if fixed else 2
            if any(key.shape[capacity_axis] < cache_position + length for key, _value in pairs):
                raise ValueError("Qwen causal KV capacity is smaller than the requested cache span")
            expected_prefix = (
                (batch, pairs[0][0].shape[1], self.config.num_key_value_heads)
                if fixed
                else (batch, self.config.num_key_value_heads, pairs[0][0].shape[2])
            )
            if any(
                key.shape[:3] != expected_prefix
                or key.shape[3] != self.config.head_dim
                or value.shape != key.shape
                for key, value in pairs
            ):
                raise ValueError(
                    "Qwen causal KV tensors must match batch, capacity, KV heads, and head width"
                )
        elif cache_position:
            raise ValueError("Qwen causal cache position requires KV storage")
        total_length = cache_position + length
        if total_length > self.config.max_position_embeddings:
            raise ValueError(
                f"{self.config.architecture} received {length} new tokens after "
                f"{cache_position} cached tokens; maximum is {self.config.max_position_embeddings}"
            )
        layer_devices = tuple(
            _execution_device(cast(QwenBlock, layer).input_layernorm) for layer in self.layers
        )
        if ids is not None:
            embedding_device = _execution_device(self.embed_tokens)
            if ids.device != embedding_device:
                ids = ids.to(embedding_device)
            hidden = self.embed_tokens(ids)
        else:
            assert embeds is not None
            hidden = embeds.to(layer_devices[0])
        if cache_key_values and any(
            key.device != device
            or value.device != device
            or key.dtype != hidden.dtype
            or value.dtype != hidden.dtype
            for (key, value), device in zip(pairs, layer_devices, strict=True)
        ):
            raise ValueError(
                "Qwen causal KV tensors must match each layer's device and model execution dtype"
            )
        mask = None
        if length > 1:
            mask = torch.full(
                (length, total_length),
                torch.finfo(hidden.dtype).min / 4,
                dtype=hidden.dtype,
                device=hidden.device,
            ).triu_(cache_position + 1)
        if attention_mask is not None:
            if attention_mask.shape != (batch, total_length):
                raise ValueError("Qwen causal attention mask must cover the full cached span")
            padding = 1.0 - attention_mask.to(device=hidden.device, dtype=hidden.dtype)
            padding = padding[:, None, None, :].expand(batch, 1, length, total_length)
            padding = padding.masked_fill(padding.to(torch.bool), torch.finfo(hidden.dtype).min / 4)
            mask = padding if mask is None else mask + padding
        if frequencies is None:
            by_device = {
                device: self.causal_frequencies(
                    length,
                    device=device,
                    start=cache_position,
                )
                for device in dict.fromkeys(layer_devices)
            }
            frequencies = tuple(by_device[device] for device in layer_devices)
        else:
            frequency_items = cast(Sequence[object], frequencies)
            if not frequency_items:
                raise ValueError("Qwen causal frequencies cannot be empty")
            if isinstance(frequency_items[0], torch.Tensor):
                valid = len(frequency_items) == 3 and all(
                    isinstance(item, torch.Tensor) for item in frequency_items
                )
            else:
                valid = len(frequency_items) == len(self.layers) and all(
                    isinstance(item, Sequence)
                    and len(item) == 3
                    and all(isinstance(tensor, torch.Tensor) for tensor in item)
                    for item in frequency_items
                )
            if not valid:
                raise ValueError(
                    "Qwen causal frequencies must be one span or one span per model layer"
                )
        masks_by_device = (
            {}
            if mask is None
            else {device: mask.to(device) for device in dict.fromkeys(layer_devices)}
        )
        new_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        prefetch_queue = make_prefetch_queue(self.layers) if prefetch else None
        try:
            for index, layer in enumerate(self.layers):
                layer = cast(QwenBlock, layer)
                prefetch_queue_pop(prefetch_queue, layer)
                device = layer_devices[index]
                if hidden.device != device:
                    hidden = hidden.to(device)
                layer_mask = None if mask is None else masks_by_device[device]
                layer_frequencies = _frequencies_for_layer(frequencies, index)
                expected_frequency_shapes = (
                    (1, 1, length, self.config.head_dim),
                    (1, 1, length, self.config.head_dim // 2),
                    (1, 1, length, self.config.head_dim // 2),
                )
                if any(
                    tensor.shape != shape
                    or tensor.device != device
                    or tensor.dtype != torch.float32
                    for tensor, shape in zip(
                        layer_frequencies,
                        expected_frequency_shapes,
                        strict=True,
                    )
                ):
                    raise ValueError(
                        "Qwen causal frequencies must match token count, head width, "
                        "layer device, and float32 dtype"
                    )
                cache = cache_key_values[index] if cache_key_values else None
                hidden, key, value = layer.forward_causal(
                    hidden,
                    layer_mask,
                    layer_frequencies,
                    cache,
                    cache_position,
                )
                new_key_values.append((key, value))
            prefetch_queue_pop(prefetch_queue, None)
        finally:
            close_prefetch_queue(prefetch_queue)
        if self.config.final_norm:
            norm_device = _execution_device(self.norm)
            if hidden.device != norm_device:
                hidden = hidden.to(norm_device)
            hidden = self.norm(hidden)
        return hidden, tuple(new_key_values)

    def allocate_causal_cache(
        self,
        batch: int,
        capacity: int,
        *,
        dtype: torch.dtype,
        fixed: bool = False,
    ) -> tuple[_QwenCausalCache, ...]:
        """Allocate caller-owned KV storage on each layer's execution device."""
        if type(batch) is not int or batch < 1 or type(capacity) is not int or capacity < 1:
            raise ValueError("Qwen causal cache batch and capacity must be positive integers")
        if capacity > self.config.max_position_embeddings:
            raise ValueError(
                f"{self.config.architecture} causal cache capacity {capacity} exceeds "
                f"maximum {self.config.max_position_embeddings}"
            )
        devices = tuple(
            _execution_device(cast(QwenBlock, layer).input_layernorm) for layer in self.layers
        )
        dinkster_kitchen = importlib.import_module("dinkster_kitchen") if fixed else None
        use_fixed = dinkster_kitchen is not None and all(
            dtype is torch.bfloat16
            and device.type == "cuda"
            and dinkster_kitchen.flash_attention_decode_is_available(device)
            for device in devices
        )
        shape = (
            (batch, capacity, self.config.num_key_value_heads, self.config.head_dim)
            if use_fixed
            else (batch, self.config.num_key_value_heads, capacity, self.config.head_dim)
        )
        return tuple(
            _QwenFixedKV(
                torch.empty(shape, device=device, dtype=dtype),
                torch.empty(shape, device=device, dtype=dtype),
                torch.empty((batch,), device=device, dtype=torch.int64),
                torch.zeros((batch,), device=device, dtype=torch.int32),
            )
            if use_fixed
            else (
                torch.empty(shape, device=device, dtype=dtype),
                torch.empty(shape, device=device, dtype=dtype),
            )
            for device in devices
        )

    def causal_frequencies(
        self,
        length: int,
        *,
        device: torch.device,
        start: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare reusable rotary frequencies for one causal token span."""
        if type(length) is not int or length < 1 or type(start) is not int or start < 0:
            raise ValueError("Qwen causal frequency span must use positive exact integer bounds")
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

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states through this profile's tied language head."""
        if hidden.ndim != 3 or hidden.shape[-1] != self.config.hidden_size:
            raise ValueError("Qwen language-head input must be [batch x tokens x hidden]")
        device = _execution_device(self.embed_tokens)
        if hidden.device != device:
            hidden = hidden.to(device)
        with materialized_embedding_weight(self.embed_tokens) as weight:
            return F.linear(hidden, weight)


class OvisTextEncoder:
    """Raw prompt to Ovis sequence conditioning, with no weighting."""

    def __init__(self, model: QwenTextModel) -> None:
        if model.config.architecture != "ovis_qwen3_2b":
            raise ValueError(
                "OvisTextEncoder requires the ovis_qwen3_2b profile, got"
                f" {model.config.architecture!r}"
            )
        self.model = model

    def encode(self, text: str) -> Conditioning[torch.Tensor]:
        tokens = tokenize_ovis_prompt(text)
        device = (
            bound_compute_device(self.model.embed_tokens) or self.model.embed_tokens.weight.device
        )
        ids = torch.tensor(tokens.ids, dtype=torch.long, device=device).unsqueeze(0)
        attention = torch.tensor(tokens.attention_mask, dtype=torch.long, device=device).unsqueeze(
            0
        )
        output = self.model(ids, attention).float()
        if self.model.config.zero_masked:
            output = output * attention.unsqueeze(-1).to(output.device, dtype=output.dtype)
        return Conditioning(output[:, tokens.slice_start :], None)


def _stacked_conditioning(
    model: QwenTextModel, ids: tuple[int, ...], attention_mask: tuple[int, ...]
) -> Conditioning[torch.Tensor]:
    """Run one row and fold the layer captures into the width.

    Mirrors the reference ``Flux2TEModel.encode_token_weights``: the
    (batch, captures, tokens, hidden) stack becomes
    (batch, tokens, captures * hidden).
    """
    device = bound_compute_device(model.embed_tokens) or model.embed_tokens.weight.device
    id_rows = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    attention = torch.tensor(attention_mask, dtype=torch.long, device=device).unsqueeze(0)
    output = model(id_rows, attention).float()
    stacked = output.movedim(1, 2).reshape(output.shape[0], output.shape[2], -1)
    return Conditioning(stacked, None)


class Flux2DevTextEncoder:
    """Raw prompt to Flux2 dev stacked Mistral3 conditioning.

    The tekken tokenizer is checkpoint-derived, so callers must hand it
    in; there is no vendored default here.
    """

    def __init__(self, model: QwenTextModel, tokenizer: TekkenBpe) -> None:
        if model.config.architecture not in ("mistral3_24b", "mistral3_24b_pruned"):
            raise ValueError(
                "Flux2DevTextEncoder requires a mistral3_24b profile, got"
                f" {model.config.architecture!r}"
            )
        self.model = model
        self._tokenizer = tokenizer

    def encode(self, text: str) -> Conditioning[torch.Tensor]:
        tokens = tokenize_flux2_dev_prompt(text, tokenizer=self._tokenizer)
        return _stacked_conditioning(self.model, tokens.ids, tokens.attention_mask)


class Flux2KleinTextEncoder:
    """Raw prompt to Flux2 Klein stacked Qwen3 conditioning."""

    def __init__(self, model: QwenTextModel) -> None:
        if model.config.architecture not in ("klein_qwen3_4b", "klein_qwen3_8b"):
            raise ValueError(
                "Flux2KleinTextEncoder requires a Klein Qwen3 profile, got"
                f" {model.config.architecture!r}"
            )
        self.model = model

    def encode(self, text: str) -> Conditioning[torch.Tensor]:
        tokens = tokenize_flux2_klein_prompt(text)
        return _stacked_conditioning(self.model, tokens.ids, tokens.attention_mask)


class ZImageTextEncoder:
    """Raw prompt to Z-Image Qwen3-4B sequence conditioning."""

    def __init__(self, model: QwenTextModel) -> None:
        if model.config.architecture != "z_image_qwen3_4b":
            raise ValueError(
                "ZImageTextEncoder requires the z_image_qwen3_4b profile, got"
                f" {model.config.architecture!r}"
            )
        self.model = model

    def encode(self, text: str) -> Conditioning[torch.Tensor]:
        tokens = tokenize_z_image_prompt(text)
        device = (
            bound_compute_device(self.model.embed_tokens) or self.model.embed_tokens.weight.device
        )
        ids = torch.tensor(tokens.ids, dtype=torch.long, device=device).unsqueeze(0)
        attention = torch.tensor(tokens.attention_mask, dtype=torch.long, device=device).unsqueeze(
            0
        )
        return Conditioning(self.model(ids, attention).float(), None)


__all__ = [
    "Flux2DevTextEncoder",
    "Flux2KleinTextEncoder",
    "OvisTextEncoder",
    "QwenAttention",
    "QwenBlock",
    "QwenMlp",
    "QwenTextModel",
    "ZImageTextEncoder",
]
