"""Unregistered Qwen2.5-VL-7B text and vision source for Qwen Image.

State names and math follow ComfyUI
2a68ce33b4c9ea6ee4283e618a74560cefb32694. The module is intentionally
direct-import-only; assembly, registration, and artifact policy belong to later
Qwen Image stages.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from dinkster_inference.qwen_image_text import (
    QWEN_IMAGE_TEXT_CONFIG,
    QwenImageTextConfig,
    select_qwen_image_output,
)

from .attention import AttentionKernel, select_attention
from .model_prefetch import close_prefetch_queue, make_prefetch_queue, prefetch_queue_pop
from .operations import INITLESS, Operations


def resize_qwen_image_content(
    content: torch.Tensor, *, target_pixels: int, multiple: int = 1
) -> torch.Tensor:
    """Resize one RGB image with the Qwen Edit area-scaling policy."""

    if content.ndim != 4 or content.shape[0] != 1 or content.shape[1] != 3:
        raise ValueError("Qwen Image content must have shape [1,3,height,width]")
    if min(content.shape[2:]) < 1 or not content.is_floating_point():
        raise ValueError("Qwen Image content must have positive floating extents")
    if type(target_pixels) is not int or target_pixels <= 0:
        raise ValueError("Qwen Image target pixels must be a positive integer")
    if type(multiple) is not int or multiple <= 0:
        raise ValueError("Qwen Image resize multiple must be a positive integer")
    scale = math.sqrt(target_pixels / (content.shape[2] * content.shape[3]))
    width = max(multiple, round(content.shape[3] * scale / multiple) * multiple)
    height = max(multiple, round(content.shape[2] * scale / multiple) * multiple)
    return F.interpolate(content, size=(height, width), mode="area")


def prepare_qwen_image_vision(
    content: torch.Tensor, *, target_pixels: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce Qwen2.5-VL flattened patches and its image grid."""

    resized = resize_qwen_image_content(content, target_pixels=target_pixels)
    factor = 28
    height = max(factor, round(resized.shape[2] / factor) * factor)
    width = max(factor, round(resized.shape[3] / factor) * factor)
    image = F.interpolate(resized, size=(height, width), mode="bilinear", align_corners=False)
    mean = image.new_tensor((0.48145466, 0.4578275, 0.40821073)).reshape(1, 3, 1, 1)
    std = image.new_tensor((0.26862954, 0.26130258, 0.27577711)).reshape(1, 3, 1, 1)
    normalized = (image - mean) / std
    grid_height = height // 14
    grid_width = width // 14
    patches = normalized.repeat(2, 1, 1, 1).reshape(
        1,
        2,
        3,
        grid_height // 2,
        2,
        14,
        grid_width // 2,
        2,
        14,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(grid_height * grid_width, 1176)
    grid = torch.tensor(((1, grid_height, grid_width),), dtype=torch.long, device=content.device)
    return patches, grid


_DEFAULT_QWEN_ATTENTION = select_attention("qwen").kernel


def _validate_rope_dims(head_dim: int, rope_dims: tuple[int, int, int]) -> None:
    if (
        len(rope_dims) != 3
        or any(type(value) is not int or value <= 0 for value in rope_dims)
        or sum(rope_dims) * 2 != head_dim
    ):
        raise ValueError("Qwen Image M-RoPE dimensions must cover half the head width")


def _language_rope(
    position_ids: torch.Tensor,
    *,
    head_dim: int,
    theta: float,
    rope_dims: tuple[int, int, int],
    device: torch.device,
    interleaved_mrope: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_rope_dims(head_dim, rope_dims)
    if position_ids.ndim != 2 or position_ids.shape[0] not in (1, 3):
        raise ValueError("Qwen Image position IDs must be [1|3, tokens]")
    numerator = torch.arange(0, head_dim, 2, device=device).float()
    inverse = 1.0 / (theta ** (numerator / head_dim))
    frequencies = (
        inverse[None, :, None] @ position_ids.to(device=device, dtype=torch.float32)[:, None, :]
    ).transpose(1, 2)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cosine = embedding.cos()
    sine = embedding.sin()
    if position_ids.shape[0] == 3 and interleaved_mrope:
        interleaved = frequencies[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            end = rope_dims[axis] * 3
            interleaved[..., offset:end:3] = frequencies[axis, ..., offset:end:3]
        embedding = torch.cat((interleaved, interleaved), dim=-1)
        cosine = embedding.cos().unsqueeze(0)
        sine = embedding.sin().unsqueeze(0)
    elif position_ids.shape[0] == 3:
        sections = rope_dims * 2
        cosine = torch.cat(
            [part[index % 3] for index, part in enumerate(cosine.split(sections, dim=-1))],
            dim=-1,
        ).unsqueeze(0)
        sine = torch.cat(
            [part[index % 3] for index, part in enumerate(sine.split(sections, dim=-1))],
            dim=-1,
        ).unsqueeze(0)
    else:
        cosine = cosine.unsqueeze(1)
        sine = sine.unsqueeze(1)
    half = sine.shape[-1] // 2
    return cosine, sine[..., :half], -sine[..., half:]


def _apply_language_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    frequencies: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine, sine, negative_sine = frequencies

    def apply(value: torch.Tensor) -> torch.Tensor:
        # Fused multiply-add matches the reference's single rounding
        # (comfy/text_encoders/llama.py apply_rope @ b78cec87); separate
        # mul+add drifts by one ulp on ~12% of roped elements.
        original_dtype = value.dtype
        embedded = value * cosine
        half = embedded.shape[-1] // 2
        embedded[..., :half].addcmul_(value[..., half:], negative_sine)
        embedded[..., half:].addcmul_(value[..., :half], sine)
        return embedded.to(original_dtype)

    return apply(query), apply(key)


class QwenImageLanguageAttention(torch.nn.Module):
    """Grouped-query language attention with Qwen2.5-VL M-RoPE."""

    _attention_kernel: AttentionKernel

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_QWEN_ATTENTION,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads:
            raise ValueError("Qwen Image language head geometry is inconsistent")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        object.__setattr__(self, "_attention_kernel", attention_kernel)
        inner_size = num_heads * head_dim
        self.q_proj = operations.linear(hidden_size, inner_size, bias=qkv_bias)
        self.k_proj = operations.linear(hidden_size, num_kv_heads * head_dim, bias=qkv_bias)
        self.v_proj = operations.linear(hidden_size, num_kv_heads * head_dim, bias=qkv_bias)
        self.o_proj = operations.linear(inner_size, hidden_size, bias=False)
        if qk_norm:
            self.q_norm = operations.rms_norm(head_dim, eps=rms_norm_eps)
            self.k_norm = operations.rms_norm(head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        *,
        rope_theta: float,
        rope_dims: tuple[int, int, int],
        interleaved_mrope: bool = False,
    ) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError("Qwen Image language hidden states must be [batch, tokens, width]")
        batch, length, _ = hidden.shape
        query = self.q_proj(hidden).view(batch, length, self.num_heads, self.head_dim)
        key = self.k_proj(hidden).view(batch, length, self.num_kv_heads, self.head_dim)
        value = self.v_proj(hidden).view(batch, length, self.num_kv_heads, self.head_dim)
        if self.q_norm is not None and self.k_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        frequencies = _language_rope(
            position_ids,
            head_dim=self.head_dim,
            theta=rope_theta,
            rope_dims=rope_dims,
            device=hidden.device,
            interleaved_mrope=interleaved_mrope,
        )
        query, key = _apply_language_rope(query, key, frequencies)
        output = self._attention_kernel(
            query,
            key,
            value,
            mask=mask,
            causal=False,
            enable_gqa=self.num_heads != self.num_kv_heads,
        )
        return self.o_proj(output.transpose(1, 2).reshape(batch, length, -1))


class _LanguageMlp(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.gate_proj = operations.linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = operations.linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = operations.linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class _LanguageBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        qkv_bias: bool,
        qk_norm: bool,
        interleaved_mrope: bool,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.self_attn = QwenImageLanguageAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            rms_norm_eps=rms_norm_eps,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.mlp = _LanguageMlp(hidden_size, intermediate_size, operations=operations)
        self.input_layernorm = operations.rms_norm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = operations.rms_norm(hidden_size, eps=rms_norm_eps)
        self.interleaved_mrope = interleaved_mrope

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        *,
        rope_theta: float,
        rope_dims: tuple[int, int, int],
    ) -> torch.Tensor:
        hidden = hidden + self.self_attn(
            self.input_layernorm(hidden),
            mask,
            position_ids,
            rope_theta=rope_theta,
            rope_dims=rope_dims,
            interleaved_mrope=self.interleaved_mrope,
        )
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


@dataclass(frozen=True)
class _LanguageShape:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    rope_dims: tuple[int, int, int]
    qkv_bias: bool
    qk_norm: bool
    final_norm: bool
    interleaved_mrope: bool
    max_position_embeddings: int | None
    architecture: str


class QwenImageLanguageModel(torch.nn.Module):
    """Operations-backed Qwen2.5-VL language tower."""

    def __init__(
        self,
        config: QwenImageTextConfig = QWEN_IMAGE_TEXT_CONFIG,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_QWEN_ATTENTION,
        _shape: _LanguageShape | None = None,
    ) -> None:
        super().__init__()
        shape = _shape or _LanguageShape(
            config.vocab_size,
            config.hidden_size,
            config.intermediate_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.rms_norm_eps,
            config.rope_theta,
            config.rope_dims,
            config.qkv_bias,
            False,
            True,
            False,
            config.max_position_embeddings,
            "Qwen Image",
        )
        _validate_rope_dims(shape.head_dim, shape.rope_dims)
        self.shape = shape
        self.embed_tokens = operations.embedding(shape.vocab_size, shape.hidden_size)
        self.layers = torch.nn.ModuleList(
            _LanguageBlock(
                hidden_size=shape.hidden_size,
                intermediate_size=shape.intermediate_size,
                num_heads=shape.num_heads,
                num_kv_heads=shape.num_kv_heads,
                head_dim=shape.head_dim,
                rms_norm_eps=shape.rms_norm_eps,
                qkv_bias=shape.qkv_bias,
                qk_norm=shape.qk_norm,
                interleaved_mrope=shape.interleaved_mrope,
                operations=operations,
                attention_kernel=attention_kernel,
            )
            for _ in range(shape.num_layers)
        )
        self.norm = (
            operations.rms_norm(shape.hidden_size, eps=shape.rms_norm_eps)
            if shape.final_norm
            else None
        )

    @classmethod
    def reduced(
        cls,
        *,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        rope_dims: tuple[int, int, int],
        head_dim: int | None = None,
        rope_theta: float = QWEN_IMAGE_TEXT_CONFIG.rope_theta,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        final_norm: bool = True,
        interleaved_mrope: bool = False,
        max_position_embeddings: int | None = None,
        architecture: str = "Qwen language model",
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_QWEN_ATTENTION,
    ) -> QwenImageLanguageModel:
        resolved_head_dim = head_dim or hidden_size // num_heads
        shape = _LanguageShape(
            vocab_size,
            hidden_size,
            intermediate_size,
            num_layers,
            num_heads,
            num_kv_heads,
            resolved_head_dim,
            QWEN_IMAGE_TEXT_CONFIG.rms_norm_eps,
            rope_theta,
            rope_dims,
            qkv_bias,
            qk_norm,
            final_norm,
            interleaved_mrope,
            max_position_embeddings,
            architecture,
        )
        return cls(operations=operations, attention_kernel=attention_kernel, _shape=shape)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2:
            raise ValueError("Qwen Image language IDs must be rank 2")
        self.validate_sequence_length(ids.shape[1])
        return self.embed_tokens(ids)

    def validate_sequence_length(self, length: int) -> None:
        limit = self.shape.max_position_embeddings
        if limit is not None and length > limit:
            raise ValueError(
                f"{self.shape.architecture} received {length} tokens; maximum is {limit}"
            )

    def _attention_mask(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor | None:
        batch, length, _ = hidden.shape
        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(hidden.dtype).reshape(batch, 1, 1, length)
            mask = mask.expand(batch, 1, length, length)
            mask = mask.masked_fill(mask.to(torch.bool), torch.finfo(hidden.dtype).min / 4)
        if length > 1:
            causal = torch.full(
                (length, length),
                torch.finfo(hidden.dtype).min / 4,
                dtype=hidden.dtype,
                device=hidden.device,
            ).triu_(1)
            mask = causal if mask is None else mask + causal
        return mask

    def forward_embeds(
        self,
        embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        *,
        deepstack_features: tuple[torch.Tensor, ...] = (),
        visual_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if embeds.ndim != 3 or embeds.shape[-1] != self.shape.hidden_size:
            raise ValueError("Qwen Image language embeddings have invalid geometry")
        self.validate_sequence_length(embeds.shape[1])
        if attention_mask is not None and attention_mask.shape != embeds.shape[:2]:
            raise ValueError("Qwen Image attention mask must match embedding rows")
        if position_ids.shape not in ((1, embeds.shape[1]), (3, embeds.shape[1])):
            raise ValueError("Qwen Image position IDs must match embedding length")
        if deepstack_features:
            if visual_mask is None or visual_mask.shape != embeds.shape[:2]:
                raise ValueError("DeepStack requires a visual mask matching embedding rows")
            visual_tokens = int(visual_mask.count_nonzero())
            if any(
                feature.shape != (visual_tokens, self.shape.hidden_size)
                for feature in deepstack_features
            ):
                raise ValueError("DeepStack features must match visual token geometry")
        elif visual_mask is not None:
            raise ValueError("a visual mask requires DeepStack features")
        mask = self._attention_mask(embeds, attention_mask)
        hidden, _ = self._run_layers(embeds, mask, position_ids, deepstack_features, visual_mask)
        return self.norm(hidden) if self.norm is not None else hidden

    def _run_layers(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        deepstack_features: tuple[torch.Tensor, ...],
        visual_mask: torch.Tensor | None,
        tap_layers: tuple[int, ...] = (),
        *,
        need_final: bool = True,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Run decoder layers, capturing each tapped layer's input.

        Tap ``k`` is the residual stream entering layer ``k`` (the
        reference's ``hidden_states[k]``). When the final state is not
        needed, layers past the last tap are skipped; their output
        cannot influence any capture.
        """

        captured: list[torch.Tensor] = []
        tap_set = frozenset(tap_layers)
        queue = make_prefetch_queue(self.layers)
        try:
            for index, layer in enumerate(self.layers):
                if index in tap_set:
                    captured.append(hidden)
                    if not need_final and len(captured) == len(tap_set):
                        break
                prefetch_queue_pop(queue, layer)
                hidden = layer(
                    hidden,
                    mask,
                    position_ids,
                    rope_theta=self.shape.rope_theta,
                    rope_dims=self.shape.rope_dims,
                )
                if index < len(deepstack_features):
                    assert visual_mask is not None
                    hidden[visual_mask] = hidden[visual_mask] + deepstack_features[index].to(hidden)
            else:
                # The queue's order contract only allows the terminal pop
                # after every layer was popped; early tap exits skip it and
                # rely on close() to release prepared prefetches.
                prefetch_queue_pop(queue, None)
            return hidden, tuple(captured)
        finally:
            close_prefetch_queue(queue)

    def tapped_states(
        self,
        ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        tap_layers: tuple[int, ...],
    ) -> torch.Tensor:
        """Stack tapped residual streams as ``(batch, taps, seq, hidden)``.

        Tap ``k`` is the input to decoder layer ``k``. The terminal tap
        ``num_layers`` is the post-decoder residual before the final norm.
        """

        if ids.ndim != 2:
            raise ValueError("Qwen Image language IDs must be rank 2")
        if attention_mask is not None and attention_mask.shape != ids.shape:
            raise ValueError("Qwen Image attention mask must match IDs")
        return self.tapped_embeds(self.embed(ids), attention_mask, tap_layers=tap_layers)

    def tapped_embeds(
        self,
        embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        tap_layers: tuple[int, ...],
    ) -> torch.Tensor:
        """Stack tapped residual streams for caller-supplied embeddings."""
        if embeds.ndim != 3 or embeds.shape[-1] != self.shape.hidden_size:
            raise ValueError("Qwen Image language embeddings have invalid geometry")
        if attention_mask is not None and attention_mask.shape != embeds.shape[:2]:
            raise ValueError("Qwen Image attention mask must match embedding rows")
        self.validate_sequence_length(embeds.shape[1])
        if not tap_layers or any(
            type(tap) is not int or not 0 <= tap <= self.shape.num_layers for tap in tap_layers
        ):
            raise ValueError("tap layers must be decoder layer indices or the terminal boundary")
        if tuple(sorted(set(tap_layers))) != tap_layers:
            raise ValueError("tap layers must be strictly increasing")
        positions = torch.arange(embeds.shape[1], device=embeds.device).unsqueeze(0)
        mask = self._attention_mask(embeds, attention_mask)
        terminal = tap_layers[-1] == self.shape.num_layers
        hidden, captured = self._run_layers(
            embeds,
            mask,
            positions,
            (),
            None,
            tap_layers[:-1] if terminal else tap_layers,
            need_final=terminal,
        )
        if terminal:
            captured = (*captured, hidden)
        return torch.stack(captured, dim=1)

    def forward(
        self, ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if ids.ndim != 2:
            raise ValueError("Qwen Image language IDs must be rank 2")
        if attention_mask is not None and attention_mask.shape != ids.shape:
            raise ValueError("Qwen Image attention mask must match IDs")
        hidden = self.embed(ids)
        positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        return self.forward_embeds(hidden, attention_mask, positions)


class _VisionPatchEmbed(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch: tuple[int, int, int],
        *,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.patch = patch
        self.hidden_size = hidden_size
        self.proj = operations.linear(3 * math.prod(patch), hidden_size, bias=False)
        self.register_state_dict_post_hook(self._reshape_saved_weight)
        self.register_load_state_dict_pre_hook(self._reshape_loaded_weight)

    @staticmethod
    def _reshape_saved_weight(
        module: torch.nn.Module,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
    ) -> None:
        patch_embed = cast(_VisionPatchEmbed, module)
        key = prefix + "proj.weight"
        state_dict[key] = state_dict[key].reshape(patch_embed.hidden_size, 3, *patch_embed.patch)

    @staticmethod
    def _reshape_loaded_weight(
        module: torch.nn.Module,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_messages: list[str],
    ) -> None:
        patch_embed = cast(_VisionPatchEmbed, module)
        key = prefix + "proj.weight"
        weight = state_dict.get(key)
        expected = (patch_embed.hidden_size, 3, *patch_embed.patch)
        if weight is not None and tuple(weight.shape) == expected:
            state_dict[key] = weight.reshape(patch_embed.hidden_size, -1)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        expected_width = 3 * math.prod(self.patch)
        if patches.ndim != 2 or patches.shape[1] != expected_width:
            raise ValueError(f"Qwen Image vision patches must be [patches, {expected_width}]")
        return self.proj(patches)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class QwenImageVisionAttention(torch.nn.Module):
    """Segmented window/full vision attention through the injected seam."""

    _attention_kernel: AttentionKernel

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_QWEN_ATTENTION,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("Qwen Image vision width must divide into attention heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        object.__setattr__(self, "_attention_kernel", attention_kernel)
        self.qkv = operations.linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = operations.linear(hidden_size, hidden_size, bias=True)

    def forward(
        self,
        hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cumulative_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.ndim != 2:
            raise ValueError("Qwen Image vision hidden states must be [patches, width]")
        if cumulative_lengths.ndim != 1 or cumulative_lengths.numel() < 2:
            raise ValueError("Qwen Image vision segments require cumulative lengths")
        boundaries = tuple(int(value) for value in cumulative_lengths.tolist())
        if (
            boundaries[0] != 0
            or boundaries[-1] != hidden.shape[0]
            or any(left >= right for left, right in zip(boundaries, boundaries[1:], strict=False))
        ):
            raise ValueError("Qwen Image vision segments must cover every patch exactly once")
        qkv = self.qkv(hidden).reshape(hidden.shape[0], 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(1, 0, 2, 3).unbind(0)
        cosine, sine = position_embeddings
        if cosine.shape != (hidden.shape[0], self.head_dim) or sine.shape != cosine.shape:
            raise ValueError("Qwen Image vision RoPE must match patch and head geometry")
        cosine = cosine.unsqueeze(1).float()
        sine = sine.unsqueeze(1).float()
        query = query * cosine + _rotate_half(query) * sine
        key = key * cosine + _rotate_half(key) * sine
        outputs: list[torch.Tensor] = []
        for start, end in zip(boundaries, boundaries[1:], strict=False):
            q = query[start:end].transpose(0, 1).unsqueeze(0)
            k = key[start:end].transpose(0, 1).unsqueeze(0)
            v = value[start:end].transpose(0, 1).unsqueeze(0)
            outputs.append(self._attention_kernel(q, k, v, causal=False))
        output = torch.cat(outputs, dim=2).transpose(1, 2).reshape(hidden.shape[0], -1)
        return self.proj(output)


class _VisionMlp(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, *, operations: Operations) -> None:
        super().__init__()
        self.gate_proj = operations.linear(hidden_size, intermediate_size, bias=True)
        self.up_proj = operations.linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = operations.linear(intermediate_size, hidden_size, bias=True)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class _VisionBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.norm1 = operations.rms_norm(hidden_size, eps=1e-6)
        self.norm2 = operations.rms_norm(hidden_size, eps=1e-6)
        self.attn = QwenImageVisionAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.mlp = _VisionMlp(hidden_size, intermediate_size, operations=operations)

    def forward(
        self,
        hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cumulative_lengths: torch.Tensor,
    ) -> torch.Tensor:
        hidden = hidden + self.attn(self.norm1(hidden), position_embeddings, cumulative_lengths)
        return hidden + self.mlp(self.norm2(hidden))


class _PatchMerger(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        spatial_merge_size: int,
        *,
        operations: Operations,
    ) -> None:
        super().__init__()
        merged = hidden_size * spatial_merge_size**2
        self.hidden_size = merged
        self.ln_q = operations.rms_norm(hidden_size, eps=1e-6)
        self.mlp = torch.nn.Sequential(
            operations.linear(merged, merged),
            torch.nn.GELU(),
            operations.linear(merged, output_size),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ln_q(hidden).reshape(-1, self.hidden_size))


@dataclass(frozen=True)
class _VisionShape:
    hidden_size: int
    output_size: int
    intermediate_size: int
    num_heads: int
    num_layers: int
    patch: tuple[int, int, int]
    spatial_merge_size: int
    window_size: int
    full_attention_blocks: tuple[int, ...]


class QwenImageVisionTransformer(torch.nn.Module):
    """Qwen2.5-VL visual tower with source window ordering."""

    def __init__(
        self,
        config: QwenImageTextConfig = QWEN_IMAGE_TEXT_CONFIG,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_QWEN_ATTENTION,
        _shape: _VisionShape | None = None,
    ) -> None:
        super().__init__()
        shape = _shape or _VisionShape(
            config.vision_hidden_size,
            config.vision_output_size,
            config.vision_intermediate_size,
            config.vision_heads,
            config.vision_layers,
            config.vision_patch,
            config.vision_spatial_merge,
            112,
            (7, 15, 23, 31),
        )
        if shape.hidden_size % shape.num_heads or shape.window_size <= 0:
            raise ValueError("Qwen Image vision configuration is inconsistent")
        self.shape = shape
        self.patch_embed = _VisionPatchEmbed(shape.hidden_size, shape.patch, operations=operations)
        self.blocks = torch.nn.ModuleList(
            _VisionBlock(
                hidden_size=shape.hidden_size,
                intermediate_size=shape.intermediate_size,
                num_heads=shape.num_heads,
                operations=operations,
                attention_kernel=attention_kernel,
            )
            for _ in range(shape.num_layers)
        )
        self.merger = _PatchMerger(
            shape.hidden_size,
            shape.output_size,
            shape.spatial_merge_size,
            operations=operations,
        )
        self.last_attention_segments: tuple[tuple[int, ...], ...] = ()

    @classmethod
    def reduced(
        cls,
        *,
        hidden_size: int,
        output_size: int,
        intermediate_size: int,
        num_heads: int,
        num_layers: int,
        patch: tuple[int, int, int],
        spatial_merge_size: int,
        window_size: int,
        full_attention_blocks: tuple[int, ...],
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_QWEN_ATTENTION,
    ) -> QwenImageVisionTransformer:
        shape = _VisionShape(
            hidden_size,
            output_size,
            intermediate_size,
            num_heads,
            num_layers,
            patch,
            spatial_merge_size,
            window_size,
            full_attention_blocks,
        )
        return cls(operations=operations, attention_kernel=attention_kernel, _shape=shape)

    def _validate_grid(self, patches: torch.Tensor, grid: torch.Tensor) -> None:
        if grid.ndim != 2 or grid.shape[1] != 3 or grid.shape[0] != 1:
            raise ValueError("Qwen Image vision grid must be one [time, height, width] row")
        if grid.dtype == torch.bool or grid.is_floating_point():
            raise ValueError("Qwen Image vision grid must use an integer dtype")
        values = tuple(int(value) for value in grid[0].tolist())
        if any(value <= 0 for value in values):
            raise ValueError("Qwen Image vision grid dimensions must be positive")
        if values[1] % self.shape.spatial_merge_size or values[2] % self.shape.spatial_merge_size:
            raise ValueError("Qwen Image vision grid must divide by the spatial merge size")
        if math.prod(values) != patches.shape[0]:
            raise ValueError("Qwen Image vision grid must account for every patch")

    def _window_index(self, grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        merge = self.shape.spatial_merge_size
        merger_window = self.shape.window_size // merge // self.shape.patch[1]
        if merger_window <= 0:
            raise ValueError("Qwen Image vision window is smaller than one merged patch")
        time, height, width = (int(value) for value in grid[0].tolist())
        llm_height = height // merge
        llm_width = width // merge
        index = torch.arange(time * llm_height * llm_width, device=grid.device).reshape(
            time, llm_height, llm_width
        )
        pad_height = (merger_window - llm_height % merger_window) % merger_window
        pad_width = (merger_window - llm_width % merger_window) % merger_window
        windows_h = (llm_height + pad_height) // merger_window
        windows_w = (llm_width + pad_width) // merger_window
        padded = F.pad(index, (0, pad_width, 0, pad_height), value=-100)
        padded = padded.reshape(time, windows_h, merger_window, windows_w, merger_window)
        padded = padded.permute(0, 1, 3, 2, 4).reshape(
            time, windows_h * windows_w, merger_window, merger_window
        )
        lengths = (padded != -100).sum((2, 3)).reshape(-1) * merge**2
        flat = padded.reshape(-1)
        window_index = flat[flat != -100]
        cumulative = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0), value=0)
        return window_index, torch.unique_consecutive(cumulative)

    def _position_embeddings(self, grid: torch.Tensor, device: torch.device) -> torch.Tensor:
        merge = self.shape.spatial_merge_size
        time, height, width = (int(value) for value in grid[0].tolist())
        height_ids = torch.arange(height, device=device).unsqueeze(1).expand(-1, width)
        height_ids = height_ids.reshape(height // merge, merge, width // merge, merge)
        height_ids = height_ids.permute(0, 2, 1, 3).flatten().repeat(time)
        width_ids = torch.arange(width, device=device).unsqueeze(0).expand(height, -1)
        width_ids = width_ids.reshape(height // merge, merge, width // merge, merge)
        width_ids = width_ids.permute(0, 2, 1, 3).flatten().repeat(time)
        positions = torch.stack((height_ids, width_ids), dim=-1)
        rotary_dim = (self.shape.hidden_size // self.shape.num_heads) // 2
        inverse = 1.0 / (
            10_000.0
            ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim)
        )
        sequence = torch.arange(max(height, width), device=device, dtype=torch.float32)
        frequencies = torch.outer(sequence, inverse)
        selected = frequencies[positions].flatten(1)
        return torch.cat((selected, selected), dim=-1)

    def forward(self, patches: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        expected_width = 3 * math.prod(self.shape.patch)
        if patches.ndim != 2 or patches.shape[1] != expected_width:
            raise ValueError(f"Qwen Image vision patches must be [patches, {expected_width}]")
        self._validate_grid(patches, grid)
        hidden = self.patch_embed(patches)
        window_index, window_lengths = self._window_index(grid.to(hidden.device))
        positions = self._position_embeddings(grid, hidden.device)
        merge_unit = self.shape.spatial_merge_size**2
        sequence_length = hidden.shape[0]
        hidden = hidden.reshape(sequence_length // merge_unit, merge_unit, -1)
        hidden = hidden[window_index].reshape(sequence_length, -1)
        positions = positions.reshape(sequence_length // merge_unit, merge_unit, -1)
        positions = positions[window_index].reshape(sequence_length, -1)
        position_embeddings = (positions.cos(), positions.sin())
        time, height, width = (int(value) for value in grid[0].tolist())
        full_lengths = torch.arange(1, time + 1, device=hidden.device, dtype=torch.int32)
        full_lengths = F.pad(full_lengths * height * width, (1, 0), value=0)
        segments: list[tuple[int, ...]] = []
        for index, block in enumerate(self.blocks):
            lengths = full_lengths if index in self.shape.full_attention_blocks else window_lengths
            segments.append(tuple(int(value) for value in lengths.tolist()))
            hidden = block(hidden, position_embeddings, lengths)
        self.last_attention_segments = tuple(segments)
        hidden = self.merger(hidden)
        return hidden[torch.argsort(window_index)]


def qwen_image_mrope_position_ids(
    *,
    sequence_length: int,
    image_start: int,
    image_length: int,
    image_grid_thw: torch.Tensor,
    attention_mask: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """Build the exact Qwen2.5-VL text/image temporal-height-width positions."""

    if type(sequence_length) is not int or sequence_length <= 0:
        raise ValueError("Qwen Image sequence length must be positive")
    if not 0 <= image_start < image_start + image_length <= sequence_length:
        raise ValueError("Qwen Image span must be non-empty and inside the sequence")
    if image_grid_thw.shape != (1, 3):
        raise ValueError("Qwen Image position grid must be one [time, height, width] row")
    if image_grid_thw.dtype == torch.bool or image_grid_thw.is_floating_point():
        raise ValueError("Qwen Image position grid must use an integer dtype")
    time, height, width = (int(value) for value in image_grid_thw[0].tolist())
    if time != 1 or height <= 0 or width <= 0 or height % 2 or width % 2:
        raise ValueError("Qwen Image position grid must be one positive mergeable image")
    if image_length != time * (height // 2) * (width // 2):
        raise ValueError("Qwen Image span length must match the merged vision grid")
    if attention_mask is not None and attention_mask.shape != (1, sequence_length):
        raise ValueError("Qwen Image position mask must match the expanded sequence")
    positions = torch.ones((3, sequence_length), dtype=torch.long, device=device)
    positions[:, :image_start] = torch.arange(image_start, device=device)
    image_end = image_start + image_length
    length_max = max(time, height, width) // 2
    next_start = image_start + length_max
    positions[0, image_start:image_end] = image_start
    positions[1, image_start:image_end] = (
        torch.arange(image_start, image_start + height // 2, device=device)
        .unsqueeze(1)
        .repeat(1, width // 2)
        .flatten()
    )
    positions[2, image_start:image_end] = (
        torch.arange(image_start, image_start + width // 2, device=device)
        .unsqueeze(0)
        .repeat(height // 2, 1)
        .flatten()
    )
    tail_length = sequence_length - image_end
    if attention_mask is None:
        positions[:, image_end:] = torch.arange(next_start, next_start + tail_length, device=device)
    else:
        tail_mask = attention_mask[0, image_end:].to(device=device)
        text_positions = tail_mask.cumsum(0) - 1 + next_start
        positions[:, image_end:] = torch.where(
            tail_mask.bool(), text_positions, positions[0, image_end:]
        )
    return positions


def _qwen_image_mrope_position_ids_for_spans(
    sequence_length: int,
    spans: Sequence[tuple[int, int, torch.Tensor]],
    device: torch.device,
) -> torch.Tensor:
    positions = torch.zeros((3, sequence_length), dtype=torch.long, device=device)
    first_start = spans[0][0]
    positions[:, :first_start] = torch.arange(first_start, device=device)
    offset = 0
    for start, length, grid in spans:
        end = start + length
        _, height, width = (int(value) for value in grid[0].tolist())
        length_max = max(height, width) // 2
        next_start = start + length_max
        positions[:, end:] = torch.arange(
            next_start + offset,
            next_start + sequence_length - end + offset,
            device=device,
        )
        positions[0, start:end] = start + offset
        positions[1, start:end] = (
            torch.arange(start + offset, start + height // 2 + offset, device=device)
            .unsqueeze(1)
            .repeat(1, width // 2)
            .flatten()
        )
        positions[2, start:end] = (
            torch.arange(start + offset, start + width // 2 + offset, device=device)
            .unsqueeze(0)
            .repeat(height // 2, 1)
            .flatten()
        )
        offset += length_max - length
    return positions


class QwenImageTextModel(torch.nn.Module):
    """Exact state container and base Qwen Image text execution policy."""

    def __init__(
        self,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_QWEN_ATTENTION,
        language: QwenImageLanguageModel | None = None,
        visual: QwenImageVisionTransformer | None = None,
    ) -> None:
        super().__init__()
        self.model = language or QwenImageLanguageModel(
            operations=operations, attention_kernel=attention_kernel
        )
        self.visual = visual or QwenImageVisionTransformer(
            operations=operations, attention_kernel=attention_kernel
        )
        self.active_image_features: torch.Tensor | None = None

    def _validate_inputs(
        self,
        ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        image_patches: Sequence[torch.Tensor],
        image_grid_thw: Sequence[torch.Tensor],
    ) -> tuple[tuple[int, ...], int]:
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
            raise ValueError("Qwen Image IDs must be one non-empty rank 2 row")
        if ids.dtype == torch.bool or ids.is_floating_point():
            raise ValueError("Qwen Image IDs must use an integer dtype")
        if attention_mask is not None:
            if attention_mask.shape != ids.shape:
                raise ValueError("Qwen Image attention mask must match IDs")
            if attention_mask.dtype == torch.bool:
                pass
            elif attention_mask.is_floating_point() or not bool(
                torch.all((attention_mask == 0) | (attention_mask == 1))
            ):
                raise ValueError("Qwen Image attention mask must be binary integers")
        placeholders = torch.nonzero(
            ids[0] == QWEN_IMAGE_TEXT_CONFIG.image_token_id, as_tuple=False
        ).flatten()
        if not image_patches:
            if placeholders.numel():
                raise ValueError("Qwen Image placeholder requires image patches")
            return (), ids.shape[1]
        if len(image_patches) != len(image_grid_thw):
            raise ValueError("Qwen Image patches and grids must have equal cardinality")
        if not 1 <= len(image_patches) <= 3 or placeholders.numel() != len(image_patches):
            raise ValueError(
                "Qwen Image execution requires one to three matched image placeholders"
            )
        expected_width = 3 * math.prod(QWEN_IMAGE_TEXT_CONFIG.vision_patch)
        expanded_length = ids.shape[1] - len(image_patches)
        for patches, grid in zip(image_patches, image_grid_thw, strict=True):
            if patches.ndim != 2 or patches.shape[1] != expected_width:
                raise ValueError(f"Qwen Image patches must be [patches, {expected_width}]")
            if grid.shape != (1, 3):
                raise ValueError("Qwen Image grid must be one [time, height, width] row")
            if grid.dtype == torch.bool or grid.is_floating_point():
                raise ValueError("Qwen Image grid must use an integer dtype")
            time, height, width = (int(value) for value in grid[0].tolist())
            merge = QWEN_IMAGE_TEXT_CONFIG.vision_spatial_merge
            if time != 1 or height <= 0 or width <= 0:
                raise ValueError("Qwen Image grid must describe one positive image")
            if height % merge or width % merge:
                raise ValueError("Qwen Image grid must divide by the spatial merge size")
            if patches.shape[0] != time * height * width:
                raise ValueError("Qwen Image grid must account for every patch")
            expanded_length += patches.shape[0] // (merge * merge)
        return tuple(int(value) for value in placeholders), expanded_length

    def forward(
        self,
        ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        image_patches: torch.Tensor | Sequence[torch.Tensor] | None = None,
        image_grid_thw: torch.Tensor | Sequence[torch.Tensor] | None = None,
        template_end: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if (image_patches is None) != (image_grid_thw is None):
            raise ValueError("Qwen Image patches and grid must be provided together")
        patches = (
            ()
            if image_patches is None
            else (image_patches,)
            if isinstance(image_patches, torch.Tensor)
            else tuple(image_patches)
        )
        grids = (
            ()
            if image_grid_thw is None
            else (image_grid_thw,)
            if isinstance(image_grid_thw, torch.Tensor)
            else tuple(image_grid_thw)
        )
        placeholders, expanded_length = self._validate_inputs(ids, attention_mask, patches, grids)
        self.model.validate_sequence_length(expanded_length)
        selection = select_qwen_image_output(
            cast(list[list[int]], ids.detach().cpu().tolist()),
            None
            if attention_mask is None
            else cast(list[list[int]], attention_mask.detach().cpu().to(torch.long).tolist()),
            template_end=template_end,
        )
        embeddings = self.model.embed(ids)
        expanded_mask = attention_mask
        position_ids = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        try:
            if placeholders:
                features = tuple(
                    self.visual(
                        value.to(device=embeddings.device, dtype=torch.float32),
                        grid.to(device=embeddings.device),
                    ).to(dtype=embeddings.dtype)
                    for value, grid in zip(patches, grids, strict=True)
                )
                self.active_image_features = torch.cat(features)
                embed_parts: list[torch.Tensor] = []
                mask_parts: list[torch.Tensor] = []
                spans: list[tuple[int, int, torch.Tensor]] = []
                source_start = 0
                expanded_start = 0
                for placeholder, feature, grid in zip(placeholders, features, grids, strict=True):
                    text_part = embeddings[:, source_start:placeholder]
                    embed_parts.extend((text_part, feature.unsqueeze(0)))
                    if attention_mask is not None:
                        mask_parts.extend(
                            (
                                attention_mask[:, source_start:placeholder],
                                torch.ones(
                                    (1, feature.shape[0]),
                                    dtype=attention_mask.dtype,
                                    device=attention_mask.device,
                                ),
                            )
                        )
                    expanded_start += text_part.shape[1]
                    spans.append((expanded_start, feature.shape[0], grid))
                    expanded_start += feature.shape[0]
                    source_start = placeholder + 1
                embed_parts.append(embeddings[:, source_start:])
                embeddings = torch.cat(embed_parts, dim=1)
                if attention_mask is not None:
                    mask_parts.append(attention_mask[:, source_start:])
                    expanded_mask = torch.cat(mask_parts, dim=1)
                position_ids = _qwen_image_mrope_position_ids_for_spans(
                    embeddings.shape[1], spans, embeddings.device
                )
            hidden = self.model.forward_embeds(embeddings, expanded_mask, position_ids)
            selected = hidden[:, selection.slice_start :]
            selected_mask = None
            if expanded_mask is not None:
                candidate = expanded_mask[:, selection.slice_start :]
                if not bool(torch.all(candidate)):
                    selected_mask = candidate
            return selected, selected_mask
        finally:
            self.active_image_features = None


__all__ = [
    "QwenImageLanguageAttention",
    "QwenImageLanguageModel",
    "QwenImageTextModel",
    "QwenImageVisionAttention",
    "QwenImageVisionTransformer",
    "prepare_qwen_image_vision",
    "qwen_image_mrope_position_ids",
    "resize_qwen_image_content",
]
