"""Native Wan 2.1 CLIP ViT-H/14 image encoder.

This is a direct port of ``CLIPVisionModelProjection``, ``CLIPVision``, and
``clip_preprocess`` from ComfyUI commit b78cec87. State names remain identical
to the official Wan 2.1 ``clip_vision_h.safetensors`` checkpoint. The public
forward accepts Comfy/Dinkster NHWC images in [0, 1] and returns the unnormalized
penultimate transformer state with shape [B, 257, 1280].
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dinkster_inference.clip_vision import WAN21_CLIP_VISION, ClipVisionConfig

from .attention import AttentionKernel, select_attention
from .image_preprocess import clip_preprocess
from .operations import INITLESS, Operations, ResidencyRouted
from .ops import cast_weight

_DEFAULT_ATTENTION = select_attention("clip").kernel


def clip_vision_preprocess(
    image: torch.Tensor, *, size: int = 224, crop: bool = True
) -> torch.Tensor:
    """Reference resize, optional center crop, and normalization."""
    if type(image) is not torch.Tensor or image.layout != torch.strided:
        raise TypeError("image must be an exact strided torch.Tensor")
    if image.ndim != 4 or image.shape[0] == 0 or image.shape[1] == 0 or image.shape[2] == 0:
        raise ValueError(f"image must be a non-empty NHWC tensor, got {tuple(image.shape)}")
    if image.shape[3] < 3:
        raise ValueError(f"image must have at least three channels, got {image.shape[3]}")
    if not image.is_floating_point():
        raise ValueError("image must use a floating-point dtype")
    if not torch.isfinite(image).all() or torch.any(image < 0) or torch.any(image > 1):
        raise ValueError("image values must be finite and in [0, 1]")

    return clip_preprocess(image.float(), size=size, crop=crop)


class ClipVisionAttention(torch.nn.Module):
    _attention_kernel: AttentionKernel

    def __init__(
        self,
        config: ClipVisionConfig,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.heads = config.num_attention_heads
        self.q_proj = operations.linear(hidden, hidden)
        self.k_proj = operations.linear(hidden, hidden)
        self.v_proj = operations.linear(hidden, hidden)
        self.out_proj = operations.linear(hidden, hidden)
        object.__setattr__(self, "_attention_kernel", attention_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, hidden = x.shape

        def split(value: torch.Tensor) -> torch.Tensor:
            return value.view(batch, sequence, self.heads, -1).transpose(1, 2)

        attended = self._attention_kernel(
            split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        )
        return self.out_proj(attended.transpose(1, 2).reshape(batch, sequence, hidden))


class ClipVisionMlp(torch.nn.Module):
    def __init__(self, config: ClipVisionConfig, *, operations: Operations) -> None:
        super().__init__()
        self.fc1 = operations.linear(config.hidden_size, config.intermediate_size)
        self.fc2 = operations.linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class ClipVisionLayer(torch.nn.Module):
    def __init__(
        self,
        config: ClipVisionConfig,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.layer_norm1 = operations.layer_norm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = ClipVisionAttention(
            config, operations=operations, attention_kernel=attention_kernel
        )
        self.layer_norm2 = operations.layer_norm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = ClipVisionMlp(config, operations=operations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class ClipVisionEncoder(torch.nn.Module):
    def __init__(
        self,
        config: ClipVisionConfig,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            ClipVisionLayer(config, operations=operations, attention_kernel=attention_kernel)
            for _ in range(config.num_hidden_layers)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        penultimate: torch.Tensor | None = None
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index == len(self.layers) - 2:
                penultimate = x.clone()
        if penultimate is None:
            raise ValueError("CLIP vision requires at least two transformer layers")
        return x, penultimate


class ClipVisionEmbeddings(ResidencyRouted, torch.nn.Module):
    def __init__(self, config: ClipVisionConfig, *, operations: Operations) -> None:
        super().__init__()
        self.class_embedding = torch.nn.Parameter(torch.empty(config.hidden_size))
        self.patch_embedding = operations.conv2d(
            config.num_channels,
            config.hidden_size,
            config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.position_embedding = operations.embedding(config.tokens, config.hidden_size)
        self.register_buffer(
            "position_ids", torch.arange(config.tokens, dtype=torch.int64).unsqueeze(0)
        )

    def _forward_owned(
        self,
        pixel_values: torch.Tensor,
        class_embedding: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        patches = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        classes = class_embedding.expand(pixel_values.shape[0], 1, -1)
        tokens = torch.cat((classes, patches), dim=1)
        return tokens + self.position_embedding(position_ids[0])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            class_embedding = cast_weight(
                self.class_embedding, device=pixel_values.device, dtype=pixel_values.dtype
            )
            position_ids = self.get_buffer("position_ids")
            return self._forward_owned(pixel_values, class_embedding, position_ids)
        with binding.lease() as lease:
            return self._forward_owned(
                pixel_values,
                lease.get("class_embedding", dtype=pixel_values.dtype),
                lease.get("position_ids", dtype=torch.int64),
            )


class ClipVisionTransformer(torch.nn.Module):
    def __init__(
        self,
        config: ClipVisionConfig,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.embeddings = ClipVisionEmbeddings(config, operations=operations)
        self.pre_layrnorm = operations.layer_norm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = ClipVisionEncoder(
            config, operations=operations, attention_kernel=attention_kernel
        )
        self.post_layernorm = operations.layer_norm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        last, penultimate = self.encoder(self.pre_layrnorm(self.embeddings(pixel_values)))
        pooled = self.post_layernorm(last[:, 0])
        return last, penultimate, pooled


class Wan21ClipVisionEncoder(torch.nn.Module):
    """Exact ViT-H/14 owner and NHWC-to-penultimate family seam."""

    def __init__(
        self,
        config: ClipVisionConfig = WAN21_CLIP_VISION,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        super().__init__()
        self.config = config
        self.vision_model = ClipVisionTransformer(
            config, operations=operations, attention_kernel=attention_kernel
        )
        self.visual_projection = operations.linear(
            config.hidden_size, config.projection_dim, bias=False
        )

    def forward(self, image: torch.Tensor, *, crop: bool = True) -> torch.Tensor:
        pixels = clip_vision_preprocess(image, size=self.config.image_size, crop=crop)
        _, penultimate, _ = self.vision_model(pixels)
        expected = (image.shape[0], self.config.tokens, self.config.hidden_size)
        if penultimate.shape != expected:
            raise RuntimeError(
                f"CLIP vision produced {tuple(penultimate.shape)}, expected {expected}"
            )
        return penultimate


class SD15IPAdapterClipVisionEncoder(Wan21ClipVisionEncoder):
    """CLIP ViT-H/14 image-projection output consumed by standard IP-Adapter."""

    def forward(self, image: torch.Tensor, *, crop: bool = True) -> torch.Tensor:
        pixels = clip_vision_preprocess(image, size=self.config.image_size, crop=crop)
        _, _, pooled = self.vision_model(pixels)
        projected = self.visual_projection(pooled)
        expected = (image.shape[0], self.config.projection_dim)
        if projected.shape != expected:
            raise RuntimeError(
                f"CLIP vision produced {tuple(projected.shape)}, expected {expected}"
            )
        return projected
