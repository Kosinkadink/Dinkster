"""Native torch DINOv3 ViT-H/16+ image encoder.

Transcribed from ``comfy/image_encoders/dino3.py`` @ 36408117: a
32-layer SwiGLU vision transformer whose full token sequence (class
token, four register tokens, then patch tokens) conditions the
TripoSplat flow model. Rotary embeddings cover ONLY the patch tokens:
per-axis angles come from patch-center coordinates in [-1, 1] expanded
over ``inv_freq = 1 / theta ** arange(0, 1, 4 / head_dim)``, tiled to
the head dimension, and the class/register prefix passes through
unrotated. Each sublayer output is scaled by a learned per-channel
lambda (LayerScale) before its residual add.

State-dict keys match the published checkpoint exactly
(:func:`dinkster_inference.dinov3.dinov3_vith_layout`); the ``mask_token``
parameter exists for checkpoint parity but inference never masks. The
rope ``inv_freq`` buffer is non-persistent and derived from the
config.

:func:`dinov3_preprocess` is the reference ``clip_preprocess`` with
the DINOv3 ImageNet statistics: NHWC [0, 1] images are bicubic-resized
(center-cropped when ``crop`` is set), quantized to 8-bit levels, and
normalized to float32 NCHW.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from dinkster_inference.dinov3 import DINOV3_VITH, DINOv3ViTConfig

from .attention import AttentionKernel, select_attention
from .operations import INITLESS, Operations, ResidencyRouted
from .ops import cast_weight

_DEFAULT_ATTENTION = select_attention("clip").kernel


def dinov3_preprocess(
    image: torch.Tensor, config: DINOv3ViTConfig = DINOV3_VITH, *, crop: bool = True
) -> torch.Tensor:
    """Reference resize, optional center crop, 8-bit quantization, and
    ImageNet normalization (clip_preprocess @ 36408117 with the DINOv3
    config's statistics)."""
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

    size = config.image_size
    image = image.float()[..., :3].movedim(-1, 1)
    if image.shape[2:] != (size, size):
        if crop:
            scale = size / min(image.shape[2], image.shape[3])
            scaled = (round(scale * image.shape[2]), round(scale * image.shape[3]))
        else:
            scaled = (size, size)
        image = F.interpolate(image, size=scaled, mode="bicubic", antialias=True)
        top = (image.shape[2] - size) // 2
        left = (image.shape[3] - size) // 2
        image = image[:, :, top : top + size, left : left + size]
    image = (255.0 * image).clamp(0, 255).round() / 255.0
    mean = image.new_tensor(config.image_mean).view(1, 3, 1, 1)
    std = image.new_tensor(config.image_std).view(1, 3, 1, 1)
    return (image - mean) / std


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_patch_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate only the trailing patch tokens; the class and register
    prefix passes through (apply_rotary_pos_emb @ 36408117)."""
    prefix = q.shape[-2] - sin.shape[-2]
    q_prefix, q_patches = q.split((prefix, sin.shape[-2]), dim=-2)
    k_prefix, k_patches = k.split((prefix, sin.shape[-2]), dim=-2)
    q_patches = (q_patches * cos) + (_rotate_half(q_patches) * sin)
    k_patches = (k_patches * cos) + (_rotate_half(k_patches) * sin)
    return torch.cat((q_prefix, q_patches), dim=-2), torch.cat((k_prefix, k_patches), dim=-2)


class DINOv3Rope(torch.nn.Module):
    """Patch-center rotary tables (DINOv3ViTRopePositionEmbedding
    @ 36408117). ``inv_freq`` is deterministic, computed on the CPU so
    meta-device construction cannot corrupt it, and non-persistent."""

    def __init__(self, config: DINOv3ViTConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        head_dim = config.hidden_size // config.num_attention_heads
        with torch.device("cpu"):
            inv_freq = 1.0 / config.rope_theta ** torch.arange(
                0, 1, 4 / head_dim, dtype=torch.float32
            )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = pixel_values.shape[-2:]
        patches_h = height // self.patch_size
        patches_w = width // self.patch_size
        device = pixel_values.device
        coords_h = torch.arange(0.5, patches_h, dtype=torch.float32, device=device) / patches_h
        coords_w = torch.arange(0.5, patches_w, dtype=torch.float32, device=device) / patches_w
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = 2.0 * coords.flatten(0, 1) - 1.0
        inv_freq = self.get_buffer("inv_freq").to(device)
        angles = 2 * math.pi * coords[:, :, None] * inv_freq[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return angles.cos().to(pixel_values.dtype), angles.sin().to(pixel_values.dtype)


class DINOv3Embeddings(ResidencyRouted, torch.nn.Module):
    """Patch conv plus the class and register token prefix
    (DINOv3ViTEmbeddings @ 36408117). ``mask_token`` is loaded for
    checkpoint parity but never applied at inference."""

    def __init__(self, config: DINOv3ViTConfig, *, operations: Operations) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.cls_token = torch.nn.Parameter(torch.empty(1, 1, hidden))
        self.mask_token = torch.nn.Parameter(torch.empty(1, 1, hidden))
        self.register_tokens = torch.nn.Parameter(
            torch.empty(1, config.num_register_tokens, hidden)
        )
        self.patch_embeddings = operations.conv2d(
            config.num_channels, hidden, config.patch_size, stride=config.patch_size
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch = pixel_values.shape[0]
        patches = self.patch_embeddings(pixel_values).flatten(2).transpose(1, 2)
        binding = self._offloaded_residency()
        if binding is None:
            cls_token = cast_weight(self.cls_token, dtype=patches.dtype, device=patches.device)
            register_tokens = cast_weight(
                self.register_tokens, dtype=patches.dtype, device=patches.device
            )
        else:
            with binding.lease() as lease:
                cls_token = lease.get("cls_token", dtype=patches.dtype)
                register_tokens = lease.get("register_tokens", dtype=patches.dtype)
        return torch.cat(
            [
                cls_token.expand(batch, -1, -1),
                register_tokens.expand(batch, -1, -1),
                patches,
            ],
            dim=1,
        )


class DINOv3LayerScale(ResidencyRouted, torch.nn.Module):
    """Learned per-channel residual scale (LayerScale @ 36408117)."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.lambda1 = torch.nn.Parameter(torch.empty(hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            return x * cast_weight(self.lambda1, dtype=x.dtype, device=x.device)
        with binding.lease() as lease:
            return x * lease.get("lambda1", dtype=x.dtype)


class DINOv3Attention(torch.nn.Module):
    """Separate q/k/v/o projections with a bias-free key projection
    (DINOv3ViTAttention @ 36408117); patch-only rope."""

    _attention_kernel: AttentionKernel

    def __init__(
        self,
        config: DINOv3ViTConfig,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.heads = config.num_attention_heads
        self.k_proj = operations.linear(hidden, hidden, bias=False)
        self.v_proj = operations.linear(hidden, hidden)
        self.q_proj = operations.linear(hidden, hidden)
        self.o_proj = operations.linear(hidden, hidden)
        object.__setattr__(self, "_attention_kernel", attention_kernel)

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        batch, sequence, hidden = x.shape

        def split(value: torch.Tensor) -> torch.Tensor:
            return value.view(batch, sequence, self.heads, -1).transpose(1, 2)

        q, k = _apply_patch_rope(split(self.q_proj(x)), split(self.k_proj(x)), *rope)
        attended = self._attention_kernel(q, k, split(self.v_proj(x)))
        return self.o_proj(attended.transpose(1, 2).reshape(batch, sequence, hidden))


class DINOv3GatedMlp(torch.nn.Module):
    """SwiGLU feed-forward (DINOv3ViTGatedMLP @ 36408117)."""

    def __init__(self, config: DINOv3ViTConfig, *, operations: Operations) -> None:
        super().__init__()
        hidden = config.hidden_size
        intermediate = config.intermediate_size
        self.gate_proj = operations.linear(hidden, intermediate)
        self.up_proj = operations.linear(hidden, intermediate)
        self.down_proj = operations.linear(intermediate, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DINOv3Mlp(torch.nn.Module):
    """Standard GELU feed-forward used by DINOv3 ViT-L/16."""

    def __init__(self, config: DINOv3ViTConfig, *, operations: Operations) -> None:
        super().__init__()
        self.up_proj = operations.linear(config.hidden_size, config.intermediate_size)
        self.down_proj = operations.linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x)))


class DINOv3Layer(torch.nn.Module):
    def __init__(
        self,
        config: DINOv3ViTConfig,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        eps = config.layer_norm_eps
        self.norm1 = operations.layer_norm(hidden, eps=eps)
        self.attention = DINOv3Attention(
            config, operations=operations, attention_kernel=attention_kernel
        )
        self.layer_scale1 = DINOv3LayerScale(hidden)
        self.norm2 = operations.layer_norm(hidden, eps=eps)
        self.mlp = (
            DINOv3GatedMlp(config, operations=operations)
            if config.mlp_kind == "swiglu"
            else DINOv3Mlp(config, operations=operations)
        )
        self.layer_scale2 = DINOv3LayerScale(hidden)

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = x + self.layer_scale1(self.attention(self.norm1(x), rope))
        return x + self.layer_scale2(self.mlp(self.norm2(x)))


class DINOv3ViTModel(torch.nn.Module):
    """The DINOv3 encoder over preprocessed pixels (DINOv3ViTModel
    @ 36408117). Returns the final-norm token sequence
    [batch, 1 + registers + patches, hidden]; the class token at
    row zero is the pooled summary."""

    _dinkster_residency_constant_buffers = frozenset({"rope_embeddings.inv_freq"})

    def __init__(
        self,
        config: DINOv3ViTConfig = DINOV3_VITH,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("hidden size must divide evenly over attention heads")
        if (config.hidden_size // config.num_attention_heads) % 4:
            raise ValueError("head dimension must be divisible by four for patch rope")
        self.config = config
        self.embeddings = DINOv3Embeddings(config, operations=operations)
        self.rope_embeddings = DINOv3Rope(config)
        self.layer = torch.nn.ModuleList(
            DINOv3Layer(config, operations=operations, attention_kernel=attention_kernel)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = operations.layer_norm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, pixel_values: torch.Tensor, *, parameter_free_norm: bool = False
    ) -> torch.Tensor:
        config = self.config
        if pixel_values.ndim != 4 or pixel_values.shape[1] != config.num_channels:
            raise ValueError(
                f"pixel values must be [batch, {config.num_channels}, height, width],"
                f" got {tuple(pixel_values.shape)}"
            )
        if not pixel_values.is_floating_point():
            raise ValueError("pixel values must use a floating-point dtype")
        height, width = pixel_values.shape[-2:]
        if height % config.patch_size or width % config.patch_size or height == 0 or width == 0:
            raise ValueError(
                f"pixel height and width must be positive multiples of {config.patch_size},"
                f" got {height}x{width}"
            )
        hidden_states = self.embeddings(pixel_values)
        rope = self.rope_embeddings(pixel_values)
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, rope)
        if parameter_free_norm:
            return F.layer_norm(hidden_states, hidden_states.shape[-1:])
        return self.norm(hidden_states)


__all__ = [
    "DINOv3Attention",
    "DINOv3Embeddings",
    "DINOv3GatedMlp",
    "DINOv3Layer",
    "DINOv3LayerScale",
    "DINOv3Mlp",
    "DINOv3Rope",
    "DINOv3ViTModel",
    "dinov3_preprocess",
]
