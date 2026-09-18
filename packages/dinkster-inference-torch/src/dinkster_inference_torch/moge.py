"""Native MoGe v1/v2 models and state-dict-driven construction.

V1: DINOv2 backbone + multi-output head (points, mask).
V2: DINOv2 encoder + neck + per-output heads (points, mask, normal, optional metric-scale MLP).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov2 import Dinov2Model
from .moge_modules import MLP, ConvStack, DINOv2Encoder, HeadV1, view_plane_uv_grid
from .operations import INITLESS, Operations, ResidencyRouted


def _remap_points(points: torch.Tensor) -> torch.Tensor:
    """Apply the exp remap: z -> exp(z), xy stays linear and gets scaled by the new z."""
    xy, z = points.split([2, 1], dim=-1)
    z = torch.exp(z)
    return torch.cat([xy * z, z], dim=-1)


def _detect_dinov2(sd: dict[str, torch.Tensor], prefix: str) -> dict[str, Any]:
    # All shipped MoGe checkpoints use plain DINOv2
    hidden = sd[prefix + "embeddings.cls_token"].shape[-1]
    layer_prefix = prefix + "encoder.layer."
    depth = 1 + max(
        int(k[len(layer_prefix) :].split(".")[0]) for k in sd if k.startswith(layer_prefix)
    )
    return {
        "hidden_size": hidden,
        "num_attention_heads": hidden // 64,
        "num_hidden_layers": depth,
        "layer_norm_eps": 1e-6,
        "use_swiglu_ffn": False,
        "position_tokens": sd[prefix + "embeddings.position_embeddings"].shape[1],
        "use_mask_token": prefix + "embeddings.mask_token" in sd,
    }


class MoGeModelV1(ResidencyRouted, nn.Module):
    """MoGe v1: DINOv2 backbone + HeadV1 (points, mask)."""

    image_mean: torch.Tensor
    image_std: torch.Tensor

    intermediate_layers = 4
    num_tokens_range = (1200, 2500)
    mask_threshold = 0.5

    def __init__(
        self,
        backbone: dict[str, Any],
        dim_upsample: Sequence[int] = (256, 128, 128),
        num_res_blocks: int = 1,
        dim_times_res_block_hidden: int = 1,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.backbone = Dinov2Model(backbone, operations=operations)
        self.head = HeadV1(
            dim_in=backbone["hidden_size"],
            dim_upsample=list(dim_upsample),
            num_res_blocks=num_res_blocks,
            dim_times_res_block_hidden=dim_times_res_block_hidden,
            operations=operations,
        )
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor, num_tokens: int) -> dict[str, torch.Tensor]:
        H, W = image.shape[-2:]
        resize = ((num_tokens * 14**2) / (H * W)) ** 0.5
        rh, rw = int(H * resize), int(W * resize)
        x = F.interpolate(image, (rh, rw), mode="bicubic", align_corners=False, antialias=True)
        binding = self._offloaded_residency()
        if binding is None:
            mean = self.image_mean.to(x)
            std = self.image_std.to(x)
        else:
            with binding.lease() as lease:
                mean = lease.get("image_mean", dtype=x.dtype)
                std = lease.get("image_std", dtype=x.dtype)
        x = (x - mean) / std
        x14 = F.interpolate(
            x, (rh // 14 * 14, rw // 14 * 14), mode="bilinear", align_corners=False, antialias=True
        )

        n_layers = len(self.backbone.encoder.layer)
        indices = list(range(n_layers - self.intermediate_layers, n_layers))
        feats = self.backbone.get_intermediate_layers(x14, indices, apply_norm=True)

        points, mask = self.head(feats, x)
        points = F.interpolate(points.float(), (H, W), mode="bilinear", align_corners=False)
        points = _remap_points(points.permute(0, 2, 3, 1))

        mask = F.interpolate(mask.float(), (H, W), mode="bilinear", align_corners=False).squeeze(1)

        return {"points": points, "mask": mask}

    @classmethod
    def from_state_dict(
        cls,
        sd: dict[str, torch.Tensor],
        *,
        operations: Operations = INITLESS,
    ) -> MoGeModelV1:
        """Detect the v1 head config from sd, build a model, and load weights."""
        n_up = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("head.upsample_blocks."))
        dim_upsample = [sd[f"head.upsample_blocks.{i}.0.0.weight"].shape[1] for i in range(n_up)]
        # Each upsample stage is Sequential[upsampler, *res_blocks]; count res blocks at level 0.
        num_res_blocks = max(
            {int(k.split(".")[3]) for k in sd if k.startswith("head.upsample_blocks.0.")}
        )
        hidden_out = sd["head.upsample_blocks.0.1.layers.2.weight"].shape[0]
        dim_times = max(hidden_out // dim_upsample[0], 1)
        model = cls(
            backbone=_detect_dinov2(sd, prefix="backbone."),
            dim_upsample=dim_upsample,
            num_res_blocks=num_res_blocks,
            dim_times_res_block_hidden=dim_times,
            operations=operations,
        )
        model.load_state_dict(sd, strict=True, assign=next(model.parameters()).is_meta)
        return model


class MoGeModelV2(nn.Module):
    """MoGe v2: DINOv2 encoder + neck + per-output heads (points/mask/normal/metric-scale)."""

    intermediate_layers = 4
    num_tokens_range = (1200, 3600)

    def __init__(
        self,
        encoder: dict[str, Any],
        neck: dict[str, Any],
        points_head: dict[str, Any],
        mask_head: dict[str, Any],
        scale_head: dict[str, Any],
        normal_head: dict[str, Any] | None = None,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.encoder = DINOv2Encoder(**encoder, operations=operations)
        self.neck = ConvStack(**neck, operations=operations)
        self.points_head = ConvStack(**points_head, operations=operations)
        self.mask_head = ConvStack(**mask_head, operations=operations)
        self.scale_head = MLP(**scale_head, operations=operations)
        if normal_head is not None:
            self.normal_head = ConvStack(**normal_head, operations=operations)

    def forward(self, image: torch.Tensor, num_tokens: int) -> dict[str, torch.Tensor]:
        B, _, H, W = image.shape
        device, dtype = image.device, image.dtype
        aspect_ratio = W / H
        base_h = round((num_tokens / aspect_ratio) ** 0.5)
        base_w = round((num_tokens * aspect_ratio) ** 0.5)

        feat_top, cls_token = self.encoder(image, base_h, base_w, return_class_token=True)

        # 5-level pyramid: feat at level 0 concatenated with UV, other levels UV-only.
        levels = [
            view_plane_uv_grid(B, base_h * (2**L), base_w * (2**L), aspect_ratio, dtype, device)
            for L in range(5)
        ]
        levels[0] = torch.cat([feat_top, levels[0]], dim=1)

        feats = self.neck(levels)

        def _resize(v: torch.Tensor) -> torch.Tensor:
            return F.interpolate(v, (H, W), mode="bilinear", align_corners=False)

        points = _remap_points(_resize(self.points_head(feats)[-1]).permute(0, 2, 3, 1))
        mask = _resize(self.mask_head(feats)[-1]).squeeze(1).sigmoid()
        metric_scale = self.scale_head(cls_token).squeeze(1).exp()

        result = {"points": points, "mask": mask, "metric_scale": metric_scale}
        if hasattr(self, "normal_head"):
            normal = _resize(self.normal_head(feats)[-1])
            result["normal"] = F.normalize(normal.permute(0, 2, 3, 1), dim=-1)
        return result

    @classmethod
    def from_state_dict(
        cls,
        sd: dict[str, torch.Tensor],
        *,
        operations: Operations = INITLESS,
    ) -> MoGeModelV2:
        """Detect the v2 encoder/neck/heads config from sd, build a model, and load weights."""
        backbone = _detect_dinov2(sd, prefix="encoder.backbone.")
        depth = backbone["num_hidden_layers"]
        n = cls.intermediate_layers
        encoder = {
            "backbone": backbone,
            "intermediate_layers": [(depth // n) * (i + 1) - 1 for i in range(n)],
            "dim_out": sd["encoder.output_projections.0.weight"].shape[0],
        }
        # Linear weights identify each scale-head stage as (out, in).
        scale_idxs = sorted({int(k.split(".")[1]) for k in sd if k.startswith("scale_head.")})
        scale_first = sd[f"scale_head.{scale_idxs[0]}.weight"]
        cfg: dict[str, Any] = {
            "encoder": encoder,
            "neck": cls._detect_convstack(sd, "neck."),
            "points_head": cls._detect_convstack(sd, "points_head."),
            "mask_head": cls._detect_convstack(sd, "mask_head."),
            "scale_head": {
                "dims": [scale_first.shape[1]]
                + [sd[f"scale_head.{i}.weight"].shape[0] for i in scale_idxs]
            },
        }
        if any(k.startswith("normal_head.") for k in sd):
            cfg["normal_head"] = cls._detect_convstack(sd, "normal_head.")
        model = cls(**cfg, operations=operations)
        model.load_state_dict(sd, strict=True, assign=next(model.parameters()).is_meta)
        return model

    @staticmethod
    def _detect_convstack(sd: dict[str, torch.Tensor], prefix: str) -> dict[str, Any]:
        """Reconstruct a ConvStack config from the keys under prefix"""
        in_keys = [
            k for k in sd if k.startswith(f"{prefix}input_blocks.") and k.endswith(".weight")
        ]
        n = 1 + max(int(k[len(f"{prefix}input_blocks.") :].split(".")[0]) for k in in_keys)

        in_shapes = [sd[f"{prefix}input_blocks.{i}.weight"].shape for i in range(n)]

        def has_out(index: int) -> bool:
            return f"{prefix}output_blocks.{index}.weight" in sd

        has_norm = f"{prefix}res_blocks.0.0.layers.0.weight" in sd

        def num_res_at(index: int) -> int:
            rb_prefix = f"{prefix}res_blocks.{index}."
            return len(
                {int(k[len(rb_prefix) :].split(".")[0]) for k in sd if k.startswith(rb_prefix)}
            )

        return {
            "dim_in": [s[1] for s in in_shapes],
            "dim_res_blocks": [s[0] for s in in_shapes],
            "dim_out": [
                sd[f"{prefix}output_blocks.{i}.weight"].shape[0] if has_out(i) else None
                for i in range(n)
            ],
            "num_res_blocks": [num_res_at(i) for i in range(n)],
            "resamplers": [
                "conv_transpose" if f"{prefix}resamplers.{i}.0.weight" in sd else "bilinear"
                for i in range(n - 1)
            ],
            "res_block_in_norm": "layer_norm" if has_norm else "none",
            "res_block_hidden_norm": "group_norm" if has_norm else "none",
        }


# Translate the Meta-style DINOv2 keys MoGe ships to the naming ComfyUI DINOv2 port expects,
# and split each fused qkv tensor into Q/K/V.
_DINOV2_TOPLEVEL_RENAMES = {
    "patch_embed.proj.weight": "embeddings.patch_embeddings.projection.weight",
    "patch_embed.proj.bias": "embeddings.patch_embeddings.projection.bias",
    "cls_token": "embeddings.cls_token",
    "pos_embed": "embeddings.position_embeddings",
    "register_tokens": "embeddings.register_tokens",
    "mask_token": "embeddings.mask_token",
    "norm.weight": "layernorm.weight",
    "norm.bias": "layernorm.bias",
}
_DINOV2_BLOCK_RENAMES = [
    ("ls1.gamma", "layer_scale1.lambda1"),
    ("ls2.gamma", "layer_scale2.lambda1"),
    ("attn.proj.", "attention.output.dense."),
    ("mlp.w12.", "mlp.weights_in."),
    ("mlp.w3.", "mlp.weights_out."),
]


def _remap_state_dict(sd: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "model" in sd and "model_config" in sd:
        sd = sd["model"]
    prefix = (
        "encoder.backbone." if any(k.startswith("encoder.backbone.") for k in sd) else "backbone."
    )
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if not k.startswith(prefix):
            out[k] = v
            continue
        rel = k[len(prefix) :]
        if rel in _DINOV2_TOPLEVEL_RENAMES:
            out[prefix + _DINOV2_TOPLEVEL_RENAMES[rel]] = v
            continue
        if not rel.startswith("blocks."):
            out[k] = v
            continue
        _, idx, sub = rel.split(".", 2)
        if sub in ("attn.qkv.weight", "attn.qkv.bias"):
            tail = sub.rsplit(".", 1)[1]
            q, kw, vw = v.chunk(3, dim=0)
            base = f"{prefix}encoder.layer.{idx}.attention.attention"
            out[f"{base}.query.{tail}"] = q
            out[f"{base}.key.{tail}"] = kw
            out[f"{base}.value.{tail}"] = vw
            continue
        for old, new in _DINOV2_BLOCK_RENAMES:
            sub = sub.replace(old, new)
        out[f"{prefix}encoder.layer.{idx}.{sub}"] = v
    return out


def build_from_state_dict(
    sd: dict[str, Any], *, operations: Operations = INITLESS
) -> MoGeModelV1 | MoGeModelV2:
    """Dispatch to v1 or v2 based on the DINOv2 backbone prefix."""
    sd = _remap_state_dict(sd)
    cls = MoGeModelV2 if any(k.startswith("encoder.backbone.") for k in sd) else MoGeModelV1
    return cls.from_state_dict(sd, operations=operations)
