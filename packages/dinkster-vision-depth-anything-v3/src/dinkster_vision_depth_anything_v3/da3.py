"""Depth Anything 3 Mono Large architecture.

The DPT structure and execution order follow ComfyUI commit
e7051b03758a1247e3adb84a5b784ffacb9a23bd.
"""

from __future__ import annotations

import math
from types import MethodType
from typing import Any, cast

import torch
from torch import nn
from transformers.models.dinov2.configuration_dinov2 import Dinov2Config
from transformers.models.dinov2.modeling_dinov2 import Dinov2Model

PATCH_SIZE = 14
OUT_LAYERS = (4, 11, 17, 23)


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, device=device)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, device=device)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.conv1(self.activation(value))
        output = self.conv2(self.activation(output))
        return output + value


class FeatureFusionBlock(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        has_residual: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.resConfUnit1 = ResidualConvUnit(features, device=device) if has_residual else None
        self.resConfUnit2 = ResidualConvUnit(features, device=device)
        self.out_conv = nn.Conv2d(features, features, 1, 1, 0, device=device)

    def forward(
        self,
        value: torch.Tensor,
        residual: torch.Tensor | None = None,
        *,
        size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if residual is not None and self.resConfUnit1 is not None:
            value = value + self.resConfUnit1(residual)
        value = self.resConfUnit2(value)
        value = torch.nn.functional.interpolate(
            value,
            size=size,
            scale_factor=None if size is not None else 2.0,
            mode="bilinear",
            align_corners=True,
        )
        return self.out_conv(value)


class Scratch(nn.Module):
    def __init__(
        self,
        channels: tuple[int, int, int, int],
        features: int,
        *,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.layer1_rn = nn.Conv2d(channels[0], features, 3, 1, 1, bias=False, device=device)
        self.layer2_rn = nn.Conv2d(channels[1], features, 3, 1, 1, bias=False, device=device)
        self.layer3_rn = nn.Conv2d(channels[2], features, 3, 1, 1, bias=False, device=device)
        self.layer4_rn = nn.Conv2d(channels[3], features, 3, 1, 1, bias=False, device=device)
        self.refinenet1 = FeatureFusionBlock(features, device=device)
        self.refinenet2 = FeatureFusionBlock(features, device=device)
        self.refinenet3 = FeatureFusionBlock(features, device=device)
        self.refinenet4 = FeatureFusionBlock(features, has_residual=False, device=device)
        self.output_conv1 = nn.Conv2d(features, features // 2, 3, 1, 1, device=device)
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, 3, 1, 1, device=device),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 1, 1, 1, 0, device=device),
        )
        self.sky_output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, 3, 1, 1, device=device),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 1, 1, 1, 0, device=device),
        )


class DPT(nn.Module):
    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__()
        channels = (256, 512, 1024, 1024)
        self.projects = nn.ModuleList(
            nn.Conv2d(1024, channel, 1, 1, 0, device=device) for channel in channels
        )
        self.resize_layers = nn.ModuleList(
            (
                nn.ConvTranspose2d(256, 256, 4, 4, 0, device=device),
                nn.ConvTranspose2d(512, 512, 2, 2, 0, device=device),
                nn.Identity(),
                nn.Conv2d(1024, 1024, 3, 2, 1, device=device),
            )
        )
        self.scratch = Scratch(channels, 256, device=device)

    def forward(
        self,
        features: list[torch.Tensor],
        *,
        height: int,
        width: int,
    ) -> dict[str, torch.Tensor]:
        batch, views, patch_count, channels = features[0].shape
        patch_height, patch_width = height // PATCH_SIZE, width // PATCH_SIZE
        resized: list[torch.Tensor] = []
        for project, resize, feature in zip(
            self.projects,
            self.resize_layers,
            features,
            strict=True,
        ):
            value = feature.reshape(batch * views, patch_count, channels)
            value = value.permute(0, 2, 1).contiguous()
            value = value.reshape(batch * views, channels, patch_height, patch_width)
            resized.append(resize(project(value)))

        layer1 = self.scratch.layer1_rn(resized[0])
        layer2 = self.scratch.layer2_rn(resized[1])
        layer3 = self.scratch.layer3_rn(resized[2])
        layer4 = self.scratch.layer4_rn(resized[3])
        output = self.scratch.refinenet4(layer4, size=layer3.shape[2:])
        output = self.scratch.refinenet3(output, layer3, size=layer2.shape[2:])
        output = self.scratch.refinenet2(output, layer2, size=layer1.shape[2:])
        output = self.scratch.refinenet1(output, layer1)
        output = self.scratch.output_conv1(output)
        output = torch.nn.functional.interpolate(
            output,
            size=(patch_height * PATCH_SIZE, patch_width * PATCH_SIZE),
            mode="bilinear",
            align_corners=True,
        )
        depth = torch.exp(self.scratch.output_conv2(output))
        sky = torch.relu(self.scratch.sky_output_conv2(output))
        return {
            "depth": depth.squeeze(1).view(batch * views, height, width),
            "sky": sky.squeeze(1).view(batch * views, height, width),
        }


def backbone_config() -> Dinov2Config:
    config = Dinov2Config(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        mlp_ratio=4,
        image_size=518,
        patch_size=PATCH_SIZE,
        use_mask_token=False,
        reshape_hidden_states=False,
    )
    config._attn_implementation = "sdpa"
    return config


def source_position_encoding(
    embeddings_module: Any,
    embeddings: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    positions = embeddings_module.position_embeddings
    patch_count = embeddings.shape[1] - 1
    position_count = positions.shape[1] - 1
    if patch_count == position_count and height == width:
        return cast("torch.Tensor", positions)
    class_position = positions[:, :1]
    patch_positions = positions[:, 1:].float()
    source_width = math.sqrt(position_count)
    target_height = height // embeddings_module.patch_size
    target_width = width // embeddings_module.patch_size
    patch_positions = torch.nn.functional.interpolate(
        patch_positions.reshape(1, int(source_width), int(source_width), -1).permute(0, 3, 1, 2),
        scale_factor=(
            (target_height + 0.1) / source_width,
            (target_width + 0.1) / source_width,
        ),
        mode="bicubic",
        antialias=False,
    )
    if patch_positions.shape[-2:] != (target_height, target_width):
        raise ValueError("Depth Anything 3 position interpolation produced the wrong shape")
    flattened = patch_positions.permute(0, 2, 3, 1).view(1, -1, embeddings.shape[-1])
    return torch.cat((class_position, flattened), dim=1).to(embeddings.dtype)


class DepthAnything3MonoLarge(nn.Module):
    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.backbone: Dinov2Model = cast("Any", Dinov2Model)(backbone_config()).to(device=device)
        self.backbone.embeddings.interpolate_pos_encoding = MethodType(
            source_position_encoding,
            self.backbone.embeddings,
        )
        self.head = DPT(device=device)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        height, width = image.shape[-2:]
        hidden_state = self.backbone.embeddings(image)
        features: list[torch.Tensor] = []
        for index, layer in enumerate(self.backbone.encoder.layer):
            hidden_state = layer(hidden_state)
            if index in OUT_LAYERS:
                normalized = self.backbone.layernorm(hidden_state)
                features.append(normalized[:, None, 1:])
        return self.head(features, height=height, width=width)


__all__ = ["DepthAnything3MonoLarge", "backbone_config", "source_position_encoding"]
