"""Native DINOv2 backbone used by MoGe geometry estimation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from .attention import AttentionKernel, select_attention
from .operations import INITLESS, Operations, ResidencyRouted

_DEFAULT_ATTENTION = select_attention("clip").kernel


class Dino2Attention(torch.nn.Module):
    _attention_kernel: AttentionKernel

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.query = operations.linear(hidden_size, hidden_size)
        self.key = operations.linear(hidden_size, hidden_size)
        self.value = operations.linear(hidden_size, hidden_size)
        object.__setattr__(self, "_attention_kernel", attention_kernel)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = value.shape

        def split(projected: torch.Tensor) -> torch.Tensor:
            return projected.view(batch, tokens, self.heads, -1).transpose(1, 2)

        attended = self._attention_kernel(
            split(self.query(value)),
            split(self.key(value)),
            split(self.value(value)),
        )
        return attended.transpose(1, 2).reshape(batch, tokens, channels)


class Dino2AttentionOutput(torch.nn.Module):
    def __init__(self, hidden_size: int, *, operations: Operations) -> None:
        super().__init__()
        self.dense = operations.linear(hidden_size, hidden_size)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.dense(value)


class Dino2AttentionBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads: int,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.attention = Dino2Attention(
            hidden_size,
            heads,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.output = Dino2AttentionOutput(hidden_size, operations=operations)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(self.attention(value))


class LayerScale(ResidencyRouted, torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.lambda1 = torch.nn.Parameter(torch.empty(hidden_size))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            scale = self.lambda1.to(device=value.device, dtype=value.dtype)
        else:
            with binding.lease() as lease:
                scale = lease.get("lambda1", dtype=value.dtype)
        return value * scale


class Dino2Mlp(torch.nn.Module):
    def __init__(self, hidden_size: int, *, operations: Operations) -> None:
        super().__init__()
        self.fc1 = operations.linear(hidden_size, hidden_size * 4)
        self.fc2 = operations.linear(hidden_size * 4, hidden_size)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(value)))


class Dino2Block(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads: int,
        layer_norm_eps: float,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.attention = Dino2AttentionBlock(
            hidden_size,
            heads,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.layer_scale1 = LayerScale(hidden_size)
        self.layer_scale2 = LayerScale(hidden_size)
        self.mlp = Dino2Mlp(hidden_size, operations=operations)
        self.norm1 = operations.layer_norm(hidden_size, eps=layer_norm_eps)
        self.norm2 = operations.layer_norm(hidden_size, eps=layer_norm_eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.layer_scale1(self.attention(self.norm1(value)))
        return value + self.layer_scale2(self.mlp(self.norm2(value)))


class Dino2Encoder(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads: int,
        layers: int,
        layer_norm_eps: float,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.layer = torch.nn.ModuleList(
            Dino2Block(
                hidden_size,
                heads,
                layer_norm_eps,
                operations=operations,
                attention_kernel=attention_kernel,
            )
            for _ in range(layers)
        )


class Dino2PatchEmbeddings(torch.nn.Module):
    def __init__(self, hidden_size: int, *, operations: Operations) -> None:
        super().__init__()
        self.projection = operations.conv2d(
            3,
            hidden_size,
            14,
            stride=14,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value).flatten(2).transpose(1, 2)


class Dino2Embeddings(ResidencyRouted, torch.nn.Module):
    position_embeddings: torch.Tensor

    def __init__(
        self,
        hidden_size: int,
        position_tokens: int,
        use_mask_token: bool,
        *,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.patch_size = 14
        self.patch_embeddings = Dino2PatchEmbeddings(hidden_size, operations=operations)
        self.position_embeddings = torch.nn.Parameter(torch.empty(1, position_tokens, hidden_size))
        self.cls_token = torch.nn.Parameter(torch.empty(1, 1, hidden_size))
        self.mask_token = (
            torch.nn.Parameter(torch.empty(1, hidden_size)) if use_mask_token else None
        )

    def _interpolate_position(
        self,
        position: torch.Tensor,
        height: int,
        width: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        class_position = position[:, :1]
        patch_position = position[:, 1:]
        count = patch_position.shape[1]
        side = int(count**0.5)
        if count != side * side:
            raise ValueError(f"DINOv2 position grid has {count} non-square patches")
        rows = height // self.patch_size
        columns = width // self.patch_size
        patch_position = patch_position.reshape(1, side, side, -1).permute(0, 3, 1, 2)
        patch_position = F.interpolate(
            patch_position,
            scale_factor=((rows + 0.1) / side, (columns + 0.1) / side),
            mode="bicubic",
            antialias=False,
        )
        if patch_position.shape[-2:] != (rows, columns):
            raise ValueError("DINOv2 position interpolation produced the wrong grid")
        patch_position = patch_position.permute(0, 2, 3, 1).flatten(1, 2)
        return torch.cat((class_position, patch_position), dim=1).to(dtype)

    def _forward_owned(
        self,
        pixel_values: torch.Tensor,
        class_token: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        value = self.patch_embeddings(pixel_values)
        value = torch.cat((class_token.expand(value.shape[0], -1, -1), value), dim=1)
        if value.shape[1] != position.shape[1]:
            position = self._interpolate_position(
                position.float(),
                pixel_values.shape[-2],
                pixel_values.shape[-1],
                value.dtype,
            )
        return value + position.to(dtype=value.dtype)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            return self._forward_owned(
                pixel_values,
                self.cls_token.to(device=pixel_values.device, dtype=pixel_values.dtype),
                self.position_embeddings.to(device=pixel_values.device),
            )
        with binding.lease() as lease:
            return self._forward_owned(
                pixel_values,
                lease.get("cls_token", dtype=pixel_values.dtype),
                lease.get("position_embeddings", dtype=self.position_embeddings.dtype),
            )


class Dinov2Model(torch.nn.Module):
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        super().__init__()
        hidden_size = int(config["hidden_size"])
        self.embeddings = Dino2Embeddings(
            hidden_size,
            int(config["position_tokens"]),
            bool(config["use_mask_token"]),
            operations=operations,
        )
        self.encoder = Dino2Encoder(
            hidden_size,
            int(config["num_attention_heads"]),
            int(config["num_hidden_layers"]),
            float(config["layer_norm_eps"]),
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.layernorm = operations.layer_norm(
            hidden_size,
            eps=float(config["layer_norm_eps"]),
        )

    def get_intermediate_layers(
        self,
        pixel_values: torch.Tensor,
        indices: Sequence[int],
        *,
        apply_norm: bool = True,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        value = self.embeddings(pixel_values)
        layers = len(self.encoder.layer)
        resolved = tuple(index if index >= 0 else layers + index for index in indices)
        targets = set(resolved)
        outputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for index, layer in enumerate(self.encoder.layer):
            value = layer(value)
            if index in targets:
                normalized = self.layernorm(value) if apply_norm else value
                outputs[index] = (normalized[:, 1:], normalized[:, 0])
            if index >= max(resolved):
                break
        return [outputs[index] for index in resolved]
