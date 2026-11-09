"""SAM 3.1 interactive image segmentation architecture."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

IMAGE_SIZE = 1008
NO_OBJECT_SCORE = -1024.0


def _cast_to_input(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return value.to(device=reference.device, dtype=reference.dtype)


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    heads: int,
    *,
    split_heads: bool = False,
) -> torch.Tensor:
    if not split_heads:
        batch, tokens, channels = query.shape
        head_dim = channels // heads
        query = query.reshape(batch, tokens, heads, head_dim).transpose(1, 2)
        key = key.reshape(batch, key.shape[1], heads, head_dim).transpose(1, 2)
        value = value.reshape(batch, value.shape[1], heads, head_dim).transpose(1, 2)
    else:
        batch, _, tokens, head_dim = query.shape
    output = F.scaled_dot_product_attention(query, key, value)
    return output.transpose(1, 2).reshape(batch, tokens, heads * head_dim)


def _rope(position: torch.Tensor, dimension: int, theta: int) -> torch.Tensor:
    scale = torch.linspace(
        0,
        (dimension - 2) / dimension,
        steps=dimension // 2,
        dtype=torch.float64,
    )
    omega = 1.0 / (theta**scale)
    angles = torch.einsum("...n,d->...nd", position.to(dtype=torch.float32), omega)
    matrix = torch.stack(
        (torch.cos(angles), -torch.sin(angles), torch.sin(angles), torch.cos(angles)),
        dim=-1,
    )
    return matrix.reshape(*matrix.shape[:-1], 2, 2).to(dtype=torch.float32)


def _rope_2d(
    width: int,
    height: int,
    dimension: int,
    *,
    theta: int = 10_000,
    position_scale: float = 1.0,
) -> torch.Tensor:
    positions = torch.arange(width * height, dtype=torch.float32)
    coordinates = torch.stack(
        (
            (positions % width) * position_scale,
            torch.div(positions, width, rounding_mode="floor") * position_scale,
        ),
        dim=-1,
    ).unsqueeze(0)
    embedded = torch.cat(
        (
            _rope(coordinates[..., 0], dimension // 2, theta),
            _rope(coordinates[..., 1], dimension // 2, theta),
        ),
        dim=-3,
    )
    return embedded.unsqueeze(1)


def _apply_rope(value: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    pairs = value.to(dtype=frequencies.dtype).reshape(*value.shape[:-1], -1, 1, 2)
    if pairs.shape[2] != 1 and frequencies.shape[2] != 1 and pairs.shape[2] != frequencies.shape[2]:
        frequencies = frequencies[:, :, : pairs.shape[2]]
    output = frequencies[..., 0] * pairs[..., 0]
    output = output + frequencies[..., 1] * pairs[..., 1]
    return output.reshape_as(value).to(dtype=value.dtype)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        layers: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        dimensions = [input_dim] + [hidden_dim] * (layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(
                dimensions[index],
                dimensions[index + 1],
                device=device,
                dtype=dtype,
            )
            for index in range(layers)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            value = F.relu(layer(value)) if index < len(self.layers) - 1 else layer(value)
        return value


class SAMAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        heads: int,
        *,
        downsample_rate: int = 1,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = heads
        internal_dim = embedding_dim // downsample_rate
        self.q_proj = nn.Linear(embedding_dim, internal_dim, device=device)
        self.k_proj = nn.Linear(embedding_dim, internal_dim, device=device)
        self.v_proj = nn.Linear(embedding_dim, internal_dim, device=device)
        self.out_proj = nn.Linear(internal_dim, embedding_dim, device=device)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return self.out_proj(
            _attention(
                self.q_proj(query),
                self.k_proj(key),
                self.v_proj(value),
                self.num_heads,
            )
        )


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        heads: int,
        *,
        skip_first_layer_pe: bool,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.skip_first_layer_pe = skip_first_layer_pe
        self.self_attn = SAMAttention(embedding_dim, heads, device=device)
        self.cross_attn_token_to_image = SAMAttention(
            embedding_dim,
            heads,
            downsample_rate=2,
            device=device,
        )
        self.cross_attn_image_to_token = SAMAttention(
            embedding_dim,
            heads,
            downsample_rate=2,
            device=device,
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 2048, device=device),
            nn.ReLU(),
            nn.Linear(2048, embedding_dim, device=device),
        )
        self.norm1 = nn.LayerNorm(embedding_dim, device=device)
        self.norm2 = nn.LayerNorm(embedding_dim, device=device)
        self.norm3 = nn.LayerNorm(embedding_dim, device=device)
        self.norm4 = nn.LayerNorm(embedding_dim, device=device)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        query_pe: torch.Tensor,
        key_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.skip_first_layer_pe:
            queries = self.norm1(self.self_attn(queries, queries, queries))
        else:
            query = queries + query_pe
            queries = self.norm1(queries + self.self_attn(query, query, queries))
        query, key = queries + query_pe, keys + key_pe
        queries = self.norm2(queries + self.cross_attn_token_to_image(query, key, keys))
        queries = self.norm3(queries + self.mlp(queries))
        query, key = queries + query_pe, keys + key_pe
        keys = self.norm4(keys + self.cross_attn_image_to_token(key, query, queries))
        return queries, keys


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device | None = None,
        depth: int = 2,
        embedding_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        **_: object,
    ) -> None:
        super().__init__()
        if (depth, embedding_dim, num_heads, mlp_dim) != (2, 256, 8, 2048):
            raise ValueError("unsupported SAM 3.1 two-way transformer dimensions")
        self.layers = nn.ModuleList(
            TwoWayAttentionBlock(
                256,
                8,
                skip_first_layer_pe=index == 0,
                device=device,
            )
            for index in range(2)
        )
        self.final_attn_token_to_image = SAMAttention(
            256,
            8,
            downsample_rate=2,
            device=device,
        )
        self.norm_final = nn.LayerNorm(256, device=device)

    def forward(
        self,
        image_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        point_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        queries, keys = point_embedding, image_embedding
        for layer in self.layers:
            queries, keys = layer(queries, keys, point_embedding, image_pe)
        query, key = queries + point_embedding, keys + image_pe
        queries = self.norm_final(queries + self.final_attn_token_to_image(query, key, keys))
        return queries, keys


class PositionEmbeddingRandom(nn.Module):
    positional_encoding_gaussian_matrix: torch.Tensor

    def __init__(self, features: int, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            torch.empty(2, features, device=device),
        )

    def _encode(self, coordinates: torch.Tensor) -> torch.Tensor:
        dtype = coordinates.dtype
        projection = self.positional_encoding_gaussian_matrix.to(
            device=coordinates.device,
            dtype=torch.float32,
        )
        projected = 2 * math.pi * (2 * coordinates.float() - 1) @ projection
        return torch.cat((projected.sin(), projected.cos()), dim=-1).to(dtype)

    def forward(
        self,
        size: tuple[int, int],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        height, width = size
        target = device or self.positional_encoding_gaussian_matrix.device
        ones = torch.ones((height, width), device=target, dtype=torch.float32)
        coordinates = torch.stack(
            (
                (ones.cumsum(1) - 0.5) / width,
                (ones.cumsum(0) - 0.5) / height,
            ),
            dim=-1,
        )
        return self._encode(coordinates).permute(2, 0, 1).unsqueeze(0)

    def forward_with_coords(
        self,
        coordinates: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        normalized = coordinates.clone()
        normalized[:, :, 0] /= image_size[1]
        normalized[:, :, 1] /= image_size[0]
        return self._encode(normalized)


def _window_partition(
    value: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    batch, height, width, channels = value.shape
    pad_height = (window_size - height % window_size) % window_size
    pad_width = (window_size - width % window_size) % window_size
    if pad_height > 0 or pad_width > 0:
        value = F.pad(value, (0, 0, 0, pad_width, 0, pad_height))
    padded_height, padded_width = height + pad_height, width + pad_width
    value = value.view(
        batch,
        padded_height // window_size,
        window_size,
        padded_width // window_size,
        window_size,
        channels,
    )
    windows = value.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, channels), (
        padded_height,
        padded_width,
    )


def _window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    padded: tuple[int, int],
    original: tuple[int, int],
) -> torch.Tensor:
    padded_height, padded_width = padded
    height, width = original
    batch = windows.shape[0] // (padded_height * padded_width // window_size // window_size)
    value = windows.view(
        batch,
        padded_height // window_size,
        padded_width // window_size,
        window_size,
        window_size,
        -1,
    )
    value = value.permute(0, 1, 3, 2, 4, 5).contiguous()
    value = value.view(batch, padded_height, padded_width, -1)
    if padded_height > height or padded_width > width:
        value = value[:, :height, :width, :].contiguous()
    return value


class ViTMLP(nn.Module):
    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1024, 4736, device=device)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4736, 1024, device=device)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(value)))


class Attention(nn.Module):
    def __init__(
        self,
        *,
        use_rope: bool,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = 16
        self.head_dim = 64
        self.use_rope = use_rope
        self.qkv = nn.Linear(1024, 3072, device=device)
        self.proj = nn.Linear(1024, 1024, device=device)

    def forward(
        self,
        value: torch.Tensor,
        frequencies: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, tokens, channels = value.shape
        projected = self.qkv(value).reshape(
            batch,
            tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = projected.permute(2, 0, 3, 1, 4).unbind(dim=0)
        if self.use_rope and frequencies is not None:
            query = _apply_rope(query, frequencies)
            key = _apply_rope(key, frequencies)
        return self.proj(_attention(query, key, value, self.num_heads, split_heads=True))


class Block(nn.Module):
    def __init__(
        self,
        *,
        window_size: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(1024, device=device)
        self.attn = Attention(use_rope=True, device=device)
        self.norm2 = nn.LayerNorm(1024, device=device)
        self.mlp = ViTMLP(device=device)

    def forward(
        self,
        value: torch.Tensor,
        frequencies: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = value
        value = self.norm1(value)
        if self.window_size > 0:
            height, width = value.shape[1:3]
            value, padded = _window_partition(value, self.window_size)
            value = value.view(value.shape[0], self.window_size**2, -1)
            value = self.attn(value, frequencies)
            value = value.view(-1, self.window_size, self.window_size, value.shape[-1])
            value = _window_unpartition(
                value,
                self.window_size,
                padded,
                (height, width),
            )
        else:
            batch, height, width, channels = value.shape
            value = value.view(batch, height * width, channels)
            value = self.attn(value, frequencies)
            value = value.view(batch, height, width, channels)
        value = residual + value
        return value + self.mlp(self.norm2(value))


class PatchEmbed(nn.Module):
    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            3,
            1024,
            kernel_size=14,
            stride=14,
            bias=False,
            device=device,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.proj(image)


class ViTDet(nn.Module):
    freqs_cis: torch.Tensor
    freqs_cis_window: torch.Tensor

    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(device=device)
        self.pos_embed = nn.Parameter(torch.empty(1, 577, 1024, device=device))
        self.ln_pre = nn.LayerNorm(1024, device=device)
        global_blocks = {7, 15, 23, 31}
        self.blocks = nn.ModuleList(
            Block(
                window_size=0 if index in global_blocks else 24,
                device=device,
            )
            for index in range(32)
        )
        self.register_buffer(
            "freqs_cis",
            _rope_2d(72, 72, 64, position_scale=24 / 72),
            persistent=False,
        )
        self.register_buffer(
            "freqs_cis_window",
            _rope_2d(24, 24, 64),
            persistent=False,
        )

    def reset_frequencies(self) -> None:
        self.freqs_cis = _rope_2d(72, 72, 64, position_scale=24 / 72)
        self.freqs_cis_window = _rope_2d(24, 24, 64)

    def _position_embedding(self, tokens: int) -> torch.Tensor:
        position = self.pos_embed
        if position.shape[1] == tokens:
            return position
        class_position = position[:, :1]
        spatial = position[:, 1:]
        old_size = int(math.sqrt(spatial.shape[1]))
        new_size = int(math.sqrt(tokens - 1))
        spatial = spatial.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
        tiles = new_size // old_size + 1
        spatial = spatial.tile((1, 1, tiles, tiles))[:, :, :new_size, :new_size]
        spatial = spatial.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)
        return torch.cat((class_position, spatial), dim=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        value = self.patch_embed(image)
        batch, channels, height, width = value.shape
        value = value.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        position = _cast_to_input(self._position_embedding(height * width + 1), value)
        value = self.ln_pre(
            (value + position[:, 1 : height * width + 1]).view(
                batch,
                height,
                width,
                channels,
            )
        )
        global_frequencies = _cast_to_input(self.freqs_cis, value)
        window_frequencies = _cast_to_input(self.freqs_cis_window, value)
        for block in self.blocks:
            block = cast("Block", block)
            frequencies = window_frequencies if block.window_size > 0 else global_frequencies
            value = block(value, frequencies)
        return value.permute(0, 3, 1, 2)


class FPNScaleConv(nn.Module):
    def __init__(
        self,
        scale: float,
        *,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if scale == 4.0:
            self.dconv_2x2_0 = nn.ConvTranspose2d(
                1024,
                512,
                kernel_size=2,
                stride=2,
                device=device,
            )
            self.dconv_2x2_1 = nn.ConvTranspose2d(
                512,
                256,
                kernel_size=2,
                stride=2,
                device=device,
            )
            projected_channels = 256
        elif scale == 2.0:
            self.dconv_2x2 = nn.ConvTranspose2d(
                1024,
                512,
                kernel_size=2,
                stride=2,
                device=device,
            )
            projected_channels = 512
        else:
            projected_channels = 1024
        self.scale = scale
        self.conv_1x1 = nn.Conv2d(projected_channels, 256, kernel_size=1, device=device)
        self.conv_3x3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, device=device)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.scale == 4.0:
            value = F.gelu(self.dconv_2x2_0(value))
            value = self.dconv_2x2_1(value)
        elif self.scale == 2.0:
            value = self.dconv_2x2(value)
        return self.conv_3x3(self.conv_1x1(value))


class PositionEmbeddingSine(nn.Module):
    def __init__(self, features: int = 256) -> None:
        super().__init__()
        if features % 2:
            raise ValueError("position feature count must be even")
        self.half_dim = features // 2
        self._cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}

    def _sincos(self, values: torch.Tensor) -> torch.Tensor:
        frequencies = 10_000.0 ** (
            2
            * (torch.arange(self.half_dim, dtype=torch.float32, device=values.device) // 2)
            / self.half_dim
        )
        encoded = values[..., None] * (2 * math.pi) / frequencies
        return torch.stack(
            (encoded[..., 0::2].sin(), encoded[..., 1::2].cos()),
            dim=-1,
        ).flatten(-2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = value.shape
        key = (height, width, value.device)
        if key not in self._cache:
            y = torch.arange(height, dtype=torch.float32, device=value.device)
            x = torch.arange(width, dtype=torch.float32, device=value.device)
            y = y / (height - 1 + 1e-6)
            x = x / (width - 1 + 1e-6)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            self._cache[key] = (
                torch.cat(
                    (self._sincos(yy), self._sincos(xx)),
                    dim=-1,
                )
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
        return self._cache[key].expand(batch, -1, -1, -1)


class InteractiveBackbone(nn.Module):
    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.trunk = ViTDet(device=device)
        self.position_encoding = PositionEmbeddingSine()
        self.propagation_convs = nn.ModuleList(
            FPNScaleConv(scale, device=device) for scale in (4.0, 2.0, 1.0)
        )
        self.interactive_convs = nn.ModuleList(
            FPNScaleConv(scale, device=device) for scale in (4.0, 2.0, 1.0)
        )

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        trunk = self.trunk(image)
        return [convolution(trunk) for convolution in self.interactive_convs]

    def tracking_features(
        self,
        image: torch.Tensor,
        *,
        interactive: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor] | None]:
        trunk = self.trunk(image)
        propagation = [convolution(trunk) for convolution in self.propagation_convs]
        positions = [
            _cast_to_input(self.position_encoding(feature), feature) for feature in propagation
        ]
        prompted = (
            [convolution(trunk) for convolution in self.interactive_convs] if interactive else None
        )
        return propagation, positions, prompted


class LayerNorm2d(nn.LayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SAMMaskDecoder(nn.Module):
    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.num_mask_tokens = 4
        self.transformer = TwoWayTransformer(device=device)
        self.iou_token = nn.Embedding(1, 256, device=device)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, 256, device=device)
        self.obj_score_token = nn.Embedding(1, 256, device=device)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2, device=device),
            LayerNorm2d(64, device=device),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, device=device),
            nn.GELU(),
        )
        self.conv_s0 = nn.Conv2d(256, 32, kernel_size=1, device=device)
        self.conv_s1 = nn.Conv2d(256, 64, kernel_size=1, device=device)
        self.output_hypernetworks_mlps = nn.ModuleList(
            MLP(256, 256, 32, 3, device=device) for _ in range(self.num_mask_tokens)
        )
        self.iou_prediction_head = MLP(256, 256, self.num_mask_tokens, 3, device=device)
        self.pred_obj_score_head = MLP(256, 256, 1, 3, device=device)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        high_res_features: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = sparse_prompt_embeddings.shape[0]
        reference = sparse_prompt_embeddings
        tokens = torch.cat(
            (
                _cast_to_input(self.obj_score_token.weight, reference),
                _cast_to_input(self.iou_token.weight, reference),
                _cast_to_input(self.mask_tokens.weight, reference),
            ),
            dim=0,
        )
        tokens = torch.cat(
            (tokens.unsqueeze(0).expand(batch, -1, -1), sparse_prompt_embeddings),
            dim=1,
        )
        source = image_embeddings
        if source.shape[0] != batch:
            source = source.expand(batch, -1, -1, -1)
        source = source + dense_prompt_embeddings
        position = image_pe.expand(batch, -1, -1, -1)
        _, channels, height, width = source.shape
        hidden, source = self.transformer(
            source.flatten(2).permute(0, 2, 1),
            position.flatten(2).permute(0, 2, 1),
            tokens,
        )
        object_token = hidden[:, 0, :]
        mask_tokens = hidden[:, 2 : 2 + self.num_mask_tokens, :]
        source = source.permute(0, 2, 1).view(batch, channels, height, width)
        first_deconv, first_norm, first_activation, second_deconv, second_activation = (
            self.output_upscaling
        )
        upscaled = first_activation(
            first_norm(first_deconv(source) + self.conv_s1(high_res_features[1]))
        )
        upscaled = second_activation(second_deconv(upscaled) + self.conv_s0(high_res_features[0]))
        hypernetwork = torch.stack(
            [
                network(mask_tokens[:, index, :])
                for index, network in enumerate(self.output_hypernetworks_mlps)
            ],
            dim=1,
        )
        masks = (hypernetwork @ upscaled.flatten(2)).view(
            batch,
            self.num_mask_tokens,
            upscaled.shape[2],
            upscaled.shape[3],
        )
        return masks[:, 0:1], self.pred_obj_score_head(object_token)


class SAMPromptEncoder(nn.Module):
    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.embed_dim = 256
        self.image_embedding_size = (72, 72)
        self.input_image_size = (IMAGE_SIZE, IMAGE_SIZE)
        self.pe_layer = PositionEmbeddingRandom(128, device=device)
        self.point_embeddings = nn.ModuleList(nn.Embedding(1, 256, device=device) for _ in range(4))
        self.not_a_point_embed = nn.Embedding(1, 256, device=device)
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=2, stride=2, device=device),
            LayerNorm2d(4, device=device),
            nn.GELU(),
            nn.Conv2d(4, 16, kernel_size=2, stride=2, device=device),
            LayerNorm2d(16, device=device),
            nn.GELU(),
            nn.Conv2d(16, 256, kernel_size=1, device=device),
        )
        self.no_mask_embed = nn.Embedding(1, 256, device=device)

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size)

    def forward(
        self,
        points: tuple[torch.Tensor, torch.Tensor],
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coordinates, labels = points
        batch = coordinates.shape[0]
        sparse = torch.empty(
            (batch, 0, self.embed_dim),
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        if boxes is None:
            coordinates = torch.cat(
                (
                    coordinates,
                    torch.zeros(batch, 1, 2, device=coordinates.device, dtype=coordinates.dtype),
                ),
                dim=1,
            )
            labels = torch.cat(
                (
                    labels,
                    -torch.ones(batch, 1, device=labels.device, dtype=labels.dtype),
                ),
                dim=1,
            )
        point_embedding = self.pe_layer.forward_with_coords(
            coordinates + 0.5,
            self.input_image_size,
        )
        for index in range(4):
            embedding = cast("nn.Embedding", self.point_embeddings[index])
            point_embedding[labels == index] += _cast_to_input(
                cast("torch.Tensor", embedding.weight),
                coordinates,
            )
        invalid = labels == -1
        point_embedding[invalid] = 0.0
        point_embedding[invalid] += _cast_to_input(
            cast("torch.Tensor", self.not_a_point_embed.weight),
            coordinates,
        )
        sparse = torch.cat((sparse, point_embedding), dim=1)
        if boxes is not None:
            corners = self.pe_layer.forward_with_coords(
                boxes.reshape(-1, 2, 2) + 0.5,
                self.input_image_size,
            )
            corners[:, 0] += _cast_to_input(
                cast("torch.Tensor", cast("nn.Embedding", self.point_embeddings[2]).weight),
                coordinates,
            )
            corners[:, 1] += _cast_to_input(
                cast("torch.Tensor", cast("nn.Embedding", self.point_embeddings[3]).weight),
                coordinates,
            )
            sparse = torch.cat((sparse, corners), dim=1)
        if masks is not None:
            dense = self.mask_downscaling(masks)
        else:
            dense = _cast_to_input(
                cast("torch.Tensor", self.no_mask_embed.weight),
                coordinates,
            )
            dense = dense.reshape(1, -1, 1, 1).expand(batch, -1, 72, 72)
        return sparse, dense


class SAM31InteractiveModel(nn.Module):
    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.backbone = InteractiveBackbone(device=device)
        self.interactive_sam_mask_decoder = SAMMaskDecoder(device=device)
        self.interactive_sam_prompt_encoder = SAMPromptEncoder(device=device)
        self.interactivity_no_mem_embed = nn.Parameter(torch.empty(1, 1, 256, device=device))

    def encode_image(self, image: torch.Tensor) -> list[torch.Tensor]:
        return self.backbone(image)

    def reset_frequencies(self) -> None:
        self.backbone.trunk.reset_frequencies()

    def segment(
        self,
        features: list[torch.Tensor],
        *,
        box: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        high_resolution = features[:-1]
        backbone = features[-1]
        batch, channels, height, width = backbone.shape
        flattened = backbone.flatten(2).permute(0, 2, 1)
        flattened = flattened + _cast_to_input(self.interactivity_no_mem_embed, flattened)
        backbone = flattened.view(batch, height, width, channels).permute(0, 3, 1, 2)
        prompt_batch = (
            mask.shape[0] if mask is not None else box.shape[0] if box is not None else batch
        )
        point_coordinates = torch.zeros(
            prompt_batch,
            1,
            2,
            device=backbone.device,
            dtype=backbone.dtype,
        )
        point_labels = -torch.ones(
            prompt_batch,
            1,
            device=backbone.device,
            dtype=torch.int32,
        )
        if mask is not None and mask.shape[-2:] != (288, 288):
            mask = F.interpolate(
                mask,
                size=(288, 288),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        sparse, dense = self.interactive_sam_prompt_encoder(
            (point_coordinates, point_labels),
            box,
            mask,
        )
        sparse = _cast_to_input(sparse, backbone)
        dense = _cast_to_input(dense, backbone)
        position = _cast_to_input(
            self.interactive_sam_prompt_encoder.get_dense_pe(),
            backbone,
        )
        masks, object_scores = self.interactive_sam_mask_decoder(
            backbone,
            position,
            sparse,
            dense,
            high_resolution,
        )
        masks = torch.where(
            (object_scores > 0)[:, None, None],
            masks,
            torch.tensor(NO_OBJECT_SCORE, device=masks.device, dtype=masks.dtype),
        )
        return F.interpolate(
            masks,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )


__all__ = ["IMAGE_SIZE", "SAM31InteractiveModel"]
