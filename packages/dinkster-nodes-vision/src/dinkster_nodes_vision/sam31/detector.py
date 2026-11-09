"""SAM 3.1 text-prompted detector and mask head."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from .sam31 import MLP, FPNScaleConv, PositionEmbeddingSine, _cast_to_input


def box_cxcywh_to_xyxy(value: torch.Tensor) -> torch.Tensor:
    center_x, center_y, width, height = value.unbind(-1)
    return torch.stack(
        (
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ),
        dim=-1,
    )


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, query_tokens, channels = query.shape
    head_width = channels // heads

    def split(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(batch, -1, heads, head_width).transpose(1, 2)

    if mask is not None and mask.ndim == 2:
        mask = mask[:, None, None, :]
    output = F.scaled_dot_product_attention(
        split(query),
        split(key),
        split(value),
        attn_mask=mask,
    )
    return output.transpose(1, 2).reshape(batch, query_tokens, channels)


class SplitMHA(nn.Module):
    def __init__(
        self,
        width: int = 256,
        heads: int = 8,
        *,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(width, width, device=device)
        self.k_proj = nn.Linear(width, width, device=device)
        self.v_proj = nn.Linear(width, width, device=device)
        self.out_proj = nn.Linear(width, width, device=device)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key = query if key is None else key
        value = key if value is None else value
        return self.out_proj(
            _attention(
                self.q_proj(query),
                self.k_proj(key),
                self.v_proj(value),
                self.heads,
                mask,
            )
        )


class MLPWithNorm(nn.Module):
    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            (
                nn.Linear(256, 2048, device=device),
                nn.Linear(2048, 256, device=device),
            )
        )
        self.out_norm = nn.LayerNorm(256, device=device)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        original = value
        value = F.relu(self.layers[0](value))
        return self.out_norm(self.layers[1](value) + original)


class EncoderLayer(nn.Module):
    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.self_attn = SplitMHA(device=device)
        self.cross_attn_image = SplitMHA(device=device)
        self.linear1 = nn.Linear(256, 2048, device=device)
        self.linear2 = nn.Linear(2048, 256, device=device)
        self.norm1 = nn.LayerNorm(256, device=device)
        self.norm2 = nn.LayerNorm(256, device=device)
        self.norm3 = nn.LayerNorm(256, device=device)

    def forward(
        self,
        value: torch.Tensor,
        position: torch.Tensor,
        text: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.norm1(value)
        query = normalized + position
        value = value + self.self_attn(query, query, normalized)
        if text is not None:
            normalized = self.norm2(value)
            value = value + self.cross_attn_image(normalized, text, text, text_mask)
        normalized = self.norm3(value)
        return value + self.linear2(F.relu(self.linear1(normalized)))


class TransformerEncoder(nn.Module):
    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(device=device) for _ in range(6))

    def forward(
        self,
        value: torch.Tensor,
        position: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            value = layer(value, position, text, text_mask)
        return value


class DecoderLayer(nn.Module):
    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.self_attn = SplitMHA(device=device)
        self.cross_attn = SplitMHA(device=device)
        self.ca_text = SplitMHA(device=device)
        self.norm1 = nn.LayerNorm(256, device=device)
        self.norm2 = nn.LayerNorm(256, device=device)
        self.norm3 = nn.LayerNorm(256, device=device)
        self.catext_norm = nn.LayerNorm(256, device=device)
        self.linear1 = nn.Linear(256, 2048, device=device)
        self.linear2 = nn.Linear(2048, 256, device=device)

    def forward(
        self,
        value: torch.Tensor,
        memory: torch.Tensor,
        position: torch.Tensor,
        memory_position: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        cross_attention_bias: torch.Tensor,
    ) -> torch.Tensor:
        query = value + position
        value = self.norm2(value + self.self_attn(query, query, value))
        value = self.catext_norm(value + self.ca_text(value + position, text, text, text_mask))
        value = self.norm1(
            value
            + self.cross_attn(
                value + position,
                memory + memory_position,
                memory,
                cross_attention_bias,
            )
        )
        return self.norm3(value + self.linear2(F.relu(self.linear1(value))))


def _sine_position(value: torch.Tensor) -> torch.Tensor:
    half = 128
    frequencies = 10_000.0 ** (
        2 * (torch.arange(half, dtype=torch.float32, device=value.device) // 2) / half
    )
    parts: list[torch.Tensor] = []
    for index in range(value.shape[-1]):
        raw = (value[..., index].float() * 2 * math.pi).unsqueeze(-1) / frequencies
        parts.append(torch.stack((raw[..., 0::2].sin(), raw[..., 1::2].cos()), dim=-1).flatten(-2))
    return torch.cat(parts, dim=-1).to(value.dtype)


class TransformerDecoder(nn.Module):
    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer(device=device) for _ in range(6))
        self.norm = nn.LayerNorm(256, device=device)
        self.query_embed = nn.Embedding(200, 256, device=device)
        self.reference_points = nn.Embedding(200, 4, device=device)
        self.ref_point_head = MLP(512, 256, 256, 2, device=device)
        self.bbox_embed = MLP(256, 256, 4, 3, device=device)
        self.boxRPB_embed_x = MLP(2, 256, 8, 2, device=device)
        self.boxRPB_embed_y = MLP(2, 256, 8, 2, device=device)
        self.presence_token = nn.Embedding(1, 256, device=device)
        self.presence_token_head = MLP(256, 256, 1, 3, device=device)
        self.presence_token_out_norm = nn.LayerNorm(256, device=device)

    @staticmethod
    def _inverse_sigmoid(value: torch.Tensor) -> torch.Tensor:
        return torch.log(value / (1 - value + 1e-6) + 1e-6)

    def _box_bias(
        self,
        references: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        boxes = box_cxcywh_to_xyxy(references)
        batch, queries, _ = boxes.shape
        rows = torch.arange(height, device=references.device, dtype=torch.float32) / height
        columns = torch.arange(width, device=references.device, dtype=torch.float32) / width
        dx = columns.view(1, 1, -1, 1) - boxes[:, :, None, 0:3:2]
        dy = rows.view(1, 1, -1, 1) - boxes[:, :, None, 1:4:2]

        def scale(delta: torch.Tensor) -> torch.Tensor:
            return torch.sign(delta * 8) * torch.log2(torch.abs(delta * 8) + 1.0) / math.log2(8)

        x_bias = self.boxRPB_embed_x(scale(dx).to(references.dtype))
        y_bias = self.boxRPB_embed_y(scale(dy).to(references.dtype))
        bias = (y_bias.unsqueeze(3) + x_bias.unsqueeze(2)).flatten(2, 3).permute(0, 3, 1, 2)
        presence = torch.zeros(
            batch,
            bias.shape[1],
            1,
            bias.shape[3],
            device=bias.device,
            dtype=bias.dtype,
        )
        if bias.shape[2] != queries:
            raise ValueError("SAM 3.1 box bias query count does not match")
        return torch.cat((presence, bias), dim=2)

    def forward(
        self,
        memory: torch.Tensor,
        memory_position: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = memory.shape[0]
        queries = _cast_to_input(self.query_embed.weight, memory).unsqueeze(0).expand(batch, -1, -1)
        presence = _cast_to_input(self.presence_token.weight, memory)[None].expand(batch, -1, -1)
        references = (
            _cast_to_input(self.reference_points.weight, memory)
            .unsqueeze(0)
            .expand(batch, -1, -1)
            .sigmoid()
        )
        for index, layer in enumerate(self.layers):
            query_position = self.ref_point_head(_sine_position(references))
            combined = torch.cat((presence, queries), dim=1)
            combined_position = torch.cat((torch.zeros_like(presence), query_position), dim=1)
            combined = layer(
                combined,
                memory,
                combined_position,
                memory_position,
                text,
                text_mask,
                self._box_bias(references, height, width),
            )
            presence, queries = combined[:, :1], combined[:, 1:]
            if index < len(self.layers) - 1:
                references = (
                    (self._inverse_sigmoid(references) + self.bbox_embed(self.norm(queries)))
                    .sigmoid()
                    .detach()
                )
        queries = self.norm(queries)
        boxes = (self._inverse_sigmoid(references) + self.bbox_embed(queries)).sigmoid()
        return queries, boxes


class Transformer(nn.Module):
    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(device=device)
        self.decoder = TransformerDecoder(device=device)


class GeometryEncoder(nn.Module):
    """Checkpoint-complete geometry state used by the text prompt class token."""

    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.points_direct_project = nn.Linear(2, 256, device=device)
        self.points_pool_project = nn.Linear(256, 256, device=device)
        self.points_pos_enc_project = nn.Linear(256, 256, device=device)
        self.boxes_direct_project = nn.Linear(4, 256, device=device)
        self.boxes_pool_project = nn.Conv2d(256, 256, kernel_size=7, device=device)
        self.boxes_pos_enc_project = nn.Linear(258, 256, device=device)
        self.label_embed = nn.Embedding(2, 256, device=device)
        self.cls_embed = nn.Embedding(1, 256, device=device)
        self.norm = nn.LayerNorm(256, device=device)
        self.img_pre_norm = nn.LayerNorm(256, device=device)
        self.encode = nn.ModuleList(EncoderLayer(device=device) for _ in range(3))
        self.encode_norm = nn.LayerNorm(256, device=device)
        self.final_proj = nn.Linear(256, 256, device=device)


class PixelDecoder(nn.Module):
    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.conv_layers = nn.ModuleList(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, device=device) for _ in range(3)
        )
        self.norms = nn.ModuleList(nn.GroupNorm(8, 256, device=device) for _ in range(3))

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        output = features[-1]
        for index, feature in enumerate(features[:-1][::-1]):
            output = F.relu(
                self.norms[index](
                    self.conv_layers[index](
                        feature + F.interpolate(output, size=feature.shape[-2:], mode="nearest")
                    )
                )
            )
        return output


class MaskPredictor(nn.Module):
    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.mask_embed = MLP(256, 256, 256, 3, device=device)

    def forward(self, queries: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bqc,bchw->bqhw", self.mask_embed(queries), pixels)


class SegmentationHead(nn.Module):
    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.pixel_decoder = PixelDecoder(device=device)
        self.mask_predictor = MaskPredictor(device=device)
        self.cross_attend_prompt = SplitMHA(device=device)
        self.cross_attn_norm = nn.LayerNorm(256, device=device)
        self.instance_seg_head = nn.Conv2d(256, 256, kernel_size=1, device=device)
        self.semantic_seg_head = nn.Conv2d(256, 1, kernel_size=1, device=device)

    def forward(
        self,
        queries: torch.Tensor,
        features: list[torch.Tensor],
        memory: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.cross_attn_norm(memory)
        memory = memory + self.cross_attend_prompt(normalized, text, text, text_mask)
        batch, _, height, width = features[-1].shape
        visual = memory[:, : height * width].permute(0, 2, 1).view(batch, 256, height, width)
        inputs = [*features[:-1], visual]
        pixels = self.instance_seg_head(self.pixel_decoder(inputs))
        return self.mask_predictor(queries, pixels)


class DotProductScoring(nn.Module):
    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.hs_proj = nn.Linear(256, 256, device=device)
        self.prompt_proj = nn.Linear(256, 256, device=device)
        self.prompt_mlp = MLPWithNorm(device=device)

    def forward(
        self,
        queries: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        prompt = self.prompt_mlp(text)
        weight = text_mask.unsqueeze(-1).to(dtype=prompt.dtype)
        pooled = (prompt * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1)
        hidden = self.hs_proj(queries)
        projected = self.prompt_proj(pooled).unsqueeze(-1).to(hidden.dtype)
        return (torch.matmul(hidden, projected) / math.sqrt(256)).clamp(-12.0, 12.0).squeeze(-1)


class SAM31Detector(nn.Module):
    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__()
        self.fpn_convs = nn.ModuleList(
            FPNScaleConv(scale, device=device) for scale in (4.0, 2.0, 1.0)
        )
        self.position_encoding = PositionEmbeddingSine()
        self.language_resizer = nn.Linear(1024, 256, device=device)
        self.transformer = Transformer(device=device)
        self.segmentation_head = SegmentationHead(device=device)
        self.geometry_encoder = GeometryEncoder(device=device)
        self.dot_prod_scoring = DotProductScoring(device=device)

    @staticmethod
    def _run_geometry_class(
        layer: EncoderLayer,
        value: torch.Tensor,
        memory: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        value = value + layer.self_attn(layer.norm1(value))
        value = value + layer.cross_attn_image(layer.norm2(value), memory + position, memory)
        return value + layer.linear2(F.relu(layer.linear1(layer.norm3(value))))

    def forward(
        self,
        trunk: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = [convolution(trunk) for convolution in self.fpn_convs]
        positions = [
            _cast_to_input(self.position_encoding(feature), feature) for feature in features
        ]
        encoded_text = self.language_resizer(text)
        batch = trunk.shape[0]
        if encoded_text.shape[0] != batch:
            encoded_text = encoded_text.expand(batch, -1, -1)
            text_mask = text_mask.expand(batch, -1)
        feature, position = features[-1], positions[-1]
        _, _, height, width = feature.shape
        flattened = feature.flatten(2).permute(0, 2, 1)
        flattened_position = position.flatten(2).permute(0, 2, 1)

        geometry = self.geometry_encoder
        class_token = geometry.norm(
            geometry.final_proj(
                _cast_to_input(geometry.cls_embed.weight, flattened)
                .view(1, 1, -1)
                .expand(batch, -1, -1)
            )
        )
        for layer in geometry.encode:
            class_token = self._run_geometry_class(
                cast("EncoderLayer", layer),
                class_token,
                flattened,
                flattened_position,
            )
        class_token = geometry.encode_norm(class_token)
        encoded_text = torch.cat((encoded_text, class_token), dim=1)
        text_mask = torch.cat(
            (
                text_mask.bool(),
                torch.ones(batch, 1, dtype=torch.bool, device=text_mask.device),
            ),
            dim=1,
        )

        memory = self.transformer.encoder(
            flattened,
            flattened_position,
            encoded_text,
            text_mask,
        )
        queries, boxes = self.transformer.decoder(
            memory,
            flattened_position,
            encoded_text,
            text_mask,
            height,
            width,
        )
        scores = self.dot_prod_scoring(queries, encoded_text, text_mask)
        masks = self.segmentation_head(queries, features, memory, encoded_text, text_mask)
        return box_cxcywh_to_xyxy(boxes), scores, masks


__all__ = ["SAM31Detector", "box_cxcywh_to_xyxy"]
