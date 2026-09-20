"""DETR ResNet-50 object detection over BHWC image batches.

The module structure and forward math mirror facebookresearch/detr at commit
29901c51d7fe8712168b8d0d64351170bc0f83e0 so the original detr-r50 checkpoint
loads with strict key matching. The pinned parity vector validates bit-identical
CPU outputs under its documented single-thread execution contract.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from dinkster_api.v1 import Detection, Region, declared_asset
from torch import nn

# COCO's 91-id category list as used by the DETR checkpoint's class head.
# Ids without annotations in the 2017 dataset keep their "N/A" placeholder so
# indices line up with the model's label outputs.
COCO_CLASSES: tuple[str, ...] = (
    "N/A",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_SHORTEST_SIDE = 800
_LONGEST_SIDE = 1333


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with fixed statistics and affine parameters.

    The eps is added before rsqrt exactly as the reference does; the frozen
    checkpoint's num_batches_tracked entries are dropped on load.
    """

    weight: torch.Tensor
    bias: torch.Tensor
    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, n: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(  # noqa: PLR0913
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        state_dict.pop(prefix + "num_batches_tracked", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        scale = weight * (running_var + 1e-5).rsqrt()
        return x * scale + (bias - running_mean * scale)


class Bottleneck(nn.Module):
    """torchvision-compatible ResNet bottleneck with frozen normalization."""

    def __init__(self, in_channels: int, width: int, stride: int, downsample: bool) -> None:
        super().__init__()
        out_channels = width * 4
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = FrozenBatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = FrozenBatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, bias=False)
        self.bn3 = FrozenBatchNorm2d(out_channels)
        self.downsample: nn.Sequential | None = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                FrozenBatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return F.relu(out)


def _make_layer(in_channels: int, width: int, blocks: int, stride: int) -> nn.Sequential:
    layers = [Bottleneck(in_channels, width, stride, downsample=True)]
    layers.extend(Bottleneck(width * 4, width, 1, downsample=False) for _ in range(1, blocks))
    return nn.Sequential(*layers)


class ResNet50Body(nn.Module):
    """ResNet-50 trunk producing the layer4 feature map."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = FrozenBatchNorm2d(64)
        self.layer1 = _make_layer(64, 64, 3, 1)
        self.layer2 = _make_layer(256, 128, 4, 2)
        self.layer3 = _make_layer(512, 256, 6, 2)
        self.layer4 = _make_layer(1024, 512, 3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


class BackboneWrapper(nn.Module):
    """Holds the trunk under a ``body`` attribute to match checkpoint keys."""

    def __init__(self) -> None:
        super().__init__()
        self.body = ResNet50Body()


def _position_embedding_sine(mask: torch.Tensor) -> torch.Tensor:
    """Sine positional embedding over the unpadded feature grid."""
    num_pos_feats = 128
    temperature = 10000
    scale = 2 * math.pi
    not_mask = ~mask
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)
    eps = 1e-6
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=mask.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class TransformerEncoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(256, 8, dropout=0.1)
        self.linear1 = nn.Linear(256, 2048)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(2048, 256)
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        q = k = src + pos
        src2 = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)[0]
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        return self.norm2(src + self.dropout2(src2))


class TransformerDecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(256, 8, dropout=0.1)
        self.multihead_attn = nn.MultiheadAttention(256, 8, dropout=0.1)
        self.linear1 = nn.Linear(256, 2048)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(2048, 256)
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(256)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)

    def forward(  # noqa: PLR0913
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        pos: torch.Tensor,
        query_pos: torch.Tensor,
    ) -> torch.Tensor:
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.multihead_attn(
            query=tgt + query_pos,
            key=memory + pos,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout3(tgt2))


class TransformerEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(TransformerEncoderLayer() for _ in range(6))

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask, pos=pos)
        return output


class TransformerDecoder(nn.Module):
    """Six decoder layers with a final norm, returning the last layer only.

    The reference stacks per-layer intermediates for auxiliary losses; at
    inference only the normed final layer is consumed, so computing exactly
    that value keeps outputs bit-identical while skipping the stack.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(TransformerDecoderLayer() for _ in range(6))
        self.norm = nn.LayerNorm(256)

    def forward(  # noqa: PLR0913
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        pos: torch.Tensor,
        query_pos: torch.Tensor,
    ) -> torch.Tensor:
        output = tgt
        for layer in self.layers:
            output = layer(
                output,
                memory,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
        return self.norm(output)


class Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()

    def forward(
        self,
        src: torch.Tensor,
        mask: torch.Tensor,
        query_embed: torch.Tensor,
        pos_embed: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = src.shape
        del channels, height, width
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, batch, 1)
        mask = mask.flatten(1)
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        return hs.transpose(0, 1)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        dims = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim, *dims], [*dims, output_dim], strict=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if index < self.num_layers - 1 else layer(x)
        return x


class DETR(nn.Module):
    """DETR R50 with 6 encoder and 6 decoder layers over 100 object queries."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.ModuleList([BackboneWrapper()])
        self.transformer = Transformer()
        self.class_embed = nn.Linear(256, len(COCO_CLASSES) + 1)
        self.bbox_embed = MLP(256, 256, 4, 3)
        self.query_embed = nn.Embedding(100, 256)
        self.input_proj = nn.Conv2d(2048, 256, kernel_size=1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one normalized CHW image; returns per-query logits and boxes."""
        features = cast("BackboneWrapper", self.backbone[0]).body(image.unsqueeze(0))
        mask = torch.zeros(
            (1, features.shape[2], features.shape[3]),
            dtype=torch.bool,
            device=features.device,
        )
        pos = _position_embedding_sine(mask)
        hs = self.transformer(self.input_proj(features), mask, self.query_embed.weight, pos)
        logits = self.class_embed(hs)
        boxes = self.bbox_embed(hs).sigmoid()
        return logits[0], boxes[0]


_MODEL: DETR | None = None
_MODEL_DIGEST = ""


def load_model() -> DETR:
    """Load the declared DETR checkpoint, cached per asset digest."""
    global _MODEL, _MODEL_DIGEST
    reference = declared_asset("detr-r50")
    if _MODEL is not None and _MODEL_DIGEST == reference.digest:
        return _MODEL
    with reference.open() as stream:
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("DETR model artifact must contain a state dictionary under 'model'")
    model = DETR()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.float().eval()
    _MODEL = model
    _MODEL_DIGEST = reference.digest
    return model


def _frames(image: object) -> list[np.ndarray]:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 4 or array.shape[0] < 1 or array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError(f"image must have non-empty BHWC shape, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("object detection requires finite pixel values")
    array = np.clip(array, 0.0, 1.0)
    if array.shape[3] == 3:
        return list(array)
    if array.shape[3] == 1:
        return list(np.repeat(array, 3, axis=3))
    if array.shape[3] == 4:
        color = array[..., :3]
        alpha = array[..., 3:4]
        return list(np.clip(color * alpha + (1.0 - alpha), 0.0, 1.0))
    raise ValueError(f"image frame must have 1, 3, or 4 channels, got {array.shape}")


def prepare_frame(frame: np.ndarray) -> torch.Tensor:
    """Resize an HWC [0, 1] frame to DETR's evaluation size and normalize.

    The shortest side scales to 800 pixels, capped so the longest side stays
    within 1333, with bilinear antialiased resampling and ImageNet
    normalization.
    """
    height, width = frame.shape[:2]
    scale = _SHORTEST_SIDE / min(height, width)
    if scale * max(height, width) > _LONGEST_SIDE:
        scale = _LONGEST_SIDE / max(height, width)
    target_height = max(1, int(round(height * scale)))
    target_width = max(1, int(round(width * scale)))
    tensor = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)
    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0]
    mean = torch.tensor(_IMAGENET_MEAN).reshape(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).reshape(3, 1, 1)
    return (tensor - mean) / std


def _parse_prompt(prompt: str, prompt_mode: str) -> frozenset[str]:
    if type(prompt) is not str:
        raise TypeError("prompt must be a string")
    if prompt_mode == "literal":
        stripped = prompt.strip()
        return frozenset((stripped.casefold(),)) if stripped else frozenset()
    if prompt_mode != "comma-separated":
        raise ValueError(f"unknown prompt mode: {prompt_mode}")
    return frozenset(name.strip().casefold() for name in prompt.split(",") if name.strip())


def _frame_detections(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    *,
    frame_height: int,
    frame_width: int,
    min_score: float,
    wanted: frozenset[str],
    max_results: int,
    result_limit_mode: str,
) -> list[Detection]:
    prob = logits.softmax(-1)
    scores, labels = prob[:, :-1].max(-1)
    centers_x, centers_y, widths, heights = boxes.unbind(-1)
    x0 = (centers_x - 0.5 * widths) * frame_width
    y0 = (centers_y - 0.5 * heights) * frame_height
    x1 = (centers_x + 0.5 * widths) * frame_width
    y1 = (centers_y + 0.5 * heights) * frame_height
    order = torch.argsort(scores, descending=True, stable=True)
    detections: list[Detection] = []
    for index in order.tolist():
        score = float(scores[index])
        if score < min_score:
            break
        label = COCO_CLASSES[int(labels[index])]
        if wanted and label.casefold() not in wanted:
            continue
        left = min(max(float(x0[index]), 0.0), float(frame_width))
        top = min(max(float(y0[index]), 0.0), float(frame_height))
        right = min(max(float(x1[index]), 0.0), float(frame_width))
        bottom = min(max(float(y1[index]), 0.0), float(frame_height))
        region = Region(left, top, max(right - left, 0.0), max(bottom - top, 0.0))
        detections.append(Detection(label, score, region))
        if result_limit_mode == "count" and len(detections) == max_results:
            break
    return detections[:max_results] if result_limit_mode == "slice-stop" else detections


def execute_detect(
    image: object,
    *,
    prompt: str,
    prompt_mode: str = "comma-separated",
    min_score: float,
    max_results: int = -1,
    result_limit_mode: str = "count",
) -> list[Detection]:
    """Detect COCO objects in every frame, concatenated in frame order.

    The prompt is a case-insensitive COCO class-name filter; comma-separated
    mode accepts multiple names. An empty prompt keeps every class, and names
    outside the class set match nothing. Detections within each frame are
    ordered by descending score.
    """
    if type(min_score) not in (int, float):
        raise TypeError("min_score must be a number")
    threshold = float(min_score)
    if not math.isfinite(threshold):
        raise ValueError("min_score must be finite")
    if result_limit_mode not in ("count", "slice-stop"):
        raise ValueError(f"unknown result limit mode: {result_limit_mode}")
    if type(max_results) is not int or (result_limit_mode == "count" and max_results < -1):
        raise ValueError("max_results must be an integer and at least -1 in count mode")
    wanted = _parse_prompt(prompt, prompt_mode)
    frames = _frames(image)
    if max_results == 0:
        return []
    model = load_model()
    detections: list[Detection] = []
    for frame in frames:
        with torch.no_grad():
            logits, boxes = model(prepare_frame(frame))
        detections.extend(
            _frame_detections(
                logits,
                boxes,
                frame_height=frame.shape[0],
                frame_width=frame.shape[1],
                min_score=threshold,
                wanted=wanted,
                max_results=max_results,
                result_limit_mode=result_limit_mode,
            )
        )
    return detections


__all__ = ["COCO_CLASSES", "DETR", "execute_detect", "load_model", "prepare_frame"]
