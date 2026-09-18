"""Depth Anything V2 Large inference over BHWC image batches."""

from __future__ import annotations

import math
from types import MethodType
from typing import Any, cast

import cv2
import numpy as np
import torch
from dinkster_api.v1 import declared_asset
from safetensors.torch import load_file
from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation, Dinov2Config

MAX_RESOLUTION = 16_384
MODEL_INPUT_SIZE = 518
MODEL_MULTIPLE = 14

_MODEL: DepthAnythingForDepthEstimation | None = None
_MODEL_DIGEST = ""


def _model_config() -> DepthAnythingConfig:
    backbone = Dinov2Config(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        image_size=MODEL_INPUT_SIZE,
        patch_size=MODEL_MULTIPLE,
        reshape_hidden_states=False,
    )
    backbone.out_features = ["stage5", "stage12", "stage18", "stage24"]
    return DepthAnythingConfig(
        backbone_config=backbone,
        patch_size=MODEL_MULTIPLE,
        reassemble_hidden_size=1024,
        reassemble_factors=[4, 2, 1, 0.5],
        neck_hidden_sizes=[256, 512, 1024, 1024],
        fusion_hidden_size=256,
        head_hidden_size=32,
        head_in_index=-1,
        depth_estimation_type="relative",
        max_depth=1,
    )


def _source_position_encoding(
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
    class_position = positions[:, 0]
    patch_positions = positions[:, 1:].float()
    dimension = embeddings.shape[-1]
    source_width = math.sqrt(position_count)
    target_height = height // embeddings_module.patch_size
    target_width = width // embeddings_module.patch_size
    patch_positions = torch.nn.functional.interpolate(
        patch_positions.reshape(1, int(source_width), int(source_width), dimension).permute(
            0, 3, 1, 2
        ),
        scale_factor=(
            (target_height + 0.1) / source_width,
            (target_width + 0.1) / source_width,
        ),
        mode="bicubic",
        antialias=False,
    )
    if patch_positions.shape[-2:] != (target_height, target_width):
        raise ValueError("Depth Anything V2 position interpolation produced the wrong shape")
    flattened = patch_positions.permute(0, 2, 3, 1).view(1, -1, dimension)
    return torch.cat((class_position.unsqueeze(0), flattened), dim=1).to(embeddings.dtype)


def _source_attention(
    attention_module: Any,
    hidden_states: torch.Tensor,
    **_kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, tokens, channels = hidden_states.shape
    weights = torch.cat(
        (
            attention_module.query.weight,
            attention_module.key.weight,
            attention_module.value.weight,
        )
    )
    bias = torch.cat(
        (
            attention_module.query.bias,
            attention_module.key.bias,
            attention_module.value.bias,
        )
    )
    qkv = torch.nn.functional.linear(hidden_states, weights, bias)
    qkv = qkv.reshape(
        batch,
        tokens,
        3,
        attention_module.num_attention_heads,
        attention_module.attention_head_size,
    ).permute(2, 0, 3, 1, 4)
    query, key, value = qkv[0] * attention_module.scaling, qkv[1], qkv[2]
    probabilities = torch.softmax(query @ key.transpose(-2, -1), dim=-1)
    output = (probabilities @ value).transpose(1, 2).reshape(batch, tokens, channels)
    return output, probabilities


def _load_model() -> DepthAnythingForDepthEstimation:
    global _MODEL, _MODEL_DIGEST
    reference = declared_asset("depth-anything-v2-large")
    if _MODEL is not None and _MODEL_DIGEST == reference.digest:
        return _MODEL
    with torch.device("meta"):
        model = DepthAnythingForDepthEstimation(_model_config())
    state = load_file(reference.local_path(), device="cpu")
    model.load_state_dict(state, strict=True, assign=True)
    model.backbone.embeddings.interpolate_pos_encoding = MethodType(
        _source_position_encoding,
        model.backbone.embeddings,
    )
    for layer in model.backbone.encoder.layer:
        attention = layer.attention.attention
        attention.forward = MethodType(_source_attention, attention)
    model.float().eval()
    _MODEL = model
    _MODEL_DIGEST = reference.digest
    return model


def _hwc3(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        frame = frame[:, :, None]
    if frame.ndim != 3 or frame.shape[2] not in (1, 3, 4):
        raise ValueError(f"image frame must have 1, 3, or 4 channels, got {frame.shape}")
    if frame.shape[2] == 3:
        return np.ascontiguousarray(frame)
    if frame.shape[2] == 1:
        return np.ascontiguousarray(np.repeat(frame, 3, axis=2))
    color = frame[:, :, :3].astype(np.float32)
    alpha = frame[:, :, 3:4].astype(np.float32) / 255.0
    return np.ascontiguousarray(
        np.clip(color * alpha + 255.0 * (1.0 - alpha), 0.0, 255.0).astype(np.uint8)
    )


def _frames(image: object) -> list[np.ndarray]:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 4 or array.shape[0] < 1 or array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError(f"image must have non-empty BHWC shape, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Depth Anything V2 preprocessing requires finite pixel values")
    raster = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    return [_hwc3(frame) for frame in raster]


def _multiple(value: float, *, minimum: int) -> int:
    rounded = int(np.round(value / MODEL_MULTIPLE) * MODEL_MULTIPLE)
    if rounded < minimum:
        rounded = int(np.ceil(value / MODEL_MULTIPLE) * MODEL_MULTIPLE)
    return rounded


def _model_input(frame: np.ndarray) -> torch.Tensor:
    height, width = frame.shape[:2]
    scale = max(MODEL_INPUT_SIZE / height, MODEL_INPUT_SIZE / width)
    target_height = _multiple(scale * height, minimum=MODEL_INPUT_SIZE)
    target_width = _multiple(scale * width, minimum=MODEL_INPUT_SIZE)
    resized = cv2.resize(
        frame / 255.0,
        (target_width, target_height),
        interpolation=cv2.INTER_CUBIC,
    )
    mean = np.array((0.485, 0.456, 0.406))
    std = np.array((0.229, 0.224, 0.225))
    normalized = (resized - mean) / std
    prepared = np.ascontiguousarray(normalized.transpose(2, 0, 1)).astype(np.float32)
    return torch.from_numpy(prepared).unsqueeze(0)


def _relative_depth(model: DepthAnythingForDepthEstimation, frame: np.ndarray) -> np.ndarray:
    model_input = _model_input(frame)
    with torch.inference_mode():
        predicted = model(model_input).predicted_depth
        resized = torch.nn.functional.interpolate(
            predicted[:, None],
            frame.shape[:2],
            mode="bilinear",
            align_corners=True,
        )[0, 0]
    depth = resized.cpu().numpy()
    minimum = float(depth.min())
    maximum = float(depth.max())
    if maximum == minimum:
        return np.zeros(frame.shape[:2], dtype=np.uint8)
    return np.clip((depth - minimum) / (maximum - minimum) * 255.0, 0.0, 255.0).astype(np.uint8)


def _validate_resolution(resolution: int) -> None:
    if type(resolution) is not int or not 64 <= resolution <= MAX_RESOLUTION:
        raise ValueError(f"resolution must be an integer between 64 and {MAX_RESOLUTION}")


def _resize_hint(depth: np.ndarray, resolution: int) -> np.ndarray:
    _validate_resolution(resolution)
    height, width = depth.shape
    scale = float(resolution) / float(min(height, width))
    target_height = int(np.round(float(height) * scale))
    target_width = int(np.round(float(width) * scale))
    rgb = np.repeat(depth[:, :, None], 3, axis=2)
    return np.ascontiguousarray(
        cv2.resize(
            rgb,
            (target_width, target_height),
            interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
        )
    )


def execute_depth_anything_v2(image: object, *, resolution: int) -> np.ndarray:
    _validate_resolution(resolution)
    frames = _frames(image)
    model = _load_model()
    outputs = [_resize_hint(_relative_depth(model, frame), resolution) for frame in frames]
    shape = outputs[0].shape
    if any(output.shape != shape for output in outputs):
        raise ValueError("Depth Anything V2 preprocessing produced inconsistent batch dimensions")
    result = np.stack(outputs).astype(np.float32) / 255.0
    return cast("np.ndarray", np.ascontiguousarray(result))


__all__ = ["execute_depth_anything_v2"]
