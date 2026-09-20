"""Depth Anything 3 Mono Large inference over BHWC image batches."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
import torch
from dinkster_api.v1 import declared_asset
from PIL import Image
from safetensors.torch import load_file

from .da3 import PATCH_SIZE, DepthAnything3MonoLarge

MAX_RESOLUTION = 16_384
MODEL_INPUT_SIZE = 504

_MODEL: DepthAnything3MonoLarge | None = None
_MODEL_DIGEST = ""


def load_model() -> DepthAnything3MonoLarge:
    global _MODEL, _MODEL_DIGEST
    reference = declared_asset("depth-anything-3-mono-large")
    if _MODEL is not None and _MODEL_DIGEST == reference.digest:
        return _MODEL
    with torch.device("meta"):
        model = DepthAnything3MonoLarge()
    stored = load_file(reference.local_path(), device="cpu")
    state = {name.removeprefix("model."): value for name, value in stored.items()}
    model.load_state_dict(state, strict=True, assign=True)
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


def frames(image: object) -> list[np.ndarray]:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 4 or array.shape[0] < 1 or array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError(f"image must have non-empty BHWC shape, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Depth Anything 3 preprocessing requires finite pixel values")
    raster = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    return [_hwc3(frame) for frame in raster]


def _round_to_patch(value: int) -> int:
    down = (value // PATCH_SIZE) * PATCH_SIZE
    up = down + PATCH_SIZE
    return max(PATCH_SIZE, up if abs(up - value) <= abs(value - down) else down)


def target_size(height: int, width: int) -> tuple[int, int]:
    scale = MODEL_INPUT_SIZE / float(max(height, width))
    return (
        _round_to_patch(round(height * scale)),
        _round_to_patch(round(width * scale)),
    )


def prepare_frame(frame: np.ndarray) -> torch.Tensor:
    height, width = target_size(*frame.shape[:2])
    raster = Image.fromarray(frame).resize((width, height), resample=Image.Resampling.LANCZOS)
    array = np.asarray(raster).astype(np.float32) / 255.0
    prepared = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).unsqueeze(0)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=prepared.dtype).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=prepared.dtype).view(1, 3, 1, 1)
    return (prepared - mean) / std


def relative_depth(model: DepthAnything3MonoLarge, frame: np.ndarray) -> np.ndarray:
    prepared = prepare_frame(frame)
    with torch.inference_mode():
        predicted = model(prepared)["depth"]
        resized = torch.nn.functional.interpolate(
            predicted[:, None],
            size=frame.shape[:2],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    return np.ascontiguousarray(resized.cpu().numpy(), dtype=np.float32)


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    minimum = float(depth.min())
    maximum = float(depth.max())
    scale = max(maximum - minimum, 1e-6)
    return np.ascontiguousarray(1.0 - np.clip((depth - minimum) / scale, 0.0, 1.0))


def _validate_resolution(resolution: int) -> None:
    if type(resolution) is not int or not 64 <= resolution <= MAX_RESOLUTION:
        raise ValueError(f"resolution must be an integer between 64 and {MAX_RESOLUTION}")


def resize_hint(depth: np.ndarray, resolution: int) -> np.ndarray:
    _validate_resolution(resolution)
    height, width = depth.shape
    scale = float(resolution) / float(min(height, width))
    target_height = int(np.round(float(height) * scale))
    target_width = int(np.round(float(width) * scale))
    resized = cv2.resize(
        depth,
        (target_width, target_height),
        interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
    )
    rgb = np.repeat(np.clip(resized, 0.0, 1.0)[:, :, None], 3, axis=2)
    return np.ascontiguousarray(rgb, dtype=np.float32)


def execute_depth_anything_v3(image: object, *, resolution: int) -> np.ndarray:
    _validate_resolution(resolution)
    inputs = frames(image)
    model = load_model()
    outputs = [
        resize_hint(normalize_depth(relative_depth(model, frame)), resolution) for frame in inputs
    ]
    shape = outputs[0].shape
    if any(output.shape != shape for output in outputs):
        raise ValueError("Depth Anything 3 preprocessing produced inconsistent batch dimensions")
    contiguous = np.ascontiguousarray(np.stack(outputs), dtype=np.float32)
    immutable = np.frombuffer(contiguous.tobytes(), dtype=np.float32).reshape(contiguous.shape)
    return cast("np.ndarray", immutable)


__all__ = [
    "execute_depth_anything_v3",
    "frames",
    "load_model",
    "normalize_depth",
    "prepare_frame",
    "relative_depth",
    "resize_hint",
    "target_size",
]
