"""ControlNet HED inference over BHWC image batches."""

from __future__ import annotations

import math
from typing import cast

import cv2
import numpy as np
import torch
from dinkster_api.v1 import declared_asset

from .cache import MODEL_CACHE

MAX_RESOLUTION = 16_384


class DoubleConvBlock(torch.nn.Module):
    def __init__(self, input_channels: int, output_channels: int, layer_count: int) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = [
            torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        ]
        layers.extend(
            torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
            for _ in range(1, layer_count)
        )
        self.convs = torch.nn.ModuleList(layers)
        self.projection = torch.nn.Conv2d(output_channels, 1, kernel_size=1)

    def forward(
        self, value: torch.Tensor, *, downsample: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if downsample:
            value = torch.nn.functional.max_pool2d(value, kernel_size=2, stride=2)
        for convolution in self.convs:
            value = convolution(value)
            value = torch.nn.functional.relu(value)
        return value, self.projection(value)


class ControlNetHED(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros((1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(3, 64, 2)
        self.block2 = DoubleConvBlock(64, 128, 2)
        self.block3 = DoubleConvBlock(128, 256, 3)
        self.block4 = DoubleConvBlock(256, 512, 3)
        self.block5 = DoubleConvBlock(512, 512, 3)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        value = value - self.norm
        projections: list[torch.Tensor] = []
        for index, block in enumerate(
            (self.block1, self.block2, self.block3, self.block4, self.block5)
        ):
            value, projection = block(value, downsample=index > 0)
            projections.append(projection)
        return tuple(projections)


def _load_model() -> ControlNetHED:
    reference = declared_asset("hed-model")
    with reference.open() as stream:
        state = torch.load(stream, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("HED model artifact must contain a state dictionary")
    model = ControlNetHED()
    model.load_state_dict(state, strict=True)
    return model.float()


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
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("HED preprocessing requires finite pixel values")
    source = np.clip(array, 0.0, 1.0) if minimum < 0.0 or maximum > 1.0 else array
    raster = np.empty(array.shape, dtype=np.uint8)
    np.multiply(source, 255.0, out=raster, casting="unsafe")
    return [_hwc3(frame) for frame in raster]


def _resize_with_pad(
    frame: np.ndarray,
    resolution: int,
    upscale_interpolation: int = cv2.INTER_CUBIC,
) -> tuple[np.ndarray, int, int]:
    if not 64 <= resolution <= MAX_RESOLUTION:
        raise ValueError(f"resolution must be between 64 and {MAX_RESOLUTION}")
    height, width = frame.shape[:2]
    scale = float(resolution) / float(min(height, width))
    target_height = int(np.round(float(height) * scale))
    target_width = int(np.round(float(width) * scale))
    resized = cv2.resize(
        frame,
        (target_width, target_height),
        interpolation=upscale_interpolation if scale > 1.0 else cv2.INTER_AREA,
    )
    pad_height = math.ceil(target_height / 64) * 64 - target_height
    pad_width = math.ceil(target_width / 64) * 64 - target_width
    padded = np.pad(resized, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge")
    return np.ascontiguousarray(padded), target_height, target_width


def _safe_step(edge: np.ndarray) -> np.ndarray:
    quantized = edge.astype(np.float32) * 3.0
    return quantized.astype(np.int32).astype(np.float32) / 2.0


def _nms(edge: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(edge.astype(np.float32), (0, 0), 3.0)
    kernels = (
        np.array(((0, 0, 0), (1, 1, 1), (0, 0, 0)), dtype=np.uint8),
        np.array(((0, 1, 0), (0, 1, 0), (0, 1, 0)), dtype=np.uint8),
        np.eye(3, dtype=np.uint8),
        np.fliplr(np.eye(3, dtype=np.uint8)),
    )
    maxima = np.zeros_like(blurred)
    for kernel in kernels:
        np.putmask(maxima, cv2.dilate(blurred, kernel=kernel) == blurred, blurred)
    output = np.zeros_like(maxima, dtype=np.uint8)
    output[maxima > 127] = 255
    return output


def _detect_frame(
    model: ControlNetHED,
    frame: np.ndarray,
    *,
    resolution: int,
    safe: bool,
    scribble: bool,
) -> np.ndarray:
    resized, target_height, target_width = _resize_with_pad(frame, resolution)
    height, width = resized.shape[:2]
    tensor = (
        torch.from_numpy(resized)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(next(model.parameters()).device)
    )
    with torch.no_grad():
        projections = model(tensor)
    edges = [
        cv2.resize(
            projection.detach().cpu().numpy().astype(np.float32)[0, 0],
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        for projection in projections
    ]
    mean = np.mean(np.stack(edges, axis=2), axis=2).astype(np.float64)
    edge = 1.0 / (1.0 + np.exp(-mean))
    if safe:
        edge = _safe_step(edge)
    detected = np.clip(edge * 255.0, 0.0, 255.0).astype(np.uint8)
    if scribble:
        detected = _nms(detected)
        detected = cv2.GaussianBlur(detected, (0, 0), 3.0)
        detected[detected > 4] = 255
        detected[detected < 255] = 0
    return _hwc3(np.ascontiguousarray(detected[:target_height, :target_width]))


def execute_hed(
    image: object,
    *,
    safe: bool,
    scribble: bool,
    resolution: int,
) -> np.ndarray:
    if type(safe) is not bool or type(scribble) is not bool:
        raise TypeError("safe and scribble must be booleans")
    with MODEL_CACHE.use("HED", _load_model) as cached:
        model = cast("ControlNetHED", cached)
        outputs = [
            _detect_frame(
                model,
                frame,
                resolution=resolution,
                safe=safe,
                scribble=scribble,
            )
            for frame in _frames(image)
        ]
    shape = outputs[0].shape
    if any(output.shape != shape for output in outputs):
        raise ValueError("HED preprocessing produced inconsistent batch dimensions")
    result = np.asarray(outputs, dtype=np.float32)
    result /= 255.0
    return cast("np.ndarray", np.ascontiguousarray(result))


__all__ = ["ControlNetHED", "DoubleConvBlock", "execute_hed"]
