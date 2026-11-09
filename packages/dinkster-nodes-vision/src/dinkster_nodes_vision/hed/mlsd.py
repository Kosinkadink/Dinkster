"""Mobile Line Segment Detection inference."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
import torch
from dinkster_api.v1 import declared_asset

from .cache import MODEL_CACHE
from .model import _frames, _hwc3, _resize_with_pad


class BlockTypeA(torch.nn.Module):
    def __init__(
        self,
        input_a: int,
        input_b: int,
        output_a: int,
        output_b: int,
        *,
        upscale: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_b, output_b, 1),
            torch.nn.BatchNorm2d(output_b),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(input_a, output_a, 1),
            torch.nn.BatchNorm2d(output_a),
            torch.nn.ReLU(inplace=True),
        )
        self.upscale = upscale

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        b = self.conv1(b)
        a = self.conv2(a)
        if self.upscale:
            b = torch.nn.functional.interpolate(
                b, scale_factor=2.0, mode="bilinear", align_corners=True
            )
        return torch.cat((a, b), dim=1)


class BlockTypeB(torch.nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, input_channels, 3, padding=1),
            torch.nn.BatchNorm2d(input_channels),
            torch.nn.ReLU(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, output_channels, 3, padding=1),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.conv1(value) + value
        value = self.conv2(value)
        return value


class BlockTypeC(torch.nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, input_channels, 3, padding=5, dilation=5),
            torch.nn.BatchNorm2d(input_channels),
            torch.nn.ReLU(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, input_channels, 3, padding=1),
            torch.nn.BatchNorm2d(input_channels),
            torch.nn.ReLU(),
        )
        self.conv3 = torch.nn.Conv2d(input_channels, output_channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.conv1(value)
        value = self.conv2(value)
        return self.conv3(value)


class ConvBNReLU(torch.nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        self.channel_pad = output_channels - input_channels
        self.stride = stride
        padding = 0 if stride == 2 else (kernel_size - 1) // 2
        super().__init__(
            torch.nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU6(inplace=True),
        )
        self.max_pool = torch.nn.MaxPool2d(kernel_size=stride, stride=stride)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        value = input
        if self.stride == 2:
            value = torch.nn.functional.pad(value, (0, 1, 0, 1), "constant", 0)
        for module in self:
            if not isinstance(module, torch.nn.MaxPool2d):
                value = module(value)
        return value


class InvertedResidual(torch.nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int, ratio: int) -> None:
        super().__init__()
        self.stride = stride
        hidden_channels = int(round(input_channels * ratio))
        self.use_res_connect = stride == 1 and input_channels == output_channels
        layers: list[torch.nn.Module] = []
        if ratio != 1:
            layers.append(ConvBNReLU(input_channels, hidden_channels, kernel_size=1))
        layers.extend(
            (
                ConvBNReLU(
                    hidden_channels,
                    hidden_channels,
                    stride=stride,
                    groups=hidden_channels,
                ),
                torch.nn.Conv2d(hidden_channels, output_channels, 1, bias=False),
                torch.nn.BatchNorm2d(output_channels),
            )
        )
        self.conv = torch.nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.conv(value)
        return value + output if self.use_res_connect else output


class MobileNetV2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        input_channels = 32
        features: list[torch.nn.Module] = [ConvBNReLU(4, input_channels, stride=2)]
        for ratio, channels, repetitions, first_stride in (
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
        ):
            for index in range(repetitions):
                stride = first_stride if index == 0 else 1
                features.append(InvertedResidual(input_channels, channels, stride, ratio))
                input_channels = channels
        self.last_channel = 1280
        self.features = torch.nn.Sequential(*features)
        self.fpn_selected = [1, 3, 6, 10, 13]

    def forward(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        selected: list[torch.Tensor] = []
        for index, module in enumerate(self.features):
            value = module(value)
            if index in self.fpn_selected:
                selected.append(value)
        if len(selected) != 5:
            raise RuntimeError("M-LSD backbone did not produce five feature maps")
        return selected[0], selected[1], selected[2], selected[3], selected[4]


class MobileV2MLSDLarge(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = MobileNetV2()
        self.block15 = BlockTypeA(64, 96, 64, 64, upscale=False)
        self.block16 = BlockTypeB(128, 64)
        self.block17 = BlockTypeA(32, 64, 64, 64)
        self.block18 = BlockTypeB(128, 64)
        self.block19 = BlockTypeA(24, 64, 64, 64)
        self.block20 = BlockTypeB(128, 64)
        self.block21 = BlockTypeA(16, 64, 64, 64)
        self.block22 = BlockTypeB(128, 64)
        self.block23 = BlockTypeC(64, 16)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        c1, c2, c3, c4, c5 = self.backbone(value)
        value = self.block15(c4, c5)
        value = self.block16(value)
        value = self.block17(c3, value)
        value = self.block18(value)
        value = self.block19(c2, value)
        value = self.block20(value)
        value = self.block21(c1, value)
        value = self.block22(value)
        return self.block23(value)[:, 7:, :, :]


def _load_mlsd() -> MobileV2MLSDLarge:
    with declared_asset("mlsd-model").open() as stream:
        state = torch.load(stream, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("mlsd-model must contain a state dictionary")
    model = MobileV2MLSDLarge()
    model.load_state_dict(cast("dict[str, torch.Tensor]", state), strict=True)
    return model


def _pred_lines(
    image: np.ndarray,
    model: torch.nn.Module,
    score_threshold: float,
    distance_threshold: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    resized = np.concatenate((image, np.ones((height, width, 1), dtype=image.dtype)), axis=-1)
    normalized = (resized.transpose((2, 0, 1))[None].astype(np.float32) / 127.5) - 1.0
    tensor = torch.from_numpy(normalized).float().to(next(model.parameters()).device)
    with torch.no_grad():
        output = model(tensor)
    _, _, map_height, map_width = output.shape
    displacement = output[:, 1:5][0]
    heat = torch.sigmoid(output[:, 0])
    maxima = torch.nn.functional.max_pool2d(heat, 3, stride=1, padding=1)
    heat = (heat * (maxima == heat)).reshape(-1)
    scores, indices = torch.topk(heat, 200, largest=True)
    points = (
        torch.stack(
            (torch.floor_divide(indices, map_width), torch.fmod(indices, map_width)), dim=-1
        )
        .detach()
        .cpu()
        .numpy()
    )
    score_values = scores.detach().cpu().numpy()
    vectors = displacement.detach().cpu().numpy().transpose((1, 2, 0))
    distances = np.sqrt(np.sum((vectors[:, :, :2] - vectors[:, :, 2:]) ** 2, axis=-1))
    segments: list[list[float]] = []
    for center, score in zip(points, score_values, strict=True):
        y, x = center
        if score > score_threshold and distances[y, x] > distance_threshold:
            start_x, start_y, end_x, end_y = vectors[y, x]
            segments.append([x + start_x, y + start_y, x + end_x, y + end_y])
    if not segments:
        return np.empty((0, 4), dtype=np.float32)
    lines = 2.0 * np.asarray(segments)
    lines[:, (0, 2)] *= width / float(map_width * 2)
    lines[:, (1, 3)] *= height / float(map_height * 2)
    return lines


def _mlsd_frame(
    model: torch.nn.Module,
    frame: np.ndarray,
    resolution: int,
    score_threshold: float,
    distance_threshold: float,
) -> np.ndarray:
    resized, target_height, target_width = _resize_with_pad(frame, resolution, cv2.INTER_AREA)
    output = np.zeros_like(resized)
    for line in _pred_lines(resized, model, score_threshold, distance_threshold):
        x_start, y_start, x_end, y_end = (int(value) for value in line)
        cv2.line(output, (x_start, y_start), (x_end, y_end), (255, 255, 255), 1)
    return _hwc3(output[:target_height, :target_width, 0])


def execute_mlsd(
    image: object,
    *,
    score_threshold: float,
    distance_threshold: float,
    resolution: int,
) -> np.ndarray:
    if not 0.01 <= score_threshold <= 2.0:
        raise ValueError("score_threshold must be between 0.01 and 2")
    if not 0.01 <= distance_threshold <= 20.0:
        raise ValueError("distance_threshold must be between 0.01 and 20")
    frames = _frames(image)

    def run(model: torch.nn.Module) -> list[np.ndarray]:
        return [
            _mlsd_frame(model, frame, resolution, score_threshold, distance_threshold)
            for frame in frames
        ]

    primary_device = MODEL_CACHE.device()
    try:
        with MODEL_CACHE.use("M-LSD", _load_mlsd, device=primary_device) as model:
            outputs = run(model)
    except torch.OutOfMemoryError:
        if primary_device.type != "cuda":
            raise
        MODEL_CACHE.discard("M-LSD")
        torch.cuda.empty_cache()
        fallback_key = "M-LSD CPU fallback"
        try:
            with MODEL_CACHE.use(fallback_key, _load_mlsd, device=torch.device("cpu")) as model:
                outputs = run(model)
        finally:
            MODEL_CACHE.discard(fallback_key)
    shape = outputs[0].shape
    if any(output.shape != shape for output in outputs):
        raise ValueError("M-LSD preprocessing produced inconsistent batch dimensions")
    result = np.asarray(outputs, dtype=np.float32)
    result /= 255.0
    return np.ascontiguousarray(result)


__all__ = ["MobileV2MLSDLarge", "execute_mlsd"]
