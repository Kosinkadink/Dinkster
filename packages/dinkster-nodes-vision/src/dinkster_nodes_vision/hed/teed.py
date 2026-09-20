"""Tiny and Efficient Edge Detector inference."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
import torch
from dinkster_api.v1 import declared_asset

from .cache import MODEL_CACHE
from .model import _frames, _hwc3, _resize_with_pad


@torch.jit.script
def smish(value: torch.Tensor) -> torch.Tensor:
    return value * torch.tanh(torch.log(1 + torch.sigmoid(value)))


class Smish(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return smish(value)


class DoubleFusion(torch.nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.DWconv1 = torch.nn.Conv2d(
            input_channels,
            input_channels * 8,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=input_channels,
        )
        self.PSconv1 = torch.nn.PixelShuffle(1)
        self.DWconv2 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1, groups=24)
        self.AF = Smish()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        attention = self.PSconv1(self.DWconv1(self.AF(value)))
        attention2 = self.PSconv1(self.DWconv2(self.AF(attention)))
        return smish((attention2 + attention).sum(1).unsqueeze(1))


class _DenseLayer(torch.nn.Sequential):
    def __init__(self, input_features: int, output_features: int) -> None:
        super().__init__()
        self.add_module(
            "conv1",
            torch.nn.Conv2d(input_features, output_features, 3, padding=2, bias=True),
        )
        self.add_module("smish1", Smish())
        self.add_module("conv2", torch.nn.Conv2d(output_features, output_features, 3, bias=True))

    def forward(
        self, input: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source, skip = input
        features = super().forward(smish(source))
        return 0.5 * (features + skip), skip


class _DenseBlock(torch.nn.Sequential):
    def __init__(self, input_features: int, output_features: int) -> None:
        super().__init__()
        self.add_module("denselayer1", _DenseLayer(input_features, output_features))


class UpConvBlock(torch.nn.Module):
    def __init__(self, input_features: int, up_scale: int) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        pads = (0, 0, 1, 3, 7)
        for index in range(up_scale):
            kernel_size = 2**up_scale
            output_features = 1 if index == up_scale - 1 else 16
            layers.extend(
                (
                    torch.nn.Conv2d(input_features, output_features, 1),
                    Smish(),
                    torch.nn.ConvTranspose2d(
                        output_features,
                        output_features,
                        kernel_size,
                        stride=2,
                        padding=pads[up_scale],
                    ),
                )
            )
            input_features = output_features
        self.features = torch.nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.features(value)


class SingleConvBlock(torch.nn.Module):
    def __init__(self, input_features: int, output_features: int, stride: int) -> None:
        super().__init__()
        self.use_ac = False
        self.conv = torch.nn.Conv2d(input_features, output_features, 1, stride=stride, bias=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value)


class DoubleConvBlock(torch.nn.Module):
    def __init__(
        self,
        input_features: int,
        middle_features: int,
        output_features: int | None = None,
        *,
        stride: int = 1,
        use_act: bool = True,
    ) -> None:
        super().__init__()
        self.use_act = use_act
        output_features = middle_features if output_features is None else output_features
        self.conv1 = torch.nn.Conv2d(input_features, middle_features, 3, padding=1, stride=stride)
        self.conv2 = torch.nn.Conv2d(middle_features, output_features, 3, padding=1)
        self.smish = Smish()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.smish(self.conv1(value))
        value = self.conv2(value)
        return self.smish(value) if self.use_act else value


class TED(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block_1 = DoubleConvBlock(3, 16, 16, stride=2)
        self.block_2 = DoubleConvBlock(16, 32, use_act=False)
        self.dblock_3 = _DenseBlock(32, 48)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.side_1 = SingleConvBlock(16, 32, 2)
        self.pre_dense_3 = SingleConvBlock(32, 48, 1)
        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(32, 1)
        self.up_block_3 = UpConvBlock(48, 2)
        self.block_cat = DoubleFusion(3)

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        block_1 = self.block_1(value)
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_3, _ = self.dblock_3(
            (block_2_down + self.side_1(block_1), self.pre_dense_3(block_2_down))
        )
        results = [
            self.up_block_1(block_1),
            self.up_block_2(block_2),
            self.up_block_3(block_3),
        ]
        results.append(self.block_cat(torch.cat(results, dim=1)))
        return results


def _load_teed(asset_id: str) -> TED:
    with declared_asset(asset_id).open() as stream:
        state = torch.load(stream, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"{asset_id} must contain a state dictionary")
    model = TED()
    model.load_state_dict(cast("dict[str, torch.Tensor]", state), strict=True)
    return model


def _teed_frame(
    model: torch.nn.Module, frame: np.ndarray, resolution: int, safe_steps: int
) -> np.ndarray:
    resized, target_height, target_width = _resize_with_pad(frame, resolution)
    height, width = resized.shape[:2]
    tensor = (
        torch.from_numpy(resized.copy())
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(next(model.parameters()).device)
    )
    with torch.no_grad():
        outputs = cast("list[torch.Tensor]", model(tensor))
    edges = [
        cv2.resize(
            output.detach().cpu().numpy().astype(np.float32)[0, 0],
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        for output in outputs
    ]
    mean = np.mean(np.stack(edges, axis=2), axis=2).astype(np.float64)
    edge = 1.0 / (1.0 + np.exp(-mean))
    if safe_steps:
        edge = (edge.astype(np.float32) * float(safe_steps + 1)).astype(np.int32).astype(
            np.float32
        ) / float(safe_steps)
    detected = np.clip(edge * 255.0, 0.0, 255.0).astype(np.uint8)
    return _hwc3(detected[:target_height, :target_width])


def execute_teed(
    image: object,
    *,
    safe_steps: int,
    resolution: int,
    asset_id: str = "teed-model",
) -> np.ndarray:
    if not 0 <= safe_steps <= 10:
        raise ValueError("safe_steps must be between 0 and 10")
    cache_key = "MTEED" if asset_id == "mteed-model" else "TEED"
    with MODEL_CACHE.use(cache_key, lambda: _load_teed(asset_id)) as model:
        outputs = [_teed_frame(model, frame, resolution, safe_steps) for frame in _frames(image)]
    shape = outputs[0].shape
    if any(output.shape != shape for output in outputs):
        raise ValueError("TEED preprocessing produced inconsistent batch dimensions")
    result = np.asarray(outputs, dtype=np.float32)
    result /= 255.0
    return np.ascontiguousarray(result)


__all__ = ["TED", "execute_teed"]
