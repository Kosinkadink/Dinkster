"""Learned realistic, anime, and manga line-art preprocessors."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import cast

import cv2
import numpy as np
import torch
from dinkster_api.v1 import declared_asset

from .cache import MODEL_CACHE
from .model import _frames, _hwc3, _resize_with_pad


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv_block = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(channels, channels, 3),
            torch.nn.InstanceNorm2d(channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(channels, channels, 3),
            torch.nn.InstanceNorm2d(channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.conv_block(value)


class RealisticGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model0 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(3, 64, 7),
            torch.nn.InstanceNorm2d(64),
            torch.nn.ReLU(inplace=True),
        )
        self.model1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
            torch.nn.InstanceNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 256, 3, stride=2, padding=1),
            torch.nn.InstanceNorm2d(256),
            torch.nn.ReLU(inplace=True),
        )
        self.model2 = torch.nn.Sequential(*(ResidualBlock(256) for _ in range(3)))
        self.model3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            torch.nn.InstanceNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            torch.nn.InstanceNorm2d(64),
            torch.nn.ReLU(inplace=True),
        )
        self.model4 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(64, 1, 7),
            torch.nn.Sigmoid(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.model0(value)
        value = self.model1(value)
        value = self.model2(value)
        value = self.model3(value)
        return self.model4(value)


class AnimeSkipBlock(torch.nn.Module):
    def __init__(
        self,
        outer_channels: int,
        inner_channels: int,
        *,
        input_channels: int | None = None,
        submodule: torch.nn.Module | None = None,
        outermost: bool = False,
        innermost: bool = False,
    ) -> None:
        super().__init__()
        self.outermost = outermost
        input_channels = outer_channels if input_channels is None else input_channels
        downconv = torch.nn.Conv2d(
            input_channels, inner_channels, kernel_size=4, stride=2, padding=1, bias=True
        )
        downrelu = torch.nn.LeakyReLU(0.2, True)
        downnorm = torch.nn.InstanceNorm2d(inner_channels)
        uprelu = torch.nn.ReLU(True)
        upnorm = torch.nn.InstanceNorm2d(outer_channels)
        if outermost:
            assert submodule is not None
            upconv = torch.nn.ConvTranspose2d(
                inner_channels * 2, outer_channels, kernel_size=4, stride=2, padding=1
            )
            modules = [downconv, submodule, uprelu, upconv, torch.nn.Tanh()]
        elif innermost:
            upconv = torch.nn.ConvTranspose2d(
                inner_channels, outer_channels, kernel_size=4, stride=2, padding=1, bias=True
            )
            modules = [downrelu, downconv, uprelu, upconv, upnorm]
        else:
            assert submodule is not None
            upconv = torch.nn.ConvTranspose2d(
                inner_channels * 2, outer_channels, kernel_size=4, stride=2, padding=1, bias=True
            )
            modules = [downrelu, downconv, downnorm, submodule, uprelu, upconv, upnorm]
        self.model = torch.nn.Sequential(*modules)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.model(value)
        return output if self.outermost else torch.cat((value, output), 1)


class AnimeGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model: torch.nn.Module = AnimeSkipBlock(512, 512, innermost=True)
        for _ in range(3):
            model = AnimeSkipBlock(512, 512, submodule=model)
        model = AnimeSkipBlock(256, 512, submodule=model)
        model = AnimeSkipBlock(128, 256, submodule=model)
        model = AnimeSkipBlock(64, 128, submodule=model)
        self.model = AnimeSkipBlock(1, 64, input_channels=3, submodule=model, outermost=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value)


class _MangaConv(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: int = 1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.BatchNorm2d(input_channels, eps=1e-3),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            ),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value)


class _MangaUpConv(torch.nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.BatchNorm2d(input_channels, eps=1e-3),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(input_channels, output_channels, 3, padding=1),
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value)


class _MangaShortcut(torch.nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.model = (
            torch.nn.Sequential(torch.nn.Conv2d(input_channels, output_channels, 1, stride=stride))
            if input_channels != output_channels or stride != 1
            else None
        )

    def forward(self, source: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return (self.model(source) if self.model is not None else source) + residual


class _MangaUpShortcut(torch.nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.model = (
            torch.nn.Sequential(
                torch.nn.Conv2d(input_channels, output_channels, 1),
                torch.nn.Upsample(scale_factor=2, mode="nearest"),
            )
            if input_channels != output_channels
            else None
        )

    def forward(self, source: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return (self.model(source) if self.model is not None else source) + residual


class _MangaBlock(torch.nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _MangaConv(input_channels, output_channels, stride)
        self.residual = _MangaConv(output_channels, output_channels)
        self.shortcut = _MangaShortcut(input_channels, output_channels, stride)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        intermediate = self.conv1(value)
        return self.shortcut(value, self.residual(intermediate))


class _MangaUpBlock(torch.nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv1 = _MangaUpConv(input_channels, output_channels)
        self.residual = _MangaConv(output_channels, output_channels)
        self.shortcut = _MangaUpShortcut(input_channels, output_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.shortcut(value, self.residual(self.conv1(value)))


class _MangaGroup(torch.nn.Module):
    def __init__(self, layers: list[torch.nn.Module]) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value)


def _manga_down(
    input_channels: int,
    output_channels: int,
    repetitions: int,
    *,
    first: bool = False,
) -> _MangaGroup:
    layers: list[torch.nn.Module] = []
    for index in range(repetitions):
        source = input_channels if index == 0 else output_channels
        stride = 2 if index == repetitions - 1 and not first else 1
        layers.append(_MangaBlock(source, output_channels, stride))
    return _MangaGroup(layers)


def _manga_up(input_channels: int, output_channels: int, repetitions: int) -> _MangaGroup:
    return _MangaGroup(
        [
            _MangaUpBlock(input_channels, output_channels),
            *(_MangaBlock(output_channels, output_channels) for _ in range(1, repetitions)),
        ]
    )


class MangaGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block0 = _manga_down(1, 24, 2, first=True)
        self.block1 = _manga_down(24, 48, 3)
        self.block2 = _manga_down(48, 96, 5)
        self.block3 = _manga_down(96, 192, 7)
        self.block4 = _manga_down(192, 384, 12)
        self.block5 = _manga_up(384, 192, 7)
        self.res1 = _MangaShortcut(192, 192)
        self.block6 = _manga_up(192, 96, 5)
        self.res2 = _MangaShortcut(96, 96)
        self.block7 = _manga_up(96, 48, 3)
        self.res3 = _MangaShortcut(48, 48)
        self.block8 = _manga_up(48, 24, 2)
        self.res4 = _MangaShortcut(24, 24)
        self.block9 = _manga_down(24, 16, 2, first=True)
        self.conv15 = _MangaConv(16, 1, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        x0 = self.block0(value)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(self.res1(x3, x5))
        x7 = self.block7(self.res2(x2, x6))
        x8 = self.block8(self.res3(x1, x7))
        return self.conv15(self.block9(self.res4(x0, x8)))


def _load_state(asset_id: str) -> dict[str, torch.Tensor]:
    with declared_asset(asset_id).open() as stream:
        state = torch.load(stream, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"{asset_id} must contain a state dictionary")
    return cast("dict[str, torch.Tensor]", state)


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    for key in list(state):
        if key.startswith("module."):
            state[key.removeprefix("module.")] = state.pop(key)
    return state


def _load_realistic(asset_id: str) -> RealisticGenerator:
    model = RealisticGenerator()
    model.load_state_dict(_load_state(asset_id), strict=True)
    return model


def _load_anime() -> AnimeGenerator:
    state = _load_state("lineart-anime-model")
    model = AnimeGenerator()
    model.load_state_dict(_strip_module_prefix(state), strict=True)
    return model


def _load_manga() -> MangaGenerator:
    state = _load_state("lineart-manga-model")
    model = MangaGenerator()
    model.load_state_dict(_strip_module_prefix(state), strict=True)
    return model


def _realistic_frame(model: torch.nn.Module, frame: np.ndarray, resolution: int) -> np.ndarray:
    resized, target_height, target_width = _resize_with_pad(frame, resolution)
    tensor = (
        torch.from_numpy(resized)
        .float()
        .to(next(model.parameters()).device)
        .div(255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    with torch.no_grad():
        line = model(tensor)[0, 0].detach().cpu().numpy()
    detected = np.clip(line * 255.0, 0.0, 255.0).astype(np.uint8)
    return _hwc3(255 - detected[:target_height, :target_width])


def _anime_frame(model: torch.nn.Module, frame: np.ndarray, resolution: int) -> np.ndarray:
    resized, target_height, target_width = _resize_with_pad(frame, resolution)
    height, width = resized.shape[:2]
    padded_height = 256 * int(np.ceil(float(height) / 256.0))
    padded_width = 256 * int(np.ceil(float(width) / 256.0))
    resized_model = cv2.resize(
        resized, (padded_width, padded_height), interpolation=cv2.INTER_CUBIC
    )
    tensor = (
        torch.from_numpy(resized_model)
        .float()
        .to(next(model.parameters()).device)
        .div(127.5)
        .sub(1.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    with torch.no_grad():
        line = model(tensor)[0, 0].mul(127.5).add(127.5).detach().cpu().numpy()
    detected = cv2.resize(
        _hwc3(np.clip(line, 0.0, 255.0).astype(np.uint8)),
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
    return np.ascontiguousarray(255 - detected[:target_height, :target_width])


def _manga_frame(model: torch.nn.Module, frame: np.ndarray, resolution: int) -> np.ndarray:
    rounded_resolution = 256 * int(np.ceil(float(resolution) / 256.0))
    resized, target_height, target_width = _resize_with_pad(frame, rounded_resolution)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    tensor = (
        torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0).to(next(model.parameters()).device)
    )
    with torch.no_grad():
        line = model(tensor)[0, 0].detach().cpu().numpy()
    detected = _hwc3(np.clip(line, 0.0, 255.0).astype(np.uint8))
    return np.ascontiguousarray(255 - detected[:target_height, :target_width])


def _execute(
    image: object,
    resolution: int,
    cache_key: str,
    factory: Callable[[], torch.nn.Module],
    transform: Callable[[torch.nn.Module, np.ndarray, int], np.ndarray],
) -> np.ndarray:
    with MODEL_CACHE.use(cache_key, factory) as model:
        outputs = [transform(model, frame, resolution) for frame in _frames(image)]
    shape = outputs[0].shape
    if any(output.shape != shape for output in outputs):
        raise ValueError("line-art preprocessing produced inconsistent batch dimensions")
    result = np.asarray(outputs, dtype=np.float32)
    result /= 255.0
    return np.ascontiguousarray(result)


def execute_realistic(image: object, *, coarse: bool, resolution: int) -> np.ndarray:
    if type(coarse) is not bool:
        raise TypeError("coarse must be a boolean")
    asset_id = "lineart-realistic-coarse-model" if coarse else "lineart-realistic-model"
    cache_key = "Realistic Lineart (coarse)" if coarse else "Realistic Lineart"
    return _execute(
        image,
        resolution,
        cache_key,
        functools.partial(_load_realistic, asset_id),
        _realistic_frame,
    )


def execute_anime(image: object, *, resolution: int) -> np.ndarray:
    return _execute(image, resolution, "Anime Lineart", _load_anime, _anime_frame)


def execute_manga(image: object, *, resolution: int) -> np.ndarray:
    return _execute(image, resolution, "Manga Lineart", _load_manga, _manga_frame)


__all__ = [
    "AnimeGenerator",
    "MangaGenerator",
    "RealisticGenerator",
    "execute_anime",
    "execute_manga",
    "execute_realistic",
]
