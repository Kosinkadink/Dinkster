"""Native ComfyUI image filter nodes."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    IMAGE_TYPE,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_image,
)
from PIL import Image

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_INT = TypeExpr.concrete(CORE_INT)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_UPSCALE_METHODS = ("nearest-exact", "bilinear", "area", "bicubic", "lanczos")
_BLEND_MODES = ("normal", "multiply", "screen", "overlay", "soft_light", "difference")
_DITHER_MODES = ("none", "floyd-steinberg", "bayer-2", "bayer-4", "bayer-8", "bayer-16")


def _number(
    name: str,
    value_type: TypeExpr,
    default: float | int,
    minimum: float | int,
    maximum: float | int,
    step: float | int | None = None,
    *,
    advanced: bool = False,
) -> InputSpec:
    return InputSpec(
        name,
        value_type,
        default=default,
        advanced=advanced,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


def _combo(name: str, options: tuple[str, ...]) -> InputSpec:
    return InputSpec(
        name,
        _COMBO,
        widget=ComboWidget(tuple(ComboOption(option) for option in options)),
    )


def _schema(
    node_type: str, display_name: str, category: str, inputs: tuple[InputSpec, ...]
) -> NodeSchema:
    return NodeSchema(
        node_type=node_type,
        display_name=display_name,
        category=category,
        inputs=inputs,
        outputs=(OutputSpec("IMAGE", _IMAGE),),
    )


def _image_alpha_fix(destination: Any, source: Any) -> tuple[Any, Any]:
    torch = importlib.import_module("torch")
    if destination.shape[-1] < source.shape[-1]:
        source = source[..., : destination.shape[-1]]
    elif destination.shape[-1] > source.shape[-1]:
        source = torch.nn.functional.pad(source, (0, 1))
        source[..., -1] = 1.0
    return destination, source


def _gaussian_kernel(kernel_size: int, sigma: float, *, device: Any, dtype: Any) -> Any:
    torch = importlib.import_module("torch")
    x, y = torch.meshgrid(
        torch.linspace(-1, 1, kernel_size, device=device),
        torch.linspace(-1, 1, kernel_size, device=device),
        indexing="ij",
    )
    distance = torch.sqrt(x * x + y * y)
    kernel = torch.exp(-(distance * distance) / (2.0 * sigma * sigma))
    return (kernel / kernel.sum()).to(dtype)


class ImageBlend(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ImageBlend",
            "Blend Images",
            "image/filters",
            (
                InputSpec("image1", _IMAGE),
                InputSpec("image2", _IMAGE),
                _number("blend_factor", _FLOAT, 0.5, 0.0, 1.0, 0.01),
                _combo("blend_mode", _BLEND_MODES),
            ),
        )

    @classmethod
    def execute(
        cls, *, image1: Any, image2: Any, blend_factor: float, blend_mode: str
    ) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        utils = importlib.import_module("dinkster_inference.utils")
        image1 = materialize_image(image1)
        image2 = materialize_image(image2)
        image1, image2 = _image_alpha_fix(image1, image2)
        image2 = image2.to(image1.device)
        if image1.shape != image2.shape:
            image2 = utils.common_upscale(
                image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "bicubic", "center"
            ).movedim(1, -1)
        blended = cls.blend_mode(image1, image2, blend_mode)
        output = torch.clamp(image1 * (1 - blend_factor) + blended * blend_factor, 0, 1)
        if image1.shape[-1] == 4:
            output[..., 3] = image1[..., 3]
        return cls.outputs(IMAGE=output)

    @classmethod
    def blend_mode(cls, image1: Any, image2: Any, mode: str) -> Any:
        torch = importlib.import_module("torch")
        if mode == "normal":
            return image2
        if mode == "multiply":
            return image1 * image2
        if mode == "screen":
            return 1 - (1 - image1) * (1 - image2)
        if mode == "overlay":
            return torch.where(
                image1 <= 0.5, 2 * image1 * image2, 1 - 2 * (1 - image1) * (1 - image2)
            )
        if mode == "soft_light":
            return torch.where(
                image2 <= 0.5,
                image1 - (1 - 2 * image2) * image1 * (1 - image1),
                image1 + (2 * image2 - 1) * (cls.g(image1) - image1),
            )
        if mode == "difference":
            return image1 - image2
        raise ValueError(f"Unsupported blend mode: {mode}")

    @classmethod
    def g(cls, image: Any) -> Any:
        torch = importlib.import_module("torch")
        return torch.where(
            image <= 0.25, ((16 * image - 12) * image + 4) * image, torch.sqrt(image)
        )


class ImageBlur(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ImageBlur",
            "Blur Image",
            "image/filters",
            (
                InputSpec("image", _IMAGE),
                _number("blur_radius", _INT, 1, 1, 31, 1),
                _number("sigma", _FLOAT, 1.0, 0.1, 10.0, 0.1),
            ),
        )

    @classmethod
    def execute(cls, *, image: Any, blur_radius: int, sigma: float) -> Mapping[str, object]:
        if blur_radius == 0:
            return cls.outputs(IMAGE=image)
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        image = materialize_image(image).to(management.get_torch_device())
        channels = image.shape[-1]
        kernel_size = blur_radius * 2 + 1
        kernel = _gaussian_kernel(kernel_size, sigma, device=image.device, dtype=image.dtype)
        kernel = kernel.repeat(channels, 1, 1).unsqueeze(1)
        tensor_image = image.movedim(-1, 1)
        padded = torch.nn.functional.pad(
            tensor_image, (blur_radius, blur_radius, blur_radius, blur_radius), "reflect"
        )
        output = torch.nn.functional.conv2d(
            padded, kernel, padding=kernel_size // 2, groups=channels
        )[:, :, blur_radius:-blur_radius, blur_radius:-blur_radius].movedim(1, -1)
        return cls.outputs(IMAGE=output.to(management.intermediate_device()))


class ImageQuantize(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ImageQuantize",
            "Quantize Image",
            "image/filters",
            (
                InputSpec("image", _IMAGE),
                _number("colors", _INT, 256, 1, 256, 1),
                _combo("dither", _DITHER_MODES),
            ),
        )

    @staticmethod
    def bayer(image: Image.Image, palette: Image.Image, order: int) -> Image.Image:
        torch = importlib.import_module("torch")

        def normalized_bayer_matrix(size: int) -> np.ndarray:
            if size == 0:
                return np.zeros((1, 1), "float32")
            quotient = 4**size
            matrix = quotient * normalized_bayer_matrix(size - 1)
            return np.bmat(((matrix - 1.5, matrix + 0.5), (matrix + 1.5, matrix - 0.5))) / quotient

        colors = len(palette.getpalette() or []) // 3
        spread = 2 * 256 / colors
        matrix = torch.from_numpy(spread * normalized_bayer_matrix(int(math.log2(order))) + 0.5)
        result = torch.from_numpy(np.array(image).astype(np.float32))
        width_tiles = math.ceil(result.shape[0] / matrix.shape[0])
        height_tiles = math.ceil(result.shape[1] / matrix.shape[1])
        tiled_matrix = matrix.tile(width_tiles, height_tiles).unsqueeze(-1)
        result.add_(tiled_matrix[: result.shape[0], : result.shape[1]]).clamp_(0, 255)
        result = result.to(dtype=torch.uint8)
        adjusted = Image.fromarray(result.cpu().numpy())
        return adjusted.quantize(palette=palette, dither=Image.Dither.NONE)

    @classmethod
    def execute(cls, *, image: Any, colors: int, dither: str) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        image = materialize_image(image)
        rgb = image[..., :3]
        result = torch.zeros_like(rgb)
        for batch_index in range(rgb.shape[0]):
            source = Image.fromarray((rgb[batch_index] * 255).to(torch.uint8).numpy(), mode="RGB")
            palette = source.quantize(colors=colors)
            if dither == "none":
                quantized = source.quantize(palette=palette, dither=Image.Dither.NONE)
            elif dither == "floyd-steinberg":
                quantized = source.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
            elif dither.startswith("bayer"):
                quantized = cls.bayer(source, palette, int(dither.split("-")[-1]))
            else:
                raise ValueError(f"Unsupported dither mode: {dither}")
            result[batch_index] = torch.tensor(np.array(quantized.convert("RGB"))).float() / 255
        if image.shape[-1] == 4:
            result = torch.cat((result, image[..., 3:]), dim=-1)
        return cls.outputs(IMAGE=result)


class ImageSharpen(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ImageSharpen",
            "Sharpen Image",
            "image/filters",
            (
                InputSpec("image", _IMAGE),
                _number("sharpen_radius", _INT, 1, 1, 31, 1, advanced=True),
                _number("sigma", _FLOAT, 1.0, 0.1, 10.0, 0.01, advanced=True),
                _number("alpha", _FLOAT, 1.0, 0.0, 5.0, 0.01, advanced=True),
            ),
        )

    @classmethod
    def execute(
        cls, *, image: Any, sharpen_radius: int, sigma: float, alpha: float
    ) -> Mapping[str, object]:
        if sharpen_radius == 0:
            return cls.outputs(IMAGE=image)
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        image = materialize_image(image).to(management.get_torch_device())
        channels = image.shape[-1]
        kernel_size = sharpen_radius * 2 + 1
        kernel = _gaussian_kernel(kernel_size, sigma, device=image.device, dtype=image.dtype) * -(
            alpha * 10
        )
        center = kernel_size // 2
        kernel[center, center] = kernel[center, center] - kernel.sum() + 1.0
        kernel = kernel.repeat(channels, 1, 1).unsqueeze(1)
        tensor_image = image.movedim(-1, 1)
        padded = torch.nn.functional.pad(
            tensor_image,
            (sharpen_radius, sharpen_radius, sharpen_radius, sharpen_radius),
            "reflect",
        )
        output = torch.nn.functional.conv2d(padded, kernel, padding=center, groups=channels)[
            :, :, sharpen_radius:-sharpen_radius, sharpen_radius:-sharpen_radius
        ].movedim(1, -1)
        return cls.outputs(IMAGE=torch.clamp(output, 0, 1).to(management.intermediate_device()))


class ImageScaleToTotalPixels(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ImageScaleToTotalPixels",
            "Scale Image to Total Pixels",
            "image/upscaling",
            (
                InputSpec("image", _IMAGE),
                _combo("upscale_method", _UPSCALE_METHODS),
                _number("megapixels", _FLOAT, 1.0, 0.01, 16.0, 0.01),
                _number("resolution_steps", _INT, 1, 1, 256, advanced=True),
            ),
        )

    @classmethod
    def execute(
        cls, *, image: Any, upscale_method: str, megapixels: float, resolution_steps: int
    ) -> Mapping[str, object]:
        utils = importlib.import_module("dinkster_inference.utils")
        samples = materialize_image(image).movedim(-1, 1)
        scale = math.sqrt(megapixels * 1024 * 1024 / (samples.shape[3] * samples.shape[2]))
        width = round(samples.shape[3] * scale / resolution_steps) * resolution_steps
        height = round(samples.shape[2] * scale / resolution_steps) * resolution_steps
        output = utils.common_upscale(samples, int(width), int(height), upscale_method, "disabled")
        return cls.outputs(IMAGE=output.movedim(1, -1))


IMAGE_FILTER_NODES = (ImageBlend, ImageBlur, ImageQuantize, ImageSharpen, ImageScaleToTotalPixels)
