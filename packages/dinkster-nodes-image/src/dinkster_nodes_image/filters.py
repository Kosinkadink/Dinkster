"""Deterministic bounded CPU image filters."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from dinkster_api.v1 import (
    DynamicComboOption,
    DynamicComboSpec,
    InputSpec,
    MirrorSpec,
    MirrorTolerance,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
)
from PIL import Image

from .geometry import FLOAT, IMAGE, INT
from .migration import with_v1_migration
from .support import combo_input as _combo
from .support import image_array as _image_array
from .support import materialized_inputs

FILTER_OPERATIONS = (
    "gaussian_blur",
    "sharpen",
    "quantize",
    "noise",
    "erode",
    "dilate",
    "open",
    "close",
    "gradient",
    "bottom_hat",
    "top_hat",
)
DITHER_MODES = ("none", "floyd-steinberg", "bayer-2", "bayer-4", "bayer-8", "bayer-16")
MAX_FILTER_RADIUS = 31
MAX_MORPHOLOGY_KERNEL = 999
MAX_SEED = 2**53 - 1

FILTER_MIRROR_PER_CHANNEL_TOLERANCE = 1.0 / 255.0

# Mirrors _gaussian_kernel / _convolve_axis / the gaussian_blur and sharpen
# execute branches. The kernel samples exp(-c^2 / (2 sigma^2)) at
# c = offset / radius (np.linspace(-1, 1, 2 * radius + 1)) and normalizes by
# the weight sum; padding reflects without repeating the border texel and
# multi-bounces when the radius exceeds the axis (np.pad mode="reflect"),
# clamping to the single texel on one-texel axes (mode="edge"). The operation
# branches index the mirrored subset of FILTER_OPERATIONS in declared order;
# uniform names follow the schema binding contract in the dinkster-schema README.
FILTER_MIRROR_SOURCE = """\
#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D u_image;
uniform int operation;
uniform int radius;
uniform float sigma;
uniform float strength;

out vec4 fragColor;

const int MAX_RADIUS = 31;

float weights[2 * MAX_RADIUS + 1];

void loadKernel() {
    float total = 0.0;
    for (int offset = -radius; offset <= radius; offset++) {
        float c = float(offset) / float(radius);
        float weight = exp(-(c * c) / (2.0 * sigma * sigma));
        weights[offset + radius] = weight;
        total += weight;
    }
    for (int i = 0; i <= 2 * radius; i++) {
        weights[i] /= total;
    }
}

int reflectIndex(int position, int size) {
    if (size == 1) {
        return 0;
    }
    int period = 2 * (size - 1);
    int wrapped = abs(position) % period;
    return wrapped < size ? wrapped : period - wrapped;
}

vec4 gaussianBlur(ivec2 center, ivec2 size) {
    if (radius == 0) {
        return texelFetch(u_image, center, 0);
    }
    loadKernel();
    vec4 accumulated = vec4(0.0);
    for (int dy = -radius; dy <= radius; dy++) {
        int y = reflectIndex(center.y + dy, size.y);
        vec4 row = vec4(0.0);
        for (int dx = -radius; dx <= radius; dx++) {
            int x = reflectIndex(center.x + dx, size.x);
            row += weights[dx + radius] * texelFetch(u_image, ivec2(x, y), 0);
        }
        accumulated += weights[dy + radius] * row;
    }
    return accumulated;
}

void main() {
    ivec2 center = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(u_image, 0);
    if (operation == 0) {
        fragColor = gaussianBlur(center, size);
    } else {
        vec4 x = texelFetch(u_image, center, 0);
        fragColor = clamp(x + 10.0 * strength * (x - gaussianBlur(center, size)), 0.0, 1.0);
    }
}
"""


def _require_finite(array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise ValueError("image filters require finite pixel values")


def _gaussian_kernel(radius: int, sigma: float) -> np.ndarray:
    if not 0 <= radius <= MAX_FILTER_RADIUS:
        raise ValueError(f"radius must be between 0 and {MAX_FILTER_RADIUS}, got {radius}")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"sigma must be finite and positive, got {sigma}")
    if radius == 0:
        return np.ones(1, dtype=np.float32)
    coordinates = np.linspace(-1.0, 1.0, 2 * radius + 1, dtype=np.float32)
    kernel = np.exp(-(coordinates * coordinates) / (2.0 * sigma * sigma))
    kernel /= kernel.sum(dtype=np.float64)
    return kernel.astype(np.float32, copy=False)


def _convolve_axis(array: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    if radius == 0:
        return np.array(array, copy=True)
    padding = [(0, 0)] * array.ndim
    padding[axis] = (radius, radius)
    mode = "reflect" if array.shape[axis] > 1 else "edge"
    padded = np.pad(array, padding, mode=mode)
    windows = np.lib.stride_tricks.sliding_window_view(padded, len(kernel), axis=axis)
    return np.einsum("...k,k->...", windows, kernel, dtype=np.float32, optimize=True)


def _gaussian_blur(array: np.ndarray, radius: int, sigma: float) -> np.ndarray:
    kernel = _gaussian_kernel(radius, sigma)
    if radius == 0:
        return np.array(array, copy=True)
    horizontal = _convolve_axis(array, kernel, 2)
    return np.ascontiguousarray(_convolve_axis(horizontal, kernel, 1), dtype=np.float32)


def _normalized_bayer_matrix(level: int) -> np.ndarray:
    matrix = np.zeros((1, 1), dtype=np.float32)
    for _ in range(level):
        q = 4 ** (int(math.log2(matrix.shape[0])) + 1)
        scaled = q * matrix
        matrix = np.block([[scaled - 1.5, scaled + 0.5], [scaled + 1.5, scaled - 0.5]]) / q
    return matrix.astype(np.float32, copy=False)


def _bayer_quantize(image: Image.Image, palette: Image.Image, order: int) -> Image.Image:
    palette_values = palette.getpalette()
    if palette_values is None:
        raise ValueError("quantization palette is unavailable")
    spread = 2.0 * 256.0 / (len(palette_values) // 3)
    matrix = spread * _normalized_bayer_matrix(int(math.log2(order))) + 0.5
    raster = np.asarray(image, dtype=np.float32)
    rows = math.ceil(raster.shape[0] / matrix.shape[0])
    columns = math.ceil(raster.shape[1] / matrix.shape[1])
    tiled = np.tile(matrix, (rows, columns))[: raster.shape[0], : raster.shape[1]]
    if raster.ndim == 3:
        tiled = tiled[..., None]
    dithered = np.clip(raster + tiled, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(dithered).quantize(palette=palette, dither=Image.Dither.NONE)


def _quantize(array: np.ndarray, colors: int, dither: str) -> np.ndarray:
    if not 1 <= colors <= 256:
        raise ValueError(f"colors must be between 1 and 256, got {colors}")
    if dither not in DITHER_MODES:
        raise ValueError(f"unknown dither mode: {dither}")
    output = np.empty_like(array)
    channels = int(array.shape[3])
    for index in range(array.shape[0]):
        raster = np.clip(array[index] * 255.0, 0.0, 255.0).astype(np.uint8)
        if channels == 1:
            image = Image.fromarray(raster[..., 0], mode="L").convert("RGB")
        else:
            image = Image.fromarray(raster[..., :3], mode="RGB")
        palette = image.quantize(colors=colors)
        if dither == "none":
            quantized = image.quantize(palette=palette, dither=Image.Dither.NONE)
        elif dither == "floyd-steinberg":
            quantized = image.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
        else:
            quantized = _bayer_quantize(image, palette, int(dither.rsplit("-", 1)[1]))
        rgb = np.asarray(quantized.convert("RGB"), dtype=np.float32) / 255.0
        if channels == 1:
            output[index, ..., 0] = rgb[..., 0]
        elif channels == 3:
            output[index] = rgb
        else:
            output[index, ..., :3] = rgb
            output[index, ..., 3] = array[index, ..., 3]
    return np.ascontiguousarray(output, dtype=np.float32)


def _extreme_filter(array: np.ndarray, kernel_size: int, *, maximum: bool) -> np.ndarray:
    if not 1 <= kernel_size <= MAX_MORPHOLOGY_KERNEL:
        raise ValueError(
            f"kernel_size must be between 1 and {MAX_MORPHOLOGY_KERNEL}, got {kernel_size}"
        )
    if kernel_size == 1:
        return np.array(array, copy=True)
    leading = kernel_size // 2
    trailing = kernel_size - leading - 1
    neutral = -1e4 if maximum else 1e4
    horizontal = np.pad(
        array,
        ((0, 0), (0, 0), (leading, trailing), (0, 0)),
        mode="constant",
        constant_values=neutral,
    )
    horizontal_windows = np.lib.stride_tricks.sliding_window_view(horizontal, kernel_size, axis=2)
    horizontal_result = (
        np.max(horizontal_windows, axis=-1) if maximum else np.min(horizontal_windows, axis=-1)
    )
    vertical = np.pad(
        horizontal_result,
        ((0, 0), (leading, trailing), (0, 0), (0, 0)),
        mode="constant",
        constant_values=neutral,
    )
    vertical_windows = np.lib.stride_tricks.sliding_window_view(vertical, kernel_size, axis=1)
    output = np.max(vertical_windows, axis=-1) if maximum else np.min(vertical_windows, axis=-1)
    return np.ascontiguousarray(output, dtype=np.float32)


def _morphology(array: np.ndarray, operation: str, kernel_size: int) -> np.ndarray:
    def erode(value: np.ndarray) -> np.ndarray:
        return _extreme_filter(value, kernel_size, maximum=False)

    def dilate(value: np.ndarray) -> np.ndarray:
        return _extreme_filter(value, kernel_size, maximum=True)

    if operation == "erode":
        return erode(array)
    if operation == "dilate":
        return dilate(array)
    if operation == "open":
        return dilate(erode(array))
    if operation == "close":
        return erode(dilate(array))
    if operation == "gradient":
        return dilate(array) - erode(array)
    if operation == "bottom_hat":
        return erode(dilate(array)) - array
    if operation == "top_hat":
        return array - dilate(erode(array))
    raise ValueError(f"unknown morphology operation: {operation}")


class ImageFilter(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        original = (
            InputSpec(
                "radius",
                INT,
                required=False,
                default=1,
                widget=NumberWidget(min=0, max=MAX_FILTER_RADIUS, step=1),
            ),
            InputSpec(
                "sigma",
                FLOAT,
                required=False,
                default=1.0,
                widget=NumberWidget(min=0.1, max=10.0, step=0.1),
            ),
            InputSpec(
                "strength",
                FLOAT,
                required=False,
                default=1.0,
                widget=NumberWidget(min=0.0, max=5.0, step=0.01),
            ),
            InputSpec(
                "colors",
                INT,
                required=False,
                default=256,
                widget=NumberWidget(min=1, max=256, step=1),
            ),
            _combo("dither", DITHER_MODES, "none", advanced=True),
            InputSpec(
                "seed",
                INT,
                required=False,
                default=0,
                widget=NumberWidget(min=0, max=MAX_SEED, step=1),
            ),
            InputSpec(
                "kernel_size",
                INT,
                required=False,
                default=3,
                widget=NumberWidget(min=1, max=MAX_MORPHOLOGY_KERNEL, step=1),
            ),
        )
        by_id = {item.id: item for item in original}
        return with_v1_migration(
            NodeSchema(
                node_type="dinkster.image.filter",
                version=2,
                display_name="Filter Image",
                category="image/filter",
                inputs=(InputSpec("image", IMAGE),),
                combos=(
                    DynamicComboSpec(
                        "operation",
                        tuple(
                            DynamicComboOption(
                                name, tuple(by_id[input_id] for input_id in input_ids)
                            )
                            for name, input_ids in (
                                ("gaussian_blur", ("radius", "sigma")),
                                ("sharpen", ("radius", "sigma", "strength")),
                                ("quantize", ("colors", "dither")),
                                ("noise", ("strength", "seed")),
                                *((name, ("kernel_size",)) for name in FILTER_OPERATIONS[4:]),
                            )
                        ),
                        default="gaussian_blur",
                    ),
                ),
                outputs=(OutputSpec("image", IMAGE, preview=True),),
                search_terms=(
                    "blur image",
                    "sharpen image",
                    "quantize image",
                    "noise image",
                    "morphology",
                    "ImageBlur",
                    "ImageSharpen",
                    "ImageQuantize",
                ),
                # Bounded because GLSL ES 3.00 does not guarantee correctly
                # rounded arithmetic and clients run on heterogeneous GPU float
                # pipelines; one 8-bit preview quantization step is far above
                # the worst-case drift of the convolution, including the 10x
                # strength amplification in sharpen. The parity corpus is
                # tests/fixtures/mirror-parity/image_filter_v1.json. The other
                # operations stay unmirrored: quantize (Pillow median cut) and
                # the morphology family are not expressible in the one-shader
                # single-pass contract, and noise depends on a seeded PCG64
                # stream a shader cannot reproduce.
                mirror=MirrorSpec(
                    kind="glsl",
                    precision="bounded",
                    tolerance=MirrorTolerance(per_channel=FILTER_MIRROR_PER_CHANNEL_TOLERANCE),
                    source=FILTER_MIRROR_SOURCE,
                    applies={"operation": ("gaussian_blur", "sharpen")},
                ),
            )
        )

    @classmethod
    @materialized_inputs
    def execute(
        cls,
        *,
        image: object,
        operation: str = "gaussian_blur",
        radius: int = 1,
        sigma: float = 1.0,
        strength: float = 1.0,
        colors: int = 256,
        dither: str = "none",
        seed: int = 0,
        kernel_size: int = 3,
    ) -> Mapping[str, object]:
        array = _image_array(image)
        _require_finite(array)
        if operation == "gaussian_blur":
            output = _gaussian_blur(array, radius, sigma)
        elif operation == "sharpen":
            if not math.isfinite(strength) or strength < 0.0:
                raise ValueError("sharpen strength must be finite and non-negative")
            blurred = _gaussian_blur(array, radius, sigma)
            output = np.clip(array + 10.0 * strength * (array - blurred), 0.0, 1.0)
        elif operation == "quantize":
            output = _quantize(array, colors, dither)
        elif operation == "noise":
            if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
                raise ValueError("noise strength must be between 0 and 1")
            if not 0 <= seed <= MAX_SEED:
                raise ValueError(f"seed must be between 0 and {MAX_SEED}, got {seed}")
            generator = np.random.Generator(np.random.PCG64(seed))
            noise = generator.standard_normal(array.shape, dtype=np.float32)
            output = np.clip(array + strength * noise, 0.0, 1.0)
        elif operation in FILTER_OPERATIONS[4:]:
            output = _morphology(array, operation, kernel_size)
        else:
            raise ValueError(f"unknown image filter: {operation}")
        return cls.outputs(image=np.ascontiguousarray(output, dtype=np.float32))


FILTER_NODES: tuple[type[Node], ...] = (ImageFilter,)


__all__ = [
    "DITHER_MODES",
    "FILTER_MIRROR_PER_CHANNEL_TOLERANCE",
    "FILTER_MIRROR_SOURCE",
    "FILTER_NODES",
    "FILTER_OPERATIONS",
    "ImageFilter",
]
