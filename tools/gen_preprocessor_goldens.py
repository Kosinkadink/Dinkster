"""Generate clean-room parity vectors for comfyui_controlnet_aux 59b1fc4.

The formulas below reproduce the pinned detector sources without importing or
executing the external package. Run this tool twice and compare the printed
sha256 before committing a refreshed fixture.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

BASELINE = "59b1fc411ede8623b2997855b8018f0b3b6cf49f"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "preprocessors_controlnet_aux_59b1fc4.json"


def _source() -> np.ndarray:
    rows, columns, channels = np.indices((65, 83, 3))
    return ((columns * 17 + rows * 29 + channels * 67 + (columns * rows) % 251) % 256).astype(
        np.uint8
    )


def _resize_with_pad(
    image: np.ndarray, resolution: int, upscale_method: int
) -> tuple[np.ndarray, int, int]:
    height, width = image.shape[:2]
    if resolution == 0:
        return image, height, width
    scale = float(resolution) / float(min(height, width))
    target_height = int(np.round(float(height) * scale))
    target_width = int(np.round(float(width) * scale))
    resized = cv2.resize(
        image,
        (target_width, target_height),
        interpolation=upscale_method if scale > 1.0 else cv2.INTER_AREA,
    )
    pad_height = math.ceil(target_height / 64) * 64 - target_height
    pad_width = math.ceil(target_width / 64) * 64 - target_width
    return (
        np.ascontiguousarray(
            np.pad(resized, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge")
        ),
        target_height,
        target_width,
    )


def _crop(image: np.ndarray, height: int, width: int) -> np.ndarray:
    return np.ascontiguousarray(image[:height, :width, ...])


def _rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = image[:, :, None]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    return np.ascontiguousarray(image)


def _canny(image: np.ndarray) -> np.ndarray:
    resized, height, width = _resize_with_pad(image, 64, cv2.INTER_CUBIC)
    return _rgb(_crop(cv2.Canny(resized, 100, 200), height, width))


def _pyramid_canny(image: np.ndarray) -> np.ndarray:
    resized, target_height, target_width = _resize_with_pad(image, 64, cv2.INTER_CUBIC)
    height, width = resized.shape[:2]
    accumulated: np.ndarray | None = None
    for scale in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        small = cv2.resize(
            resized,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
        edge = np.stack(
            [
                cv2.Canny(small[:, :, channel], 64, 128).astype(np.float32) / 255.0
                for channel in range(3)
            ],
            axis=2,
        )
        if accumulated is None:
            accumulated = edge
        else:
            accumulated = cv2.resize(
                accumulated,
                (edge.shape[1], edge.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            accumulated = accumulated * 0.75 + edge * 0.25
    assert accumulated is not None
    combined = np.sum(accumulated, axis=2, dtype=np.float32)
    minimum = np.percentile(combined, 1)
    maximum = np.percentile(combined, 99)
    combined -= minimum
    combined /= maximum - minimum
    result = np.clip(combined * 255.0, 0.0, 255.0).astype(np.uint8)
    return _rgb(_crop(result, target_height, target_width))


def _lineart(image: np.ndarray) -> np.ndarray:
    resized, height, width = _resize_with_pad(image, 64, cv2.INTER_CUBIC)
    source = resized.astype(np.float32)
    blurred = cv2.GaussianBlur(source, (0, 0), 2.5)
    intensity = np.min(blurred - source, axis=2).clip(0, 255)
    intensity /= max(16, np.median(intensity[intensity > 7]))
    result = (intensity * 127).clip(0, 255).astype(np.uint8)
    return _rgb(_crop(result, height, width))


def _scribble(image: np.ndarray) -> np.ndarray:
    resized, height, width = _resize_with_pad(image, 64, cv2.INTER_AREA)
    result = np.zeros_like(resized, dtype=np.uint8)
    result[np.min(resized, axis=2) < 127] = 255
    return _crop(255 - result, height, width)


def _xdog(image: np.ndarray) -> np.ndarray:
    resized, height, width = _resize_with_pad(image, 64, cv2.INTER_CUBIC)
    narrow = cv2.GaussianBlur(resized.astype(np.float32), (0, 0), 0.5)
    wide = cv2.GaussianBlur(resized.astype(np.float32), (0, 0), 5.0)
    dog = (255 - np.min(wide - narrow, axis=2)).clip(0, 255).astype(np.uint8)
    result = np.zeros_like(resized, dtype=np.uint8)
    result[2 * (255 - dog) > 32] = 255
    return _crop(result, height, width)


def _binary(image: np.ndarray, threshold: int) -> np.ndarray:
    resized, height, width = _resize_with_pad(image, 64, cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    if threshold in (0, 255):
        _, result = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
    else:
        _, result = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return _rgb(_crop(255 - result, height, width))


def _palette(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    target_height = 64
    target_width = int(round(width / height * 64))
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    reduced = cv2.resize(
        resized,
        (target_width // 64, target_height // 64),
        interpolation=cv2.INTER_CUBIC,
    )
    return cv2.resize(
        reduced,
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    )


def _recolor(image: np.ndarray, mode: str, gamma: float) -> np.ndarray:
    resized, height, width = _resize_with_pad(image, 64, cv2.INTER_CUBIC)
    conversion = cv2.COLOR_BGR2LAB if mode == "luminance" else cv2.COLOR_BGR2HSV
    channel = cv2.cvtColor(resized, conversion)[:, :, 0 if mode == "luminance" else 2]
    result = channel.astype(np.float32) / 255.0
    result = result**gamma
    result = (result * 255.0).clip(0, 255).astype(np.uint8)
    return _rgb(_crop(result, height, width))


def _noise_disk(height: int, width: int, generator: np.random.Generator) -> np.ndarray:
    noise = generator.uniform(
        low=0,
        high=1,
        size=((height // 256) + 2, (width // 256) + 2, 1),
    )
    noise = cv2.resize(
        noise,
        (width + 512, height + 512),
        interpolation=cv2.INTER_CUBIC,
    )
    noise = noise[256 : 256 + height, 256 : 256 + width]
    noise -= np.min(noise)
    noise /= np.max(noise)
    return noise[:, :, None] if noise.ndim == 2 else noise


def _shuffle(image: np.ndarray) -> np.ndarray:
    resized, target_height, target_width = _resize_with_pad(image, 64, cv2.INTER_CUBIC)
    height, width = resized.shape[:2]
    generator = np.random.default_rng(7)
    x = _noise_disk(height, width, generator) * float(width - 1)
    y = _noise_disk(height, width, generator) * float(height - 1)
    flow = np.concatenate((x, y), axis=2).astype(np.float32)
    result = cv2.remap(resized, flow[:, :, 0], flow[:, :, 1], cv2.INTER_LINEAR)
    return _crop(result, target_height, target_width)


def _tile_pyramid(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    height = int(np.round(height / 64.0)) * 64
    width = int(np.round(width / 64.0)) * 64
    result = cv2.resize(image, (width // 8, height // 8), interpolation=cv2.INTER_AREA)
    for _ in range(3):
        result = cv2.pyrUp(result)
    return result


def _tile_blur(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (3, 3), sigmaX=1.5)


def _guided(image: np.ndarray) -> np.ndarray:
    source = image.astype(np.float32) / 255.0
    height, width = source.shape[:2]
    guide = cv2.resize(source, (width // 2, height // 2), interpolation=cv2.INTER_NEAREST)
    window = 2 * int(5 / 2) + 1
    red, green, blue = (guide[:, :, index] for index in range(3))
    red_mean = cv2.blur(red, (window, window))
    green_mean = cv2.blur(green, (window, window))
    blue_mean = cv2.blur(blue, (window, window))
    rr = cv2.blur(red**2, (window, window)) - red_mean**2 + 0.01
    rg = cv2.blur(red * green, (window, window)) - red_mean * green_mean
    rb = cv2.blur(red * blue, (window, window)) - red_mean * blue_mean
    gg = cv2.blur(green**2, (window, window)) - green_mean**2 + 0.01
    gb = cv2.blur(green * blue, (window, window)) - green_mean * blue_mean
    bb = cv2.blur(blue**2, (window, window)) - blue_mean**2 + 0.01
    rr_inv = gg * bb - gb * gb
    rg_inv = gb * rb - rg * bb
    rb_inv = rg * gb - gg * rb
    gg_inv = rr * bb - rb * rb
    gb_inv = rb * rg - rr * gb
    bb_inv = rr * gg - rg * rg
    covariance = rr_inv * rr + rg_inv * rg + rb_inv * rb
    inverses = [value / covariance for value in (rr_inv, rg_inv, rb_inv, gg_inv, gb_inv, bb_inv)]
    rr_inv, rg_inv, rb_inv, gg_inv, gb_inv, bb_inv = inverses
    output = np.array(source)
    for channel in range(3):
        plane = guide[:, :, channel]
        plane_mean = cv2.blur(plane, (window, window))
        red_cov = cv2.blur(red * plane, (window, window)) - red_mean * plane_mean
        green_cov = cv2.blur(green * plane, (window, window)) - green_mean * plane_mean
        blue_cov = cv2.blur(blue * plane, (window, window)) - blue_mean * plane_mean
        a_red = rr_inv * red_cov + rg_inv * green_cov + rb_inv * blue_cov
        a_green = rg_inv * red_cov + gg_inv * green_cov + gb_inv * blue_cov
        a_blue = rb_inv * red_cov + gb_inv * green_cov + bb_inv * blue_cov
        offset = plane_mean - a_red * red_mean - a_green * green_mean - a_blue * blue_mean
        coefficient_values = (
            cv2.blur(a_red, (window, window)),
            cv2.blur(a_green, (window, window)),
            cv2.blur(a_blue, (window, window)),
            cv2.blur(offset, (window, window)),
        )
        coefficients = [
            cv2.resize(value, (width, height), interpolation=cv2.INTER_LINEAR)
            for value in coefficient_values
        ]
        output[:, :, channel] = (
            coefficients[0] * source[:, :, 0]
            + coefficients[1] * source[:, :, 1]
            + coefficients[2] * source[:, :, 2]
            + coefficients[3]
        )
    return np.asarray(np.clip(np.uint8(255.0 * output), 0, 255), dtype=np.uint8)


def _tile_guided(image: np.ndarray) -> np.ndarray:
    bgr = image[:, :, ::-1]
    processed = _guided(_tile_blur(bgr))
    height, width = processed.shape[:2]
    reduced = cv2.resize(processed, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_CUBIC)[:, :, ::-1]


def _tile_simple(image: np.ndarray) -> np.ndarray:
    bgr = image[:, :, ::-1]
    height, width = bgr.shape[:2]
    reduced = cv2.resize(bgr, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(reduced, (width, height), interpolation=cv2.INTER_LANCZOS4)
    return _tile_blur(restored)[:, :, ::-1]


def _record(array: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array, dtype=np.uint8)
    return {
        "shape": list(contiguous.shape),
        "uint8Base64": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def build_goldens() -> dict[str, object]:
    source = _source()
    cases = {
        "binary_fixed": _binary(source, 100),
        "binary_otsu": _binary(source, 0),
        "canny": _canny(source),
        "color_palette": _palette(source),
        "content_shuffle": _shuffle(source),
        "lineart": _lineart(source),
        "pyramid_canny": _pyramid_canny(source),
        "recolor_intensity": _recolor(source, "intensity", 0.8),
        "recolor_luminance": _recolor(source, "luminance", 1.2),
        "scribble": _scribble(source),
        "scribble_xdog": _xdog(source),
        "tile_guided": _tile_guided(source),
        "tile_pyramid": _tile_pyramid(source),
        "tile_simple": _tile_simple(source),
    }
    return {
        "baseline": BASELINE,
        "opencv": cv2.__version__,
        "source": _record(source),
        "cases": {name: _record(value) for name, value in cases.items()},
    }


def main() -> None:
    payload = json.dumps(build_goldens(), indent=2, sort_keys=True) + "\n"
    OUT.write_text(payload, encoding="utf-8")
    print(hashlib.sha256(payload.encode()).hexdigest())


if __name__ == "__main__":
    main()
