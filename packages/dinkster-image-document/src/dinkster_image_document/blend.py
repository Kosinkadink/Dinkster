"""NumPy compositor blend modes matching ComfyUI's LayerStyle formulas."""

from __future__ import annotations

import numpy as np

from .format import BLEND_MODES

_EPSILON = 1e-7


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    return np.where(
        image <= 0.0031308,
        12.92 * image,
        1.055 * np.power(np.maximum(image, 0.0), 1.0 / 2.4) - 0.055,
    )


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    return np.where(
        image <= 0.04045,
        image / 12.92,
        np.power((np.maximum(image, 0.0) + 0.055) / 1.055, 2.4),
    )


def _normal(_bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    return top


def _dissolve(bottom: np.ndarray, top: np.ndarray, seed: int) -> np.ndarray:
    del bottom, seed
    return top


def _multiply(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    return bottom * top


def _screen(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - bottom) * (1.0 - top)


def _overlay(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    return np.where(
        bottom <= 0.5,
        2.0 * bottom * top,
        1.0 - 2.0 * (1.0 - bottom) * (1.0 - top),
    )


def _soft_light(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    curve = np.where(
        bottom <= 0.25,
        ((16.0 * bottom - 12.0) * bottom + 4.0) * bottom,
        np.sqrt(np.maximum(bottom, 0.0)),
    )
    return np.where(
        top <= 0.5,
        bottom - (1.0 - 2.0 * top) * bottom * (1.0 - bottom),
        bottom + (2.0 * top - 1.0) * (curve - bottom),
    )


def _hard_light(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    return _overlay(top, bottom)


def _color_dodge(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    return np.where(top >= 1.0, 1.0, np.minimum(1.0, bottom / np.maximum(1.0 - top, _EPSILON)))


def _linear_dodge(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    return np.minimum(1.0, bottom + top)


def _color_burn(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    return np.where(
        top <= 0.0,
        0.0,
        1.0 - np.minimum(1.0, (1.0 - bottom) / np.maximum(top, _EPSILON)),
    )


def _linear_burn(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, bottom + top - 1.0)


def _vivid_light(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    return np.where(
        top <= 0.5,
        _color_burn(bottom, 2.0 * top),
        _color_dodge(bottom, 2.0 * top - 1.0),
    )


def _linear_light(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    return np.clip(bottom + 2.0 * top - 1.0, 0.0, 1.0)


def _difference(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    return np.abs(bottom - top)


def _exclusion(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    return bottom + top - 2 * bottom * top


def _pin_light(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    return np.where(top > 0.5, np.maximum(bottom, 2 * top - 1), np.minimum(bottom, 2 * top))


def _hard_mix(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    return (bottom + top >= 1).astype(bottom.dtype)


def _subtract(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    return bottom - top


def _divide(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    return bottom / np.maximum(top, 1e-6)


def _grain_extract(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    return bottom - top + 0.5


def _grain_merge(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    return bottom + top - 0.5


def _darken(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    return np.minimum(bottom, top)


def _lighten(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    return np.maximum(bottom, top)


def _luminosity(color: np.ndarray) -> np.ndarray:
    return np.sum(color * np.array((0.3, 0.59, 0.11), dtype=np.float32), axis=-1, keepdims=True)


def _saturation(color: np.ndarray) -> np.ndarray:
    return np.max(color, axis=-1, keepdims=True) - np.min(color, axis=-1, keepdims=True)


def _clip_color(color: np.ndarray) -> np.ndarray:
    luminosity = _luminosity(color)
    minimum = np.min(color, axis=-1, keepdims=True)
    maximum = np.max(color, axis=-1, keepdims=True)
    below = minimum < 0.0
    above = maximum > 1.0
    low_denominator = np.where(np.abs(luminosity - minimum) < _EPSILON, 1.0, luminosity - minimum)
    high_denominator = np.where(np.abs(maximum - luminosity) < _EPSILON, 1.0, maximum - luminosity)
    clipped = np.where(
        below,
        luminosity + (color - luminosity) * luminosity / low_denominator,
        color,
    )
    return np.where(
        above,
        luminosity + (clipped - luminosity) * (1.0 - luminosity) / high_denominator,
        clipped,
    )


def _set_luminosity(color: np.ndarray, luminosity: np.ndarray) -> np.ndarray:
    return _clip_color(color + luminosity - _luminosity(color))


def _set_saturation(color: np.ndarray, saturation: np.ndarray) -> np.ndarray:
    indexes = np.argsort(color, axis=-1)
    sorted_color = np.take_along_axis(color, indexes, axis=-1)
    minimum = sorted_color[..., 0:1]
    middle = sorted_color[..., 1:2]
    maximum = sorted_color[..., 2:3]
    denominator = maximum - minimum
    safe = np.where(denominator > _EPSILON, denominator, 1.0)
    sorted_result = np.concatenate(
        (
            np.zeros_like(minimum),
            np.where(denominator > _EPSILON, (middle - minimum) * saturation / safe, 0.0),
            np.where(denominator > _EPSILON, saturation, 0.0),
        ),
        axis=-1,
    )
    inverse = np.argsort(indexes, axis=-1)
    return np.take_along_axis(sorted_result, inverse, axis=-1)


def _hue(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    return _set_luminosity(_set_saturation(top, _saturation(bottom)), _luminosity(bottom))


def _saturation_mode(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    return _set_luminosity(_set_saturation(bottom, _saturation(top)), _luminosity(bottom))


def _luminosity_mode(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    return _set_luminosity(bottom, _luminosity(top))


def _color(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    return _set_luminosity(top, _luminosity(bottom))


def dissolve_alpha(alpha: np.ndarray, seed: int) -> np.ndarray:
    """Turn fractional source opacity into deterministic pixel coverage."""

    rng = np.random.default_rng(seed)
    noise = rng.random(alpha.shape[:2], dtype=np.float32)[..., None]
    return (noise < alpha).astype(np.float32)


def blend(
    bottom: np.ndarray,
    top: np.ndarray,
    mode: str,
    *,
    seed: int,
) -> np.ndarray:
    """Blend straight RGB layers already expressed in one color space."""

    if mode not in BLEND_MODES:
        raise ValueError(f"unknown compositor blend mode: {mode}")
    if mode == "dissolve":
        result = _dissolve(bottom, top, seed)
    else:
        operation = {
            "normal": _normal,
            "multiply": _multiply,
            "screen": _screen,
            "overlay": _overlay,
            "soft_light": _soft_light,
            "hard_light": _hard_light,
            "color_dodge": _color_dodge,
            "linear_dodge": _linear_dodge,
            "color_burn": _color_burn,
            "linear_burn": _linear_burn,
            "vivid_light": _vivid_light,
            "linear_light": _linear_light,
            "difference": _difference,
            "exclusion": _exclusion,
            "pin_light": _pin_light,
            "hard_mix": _hard_mix,
            "subtract": _subtract,
            "divide": _divide,
            "grain_extract": _grain_extract,
            "grain_merge": _grain_merge,
            "darken": _darken,
            "lighten": _lighten,
            "hue": _hue,
            "saturation": _saturation_mode,
            "luminosity": _luminosity_mode,
            "color": _color,
        }[mode]
        result = operation(bottom, top)
    return np.clip(result, 0.0, 1.0)


__all__ = ["blend", "dissolve_alpha", "linear_to_srgb", "srgb_to_linear"]
