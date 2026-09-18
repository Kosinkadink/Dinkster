from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from dinkster_nodes_image.composition import (
    BLEND_MODES,
    PORTER_DUFF_MODES,
    ImageComposite,
    PorterDuffComposite,
)

GOLDEN = cast(
    "dict[str, object]",
    json.loads(
        (Path(__file__).parent / "goldens" / "image_geometry_e20d433a.json").read_text(
            encoding="utf-8"
        )
    ),
)
GOLDEN_CASES = cast("dict[str, object]", GOLDEN["cases"])


def _golden_array(name: str) -> np.ndarray:
    record = cast("dict[str, object]", GOLDEN_CASES[name])
    return np.asarray(cast("list[float]", record["values"]), dtype=np.float32).reshape(
        cast("list[int]", record["shape"])
    )


def _golden_source() -> np.ndarray:
    record = cast("dict[str, object]", GOLDEN["source"])
    return np.asarray(cast("list[float]", record["values"]), dtype=np.float32).reshape(
        cast("list[int]", record["shape"])
    )


def _solid(value: float, *, batch: int = 1, height: int = 2, width: int = 3) -> np.ndarray:
    return np.full((batch, height, width, 3), value, dtype=np.float32)


@pytest.mark.parametrize("mask_polarity", ["coverage", "transparency"])
def test_core_composite_mappings_match_comfy_goldens(mask_polarity: str) -> None:
    destination = _golden_source()
    source = np.flip(destination[:, :2, :3, :], axis=(1, 2)).copy()
    mask = np.asarray([[[0.0, 0.25, 0.5], [0.75, 1.0, 0.25]]], dtype=np.float32)
    if mask_polarity == "transparency":
        mask = 1.0 - mask
    direct = ImageComposite.execute(
        destination=destination,
        source=source,
        x=1,
        y=1,
        mask=mask,
        mask_polarity=mask_polarity,
        clamp_output=False,
        batch_policy="destination_repeat",
    )["image"]
    resized = ImageComposite.execute(
        destination=destination,
        source=source,
        mask=mask,
        source_resize="stretch",
        mask_polarity=mask_polarity,
        interpolation="bilinear",
        clamp_output=False,
        batch_policy="destination_repeat",
    )["image"]
    np.testing.assert_array_equal(direct, _golden_array("ImageCompositeMasked"))
    # Float32 bilinear mask and image interpolation drifts by 2.99e-8; 5e-8 leaves 2.01e-8.
    np.testing.assert_allclose(
        np.asarray(resized),
        _golden_array("ImageCompositeMasked:resize-source"),
        rtol=0,
        atol=5e-8,
    )

    for blend_mode in BLEND_MODES:
        source_mode = "difference" if blend_mode == "signed_difference" else blend_mode
        result = ImageComposite.execute(
            destination=destination,
            source=source,
            blend_mode=blend_mode,
            factor=0.4,
            source_resize="fill",
            interpolation="bicubic",
            clamp_output=True,
        )["image"]
        # Float32 cubic and blend accumulation drift by at most 1.20e-7; 2e-7 leaves 8e-8.
        np.testing.assert_allclose(
            np.asarray(result),
            _golden_array(f"ImageBlend:{source_mode}"),
            rtol=0,
            atol=2e-7,
        )


@pytest.mark.parametrize("mask_polarity", ["coverage", "transparency"])
def test_core_porter_duff_mappings_match_comfy_goldens(mask_polarity: str) -> None:
    destination = _golden_source()
    source = np.flip(destination[:, :2, :3, :], axis=(1, 2)).copy()
    source_alpha = np.asarray([[[0.0, 0.25, 0.5], [0.75, 1.0, 0.25]]], dtype=np.float32)
    destination_alpha = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(1, 3, 4)
    result = PorterDuffComposite.execute(
        source=source,
        source_alpha_mask=1.0 - source_alpha if mask_polarity == "coverage" else source_alpha,
        destination=destination,
        destination_alpha_mask=1.0 - destination_alpha
        if mask_polarity == "coverage"
        else destination_alpha,
        mode="SRC_OVER",
        mask_polarity=mask_polarity,
        batch_policy="truncate_to_shortest",
    )
    # Cubic and premultiplied-alpha accumulation drift by 2.39e-7; 3e-7 leaves 6.1e-8.
    np.testing.assert_allclose(
        np.asarray(result["image"]),
        _golden_array("PorterDuffImageComposite:image"),
        rtol=0,
        atol=3e-7,
    )
    np.testing.assert_allclose(
        1.0 - np.asarray(result["alpha_mask"])
        if mask_polarity == "coverage"
        else np.asarray(result["alpha_mask"]),
        _golden_array("PorterDuffImageComposite:mask"),
        rtol=0,
        atol=3e-7,
    )


def test_core_porter_duff_center_crops_mismatched_aspects() -> None:
    result = PorterDuffComposite.execute(
        source=_golden_array("PorterDuffImageComposite:aspect-source"),
        source_alpha_mask=_golden_array("PorterDuffImageComposite:aspect-source-mask"),
        destination=_golden_array("PorterDuffImageComposite:aspect-destination"),
        destination_alpha_mask=_golden_array("PorterDuffImageComposite:aspect-destination-mask"),
        mode="SRC_OVER",
        mask_polarity="transparency",
        batch_policy="truncate_to_shortest",
    )
    # Center-crop, cubic, and alpha accumulation drift by 1.79e-7; 3e-7 leaves 1.21e-7.
    np.testing.assert_allclose(
        np.asarray(result["image"]),
        _golden_array("PorterDuffImageComposite:aspect-image"),
        rtol=0,
        atol=3e-7,
    )
    np.testing.assert_allclose(
        np.asarray(result["alpha_mask"]),
        _golden_array("PorterDuffImageComposite:aspect-mask"),
        rtol=0,
        atol=3e-7,
    )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("normal", 0.75),
        ("multiply", 0.1875),
        ("screen", 0.8125),
        ("overlay", 0.375),
        ("soft_light", 0.375),
        ("signed_difference", 0.0),
    ],
)
def test_each_blend_mode_matches_display_rgb_formula(mode: str, expected: float) -> None:
    result = ImageComposite.execute(
        destination=_solid(0.25),
        source=_solid(0.75),
        blend_mode=mode,
    )
    np.testing.assert_allclose(np.asarray(result["image"]), expected, rtol=0, atol=1e-7)


def test_blend_factor_interpolates_and_overlay_uses_destination_branch() -> None:
    result = ImageComposite.execute(
        destination=_solid(0.75),
        source=_solid(0.25),
        blend_mode="overlay",
        factor=0.5,
    )
    np.testing.assert_allclose(np.asarray(result["image"]), 0.6875, rtol=0, atol=1e-7)


def test_composite_output_clamping_is_explicit() -> None:
    destination = _solid(1.5, height=1, width=1)
    source = _solid(2.0, height=1, width=1)
    clamped = ImageComposite.execute(destination=destination, source=source)["image"]
    unclamped = ImageComposite.execute(
        destination=destination,
        source=source,
        clamp_output=False,
    )["image"]
    np.testing.assert_array_equal(clamped, 1.0)
    np.testing.assert_array_equal(unclamped, 2.0)


def test_composite_offsets_clip_both_sides_and_out_of_bounds_is_a_noop() -> None:
    destination = np.zeros((1, 3, 4, 3), dtype=np.float32)
    source = np.ones((1, 2, 3, 3), dtype=np.float32)

    positive = np.asarray(
        ImageComposite.execute(destination=destination, source=source, x=3, y=2)["image"]
    )
    expected_positive = np.zeros_like(destination)
    expected_positive[:, 2, 3] = 1.0
    np.testing.assert_array_equal(positive, expected_positive)

    negative = np.asarray(
        ImageComposite.execute(destination=destination, source=source, x=-2, y=-1)["image"]
    )
    expected_negative = np.zeros_like(destination)
    expected_negative[:, 0, 0] = 1.0
    np.testing.assert_array_equal(negative, expected_negative)

    outside = ImageComposite.execute(destination=destination, source=source, x=5, y=-4)
    np.testing.assert_array_equal(outside["image"], destination)

    extended_destination = np.full((1, 2, 2, 3), -0.5, dtype=np.float32)
    partial = np.asarray(
        ImageComposite.execute(
            destination=extended_destination,
            source=np.ones((1, 1, 1, 3), dtype=np.float32),
            x=1,
            y=1,
        )["image"]
    )
    np.testing.assert_array_equal(partial[:, :1, :, :], -0.5)
    np.testing.assert_array_equal(partial[:, 1:, :1, :], -0.5)
    np.testing.assert_array_equal(partial[:, 1, 1, :], 1.0)


def test_composite_can_stretch_or_center_fill_the_source() -> None:
    destination = np.zeros((1, 2, 4, 1), dtype=np.float32)
    source = np.arange(6, dtype=np.float32).reshape(1, 2, 3, 1) / 5
    stretched = np.asarray(
        ImageComposite.execute(
            destination=destination,
            source=source,
            source_resize="stretch",
            interpolation="nearest-exact",
        )["image"]
    )
    filled = np.asarray(
        ImageComposite.execute(
            destination=destination,
            source=source,
            source_resize="fill",
            interpolation="nearest-exact",
        )["image"]
    )
    assert stretched.shape == destination.shape
    assert filled.shape == destination.shape
    np.testing.assert_array_equal(stretched[:, :, (0, 2), :], source[:, :, (0, 1), :])
    np.testing.assert_array_equal(filled[:, 0, :, :], source[:, 0, (0, 1, 1, 2), :])


def test_composite_mask_polarity_and_factor_are_explicit() -> None:
    destination = _solid(0.0, width=2)
    source = _solid(1.0, width=2)
    mask = np.asarray([[[0.25, 0.75], [0.25, 0.75]]], dtype=np.float32)
    opacity = ImageComposite.execute(
        destination=destination,
        source=source,
        mask=mask,
        mask_polarity="coverage",
        factor=0.5,
    )["image"]
    transparency = ImageComposite.execute(
        destination=destination,
        source=source,
        mask=mask,
        mask_polarity="transparency",
        factor=0.5,
    )["image"]
    np.testing.assert_allclose(np.asarray(opacity)[0, 0, :, 0], [0.125, 0.375])
    np.testing.assert_allclose(np.asarray(transparency)[0, 0, :, 0], [0.375, 0.125])


def test_composite_batch_policies_are_not_silently_interchangeable() -> None:
    destination = _solid(0.0, batch=2)
    source = _solid(1.0, batch=1)
    broadcast = ImageComposite.execute(destination=destination, source=source)["image"]
    assert np.asarray(broadcast).shape == (2, 2, 3, 3)

    with pytest.raises(ValueError, match="equal batches"):
        ImageComposite.execute(
            destination=destination,
            source=source,
            batch_policy="strict",
        )

    source_three = np.stack((_solid(0.1)[0], _solid(0.2)[0], _solid(0.3)[0]))
    repeated = np.asarray(
        ImageComposite.execute(
            destination=destination,
            source=source_three,
            batch_policy="destination_repeat",
        )["image"]
    )
    np.testing.assert_allclose(repeated[:, 0, 0, 0], [0.1, 0.2])

    with pytest.raises(ValueError, match="each batch"):
        ImageComposite.execute(destination=destination, source=source_three)


def test_composite_reconciles_rgb_rgba_and_rejects_lossy_grayscale_destination() -> None:
    rgb = _solid(0.2)
    rgba_destination = np.zeros((1, 2, 3, 4), dtype=np.float32)
    result = np.asarray(ImageComposite.execute(destination=rgba_destination, source=rgb)["image"])
    np.testing.assert_array_equal(result[..., 3], 1.0)

    rgba_source = np.concatenate((rgb, np.full((1, 2, 3, 1), 0.7, np.float32)), axis=3)
    result = ImageComposite.execute(destination=rgb, source=rgba_source)["image"]
    assert np.asarray(result).shape == rgb.shape

    with pytest.raises(ValueError, match="cannot be conformed"):
        ImageComposite.execute(
            destination=np.zeros((1, 2, 3, 1), dtype=np.float32),
            source=rgb,
        )


def _porter_duff_reference(
    source: float,
    source_opacity: float,
    destination: float,
    destination_opacity: float,
    mode: str,
) -> tuple[float, float]:
    source_color = source * source_opacity
    destination_color = destination * destination_opacity
    if mode == "ADD":
        alpha = min(source_opacity + destination_opacity, 1.0)
        color = min(source_color + destination_color, 1.0)
    elif mode == "CLEAR":
        alpha, color = 0.0, 0.0
    elif mode == "DARKEN":
        alpha = source_opacity + destination_opacity - source_opacity * destination_opacity
        color = (
            (1 - destination_opacity) * source_color
            + (1 - source_opacity) * destination_color
            + min(source_color, destination_color)
        )
    elif mode == "DST":
        alpha, color = destination_opacity, destination_color
    elif mode == "DST_ATOP":
        alpha = source_opacity
        color = source_opacity * destination_color + (1 - destination_opacity) * source_color
    elif mode == "DST_IN":
        alpha = source_opacity * destination_opacity
        color = destination_color * source_opacity
    elif mode == "DST_OUT":
        alpha = (1 - source_opacity) * destination_opacity
        color = (1 - source_opacity) * destination_color
    elif mode == "DST_OVER":
        alpha = destination_opacity + (1 - destination_opacity) * source_opacity
        color = destination_color + (1 - destination_opacity) * source_color
    elif mode == "LIGHTEN":
        alpha = source_opacity + destination_opacity - source_opacity * destination_opacity
        color = (
            (1 - destination_opacity) * source_color
            + (1 - source_opacity) * destination_color
            + max(source_color, destination_color)
        )
    elif mode == "MULTIPLY":
        alpha = source_opacity * destination_opacity
        color = source_color * destination_color
    elif mode == "OVERLAY":
        alpha = source_opacity + destination_opacity - source_opacity * destination_opacity
        color = (
            2 * source_color * destination_color
            if 2 * destination_color < destination_opacity
            else source_opacity * destination_opacity
            - 2 * (destination_opacity - source_color) * (source_opacity - destination_color)
        )
    elif mode == "SCREEN":
        alpha = source_opacity + destination_opacity - source_opacity * destination_opacity
        color = source_color + destination_color - source_color * destination_color
    elif mode == "SRC":
        alpha, color = source_opacity, source_color
    elif mode == "SRC_ATOP":
        alpha = destination_opacity
        color = destination_opacity * source_color + (1 - source_opacity) * destination_color
    elif mode == "SRC_IN":
        alpha = source_opacity * destination_opacity
        color = source_color * destination_opacity
    elif mode == "SRC_OUT":
        alpha = (1 - destination_opacity) * source_opacity
        color = (1 - destination_opacity) * source_color
    elif mode == "SRC_OVER":
        alpha = source_opacity + (1 - source_opacity) * destination_opacity
        color = source_color + (1 - source_opacity) * destination_color
    elif mode == "XOR":
        alpha = (1 - destination_opacity) * source_opacity + (
            1 - source_opacity
        ) * destination_opacity
        color = (1 - destination_opacity) * source_color + (1 - source_opacity) * destination_color
    else:
        raise AssertionError(mode)
    return (0.0 if alpha <= 1e-5 else min(max(color / alpha, 0.0), 1.0), alpha)


@pytest.mark.parametrize("mode", PORTER_DUFF_MODES)
def test_each_porter_duff_mode_matches_premultiplied_reference(mode: str) -> None:
    source, source_opacity = 0.8, 0.5
    destination, destination_opacity = 0.2, 0.25
    expected_image, expected_opacity = _porter_duff_reference(
        source, source_opacity, destination, destination_opacity, mode
    )
    result = PorterDuffComposite.execute(
        source=_solid(source, height=1, width=1),
        source_alpha_mask=np.full((1, 1, 1), source_opacity, np.float32),
        destination=_solid(destination, height=1, width=1),
        destination_alpha_mask=np.full((1, 1, 1), destination_opacity, np.float32),
        mode=mode,
        mask_polarity="coverage",
    )
    np.testing.assert_allclose(np.asarray(result["image"]), expected_image, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(result["alpha_mask"]), expected_opacity, rtol=0, atol=1e-6
    )


def test_porter_duff_transparency_polarity_and_batch_truncation() -> None:
    source = _solid(1.0, batch=3, height=1, width=1)
    destination = _solid(0.0, batch=2, height=1, width=1)
    source_transparency = np.zeros((1, 1, 1), dtype=np.float32)
    destination_transparency = np.ones((2, 1, 1), dtype=np.float32)
    result = PorterDuffComposite.execute(
        source=source,
        source_alpha_mask=source_transparency,
        destination=destination,
        destination_alpha_mask=destination_transparency,
        mode="SRC_OVER",
        batch_policy="truncate_to_shortest",
    )
    assert np.asarray(result["image"]).shape[0] == 1
    np.testing.assert_array_equal(result["image"], 1.0)
    np.testing.assert_array_equal(result["alpha_mask"], 0.0)


def test_composition_rejects_invalid_inputs_and_exposes_all_modes() -> None:
    assert set(BLEND_MODES) == {
        "normal",
        "multiply",
        "screen",
        "overlay",
        "soft_light",
        "signed_difference",
    }
    assert len(PORTER_DUFF_MODES) == 18
    with pytest.raises(ValueError, match="factor"):
        ImageComposite.execute(destination=_solid(0), source=_solid(1), factor=1.1)
    with pytest.raises(ValueError, match="blend mode"):
        ImageComposite.execute(destination=_solid(0), source=_solid(1), blend_mode="bad")
    with pytest.raises(ValueError, match="source resize mode"):
        ImageComposite.execute(destination=_solid(0), source=_solid(1), source_resize="bad")
    with pytest.raises(ValueError, match="channels must match"):
        PorterDuffComposite.execute(
            source=np.zeros((1, 1, 1, 1), np.float32),
            source_alpha_mask=np.zeros((1, 1, 1), np.float32),
            destination=_solid(0, height=1, width=1),
            destination_alpha_mask=np.zeros((1, 1, 1), np.float32),
        )
