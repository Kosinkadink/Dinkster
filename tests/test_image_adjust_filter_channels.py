from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from dinkster_nodes_image.adjust import ADJUST_OPERATIONS, ImageAdjust
from dinkster_nodes_image.channels import ImageAlphaJoin, ImageChannelMerge, ImageChannelSplit
from dinkster_nodes_image.filters import DITHER_MODES, FILTER_OPERATIONS, ImageFilter

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


def _image(*, batch: int = 1, channels: int = 3) -> np.ndarray:
    values = np.linspace(0.0, 1.0, batch * 3 * 4 * channels, dtype=np.float32)
    return values.reshape(batch, 3, 4, channels)


def test_adjust_and_filter_dynamic_combo_schemas_and_materialized_execution() -> None:
    adjust = ImageAdjust.define_schema()
    image_filter = ImageFilter.define_schema()
    assert adjust.version == image_filter.version == 2
    assert tuple(item.id for item in adjust.inputs) == ("image",)
    adjust_options = [
        (option.key, tuple(item.id for item in option.inputs))
        for option in adjust.combos[0].options
    ]
    assert adjust_options == [
        ("invert", ()),
        ("normalize", ("mean", "standard_deviation")),
        ("brightness", ("factor",)),
        ("contrast", ("factor",)),
    ]
    expected_filter = {
        "gaussian_blur": ("radius", "sigma"),
        "sharpen": ("radius", "sigma", "strength"),
        "quantize": ("colors", "dither"),
        "noise": ("strength", "seed"),
        **{name: ("kernel_size",) for name in FILTER_OPERATIONS[4:]},
    }
    assert {
        option.key: tuple(item.id for item in option.inputs)
        for option in image_filter.combos[0].options
    } == expected_filter

    source = _image()
    expected = ImageAdjust.execute(image=source, operation="contrast", factor=2.0)
    materialized = ImageAdjust.execute(
        **{"image": source, "operation": "contrast", "operation.factor": 2.0}
    )
    np.testing.assert_array_equal(materialized["image"], expected["image"])


def test_core_adjust_filter_mappings_match_comfy_goldens() -> None:
    source = _golden_source()
    cases = {
        "ImageInvert": ImageAdjust.execute(image=source, operation="invert")["image"],
        "NormalizeImages": ImageAdjust.execute(
            image=source,
            operation="normalize",
            mean=0.4,
            standard_deviation=0.25,
        )["image"],
        "AdjustBrightness": ImageAdjust.execute(
            image=source,
            operation="brightness",
            factor=1.25,
        )["image"],
        "AdjustContrast": ImageAdjust.execute(
            image=source,
            operation="contrast",
            factor=1.25,
        )["image"],
    }
    for name, result in cases.items():
        np.testing.assert_array_equal(result, _golden_array(name))

    filtered = {
        "ImageBlur": ImageFilter.execute(
            image=source,
            operation="gaussian_blur",
            radius=1,
            sigma=1.0,
        )["image"],
        "ImageSharpen": ImageFilter.execute(
            image=source,
            operation="sharpen",
            radius=1,
            sigma=1.0,
            strength=0.1,
        )["image"],
    }
    for name, result in filtered.items():
        # Float32 convolution accumulation drifts by 1.20e-7; 2e-7 leaves 8e-8.
        np.testing.assert_allclose(np.asarray(result), _golden_array(name), rtol=0, atol=2e-7)

    for dither in DITHER_MODES:
        result = ImageFilter.execute(
            image=source,
            operation="quantize",
            colors=4,
            dither=dither,
        )["image"]
        np.testing.assert_array_equal(result, _golden_array(f"ImageQuantize:{dither}"))


def test_core_filters_match_large_kernel_comfy_goldens() -> None:
    source = _golden_array("ImageFilter:stress-source")
    blurred = ImageFilter.execute(
        image=source,
        operation="gaussian_blur",
        radius=31,
        sigma=0.1,
    )["image"]
    sharpened = ImageFilter.execute(
        image=source,
        operation="sharpen",
        radius=31,
        sigma=0.1,
        strength=5.0,
    )["image"]
    # Mint-host Gaussian kernel drift is 1.7e-6 here; 8e-5 also covers
    # GPU-executed references, whose CPU/GPU kernel math drifts by 5.22e-5.
    np.testing.assert_allclose(
        np.asarray(blurred), _golden_array("ImageBlur:stress"), rtol=0, atol=8e-5
    )
    # The 50x sharpen gain amplifies that drift to 6.1e-6 here and 0.02087
    # on GPU-executed references; 0.022 leaves 0.00113 over the worst case.
    np.testing.assert_allclose(
        np.asarray(sharpened), _golden_array("ImageSharpen:stress"), rtol=0, atol=0.022
    )


@pytest.mark.parametrize(
    "operation",
    ("erode", "dilate", "open", "close", "gradient", "bottom_hat", "top_hat"),
)
def test_core_morphology_mapping_matches_comfy_goldens(operation: str) -> None:
    source = _golden_array("Morphology:source")
    result = ImageFilter.execute(
        image=source,
        operation=operation,
        kernel_size=4,
    )["image"]
    np.testing.assert_array_equal(result, _golden_array(f"Morphology:{operation}"))


@pytest.mark.parametrize("mask_polarity", ["coverage", "transparency"])
def test_core_channel_mappings_match_comfy_goldens(mask_polarity: str) -> None:
    source = _golden_source()
    alpha = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(1, 3, 4)
    rgba = np.concatenate((source, alpha[..., None]), axis=3)
    split = ImageChannelSplit.execute(image=rgba, mask_polarity=mask_polarity)
    np.testing.assert_array_equal(split["image"], _golden_array("SplitImageWithAlpha:image"))
    # NumPy and Torch linspace inputs differ by 5.97e-8; 1e-7 leaves 4e-8.
    np.testing.assert_allclose(
        1.0 - np.asarray(split["alpha_mask"])
        if mask_polarity == "coverage"
        else np.asarray(split["alpha_mask"]),
        _golden_array("SplitImageWithAlpha:mask"),
        rtol=0,
        atol=1e-7,
    )
    joined = ImageAlphaJoin.execute(
        image=source,
        alpha_mask=1.0 - alpha if mask_polarity == "coverage" else alpha,
        mask_polarity=mask_polarity,
        batch_policy="cyclic_repeat",
    )["image"]
    # Float32 alpha subtraction differs by 5.97e-8; 1e-7 leaves 4e-8.
    np.testing.assert_allclose(
        np.asarray(joined), _golden_array("JoinImageWithAlpha"), rtol=0, atol=1e-7
    )

    ycbcr = ImageChannelSplit.execute(image=source, color_space="ycbcr")
    ycbcr_repeated = ImageChannelSplit.execute(
        image=source,
        color_space="ycbcr",
        channel_layout="rgb_repeated",
    )
    for output, name in (
        ("channel_1", "Y"),
        ("channel_2", "U"),
        ("channel_3", "V"),
    ):
        np.testing.assert_array_equal(
            ycbcr_repeated[output], _golden_array(f"ImageRGBToYUV:{name}")
        )
    restored = ImageChannelMerge.execute(
        channel_1=ycbcr["channel_1"],
        channel_2=ycbcr["channel_2"],
        channel_3=ycbcr["channel_3"],
        color_space="ycbcr",
        batch_policy="strict",
    )["image"]
    # Rounded YCbCr constants drift by 7.92e-6; 1e-5 leaves 2.08e-6.
    np.testing.assert_allclose(
        np.asarray(restored), _golden_array("ImageYUVToRGB"), rtol=0, atol=1e-5
    )


def test_adjust_operations_match_core_formulas_and_preserve_layout() -> None:
    image = np.asarray([[[[0.0, 0.25, 0.75], [1.0, 0.5, 0.1]]]], dtype=np.float32)
    invert = ImageAdjust.execute(image=image, operation="invert")["image"]
    normalized = ImageAdjust.execute(
        image=image,
        operation="normalize",
        mean=0.5,
        standard_deviation=0.25,
    )["image"]
    brightness = ImageAdjust.execute(image=image, operation="brightness", factor=2.0)["image"]
    contrast = ImageAdjust.execute(image=image, operation="contrast", factor=2.0)["image"]
    np.testing.assert_array_equal(invert, 1.0 - image)
    np.testing.assert_array_equal(normalized, (image - 0.5) / 0.25)
    np.testing.assert_array_equal(brightness, np.clip(image * 2.0, 0.0, 1.0))
    np.testing.assert_array_equal(contrast, np.clip((image - 0.5) * 2.0 + 0.5, 0.0, 1.0))
    assert all(
        np.asarray(value).shape == image.shape
        for value in (invert, normalized, brightness, contrast)
    )
    assert all(
        np.asarray(value).dtype == np.float32
        for value in (invert, normalized, brightness, contrast)
    )
    assert set(ADJUST_OPERATIONS) == {"invert", "normalize", "brightness", "contrast"}


def test_adjust_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="standard_deviation"):
        ImageAdjust.execute(
            image=_image(),
            operation="normalize",
            standard_deviation=0.0,
        )
    with pytest.raises(ValueError, match="non-negative"):
        ImageAdjust.execute(image=_image(), operation="contrast", factor=-1.0)
    with pytest.raises(ValueError, match="unknown"):
        ImageAdjust.execute(image=_image(), operation="gamma")


def test_gaussian_blur_and_sharpen_handle_tiny_images_radius_zero_and_batches() -> None:
    impulse = np.zeros((2, 3, 3, 1), dtype=np.float32)
    impulse[:, 1, 1, 0] = 1.0
    unchanged = ImageFilter.execute(
        image=impulse,
        operation="gaussian_blur",
        radius=0,
    )["image"]
    blurred = np.asarray(
        ImageFilter.execute(image=impulse, operation="gaussian_blur", radius=1, sigma=1.0)["image"]
    )
    np.testing.assert_array_equal(unchanged, impulse)
    assert blurred.shape == impulse.shape
    assert blurred.dtype == np.float32
    assert np.all(blurred[:, 1, 1, 0] < 1.0)
    np.testing.assert_array_equal(blurred[0], blurred[1])
    assert np.all(blurred[:, 0, 0, 0] > blurred[:, 1, 1, 0])

    one_pixel = np.asarray([[[[0.25, 0.5, 0.75]]]], dtype=np.float32)
    tiny = np.asarray(
        ImageFilter.execute(
            image=one_pixel,
            operation="gaussian_blur",
            radius=3,
            sigma=1.0,
        )["image"]
    )
    np.testing.assert_allclose(tiny, one_pixel, rtol=0, atol=1e-6)

    no_sharpen = np.asarray(
        ImageFilter.execute(
            image=impulse,
            operation="sharpen",
            radius=1,
            strength=0.0,
        )["image"]
    )
    np.testing.assert_allclose(no_sharpen, impulse, rtol=0, atol=1e-7)


def test_gaussian_coordinates_and_sharpen_scale_match_core_formulas() -> None:
    impulse = np.zeros((1, 5, 5, 1), dtype=np.float32)
    impulse[:, 2, 2, :] = 1.0
    blurred = np.asarray(
        ImageFilter.execute(
            image=impulse,
            operation="gaussian_blur",
            radius=1,
            sigma=1.0,
        )["image"]
    )
    side = math.exp(-0.5)
    center_weight = 1.0 / (1.0 + 2.0 * side)
    side_weight = side * center_weight
    np.testing.assert_allclose(blurred[0, 2, 2, 0], center_weight**2, rtol=0, atol=1e-7)
    np.testing.assert_allclose(blurred[0, 2, 1, 0], center_weight * side_weight, rtol=0, atol=1e-7)

    sharpened = np.asarray(
        ImageFilter.execute(
            image=impulse,
            operation="sharpen",
            radius=1,
            sigma=1.0,
            strength=0.05,
        )["image"]
    )
    expected = np.clip(impulse + 0.5 * (impulse - blurred), 0.0, 1.0)
    np.testing.assert_allclose(sharpened, expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize("dither", DITHER_MODES)
def test_quantize_is_deterministic_for_each_dither_and_preserves_channel_layout(
    dither: str,
) -> None:
    rgba = _image(channels=4)
    first = np.asarray(
        ImageFilter.execute(
            image=rgba,
            operation="quantize",
            colors=4,
            dither=dither,
        )["image"]
    )
    second = np.asarray(
        ImageFilter.execute(
            image=rgba,
            operation="quantize",
            colors=4,
            dither=dither,
        )["image"]
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[..., 3], rgba[..., 3])
    assert first.shape == rgba.shape
    assert first.dtype == np.float32

    gray = _image(channels=1)
    gray_result = ImageFilter.execute(
        image=gray,
        operation="quantize",
        colors=2,
        dither=dither,
    )["image"]
    assert np.asarray(gray_result).shape == gray.shape


def test_seeded_noise_is_local_deterministic_and_batch_preserving() -> None:
    image = _image(batch=2)
    np.random.seed(1234)
    state_before = pickle.dumps(np.random.get_state())
    first = np.asarray(
        ImageFilter.execute(image=image, operation="noise", seed=99, strength=0.2)["image"]
    )
    state_after = pickle.dumps(np.random.get_state())
    second = np.asarray(
        ImageFilter.execute(image=image, operation="noise", seed=99, strength=0.2)["image"]
    )
    different = np.asarray(
        ImageFilter.execute(image=image, operation="noise", seed=100, strength=0.2)["image"]
    )
    assert state_before == state_after
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)
    assert np.asarray(first).shape == image.shape
    assert np.asarray(first).dtype == np.float32


@pytest.mark.parametrize("operation", FILTER_OPERATIONS[4:])
def test_each_image_morphology_operation_is_bounded_and_shape_preserving(operation: str) -> None:
    image = np.zeros((1, 5, 5, 1), dtype=np.float32)
    image[:, 1:4, 1:4, :] = 1.0
    image[:, 2, 2, :] = 0.0
    result = np.asarray(
        ImageFilter.execute(
            image=image,
            operation=operation,
            kernel_size=3,
        )["image"]
    )
    assert result.shape == image.shape
    assert result.dtype == np.float32
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_morphology_has_neutral_boundaries_and_expected_impulse_behavior() -> None:
    impulse = np.zeros((1, 3, 3, 1), dtype=np.float32)
    impulse[:, 1, 1, :] = 1.0
    dilated = ImageFilter.execute(
        image=impulse,
        operation="dilate",
        kernel_size=3,
    )["image"]
    eroded = ImageFilter.execute(
        image=np.ones_like(impulse),
        operation="erode",
        kernel_size=3,
    )["image"]
    np.testing.assert_array_equal(dilated, 1.0)
    np.testing.assert_array_equal(eroded, 1.0)

    centered = np.zeros((1, 5, 5, 1), dtype=np.float32)
    centered[:, 2, 2, :] = 1.0
    top_hat = ImageFilter.execute(
        image=centered,
        operation="top_hat",
        kernel_size=3,
    )["image"]
    bottom_hat = ImageFilter.execute(
        image=centered,
        operation="bottom_hat",
        kernel_size=3,
    )["image"]
    np.testing.assert_array_equal(top_hat, centered)
    np.testing.assert_array_equal(bottom_hat, 0.0)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"operation": "gaussian_blur", "radius": 32}, "radius"),
        ({"operation": "gaussian_blur", "sigma": 0.0}, "sigma"),
        ({"operation": "quantize", "colors": 0}, "colors"),
        ({"operation": "noise", "seed": -1}, "seed"),
        ({"operation": "erode", "kernel_size": 1000}, "kernel_size"),
        ({"operation": "unknown"}, "unknown image filter"),
    ],
)
def test_filter_rejects_invalid_inputs(arguments: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ImageFilter.execute(
            image=_image(),
            **arguments,  # pyright: ignore[reportArgumentType]
        )
    assert len(FILTER_OPERATIONS) == 11


def test_filter_rejects_nonfinite_pixels() -> None:
    image = _image()
    image[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ImageFilter.execute(image=image, operation="noise")


def test_channel_split_has_rgb_planes_and_explicit_alpha_polarity() -> None:
    rgba = np.asarray([[[[0.1, 0.2, 0.3, 0.75]]]], dtype=np.float32)
    transparency = ImageChannelSplit.execute(
        image=rgba,
        mask_polarity="transparency",
    )
    opacity = ImageChannelSplit.execute(image=rgba, mask_polarity="coverage")
    np.testing.assert_array_equal(transparency["image"], rgba[..., :3])
    np.testing.assert_array_equal(transparency["channel_1"], rgba[..., 0:1])
    np.testing.assert_array_equal(transparency["channel_2"], rgba[..., 1:2])
    np.testing.assert_array_equal(transparency["channel_3"], rgba[..., 2:3])
    np.testing.assert_allclose(np.asarray(transparency["alpha_mask"]), 0.25)
    np.testing.assert_allclose(np.asarray(opacity["alpha_mask"]), 0.75)

    rgb = ImageChannelSplit.execute(image=rgba[..., :3])
    np.testing.assert_array_equal(rgb["alpha_mask"], 0.0)

    grayscale = rgba[..., :1]
    preserved = ImageChannelSplit.execute(image=grayscale, single_channel_image="preserve")
    np.testing.assert_array_equal(preserved["image"], grayscale)
    assert np.asarray(ImageChannelSplit.execute(image=grayscale)["image"]).shape == (1, 1, 1, 3)


def test_rgb_channel_split_merge_round_trip_and_batch_broadcast() -> None:
    image = _image(batch=2)
    split = ImageChannelSplit.execute(image=image)
    merged = np.asarray(
        ImageChannelMerge.execute(
            channel_1=split["channel_1"],
            channel_2=split["channel_2"],
            channel_3=split["channel_3"],
        )["image"]
    )
    np.testing.assert_array_equal(merged, image)

    broadcast = np.asarray(
        ImageChannelMerge.execute(
            channel_1=np.asarray(split["channel_1"])[0:1],
            channel_2=split["channel_2"],
            channel_3=split["channel_3"],
        )["image"]
    )
    np.testing.assert_array_equal(broadcast[:, ..., 0], np.repeat(image[0:1, ..., 0], 2, axis=0))
    np.testing.assert_array_equal(broadcast[:, ..., 1:], image[:, ..., 1:])

    with pytest.raises(ValueError, match="equal batches"):
        ImageChannelMerge.execute(
            channel_1=np.asarray(split["channel_1"])[0:1],
            channel_2=split["channel_2"],
            channel_3=split["channel_3"],
            batch_policy="strict",
        )


def test_ycbcr_split_merge_is_directional_and_value_close() -> None:
    image = _image()
    split = ImageChannelSplit.execute(image=image, color_space="ycbcr")
    merged = ImageChannelMerge.execute(
        channel_1=split["channel_1"],
        channel_2=split["channel_2"],
        channel_3=split["channel_3"],
        color_space="ycbcr",
    )["image"]
    np.testing.assert_allclose(np.asarray(merged), image, rtol=0, atol=3e-4)

    red = np.asarray([[[[1.0, 0.0, 0.0]]]], dtype=np.float32)
    red_split = ImageChannelSplit.execute(image=red, color_space="ycbcr")
    np.testing.assert_allclose(np.asarray(red_split["channel_1"]), 0.299, rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        np.asarray(red_split["channel_2"]), 0.5 - 0.299 * 0.564, rtol=0, atol=1e-7
    )
    np.testing.assert_allclose(
        np.asarray(red_split["channel_3"]), 0.5 + (1.0 - 0.299) * 0.713, rtol=0, atol=1e-7
    )
    repeated = ImageChannelSplit.execute(
        image=red,
        color_space="ycbcr",
        channel_layout="rgb_repeated",
    )
    assert np.asarray(repeated["channel_1"]).shape[-1] == 3
    np.testing.assert_array_equal(
        np.asarray(repeated["channel_1"])[..., 0],
        np.asarray(repeated["channel_1"])[..., 2],
    )


def test_channel_merge_and_alpha_join_keep_distinct_required_arities() -> None:
    split_schema = ImageChannelSplit.schema()
    merge_schema = ImageChannelMerge.schema()
    join_schema = ImageAlphaJoin.schema()
    assert [item.id for item in split_schema.inputs if item.required] == ["image"]
    assert [item.id for item in merge_schema.inputs if item.required] == [
        "channel_1",
        "channel_2",
        "channel_3",
    ]
    assert [item.id for item in join_schema.inputs if item.required] == ["image", "alpha_mask"]


def test_alpha_join_resizes_mask_and_supports_core_cyclic_batch_policy() -> None:
    image = _image(batch=2)
    transparency = np.asarray([[[0.0, 1.0]]], dtype=np.float32)
    result = np.asarray(
        ImageAlphaJoin.execute(
            image=image,
            alpha_mask=transparency,
            mask_polarity="transparency",
            batch_policy="cyclic_repeat",
        )["image"]
    )
    assert result.shape == (2, 3, 4, 4)
    np.testing.assert_array_equal(result[..., :3], image)
    assert np.all(result[:, :, 0, 3] > result[:, :, -1, 3])


def test_merge_optional_alpha_and_invalid_channel_shapes() -> None:
    plane = np.ones((1, 2, 3, 1), dtype=np.float32)
    transparency = np.full((1, 2, 3), 0.25, dtype=np.float32)
    rgba = ImageChannelMerge.execute(
        channel_1=plane,
        channel_2=plane,
        channel_3=plane,
        alpha_mask=transparency,
    )["image"]
    assert np.asarray(rgba).shape == (1, 2, 3, 4)
    np.testing.assert_allclose(np.asarray(rgba)[..., 3], 0.75)

    with pytest.raises(ValueError, match="spatial dimensions"):
        ImageChannelMerge.execute(
            channel_1=plane,
            channel_2=np.ones((1, 3, 3, 1), dtype=np.float32),
            channel_3=plane,
        )
    with pytest.raises(ValueError, match="alpha mask polarity"):
        ImageChannelMerge.execute(
            channel_1=plane,
            channel_2=plane,
            channel_3=plane,
            mask_polarity="unknown",
        )
