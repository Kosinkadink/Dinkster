from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from dinkster_api.v1 import ABSENT, TypeRegistry
from dinkster_nodes_image import (
    DETECTION_TYPE,
    REGION_TYPE,
    RESERVED_TYPE_IDS,
    SEGMENTATION_TYPE,
    VECTOR_TYPE,
    ImageCrop,
    ImageInfo,
    ImageResize,
    ImageTransform,
    ImageUncrop,
    MakeRegion,
    Region,
    RegionInfo,
    register_image_types,
)

GOLDEN_PATH = Path(__file__).parent / "goldens" / "image_geometry_e20d433a.json"
GOLDEN = cast("dict[str, object]", json.loads(GOLDEN_PATH.read_text(encoding="utf-8")))
GOLDEN_CASES = cast("dict[str, object]", GOLDEN["cases"])


def _image(height: int = 3, width: int = 4, batch: int = 1, channels: int = 3) -> np.ndarray:
    values = np.arange(batch * height * width * channels, dtype=np.float32)
    return values.reshape(batch, height, width, channels) / max(1, values.size - 1)


def _golden_array(name: str) -> np.ndarray:
    record = cast("dict[str, object]", GOLDEN_CASES[name])
    shape = cast("list[int]", record["shape"])
    values = cast("list[float]", record["values"])
    return np.asarray(values, dtype=np.float32).reshape(shape)


def _golden_source() -> np.ndarray:
    record = cast("dict[str, object]", GOLDEN["source"])
    return np.asarray(cast("list[float]", record["values"]), dtype=np.float32).reshape(
        cast("list[int]", record["shape"])
    )


def test_comfy_geometry_golden_has_the_required_pin_and_cases() -> None:
    assert GOLDEN["baseline"] == "e20d433a4966dcc88fa5abbae6ace824cb78b263"
    assert set(GOLDEN_CASES) == {
        "AdjustBrightness",
        "AdjustContrast",
        "GetImageSize",
        "ImageBlend:difference",
        "ImageBlend:multiply",
        "ImageBlend:normal",
        "ImageBlend:overlay",
        "ImageBlend:screen",
        "ImageBlend:soft_light",
        "ImageBlur",
        "ImageBlur:stress",
        "ImageCompositeMasked",
        "ImageCompositeMasked:resize-source",
        "ImageCrop",
        "ImageCrop:clamped-origin",
        "ImageCropV2",
        "ImageFlip:x-axis: vertically",
        "ImageFlip:y-axis: horizontally",
        "ImagePadForOutpaint:image",
        "ImagePadForOutpaint:mask",
        "ImageQuantize:bayer-16",
        "ImageQuantize:bayer-2",
        "ImageQuantize:bayer-4",
        "ImageQuantize:bayer-8",
        "ImageQuantize:floyd-steinberg",
        "ImageQuantize:none",
        "ImageRGBToYUV:U",
        "ImageRGBToYUV:V",
        "ImageRGBToYUV:Y",
        "ImageRotate:180 degrees",
        "ImageRotate:270 degrees",
        "ImageRotate:90 degrees",
        "ImageRotate:none",
        "ImageScale:area",
        "ImageScale:bilinear-downscale",
        "ImageScale:bilinear-center",
        "ImageScale:bicubic-nonlinear",
        "ImageScale:bicubic-nonlinear-source",
        "ImageScale:lanczos",
        "ImageScale:nearest-exact",
        "ImageScale:nearest-exact-center-odd",
        "ImageScaleBy",
        "ImageScaleBy:bicubic-nonlinear",
        "ImageScaleToMaxDimension",
        "ImageScaleToMaxDimension:bicubic-nonlinear",
        "ImageScaleToTotalPixels",
        "ImageScaleToTotalPixels:bicubic-nonlinear",
        "ImageSharpen",
        "ImageSharpen:stress",
        "ImageFilter:stress-source",
        "ImageInvert",
        "ImageYUVToRGB",
        "JoinImageWithAlpha",
        "NormalizeImages",
        "Morphology:bottom_hat",
        "Morphology:close",
        "Morphology:dilate",
        "Morphology:erode",
        "Morphology:gradient",
        "Morphology:open",
        "Morphology:source",
        "Morphology:top_hat",
        "PorterDuffImageComposite:image",
        "PorterDuffImageComposite:mask",
        "PorterDuffImageComposite:aspect-destination",
        "PorterDuffImageComposite:aspect-destination-mask",
        "PorterDuffImageComposite:aspect-image",
        "PorterDuffImageComposite:aspect-mask",
        "PorterDuffImageComposite:aspect-source",
        "PorterDuffImageComposite:aspect-source-mask",
        "PrimitiveBoundingBox",
        "SplitImageWithAlpha:image",
        "SplitImageWithAlpha:mask",
    }


def test_core_region_crop_transform_pad_and_info_match_comfy_goldens() -> None:
    source = _golden_source()
    made_region = cast("Region", MakeRegion.execute(x=1, y=2, width=3, height=4)["region"])
    assert made_region.to_record() == GOLDEN_CASES["PrimitiveBoundingBox"]

    crop = ImageCrop.execute(
        image=source,
        x=1,
        y=1,
        width=2,
        height=2,
        rounding="floor",
    )
    crop_v2 = ImageCrop.execute(
        image=source,
        region=Region(1, 0, 3, 2),
        rounding="floor",
    )
    np.testing.assert_array_equal(crop["image"], _golden_array("ImageCrop"))
    np.testing.assert_array_equal(crop_v2["image"], _golden_array("ImageCropV2"))

    transforms = {
        "ImageRotate:none": ImageTransform.execute(image=source, operation="rotate_90", steps=0)[
            "image"
        ],
        "ImageRotate:90 degrees": ImageTransform.execute(
            image=source, operation="rotate_90", steps=1
        )["image"],
        "ImageRotate:180 degrees": ImageTransform.execute(
            image=source, operation="rotate_90", steps=2
        )["image"],
        "ImageRotate:270 degrees": ImageTransform.execute(
            image=source, operation="rotate_90", steps=3
        )["image"],
        "ImageFlip:x-axis: vertically": ImageTransform.execute(
            image=source, operation="flip_vertical"
        )["image"],
        "ImageFlip:y-axis: horizontally": ImageTransform.execute(
            image=source, operation="flip_horizontal"
        )["image"],
    }
    for name, actual in transforms.items():
        np.testing.assert_array_equal(actual, _golden_array(name))

    padded = ImageTransform.execute(
        image=source,
        operation="pad",
        left=1,
        top=1,
        right=2,
        bottom=1,
        feathering=1,
    )
    np.testing.assert_array_equal(padded["image"], _golden_array("ImagePadForOutpaint:image"))
    np.testing.assert_array_equal(padded["mask"], _golden_array("ImagePadForOutpaint:mask"))

    info = ImageInfo.execute(image=source)
    assert [info["width"], info["height"], info["count"]] == GOLDEN_CASES["GetImageSize"]


def test_core_resize_mappings_match_comfy_goldens() -> None:
    source = _golden_source()
    cases: tuple[tuple[str, dict[str, object], float], ...] = (
        (
            "ImageScale:nearest-exact",
            {"width": 6, "height": 5, "interpolation": "nearest-exact"},
            0.0,
        ),
        (
            "ImageScale:bilinear-center",
            {"width": 2, "height": 4, "mode": "fill", "interpolation": "bilinear"},
            1e-6,
        ),
        (
            "ImageScale:bilinear-downscale",
            {"width": 3, "height": 2, "interpolation": "bilinear"},
            1e-7,
        ),
        (
            "ImageScale:area",
            {"width": 3, "height": 2, "interpolation": "area"},
            1e-7,
        ),
        (
            "ImageScale:lanczos",
            {"width": 5, "height": 4, "interpolation": "lanczos"},
            0.0,
        ),
        (
            "ImageScaleBy",
            {"target": "factor", "factor": 1.5, "interpolation": "bicubic"},
            4e-7,
        ),
        (
            "ImageScaleToTotalPixels",
            {
                "target": "total_pixels",
                "megapixels": 48 / (1024 * 1024),
                "resolution_steps": 2,
                "interpolation": "nearest-exact",
            },
            0.0,
        ),
        (
            "ImageScaleToMaxDimension",
            {"target": "longest", "size": 8, "interpolation": "nearest-exact"},
            0.0,
        ),
    )
    for name, arguments, tolerance in cases:
        result = ImageResize.execute(
            image=source,
            **arguments,  # pyright: ignore[reportArgumentType]
        )
        actual = np.asarray(result["image"])
        expected = _golden_array(name)
        if tolerance == 0:
            np.testing.assert_array_equal(actual, expected)
        else:
            # Float32 separable interpolation drifts by at most 2.99e-7; 4e-7 leaves 1.01e-7.
            np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)


def test_core_resize_alias_tolerances_cover_nonlinear_bicubic_domain() -> None:
    source = _golden_array("ImageScale:bicubic-nonlinear-source")
    cases: tuple[tuple[str, dict[str, object]], ...] = (
        ("ImageScale:bicubic-nonlinear", {"width": 7, "height": 7}),
        ("ImageScaleBy:bicubic-nonlinear", {"target": "factor", "factor": 1.4}),
        (
            "ImageScaleToTotalPixels:bicubic-nonlinear",
            {
                "target": "total_pixels",
                "megapixels": 49 / (1024 * 1024),
                "resolution_steps": 1,
            },
        ),
        (
            "ImageScaleToMaxDimension:bicubic-nonlinear",
            {"target": "longest", "size": 7},
        ),
    )
    for name, arguments in cases:
        result = ImageResize.execute(
            image=source,
            interpolation="bicubic",
            **arguments,  # pyright: ignore[reportArgumentType]
        )["image"]
        # Float32 separable accumulation drifts by at most 1.44e-6; 2e-6 leaves 5.6e-7.
        np.testing.assert_allclose(
            np.asarray(result),
            _golden_array(name),
            rtol=0,
            atol=2e-6,
        )


def test_region_is_frozen_validated_and_has_deterministic_codec() -> None:
    region = Region(x=1, y=2.5, width=3, height=4)
    assert region.right == 4
    assert region.bottom == 6.5
    assert region.to_record() == {"x": 1, "y": 2.5, "width": 3, "height": 4}
    with pytest.raises(FrozenInstanceError):
        region.x = 2  # type: ignore[misc]

    registry = TypeRegistry()
    register_image_types(registry)
    register_image_types(registry)
    spec = registry.spec(REGION_TYPE)
    assert spec.encode(region) == b'{"height":4,"width":3,"x":1,"y":2.5}'
    assert spec.decode(spec.encode(region)) == region
    assert spec.coerce is not None
    assert spec.coerce({"x": 1, "y": 2.5, "width": 3, "height": 4}) == region
    assert spec.meta is not None
    assert spec.meta(region) == region.to_record()


def test_region_info_preserves_the_region_and_exposes_coordinates() -> None:
    region = Region(x=1, y=2.5, width=3, height=4)
    assert RegionInfo.execute(region=region) == {
        "region": region,
        "x": 1.0,
        "y": 2.5,
        "width": 3.0,
        "height": 4.0,
        "integer_x": 1,
        "integer_y": 2,
        "integer_width": 3,
        "integer_height": 5,
    }
    assert RegionInfo.execute(region=region, integer_rounding="round")["integer_height"] == 4


@pytest.mark.parametrize(
    ("values", "error"),
    [
        ((True, 0, 1, 1), TypeError),
        ((math.inf, 0, 1, 1), ValueError),
        ((0, 0, -1, 1), ValueError),
    ],
)
def test_region_rejects_invalid_coordinates(
    values: tuple[object, object, object, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        Region(*values)  # type: ignore[arg-type]


def test_region_codec_rejects_invalid_records() -> None:
    registry = TypeRegistry()
    register_image_types(registry)
    spec = registry.spec(REGION_TYPE)
    assert spec.coerce is not None
    with pytest.raises(TypeError, match="expects an object"):
        spec.coerce([])
    with pytest.raises(ValueError, match="requires exactly"):
        spec.coerce({"x": 1, "y": 2, "width": 3})
    with pytest.raises(ValueError, match="invalid dinkster.region payload"):
        spec.decode(b"not json")


def test_future_spatial_type_names_are_reserved() -> None:
    assert VECTOR_TYPE == "dinkster.vector"
    assert SEGMENTATION_TYPE == "dinkster.segmentation"
    assert DETECTION_TYPE == "dinkster.detection"
    assert RESERVED_TYPE_IDS == {VECTOR_TYPE, SEGMENTATION_TYPE}


def test_make_region_returns_the_shared_value() -> None:
    assert MakeRegion.execute(x=1, y=2, width=3, height=4) == {"region": Region(1, 2, 3, 4)}


@pytest.mark.parametrize(
    ("arguments", "shape"),
    [
        ({"target": "dimensions", "width": 6, "height": 5}, (1, 5, 6, 3)),
        ({"target": "dimensions", "width": 0, "height": 6}, (1, 6, 8, 3)),
        ({"target": "dimensions", "width": 8, "height": 0}, (1, 6, 8, 3)),
        ({"target": "width", "width": 8}, (1, 6, 8, 3)),
        ({"target": "width", "width": 0}, (1, 3, 4, 3)),
        ({"target": "height", "height": 6}, (1, 6, 8, 3)),
        ({"target": "height", "height": 0}, (1, 3, 4, 3)),
        ({"target": "longest", "size": 8}, (1, 6, 8, 3)),
        ({"target": "shortest", "size": 6}, (1, 6, 8, 3)),
        ({"target": "factor", "factor": 2}, (1, 6, 8, 3)),
    ],
)
def test_resize_target_policies(arguments: dict[str, object], shape: tuple[int, ...]) -> None:
    result = ImageResize.execute(
        image=_image(),
        interpolation="nearest",
        **arguments,  # pyright: ignore[reportArgumentType]
    )
    assert np.asarray(result["image"]).shape == shape


@pytest.mark.parametrize(("target", "size"), [("width", {"width": 0}), ("height", {"height": 0})])
def test_resize_zero_one_side_target_is_an_exact_noop(target: str, size: dict[str, int]) -> None:
    image = np.linspace(0.0, 1.0, 60, dtype=np.float32).reshape(1, 4, 5, 3)
    mask = np.linspace(0.0, 1.0, 20, dtype=np.float32).reshape(1, 4, 5)
    image_result = ImageResize.execute(
        image=image,
        mask=mask,
        target=target,
        interpolation="lanczos",
        **size,  # pyright: ignore[reportArgumentType]
    )
    mask_result = ImageResize.execute(
        image=mask,
        target=target,
        interpolation="lanczos",
        **size,  # pyright: ignore[reportArgumentType]
    )

    np.testing.assert_array_equal(image_result["image"], image)
    np.testing.assert_array_equal(image_result["mask"], mask)
    np.testing.assert_array_equal(mask_result["image"], mask)


def test_resize_total_pixels_uses_comfy_megapixels_and_resolution_steps() -> None:
    source = _image(height=2, width=4)
    one_source_area_in_megapixels = 8 / (1024 * 1024)
    unchanged = ImageResize.execute(
        image=source,
        target="total_pixels",
        megapixels=one_source_area_in_megapixels,
        resolution_steps=1,
        interpolation="nearest",
    )
    stepped = ImageResize.execute(
        image=source,
        target="total_pixels",
        megapixels=one_source_area_in_megapixels,
        resolution_steps=4,
        interpolation="nearest",
    )
    assert np.asarray(unchanged["image"]).shape == source.shape
    assert np.asarray(stepped["image"]).shape == (1, 4, 4, 3)


def test_resize_to_multiple_floors_each_source_dimension() -> None:
    result = ImageResize.execute(
        image=_image(height=7, width=10),
        width=0,
        height=0,
        divisibility="crop",
        multiple_of=4,
        interpolation="nearest",
    )
    assert np.asarray(result["image"]).shape == (1, 4, 8, 3)

    tiny = ImageResize.execute(
        image=_image(height=2, width=3),
        width=0,
        height=0,
        divisibility="crop",
        multiple_of=8,
    )
    assert np.asarray(tiny["image"]).shape == (1, 1, 1, 3)


def test_resize_to_multiple_cover_resizes_then_center_crops_image_and_mask() -> None:
    image = _image(height=7, width=10)
    mask = np.arange(70, dtype=np.float32).reshape(1, 7, 10) / 69

    result = ImageResize.execute(
        image=image,
        mask=mask,
        target="multiple_cover",
        multiple_of=4,
        interpolation="nearest-exact",
    )
    resized = ImageResize.execute(
        image=image,
        mask=mask,
        target="dimensions",
        width=8,
        height=6,
        interpolation="nearest-exact",
    )

    np.testing.assert_array_equal(result["image"], np.asarray(resized["image"])[:, 1:5])
    np.testing.assert_array_equal(result["mask"], np.asarray(resized["mask"])[:, 1:5])


@pytest.mark.parametrize(("height", "width", "multiple"), [(8, 12, 4), (3, 2, 8), (5, 7, 1)])
def test_resize_to_multiple_cover_is_noop_when_no_valid_crop_exists(
    height: int,
    width: int,
    multiple: int,
) -> None:
    image = _image(height=height, width=width)
    result = ImageResize.execute(
        image=image,
        target="multiple_cover",
        multiple_of=multiple,
        interpolation="lanczos",
    )
    np.testing.assert_array_equal(result["image"], image)


def test_resize_can_match_a_reference_image() -> None:
    result = ImageResize.execute(
        image=_image(height=3, width=4),
        reference=np.zeros((1, 5, 7), dtype=np.float32),
        target="match",
        interpolation="nearest-exact",
    )
    assert np.asarray(result["image"]).shape == (1, 5, 7, 3)
    with pytest.raises(ValueError, match="requires reference"):
        ImageResize.execute(image=_image(), target="match")


@pytest.mark.parametrize(
    "interpolation", ("nearest", "nearest-exact", "bilinear", "bicubic", "lanczos", "area")
)
def test_resize_supports_every_interpolation(interpolation: str) -> None:
    result = ImageResize.execute(image=_image(), width=5, height=4, interpolation=interpolation)
    assert np.asarray(result["image"]).shape == (1, 4, 5, 3)


def test_resize_distinguishes_nearest_from_nearest_exact() -> None:
    source = np.array([[[[0.0], [1.0]]]], dtype=np.float32)
    nearest = ImageResize.execute(
        image=source,
        width=3,
        height=1,
        interpolation="nearest",
    )
    nearest_exact = ImageResize.execute(
        image=source,
        width=3,
        height=1,
        interpolation="nearest-exact",
    )
    np.testing.assert_array_equal(nearest["image"], [[[[0.0], [0.0], [1.0]]]])
    np.testing.assert_array_equal(nearest_exact["image"], [[[[0.0], [1.0], [1.0]]]])


def test_resize_fit_fill_and_pad_have_distinct_geometry() -> None:
    source = _image(height=2, width=4, channels=1)
    fit = np.asarray(ImageResize.execute(image=source, width=4, height=4, mode="fit")["image"])
    fill = np.asarray(ImageResize.execute(image=source, width=4, height=4, mode="fill")["image"])
    padded = np.asarray(
        ImageResize.execute(
            image=source,
            width=4,
            height=4,
            mode="pad",
            pad_value=0.25,
        )["image"]
    )
    assert fit.shape == (1, 2, 4, 1)
    assert fill.shape == (1, 4, 4, 1)
    assert padded.shape == (1, 4, 4, 1)
    np.testing.assert_array_equal(padded[:, 0], 0.25)
    np.testing.assert_array_equal(padded[:, 3], 0.25)


def test_resize_fill_splits_odd_crop_extents_without_rescaling_extra_pixels() -> None:
    columns = np.broadcast_to(
        np.arange(6, dtype=np.float32)[None, None, :, None],
        (1, 3, 6, 1),
    ).copy()
    result = ImageResize.execute(
        image=columns,
        mask=columns[..., 0],
        width=3,
        height=3,
        mode="fill",
        interpolation="nearest-exact",
    )
    np.testing.assert_array_equal(np.asarray(result["image"])[0, 0, :, 0], [1, 2, 3])
    np.testing.assert_array_equal(np.asarray(result["mask"])[0, 0], [1, 2, 3])

    one_pixel = ImageResize.execute(
        image=columns[:, :, :4],
        width=3,
        height=3,
        mode="fill",
        interpolation="nearest-exact",
    )
    np.testing.assert_array_equal(np.asarray(one_pixel["image"])[0, 0, :, 0], [0, 1, 2])

    rows = np.swapaxes(columns, 1, 2)
    vertical = ImageResize.execute(
        image=rows,
        width=3,
        height=3,
        mode="fill",
        interpolation="nearest-exact",
    )
    np.testing.assert_array_equal(np.asarray(vertical["image"])[0, :, 0, 0], [1, 2, 3])


def test_resize_applies_identical_geometry_to_optional_mask() -> None:
    image = np.zeros((1, 2, 2, 3), dtype=np.float32)
    mask = np.array([[[0, 1], [1, 0]]], dtype=np.float32)
    result = ImageResize.execute(
        image=image,
        mask=mask,
        width=4,
        height=4,
        interpolation="nearest",
    )
    assert np.asarray(result["image"]).shape == (1, 4, 4, 3)
    np.testing.assert_array_equal(
        result["mask"],
        np.repeat(np.repeat(mask, 2, axis=1), 2, axis=2),
    )
    assert ImageResize.execute(image=image, width=1, height=1)["mask"] is ABSENT


def test_resize_intents_compose_fit_fill_divisibility_and_mask_geometry() -> None:
    image = _image(height=3, width=5, channels=1)
    mask = np.arange(15, dtype=np.float32).reshape(1, 3, 5) / 14
    fitted = ImageResize.execute(
        image=image,
        width=8,
        height=8,
        mode="fit",
        interpolation="nearest-exact",
        divisibility="crop",
        multiple_of=4,
    )
    assert np.asarray(fitted["image"]).shape == (1, 4, 8, 1)

    cropped = ImageResize.execute(
        image=image,
        mask=mask,
        width=2,
        height=2,
        mode="fill",
        mode_anchor="right",
        interpolation="nearest-exact",
    )
    assert np.asarray(cropped["image"]).shape == (1, 2, 2, 1)
    assert np.asarray(cropped["mask"]).shape == (1, 2, 2)
    np.testing.assert_array_equal(np.asarray(cropped["image"])[..., 0], cropped["mask"])

    columns = np.broadcast_to(
        np.arange(4, dtype=np.float32)[None, None, :, None],
        (1, 3, 4, 1),
    ).copy()
    left = ImageResize.execute(
        image=columns,
        width=3,
        height=3,
        mode="fill",
        mode_anchor="left",
        interpolation="nearest-exact",
    )
    right = ImageResize.execute(
        image=columns,
        width=3,
        height=3,
        mode="fill",
        mode_anchor="right",
        interpolation="nearest-exact",
    )
    np.testing.assert_array_equal(np.asarray(left["image"])[0, 0, :, 0], [0, 1, 2])
    np.testing.assert_array_equal(np.asarray(right["image"])[0, 0, :, 0], [1, 2, 3])

    mismatched_mask = ImageResize.execute(
        image=_image(batch=2),
        mask=np.zeros((1, 6, 8), dtype=np.float32),
        width=8,
        height=6,
    )
    assert np.asarray(mismatched_mask["mask"]).shape == (2, 6, 8)


def test_resize_padding_mask_is_meaningful_for_large_batches() -> None:
    image = np.ones((65, 2, 4, 1), dtype=np.float32)
    result = ImageResize.execute(image=image, width=4, height=4, mode="pad")
    mask = np.asarray(result["mask"])
    assert mask.shape == (65, 4, 4)
    np.testing.assert_array_equal(mask[:, (0, 3), :], 1.0)
    np.testing.assert_array_equal(mask[:, 1:3, :], 0.0)
    assert ImageResize.execute(image=image, width=4, height=2)["mask"] is ABSENT


def test_resize_padding_styles_preserve_distinct_pixel_intents() -> None:
    image = np.asarray(
        [[[[0.0], [0.25], [0.75], [1.0]], [[1.0], [0.75], [0.25], [0.0]]]],
        dtype=np.float32,
    )
    results = {
        style: np.asarray(
            ImageResize.execute(
                image=image,
                width=4,
                height=4,
                mode="pad",
                mode_padding=style,
                pad_value=0.125,
                interpolation="nearest-exact",
            )["image"]
        )
        for style in ("constant", "edge_average", "edge_pixel", "blurred_background")
    }

    assert all(result.shape == (1, 4, 4, 1) for result in results.values())
    np.testing.assert_array_equal(results["constant"][:, 0], 0.125)
    np.testing.assert_array_equal(results["edge_average"][:, 0], 0.5)
    np.testing.assert_array_equal(results["edge_pixel"][:, 0], image[:, 0])
    assert np.isfinite(results["blurred_background"]).all()
    assert not np.array_equal(results["blurred_background"][:, 0], results["constant"][:, 0])
    for result in results.values():
        np.testing.assert_array_equal(result[:, 1:3], image)


@pytest.mark.parametrize(
    ("pad_color", "expected"),
    [("0, 0, 0", 0.0), ("255, 0, 0", 0.2126)],
)
def test_resize_rgb_pad_color_maps_to_mask_luma(pad_color: str, expected: float) -> None:
    mask = np.ones((1, 2, 4), dtype=np.float32)
    result = ImageResize.execute(
        image=mask,
        width=4,
        height=4,
        mode="pad",
        interpolation="nearest-exact",
        pad_color=pad_color,
    )
    output = np.asarray(result["image"])
    assert output.shape == (1, 4, 4)
    np.testing.assert_allclose(output[:, (0, 3), :], expected, rtol=0.0, atol=1e-6)
    np.testing.assert_array_equal(output[:, 1:3, :], 1.0)


def test_resize_apply_conditions_padding_and_divisibility_are_independent_intents() -> None:
    image = _image(height=3, width=5, channels=1)
    skipped = ImageResize.execute(
        image=image,
        width=2,
        height=2,
        mode="stretch",
        apply="only_if_smaller",
        divisibility="crop",
        multiple_of=2,
    )
    assert np.asarray(skipped["image"]).shape == (1, 2, 4, 1)
    np.testing.assert_array_equal(np.asarray(skipped["image"]), image[:, :2, :4, :])

    padded = ImageResize.execute(
        image=image,
        width=8,
        height=8,
        mode="pad",
        interpolation="nearest-exact",
    )
    assert np.asarray(padded["image"]).shape == (1, 8, 8, 1)
    np.testing.assert_array_equal(np.asarray(padded["image"])[:, :1], 0)

    with pytest.raises(ValueError, match="output dimensions exceed"):
        ImageResize.execute(
            image=image,
            width=20_000,
            height=1,
            mode="pad",
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"target": "missing"}, "unknown resize target"),
        ({"interpolation": "missing"}, "unknown interpolation"),
        ({"mode": "missing"}, "unknown resize mode"),
        ({"width": 16_385}, "output dimensions exceed"),
        ({"target": "factor", "factor": math.nan}, "factor must be finite"),
        ({"target": "total_pixels", "megapixels": 0}, "megapixels must be"),
        ({"divisibility": "missing"}, "unknown divisibility"),
        ({"divisibility": "crop", "multiple_of": -1}, "must be non-negative"),
    ],
)
def test_resize_rejects_invalid_parameters(arguments: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ImageResize.execute(image=_image(), **arguments)  # pyright: ignore[reportArgumentType]


def test_resize_rejects_invalid_image_and_mask_shapes() -> None:
    with pytest.raises(ValueError, match="BHWC"):
        ImageResize.execute(image=np.zeros((4, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="batch must match image or be one"):
        ImageResize.execute(image=_image(), mask=np.zeros((2, 2, 2), dtype=np.float32))


def test_transform_exact_quarter_turns_and_flips() -> None:
    image = _image(height=2, width=3, channels=1)
    rotated = ImageTransform.execute(image=image, operation="rotate_90", steps=1)
    horizontal = ImageTransform.execute(image=image, operation="flip_horizontal")
    vertical = ImageTransform.execute(image=image, operation="flip_vertical")
    np.testing.assert_array_equal(rotated["image"], np.rot90(image, k=-1, axes=(1, 2)))
    np.testing.assert_array_equal(horizontal["image"], image[:, :, ::-1, :])
    np.testing.assert_array_equal(vertical["image"], image[:, ::-1, :, :])
    assert rotated["mask"] is ABSENT


def test_transform_translate_shear_and_expand_rotate() -> None:
    image = _image(height=3, width=4, channels=1)
    translated = np.asarray(
        ImageTransform.execute(
            image=image,
            operation="translate",
            x=1,
            interpolation="nearest",
        )["image"]
    )
    np.testing.assert_array_equal(translated[:, :, 1:, :], image[:, :, :-1, :])
    np.testing.assert_array_equal(translated[:, :, 0, :], 0)
    sheared = ImageTransform.execute(
        image=image,
        operation="shear",
        x=1,
        y=-1,
        interpolation="nearest",
    )
    expanded = ImageTransform.execute(
        image=image,
        operation="rotate",
        angle=45,
        expand="true",
    )
    assert np.asarray(sheared["image"]).shape == image.shape
    assert np.asarray(expanded["image"]).shape[1:3] == (5, 6)


def test_transform_pad_matches_comfy_outpaint_contract() -> None:
    image = np.ones((2, 5, 5, 1), dtype=np.float32)
    result = ImageTransform.execute(
        image=image,
        operation="pad",
        left=1,
        top=1,
        right=2,
        bottom=1,
        feathering=2,
    )
    padded = np.asarray(result["image"])
    mask = np.asarray(result["mask"])
    assert padded.shape == (2, 7, 8, 1)
    assert mask.shape == (1, 7, 8)
    np.testing.assert_array_equal(padded[:, 1:6, 1:6, :], image)
    np.testing.assert_array_equal(padded[:, 0, :, :], 0.5)
    assert mask[0, 1, 1] == 1.0
    assert mask[0, 2, 2] == 0.25
    assert mask[0, 3, 3] == 0.0
    assert mask[0, 0, 0] == 1.0


@pytest.mark.parametrize(
    "arguments",
    (
        {"operation": "missing"},
        {"operation": "translate", "units": "missing"},
        {"operation": "rotate", "interpolation": "missing"},
        {"operation": "pad", "left": -1},
    ),
)
def test_transform_rejects_invalid_parameters(arguments: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ImageTransform.execute(
            image=_image(),
            **arguments,  # pyright: ignore[reportArgumentType]
        )


def test_crop_clips_or_pads_fractional_regions_with_explicit_rounding() -> None:
    image = _image(height=3, width=4, channels=1)
    clipped = ImageCrop.execute(
        image=image,
        region=Region(-0.2, 0.8, 3.1, 3),
        rounding="expand",
        outside="clip",
    )
    padded = ImageCrop.execute(
        image=image,
        region=Region(-1, -1, 3, 3),
        outside="pad",
        fill=0.25,
    )
    assert clipped["region"] == Region(0, 0, 3, 3)
    np.testing.assert_array_equal(clipped["image"], image[:, :, :3, :])
    assert padded["region"] == Region(-1, -1, 3, 3)
    assert np.asarray(padded["image"]).shape == (1, 3, 3, 1)
    np.testing.assert_array_equal(np.asarray(padded["image"])[:, 0, :, :], 0.25)
    np.testing.assert_array_equal(np.asarray(padded["image"])[:, :, 0, :], 0.25)


def test_crop_comfy_policy_clamps_the_origin_before_slicing() -> None:
    result = ImageCrop.execute(
        image=_golden_source(),
        x=20,
        y=20,
        width=2,
        height=2,
        rounding="floor",
        outside="comfy",
    )
    np.testing.assert_array_equal(result["image"], _golden_array("ImageCrop:clamped-origin"))
    assert result["region"] == Region(3, 2, 2, 2)


def test_crop_region_overrides_coordinates_and_zero_size_extends_to_edge() -> None:
    image = _image(channels=1)
    from_region = ImageCrop.execute(
        image=image,
        region=Region(1, 1, 2, 1),
        x=0,
        y=0,
        width=4,
        height=3,
    )
    to_edge = ImageCrop.execute(image=image, x=1, y=1, width=0, height=0)
    np.testing.assert_array_equal(from_region["image"], image[:, 1:2, 1:3, :])
    np.testing.assert_array_equal(to_edge["image"], image[:, 1:, 1:, :])


def test_crop_placement_applies_offsets_and_reports_the_clipped_region() -> None:
    image = _image(height=5, width=7, channels=1)
    result = ImageCrop.execute(
        image=image,
        placement="bottom_right",
        width=4,
        height=3,
        x=-1,
        y=1,
        rounding="floor",
    )
    assert result["region"] == Region(2, 3, 4, 2)
    np.testing.assert_array_equal(result["image"], image[:, 3:5, 2:6, :])


def test_crop_derives_union_region_from_mask_and_returns_synchronized_mask() -> None:
    image = _image(height=4, width=5, batch=2, channels=1)
    mask = np.zeros((1, 4, 5), dtype=np.float32)
    mask[:, 1:3, 2:5] = 0.75
    result = ImageCrop.execute(
        image=image,
        mask=mask,
        mask_threshold=0.5,
        padding=1,
    )
    assert result["region"] == Region(1, 0, 4, 4)
    np.testing.assert_array_equal(result["image"], image[:, :, 1:5, :])
    expected_mask = np.broadcast_to(mask[:, :, 1:5], (2, 4, 4))
    np.testing.assert_array_equal(result["mask"], expected_mask)


def test_crop_mask_blur_uses_essentials_kernel_size_semantics() -> None:
    image = _image(height=5, width=5, channels=1)
    mask = np.zeros((1, 5, 5), dtype=np.float32)
    mask[:, 2, 2] = 1.0
    result = ImageCrop.execute(image=image, mask=mask, mask_blur=3)
    assert result["region"] == Region(1, 1, 3, 3)
    assert np.asarray(result["image"]).shape == (1, 3, 3, 1)
    assert np.asarray(result["mask"]).shape == (1, 3, 3)
    assert np.all(np.asarray(result["mask"]) > 0)


def test_crop_rejects_empty_or_mismatched_masks_and_disjoint_regions() -> None:
    image = _image()
    with pytest.raises(ValueError, match="selected pixels"):
        ImageCrop.execute(image=image, mask=np.zeros((1, 3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="dimensions must match"):
        ImageCrop.execute(image=image, mask=np.zeros((1, 2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="does not intersect"):
        ImageCrop.execute(image=image, region=Region(10, 10, 2, 2))


def test_uncrop_replaces_blends_and_clips_the_crop() -> None:
    base = np.zeros((1, 4, 5, 1), dtype=np.float32)
    crop = np.ones((1, 2, 3, 1), dtype=np.float32)
    replaced = np.asarray(
        ImageUncrop.execute(base=base, crop=crop, region=Region(1, 1, 3, 2))["image"]
    )
    blended = np.asarray(
        ImageUncrop.execute(
            base=base,
            crop=crop,
            region=Region(-1, 1, 3, 2),
            opacity=0.5,
        )["image"]
    )
    np.testing.assert_array_equal(replaced[:, 1:3, 1:4, :], 1)
    assert replaced.sum() == 6
    np.testing.assert_array_equal(blended[:, 1:3, :2, :], 0.5)
    assert blended.sum() == 2


def test_uncrop_resizes_crop_and_mask_and_broadcasts_batches() -> None:
    base = np.zeros((2, 4, 4, 1), dtype=np.float32)
    crop = np.ones((1, 1, 1, 1), dtype=np.float32)
    mask = np.array([[[1, 0], [0.5, 0]]], dtype=np.float32)
    result = np.asarray(
        ImageUncrop.execute(
            base=base,
            crop=crop,
            mask=mask,
            region=Region(1, 1, 2, 2),
            interpolation="nearest",
        )["image"]
    )
    expected = np.array([[1, 0], [0.5, 0]], dtype=np.float32)
    np.testing.assert_array_equal(result[0, 1:3, 1:3, 0], expected)
    np.testing.assert_array_equal(result[1, 1:3, 1:3, 0], expected)


def test_uncrop_matches_mtb_border_blending_reference() -> None:
    base = np.zeros((1, 5, 5, 1), dtype=np.float32)
    crop = np.array([[[[0.0], [1.0]], [[0.25], [0.75]]]], dtype=np.float32)
    result = np.asarray(
        ImageUncrop.execute(
            base=base,
            crop=crop,
            region=Region(1, 1, 3, 3),
            border_blending=1.0,
            rounding="floor",
            interpolation="bicubic",
            outside="ignore",
        )["image"]
    )
    expected = np.array(
        [
            [-0.06502156, 0.38050297, 0.64415139],
            [0.04558108, 0.50000006, 0.71542466],
            [0.13439648, 0.38050297, 0.44473344],
        ],
        dtype=np.float32,
    )
    # Float32 resize and Gaussian accumulation drift by 1.79e-7; 3e-7 leaves 1.21e-7.
    np.testing.assert_allclose(result[0, 1:4, 1:4, 0], expected, rtol=0, atol=3e-7)


def test_uncrop_can_ignore_a_disjoint_region() -> None:
    base = _image()
    result = ImageUncrop.execute(
        base=base,
        crop=_image(),
        region=Region(20, 20, 2, 2),
        outside="ignore",
    )
    np.testing.assert_array_equal(result["image"], base)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"region": Region(0, 0, 0, 1)}, "positive dimensions"),
        ({"region": Region(10, 10, 1, 1)}, "does not intersect"),
        ({"region": Region(0, 0, 1, 1), "opacity": 2}, "opacity must be"),
        ({"region": Region(0, 0, 1, 1), "border_blending": 2}, "border_blending must be"),
    ],
)
def test_uncrop_rejects_invalid_regions_and_opacity(
    arguments: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ImageUncrop.execute(
            base=_image(),
            crop=_image(),
            **arguments,  # pyright: ignore[reportArgumentType]
        )


def test_uncrop_rejects_channels_and_nonbroadcastable_batches() -> None:
    with pytest.raises(ValueError, match="channels must match"):
        ImageUncrop.execute(
            base=_image(channels=3),
            crop=_image(channels=4),
            region=Region(0, 0, 4, 3),
        )
    with pytest.raises(ValueError, match="batch must be"):
        ImageUncrop.execute(
            base=_image(batch=2),
            crop=_image(batch=3),
            region=Region(0, 0, 4, 3),
        )


def test_image_info_reports_stable_dimensions_statistics_and_histogram() -> None:
    image = np.array([[[[-1.0], [0.25]], [[0.75], [2.0]]]], dtype=np.float32)
    result = ImageInfo.execute(image=image, histogram_bins=2)
    assert result == {
        "width": 2,
        "height": 2,
        "count": 1,
        "channels": 1,
        "mean": 0.5,
        "minimum": -1.0,
        "maximum": 2.0,
        "histogram": [2, 2],
    }
    with pytest.raises(ValueError, match="between 2 and 4096"):
        ImageInfo.execute(image=image, histogram_bins=1)
