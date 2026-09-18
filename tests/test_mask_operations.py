from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from dinkster_api.v1 import ComboWidget, InputSpec
from dinkster_nodes_image import IMAGE_NODES
from dinkster_nodes_image.drawing import MakeImage
from dinkster_nodes_image.mask import (
    MASK_NODES,
    ImageToMask,
    MakeMask,
    MaskCombine,
    MaskInfo,
    MaskMorphology,
    MaskToImage,
    TextMask,
)
from dinkster_nodes_image.types import Region
from dinkster_schema import build_schemas, schema_from_wire, schema_to_wire

GOLDEN_PATH = Path(__file__).parent / "goldens" / "image_mask_e20d433a.json"
GOLDEN = cast("dict[str, object]", json.loads(GOLDEN_PATH.read_text(encoding="utf-8")))
GOLDEN_CASES = cast("dict[str, object]", GOLDEN["cases"])


def _run(node: type, **kwargs: object) -> dict[str, object]:
    return node.execute(**kwargs)  # pyright: ignore[reportUnknownMemberType]


def _mask(result: dict[str, object]) -> np.ndarray:
    return np.asarray(result["mask"], dtype=np.float32)


def _golden_array(name: str) -> np.ndarray:
    record = cast("dict[str, object]", GOLDEN_CASES[name])
    return np.asarray(cast("list[float]", record["values"]), dtype=np.float32).reshape(
        cast("list[int]", record["shape"])
    )


def _golden_source(name: str) -> np.ndarray:
    record = cast("dict[str, object]", GOLDEN[name])
    return np.asarray(cast("list[float]", record["values"]), dtype=np.float32).reshape(
        cast("list[int]", record["shape"])
    )


def test_comfy_mask_golden_has_the_required_pin_and_cases() -> None:
    assert GOLDEN["baseline"] == "e20d433a4966dcc88fa5abbae6ace824cb78b263"
    assert set(GOLDEN_CASES) == {
        "CropMask",
        "EmptyImage",
        "FeatherMask",
        "GrowMask:square",
        "GrowMask:tapered-erode",
        "ImageBatch",
        "ImageColorToMask",
        "ImageFromBatch",
        "ImageStitch",
        "ImageToMask:alpha",
        "ImageToMask:blue",
        "ImageToMask:green",
        "ImageToMask:red",
        "InvertMask",
        "MaskComposite:add",
        "MaskComposite:and",
        "MaskComposite:multiply",
        "MaskComposite:or",
        "MaskComposite:source-singleton",
        "MaskComposite:subtract",
        "MaskComposite:xor",
        "MaskToImage",
        "RebatchImages",
        "RepeatImageBatch",
        "SolidMask",
        "ThresholdMask",
    }


def test_core_mask_and_image_generation_match_comfy_goldens() -> None:
    source_mask = _golden_source("sourceMask")
    source_image = _golden_source("sourceImage")

    actual: dict[str, object] = {
        "SolidMask": MakeMask.execute(operation="solid", width=5, height=4, foreground=0.25)[
            "mask"
        ],
        "InvertMask": MaskMorphology.execute(mask=source_mask, operation="invert")["mask"],
        "CropMask": MaskMorphology.execute(
            mask=source_mask, operation="crop", x=1, y=1, width=3, height=2
        )["mask"],
        "FeatherMask": MaskMorphology.execute(
            mask=source_mask,
            operation="feather_edges",
            left=2,
            top=1,
            right=2,
            bottom=1,
        )["mask"],
        "GrowMask:square": MaskMorphology.execute(
            mask=source_mask,
            operation="grow_erode",
            radius=2,
            tapered_corners=False,
            edge_policy="reflect",
        )["mask"],
        "GrowMask:tapered-erode": MaskMorphology.execute(
            mask=source_mask,
            operation="grow_erode",
            radius=-1,
            tapered_corners=True,
            edge_policy="reflect",
        )["mask"],
        "ThresholdMask": MaskMorphology.execute(
            mask=source_mask, operation="threshold", threshold=0.5
        )["mask"],
        "ImageColorToMask": ImageToMask.execute(
            image=source_image,
            policy="exact_color",
            color_source="integer",
            color_value=0x8040FF,
        )["mask"],
        "MaskToImage": MaskToImage.execute(mask=source_mask)["image"],
        "EmptyImage": MakeImage.execute(
            operation="solid",
            width=3,
            height=2,
            batch_size=2,
            color_source="integer",
            color_value=0x3366CC,
        )["image"],
    }
    for channel in ("red", "green", "blue", "alpha"):
        actual[f"ImageToMask:{channel}"] = ImageToMask.execute(image=source_image, channel=channel)[
            "mask"
        ]
    source = np.flip(source_mask, axis=(1, 2))
    for operation in ("multiply", "add", "subtract", "and", "or", "xor"):
        actual[f"MaskComposite:{operation}"] = MaskCombine.execute(
            destination=source_mask, source=source, x=1, y=1, operation=operation
        )["mask"]
    destination_batch = np.concatenate((source_mask, 1.0 - source_mask), axis=0)
    actual["MaskComposite:source-singleton"] = MaskCombine.execute(
        destination=destination_batch,
        source=source,
        x=1,
        y=1,
        operation="add",
        batch_policy="source_singleton",
    )["mask"]

    for name, value in actual.items():
        np.testing.assert_array_equal(np.asarray(value), _golden_array(name), err_msg=name)


@pytest.mark.parametrize(
    ("operation", "kwargs"),
    [
        ("solid", {"foreground": 0.25}),
        ("rectangle", {"x": 2, "y": 1, "shape_width": 5, "shape_height": 3}),
        ("ellipse", {"x": 2, "y": 1, "shape_width": 5, "shape_height": 3}),
        ("linear_gradient", {"angle": 30.0, "background": 0.2, "foreground": 0.8}),
        ("radial_gradient", {"center_x": 0.4, "center_y": 0.6, "radius": 0.7}),
        ("checkerboard", {"tile_size": 3}),
    ],
)
def test_make_mask_operations_are_batched_float32_and_deterministic(
    operation: str, kwargs: dict[str, object]
) -> None:
    first = _mask(_run(MakeMask, operation=operation, width=11, height=7, batch_size=3, **kwargs))
    second = _mask(_run(MakeMask, operation=operation, width=11, height=7, batch_size=3, **kwargs))

    assert first.shape == (3, 7, 11)
    assert first.dtype == np.float32
    assert first.flags.c_contiguous
    assert np.array_equal(first, second)
    assert np.array_equal(first[0], first[2])


def test_make_mask_shapes_and_empty_content() -> None:
    empty = _mask(_run(MakeMask, operation="solid", width=5, height=4, foreground=0.0))
    rectangle = _mask(
        _run(
            MakeMask,
            operation="rectangle",
            width=5,
            height=5,
            x=1,
            y=1,
            shape_width=2,
            shape_height=2,
        )
    )
    ellipse = _mask(
        _run(
            MakeMask,
            operation="ellipse",
            width=5,
            height=5,
            x=0,
            y=0,
            shape_width=4,
            shape_height=4,
        )
    )

    assert not np.any(empty)
    assert rectangle.sum() == 4.0
    assert ellipse[0, 2, 2] == 1.0
    assert ellipse[0, 0, 0] == 0.0


def test_make_mask_extended_generators() -> None:
    triangle = _mask(
        _run(
            MakeMask,
            operation="triangle",
            width=9,
            height=9,
            batch_size=2,
            x=4,
            y=4,
            shape_origin="center",
            shape_width=4,
            shape_height=4,
            grow=2,
        )
    )
    polygon = _mask(
        _run(
            MakeMask,
            operation="polygon",
            width=7,
            height=6,
            points_x=[1, 5, 3],
            points_y=[4, 4, 1],
            foreground=0.75,
        )
    )
    frame_gradient = _mask(
        _run(MakeMask, operation="frame_gradient", width=3, height=2, batch_size=2)
    )
    noise = _mask(
        _run(
            MakeMask,
            operation="noise",
            width=8,
            height=5,
            batch_size=2,
            seed=671,
            noise_mode="binary",
            noise_density=0.25,
        )
    )
    noise_again = _mask(
        _run(
            MakeMask,
            operation="noise",
            width=8,
            height=5,
            batch_size=2,
            seed=671,
            noise_mode="binary",
            noise_density=0.25,
        )
    )

    assert triangle[1].sum() > triangle[0].sum() > 0
    assert polygon[0, 3, 3] == 0.75
    assert polygon[0, 0, 0] == 0.0
    assert np.array_equal(
        frame_gradient,
        np.array(
            [
                [[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]],
                [[0.5, 0.0, -0.5], [0.5, 0.0, -0.5]],
            ],
            dtype=np.float32,
        ),
    )
    assert set(np.unique(noise)) <= {0.0, 1.0}
    assert np.array_equal(noise, noise_again)
    assert not np.array_equal(noise[0], noise[1])


@pytest.mark.parametrize(
    "transition_type",
    [
        "horizontal_slide",
        "vertical_slide",
        "horizontal_bar",
        "vertical_bar",
        "center_box",
        "horizontal_door",
        "vertical_door",
        "circle",
        "fade",
    ],
)
def test_make_mask_transition_types_have_defined_endpoints(transition_type: str) -> None:
    transition = _mask(
        _run(
            MakeMask,
            operation="transition",
            width=7,
            height=5,
            batch_size=3,
            start_frame=0,
            end_frame=3,
            transition_type=transition_type,
        )
    )
    assert transition[0].sum() == (1.0 if transition_type == "circle" else 0.0)
    assert np.all(transition[-1] == 1.0)


@pytest.mark.parametrize(
    ("timing", "middle"),
    [("linear", 0.5), ("in", 0.25), ("out", 0.75), ("in_out", 0.5)],
)
def test_make_mask_transition_timing_and_degenerate_ranges(timing: str, middle: float) -> None:
    transition = _mask(
        _run(
            MakeMask,
            operation="transition",
            width=2,
            height=2,
            batch_size=3,
            transition_type="fade",
            timing_function=timing,
        )
    )
    single = _mask(
        _run(
            MakeMask,
            operation="transition",
            width=2,
            height=2,
            batch_size=1,
            transition_type="fade",
        )
    )
    reversed_range = _mask(
        _run(
            MakeMask,
            operation="transition",
            width=2,
            height=2,
            batch_size=4,
            start_frame=3,
            end_frame=1,
            transition_type="fade",
        )
    )

    assert np.allclose(transition[:, 0, 0], [0.0, middle, 1.0])
    assert np.all(single == 1.0)
    assert np.array_equal(reversed_range[:, 0, 0], [0.0, 0.0, 0.0, 1.0])


def test_text_mask_uses_pillow_bundled_default_font_deterministically() -> None:
    mask = _mask(
        _run(
            TextMask,
            text="Dinkster\n671",
            width=64,
            height=40,
            x=2,
            y=3,
            font_size=12,
            line_spacing=2,
        )
    )

    assert mask.shape == (1, 40, 64)
    assert hashlib.sha256(mask.tobytes()).hexdigest() == (
        "9980a74a09126377f4b2b53faac2585a41c6c641be918d84415e5a027cbfca4d"
    )
    assert np.array_equal(
        mask,
        _mask(
            _run(
                TextMask,
                text="Dinkster\n671",
                width=64,
                height=40,
                x=2,
                y=3,
                font_size=12,
                line_spacing=2,
            )
        ),
    )


def test_text_mask_color_rotation_and_outputs() -> None:
    regular = _run(
        TextMask,
        text="A",
        width=32,
        height=32,
        batch_size=2,
        x=4,
        y=5,
        font_size=16,
        color="white",
        mask_value="color_red",
        start_rotation=0.0,
        end_rotation=90.0,
    )
    inverted = _run(
        TextMask,
        text="A",
        width=32,
        height=32,
        batch_size=2,
        x=4,
        y=5,
        font_size=16,
        color="white",
        mask_value="color_red",
        start_rotation=0.0,
        end_rotation=90.0,
        invert=True,
    )
    mask = np.asarray(regular["mask"], dtype=np.float32)
    image = np.asarray(regular["image"], dtype=np.float32)

    assert mask.shape == (2, 32, 32)
    assert image.shape == (2, 32, 32, 3)
    assert not np.array_equal(mask[0], mask[1])
    assert np.array_equal(np.asarray(inverted["mask"]), 1.0 - mask)
    assert np.array_equal(np.asarray(inverted["image"]), 1.0 - image)
    assert np.array_equal(np.asarray(regular["inverse_mask"]), 1.0 - mask)


def test_threshold_is_strict_and_empty_mask_stays_empty() -> None:
    values = np.array([[[0.49, 0.5, 0.51], [0.0, 1.0, 0.5]]], dtype=np.float32)
    thresholded = _mask(_run(MaskMorphology, mask=values, operation="threshold", threshold=0.5))
    empty = _mask(
        _run(
            MaskMorphology,
            mask=np.zeros((2, 4, 5), dtype=np.float32),
            operation="threshold",
            threshold=0.0,
        )
    )

    assert np.array_equal(thresholded, [[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])
    assert empty.shape == (2, 4, 5)
    assert not np.any(empty)


def test_remove_small_components_uses_area_boundary_and_preserves_values() -> None:
    source = np.zeros((1, 6, 8), dtype=np.float32)
    source[0, 1, 1:3] = [0.6, 0.8]
    source[0, 3:5, 4:7] = [[0.51, 0.7, 0.9], [1.0, 0.75, 0.55]]
    source[0, 0, 7] = 0.4
    original = source.copy()

    result = _run(
        MaskMorphology,
        mask=source,
        operation="remove_small_components",
        minimum_area=6,
        threshold=0.5,
        connectivity="4",
    )
    cleaned = np.asarray(result["mask"])

    assert np.array_equal(source, original)
    assert np.array_equal(cleaned[0, 1, 1:3], [0.0, 0.0])
    assert np.array_equal(cleaned[0, 3:5, 4:7], original[0, 3:5, 4:7])
    assert cleaned[0, 0, 7] == 0.4
    assert np.array_equal(np.asarray(result["inverse_mask"]), 1.0 - cleaned)


def test_remove_small_components_connectivity_is_per_frame_and_threshold_is_strict() -> None:
    source = np.zeros((2, 5, 5), dtype=np.float32)
    source[0, 1, 1] = 0.75
    source[0, 2, 2] = 0.9
    source[1, 1, 1:4] = [0.5, 0.6, 0.7]

    four_connected = _mask(
        _run(
            MaskMorphology,
            mask=source,
            operation="remove_small_components",
            minimum_area=2,
            threshold=0.5,
            connectivity="4",
        )
    )
    eight_connected = _mask(
        _run(
            MaskMorphology,
            mask=source,
            operation="remove_small_components",
            minimum_area=2,
            threshold=0.5,
            connectivity="8",
        )
    )

    assert not np.any(four_connected[0])
    assert np.array_equal(eight_connected[0], source[0])
    assert np.array_equal(four_connected[1], source[1])
    assert np.array_equal(eight_connected[1], source[1])


def test_mask_cleanup_handles_empty_full_holes_and_border_background() -> None:
    empty = np.zeros((1, 4, 5), dtype=np.float32)
    full = np.ones((1, 4, 5), dtype=np.float32)
    ring = np.ones((1, 5, 6), dtype=np.float32)
    ring[0, 2, 2:4] = 0.0
    ring[0, 0:3, 5] = 0.0

    assert not np.any(
        _mask(
            _run(
                MaskMorphology,
                mask=empty,
                operation="remove_small_components",
                minimum_area=1,
            )
        )
    )
    assert np.all(
        _mask(
            _run(
                MaskMorphology,
                mask=full,
                operation="remove_small_components",
                minimum_area=full.shape[1] * full.shape[2],
            )
        )
        == 1.0
    )
    filled = _mask(_run(MaskMorphology, mask=ring, operation="fill_holes"))
    assert np.all(filled[0, 2, 2:4] == 1.0)
    assert np.all(filled[0, 0:3, 5] == 0.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"minimum_area": 0}, "minimum_area"),
        ({"minimum_area": 1.5}, "minimum_area"),
        ({"connectivity": "6"}, "connectivity"),
        ({"threshold": float("nan")}, "threshold"),
    ],
)
def test_remove_small_components_rejects_invalid_controls(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        _run(
            MaskMorphology,
            mask=np.zeros((1, 2, 2), dtype=np.float32),
            operation="remove_small_components",
            **kwargs,
        )


def test_mask_cleanup_is_registered_and_round_trips_through_schema_wire() -> None:
    assert MaskMorphology in IMAGE_NODES
    schema = build_schemas(IMAGE_NODES)["dinkster.mask.morphology"]
    operation = schema.combos[0]
    cleanup = operation.option("remove_small_components")
    assert cleanup is not None
    assert tuple(item.id for item in cleanup.inputs) == (
        "minimum_area",
        "threshold",
        "connectivity",
    )
    minimum_area = cleanup.inputs[0]
    assert isinstance(minimum_area, InputSpec)
    assert minimum_area.default == 64
    connectivity = cleanup.inputs[2]
    assert isinstance(connectivity, InputSpec)
    assert connectivity.default == "8"
    assert connectivity.widget == ComboWidget(options=("4", "8"))
    assert operation.option("fill_holes") is not None

    wire = schema_to_wire(schema)
    restored = schema_from_wire(json.loads(json.dumps(wire)))
    assert schema_to_wire(restored) == wire
    restored_cleanup = restored.combos[0].option("remove_small_components")
    assert restored_cleanup is not None
    assert tuple(item.id for item in restored_cleanup.inputs) == (
        "minimum_area",
        "threshold",
        "connectivity",
    )


def test_grow_distinguishes_square_tapered_iterations_and_edge_policy() -> None:
    seed = np.zeros((1, 7, 7), dtype=np.float32)
    seed[0, 3, 3] = 1.0
    square = _mask(
        _run(
            MaskMorphology,
            mask=seed,
            operation="grow_erode",
            radius=1,
            iterations=1,
            tapered_corners=False,
        )
    )
    tapered = _mask(
        _run(
            MaskMorphology,
            mask=seed,
            operation="grow_erode",
            radius=1,
            iterations=1,
            tapered_corners=True,
        )
    )
    twice = _mask(
        _run(
            MaskMorphology,
            mask=seed,
            operation="grow_erode",
            radius=1,
            iterations=2,
            tapered_corners=False,
        )
    )

    assert square.sum() == 9.0
    assert tapered.sum() == 5.0
    assert twice.sum() == 25.0

    corner = np.zeros((1, 3, 3), dtype=np.float32)
    corner[0, 0, 0] = 1.0
    constant = _mask(
        _run(
            MaskMorphology,
            mask=corner,
            operation="grow_erode",
            radius=1,
            edge_policy="constant",
        )
    )
    wrap = _mask(
        _run(MaskMorphology, mask=corner, operation="grow_erode", radius=1, edge_policy="wrap")
    )
    assert constant.sum() == 3.0
    assert wrap.sum() == 5.0


def test_erode_open_close_and_kernel_size_have_distinct_effects() -> None:
    full = np.ones((1, 7, 7), dtype=np.float32)
    eroded = _mask(
        _run(
            MaskMorphology,
            mask=full,
            operation="grow_erode",
            radius=-1,
            edge_policy="constant",
        )
    )
    assert eroded.sum() == 25.0

    isolated = np.zeros((1, 7, 7), dtype=np.float32)
    isolated[0, 1, 1] = 1.0
    isolated[0, 4:6, 4:6] = 1.0
    opened = _mask(_run(MaskMorphology, mask=isolated, operation="open", kernel_size=3))
    assert not np.any(opened)

    hole = np.ones((1, 7, 7), dtype=np.float32)
    hole[0, 3, 3] = 0.0
    closed = _mask(
        _run(
            MaskMorphology,
            mask=hole,
            operation="close",
            kernel_size=3,
            edge_policy="replicate",
        )
    )
    assert np.all(closed == 1.0)


def test_grow_and_erode_preserve_order_for_random_batches() -> None:
    source = np.random.default_rng(671).random((3, 8, 9), dtype=np.float32)
    grown = _mask(
        _run(
            MaskMorphology,
            mask=source,
            operation="grow_erode",
            radius=3,
            tapered_corners=True,
            edge_policy="replicate",
        )
    )
    eroded = _mask(
        _run(
            MaskMorphology,
            mask=source,
            operation="grow_erode",
            radius=-3,
            tapered_corners=False,
            edge_policy="replicate",
        )
    )

    assert np.all(grown >= source)
    assert np.all(eroded <= source)
    assert np.array_equal(
        grown,
        _mask(
            _run(
                MaskMorphology,
                mask=source,
                operation="grow_erode",
                radius=3,
                tapered_corners=True,
                edge_policy="replicate",
            )
        ),
    )


def test_grow_blur_temporal_controls() -> None:
    source = np.zeros((3, 9, 9), dtype=np.float32)
    source[:, 4, 4] = 1.0
    expanded = _mask(
        _run(
            MaskMorphology,
            mask=source,
            operation="grow_blur",
            radius=0,
            incremental_expandrate=1.0,
            tapered_corners=True,
        )
    )
    blended_source = np.zeros((2, 5, 5), dtype=np.float32)
    blended_source[0, 2, 1] = 1.0
    blended_source[1, 2, 3] = 1.0
    blended = _mask(
        _run(
            MaskMorphology,
            mask=blended_source,
            operation="grow_blur",
            radius=0,
            lerp_alpha=0.5,
        )
    )
    ring = np.ones((1, 5, 5), dtype=np.float32)
    ring[:, 2, 2] = 0.0
    filled = _mask(
        _run(
            MaskMorphology,
            mask=ring,
            operation="grow_blur",
            radius=0,
            fill_holes=True,
        )
    )
    flipped_blurred = _mask(
        _run(
            MaskMorphology,
            mask=source[:1],
            operation="grow_blur",
            radius=0,
            flip_input=True,
            blur_amount=0.75,
        )
    )

    assert [float(frame.sum()) for frame in expanded] == [1.0, 5.0, 13.0]
    assert blended[1, 2, 1] == 0.5
    assert blended[1, 2, 3] == 0.5
    assert filled[0, 2, 2] == 1.0
    assert 0.0 < flipped_blurred[0, 4, 4] < 1.0
    assert flipped_blurred[0, 4, 3] > flipped_blurred[0, 4, 4]


@pytest.mark.parametrize(
    ("operation", "controls"),
    [
        ("grow_erode", {"radius": 4096, "iterations": 4096}),
        ("grow_blur", {"radius": 4096, "iterations": 4096}),
        ("open", {"kernel_size": 4097, "iterations": 2}),
        ("blur", {"radius": 4096}),
        ("round", {"radius": 4096, "iterations": 4096}),
    ],
)
def test_morphology_rejects_schema_valid_kernel_work_exhaustion(
    operation: str, controls: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="kernel work"):
        _run(
            MaskMorphology,
            mask=np.zeros((2, 128, 128), dtype=np.float32),
            operation=operation,
            **controls,
        )


def test_feather_blur_offset_remap_round_block_and_invert() -> None:
    full = np.ones((1, 5, 5), dtype=np.float32)
    feathered = _mask(
        _run(
            MaskMorphology,
            mask=full,
            operation="feather_edges",
            left=2,
            top=2,
            right=2,
            bottom=2,
        )
    )
    assert feathered[0, 0, 0] == 0.25
    assert feathered[0, 2, 2] == 1.0

    impulse = np.zeros((1, 5, 5), dtype=np.float32)
    impulse[0, 2, 2] = 1.0
    blurred = _mask(
        _run(
            MaskMorphology,
            mask=impulse,
            operation="blur",
            radius=2,
            sigma=1.25,
            edge_policy="reflect",
        )
    )
    assert 0.0 < blurred[0, 2, 2] < 1.0
    assert blurred[0, 2, 2] > blurred[0, 0, 0]
    assert np.array_equal(
        blurred,
        _mask(
            _run(
                MaskMorphology,
                mask=impulse,
                operation="blur",
                radius=2,
                sigma=1.25,
                edge_policy="reflect",
            )
        ),
    )

    offset = _mask(
        _run(
            MaskMorphology,
            mask=impulse,
            operation="offset",
            x=-2,
            y=1,
            edge_policy="constant",
        )
    )
    assert offset[0, 3, 0] == 1.0
    wrapped = _mask(
        _run(
            MaskMorphology,
            mask=impulse,
            operation="offset",
            x=3,
            y=0,
            edge_policy="wrap",
        )
    )
    assert wrapped[0, 2, 0] == 1.0

    ramp = np.array([[[0.0, 0.25, 0.5, 0.75, 1.0]]], dtype=np.float32)
    remapped = _mask(
        _run(
            MaskMorphology,
            mask=ramp,
            operation="remap",
            input_low=0.25,
            input_high=0.75,
            output_low=0.0,
            output_high=1.0,
        )
    )
    assert np.allclose(remapped, [[[0.0, 0.0, 0.5, 1.0, 1.0]]])
    rough = np.zeros((1, 7, 7), dtype=np.float32)
    rough[0, 2:5, 2:5] = 1.0
    rough[0, 1, 1] = 1.0
    rounded = _mask(_run(MaskMorphology, mask=rough, operation="round", radius=1))
    blocked = _mask(_run(MaskMorphology, mask=ramp, operation="block", kernel_size=2))
    inverted = _mask(_run(MaskMorphology, mask=ramp, operation="invert"))
    assert rounded[0, 1, 1] == 0.0
    assert rounded[0, 3, 3] == 1.0
    assert np.allclose(blocked, [[[0.125, 0.125, 0.625, 0.625, 1.0]]])
    assert np.array_equal(inverted, 1.0 - ramp)


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("multiply", [0.0, 0.0, 0.5, 0.5]),
        ("add", [0.0, 0.5, 1.0, 1.0]),
        ("subtract", [0.0, 0.0, 0.0, 0.5]),
        ("min", [0.0, 0.0, 0.5, 0.5]),
        ("max", [0.0, 0.5, 1.0, 1.0]),
        ("and", [0.0, 0.0, 0.0, 0.0]),
        ("or", [0.0, 0.0, 1.0, 1.0]),
        ("xor", [0.0, 0.0, 1.0, 1.0]),
    ],
)
def test_combine_modes(operation: str, expected: list[float]) -> None:
    destination = np.array([[[0.0, 0.0, 0.5, 1.0]]], dtype=np.float32)
    source = np.array([[[0.0, 0.5, 1.0, 0.5]]], dtype=np.float32)
    result = _mask(_run(MaskCombine, destination=destination, source=source, operation=operation))
    assert np.array_equal(result, np.array([[expected]], dtype=np.float32))


def test_combine_clips_signed_placement_and_preserves_fully_oob_destination() -> None:
    destination = np.zeros((1, 3, 4), dtype=np.float32)
    source = np.ones((1, 2, 3), dtype=np.float32)
    clipped = _mask(
        _run(MaskCombine, destination=destination, source=source, operation="add", x=-1, y=2)
    )
    oob = _mask(
        _run(MaskCombine, destination=destination, source=source, operation="add", x=20, y=20)
    )

    assert np.array_equal(clipped[0, 2], [1.0, 1.0, 0.0, 0.0])
    assert clipped.sum() == 2.0
    assert np.array_equal(oob, destination)


def test_combine_batch_policies_control_singleton_direction() -> None:
    destination = np.zeros((3, 2, 2), dtype=np.float32)
    source = np.ones((1, 2, 2), dtype=np.float32)
    broadcast = _mask(_run(MaskCombine, destination=destination, source=source, operation="max"))
    assert broadcast.shape == (3, 2, 2)
    assert np.all(broadcast == 1.0)

    paired = _mask(
        _run(
            MaskCombine,
            destination=np.zeros((2, 2, 2), dtype=np.float32),
            source=np.stack(
                [np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)]
            ),
            operation="max",
        )
    )
    assert paired[0].sum() == 0.0
    assert paired[1].sum() == 4.0

    with pytest.raises(ValueError, match="singleton_broadcast"):
        _run(
            MaskCombine,
            destination=np.zeros((2, 2, 2), dtype=np.float32),
            source=np.zeros((3, 2, 2), dtype=np.float32),
        )

    with pytest.raises(ValueError, match="source_singleton"):
        _run(
            MaskCombine,
            destination=np.zeros((1, 2, 2), dtype=np.float32),
            source=np.zeros((2, 2, 2), dtype=np.float32),
            batch_policy="source_singleton",
        )


def test_image_to_mask_channels_and_alpha_polarities() -> None:
    image = np.array([[[[0.1, 0.2, 0.3, 0.25]]]], dtype=np.float32)
    assert np.allclose(_mask(_run(ImageToMask, image=image, channel="red")), [[[0.1]]])
    assert np.allclose(_mask(_run(ImageToMask, image=image, channel="alpha")), [[[0.25]]])
    assert np.allclose(_mask(_run(ImageToMask, image=image, channel="transparency")), [[[0.75]]])

    rgb = image[..., :3]
    with pytest.raises(ValueError, match="alpha extraction"):
        _run(ImageToMask, image=rgb, channel="alpha")
    with pytest.raises(ValueError, match="transparency extraction"):
        _run(ImageToMask, image=rgb, channel="transparency")


def test_image_to_mask_exact_color_rounds_to_eight_bit() -> None:
    image = np.array([[[[128 / 255, 64 / 255, 1.0], [0.501, 0.249, 0.999]]]], dtype=np.float32)
    result = _mask(
        _run(
            ImageToMask,
            image=image,
            policy="exact_color",
            color="#8040ff",
        )
    )
    assert np.array_equal(result, [[[1.0, 0.0]]])


def test_image_to_mask_tolerance_metrics_are_explicit() -> None:
    image = np.array([[[[0.2, 0.0, 0.0, 0.0]]]], dtype=np.float32)
    max_channel = _mask(
        _run(
            ImageToMask,
            image=image,
            policy="tolerance_color",
            color="#000000ff",
            tolerance=0.15,
            metric="max_channel",
        )
    )
    rgb_rms = _mask(
        _run(
            ImageToMask,
            image=image,
            policy="tolerance_color",
            color="#000000ff",
            tolerance=0.15,
            metric="euclidean_rgb",
        )
    )
    rgba_rms = _mask(
        _run(
            ImageToMask,
            image=image,
            policy="tolerance_color",
            color="#000000ff",
            tolerance=0.15,
            metric="euclidean_rgba",
        )
    )
    assert max_channel.item() == 0.0
    assert rgb_rms.item() == 1.0
    assert rgba_rms.item() == 0.0


def test_image_to_mask_color_sources_and_invert() -> None:
    image = np.array(
        [[[[128 / 255, 64 / 255, 1.0], [130 / 255, 64 / 255, 1.0], [0.0, 0.0, 0.0]]]],
        dtype=np.float32,
    )
    from_hex = _mask(
        _run(ImageToMask, image=image, policy="exact_color", color_source="hex", color="#8040ff")
    )
    from_integer = _mask(
        _run(
            ImageToMask,
            image=image,
            policy="exact_color",
            color_source="integer",
            color_value=0x8040FF,
        )
    )
    from_channels = _mask(
        _run(
            ImageToMask,
            image=image,
            policy="tolerance_color",
            color_source="channels",
            red=128,
            green=64,
            blue=255,
            tolerance=2 / 255,
            metric="euclidean_rgb_sum",
        )
    )
    inverted = _mask(
        _run(
            ImageToMask,
            image=image,
            policy="exact_color",
            color_source="integer",
            color_value=0x8040FF,
            invert=True,
        )
    )

    assert np.array_equal(from_hex, [[[1.0, 0.0, 0.0]]])
    assert np.array_equal(from_integer, from_hex)
    assert np.array_equal(from_channels, [[[1.0, 1.0, 0.0]]])
    assert np.array_equal(inverted, 1.0 - from_hex)


def test_mask_to_image_shapes_channels_and_polarity() -> None:
    mask = np.array([[[0.25, 0.75]], [[1.0, 0.0]]], dtype=np.float32)
    rgb = np.asarray(_run(MaskToImage, mask=mask, channels="rgb")["image"])
    opacity = np.asarray(
        _run(MaskToImage, mask=mask, channels="rgba", mask_polarity="coverage")["image"]
    )
    transparency = np.asarray(
        _run(
            MaskToImage,
            mask=mask,
            channels="rgba",
            mask_polarity="transparency",
        )["image"]
    )

    assert rgb.shape == (2, 1, 2, 3)
    assert opacity.shape == (2, 1, 2, 4)
    assert np.array_equal(rgb[..., 0], mask)
    assert np.array_equal(opacity[..., 3], mask)
    assert np.array_equal(transparency[..., 3], 1.0 - mask)


def test_mask_info_bounds_and_region_mask_creation_clip_to_canvas() -> None:
    mask = np.zeros((2, 5, 6), dtype=np.float32)
    mask[0, 1:4, 2:5] = 1.0
    region = _run(MaskInfo, mask=mask, threshold=0.5)["bounds"]
    assert region == Region(x=2, y=1, width=3, height=3)

    made = _mask(
        _run(
            MakeMask,
            operation="region",
            region=Region(x=-2, y=2, width=4, height=5),
            width=5,
            height=4,
            batch_size=2,
            foreground=0.75,
        )
    )
    assert made.shape == (2, 4, 5)
    assert np.all(made[:, 2:, :2] == 0.75)
    assert made.sum() == 6.0


def test_mask_info_reports_stable_union_bounds_and_empty_bounds() -> None:
    mask = np.zeros((2, 5, 6), dtype=np.float32)
    mask[0, 1:3, 2:5] = 0.5
    mask[1, 4, 0] = 1.0
    info = _run(MaskInfo, mask=mask, threshold=0.5)
    returned_mask = info.pop("mask")

    assert info == {
        "width": 6,
        "height": 5,
        "count": 2,
        "area": 1,
        "bounds": Region(x=0, y=4, width=1, height=1),
    }
    assert np.array_equal(np.asarray(returned_mask), mask)

    empty = _run(MaskInfo, mask=np.zeros((3, 4, 5), dtype=np.float32), threshold=0.0)
    empty_mask = empty.pop("mask")
    assert empty == {
        "width": 5,
        "height": 4,
        "count": 3,
        "area": 0,
        "bounds": Region(x=0, y=0, width=0, height=0),
    }
    assert not np.any(np.asarray(empty_mask))


def test_mask_schema_ids_and_required_inputs_are_distinct() -> None:
    schemas = {node.define_schema().node_type: node.define_schema() for node in MASK_NODES}
    assert len(schemas) == len(MASK_NODES)
    assert {
        node_type: tuple(item.id for item in schema.inputs if item.required)
        for node_type, schema in schemas.items()
    } == {
        "dinkster.mask.make": (),
        "dinkster.mask.text": ("text",),
        "dinkster.mask.morphology": ("mask",),
        "dinkster.mask.combine": ("destination", "source"),
        "dinkster.image.to_mask": ("image",),
        "dinkster.mask.to_image": ("mask",),
        "dinkster.mask.info": ("mask",),
        "dinkster.mask.polarity": ("mask",),
    }


@pytest.mark.parametrize(
    ("node", "kwargs", "match"),
    [
        (MakeMask, {"operation": "solid", "width": 0, "height": 4}, "width"),
        (
            MaskMorphology,
            {"mask": np.zeros((2, 3), dtype=np.float32), "operation": "threshold"},
            "BHW",
        ),
        (
            MaskMorphology,
            {"mask": np.zeros((1, 2, 2), dtype=np.float32), "operation": "blur", "radius": -1},
            "blur radius",
        ),
        (
            MaskMorphology,
            {
                "mask": np.zeros((1, 2, 2), dtype=np.float32),
                "operation": "crop",
                "x": 2,
                "y": 0,
                "width": 1,
                "height": 1,
            },
            "overlap",
        ),
        (
            ImageToMask,
            {"image": np.zeros((1, 2, 2, 2), dtype=np.float32), "channel": "red"},
            "channels",
        ),
        (MakeMask, {"operation": "region", "region": "not a region"}, "Region"),
        (
            TextMask,
            {"text": "", "width": 8192, "height": 8192},
            "image operation limit",
        ),
    ],
)
def test_mask_nodes_reject_invalid_inputs(
    node: type, kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        _run(node, **kwargs)
