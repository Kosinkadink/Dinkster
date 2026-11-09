from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from dinkster_api.v1 import ABSENT
from dinkster_nodes_image import ImageResize

GOLDEN_PATH = Path(__file__).parent / "goldens" / "kj_resize_v2_827fe6ee.json"
GOLDEN_BYTES = GOLDEN_PATH.read_bytes()
GOLDEN = cast("dict[str, Any]", json.loads(GOLDEN_BYTES))

MODE_MAP = {
    "stretch": "stretch",
    "resize": "fit",
    "pad": "pad",
    "pad_edge": "pad",
    "pad_edge_pixel": "pad",
    "crop": "fill",
    "pillarbox_blur": "pad",
}
PADDING_MAP = {
    "pad": "constant",
    "pad_edge": "edge_average",
    "pad_edge_pixel": "edge_pixel",
    "pillarbox_blur": "blurred_background",
}


def _array(record: dict[str, Any]) -> np.ndarray:
    return np.asarray(record["values"], dtype=np.float32).reshape(record["shape"])


def test_kj_resize_v2_golden_has_exact_pins_and_cases() -> None:
    assert hashlib.sha256(GOLDEN_BYTES).hexdigest() == (
        "9c32d4238965b67ddbfc1e87ad6c571e5ec60d8c3e356f83decaba8111ee6452"
    )
    assert GOLDEN["baselines"] == {
        "comfyui": "b78cec879b9460d5cb25228a83a942fb78d2cd24",
        "comfyui-kjnodes": "827fe6ee0ed7348d8daa988ed852bedf1272380c",
    }
    assert set(GOLDEN["cases"]) == {
        "stretch_landscape",
        "resize_portrait",
        "crop_landscape",
        "pad_rgb_landscape",
        "pad_rgb_portrait",
        "pad_edge_landscape",
        "pad_edge_bicubic_landscape",
        "pad_edge_pixel_portrait",
        "pillarbox_blur_landscape",
        "total_pixels_landscape",
        "total_pixels_portrait",
    }


def test_kj_resize_v2_modes_map_to_natural_resize_intents() -> None:
    sources = {name: _array(record) for name, record in GOLDEN["sources"].items()}
    masks = {name: _array(record) for name, record in GOLDEN["masks"].items()}
    for case in GOLDEN["cases"].values():
        mask_name = cast("str | None", case["mask"])
        mode = cast("str", case["mode"])
        total_pixels = mode == "total_pixels"
        result = ImageResize.execute(
            image=sources[case["source"]],
            mask=None if mask_name is None else masks[mask_name],
            width=case["width"],
            height=case["height"],
            target="total_pixels" if total_pixels else "dimensions",
            megapixels=(case["width"] * case["height"] / (1024 * 1024)),
            mode="stretch" if total_pixels else MODE_MAP[mode],
            mode_padding=PADDING_MAP.get(mode, "constant"),
            interpolation=case["interpolation"],
            divisibility="crop",
            multiple_of=case["divisible_by"],
            pad_color=case["pad_color"],
            mode_anchor="center" if case["anchor"] == "disabled" else case["anchor"],
        )
        actual_image = np.asarray(result["image"])
        assert actual_image.shape[1] > 0 and actual_image.shape[2] > 0
        multiple = case["divisible_by"]
        if multiple > 1 and min(actual_image.shape[1:3]) >= multiple:
            assert actual_image.shape[1] % multiple == actual_image.shape[2] % multiple == 0
        if mask_name is not None or mode in PADDING_MAP:
            assert np.asarray(result["mask"]).shape == actual_image.shape[:3]
        else:
            assert result["mask"] is ABSENT


def test_kj_resize_v2_invalid_pad_color_falls_back_to_black() -> None:
    image = np.ones((1, 2, 4, 3), dtype=np.float32)
    arguments = {
        "image": image,
        "width": 8,
        "height": 8,
        "mode": "pad",
        "interpolation": "nearest-exact",
        "divisibility": "crop",
        "multiple_of": 1,
        "mode_anchor": "center",
    }

    invalid = ImageResize.execute(**arguments, pad_color="not a color")
    black = ImageResize.execute(**arguments, pad_color="0, 0, 0")

    np.testing.assert_array_equal(invalid["image"], black["image"])
    np.testing.assert_array_equal(invalid["mask"], black["mask"])


def test_kj_resize_v2_missing_mask_is_absent_without_padding() -> None:
    result = ImageResize.execute(
        image=np.zeros((1, 3, 5, 3), dtype=np.float32),
        width=10,
        height=6,
        mode="stretch",
        interpolation="nearest-exact",
        divisibility="crop",
        multiple_of=1,
    )

    assert result["mask"] is ABSENT


@pytest.mark.parametrize(
    ("color", "expected"),
    [
        ("red", [1, 0, 0, 1]),
        ("#ff0000", [1, 0, 0, 1]),
        ("1,0,0", [1, 0, 0, 1]),
        ("0.5", [127 / 255, 127 / 255, 127 / 255, 1]),
        ("#ff000080", [1, 0, 0, 128 / 255]),
        ("1,0,0,0", [1, 0, 0, 0]),
    ],
)
def test_rgba_padding_defaults_to_opaque_alpha(color: str, expected: list[float]) -> None:
    image = np.full((1, 2, 4, 4), 0.25, dtype=np.float32)
    result = ImageResize.execute(
        image=image,
        width=8,
        height=8,
        mode="pad",
        pad_color=color,
        interpolation="nearest-exact",
        divisibility="crop",
        multiple_of=1,
        mode_anchor="center",
    )
    output = np.asarray(result["image"])
    assert output.shape == (1, 8, 8, 4)
    np.testing.assert_array_equal(output[0, 0, 0], np.asarray(expected, dtype=np.float32))
    np.testing.assert_array_equal(output[:, 2:6], 0.25)


def test_kj_resize_v2_large_batch_keeps_generated_padding_mask() -> None:
    result = ImageResize.execute(
        image=np.zeros((65, 2, 4, 3), dtype=np.float32),
        width=8,
        height=8,
        mode="pad",
        interpolation="nearest-exact",
        divisibility="crop",
        multiple_of=1,
        mode_anchor="center",
    )

    assert np.asarray(result["image"]).shape == (65, 8, 8, 3)
    output_mask = np.asarray(result["mask"])
    assert output_mask.shape == (65, 8, 8)
    np.testing.assert_array_equal(output_mask[:, :2], 1.0)
    np.testing.assert_array_equal(output_mask[:, 3:5], 0.0)
