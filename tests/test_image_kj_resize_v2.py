from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from dinkster_image.geometry import ImageResize

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
