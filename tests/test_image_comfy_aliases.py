from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from dinkster_api.v1 import ABSENT, annotate_image, media_semantics
from dinkster_nodes_image import ImageBatchCombine, ImageCompare, ImageResize


def test_native_image_batch_and_compare_preserve_alpha_semantics() -> None:
    rgb = np.full((1, 3, 5, 3), 0.25, np.float32)
    rgba = np.full((2, 6, 10, 4), 0.5, np.float32)
    batch = ImageBatchCombine.execute(
        images={"0.value": rgb, "1.value": rgba},
        shape_policy="resize_to_first",
        channel_policy="pad_with_one",
    )["image"]
    assert np.asarray(batch).shape == (3, 3, 5, 4)
    np.testing.assert_array_equal(np.asarray(batch)[0, ..., 3], 1.0)

    for image_a, image_b in ((rgb, rgba), (rgb, None), (None, rgba), (None, None)):
        result = ImageCompare.execute(image_a=image_a, image_b=image_b)
        for name, expected in (("image_a", image_a), ("image_b", image_b)):
            if expected is None:
                assert result[name] is ABSENT
            else:
                np.testing.assert_array_equal(result[name], expected)

    premultiplied = annotate_image(rgba, alpha="premultiplied")
    preview = ImageCompare.execute(image_a=premultiplied, image_b=np.empty((0, 2, 3, 4)))
    assert media_semantics(preview["image_a"]) == media_semantics(premultiplied)
    np.testing.assert_array_equal(preview["image_a"], premultiplied)
    assert preview["image_b"] is ABSENT


def test_native_image_batch_and_resize_match_pinned_goldens() -> None:
    fixture = json.loads(
        (Path(__file__).parent / "goldens" / "image_aliases_b78cec87.json").read_text(
            encoding="utf-8"
        )
    )
    assert fixture["baseline"] == "b78cec879b9460d5cb25228a83a942fb78d2cd24"

    def array(value: dict[str, Any]) -> np.ndarray:
        return np.array(value["values"], np.float32).reshape(value["shape"])

    batch = ImageBatchCombine.execute(
        images={"1": array(fixture["rgb"]), "2": array(fixture["rgba"])},
        shape_policy="resize_to_first",
        channel_policy="pad_with_one",
    )["image"]
    np.testing.assert_allclose(np.asarray(batch), array(fixture["batch"]), rtol=0, atol=1e-7)
    for case in fixture["resize"]:
        result = ImageResize.execute(
            image=array(fixture["rgb"]),
            width=4,
            height=4,
            mode="pad",
            fit_rounding="floor",
            interpolation=case["method"],
            pad_value=float(case["color"] == "white"),
        )["image"]
        if case["method"] in ("nearest-exact", "lanczos"):
            np.testing.assert_array_equal(result, array(case["image"]))
        else:
            np.testing.assert_allclose(np.asarray(result), array(case["image"]), rtol=0, atol=1e-7)
