from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, cast

from dinkster_image._draft_initialization import IMAGE_NODES, MakeRegion, Region

ROOT = Path(__file__).parent.parent


def test_region_preserves_bounding_box_coordinates() -> None:
    region = cast("Region", MakeRegion.execute(x=7, y=11, width=13, height=17)["region"])
    assert region.to_record() == {"x": 7, "y": 11, "width": 13, "height": 17}


def test_image_pack_has_no_foundation_dependency() -> None:
    manifest = tomllib.loads(
        (ROOT / "packages" / "dinkster-nodes-image" / "dinkster-pack.toml").read_text(
            encoding="utf-8"
        )
    )
    assert "dinkster-nodes-foundation" not in manifest["pack"].get("dependencies", {})


def test_image_batch_resize_and_compare_execute() -> None:
    import numpy as np
    from dinkster_api.v1 import ABSENT, annotate_image, media_semantics
    from dinkster_image._draft_initialization import (
        ImageBatchCombine,
        ImageCompare,
        ImageResize,
    )

    rgb = np.full((1, 3, 5, 3), 0.25, np.float32)
    rgba = np.full((2, 6, 10, 4), 0.5, np.float32)
    batch = ImageBatchCombine.execute(
        images={"0.value": rgb, "1.value": rgba},
        shape_policy="resize_to_first",
        channel_policy="pad_with_one",
    )["image"]
    assert np.asarray(batch).shape == (3, 3, 5, 4)
    np.testing.assert_array_equal(np.asarray(batch)[0, ..., 3], 1.0)
    for pad in (0.0, 1.0):
        result = ImageResize.execute(
            image=rgb,
            width=4,
            height=4,
            mode="pad",
            fit_rounding="floor",
            interpolation="nearest-exact",
            pad_value=pad,
        )["image"]
        assert np.asarray(result).shape == (1, 4, 4, 3)
        np.testing.assert_array_equal(np.asarray(result)[:, (0, 3)], pad)
        np.testing.assert_array_equal(np.asarray(result)[:, 1:3], 0.25)
    for a, b in ((rgb, rgba), (rgb, None), (None, rgba), (None, None)):
        result = ImageCompare.execute(image_a=a, image_b=b)
        for name, expected in (("image_a", a), ("image_b", b)):
            if expected is None:
                assert result[name] is ABSENT
            else:
                np.testing.assert_array_equal(result[name], expected)
    premultiplied = annotate_image(rgba, alpha="premultiplied")
    preview = ImageCompare.execute(image_a=premultiplied, image_b=np.empty((0, 2, 3, 4)))
    assert media_semantics(preview["image_a"]) == media_semantics(premultiplied)
    np.testing.assert_array_equal(preview["image_a"], premultiplied)
    assert preview["image_b"] is ABSENT


def test_exact_image_compare_has_no_outputs_and_previews_each_nonempty_side() -> None:
    from collections.abc import Mapping
    from io import BytesIO

    import numpy as np
    from dinkster_image.compare import NativeImageCompare
    from dinkster_schema import schema_from_wire, schema_to_wire, use_reporter
    from PIL import Image

    schema = NativeImageCompare.schema()
    assert NativeImageCompare in IMAGE_NODES
    assert schema.node_type == "ImageCompare" and not schema.aliases
    assert [spec.id for spec in schema.inputs] == ["image_a", "image_b", "compare_view"]
    compare_view = schema.input("compare_view")
    assert compare_view is not None and compare_view.required
    assert schema.outputs == () and schema.output_node and schema.emits_previews
    assert schema_from_wire(schema_to_wire(schema)) == schema
    a = np.zeros((2, 2, 3, 3), dtype=np.float32)
    a[0, ..., 0] = 0.03
    a[1, ..., 1] = 0.61
    b = np.zeros((1, 3, 1, 3), dtype=np.float32)
    b[..., 2] = 0.91
    reports: list[tuple[Mapping[str, object], bytes | None]] = []

    def capture(name: str, data: Mapping[str, object], blob: bytes | None) -> None:
        assert name == "preview"
        reports.append((data, blob))

    with use_reporter(capture):
        assert NativeImageCompare.execute(image_a=a, image_b=b) == {}
        assert NativeImageCompare.execute(image_a=np.empty((0, 2, 3, 3))) == {}
    assert [(data["stream"], data["frameIndex"], data["frameCount"]) for data, _ in reports] == [
        ("image_a", 0, 2),
        ("image_a", 1, 2),
        ("image_b", 0, 1),
    ]
    for (_, blob), pixels in zip(reports, (a[0], a[1], b[0]), strict=True):
        assert blob is not None
        with Image.open(BytesIO(blob)) as decoded:
            np.testing.assert_array_equal(np.array(decoded), (pixels * 255).astype(np.uint8))


def test_image_batch_and_resize_match_pinned_goldens() -> None:
    import numpy as np
    from dinkster_image._draft_initialization import ImageBatchCombine, ImageResize

    fixture = json.loads((ROOT / "tests/goldens/image_aliases_b78cec87.json").read_text())
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
