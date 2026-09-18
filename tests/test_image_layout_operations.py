from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from dinkster_nodes_image.layout import ImageGridCompose, ImageGridDecompose, ImageStitch
from dinkster_schema import schema_to_wire

GOLDEN = cast(
    "dict[str, object]",
    json.loads(
        (Path(__file__).parent / "goldens" / "image_mask_e20d433a.json").read_text(encoding="utf-8")
    ),
)


def _golden_record(record: object) -> np.ndarray:
    value = cast("dict[str, object]", record)
    return np.asarray(cast("list[float]", value["values"]), dtype=np.float32).reshape(
        cast("list[int]", value["shape"])
    )


def _image(values: list[float], *, height: int = 1, width: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape((-1, 1, 1, 1))
    return array.repeat(height, axis=1).repeat(width, axis=2)


def test_stitch_spacing_color_is_conditional_on_nonzero_width() -> None:
    schema = ImageStitch.schema()
    assert schema.inputs[3].id == "spacing_width"
    group = schema.widget_groups[0]
    assert group.input == "spacing_width"
    assert group.values == tuple(range(1, 1025))
    assert group.members == ("spacing_color",)
    wire_groups = cast("list[dict[str, object]]", schema_to_wire(schema)["widgetGroups"])
    assert len(cast("list[object]", wire_groups[0]["values"])) == 1024


def test_grid_compose_uses_row_major_dynamic_inputs() -> None:
    output = ImageGridCompose.execute(
        images={str(index): _image([value]) for index, value in enumerate((1, 2, 3, 4))},
        columns=2,
    )["image"]
    np.testing.assert_array_equal(np.asarray(output)[0, :, :, 0], [[1, 2], [3, 4]])


def test_grid_compose_preserves_batches_and_requires_compatible_rows() -> None:
    first = _image([1, 2], height=2, width=3)
    second = _image([3, 4], height=2, width=3)
    output = np.asarray(ImageGridCompose.execute(images={"a": first, "b": second})["image"])
    assert output.shape == (2, 2, 6, 1)
    np.testing.assert_array_equal(output[:, :, :3], first)
    np.testing.assert_array_equal(output[:, :, 3:], second)
    with pytest.raises(ValueError, match="identical heights"):
        ImageGridCompose.execute(images={"a": first, "b": _image([3, 4], height=3)})

    varied = ImageGridCompose.execute(
        images={
            "a": _image([1], height=2, width=1),
            "b": _image([2], height=2, width=3),
            "c": _image([3], height=3, width=2),
            "d": _image([4], height=3, width=2),
        },
        columns=2,
    )["image"]
    assert np.asarray(varied).shape == (1, 5, 4, 1)


def test_grid_decompose_matches_kj_crop_and_batch_order() -> None:
    source = np.arange(2 * 5 * 7, dtype=np.float32).reshape((2, 5, 7, 1))
    output = np.asarray(ImageGridDecompose.execute(image=source, columns=3, rows=2)["image"])
    assert output.shape == (12, 2, 2, 1)
    np.testing.assert_array_equal(output[0], source[0, :2, :2])
    np.testing.assert_array_equal(output[5], source[0, 2:4, 4:6])
    np.testing.assert_array_equal(output[6], source[1, :2, :2])


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        ("right", [0.25, 0, 0, 0.75]),
        ("left", [0.75, 0, 0, 0.25]),
    ],
)
def test_horizontal_stitch_direction_and_even_spacing(
    direction: str, expected: list[float]
) -> None:
    output = ImageStitch.execute(
        first=_image([0.25]),
        second=_image([0.75]),
        direction=direction,
        spacing_width=1,
        spacing_color="black",
    )["image"]
    np.testing.assert_allclose(np.asarray(output)[0, 0, :, 0], expected, rtol=0, atol=1 / 255)


def test_stitch_extends_short_batches_and_matches_second_image_size() -> None:
    first = _image([0.1], height=4, width=2)
    second = _image([0.2, 0.3], height=2, width=3)
    output = np.asarray(ImageStitch.execute(first=first, second=second)["image"])
    assert output.shape == (2, 4, 8, 1)
    np.testing.assert_array_equal(output[:, :, :2], np.repeat(first, 2, axis=0))
    np.testing.assert_allclose(output[0, :, 2:], 0.2, rtol=0, atol=1 / 255)
    np.testing.assert_allclose(output[1, :, 2:], 0.3, rtol=0, atol=1 / 255)


def test_stitch_padding_and_channel_extension_match_core_contract() -> None:
    first = np.full((1, 2, 1, 1), 0.25, dtype=np.float32)
    second = np.ones((1, 5, 1, 3), dtype=np.float32)
    output = np.asarray(
        ImageStitch.execute(
            first=first,
            second=second,
            match_image_size=False,
            spacing_color="white",
        )["image"]
    )
    assert output.shape == (1, 5, 2, 3)
    np.testing.assert_array_equal(output[0, 1:3, 0, 0], 0.25)
    np.testing.assert_array_equal(output[0, :, 1], 1)


def test_stitch_without_second_image_is_identity() -> None:
    source = _image([1, 2], height=3, width=4)
    np.testing.assert_array_equal(ImageStitch.execute(first=source)["image"], source)


def test_image_stitch_matches_pinned_comfy_golden() -> None:
    source = _golden_record(GOLDEN["sourceBatchImage"])
    second = _golden_record(GOLDEN["secondBatchImage"])
    actual = ImageStitch.execute(
        first=source[:1],
        second=second[:, :, :1, :],
        direction="right",
        match_image_size=False,
        spacing_width=2,
        spacing_color="red",
    )["image"]
    cases = cast("dict[str, object]", GOLDEN["cases"])
    np.testing.assert_array_equal(actual, _golden_record(cases["ImageStitch"]))
