from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from dinkster_nodes_image.batch import (
    BATCH_NODES,
    ImageBatchCombine,
    ImageBatchEdit,
    ImageBatchToList,
    ImageListToBatch,
    ImageRebatch,
    MaskBatchCombine,
    MaskBatchEdit,
)

GOLDEN = cast(
    "dict[str, object]",
    json.loads(
        (Path(__file__).parent / "goldens" / "image_mask_e20d433a.json").read_text(encoding="utf-8")
    ),
)
GOLDEN_CASES = cast("dict[str, object]", GOLDEN["cases"])


def _golden_record(record: object) -> np.ndarray:
    value = cast("dict[str, object]", record)
    return np.asarray(cast("list[float]", value["values"]), dtype=np.float32).reshape(
        cast("list[int]", value["shape"])
    )


def _golden(name: str) -> np.ndarray:
    return _golden_record(GOLDEN_CASES[name])


def _image(values: list[float], *, height: int = 1, width: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape((-1, 1, 1, 1))
    return array.repeat(height, axis=1).repeat(width, axis=2)


def _mask(values: list[float], *, height: int = 1, width: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape((-1, 1, 1))
    return array.repeat(height, axis=1).repeat(width, axis=2)


def _frames(value: object) -> list[float]:
    array = np.asarray(value)
    return [float(item) for item in array[:, 0, 0].reshape(-1)]


def test_batch_edit_consolidates_core_and_ecosystem_order_operations() -> None:
    image = _image([0, 1, 2, 3])
    assert _frames(
        ImageBatchEdit.execute(image=image, operation="range", start=-2, count=10)["image"]
    ) == [2, 3]
    assert _frames(
        ImageBatchEdit.execute(image=image, operation="repeat_all", amount=2)["image"]
    ) == [0, 1, 2, 3, 0, 1, 2, 3]
    assert _frames(
        ImageBatchEdit.execute(image=image, operation="repeat_each", amount=2)["image"]
    ) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert _frames(ImageBatchEdit.execute(image=image, operation="reverse")["image"]) == [
        3,
        2,
        1,
        0,
    ]

    first = ImageBatchEdit.execute(image=image, operation="shuffle", seed=17)["image"]
    second = ImageBatchEdit.execute(image=image, operation="shuffle", seed=17)["image"]
    np.testing.assert_array_equal(first, second)
    assert sorted(_frames(first)) == [0, 1, 2, 3]

    assert _frames(ImageBatchEdit.execute(image=image, operation="pingpong")["image"]) == [
        0,
        1,
        2,
        3,
        2,
        1,
    ]
    assert _frames(ImageBatchEdit.execute(image=_image([0, 1]), operation="pingpong")["image"]) == [
        0,
        1,
    ]
    assert _frames(ImageBatchEdit.execute(image=_image([0]), operation="pingpong")["image"]) == [0]


@pytest.mark.parametrize(
    ("method", "size", "expected"),
    [
        ("expand", 3, [0, 2, 3]),
        ("expand", 6, [0, 1, 1, 2, 3, 3]),
        ("repeat_all", 6, [0, 1, 2, 3, 0, 1]),
        ("repeat_first", 6, [0, 0, 0, 1, 2, 3]),
        ("repeat_last", 6, [0, 1, 2, 3, 3, 3]),
    ],
)
def test_batch_resize_count_matches_essentials_expansion(
    method: str, size: int, expected: list[float]
) -> None:
    output = ImageBatchEdit.execute(
        image=_image([0, 1, 2, 3]),
        operation="resize_count",
        expansion=method,
        size=size,
    )["image"]
    assert _frames(output) == expected


def test_batch_edit_range_policy_and_shuffle_do_not_leak_state() -> None:
    image = _image([0, 1, 2])
    with pytest.raises(ValueError, match="exceeds batch size"):
        ImageBatchEdit.execute(
            image=image,
            operation="range",
            start=2,
            count=2,
            range_end="error",
        )
    np.random.seed(1234)
    expected = np.random.random(4)
    np.random.seed(1234)
    ImageBatchEdit.execute(image=image, operation="shuffle", seed=99)
    np.testing.assert_array_equal(np.random.random(4), expected)


@pytest.mark.parametrize(
    ("seed", "expected"),
    [
        (0, [4, 1, 7, 5, 3, 9, 0, 8, 6, 2]),
        (1, [5, 6, 1, 2, 0, 8, 9, 3, 7, 4]),
        (42, [2, 6, 1, 8, 4, 5, 0, 9, 3, 7]),
        ((1 << 32) + 1, [5, 6, 1, 2, 0, 8, 9, 3, 7, 4]),
        ((1 << 53) - 1, [1, 4, 6, 2, 0, 5, 8, 7, 9, 3]),
    ],
)
def test_shuffle_matches_pytorch_cpu_randperm(seed: int, expected: list[float]) -> None:
    output = ImageBatchEdit.execute(
        image=_image([float(index) for index in range(10)]),
        operation="shuffle",
        seed=seed,
    )["image"]
    assert _frames(output) == expected


def test_kj_repeat_interleaving_preserves_or_generates_masks() -> None:
    image = _image([1, 2], height=2, width=3)
    generated = ImageBatchEdit.execute(
        image=image,
        operation="repeat_each",
        amount=3,
        generate_repeat_marker=True,
    )
    assert _frames(generated["image"]) == [1, 1, 1, 2, 2, 2]
    assert _frames(generated["mask"]) == [1, 0, 0, 1, 0, 0]
    assert np.asarray(generated["mask"]).shape == (6, 2, 3)

    repeated = ImageBatchEdit.execute(
        image=image,
        mask=_mask([0.25, 0.75], height=2, width=3),
        operation="repeat_each",
        amount=2,
        generate_repeat_marker=True,
    )
    assert _frames(repeated["mask"]) == [0.25, 0.25, 0.75, 0.75]


def test_image_batch_combine_supports_dynamic_concat_insert_and_replace() -> None:
    original = _image([1, 2, 3])
    replacement = _image([8, 9])
    assert _frames(
        ImageBatchCombine.execute(images={"a": original, "b": replacement})["image"]
    ) == [1, 2, 3, 8, 9]
    assert _frames(
        ImageBatchCombine.execute(
            images={"a": original, "b": replacement},
            operation="insert_indexed",
            indexes="1, 4",
        )["image"]
    ) == [1, 8, 2, 3, 9]
    assert _frames(
        ImageBatchCombine.execute(
            images={"a": original, "b": replacement},
            operation="replace_indexed",
            indexes="0, -1",
        )["image"]
    ) == [8, 2, 9]
    assert _frames(
        ImageBatchCombine.execute(
            images={"a": original, "b": replacement},
            operation="replace_range",
            index=1,
        )["image"]
    ) == [1, 8, 9]


def test_image_batch_combine_has_explicit_spatial_and_channel_policies() -> None:
    small = _image([1], height=2, width=2)
    large = _image([2], height=4, width=4)
    with pytest.raises(ValueError, match="dimensions must match"):
        ImageBatchCombine.execute(images={"small": small, "large": large})
    resized = ImageBatchCombine.execute(
        images={"small": small, "large": large},
        shape_policy="resize_to_first",
        interpolation="nearest-exact",
    )["image"]
    assert np.asarray(resized).shape == (2, 2, 2, 1)

    rgb = np.ones((1, 2, 2, 3), dtype=np.float32)
    padded = ImageBatchCombine.execute(
        images={"gray": small, "rgb": rgb},
        channel_policy="pad_with_one",
    )["image"]
    assert np.asarray(padded).shape == (2, 2, 2, 3)
    np.testing.assert_array_equal(np.asarray(padded)[0], 1)


def test_kj_dynamic_batch_count_preserves_late_and_missing_inputs() -> None:
    output = ImageBatchCombine.execute(
        images={
            "1": _image([1, 2]),
            "3": _image([8]),
            "5": _image([99]),
        },
        input_count=4,
        shape_policy="resize_to_first",
        channel_policy="pad_with_one",
    )["image"]
    assert _frames(output) == [1, 2, 0, 0, 8, 0, 0]


def test_replace_image_and_mask_pairs_preserves_both_output_contracts() -> None:
    images = ImageBatchCombine.execute(
        images={"1": _image([1, 2, 3]), "2": _image([8, 9])},
        masks={"1": _mask([0.1, 0.2, 0.3]), "2": _mask([0.8, 0.9])},
        operation="replace_range",
        index=1,
    )
    assert _frames(images["image"]) == [1, 8, 9]
    assert _frames(images["mask"]) == pytest.approx([0.1, 0.8, 0.9])

    masks = MaskBatchCombine.execute(
        masks={"1": _mask([0.1, 0.2, 0.3]), "2": _mask([0.8, 0.9])},
        operation="replace_range",
        index=1,
    )["mask"]
    assert _frames(masks) == pytest.approx([0.1, 0.8, 0.9])

    fallback = ImageBatchCombine.execute(
        images={"1": _image([1, 2]), "2": _image([8])},
        operation="replace_range",
        index=1,
        mask_fallback="zero_64",
    )["mask"]
    np.testing.assert_array_equal(fallback, np.zeros((1, 64, 64), dtype=np.float32))


def test_mask_batch_edit_and_combine_share_batch_order_contracts() -> None:
    first, second = _mask([1, 2]), _mask([8, 9])
    combined = MaskBatchCombine.execute(masks={"first": first, "second": second})["mask"]
    assert _frames(combined) == [1, 2, 8, 9]
    repeated = MaskBatchEdit.execute(mask=combined, operation="repeat_each", amount=2)["mask"]
    assert _frames(repeated) == [1, 1, 2, 2, 8, 8, 9, 9]
    looped = MaskBatchEdit.execute(mask=combined, operation="pingpong")["mask"]
    assert _frames(looped) == [1, 2, 8, 9, 8, 2]


def test_image_list_conversions_and_rebatch_preserve_frame_order() -> None:
    image = _image([1, 2, 3], height=2, width=2)
    values = ImageBatchToList.execute(image=image)["images"]
    assert isinstance(values, list)
    assert [np.asarray(item).shape for item in values] == [(1, 2, 2, 1)] * 3
    np.testing.assert_array_equal(ImageListToBatch.execute(images=values)["image"], image)

    source = [_image([0, 1]), _image([2, 3, 4])]
    rebatched = ImageRebatch.execute(images=source, batch_size=2)["images"]
    assert isinstance(rebatched, list)
    assert [_frames(item) for item in rebatched] == [[0, 1], [2, 3], [4]]


def test_image_batch_schema_ids_are_unique_and_dynamic_inputs_are_bounded() -> None:
    schemas = {node.define_schema().node_type: node.define_schema() for node in BATCH_NODES}
    assert len(schemas) == len(BATCH_NODES) == 7
    image_families = schemas["dinkster.image.batch.combine"].input_families
    assert (image_families[0].min_members, image_families[0].max_members) == (1, 1000)
    assert (image_families[1].min_members, image_families[1].max_members) == (0, 2)
    assert schemas["dinkster.mask.batch.combine"].input_families[0].min_members == 2


def test_core_batch_operations_match_pinned_comfy_goldens() -> None:
    source = _golden_record(GOLDEN["sourceBatchImage"])
    second = _golden_record(GOLDEN["secondBatchImage"])
    np.testing.assert_array_equal(
        ImageBatchEdit.execute(image=source, operation="repeat_all", amount=2)["image"],
        _golden("RepeatImageBatch"),
    )
    np.testing.assert_array_equal(
        ImageBatchEdit.execute(image=source, operation="range", start=-2, count=2)["image"],
        _golden("ImageFromBatch"),
    )
    np.testing.assert_array_equal(
        ImageBatchCombine.execute(
            images={"first": source[:1], "second": second},
            channel_policy="pad_with_one",
        )["image"],
        _golden("ImageBatch"),
    )
    result = ImageRebatch.execute(images=[source[:2], source[2:]], batch_size=2)["images"]
    assert isinstance(result, list)
    record = cast("dict[str, object]", GOLDEN_CASES["RebatchImages"])
    expected = [_golden_record(item) for item in cast("list[object]", record["items"])]
    assert len(result) == len(expected)
    for actual, golden in zip(result, expected, strict=True):
        np.testing.assert_array_equal(actual, golden)


def test_dynamic_batch_constructs_and_materialized_execution() -> None:
    expected = {
        ImageBatchEdit: ("operation",),
        MaskBatchEdit: ("operation",),
        ImageBatchCombine: ("operation", "shape_policy"),
        MaskBatchCombine: ("operation", "shape_policy"),
        ImageListToBatch: ("shape_policy",),
    }
    for node, combo_ids in expected.items():
        schema = node.define_schema()
        assert schema.version == 2
        assert tuple(combo.id for combo in schema.combos) == combo_ids

    image = _image([1, 2])
    mask = _mask([1, 2])
    np.testing.assert_array_equal(
        cast("Any", ImageBatchEdit.execute)(
            image=image, operation="range", **{"operation.start": 1}
        )["image"],
        ImageBatchEdit.execute(image=image, operation="range", start=1)["image"],
    )
    np.testing.assert_array_equal(
        cast("Any", MaskBatchEdit.execute)(mask=mask, operation="range", **{"operation.start": 1})[
            "mask"
        ],
        MaskBatchEdit.execute(mask=mask, operation="range", start=1)["mask"],
    )
    images = {"a": image[:1], "b": image[1:]}
    np.testing.assert_array_equal(
        cast("Any", ImageBatchCombine.execute)(
            images=images, operation="replace_range", **{"operation.index": 0}
        )["image"],
        ImageBatchCombine.execute(images=images, operation="replace_range", index=0)["image"],
    )
    masks = {"a": mask[:1], "b": mask[1:]}
    np.testing.assert_array_equal(
        cast("Any", MaskBatchCombine.execute)(
            masks=masks, operation="replace_range", **{"operation.index": 0}
        )["mask"],
        MaskBatchCombine.execute(masks=masks, operation="replace_range", index=0)["mask"],
    )
    np.testing.assert_array_equal(
        cast("Any", ImageListToBatch.execute)(
            images=[image],
            shape_policy="resize_to_first",
            **{"shape_policy.interpolation": "nearest-exact"},
        )["image"],
        ImageListToBatch.execute(
            images=[image], shape_policy="resize_to_first", interpolation="nearest-exact"
        )["image"],
    )
