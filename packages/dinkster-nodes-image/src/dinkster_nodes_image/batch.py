"""Deterministic image and mask batch operations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

import numpy as np
from dinkster_api.v1 import (
    ABSENT,
    CORE_STRING,
    DynamicComboOption,
    DynamicComboSpec,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)

from .geometry import BOOLEAN, IMAGE, INT, MASK
from .migration import with_v1_migration
from .support import (
    INTERPOLATION_OPTIONS,
    MAX_DIMENSION,
    Interpolation,
    check_output_size,
    combo_input,
    image_array,
    indexed_family_values,
    mask_array,
    materialized_inputs,
    number_input,
    resize_to_fill,
)

IMAGE_LIST = TypeExpr.list_of(IMAGE)
STRING = TypeExpr.concrete(CORE_STRING)

_EDIT_OPERATIONS = (
    "range",
    "repeat_all",
    "repeat_each",
    "resize_count",
    "reverse",
    "shuffle",
    "pingpong",
)
_EXPANSION_METHODS = ("expand", "repeat_all", "repeat_first", "repeat_last")
_COMBINE_OPERATIONS = ("concat", "insert_indexed", "replace_indexed", "replace_range")
_MASK_COMBINE_OPERATIONS = ("concat", "replace_range")


def _check_batch_output_size(array: np.ndarray, count: int) -> None:
    shape = (count, *array.shape[1:])
    checked_shape = shape if array.ndim == 4 else (*shape, 1)
    check_output_size(cast("tuple[int, ...]", checked_shape))


def _edit_inputs(value_id: str, value_type: TypeExpr) -> tuple[InputSpec, ...]:
    return (
        InputSpec(value_id, value_type),
        combo_input("operation", _EDIT_OPERATIONS, "range"),
        number_input("start", INT, 0, minimum=-MAX_DIMENSION, maximum=MAX_DIMENSION, step=1),
        number_input("count", INT, 1, minimum=1, maximum=MAX_DIMENSION, step=1),
        number_input("amount", INT, 1, minimum=1, maximum=4096, step=1),
        number_input("size", INT, 1, minimum=1, maximum=4096, step=1),
        combo_input("expansion", _EXPANSION_METHODS, "expand"),
        number_input("seed", INT, 0, minimum=0, maximum=(1 << 53) - 1, step=1),
        combo_input("range_end", ("clamp", "error"), "clamp", advanced=True),
    )


def _resize_count(array: np.ndarray, size: int, method: str) -> np.ndarray:
    if size < 1:
        raise ValueError(f"size must be positive, got {size}")
    if method not in _EXPANSION_METHODS:
        raise ValueError(f"unknown expansion method: {method}")
    count = int(array.shape[0])
    if count == size:
        return array
    _check_batch_output_size(array, size)
    if size == 1:
        return np.ascontiguousarray(array[:1])
    if method == "expand":
        if size < count:
            indices = np.rint(np.arange(size) * (count - 1) / (size - 1)).astype(np.int64)
        else:
            indices = np.floor((np.arange(size) + 0.5) * count / size).astype(np.int64)
        return np.ascontiguousarray(array[np.minimum(indices, count - 1)])
    if method == "repeat_all":
        repetitions = (math.ceil(size / count), *(1 for _ in array.shape[1:]))
        return np.ascontiguousarray(np.tile(array, repetitions)[:size])
    if size < count:
        return np.ascontiguousarray(array[:size])
    repeated = np.repeat(
        array[:1] if method == "repeat_first" else array[-1:], size - count, axis=0
    )
    values = (repeated, array) if method == "repeat_first" else (array, repeated)
    return np.ascontiguousarray(np.concatenate(values, axis=0))


def _torch_cpu_randperm(size: int, seed: int) -> np.ndarray:
    """Match PyTorch's CPU randperm sequence without making this pack torch-dependent."""

    state = [seed & 0xFFFFFFFF]
    for index in range(1, 624):
        previous = state[-1]
        state.append((1812433253 * (previous ^ (previous >> 30)) + index) & 0xFFFFFFFF)
    state_index = 624

    def random_uint32() -> int:
        nonlocal state_index
        if state_index == 624:
            for index in range(624):
                value = (state[index] & 0x80000000) | (state[(index + 1) % 624] & 0x7FFFFFFF)
                state[index] = state[(index + 397) % 624] ^ (value >> 1)
                if value & 1:
                    state[index] ^= 0x9908B0DF
            state_index = 0
        value = state[state_index]
        state_index += 1
        value ^= value >> 11
        value ^= (value << 7) & 0x9D2C5680
        value ^= (value << 15) & 0xEFC60000
        value ^= value >> 18
        return value & 0xFFFFFFFF

    permutation = np.arange(size, dtype=np.int64)
    for index in range(size - 1):
        selected = index + random_uint32() % (size - index)
        permutation[index], permutation[selected] = permutation[selected], permutation[index]
    return permutation


def _edit_batch(
    array: np.ndarray,
    *,
    operation: str,
    start: int,
    count: int,
    amount: int,
    size: int,
    expansion: str,
    seed: int,
    range_end: str,
) -> np.ndarray:
    batch = int(array.shape[0])
    if operation == "range":
        if count < 1:
            raise ValueError(f"count must be positive, got {count}")
        if range_end not in ("clamp", "error"):
            raise ValueError(f"unknown range-end policy: {range_end}")
        index = start + batch if start < 0 else start
        if range_end == "clamp":
            index = min(max(index, 0), batch - 1)
        elif not 0 <= index < batch:
            raise ValueError(f"start must select a batch item, got {start} for batch size {batch}")
        end = index + count
        if end > batch and range_end == "error":
            raise ValueError(f"range ending at {end} exceeds batch size {batch}")
        return np.ascontiguousarray(array[index : min(end, batch)])
    if operation == "repeat_all":
        if amount < 1:
            raise ValueError(f"amount must be positive, got {amount}")
        _check_batch_output_size(array, batch * amount)
        repetitions = (amount, *(1 for _ in array.shape[1:]))
        return np.ascontiguousarray(np.tile(array, repetitions))
    if operation == "repeat_each":
        if amount < 1:
            raise ValueError(f"amount must be positive, got {amount}")
        _check_batch_output_size(array, batch * amount)
        return np.ascontiguousarray(np.repeat(array, amount, axis=0))
    if operation == "resize_count":
        return _resize_count(array, size, expansion)
    if operation == "reverse":
        return np.ascontiguousarray(array[::-1])
    if operation == "shuffle":
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")
        permutation = _torch_cpu_randperm(batch, seed)
        return np.ascontiguousarray(array[permutation])
    if operation == "pingpong":
        if batch <= 2:
            return np.ascontiguousarray(array)
        _check_batch_output_size(array, 2 * batch - 2)
        return np.ascontiguousarray(np.concatenate((array, array[-2:0:-1]), axis=0))
    raise ValueError(f"unknown batch edit operation: {operation}")


class ImageBatchEdit(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        specs = {spec.id: spec for spec in _edit_inputs("image", IMAGE)}
        specs["generate_repeat_marker"] = InputSpec(
            "generate_repeat_marker",
            BOOLEAN,
            required=False,
            default=False,
            advanced=True,
        )
        return with_v1_migration(
            NodeSchema(
                version=2,
                node_type="dinkster.image.batch.edit",
                display_name="Edit Image Batch",
                category="image/batch",
                inputs=(
                    specs["image"],
                    InputSpec("mask", MASK, required=False, advanced=True),
                ),
                combos=(
                    DynamicComboSpec(
                        "operation",
                        tuple(
                            DynamicComboOption(key, tuple(specs[name] for name in names))
                            for key, names in (
                                ("range", ("start", "count", "range_end")),
                                ("repeat_all", ("amount",)),
                                ("repeat_each", ("amount", "generate_repeat_marker")),
                                ("resize_count", ("size", "expansion")),
                                ("reverse", ()),
                                ("shuffle", ("seed",)),
                                ("pingpong", ()),
                            )
                        ),
                        default="range",
                    ),
                ),
                outputs=(
                    OutputSpec("image", IMAGE, preview=True),
                    OutputSpec("mask", MASK, optional=True),
                ),
                search_terms=(
                    "ImageFromBatch",
                    "RepeatImageBatch",
                    "repeat each image",
                    "reverse image batch",
                    "shuffle image batch",
                    "pingpong",
                    "ping-pong loop",
                ),
            ),
            current_only_choices={"operation": frozenset({"pingpong"})},
        )

    @classmethod
    @materialized_inputs
    def execute(
        cls,
        *,
        image: object,
        operation: str = "range",
        start: int = 0,
        count: int = 1,
        amount: int = 1,
        size: int = 1,
        expansion: str = "expand",
        seed: int = 0,
        range_end: str = "clamp",
        mask: object | None = None,
        generate_repeat_marker: bool = False,
    ) -> Mapping[str, object]:
        source_image = image_array(image)
        edited_image = _edit_batch(
            source_image,
            operation=operation,
            start=start,
            count=count,
            amount=amount,
            size=size,
            expansion=expansion,
            seed=seed,
            range_end=range_end,
        )
        if mask is not None:
            edited_mask: object = _edit_batch(
                mask_array(mask),
                operation=operation,
                start=start,
                count=count,
                amount=amount,
                size=size,
                expansion=expansion,
                seed=seed,
                range_end=range_end,
            )
        elif generate_repeat_marker:
            if operation != "repeat_each":
                raise ValueError("repeat markers require the repeat_each operation")
            edited_mask = np.zeros(
                (len(edited_image), int(source_image.shape[1]), int(source_image.shape[2])),
                dtype=np.float32,
            )
            edited_mask[::amount] = 1.0
        else:
            edited_mask = ABSENT
        return cls.outputs(image=edited_image, mask=edited_mask)


class MaskBatchEdit(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        specs = {spec.id: spec for spec in _edit_inputs("mask", MASK)}
        return with_v1_migration(
            NodeSchema(
                version=2,
                node_type="dinkster.mask.batch.edit",
                display_name="Edit Mask Batch",
                category="mask/batch",
                inputs=(specs["mask"],),
                combos=(
                    DynamicComboSpec(
                        "operation",
                        tuple(
                            DynamicComboOption(key, tuple(specs[name] for name in names))
                            for key, names in (
                                ("range", ("start", "count", "range_end")),
                                ("repeat_all", ("amount",)),
                                ("repeat_each", ("amount",)),
                                ("resize_count", ("size", "expansion")),
                                ("reverse", ()),
                                ("shuffle", ("seed",)),
                                ("pingpong", ()),
                            )
                        ),
                        default="range",
                    ),
                ),
                outputs=(OutputSpec("mask", MASK, preview=True),),
                search_terms=("select mask", "expand mask batch", "repeat mask batch"),
            ),
            current_only_choices={"operation": frozenset({"pingpong"})},
        )

    @classmethod
    @materialized_inputs
    def execute(
        cls,
        *,
        mask: object,
        operation: str = "range",
        start: int = 0,
        count: int = 1,
        amount: int = 1,
        size: int = 1,
        expansion: str = "expand",
        seed: int = 0,
        range_end: str = "clamp",
    ) -> Mapping[str, object]:
        return cls.outputs(
            mask=_edit_batch(
                mask_array(mask),
                operation=operation,
                start=start,
                count=count,
                amount=amount,
                size=size,
                expansion=expansion,
                seed=seed,
                range_end=range_end,
            )
        )


def _family_values(values: Mapping[str, object], subject: str) -> list[object]:
    if not values:
        raise ValueError(f"{subject} must not be empty")
    return list(values.values())


def _normalize_images(
    values: Mapping[str, object],
    *,
    input_count: int,
    shape_policy: str,
    channel_policy: str,
    interpolation: str,
) -> list[np.ndarray]:
    if shape_policy not in ("strict", "resize_to_first"):
        raise ValueError(f"unknown shape policy: {shape_policy}")
    if channel_policy not in ("strict", "pad_with_one"):
        raise ValueError(f"unknown channel policy: {channel_policy}")
    ordered = indexed_family_values(
        values,
        count=input_count,
        member_prefix="image_",
        subject="images",
    )
    arrays = [
        image_array(value, subject=f"image {index}") if value is not None else None
        for index, value in enumerate(ordered, start=1)
    ]
    first = arrays[0]
    assert first is not None
    max_channels = max(int(array.shape[3]) for array in arrays if array is not None)
    normalized: list[np.ndarray] = []
    for index, array in enumerate(arrays, start=1):
        if array is None:
            normalized.append(
                np.zeros(
                    (len(first), int(first.shape[1]), int(first.shape[2]), max_channels),
                    dtype=np.float32,
                )
            )
            continue
        if array.shape[1:3] != first.shape[1:3]:
            if shape_policy == "strict":
                raise ValueError(
                    f"image {index} dimensions must match the first image, got "
                    f"{array.shape[1:3]} and {first.shape[1:3]}"
                )
            array = resize_to_fill(
                array,
                int(first.shape[2]),
                int(first.shape[1]),
                cast("Interpolation", interpolation),
            )
        if array.shape[3] != max_channels:
            if channel_policy == "strict":
                raise ValueError("all image channel counts must match")
            padding = np.ones(
                (*array.shape[:3], max_channels - int(array.shape[3])), dtype=np.float32
            )
            array = np.concatenate((array, padding), axis=3)
        normalized.append(np.ascontiguousarray(array))
    return normalized


def _parse_indexes(value: str) -> list[int]:
    if not value.strip():
        return []
    try:
        return [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError("indexes must be a comma-separated list of integers") from exc


def _combine_arrays(
    arrays: list[np.ndarray],
    *,
    operation: str,
    index: int,
    indexes: str,
    subject: str,
) -> np.ndarray:
    first = arrays[0]
    if operation == "concat":
        count = sum(int(array.shape[0]) for array in arrays)
        _check_batch_output_size(first, count)
        return np.ascontiguousarray(np.concatenate(arrays, axis=0))
    if len(arrays) != 2:
        raise ValueError(f"{operation} requires exactly two {subject} inputs")
    original, replacement = arrays
    if operation == "replace_range":
        end = index + int(replacement.shape[0])
        if index < 0 or index >= len(original) or end > len(original):
            raise ValueError(f"replacement range is outside the original {subject} batch")
        replaced = original.copy()
        replaced[index:end] = replacement
        return np.ascontiguousarray(replaced)
    parsed = _parse_indexes(indexes)
    if not parsed:
        return original
    if operation == "replace_indexed":
        replaced = original.copy()
        for target, image in zip(parsed, replacement, strict=False):
            if not -len(replaced) <= target < len(replaced):
                raise ValueError(f"replacement index {target} is outside the {subject} batch")
            replaced[target] = image
        return np.ascontiguousarray(replaced)
    if operation == "insert_indexed":
        output_count = len(original) + len(parsed)
        if parsed != sorted(parsed) or len(set(parsed)) != len(parsed):
            raise ValueError("insertion indexes must be unique and increasing")
        if parsed[0] < 0 or parsed[-1] >= output_count:
            raise ValueError("insertion indexes are outside the resulting image batch")
        _check_batch_output_size(first, output_count)
        inserted = iter(enumerate(parsed))
        next_insert = next(inserted, None)
        source_index = 0
        output: list[np.ndarray] = []
        for output_index in range(output_count):
            if next_insert is not None and output_index == next_insert[1]:
                output.append(replacement[next_insert[0] % len(replacement)])
                next_insert = next(inserted, None)
            else:
                if source_index >= len(original):
                    raise ValueError("insertion indexes do not leave room for the original batch")
                output.append(original[source_index])
                source_index += 1
        return np.ascontiguousarray(np.stack(output, axis=0))
    raise ValueError(f"unknown batch combine operation: {operation}")


class ImageBatchCombine(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_v1_migration(
            NodeSchema(
                version=2,
                node_type="dinkster.image.batch.combine",
                display_name="Combine Image Batches",
                category="image/batch",
                inputs=(
                    combo_input("channel_policy", ("strict", "pad_with_one"), "strict"),
                    number_input(
                        "input_count", INT, 0, minimum=0, maximum=1000, step=1, hidden=True
                    ),
                    combo_input("mask_fallback", ("absent", "zero_64"), "absent", advanced=True),
                ),
                combos=(
                    DynamicComboSpec(
                        "operation",
                        (
                            DynamicComboOption("concat"),
                            DynamicComboOption(
                                "replace_range",
                                (
                                    number_input(
                                        "index",
                                        INT,
                                        0,
                                        minimum=-MAX_DIMENSION,
                                        maximum=MAX_DIMENSION,
                                        step=1,
                                    ),
                                ),
                            ),
                            DynamicComboOption(
                                "insert_indexed",
                                (InputSpec("indexes", STRING, required=False, default="0"),),
                            ),
                            DynamicComboOption(
                                "replace_indexed",
                                (InputSpec("indexes", STRING, required=False, default="0"),),
                            ),
                        ),
                        default="concat",
                    ),
                    DynamicComboSpec(
                        "shape_policy",
                        (
                            DynamicComboOption("strict"),
                            DynamicComboOption(
                                "resize_to_first",
                                (
                                    combo_input(
                                        "interpolation",
                                        INTERPOLATION_OPTIONS,
                                        "bilinear",
                                        advanced=True,
                                    ),
                                ),
                            ),
                        ),
                        default="strict",
                    ),
                ),
                input_families=(
                    InputFamilySpec(
                        "images", IMAGE, min_members=1, max_members=1000, member_prefix="image_"
                    ),
                    InputFamilySpec(
                        "masks", MASK, min_members=0, max_members=2, member_prefix="mask_"
                    ),
                ),
                outputs=(
                    OutputSpec("image", IMAGE, preview=True),
                    OutputSpec("mask", MASK, optional=True),
                ),
                search_terms=("ImageBatch", "combine images", "insert images", "replace images"),
            )
        )

    @classmethod
    @materialized_inputs
    def execute(
        cls,
        *,
        images: Mapping[str, object],
        masks: Mapping[str, object] | None = None,
        operation: str = "concat",
        index: int = 0,
        indexes: str = "0",
        shape_policy: str = "strict",
        channel_policy: str = "strict",
        interpolation: str = "bilinear",
        input_count: int = 0,
        mask_fallback: str = "absent",
    ) -> Mapping[str, object]:
        arrays = _normalize_images(
            images,
            input_count=input_count,
            shape_policy=shape_policy,
            channel_policy=channel_policy,
            interpolation=interpolation,
        )
        combined_image = _combine_arrays(
            arrays,
            operation=operation,
            index=index,
            indexes=indexes,
            subject="image",
        )
        if masks:
            mask_arrays = _normalize_masks(
                masks,
                shape_policy=shape_policy,
                interpolation="nearest-exact",
            )
            combined_mask: object = _combine_arrays(
                mask_arrays,
                operation=operation,
                index=index,
                indexes=indexes,
                subject="mask",
            )
        elif mask_fallback == "zero_64":
            combined_mask = np.zeros((1, 64, 64), dtype=np.float32)
        elif mask_fallback == "absent":
            combined_mask = ABSENT
        else:
            raise ValueError(f"unknown mask fallback: {mask_fallback}")
        return cls.outputs(
            image=combined_image,
            mask=combined_mask,
        )


def _normalize_masks(
    values: Mapping[str, object],
    *,
    shape_policy: str,
    interpolation: str,
) -> list[np.ndarray]:
    if shape_policy not in ("strict", "resize_to_first"):
        raise ValueError(f"unknown shape policy: {shape_policy}")
    arrays = [
        mask_array(value, subject=f"mask {index}")
        for index, value in enumerate(_family_values(values, "masks"))
    ]
    first = arrays[0]
    normalized: list[np.ndarray] = []
    for index, array in enumerate(arrays):
        if array.shape[1:] != first.shape[1:]:
            if shape_policy == "strict":
                raise ValueError(
                    f"mask {index} dimensions must match the first mask, got "
                    f"{array.shape[1:]} and {first.shape[1:]}"
                )
            array = resize_to_fill(
                array,
                int(first.shape[2]),
                int(first.shape[1]),
                cast("Interpolation", interpolation),
            )
        normalized.append(array)
    return normalized


class MaskBatchCombine(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_v1_migration(
            NodeSchema(
                version=2,
                node_type="dinkster.mask.batch.combine",
                display_name="Combine Mask Batches",
                category="mask/batch",
                inputs=(),
                combos=(
                    DynamicComboSpec(
                        "operation",
                        (
                            DynamicComboOption("concat"),
                            DynamicComboOption(
                                "replace_range",
                                (
                                    number_input(
                                        "index",
                                        INT,
                                        0,
                                        minimum=-MAX_DIMENSION,
                                        maximum=MAX_DIMENSION,
                                        step=1,
                                    ),
                                ),
                            ),
                        ),
                        default="concat",
                    ),
                    DynamicComboSpec(
                        "shape_policy",
                        (
                            DynamicComboOption("strict"),
                            DynamicComboOption(
                                "resize_to_first",
                                (
                                    combo_input(
                                        "interpolation",
                                        INTERPOLATION_OPTIONS,
                                        "bicubic",
                                        advanced=True,
                                    ),
                                ),
                            ),
                        ),
                        default="strict",
                    ),
                ),
                input_families=(
                    InputFamilySpec(
                        "masks", MASK, min_members=2, max_members=100, member_prefix="mask_"
                    ),
                ),
                outputs=(OutputSpec("mask", MASK, preview=True),),
                search_terms=("MaskBatch", "combine masks"),
            )
        )

    @classmethod
    @materialized_inputs
    def execute(
        cls,
        *,
        masks: Mapping[str, object],
        operation: str = "concat",
        index: int = 0,
        shape_policy: str = "strict",
        interpolation: str = "bicubic",
    ) -> Mapping[str, object]:
        arrays = _normalize_masks(
            masks,
            shape_policy=shape_policy,
            interpolation=interpolation,
        )
        return cls.outputs(
            mask=_combine_arrays(
                arrays,
                operation=operation,
                index=index,
                indexes="",
                subject="mask",
            )
        )


def _as_image_list(value: object) -> list[np.ndarray]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("images must be a typed list")
    values = cast("list[object] | tuple[object, ...]", value)
    if not values:
        raise ValueError("images must not be empty")
    return [image_array(item, subject=f"image {index}") for index, item in enumerate(values)]


def _batch_from_list(values: object, *, shape_policy: str, interpolation: str) -> np.ndarray:
    if shape_policy not in ("strict", "resize_to_first"):
        raise ValueError(f"unknown shape policy: {shape_policy}")
    arrays = _as_image_list(values)
    first = arrays[0]
    normalized: list[np.ndarray] = []
    for index, array in enumerate(arrays):
        if array.shape[3] != first.shape[3]:
            raise ValueError(
                f"image {index} channels must match the first image, got "
                f"{array.shape[3]} and {first.shape[3]}"
            )
        if array.shape[1:3] != first.shape[1:3]:
            if shape_policy == "strict":
                raise ValueError(
                    f"image {index} dimensions must match the first image, got "
                    f"{array.shape[1:3]} and {first.shape[1:3]}"
                )
            array = resize_to_fill(
                array,
                int(first.shape[2]),
                int(first.shape[1]),
                cast("Interpolation", interpolation),
            )
        normalized.append(array)
    _check_batch_output_size(first, sum(len(array) for array in normalized))
    return np.ascontiguousarray(np.concatenate(normalized, axis=0))


class ImageBatchToList(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.image.batch.to_list",
            display_name="Image Batch to List",
            category="image/batch",
            inputs=(InputSpec("image", IMAGE),),
            outputs=(OutputSpec("images", IMAGE_LIST),),
            search_terms=("ImageBatchToList", "split image batch"),
        )

    @classmethod
    def execute(cls, *, image: object) -> Mapping[str, object]:
        array = image_array(image)
        return cls.outputs(
            images=[np.ascontiguousarray(array[index : index + 1]) for index in range(len(array))]
        )


class ImageListToBatch(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_v1_migration(
            NodeSchema(
                version=2,
                node_type="dinkster.image.list.to_batch",
                display_name="Image List to Batch",
                category="image/batch",
                inputs=(InputSpec("images", IMAGE_LIST),),
                combos=(
                    DynamicComboSpec(
                        "shape_policy",
                        (
                            DynamicComboOption("strict"),
                            DynamicComboOption(
                                "resize_to_first",
                                (
                                    combo_input(
                                        "interpolation",
                                        INTERPOLATION_OPTIONS,
                                        "bicubic",
                                        advanced=True,
                                    ),
                                ),
                            ),
                        ),
                        default="strict",
                    ),
                ),
                outputs=(OutputSpec("image", IMAGE, preview=True),),
                search_terms=("ImageListToBatch", "combine image list"),
            )
        )

    @classmethod
    @materialized_inputs
    def execute(
        cls,
        *,
        images: object,
        shape_policy: str = "strict",
        interpolation: str = "bicubic",
    ) -> Mapping[str, object]:
        return cls.outputs(
            image=_batch_from_list(
                images,
                shape_policy=shape_policy,
                interpolation=interpolation,
            )
        )


class ImageRebatch(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.image.rebatch",
            display_name="Rebatch Images",
            category="image/batch",
            inputs=(
                InputSpec("images", IMAGE_LIST),
                number_input("batch_size", INT, 1, minimum=1, maximum=4096, step=1),
            ),
            outputs=(OutputSpec("images", IMAGE_LIST),),
            search_terms=("RebatchImages", "change image batch size"),
        )

    @classmethod
    def execute(
        cls,
        *,
        images: object,
        batch_size: int = 1,
    ) -> Mapping[str, object]:
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        arrays = _as_image_list(images)
        frames = [array[index : index + 1] for array in arrays for index in range(len(array))]
        return cls.outputs(
            images=[
                np.ascontiguousarray(np.concatenate(frames[index : index + batch_size], axis=0))
                for index in range(0, len(frames), batch_size)
            ]
        )


BATCH_NODES: tuple[type[Node], ...] = (
    ImageBatchEdit,
    MaskBatchEdit,
    ImageBatchCombine,
    MaskBatchCombine,
    ImageBatchToList,
    ImageListToBatch,
    ImageRebatch,
)


__all__ = [
    "BATCH_NODES",
    "IMAGE_LIST",
    "ImageBatchCombine",
    "ImageBatchEdit",
    "ImageBatchToList",
    "ImageListToBatch",
    "ImageRebatch",
    "MaskBatchCombine",
    "MaskBatchEdit",
]
