"""Apply schema media policies at the ordinary node invocation boundary."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

from dinkster_schema import NodeSchema, OutputSpec
from dinkster_values import (
    Value,
    ValueMeta,
    annotate_image,
    annotate_mask,
    image_array_meta,
    iter_value_tree,
    make_list_value,
    mask_array_meta,
    media_semantics,
    parse_asset_type_id,
    parse_list_type_id,
    runtime_type_atom,
)
from dinkster_values.model import PyObjPayload


def media_input_value(obj: object, type_id: str, original: Value) -> Value:
    """Describe invoked pixels without hashing decoded assets into cache identity."""
    if runtime_type_atom(type_id) not in (
        "comfy.IMAGE",
        "dinkster.image",
        "comfy.MASK",
        "dinkster.mask",
    ):
        return original
    element_type = parse_list_type_id(type_id)
    if element_type is not None and isinstance(obj, (list, tuple)):
        children = tuple(
            media_input_value(item, element_type, original)
            for item in cast("Sequence[object]", obj)
        )
        return replace(make_list_value(element_type, children), fingerprint=original.fingerprint)
    if not hasattr(obj, "shape"):
        return original
    if type_id in ("comfy.IMAGE", "dinkster.image"):
        metadata = image_array_meta(obj)
    elif type_id in ("comfy.MASK", "dinkster.mask"):
        metadata = mask_array_meta(obj)
    else:
        return original
    return replace(original, type_id=type_id, meta=ValueMeta(metadata), payload=PyObjPayload(obj))


def coercion_drops_alpha(value: Value, result: object) -> bool:
    """Inspect source headers only; unknown formats never block a custom decoder."""
    sources = [
        child
        for child in iter_value_tree(value)
        if parse_asset_type_id(child.type_id) in ("comfy.IMAGE", "dinkster.image")
    ]
    if not sources:
        return False
    outputs = cast("Sequence[object]", result) if isinstance(result, (list, tuple)) else [result]
    shapes = [cast("Sequence[int]", getattr(item, "shape", ())) for item in outputs]
    if all(len(shape) in (3, 4) and shape[-1] in (2, 4) for shape in shapes):
        return False
    try:
        image_module = cast("Any", importlib.import_module("PIL.Image"))
    except ImportError:
        return False
    for source in sources:
        try:
            asset = cast("Any", source.resolve())
            with asset.open() as handle, image_module.open(handle) as image:
                if "A" in image.getbands() or "transparency" in image.info:
                    return True
        except (OSError, ValueError, AttributeError):
            continue
    return False


def apply_alpha_policy(obj: object, type_id: str, policy: str) -> object:
    element_type = parse_list_type_id(type_id)
    if element_type is not None and isinstance(obj, (list, tuple)):
        return [
            apply_alpha_policy(item, element_type, policy) for item in cast("Sequence[object]", obj)
        ]
    if type_id not in {"comfy.IMAGE", "dinkster.image"} or policy not in (
        "require",
        "create_if_missing",
    ):
        return obj
    array = cast("Any", obj)
    if len(array.shape) not in (3, 4):
        raise ValueError("IMAGE alpha policy requires a channels-last image")
    if array.shape[-1] in (2, 4):
        return obj
    if policy == "require":
        raise ValueError("IMAGE input requires an alpha channel")
    if hasattr(obj, "detach"):
        torch = cast("Any", importlib.import_module("torch"))
        result = torch.cat((array, array.new_ones((*array.shape[:-1], 1))), dim=-1)
    else:
        np = cast("Any", importlib.import_module("numpy"))
        result = np.concatenate(
            (array, np.ones((*array.shape[:-1], 1), dtype=array.dtype)), axis=-1
        )
    return annotate_image(result, color=cast("Any", media_semantics(obj).get("color")))


def prepare_media_output(
    obj: object,
    type_id: str,
    spec: OutputSpec,
    schema: NodeSchema,
    inputs: Mapping[str, Value],
) -> object:
    element_type = parse_list_type_id(type_id)
    if element_type is not None and isinstance(obj, (list, tuple)):
        return [
            prepare_media_output(item, element_type, spec, schema, inputs)
            for item in cast("Sequence[object]", obj)
        ]
    result = apply_alpha_policy(obj, type_id, spec.alpha_policy)
    is_image = type_id in {"comfy.IMAGE", "dinkster.image"}
    is_mask = type_id in {"comfy.MASK", "dinkster.mask"}
    if (not is_image and not is_mask) or not hasattr(result, "shape"):
        return result
    explicit = hasattr(result, "_dinkster_media")
    inherited: dict[str, object] = {}
    sources: list[dict[str, object]] = []
    for input_spec in schema.inputs:
        root = inputs.get(input_spec.id)
        if root is None or input_spec.alpha_policy == "drop":
            continue
        for value in iter_value_tree(root):
            if is_image and value.type_id in {"comfy.IMAGE", "dinkster.image"}:
                channels = value.meta.get("channels")
                alpha = (
                    cast("Mapping[str, object]", channels).get("alpha")
                    if isinstance(channels, Mapping)
                    else None
                )
                sources.append({"color": value.meta.get("color"), "alpha": alpha})
            elif is_mask and value.type_id in {"comfy.MASK", "dinkster.mask"}:
                sources.append(
                    {
                        "polarity": value.meta.get("polarity", "coverage"),
                        "semantic": value.meta.get("semantic", "selection"),
                    }
                )
    if sources and all(source == sources[0] for source in sources):
        inherited = sources[0]
    if is_mask:
        current = media_semantics(result) if explicit else inherited
        polarity = spec.mask_polarity or current.get("polarity", "coverage")
        semantic = spec.mask_semantic or current.get("semantic", "selection")
        metadata = mask_array_meta(result)
        if polarity == metadata["polarity"] and semantic == metadata["semantic"]:
            return result
        return annotate_mask(result, polarity=cast("str", polarity), semantic=cast("str", semantic))
    if explicit or not inherited:
        return result
    shape = cast("Any", result).shape
    alpha = inherited.get("alpha") if shape[-1] in (2, 4) else None
    return annotate_image(
        result,
        alpha=cast("str", alpha) if alpha in ("straight", "premultiplied") else None,
        color=cast("Mapping[str, object] | None", inherited.get("color")),
    )
