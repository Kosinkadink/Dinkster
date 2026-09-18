"""Nonblocking media contract diagnostics over value envelopes, without decoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from dinkster_values import Value, iter_value_tree

from .model import NodeSchema


def _has_alpha(value: Value) -> bool:
    if value.type_id not in {"comfy.IMAGE", "dinkster.image"}:
        return False
    channels = value.meta.get("channels")
    if isinstance(channels, Mapping):
        return cast("Mapping[str, object]", channels).get("alpha") in ("straight", "premultiplied")
    shape = value.meta.get("shape")
    return (
        isinstance(shape, Sequence)
        and len(cast("Sequence[object]", shape)) in (3, 4)
        and shape[-1] in (2, 4)
    )


def media_diagnostics(
    schema: NodeSchema,
    inputs: Mapping[str, Value],
    outputs: Mapping[str, Value],
) -> list[dict[str, object]]:
    diagnostics: list[dict[str, object]] = []
    alpha_inputs = [
        spec.id
        for spec in schema.inputs
        if spec.alpha_policy != "drop"
        and spec.id in inputs
        and any(_has_alpha(value) for value in iter_value_tree(inputs[spec.id]))
    ]
    opaque_inputs = {
        value.fingerprint
        for root in inputs.values()
        for value in iter_value_tree(root)
        if value.type_id in {"comfy.IMAGE", "dinkster.image"} and not _has_alpha(value)
    }
    for spec in schema.inputs:
        if spec.mask_polarity is None or spec.id not in inputs:
            continue
        for value in iter_value_tree(inputs[spec.id]):
            if value.type_id not in {"comfy.MASK", "dinkster.mask"}:
                continue
            actual = value.meta.get("polarity", "coverage")
            if actual != spec.mask_polarity:
                diagnostics.append(
                    {
                        "code": "mask_polarity_mismatch",
                        "inputId": spec.id,
                        "expected": spec.mask_polarity,
                        "actual": actual,
                    }
                )
                break
    for spec in schema.outputs:
        if spec.id not in outputs:
            continue
        value = outputs[spec.id]
        if spec.alpha_policy != "drop" and alpha_inputs:
            if any(
                child.type_id in {"comfy.IMAGE", "dinkster.image"}
                and not _has_alpha(child)
                and child.fingerprint not in opaque_inputs
                for child in iter_value_tree(value)
            ):
                diagnostics.append(
                    {"code": "alpha_dropped", "outputId": spec.id, "inputIds": alpha_inputs}
                )
    return diagnostics
