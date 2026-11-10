"""Native ComfyUI conditioning transform nodes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_FLOAT,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    first_party_schema,
)

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)

_FLOAT = TypeExpr.concrete(CORE_FLOAT)


def _number(
    name: str,
    type_: TypeExpr,
    default: float | int,
    minimum: float | int,
    maximum: float | int,
    step: float | int,
) -> InputSpec:
    return InputSpec(
        name,
        type_,
        default=default,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


def conditioning_set_values(
    conditioning: Sequence[Sequence[Any]], values: Mapping[str, object], *, append: bool = False
) -> list[list[Any]]:
    result: list[list[Any]] = []
    for entry in conditioning:
        metadata = entry[1].copy()
        for key, value in values.items():
            old_value = metadata.get(key)
            metadata[key] = old_value + value if append and old_value is not None else value
        result.append([entry[0], metadata])
    return result


class ConditioningStableAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="ConditioningStableAudio",
            category="model/conditioning/stable audio",
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                _number("seconds_start", _FLOAT, 0.0, 0.0, 1000.0, 0.1),
                _number("seconds_total", _FLOAT, 47.0, 0.0, 1000.0, 0.1),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING),
                OutputSpec("negative", _CONDITIONING),
            ),
        )
        return first_party_schema(schema)

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        seconds_start: float,
        seconds_total: float,
    ) -> Mapping[str, object]:
        values = {"seconds_start": seconds_start, "seconds_total": seconds_total}
        return cls.outputs(
            positive=conditioning_set_values(positive, values),
            negative=conditioning_set_values(negative, values),
        )
