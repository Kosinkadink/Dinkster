"""Native ComfyUI conditioning transform nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_image,
)

_CATEGORY = "model/conditioning/transform"
_CONDITIONING = TypeExpr.concrete("CONDITIONING")
_MASK = TypeExpr.concrete("MASK")
_FLOAT = TypeExpr.concrete("FLOAT")
_INT = TypeExpr.concrete("INT")
_COMBO = TypeExpr.concrete("COMBO")


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


def _schema(node_type: str, display_name: str, inputs: tuple[InputSpec, ...]) -> NodeSchema:
    return NodeSchema(
        node_type=node_type,
        display_name=display_name,
        category=_CATEGORY,
        inputs=inputs,
        outputs=(OutputSpec("CONDITIONING", _CONDITIONING),),
    )


def _set_values(
    conditioning: Sequence[Sequence[Any]], values: Mapping[str, object]
) -> list[list[Any]]:
    return [[entry[0], {**entry[1], **values}] for entry in conditioning]


class ConditioningCombine(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ConditioningCombine",
            "Conditioning (Combine)",
            (
                InputSpec("conditioning_1", _CONDITIONING),
                InputSpec("conditioning_2", _CONDITIONING),
            ),
        )

    @classmethod
    def execute(
        cls, *, conditioning_1: list[list[Any]], conditioning_2: list[list[Any]]
    ) -> Mapping[str, object]:
        return cls.outputs(CONDITIONING=conditioning_1 + conditioning_2)


class ConditioningAverage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ConditioningAverage",
            "Conditioning (Average)",
            (
                InputSpec("conditioning_to", _CONDITIONING),
                InputSpec("conditioning_from", _CONDITIONING),
                _number("conditioning_to_strength", _FLOAT, 1.0, 0.0, 1.0, 0.01),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        conditioning_to: Sequence[Sequence[Any]],
        conditioning_from: Sequence[Sequence[Any]],
        conditioning_to_strength: float,
    ) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        cond_from = conditioning_from[0][0]
        pooled_output_from = conditioning_from[0][1].get("pooled_output")
        result: list[list[Any]] = []
        for text, metadata in conditioning_to:
            pooled_output_to = metadata.get("pooled_output", pooled_output_from)
            source = cond_from[:, : text.shape[1]]
            if source.shape[1] < text.shape[1]:
                source = torch.cat(
                    (source, torch.zeros((1, text.shape[1] - source.shape[1], text.shape[2]))),
                    dim=1,
                )
            values = metadata.copy()
            if pooled_output_from is not None and pooled_output_to is not None:
                values["pooled_output"] = torch.mul(
                    pooled_output_to, conditioning_to_strength
                ) + torch.mul(pooled_output_from, 1.0 - conditioning_to_strength)
            elif pooled_output_from is not None:
                values["pooled_output"] = pooled_output_from
            result.append(
                [
                    torch.mul(text, conditioning_to_strength)
                    + torch.mul(source, 1.0 - conditioning_to_strength),
                    values,
                ]
            )
        return cls.outputs(CONDITIONING=result)


class ConditioningConcat(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ConditioningConcat",
            "Conditioning (Concat)",
            (
                InputSpec("conditioning_to", _CONDITIONING),
                InputSpec("conditioning_from", _CONDITIONING),
            ),
        )

    @classmethod
    def execute(
        cls, *, conditioning_to: Sequence[Sequence[Any]], conditioning_from: Sequence[Sequence[Any]]
    ) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        source = conditioning_from[0][0]
        return cls.outputs(
            CONDITIONING=[
                [torch.cat((text, source), 1), metadata.copy()]
                for text, metadata in conditioning_to
            ]
        )


class ConditioningMultiply(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ConditioningMultiply",
            "Conditioning (Multiply)",
            (
                InputSpec("conditioning", _CONDITIONING),
                _number("multiplier", _FLOAT, 1.0, -100.0, 100.0, 0.01),
            ),
        )

    @classmethod
    def execute(
        cls, *, conditioning: Sequence[Sequence[Any]], multiplier: float
    ) -> Mapping[str, object]:
        result: list[list[Any]] = []
        for text, metadata in conditioning:
            values: dict[str, object] = {}
            pooled_output = metadata.get("pooled_output")
            if pooled_output is not None:
                values["pooled_output"] = pooled_output * multiplier
            result.extend(_set_values([[text * multiplier, metadata]], values))
        return cls.outputs(CONDITIONING=result)


class ConditioningSetArea(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ConditioningSetArea",
            "Conditioning (Set Area)",
            (
                InputSpec("conditioning", _CONDITIONING),
                _number("width", _INT, 64, 64, 16384, 8),
                _number("height", _INT, 64, 64, 16384, 8),
                _number("x", _INT, 0, 0, 16384, 8),
                _number("y", _INT, 0, 0, 16384, 8),
                _number("strength", _FLOAT, 1.0, 0.0, 10.0, 0.01),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        conditioning: Sequence[Sequence[Any]],
        width: int,
        height: int,
        x: int,
        y: int,
        strength: float,
    ) -> Mapping[str, object]:
        return cls.outputs(
            CONDITIONING=_set_values(
                conditioning,
                {
                    "area": (height // 8, width // 8, y // 8, x // 8),
                    "strength": strength,
                    "set_area_to_bounds": False,
                },
            )
        )


class ConditioningSetAreaPercentage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ConditioningSetAreaPercentage",
            "Conditioning (Set Area with Percentage)",
            (
                InputSpec("conditioning", _CONDITIONING),
                _number("width", _FLOAT, 1.0, 0.0, 1.0, 0.01),
                _number("height", _FLOAT, 1.0, 0.0, 1.0, 0.01),
                _number("x", _FLOAT, 0.0, 0.0, 1.0, 0.01),
                _number("y", _FLOAT, 0.0, 0.0, 1.0, 0.01),
                _number("strength", _FLOAT, 1.0, 0.0, 10.0, 0.01),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        conditioning: Sequence[Sequence[Any]],
        width: float,
        height: float,
        x: float,
        y: float,
        strength: float,
    ) -> Mapping[str, object]:
        return cls.outputs(
            CONDITIONING=_set_values(
                conditioning,
                {
                    "area": ("percentage", height, width, y, x),
                    "strength": strength,
                    "set_area_to_bounds": False,
                },
            )
        )


class ConditioningSetAreaStrength(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ConditioningSetAreaStrength",
            "Conditioning (Set Area Strength)",
            (
                InputSpec("conditioning", _CONDITIONING),
                _number("strength", _FLOAT, 1.0, 0.0, 10.0, 0.01),
            ),
        )

    @classmethod
    def execute(
        cls, *, conditioning: Sequence[Sequence[Any]], strength: float
    ) -> Mapping[str, object]:
        return cls.outputs(CONDITIONING=_set_values(conditioning, {"strength": strength}))


class ConditioningSetMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ConditioningSetMask",
            "Conditioning (Set Mask)",
            (
                InputSpec("conditioning", _CONDITIONING),
                InputSpec("mask", _MASK),
                _number("strength", _FLOAT, 1.0, 0.0, 10.0, 0.01),
                InputSpec("set_cond_area", _COMBO, widget=ComboWidget(("default", "mask bounds"))),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        conditioning: Sequence[Sequence[Any]],
        mask: Any,
        strength: float,
        set_cond_area: str,
    ) -> Mapping[str, object]:
        hooks = importlib.import_module("dinkster_inference.hooks")
        return cls.outputs(
            CONDITIONING=hooks.set_mask_for_conditioning(
                conditioning, materialize_image(mask), set_cond_area, strength
            )
        )


class ConditioningZeroOut(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ConditioningZeroOut",
            "Conditioning Zero Out",
            (InputSpec("conditioning", _CONDITIONING),),
        )

    @classmethod
    def execute(cls, *, conditioning: Sequence[Sequence[Any]]) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        result: list[list[Any]] = []
        for text, metadata in conditioning:
            values = metadata.copy()
            for key in (
                "pooled_output",
                "conditioning_lyrics",
                "conditioning_scale",
                "direct_context",
            ):
                value = values.get(key)
                if value is not None:
                    values[key] = torch.zeros_like(value)
            result.append([torch.zeros_like(text), values])
        return cls.outputs(CONDITIONING=result)


class ConditioningSetTimestepRange(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ConditioningSetTimestepRange",
            "",
            (
                InputSpec("conditioning", _CONDITIONING),
                _number("start", _FLOAT, 0.0, 0.0, 1.0, 0.001),
                _number("end", _FLOAT, 1.0, 0.0, 1.0, 0.001),
            ),
        )

    @classmethod
    def execute(
        cls, *, conditioning: Sequence[Sequence[Any]], start: float, end: float
    ) -> Mapping[str, object]:
        hooks = importlib.import_module("dinkster_inference.hooks")
        return cls.outputs(
            CONDITIONING=hooks.set_timesteps_for_conditioning(conditioning, (start, end))
        )


CONDITIONING_NODES = (
    ConditioningAverage,
    ConditioningCombine,
    ConditioningConcat,
    ConditioningMultiply,
    ConditioningSetArea,
    ConditioningSetAreaPercentage,
    ConditioningSetAreaStrength,
    ConditioningSetMask,
    ConditioningZeroOut,
    ConditioningSetTimestepRange,
)
