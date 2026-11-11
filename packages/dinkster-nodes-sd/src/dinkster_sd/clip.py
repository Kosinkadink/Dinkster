"""Core text encoding and CLIP layer selection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CLIP_VISION_OUTPUT_TYPE,
    CONDITIONING_TYPE,
    CORE_FLOAT,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_clip_vision_output,
    with_identity_replacement,
)


class unCLIPConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.sd.un_clip_conditioning",
            display_name="unCLIPConditioning",
            category="model/conditioning",
            inputs=(
                InputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("clip_vision_output", TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE)),
                InputSpec(
                    "strength",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=-10.0, max=10.0, step=0.01),
                ),
                InputSpec(
                    "noise_augmentation",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("CONDITIONING", TypeExpr.concrete(CONDITIONING_TYPE)),),
        )
        return with_identity_replacement(schema, "unCLIPConditioning")

    @classmethod
    def execute(
        cls,
        *,
        conditioning: list[list[Any]],
        clip_vision_output: object,
        strength: float,
        noise_augmentation: float,
    ) -> Mapping[str, object]:
        if strength == 0:
            return cls.outputs(CONDITIONING=conditioning)
        output = materialize_clip_vision_output(clip_vision_output)
        value = {
            "clip_vision_output": output,
            "strength": strength,
            "noise_augmentation": noise_augmentation,
        }
        result: list[list[Any]] = []
        for text, metadata in conditioning:
            values = metadata.copy()
            previous = values.get("unclip_conditioning")
            values["unclip_conditioning"] = [value] if previous is None else previous + [value]
            result.append([text, values])
        return cls.outputs(CONDITIONING=result)
