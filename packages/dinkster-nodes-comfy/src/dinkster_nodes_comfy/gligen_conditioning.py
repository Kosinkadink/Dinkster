"""Textbox positioning through GLIGEN conditioning metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_INT,
    CORE_STRING,
    GLIGEN_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
)


class GLIGENTextBoxApply(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="GLIGENTextBoxApply",
            display_name="Apply GLIGEN Text Box",
            category="model/conditioning/gligen",
            inputs=(
                InputSpec("conditioning_to", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec("gligen_textbox_model", TypeExpr.concrete(GLIGEN_TYPE)),
                InputSpec(
                    "text",
                    TypeExpr.concrete(CORE_STRING),
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                *(
                    InputSpec(
                        name,
                        TypeExpr.concrete(CORE_INT),
                        default=default,
                        widget=NumberWidget(min=minimum, max=16384, step=8),
                    )
                    for name, default, minimum in (
                        ("width", 64, 8),
                        ("height", 64, 8),
                        ("x", 0, 0),
                        ("y", 0, 0),
                    )
                ),
            ),
            outputs=(OutputSpec("CONDITIONING", TypeExpr.concrete(CONDITIONING_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        conditioning_to: Sequence[Sequence[Any]],
        clip: Any,
        gligen_textbox_model: object,
        text: str,
        width: int,
        height: int,
        x: int,
        y: int,
    ) -> Mapping[str, object]:
        _, pooled = clip.encode_from_tokens(clip.tokenize(text), return_pooled="unprojected")
        result: list[list[Any]] = []
        for embedding, metadata in conditioning_to:
            values = metadata.copy()
            previous: list[Any] = values["gligen"][2] if "gligen" in values else []
            values["gligen"] = (
                "position",
                gligen_textbox_model,
                previous + [(pooled, height // 8, width // 8, y // 8, x // 8)],
            )
            result.append([embedding, values])
        return cls.outputs(CONDITIONING=result)
