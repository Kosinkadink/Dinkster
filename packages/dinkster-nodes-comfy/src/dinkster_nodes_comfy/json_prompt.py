"""Structured Ideogram prompts using native array and color inputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    ARRAY_TYPE,
    COLORS_TYPE,
    CORE_STRING,
    DICT_TYPE,
    DynamicComboOption,
    DynamicComboSpec,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    StringWidget,
    TypeExpr,
)

_STRING = TypeExpr.concrete(CORE_STRING)


class BuildJsonPromptIdeogram(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="BuildJsonPromptIdeogram",
            display_name="Build JSON Prompt (Ideogram)",
            category="text",
            description="Build a JSON prompt for the Ideogram 4 model.",
            inputs=(
                InputSpec("element", TypeExpr.concrete(ARRAY_TYPE)),
                InputSpec(
                    "high_level_description",
                    _STRING,
                    default="",
                    widget=StringWidget(multiline=True),
                ),
                InputSpec("background", _STRING, default="", widget=StringWidget(multiline=True)),
                InputSpec("aesthetics", _STRING, default="", widget=StringWidget(multiline=False)),
                InputSpec("lighting", _STRING, default="", widget=StringWidget(multiline=False)),
                InputSpec("medium", _STRING, default="", widget=StringWidget(multiline=False)),
                InputSpec("color_palette", TypeExpr.concrete(COLORS_TYPE), default=[]),
            ),
            combos=(
                DynamicComboSpec(
                    "style",
                    (
                        DynamicComboOption("none"),
                        DynamicComboOption(
                            "photo",
                            (
                                InputSpec(
                                    "photo",
                                    _STRING,
                                    default="",
                                    widget=StringWidget(multiline=False),
                                ),
                            ),
                        ),
                        DynamicComboOption(
                            "art_style",
                            (
                                InputSpec(
                                    "art_style",
                                    _STRING,
                                    default="",
                                    widget=StringWidget(multiline=False),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            input_order=(
                "element",
                "high_level_description",
                "background",
                "style",
                "aesthetics",
                "lighting",
                "medium",
                "color_palette",
            ),
            outputs=(OutputSpec("prompt", TypeExpr.concrete(DICT_TYPE), display_name="prompt"),),
        )

    @classmethod
    def execute(
        cls,
        *,
        element: Any,
        style: str,
        high_level_description: str = "",
        background: str = "",
        aesthetics: str = "",
        lighting: str = "",
        medium: str = "",
        color_palette: Any = None,
        **inputs: Any,
    ) -> Mapping[str, object]:
        colors: Any = color_palette or []
        if isinstance(colors, dict):
            colors = cast("dict[object, Any]", colors).values()
        palette = [
            color.upper()
            for color in cast("Iterable[Any]", colors)
            if isinstance(color, str) and color
        ]
        caption: dict[str, Any] = {}
        if high_level_description.strip():
            caption["high_level_description"] = high_level_description
        if style != "none":
            description: dict[str, Any] = {"aesthetics": aesthetics, "lighting": lighting}
            if style == "photo":
                description["photo"] = inputs.get("style.photo", "")
                description["medium"] = medium
            else:
                description["medium"] = medium
                description["art_style"] = inputs.get("style.art_style", "")
            if palette:
                description["color_palette"] = palette
            caption["style_description"] = description
        caption["compositional_deconstruction"] = {
            "background": background,
            "elements": element if isinstance(element, list) else [],
        }
        return cls.outputs(prompt=caption)
