"""Native color and resolution utilities from the fixed ComfyUI reference."""

from __future__ import annotations

import math
from collections.abc import Mapping

from dinkster_api.v1 import (
    COLOR_TYPE,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    RESOLUTION_PREVIEW_TYPE,
    ColorWidget,
    ComboOption,
    ComboWidget,
    CustomWidgetDescriptor,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
)

_ASPECT_RATIOS = {
    "1:1 (Square)": (1, 1),
    "2:3 (Portrait Photo)": (2, 3),
    "3:2 (Photo)": (3, 2),
    "3:4 (Portrait Standard)": (3, 4),
    "4:3 (Standard)": (4, 3),
    "9:16 (Portrait Widescreen)": (9, 16),
    "16:9 (Widescreen)": (16, 9),
    "21:9 (Ultrawide)": (21, 9),
}
_INT = TypeExpr.concrete(CORE_INT)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_COLOR = TypeExpr.concrete(COLOR_TYPE)


class ColorToRGBInt(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "ColorToRGBInt",
            display_name="Color Picker",
            category="utilities",
            description="Return a color RGB integer value and hexadecimal representation.",
            inputs=(InputSpec("color", _COLOR, default="#ffffff", widget=ColorWidget()),),
            outputs=(
                OutputSpec("rgb_int", _INT),
                OutputSpec("hex", _COLOR),
                OutputSpec("alpha", _FLOAT),
            ),
        )

    @classmethod
    def execute(cls, *, color: str) -> Mapping[str, object]:
        message = "Color must be in format #RRGGBB or #RRGGBBAA"
        if len(color) not in (7, 9) or color[0] != "#":
            raise ValueError(message)
        try:
            int(color[1:], 16)
        except ValueError:
            raise ValueError(message) from None
        alpha = int(color[7:9], 16) / 255.0 if len(color) == 9 else 1.0
        color = color[:7]
        try:
            red, green, blue = (int(color[index : index + 2], 16) for index in (1, 3, 5))
        except ValueError:
            # The reference color helper falls back to white after whole-string validation.
            red, green, blue = 255, 255, 255
        return cls.outputs(rgb_int=red * 256 * 256 + green * 256 + blue, hex=color, alpha=alpha)


class ResolutionSelector(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "ResolutionSelector",
            display_name="Resolution Selector",
            category="utilities",
            description=(
                "Calculate width and height from aspect ratio and megapixel target. Useful for "
                "setting up Empty Latent Image dimensions."
            ),
            inputs=(
                InputSpec(
                    "aspect_ratio",
                    TypeExpr.concrete(CORE_COMBO),
                    default="1:1 (Square)",
                    doc="The aspect ratio for the output dimensions.",
                    widget=ComboWidget(tuple(ComboOption(value) for value in _ASPECT_RATIOS)),
                ),
                InputSpec(
                    "megapixels",
                    _FLOAT,
                    default=1.0,
                    doc="Target total megapixels. 1.0 MP is approximately 1024x1024 for square.",
                    widget=NumberWidget(min=0.1, max=16.0, step=0.1),
                ),
                InputSpec(
                    "preview",
                    TypeExpr.concrete(RESOLUTION_PREVIEW_TYPE),
                    required=False,
                    default=None,
                    doc="Live preview of the calculated output resolution.",
                    widget=CustomWidgetDescriptor(
                        "RESOLUTION_PREVIEW",
                        {
                            "ratio_widget": "aspect_ratio",
                            "megapixels_widget": "megapixels",
                            "multiple_widget": "multiple",
                        },
                    ),
                ),
                InputSpec(
                    "multiple",
                    _INT,
                    default=8,
                    advanced=True,
                    doc="Nearest multiple of the result to set the selected resolution to.",
                    widget=NumberWidget(min=8, max=128, step=4),
                ),
            ),
            outputs=(
                OutputSpec(
                    "width",
                    _INT,
                    doc="Calculated width in pixels multiplied by the selected multiple.",
                ),
                OutputSpec(
                    "height",
                    _INT,
                    doc="Calculated height in pixels multiplied by the selected multiple.",
                ),
            ),
        )

    @classmethod
    def execute(
        cls, *, aspect_ratio: str, megapixels: float, multiple: int, preview: object = None
    ) -> Mapping[str, object]:
        width_ratio, height_ratio = _ASPECT_RATIOS[aspect_ratio]
        scale = math.sqrt(megapixels * 1024 * 1024 / (width_ratio * height_ratio))
        return cls.outputs(
            width=round(width_ratio * scale / multiple) * multiple,
            height=round(height_ratio * scale / multiple) * multiple,
        )


COLOR_RESOLUTION_NODES = (ColorToRGBInt, ResolutionSelector)
