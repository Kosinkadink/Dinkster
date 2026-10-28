"""Native training loss graph preview."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_STRING,
    LOSS_MAP_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    StringWidget,
    TypeExpr,
    render_image_png,
    report_preview,
)


class LossGraphNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "LossGraphNode",
            display_name="Plot Loss Graph",
            category="model/training",
            search_terms=("training chart", "training visualization", "plot loss"),
            inputs=(
                InputSpec(
                    "loss", TypeExpr.concrete(LOSS_MAP_TYPE), doc="Loss map from training node."
                ),
                InputSpec(
                    "filename_prefix",
                    TypeExpr.concrete(CORE_STRING),
                    default="loss_graph",
                    doc="Prefix for the saved loss graph image.",
                    widget=StringWidget(multiline=False),
                ),
            ),
            output_node=True,
            emits_previews=True,
        )

    @classmethod
    def execute(cls, *, loss: Mapping[str, Any], filename_prefix: str) -> Mapping[str, object]:
        image = importlib.import_module("PIL.Image")
        image_draw = importlib.import_module("PIL.ImageDraw")
        image_font = importlib.import_module("PIL.ImageFont")
        np = importlib.import_module("numpy")
        loss_values = loss["loss"]
        width, height, margin = 800, 480, 40
        graph = image.new("RGB", (width + margin, height + margin), "white")
        draw = image_draw.Draw(graph)
        min_loss, max_loss = min(loss_values), max(loss_values)
        scaled_loss = [(value - min_loss) / (max_loss - min_loss) for value in loss_values]
        steps = len(loss_values)
        previous = (margin, height - int(scaled_loss[0] * height))
        for index, value in enumerate(scaled_loss[1:], start=1):
            point = (margin + int(index / steps * width), height - int(value * height))
            draw.line([previous, point], fill="blue", width=2)
            previous = point
        draw.line([(margin, 0), (margin, height)], fill="black", width=2)
        draw.line([(margin, height), (width + margin, height)], fill="black", width=2)
        try:
            font = image_font.truetype("arial.ttf", 12)
        except OSError:
            font = image_font.load_default()
        draw.text((5, height // 2), "Loss", font=font, fill="black")
        draw.text((width // 2, height + 10), "Steps", font=font, fill="black")
        draw.text((margin - 30, 0), f"{max_loss:.2f}", font=font, fill="black")
        draw.text((margin - 30, height - 10), f"{min_loss:.2f}", font=font, fill="black")
        pixels = np.array(graph).astype(np.float32) / 255.0
        # The fixed node ignores the prefix and publishes only a temporary preview.
        report_preview(render_image_png(pixels), mime="image/png", width=840, height=520)
        return cls.outputs()
