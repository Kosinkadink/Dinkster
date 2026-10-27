"""Native text overlay with one shared raster per image batch."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_STRING,
    IMAGE_TYPE,
    ColorWidget,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    materialize_image,
)


class TextOverlay(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="TextOverlay",
            display_name="Draw Text Overlay",
            category="text",
            description="Draw text overlay on an image or batch of images.",
            search_terms=(
                "text",
                "label",
                "caption",
                "subtitle",
                "watermark",
                "title",
                "addlabel",
                "overlay",
            ),
            inputs=(
                InputSpec("images", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec(
                    "text",
                    TypeExpr.concrete(CORE_STRING),
                    default="",
                    widget=StringWidget(multiline=True),
                ),
                InputSpec(
                    "font_size",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=5.0,
                    widget=NumberWidget(min=0.5, max=50.0, step=0.5),
                    doc="Font size as a percentage of the image height.",
                ),
                InputSpec(
                    "color",
                    TypeExpr.concrete(CORE_STRING),
                    default="#ffffff",
                    widget=ColorWidget(),
                    doc="Color of the text.",
                ),
                InputSpec(
                    "position",
                    TypeExpr.concrete(CORE_COMBO),
                    default="top",
                    widget=ComboWidget(tuple(ComboOption(value) for value in ("top", "bottom"))),
                ),
                InputSpec(
                    "align",
                    TypeExpr.concrete(CORE_COMBO),
                    default="left",
                    widget=ComboWidget(
                        tuple(ComboOption(value) for value in ("left", "center", "right"))
                    ),
                ),
                InputSpec(
                    "outline",
                    TypeExpr.concrete(CORE_BOOLEAN),
                    default=True,
                    doc="Draw a black outline around the text.",
                ),
            ),
            outputs=(OutputSpec("images", TypeExpr.concrete(IMAGE_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        images: Any,
        text: str,
        font_size: float,
        color: str,
        position: str,
        align: str,
        outline: bool,
    ) -> Mapping[str, object]:
        if text.strip() == "":
            return cls.outputs(images=images)
        images = materialize_image(images)
        torch = importlib.import_module("torch")
        image_color = importlib.import_module("PIL.ImageColor")
        text = text.replace("\\n", "\n").replace("\\t", "\t")
        parsed = image_color.getrgb(color)
        text_rgba = (*parsed, 255) if len(parsed) == 3 else parsed
        outline_rgba = (0, 0, 0, 255) if outline else (0, 0, 0, 0)
        height, width = images.shape[1:3]
        overlay_rgb, overlay_alpha = cls._render(
            width, height, text, position, align, font_size, text_rgba, outline_rgba
        )
        overlay_rgb = overlay_rgb.to(device=images.device, dtype=images.dtype)
        overlay_alpha = overlay_alpha.to(device=images.device, dtype=images.dtype)
        if images.shape[-1] == 4:
            rgb, alpha = images[..., :3], images[..., 3:]
            out_alpha = overlay_alpha + (1.0 - overlay_alpha) * alpha
            out_rgb = overlay_rgb * overlay_alpha + (1.0 - overlay_alpha) * rgb * alpha
            out_rgb = torch.where(out_alpha > 1e-5, out_rgb / out_alpha, torch.zeros_like(out_rgb))
            return cls.outputs(images=torch.cat((torch.clamp(out_rgb, 0, 1), out_alpha), dim=-1))
        return cls.outputs(images=images * (1.0 - overlay_alpha) + overlay_rgb * overlay_alpha)

    @classmethod
    def _render(
        cls,
        width: int,
        height: int,
        text: str,
        position: str,
        align: str,
        font_size: float,
        text_rgba: Any,
        outline_rgba: Any,
    ) -> tuple[Any, Any]:
        image_module = importlib.import_module("PIL.Image")
        image_draw = importlib.import_module("PIL.ImageDraw")
        image_font = importlib.import_module("PIL.ImageFont")
        np = importlib.import_module("numpy")
        torch = importlib.import_module("torch")
        layer = image_module.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = image_draw.Draw(layer)
        margin = int(round(1.0 / 100.0 * min(width, height)))
        max_width, max_height = max(1, width - 2 * margin), max(1, height - 2 * margin)
        size = max(1, int(round(font_size / 100.0 * height)))
        floor = min(size, max(10, int(round(2.0 / 100.0 * height))))
        while True:
            font = image_font.load_default(size=size)
            stroke = max(1, int(round(size * 0.04))) if outline_rgba[3] > 0 else 0
            block = "\n".join(cls._wrap(text, font, max_width))
            single = draw.textbbox((0, 0), "Ay", font=font, stroke_width=stroke)
            double = draw.multiline_textbbox(
                (0, 0), "Ay\nAy", font=font, spacing=0, stroke_width=stroke
            )
            advance = (double[3] - double[1]) - (single[3] - single[1])
            spacing = int(round(size * 1.2 - advance))
            box = draw.multiline_textbbox(
                (0, 0), block, font=font, spacing=spacing, stroke_width=stroke
            )
            if box[3] - box[1] <= max_height or size <= floor:
                break
            size = max(floor, int(size * 0.9))
        anchor, x = {
            "left": ("l", margin),
            "center": ("m", width / 2),
            "right": ("r", width - margin),
        }[align]
        y = height - margin - box[3] if position == "bottom" else margin - box[1]
        draw.multiline_text(
            (x, y),
            block,
            font=font,
            fill=text_rgba,
            anchor=anchor + "a",
            align=align,
            spacing=spacing,
            stroke_width=stroke,
            stroke_fill=outline_rgba,
        )
        overlay = np.array(layer).astype(np.float32) / 255.0
        return torch.from_numpy(overlay[:, :, :3]), torch.from_numpy(overlay[:, :, 3:4])

    @staticmethod
    def _wrap(text: str, font: Any, max_width: int) -> list[str]:
        lines: list[str] = []
        for raw_line in text.split("\n"):
            words = raw_line.split()
            if not words:
                lines.append("")
                continue
            current = ""
            for word in words:
                while font.getlength(word) > max_width and len(word) > 1:
                    cut = 1
                    while cut < len(word) and font.getlength(word[: cut + 1]) <= max_width:
                        cut += 1
                    if current:
                        lines.append(current)
                        current = ""
                    lines.append(word[:cut])
                    word = word[cut:]
                candidate = word if not current else current + " " + word
                if not current or font.getlength(candidate) <= max_width:
                    current = candidate
                else:
                    lines.append(current)
                    current = word
            if current:
                lines.append(current)
        return lines
