"""Native Flux image scaling and layout prompt nodes."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from typing import cast

from dinkster_api.v1 import (
    ARRAY_TYPE,
    CORE_STRING,
    IMAGE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    StringWidget,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)

_ARRAY = TypeExpr.concrete(ARRAY_TYPE)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_STRING = TypeExpr.concrete(CORE_STRING)

_PREFERRED_KONTEXT_RESOLUTIONS = (
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
)


class FluxKontextImageScale(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="FluxKontextImageScale",
            category="model/conditioning/flux",
            description="This node resizes the image to one that is more optimal for flux kontext.",
            inputs=(InputSpec("image", _IMAGE),),
            outputs=(OutputSpec("IMAGE", _IMAGE),),
        )

    @classmethod
    def execute(cls, *, image: object) -> Mapping[str, object]:
        value = materialize_image(image)
        aspect_ratio = value.shape[2] / value.shape[1]
        _, width, height = min(
            (abs(aspect_ratio - width / height), width, height)
            for width, height in _PREFERRED_KONTEXT_RESOLUTIONS
        )
        initialize_inference_process()
        utils = importlib.import_module("dinkster_inference.utils")
        return cls.outputs(
            IMAGE=utils.common_upscale(
                value.movedim(-1, 1), width, height, "lanczos", "center"
            ).movedim(1, -1)
        )


class Flux3LayoutPrompt(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="Flux3LayoutPrompt",
            display_name="FLUX 3 Image Layout Prompt",
            category="model/conditioning/flux",
            description=(
                "Appends the boxes from Create Bounding Boxes to a FLUX 3 Image prompt as its "
                "layout table."
            ),
            inputs=(
                InputSpec(
                    "prompt",
                    _STRING,
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                    doc="The prompt text. Cite box N as <obj_N> to tie words to it.",
                ),
                InputSpec(
                    "element",
                    _ARRAY,
                    doc=(
                        "Prompt elements from the node Create Bounding Boxes. Colour palettes are "
                        "not used, describe colours in the description."
                    ),
                ),
            ),
            outputs=(OutputSpec("prompt", _STRING, display_name="prompt"),),
        )

    @classmethod
    def execute(cls, *, prompt: str, element: object) -> Mapping[str, object]:
        if not isinstance(element, list):
            return cls.outputs(prompt=prompt)
        rows: list[dict[str, object]] = []
        for index, item in enumerate(cast("list[Mapping[str, object]]", element)):
            description = item.get("desc", "")
            if item.get("type") == "text" and item.get("text"):
                description = f'{description} "{item["text"]}"'.strip()
            rows.append({"id": f"obj_{index + 1}", "bbox": item["bbox"], "desc": description})
        if not rows:
            return cls.outputs(prompt=prompt)
        return cls.outputs(prompt=f"{prompt.rstrip()} {json.dumps(rows, ensure_ascii=False)}")


FLUX_IMAGE_LAYOUT_NODES = (FluxKontextImageScale, Flux3LayoutPrompt)
