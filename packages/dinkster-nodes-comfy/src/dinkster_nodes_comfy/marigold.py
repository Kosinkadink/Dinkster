"""Convert decoded Marigold predictions into display images."""

from __future__ import annotations

import importlib
from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_COMBO,
    IMAGE_TYPE,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
)


class MarigoldV2PostProcess(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MarigoldV2PostProcess",
            display_name="Marigold V2 Post-Process",
            category="image/geometry estimation",
            description=(
                "Turns a decoded Marigold V2 prediction into an image: normalized depth "
                "with near as bright, unit surface normals, or sRGB albedo."
            ),
            inputs=(
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec(
                    "prediction",
                    TypeExpr.concrete(CORE_COMBO),
                    widget=ComboWidget(
                        tuple(ComboOption(name) for name in ("depth", "normals", "albedo"))
                    ),
                ),
            ),
            outputs=(OutputSpec("IMAGE", TypeExpr.concrete(IMAGE_TYPE)),),
        )

    @classmethod
    def execute(cls, *, image: object, prediction: str) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        samples = materialize_image(image)
        if prediction == "normals":
            output = (torch.nn.functional.normalize(samples * 2.0 - 1.0, dim=-1) + 1.0) * 0.5
        elif prediction == "albedo":
            output = torch.where(
                samples <= 0.0031308,
                samples * 12.92,
                1.055 * samples.clamp(min=0.0031308) ** (1.0 / 2.4) - 0.055,
            )
        else:
            depth = samples.mean(dim=-1, keepdim=True)
            low = depth.amin(dim=(1, 2, 3), keepdim=True)
            high = depth.amax(dim=(1, 2, 3), keepdim=True)
            output = ((high - depth) / (high - low).clamp(min=1e-6)).repeat(1, 1, 1, 3)
        return cls.outputs(IMAGE=output)
