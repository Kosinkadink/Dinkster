"""ComfyUI image and latent dimension inspection nodes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    first_party_schema,
    materialize_image,
    report_progress,
)

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_INT = TypeExpr.concrete(CORE_INT)


class GetImageSize(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="GetImageSize",
            display_name="Get Image Size",
            description="Returns the width, height, and batch size of the image.",
            category="image",
            inputs=(InputSpec("image", _IMAGE),),
            outputs=(
                OutputSpec("width", _INT),
                OutputSpec("height", _INT),
                OutputSpec("batch_size", _INT),
            ),
            search_terms=("dimensions", "resolution", "image info"),
        )
        return first_party_schema(schema)

    @classmethod
    def execute(cls, *, image: Any) -> Mapping[str, object]:
        image = materialize_image(image)
        batch_size, height, width = image.shape[:3]
        report_progress(1, 1, text=f"width: {width}, height: {height}\n batch size: {batch_size}")
        return cls.outputs(width=width, height=height, batch_size=batch_size)
