"""Native ComfyUI image crop and bounding-box primitive nodes."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    BOUNDING_BOX_TYPE,
    CORE_INT,
    IMAGE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    with_identity_replacement,
)

_BOUNDING_BOX = TypeExpr.concrete(BOUNDING_BOX_TYPE)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
_MAX_RESOLUTION = 16_384


class PrimitiveBoundingBox(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.utilities.primitive_bounding_box",
            display_name="Bounding Box",
            category="utilities/primitive",
            inputs=(
                InputSpec("x", _INT, default=0, widget=NumberWidget(min=0, max=_MAX_RESOLUTION)),
                InputSpec("y", _INT, default=0, widget=NumberWidget(min=0, max=_MAX_RESOLUTION)),
                InputSpec(
                    "width", _INT, default=512, widget=NumberWidget(min=1, max=_MAX_RESOLUTION)
                ),
                InputSpec(
                    "height", _INT, default=512, widget=NumberWidget(min=1, max=_MAX_RESOLUTION)
                ),
            ),
            outputs=(OutputSpec("BOUNDING_BOX", _BOUNDING_BOX),),
        )
        return with_identity_replacement(schema, "PrimitiveBoundingBox")

    @classmethod
    def execute(cls, *, x: int, y: int, width: int, height: int) -> Mapping[str, object]:
        return cls.outputs(BOUNDING_BOX={"x": x, "y": y, "width": width, "height": height})
