"""Asset-backed layer document import, export, and compatibility rendering."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    ASSET_TYPE,
    AssetRef,
    AssetWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    StringWidget,
    TypeExpr,
)
from dinkster_image_document import IMAGE_DOCUMENT_MEDIA_TYPE
from dinkster_image_document.document import ImageDocument, flatten

IMAGE = TypeExpr.concrete("dinkster.image")
ASSET = TypeExpr.concrete(ASSET_TYPE)
STRING = TypeExpr.concrete("core.string")


class RenderImageDocument(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.render_image_document",
            display_name="Render Image Document",
            category="image/io",
            description="Deprecated: use Load Layers and Flatten Layers.",
            inputs=(
                InputSpec("document", ASSET, widget=AssetWidget((IMAGE_DOCUMENT_MEDIA_TYPE,))),
                InputSpec(
                    "selector", STRING, required=False, default="composite", widget=StringWidget()
                ),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True),),
        )

    @classmethod
    def execute(cls, *, document: AssetRef, selector: str = "composite") -> Mapping[str, object]:
        value = ImageDocument(
            document.read_bytes(), lambda wire: AssetRef.from_wire(wire, document.resolver)
        )
        image, _ = flatten(value, selector)
        return cls.outputs(image=image)


IMAGE_DOCUMENT_NODES: tuple[type[Node], ...] = (RenderImageDocument,)
