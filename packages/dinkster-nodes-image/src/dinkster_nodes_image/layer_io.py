"""Layered-file import and library export nodes."""

from __future__ import annotations

import os
from collections.abc import Mapping

from dinkster_api.v1 import (
    ASSET_TYPE,
    SAVE_TARGET_TYPE,
    AssetRef,
    AssetWidget,
    AssetWriter,
    ComboWidget,
    InputSpec,
    MountSnapshotWriter,
    Node,
    NodeSchema,
    OutputSpec,
    SaveTargetWidget,
    TypeExpr,
)
from dinkster_image_document import IMAGE_DOCUMENT_MEDIA_TYPE
from dinkster_image_document.document import ImageDocument
from dinkster_image_document.interchange import export_ora, export_psd, load_document

LAYERS = TypeExpr.concrete("dinkster.layers")
ASSET = TypeExpr.concrete(ASSET_TYPE)


class LoadLayers(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.layers.load",
            display_name="Load Layers",
            category="image/io",
            inputs=(InputSpec("document", ASSET, widget=AssetWidget()),),
            outputs=(OutputSpec("layers", LAYERS),),
        )

    @classmethod
    def execute(cls, *, document: AssetRef) -> Mapping[str, object]:
        loaded = load_document(document.read_bytes())
        return cls.outputs(
            layers=ImageDocument(
                loaded.data, lambda wire: AssetRef.from_wire(wire, document.resolver)
            )
        )


class SaveLayers(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.layers.save",
            display_name="Save Layers",
            category="image/io",
            inputs=(
                InputSpec("layers", LAYERS),
                InputSpec("target", TypeExpr.concrete(SAVE_TARGET_TYPE), widget=SaveTargetWidget()),
                InputSpec(
                    "format",
                    TypeExpr.concrete("core.combo"),
                    required=False,
                    default="native",
                    widget=ComboWidget(options=("native", "ora", "psd")),
                ),
            ),
            outputs=(OutputSpec("document", ASSET),),
            output_node=True,
            idempotent=False,
        )

    @classmethod
    def execute(
        cls, *, layers: ImageDocument, target: object, format: str = "native"
    ) -> Mapping[str, object]:
        if format not in ("native", "ora", "psd"):
            raise ValueError("layer export format must be native, ora, or psd")
        data = (
            layers.data
            if format == "native"
            else export_ora(layers)
            if format == "ora"
            else export_psd(layers)
        )
        suffix, media_type = {
            "native": (".json", IMAGE_DOCUMENT_MEDIA_TYPE),
            "ora": (".ora", "image/openraster"),
            "psd": (".psd", "image/vnd.adobe.photoshop"),
        }[format]
        reference = AssetWriter(
            MountSnapshotWriter(os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", ""))
        ).save_bytes(
            target,
            data,
            suffix=suffix,
            media_type=media_type,
        )
        return cls.outputs(document=reference)
