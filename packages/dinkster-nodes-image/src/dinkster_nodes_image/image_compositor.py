"""Layered image composition through the shared image-document renderer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from dinkster_api.v1 import (
    COMPOSITOR_TYPE,
    IMAGE_TYPE,
    LAYERS_TYPE,
    MASK_TYPE,
    CompositorWidget,
    InputSpec,
    MappingSource,
    Node,
    NodeSchema,
    OutputSpec,
    ReplacementCase,
    ReplacementRule,
    TypeExpr,
    flatten,
    report_event,
    report_preview,
)

from .layer_recipe import prepare_compositor


class ImageCompositor(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.image.image_compositor",
            display_name="Create Layered Image",
            category="image",
            editor_role="compositor",
            search_terms=(
                "ImageCompositor",
                "compositor",
                "composite",
                "layer",
                "layers",
                "layer editor",
                "psd",
            ),
            inputs=(
                InputSpec("layers", TypeExpr.concrete(LAYERS_TYPE)),
                InputSpec(
                    "compositor",
                    TypeExpr.concrete(COMPOSITOR_TYPE),
                    default={},
                    socketless=True,
                    widget=CompositorWidget(),
                ),
            ),
            outputs=(
                OutputSpec("IMAGE", TypeExpr.concrete(IMAGE_TYPE), preview=True),
                OutputSpec("MASK", TypeExpr.concrete(MASK_TYPE)),
            ),
            output_node=True,
            emits_previews=True,
            idempotent=False,
        )
        return replace(
            schema,
            replacements=(
                ReplacementRule(
                    from_type="ImageCompositor",
                    cases=(
                        ReplacementCase.build(
                            schema.node_type,
                            inputs={spec.id: MappingSource.copy(spec.id) for spec in schema.inputs},
                            outputs={spec.id: spec.id for spec in schema.outputs},
                        ),
                    ),
                ),
            ),
        )

    @classmethod
    def execute(cls, *, layers: object, compositor: object = None) -> Mapping[str, object]:
        document, input_digest, stale = prepare_compositor(layers, compositor)
        image, mask = flatten(document)
        record = document.to_record()
        streams: list[str] = []
        for index, identifier in enumerate(record["rootLayerIds"]):
            stream = f"compositor.layer.{index}"
            preview = document.render(f"layer:{identifier}")
            report_preview(
                preview.png,
                mime="image/png",
                width=preview.width,
                height=preview.height,
                stream=stream,
            )
            streams.append(stream)
        report_event(
            "dinkster.compositor.state",
            {
                "version": 2,
                "document": record,
                "documentDigest": input_digest,
                "stale": stale,
                "layerStreams": streams,
            },
        )
        return cls.outputs(IMAGE=image, MASK=mask)
