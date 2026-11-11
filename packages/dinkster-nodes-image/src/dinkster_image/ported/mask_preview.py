"""Native ComfyUI mask preview node."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    MASK_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
    render_mask_png,
    report_preview,
    with_identity_replacement,
)

_MASK = TypeExpr.concrete(MASK_TYPE)


class MaskPreview(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.image.mask_preview",
            display_name="Preview Mask",
            description="Preview the masks without saving them to the ComfyUI output directory.",
            category="image/mask",
            search_terms=(
                "preview",
                "preview mask",
                "show mask",
                "view mask",
                "inspect mask",
                "debug mask",
            ),
            inputs=(InputSpec("mask", _MASK),),
            outputs=(OutputSpec("mask", _MASK, preview=True),),
            emits_previews=True,
            output_node=True,
        )
        return with_identity_replacement(schema, "MaskPreview")

    @classmethod
    def execute(cls, *, mask: Any) -> Mapping[str, object]:
        mask = materialize_image(mask)
        report_preview(
            render_mask_png(mask),
            mime="image/png",
            width=int(mask.shape[2]),
            height=int(mask.shape[1]),
        )
        return cls.outputs(mask=mask)
