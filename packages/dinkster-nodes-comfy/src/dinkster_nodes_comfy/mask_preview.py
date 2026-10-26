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
)

_MASK = TypeExpr.concrete(MASK_TYPE)


class MaskPreview(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MaskPreview",
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
        )

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
