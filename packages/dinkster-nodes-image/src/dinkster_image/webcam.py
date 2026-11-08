"""Decode browser-captured camera frames from host-granted image assets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_INT,
    IMAGE_TYPE,
    IMAGE_WORKING_SET_LIMIT_BYTES,
    WEBCAM_TYPE,
    AssetRef,
    CustomWidgetDescriptor,
    InputSpec,
    LocalAssetRef,
    MappingSource,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    ReplacementCase,
    ReplacementRule,
    TypeExpr,
    open_image_components,
)


class WebcamCapture(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.image.webcam_capture",
            display_name="Webcam Capture",
            category="image",
            search_terms=(
                "WebcamCapture",
                "camera input",
                "live capture",
                "camera feed",
                "snapshot",
            ),
            inputs=(
                InputSpec(
                    "image", TypeExpr.concrete(WEBCAM_TYPE), widget=CustomWidgetDescriptor("WEBCAM")
                ),
                *(
                    InputSpec(
                        name,
                        TypeExpr.concrete(CORE_INT),
                        default=0,
                        widget=NumberWidget(min=0, max=16384, step=1),
                    )
                    for name in ("width", "height")
                ),
                InputSpec("capture_on_queue", TypeExpr.concrete(CORE_BOOLEAN), default=True),
            ),
            outputs=(OutputSpec("IMAGE", TypeExpr.concrete(IMAGE_TYPE)),),
        )
        return replace(
            schema,
            replacements=(
                ReplacementRule(
                    from_type="WebcamCapture",
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
    def execute(
        cls,
        *,
        image: object,
        width: int = 0,
        height: int = 0,
        capture_on_queue: bool = True,
    ) -> Mapping[str, object]:
        # Dimensions and queue capture configure the browser, not the saved frame decoder.
        if not isinstance(image, (AssetRef, LocalAssetRef)):
            raise TypeError("image must be an AssetRef or LocalAssetRef")
        with open_image_components(
            image,
            max_dimension=16384,
            max_source_bytes=IMAGE_WORKING_SET_LIMIT_BYTES,
            max_working_set_bytes=IMAGE_WORKING_SET_LIMIT_BYTES // 2,
        ) as frames:
            frame = next(iter(frames))
            pixels = np.ascontiguousarray(frame.pixels)[None]
        return cls.outputs(IMAGE=pixels)
