"""Paired image previews with independent optional sides."""

from collections.abc import Mapping
from io import BytesIO
from typing import Any, cast

import numpy as np
from dinkster_api.v1 import (
    ABSENT,
    IMAGECOMPARE_TYPE,
    CustomWidgetDescriptor,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    copy_media_semantics,
    report_preview,
)
from PIL import Image

from .geometry import IMAGE
from .support import image_array


class ImageCompare(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.image.compare",
            display_name="Compare Images",
            category="image",
            inputs=tuple(
                InputSpec(name, IMAGE, required=False, default=None)
                for name in ("image_a", "image_b")
            ),
            outputs=tuple(
                OutputSpec(name, IMAGE, optional=True, preview=True)
                for name in ("image_a", "image_b")
            ),
            output_node=True,
            search_terms=("compare images", "before after"),
        )

    @classmethod
    def execute(cls, *, image_a: object = None, image_b: object = None) -> Mapping[str, object]:
        previews: dict[str, object] = {}
        for name, value in (("image_a", image_a), ("image_b", image_b)):
            array = cast("np.ndarray[Any, Any]", value) if isinstance(value, np.ndarray) else None
            empty = array is not None and array.ndim == 4 and array.shape[0] == 0
            previews[name] = (
                ABSENT
                if value is None or empty
                else copy_media_semantics(
                    cast("object", value), image_array(cast("object", value), subject=name)
                )
            )
        return cls.outputs(**previews)


class NativeImageCompare(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ImageCompare",
            display_name="Compare Images",
            category="image",
            inputs=(
                *ImageCompare.schema().inputs,
                InputSpec(
                    "compare_view",
                    TypeExpr.concrete(IMAGECOMPARE_TYPE),
                    default=None,
                    socketless=True,
                    widget=CustomWidgetDescriptor("IMAGECOMPARE"),
                ),
            ),
            output_node=True,
            emits_previews=True,
            idempotent=False,
        )

    @classmethod
    def execute(
        cls, *, image_a: object = None, image_b: object = None, compare_view: object = None
    ) -> Mapping[str, object]:
        images = ImageCompare.execute(image_a=image_a, image_b=image_b)
        for name, value in images.items():
            if value is ABSENT:
                continue
            batch = image_array(value, subject=name)
            for index in range(batch.shape[0]):
                raster = np.clip(batch[index] * 255.0, 0, 255).astype(np.uint8)
                encoded = BytesIO()
                Image.fromarray(raster).save(encoded, format="PNG")
                report_preview(
                    encoded.getvalue(),
                    mime="image/png",
                    width=int(batch.shape[2]),
                    height=int(batch.shape[1]),
                    stream=name,
                    frame_index=index,
                    frame_count=int(batch.shape[0]),
                )
        return cls.outputs()
