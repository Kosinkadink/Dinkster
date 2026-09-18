"""Paired image previews with independent optional sides."""

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from dinkster_api.v1 import ABSENT, InputSpec, Node, NodeSchema, OutputSpec, copy_media_semantics

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
