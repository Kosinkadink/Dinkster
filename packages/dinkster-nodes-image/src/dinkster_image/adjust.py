"""Deterministic pointwise image adjustments."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from dinkster_api.v1 import (
    DynamicComboOption,
    DynamicComboSpec,
    InputSpec,
    MirrorSpec,
    MirrorTolerance,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
)

from .geometry import FLOAT, IMAGE
from .migration import with_v1_migration
from .support import image_array as _image_array
from .support import materialized_inputs

ADJUST_OPERATIONS = ("invert", "normalize", "brightness", "contrast")

ADJUST_MIRROR_PER_CHANNEL_TOLERANCE = 1.0 / 255.0

# The operation branches index ADJUST_OPERATIONS in declared order; the
# uniform names follow the schema binding contract in the dinkster-schema README.
ADJUST_MIRROR_SOURCE = """\
#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D u_image;
uniform int operation;
uniform float factor;
uniform float mean;
uniform float standard_deviation;

out vec4 fragColor;

void main() {
    vec4 x = texelFetch(u_image, ivec2(gl_FragCoord.xy), 0);
    if (operation == 0) {
        fragColor = 1.0 - x;
    } else if (operation == 1) {
        fragColor = (x - mean) / standard_deviation;
    } else if (operation == 2) {
        fragColor = clamp(x * factor, 0.0, 1.0);
    } else {
        fragColor = clamp((x - 0.5) * factor + 0.5, 0.0, 1.0);
    }
}
"""


class ImageAdjust(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        factor = InputSpec(
            "factor",
            FLOAT,
            required=False,
            default=1.0,
            widget=NumberWidget(min=0.0, max=10.0, step=0.01),
        )
        mean = InputSpec(
            "mean",
            FLOAT,
            required=False,
            default=0.5,
            widget=NumberWidget(min=-10.0, max=10.0, step=0.01),
        )
        standard_deviation = InputSpec(
            "standard_deviation",
            FLOAT,
            required=False,
            default=0.5,
            widget=NumberWidget(min=0.001, max=10.0, step=0.001),
        )
        return with_v1_migration(
            NodeSchema(
                node_type="dinkster.image.adjust",
                version=2,
                display_name="Adjust Image",
                category="image/adjust",
                inputs=(InputSpec("image", IMAGE),),
                combos=(
                    DynamicComboSpec(
                        "operation",
                        tuple(
                            DynamicComboOption(name, inputs)
                            for name, inputs in (
                                ("invert", ()),
                                ("normalize", (mean, standard_deviation)),
                                ("brightness", (factor,)),
                                ("contrast", (factor,)),
                            )
                        ),
                        default="brightness",
                    ),
                ),
                outputs=(OutputSpec("image", IMAGE, preview=True),),
                search_terms=(
                    "invert image",
                    "normalize image",
                    "brightness",
                    "contrast",
                    "ImageInvert",
                    "NormalizeImages",
                ),
                # Bounded because GLSL ES 3.00 does not guarantee correctly
                # rounded arithmetic and clients run on heterogeneous GPU float
                # pipelines; one 8-bit preview quantization step is far above the
                # worst-case drift of these pointwise operations. The parity
                # corpus is tests/fixtures/mirror-parity/image_adjust_v1.json.
                mirror=MirrorSpec(
                    kind="glsl",
                    precision="bounded",
                    tolerance=MirrorTolerance(per_channel=ADJUST_MIRROR_PER_CHANNEL_TOLERANCE),
                    source=ADJUST_MIRROR_SOURCE,
                ),
            )
        )

    @classmethod
    @materialized_inputs
    def execute(
        cls,
        *,
        image: object,
        operation: str = "brightness",
        factor: float = 1.0,
        mean: float = 0.5,
        standard_deviation: float = 0.5,
    ) -> Mapping[str, object]:
        if not all(math.isfinite(value) for value in (factor, mean, standard_deviation)):
            raise ValueError("adjustment parameters must be finite")
        array = _image_array(image)
        if operation == "invert":
            output = 1.0 - array
        elif operation == "normalize":
            if standard_deviation <= 0.0:
                raise ValueError("standard_deviation must be positive")
            output = (array - mean) / standard_deviation
        elif operation == "brightness":
            if factor < 0.0:
                raise ValueError("brightness factor must be non-negative")
            output = np.clip(array * factor, 0.0, 1.0)
        elif operation == "contrast":
            if factor < 0.0:
                raise ValueError("contrast factor must be non-negative")
            output = np.clip((array - 0.5) * factor + 0.5, 0.0, 1.0)
        else:
            raise ValueError(f"unknown image adjustment: {operation}")
        return cls.outputs(image=np.ascontiguousarray(output, dtype=np.float32))


ADJUST_NODES: tuple[type[Node], ...] = (ImageAdjust,)


__all__ = [
    "ADJUST_MIRROR_PER_CHANNEL_TOLERANCE",
    "ADJUST_MIRROR_SOURCE",
    "ADJUST_NODES",
    "ADJUST_OPERATIONS",
    "ImageAdjust",
]
