"""Native autoregressive video sampler construction."""

from __future__ import annotations

import importlib
from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_INT,
    SAMPLER_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)


class SamplerARVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SamplerARVideo",
            display_name="Sampler AR Video",
            category="model/sampling/samplers",
            inputs=(
                InputSpec(
                    "num_frame_per_block",
                    TypeExpr.concrete(CORE_INT),
                    default=1,
                    widget=NumberWidget(min=1, max=64),
                    doc=(
                        "Frames per autoregressive block. 1 = framewise, "
                        "3 = chunkwise. Must match the checkpoint's training mode."
                    ),
                ),
            ),
            outputs=(OutputSpec("SAMPLER", TypeExpr.concrete(SAMPLER_TYPE)),),
        )

    @classmethod
    def execute(cls, *, num_frame_per_block: int) -> Mapping[str, object]:
        initialize_inference_process()
        samplers = importlib.import_module("dinkster_inference.samplers")
        return cls.outputs(
            SAMPLER=samplers.ksampler("ar_video", {"num_frame_per_block": num_frame_per_block})
        )
