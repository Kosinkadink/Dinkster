"""MiniMax H3 audio/video sampling shifts."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_FLOAT,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)


class MiniMaxH3SigmaShift(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        model = TypeExpr.concrete(MODEL_TYPE)
        number = TypeExpr.concrete(CORE_FLOAT)
        return NodeSchema(
            node_type="MiniMaxH3SigmaShift",
            display_name="ModelSamplingMiniMaxH3",
            description="Set the video/audio flow shifts.",
            category="model/patch/minimax",
            search_terms=("sigma shift", "minimax shift"),
            inputs=(
                InputSpec("model", model),
                InputSpec(
                    "shift_video",
                    number,
                    default=12.0,
                    widget=NumberWidget(min=0.01, max=100.0, step=0.01),
                ),
                InputSpec(
                    "shift_audio",
                    number,
                    default=3.0,
                    widget=NumberWidget(min=0.01, max=100.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("MODEL", model),),
        )

    @classmethod
    def execute(cls, *, model: Any, shift_video: float, shift_audio: float) -> Mapping[str, object]:
        initialize_inference_process()
        sampling = importlib.import_module("dinkster_inference.model_sampling")
        result = model.clone()

        class ModelSamplingAdvanced(sampling.ModelSamplingAV, sampling.CONST):
            pass

        original = result.get_model_object("model_sampling")
        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift_video, audio_shift=shift_audio)
        if hasattr(original, "noise_scale"):
            model_sampling.set_noise_scale(original.noise_scale)
        result.add_object_patch("model_sampling", model_sampling)
        options = result.model_options["transformer_options"] = result.model_options.get(
            "transformer_options", {}
        ).copy()
        options["minimax_h3_sigma_shift_video"] = shift_video
        options["minimax_h3_sigma_shift_audio"] = shift_audio
        return cls.outputs(MODEL=result)
