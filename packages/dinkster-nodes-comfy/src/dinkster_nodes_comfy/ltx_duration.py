"""Predict LTXV shot duration with the pinned caption duration head."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_FLOAT,
    CORE_INT,
    MODEL_PATCH_TYPE,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)


class LTXVDurationPredictor(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        floating = TypeExpr.concrete(CORE_FLOAT)
        return NodeSchema(
            node_type="LTXVDurationPredictor",
            display_name="LTXV Duration Predictor",
            category="model/conditioning/ltxv",
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec(
                    "duration_head",
                    TypeExpr.concrete(MODEL_PATCH_TYPE),
                    doc="LTX 2.4 duration head loaded with ModelPatchLoader.",
                ),
                InputSpec(
                    "frame_rate",
                    floating,
                    default=24.0,
                    widget=NumberWidget(min=1.0, max=120.0, step=0.01),
                ),
                InputSpec(
                    "min_seconds",
                    floating,
                    default=1.0,
                    widget=NumberWidget(min=0.5, max=120.0, step=0.1),
                ),
                InputSpec(
                    "max_seconds",
                    floating,
                    default=20.0,
                    widget=NumberWidget(min=0.5, max=120.0, step=0.1),
                ),
            ),
            outputs=(
                OutputSpec("num_frames", TypeExpr.concrete(CORE_INT)),
                OutputSpec("seconds", floating, doc="Raw (unclamped) predicted duration."),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        positive: Any,
        duration_head: Any,
        frame_rate: float,
        min_seconds: float,
        max_seconds: float,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        duration = importlib.import_module("dinkster_inference.ldm.lightricks.duration_head")
        management = importlib.import_module("dinkster_inference.model_management")
        torch = importlib.import_module("torch")
        diffusion = model.model.diffusion_model
        head = duration_head.model
        if not isinstance(head, duration.DurationHead):
            raise ValueError("The connected model_patch is not an LTX duration head.")
        context, metadata = positive[0]
        if context.shape[0] != 1:
            context = context[:1]
        management.load_models_gpu([model, duration_head])
        device = model.load_device
        head = head.to(device)
        with torch.no_grad():
            context = context.to(device=device, dtype=model.model.get_dtype_inference())
            processed = diffusion.preprocess_text_embeds(
                context, unprocessed=metadata.get("unprocessed_ltxav_embeds", False)
            )
            video = processed[..., : diffusion.cross_attention_dim].float()
            audio = processed[..., diffusion.cross_attention_dim :].float()
            seconds = float(head(video, audio)[0])
        frames = duration.seconds_to_num_frames(seconds, frame_rate, min_seconds, max_seconds)
        return cls.outputs(num_frames=frames, seconds=seconds)
