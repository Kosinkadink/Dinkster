from __future__ import annotations

import importlib
from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_INT,
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    with_identity_replacement,
)


class EmptyMiniMaxMusic3LatentAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.minimax.empty_mini_max_music3_latent_audio",
            display_name="Empty MiniMax Music3 Latent Audio",
            category="model/latent/minimax music",
            description="Creates an empty MiniMax Music3 audio latent for the requested duration.",
            inputs=(
                InputSpec(
                    "seconds",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=120.0,
                    widget=NumberWidget(min=0.04, max=360.0, step=0.04),
                ),
                InputSpec(
                    "batch_size",
                    TypeExpr.concrete(CORE_INT),
                    default=1,
                    widget=NumberWidget(min=1, max=4096),
                ),
            ),
            outputs=(OutputSpec("LATENT", TypeExpr.concrete(LATENT_TYPE)),),
        )
        return with_identity_replacement(schema, "EmptyMiniMaxMusic3LatentAudio")

    @classmethod
    def execute(cls, *, seconds: float, batch_size: int) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        ar = importlib.import_module("dinkster_inference.ldm.minimax_music.ar")
        dit = importlib.import_module("dinkster_inference.ldm.minimax_music.dit")
        frames = min(ar.MAX_AUDIO_FRAMES, max(1, round(seconds * ar.AUDIO_FRAMES_PER_SECOND)))
        latent = torch.zeros(
            (batch_size, 128, dit.latent_length(frames)),
            device=management.intermediate_device(),
            dtype=management.intermediate_dtype(),
        )
        return cls.outputs(
            LATENT={"samples": latent, "type": "audio", "downscale_ratio_temporal": 512}
        )
