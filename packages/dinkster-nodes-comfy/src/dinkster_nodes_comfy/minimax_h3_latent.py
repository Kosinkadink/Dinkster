"""Native MiniMax H3 joint audio and video latent construction."""

from __future__ import annotations

import importlib
from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_INT,
    LATENT_TYPE,
    InputSpec,
    LatentStream,
    MultiStreamLatent,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)


class EmptyMiniMaxH3LatentAV(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        integer = TypeExpr.concrete(CORE_INT)
        return NodeSchema(
            node_type="EmptyMiniMaxH3LatentAV",
            display_name="Empty MiniMax H3 AV Latent",
            category="model/latent/minimax",
            description=(
                "Joint video+audio latent for MiniMax H3. Duration snaps to the model's "
                "17k+5 frame grid at 24 fps."
            ),
            inputs=(
                InputSpec(
                    "width",
                    integer,
                    default=1344,
                    widget=NumberWidget(min=32, max=16384, step=32),
                ),
                InputSpec(
                    "height",
                    integer,
                    default=768,
                    widget=NumberWidget(min=32, max=16384, step=32),
                ),
                InputSpec(
                    "length",
                    integer,
                    default=124,
                    widget=NumberWidget(min=5, max=3600, step=17),
                    doc=(
                        "Frame count at 24 fps, snapped up to the model's 17k+5 grid "
                        "(124 = ~5s; trained range is ~124-362, longer is untested)"
                    ),
                ),
            ),
            outputs=(OutputSpec("LATENT", TypeExpr.concrete(LATENT_TYPE)),),
        )

    @classmethod
    def execute(cls, *, width: int, height: int, length: int) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        frame_count = max(5, length)
        frame_count += (5 - frame_count) % 17
        video_frames = 2 + ((frame_count - 5) // 17) * 5
        audio_frames = round(frame_count / 24 * 40)
        device = management.intermediate_device()
        video = torch.zeros((1, 24, video_frames, height // 16, width // 16), device=device)
        audio = torch.zeros((1, 32, 2, audio_frames), device=device)
        return cls.outputs(
            LATENT={
                "samples": MultiStreamLatent(
                    (LatentStream("video", video), LatentStream("audio", audio))
                )
            }
        )
