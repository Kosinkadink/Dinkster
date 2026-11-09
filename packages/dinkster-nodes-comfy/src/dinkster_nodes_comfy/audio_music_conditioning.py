"""Native MiniMax Music3 and reference-timbre conditioning."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)


class MiniMaxMusic3TextEncode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MiniMaxMusic3TextEncode",
            display_name="MiniMax Music3 Text Encode",
            category="model/conditioning/minimax music",
            description=(
                "Uses a MiniMax Music3 CLIP model to generate the acoustic conditioning sequence."
            ),
            inputs=(
                InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec(
                    "caption",
                    TypeExpr.concrete(CORE_STRING),
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                InputSpec(
                    "lyrics",
                    TypeExpr.concrete(CORE_STRING),
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                InputSpec(
                    "seed",
                    TypeExpr.concrete(CORE_INT),
                    default=0,
                    widget=NumberWidget(
                        min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate="fixed"
                    ),
                ),
                InputSpec(
                    "max_duration",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=120.0,
                    widget=NumberWidget(min=0.04, max=360.0, step=0.04),
                    doc="Maximum duration in seconds; the model can end the song earlier.",
                ),
                InputSpec(
                    "cfg_scale",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.5,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1, round=0.01),
                    advanced=True,
                ),
                InputSpec(
                    "top_k",
                    TypeExpr.concrete(CORE_INT),
                    default=50,
                    widget=NumberWidget(min=1, max=16384),
                    advanced=True,
                ),
            ),
            outputs=(
                OutputSpec("CONDITIONING", _CONDITIONING),
                OutputSpec("seconds", TypeExpr.concrete(CORE_FLOAT), display_name="seconds"),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        clip: Any,
        caption: str,
        lyrics: str,
        seed: int,
        max_duration: float,
        cfg_scale: float,
        top_k: int,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        ar = importlib.import_module("dinkster_inference.ldm.minimax_music.ar")
        frames = min(ar.MAX_AUDIO_FRAMES, max(1, round(max_duration * ar.AUDIO_FRAMES_PER_SECOND)))
        tokens = clip.tokenize(
            caption,
            lyrics=lyrics,
            seed=seed,
            max_audio_frames=frames,
            cfg_scale=cfg_scale,
            top_k=top_k,
        )
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        for hidden, metadata in conditioning:
            metadata["conditioning_scale"] = torch.ones(
                (hidden.shape[0], 1, 1), device=hidden.device, dtype=hidden.dtype
            )
        return cls.outputs(
            CONDITIONING=conditioning,
            seconds=conditioning[0][0].shape[1] / ar.AUDIO_FRAMES_PER_SECOND,
        )


class ReferenceTimbreAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ReferenceTimbreAudio",
            display_name="Set Reference Audio",
            category="model/conditioning",
            description="This node sets the reference audio for ace step 1.5",
            inputs=(
                InputSpec("conditioning", _CONDITIONING),
                InputSpec("latent", TypeExpr.concrete(LATENT_TYPE), required=False),
            ),
            outputs=(OutputSpec("CONDITIONING", _CONDITIONING),),
        )

    @classmethod
    def execute(
        cls, *, conditioning: Sequence[Sequence[Any]], latent: object = None
    ) -> Mapping[str, object]:
        if latent is None:
            return cls.outputs(CONDITIONING=conditioning)
        initialize_inference_process()
        samples = materialize_latent(latent)["samples"]
        result: list[list[Any]] = []
        for hidden, metadata in conditioning:
            references = metadata.get("reference_audio_timbre_latents")
            values = [samples] if references is None else references + [samples]
            result.append([hidden, {**metadata, "reference_audio_timbre_latents": values}])
        return cls.outputs(CONDITIONING=result)


AUDIO_MUSIC_CONDITIONING_NODES = (MiniMaxMusic3TextEncode, ReferenceTimbreAudio)
