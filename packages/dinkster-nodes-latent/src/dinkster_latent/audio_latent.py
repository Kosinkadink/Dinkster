"""Native audio latent construction and dimension inspection."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_INT,
    LATENT_TYPE,
    InputSpec,
    MultiStreamLatent,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
    report_progress,
    with_identity_replacement,
)


def _duration_inputs(
    seconds: float, step: float, *, minimum: float = 1.0, batch_doc: str = ""
) -> tuple[InputSpec, ...]:
    return (
        InputSpec(
            "seconds",
            TypeExpr.concrete(CORE_FLOAT),
            default=seconds,
            widget=NumberWidget(min=minimum, max=1000.0, step=step),
        ),
        InputSpec(
            "batch_size",
            TypeExpr.concrete(CORE_INT),
            default=1,
            widget=NumberWidget(min=1, max=4096),
            doc=batch_doc,
        ),
    )


class EmptyLatentAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            NodeSchema(
                node_type="dinkster.latent.empty_latent_audio",
                display_name="Empty Latent Audio",
                category="model/latent",
                inputs=_duration_inputs(
                    47.6, 0.1, batch_doc="The number of latent images in the batch."
                ),
                outputs=(OutputSpec("LATENT", TypeExpr.concrete(LATENT_TYPE)),),
            ),
            "EmptyLatentAudio",
        )

    @classmethod
    def execute(cls, *, seconds: float, batch_size: int) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        length = round((seconds * 44100 / 2048) / 2) * 2
        latent = torch.zeros([batch_size, 64, length], device=management.intermediate_device())
        return cls.outputs(
            LATENT={"samples": latent, "type": "audio", "downscale_ratio_temporal": 2048}
        )


class GetAudioLatentSize(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            NodeSchema(
                node_type="dinkster.latent.get_audio_latent_size",
                display_name="Get Audio Latent Size",
                search_terms=("dimensions", "audio latent info", "audio latent length"),
                description=(
                    "Returns the length, audio channel count, and batch size of an audio latent."
                ),
                category="model/latent",
                inputs=(
                    InputSpec(
                        "latent",
                        TypeExpr.concrete(LATENT_TYPE),
                        doc="The audio or audio/video latent to inspect.",
                    ),
                ),
                outputs=(
                    OutputSpec(
                        "length", TypeExpr.concrete(CORE_INT), doc="The audio latent length."
                    ),
                    OutputSpec(
                        "audio_channels",
                        TypeExpr.concrete(CORE_INT),
                        doc="The number of audio channels. "
                        "Latents without an audio channel dimension "
                        "have one channel.",
                    ),
                    OutputSpec(
                        "batch_size",
                        TypeExpr.concrete(CORE_INT),
                        doc="The number of latents in the batch.",
                    ),
                ),
            ),
            "GetAudioLatentSize",
        )

    @classmethod
    def execute(cls, *, latent: object) -> Mapping[str, object]:
        initialize_inference_process()
        samples = cast("Mapping[str, Any]", materialize_latent(latent))["samples"]
        if isinstance(samples, MultiStreamLatent):
            samples = cast("MultiStreamLatent[Any]", samples).streams[-1].payload
        elif samples.is_nested:
            samples = samples.unbind()[-1]
        if samples.ndim == 4:
            batch_size, _, audio_channels, length = samples.shape
        else:
            batch_size, _, length = samples.shape
            audio_channels = 1
        report_progress(
            1,
            1,
            text=f"length: {length}, audio channels: {audio_channels}\n batch size: {batch_size}",
        )
        return cls.outputs(length=length, audio_channels=audio_channels, batch_size=batch_size)
