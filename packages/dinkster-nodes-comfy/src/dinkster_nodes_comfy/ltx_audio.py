"""Native LTXV audio VAE nodes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    AUDIO_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)

from .audio_vae import VAEEncodeAudio

_AUDIO = TypeExpr.concrete(AUDIO_TYPE)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)


class LTXVAudioVAEEncode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LTXVAudioVAEEncode",
            display_name="LTXV Audio VAE Encode",
            category="model/latent/ltxv",
            inputs=(
                InputSpec("audio", _AUDIO, doc="The audio to be encoded."),
                InputSpec(
                    "audio_vae",
                    _VAE,
                    display_name="Audio VAE",
                    doc="The Audio VAE model to use for encoding.",
                ),
            ),
            outputs=(OutputSpec("LATENT", _LATENT, display_name="Audio Latent"),),
        )

    @classmethod
    def execute(cls, *, audio: Any, audio_vae: Any) -> Mapping[str, object]:
        return VAEEncodeAudio.execute(audio=audio, vae=audio_vae)


class LTXVAudioVAEDecode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LTXVAudioVAEDecode",
            display_name="LTXV Audio VAE Decode",
            category="model/latent/ltxv",
            inputs=(
                InputSpec("samples", _LATENT, doc="The latent to be decoded."),
                InputSpec(
                    "audio_vae",
                    _VAE,
                    display_name="Audio VAE",
                    doc="The Audio VAE model used for decoding the latent.",
                ),
            ),
            outputs=(OutputSpec("AUDIO", _AUDIO, display_name="Audio"),),
        )

    @classmethod
    def execute(cls, *, samples: object, audio_vae: Any) -> Mapping[str, object]:
        initialize_inference_process()
        latent = cast("Mapping[str, Any]", materialize_latent(samples))["samples"]
        if latent.is_nested:
            latent = latent.unbind()[-1]
        audio = audio_vae.decode(latent).movedim(-1, 1).to(latent.device)
        return cls.outputs(
            AUDIO={
                "waveform": audio,
                "sample_rate": int(audio_vae.first_stage_model.output_sample_rate),
            }
        )


LTX_AUDIO_NODES = (LTXVAudioVAEEncode, LTXVAudioVAEDecode)
