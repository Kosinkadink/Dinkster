"""Native LTXV audio VAE nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    AUDIO_TYPE,
    CORE_FLOAT,
    CORE_INT,
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
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


class LTXVEmptyLatentAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LTXVEmptyLatentAudio",
            display_name="Empty LTXV Latent Audio",
            category="model/latent/ltxv",
            inputs=(
                InputSpec(
                    "frames_number",
                    TypeExpr.concrete(CORE_INT),
                    default=97,
                    widget=NumberWidget(min=1, max=1000, step=1, display="number"),
                    doc="Number of frames.",
                ),
                InputSpec(
                    "frame_rate",
                    TypeExpr.union(CORE_FLOAT, CORE_INT),
                    default=25.0,
                    widget=NumberWidget(min=1.0, max=1000.0, step=0.01, display="number"),
                    doc="Number of frames per second.",
                ),
                InputSpec(
                    "batch_size",
                    TypeExpr.concrete(CORE_INT),
                    default=1,
                    widget=NumberWidget(min=1, max=4096, display="number"),
                    doc="The number of latent audio samples in the batch.",
                ),
                InputSpec(
                    "audio_vae",
                    _VAE,
                    display_name="Audio VAE",
                    doc="The Audio VAE model to get configuration from.",
                ),
            ),
            outputs=(OutputSpec("LATENT", _LATENT, display_name="Latent"),),
        )

    @classmethod
    def execute(
        cls, *, frames_number: int, frame_rate: float, batch_size: int, audio_vae: Any
    ) -> Mapping[str, object]:
        assert audio_vae is not None, "Audio VAE model is required"
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        first_stage = audio_vae.first_stage_model
        latent = torch.zeros(
            (
                batch_size,
                audio_vae.latent_channels,
                first_stage.num_of_latents_from_frames(frames_number, frame_rate),
                first_stage.latent_frequency_bins,
            ),
            device=management.intermediate_device(),
        )
        return cls.outputs(LATENT={"samples": latent, "type": "audio"})


LTX_AUDIO_NODES = (LTXVAudioVAEEncode, LTXVAudioVAEDecode, LTXVEmptyLatentAudio)
