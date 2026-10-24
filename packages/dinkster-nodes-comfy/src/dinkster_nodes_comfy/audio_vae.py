"""Native ComfyUI audio VAE nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    AUDIO_TYPE,
    CORE_INT,
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_audio,
    materialize_latent,
)

_AUDIO = TypeExpr.concrete(AUDIO_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)


def _decode_audio(
    vae: Any, samples: object, tile: int | None = None, overlap: int | None = None
) -> dict[str, object]:
    samples = cast("Mapping[str, Any]", materialize_latent(samples))
    latent = samples["samples"]
    if latent.is_nested:
        latent = latent.unbind()[-1]

    if tile is None:
        audio = vae.decode(latent).movedim(-1, 1)
    else:
        audio = vae.decode_tiled(latent, tile_x=tile, tile_y=tile, overlap=overlap).movedim(-1, 1)

    std = audio.std(dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio /= std
    sample_rate = samples.get(
        "sample_rate",
        getattr(vae, "audio_sample_rate_output", getattr(vae, "audio_sample_rate", 44100)),
    )
    return {"waveform": audio, "sample_rate": sample_rate}


class VAEEncodeAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="VAEEncodeAudio",
            display_name="VAE Encode Audio",
            category="model/latent",
            search_terms=("audio to latent",),
            inputs=(InputSpec("audio", _AUDIO), InputSpec("vae", _VAE)),
            outputs=(OutputSpec("LATENT", _LATENT),),
        )

    @classmethod
    def execute(cls, *, audio: Any, vae: Any) -> Mapping[str, object]:
        if audio is None:
            raise ValueError(
                "VAEEncodeAudio: input audio is None (source video may have no audio track)."
            )
        source = cast("Mapping[str, Any]", materialize_audio(audio))
        sample_rate = source["sample_rate"]
        vae_sample_rate = getattr(vae, "audio_sample_rate", 44100)
        waveform = source["waveform"]
        if vae_sample_rate != sample_rate:
            inference_audio: Any = importlib.import_module("dinkster_inference.audio")
            waveform = inference_audio.resample(waveform, sample_rate, vae_sample_rate)
        return cls.outputs(LATENT={"samples": vae.encode(waveform.movedim(1, -1))})


class VAEDecodeAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="VAEDecodeAudio",
            display_name="VAE Decode Audio",
            category="model/latent",
            search_terms=("latent to audio",),
            inputs=(InputSpec("samples", _LATENT), InputSpec("vae", _VAE)),
            outputs=(OutputSpec("AUDIO", _AUDIO),),
        )

    @classmethod
    def execute(cls, *, samples: object, vae: Any) -> Mapping[str, object]:
        return cls.outputs(AUDIO=_decode_audio(vae, samples))


class VAEDecodeAudioTiled(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="VAEDecodeAudioTiled",
            display_name="VAE Decode Audio (Tiled)",
            category="model/latent",
            search_terms=("latent to audio",),
            inputs=(
                InputSpec("samples", _LATENT),
                InputSpec("vae", _VAE),
                InputSpec(
                    "tile_size", _INT, default=512, widget=NumberWidget(min=32, max=8192, step=8)
                ),
                InputSpec(
                    "overlap", _INT, default=64, widget=NumberWidget(min=0, max=1024, step=8)
                ),
            ),
            outputs=(OutputSpec("AUDIO", _AUDIO),),
        )

    @classmethod
    def execute(
        cls, *, samples: object, vae: Any, tile_size: int = 512, overlap: int = 64
    ) -> Mapping[str, object]:
        return cls.outputs(AUDIO=_decode_audio(vae, samples, tile_size, overlap))


AUDIO_VAE_NODES = (VAEEncodeAudio, VAEDecodeAudio, VAEDecodeAudioTiled)
