"""Native LTXV audio and video latent composition."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    LATENT_TYPE,
    InputSpec,
    LatentStream,
    MultiStreamLatent,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)

_LATENT = TypeExpr.concrete(LATENT_TYPE)


def _streams(value: object) -> tuple[object, object]:
    if type(value) is MultiStreamLatent:
        streams = cast("MultiStreamLatent[object]", value)
        return streams.by_role("video"), streams.by_role("audio")
    if getattr(value, "is_nested", False):
        streams = value.unbind()
        return streams[0], streams[1]
    raise ValueError("AV latent samples must contain video and audio streams")


def _mask_streams(value: object) -> tuple[object, object]:
    return _streams(value)


class LTXVConcatAVLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LTXVConcatAVLatent",
            display_name="Concat AV Latent",
            description=(
                "Merge a video latent and an audio latent into a joint AV latent "
                "(any AV model, e.g. LTXV or MiniMax H3)."
            ),
            category="model/latent/ltxv",
            inputs=(
                InputSpec("video_latent", _LATENT),
                InputSpec("audio_latent", _LATENT),
            ),
            outputs=(OutputSpec("LATENT", _LATENT, display_name="latent"),),
        )

    @staticmethod
    def fit_audio(
        reference: object, audio: object, noise_mask: object | None
    ) -> tuple[object, object | None]:
        dims = [
            index for index in range(reference.ndim) if reference.shape[index] != audio.shape[index]
        ]
        if not dims:
            return audio, noise_mask
        if len(dims) > 1 or dims[0] < 2:
            raise ValueError(
                f"audio latent {tuple(audio.shape)} cannot be fitted to {tuple(reference.shape)}"
            )
        dimension = dims[0]
        length = reference.shape[dimension]
        if noise_mask is not None:
            reshape_mask = importlib.import_module("dinkster_inference.utils").reshape_mask
            noise_mask = reshape_mask(noise_mask, audio.shape)
        if audio.shape[dimension] > length:
            audio = audio.narrow(dimension, 0, length)
            if noise_mask is not None:
                noise_mask = noise_mask.narrow(dimension, 0, length)
        else:
            torch = importlib.import_module("torch")
            pad = torch.zeros_like(audio.narrow(dimension, 0, 1)).repeat(
                [
                    length - audio.shape[dimension] if index == dimension else 1
                    for index in range(audio.ndim)
                ]
            )
            audio = torch.cat([audio, pad], dim=dimension)
            if noise_mask is not None:
                noise_mask = torch.cat([noise_mask, torch.ones_like(pad)], dim=dimension)
        return audio, noise_mask

    @classmethod
    def execute(cls, *, video_latent: object, audio_latent: object) -> Mapping[str, object]:
        initialize_inference_process()
        video = cast("Mapping[str, Any]", materialize_latent(video_latent))
        audio = cast("Mapping[str, Any]", materialize_latent(audio_latent))
        output = dict(video)
        output.update(audio)
        video_samples = video["samples"]
        audio_samples = audio["samples"]
        video_mask = video.get("noise_mask")
        audio_mask = audio.get("noise_mask")
        if type(video_samples) is MultiStreamLatent or getattr(video_samples, "is_nested", False):
            video_samples, prior_audio = _streams(video_samples)
            if video_mask is not None:
                video_mask, _ = _mask_streams(video_mask)
            audio_samples, audio_mask = cls.fit_audio(prior_audio, audio_samples, audio_mask)
        if video_mask is not None or audio_mask is not None:
            torch = importlib.import_module("torch")
            if video_mask is None:
                video_mask = torch.ones_like(video_samples)
            if audio_mask is None:
                audio_mask = torch.ones_like(audio_samples)
            output["noise_mask"] = MultiStreamLatent(
                (LatentStream("video", video_mask), LatentStream("audio", audio_mask))
            )
        output["samples"] = MultiStreamLatent(
            (LatentStream("video", video_samples), LatentStream("audio", audio_samples))
        )
        return cls.outputs(LATENT=output)


class LTXVSeparateAVLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LTXVSeparateAVLatent",
            display_name="Separate AV Latent",
            description=(
                "Split a joint AV latent into its video and audio latents "
                "(any AV model, e.g. LTXV or MiniMax H3)."
            ),
            category="model/latent/ltxv",
            inputs=(InputSpec("av_latent", _LATENT),),
            outputs=(
                OutputSpec("VIDEO_LATENT", _LATENT, display_name="video_latent"),
                OutputSpec("AUDIO_LATENT", _LATENT, display_name="audio_latent"),
            ),
        )

    @classmethod
    def execute(cls, *, av_latent: object) -> Mapping[str, object]:
        initialize_inference_process()
        latent = cast("Mapping[str, Any]", materialize_latent(av_latent))
        video_samples, audio_samples = _streams(latent["samples"])
        video = dict(latent)
        audio = dict(latent)
        video["samples"] = video_samples
        audio["samples"] = audio_samples
        if latent.get("noise_mask") is not None:
            video_mask, audio_mask = _mask_streams(latent["noise_mask"])
            video["noise_mask"] = video_mask
            audio["noise_mask"] = audio_mask
        return cls.outputs(VIDEO_LATENT=video, AUDIO_LATENT=audio)


LTX_AV_LATENT_NODES = (LTXVConcatAVLatent, LTXVSeparateAVLatent)
