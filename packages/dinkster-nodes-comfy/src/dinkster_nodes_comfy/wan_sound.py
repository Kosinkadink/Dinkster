"""Native Wan sound and HuMo audio conditioning."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    AUDIO_ENCODER_OUTPUT_TYPE,
    CONDITIONING_TYPE,
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
    materialize_latent,
)

from .conditioning import conditioning_set_values

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_AUDIO = TypeExpr.concrete(AUDIO_ENCODER_OUTPUT_TYPE)
_OUTPUTS = (
    OutputSpec("positive", _CONDITIONING),
    OutputSpec("negative", _CONDITIONING),
    OutputSpec("latent", _LATENT),
)


def _dimensions(length: int) -> tuple[InputSpec, ...]:
    return (
        InputSpec("width", _INT, default=832, widget=NumberWidget(min=16, max=16384, step=16)),
        InputSpec("height", _INT, default=480, widget=NumberWidget(min=16, max=16384, step=16)),
        _length(length),
        InputSpec("batch_size", _INT, default=1, widget=NumberWidget(min=1, max=4096)),
    )


def _length(default: int) -> InputSpec:
    return InputSpec("length", _INT, default=default, widget=NumberWidget(min=1, max=16384, step=4))


def _runtime() -> tuple[Any, Any, Any]:
    initialize_inference_process()
    return (
        importlib.import_module("torch"),
        importlib.import_module("dinkster_inference.model_management"),
        importlib.import_module("dinkster_inference.utils"),
    )


def _interpolate(features: Any, input_fps: int, output_fps: int, torch: Any) -> Any:
    features = features.transpose(1, 2)
    output_len = int(features.shape[2] / float(input_fps) * output_fps)
    return torch.nn.functional.interpolate(
        features, size=output_len, align_corners=True, mode="linear"
    ).transpose(1, 2)


def _audio_bucket(features: Any, batch_frames: int, torch: Any) -> Any:
    np = importlib.import_module("numpy")
    layers, frames, dimensions = features.shape
    repeats = int(frames / (batch_frames * (30 / 16))) + 1
    count = repeats * batch_frames
    padded = math.ceil(repeats * batch_frames / 16 * 30)
    times = np.linspace(0, count / 16, count, endpoint=False)
    indices = np.clip(np.round(times * 30).astype(int), 0, padded - 1)
    buckets: list[Any] = []
    for index in indices:
        if index < frames:
            value = features[:, [index]].flatten(start_dim=-2, end_dim=-1)
            if layers == 1:
                value = features[0][[index]].flatten()
        else:
            shape = [layers, dimensions] if layers > 1 else [dimensions]
            value = torch.zeros(shape, device=features.device)
        buckets.append(value)
    return torch.cat([value.unsqueeze(0) for value in buckets], dim=0)


def _append_reference(conditioning: Sequence[Sequence[Any]], latent: Any) -> list[list[Any]]:
    return [
        [
            text,
            {**metadata, "reference_latents": (metadata.get("reference_latents") or []) + [latent]},
        ]
        for text, metadata in conditioning
    ]


def _sound_conditioning(
    positive: Sequence[Sequence[Any]],
    negative: Sequence[Sequence[Any]],
    vae: Any,
    width: int,
    height: int,
    length: int,
    batch_size: int,
    *,
    frame_offset: int = 0,
    ref_image: object = None,
    audio_encoder_output: object = None,
    control_video: object = None,
    ref_motion: object = None,
    ref_motion_latent: Any = None,
) -> Mapping[str, object]:
    torch, management, utils = _runtime()
    latent_t = (length - 1) // 4 + 1
    if audio_encoder_output is not None:
        audio = materialize_latent(audio_encoder_output)
        features = _interpolate(torch.cat(audio["encoded_audio_all_layers"]), 50, 30, torch)
        bucket = _audio_bucket(features, latent_t * 4, torch).unsqueeze(0)
        if bucket.ndim == 3:
            bucket = bucket.permute(0, 2, 1)
        elif bucket.ndim == 4:
            bucket = bucket.permute(0, 2, 3, 1)
        bucket = bucket[:, :, :, frame_offset : frame_offset + latent_t * 4]
        if bucket.shape[3] > 0:
            positive = conditioning_set_values(positive, {"audio_embed": bucket})
            negative = conditioning_set_values(negative, {"audio_embed": bucket * 0.0})

    if ref_image is not None:
        image = materialize_image(ref_image)
        image = utils.common_upscale(
            image[:1].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        reference = vae.encode(image[:, :, :, :3])
        positive = _append_reference(positive, reference)
        negative = _append_reference(negative, reference)

    if ref_motion is not None:
        motion = materialize_image(ref_motion)[-73:]
        motion = utils.common_upscale(
            motion.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        if motion.shape[0] < 73:
            padded = torch.ones([73, height, width, 3]) * 0.5
            padded[-motion.shape[0] :] = motion
            motion = padded
        ref_motion_latent = vae.encode(motion[:, :, :, :3])
    if ref_motion_latent is not None:
        motion_latent = ref_motion_latent[:, :, -19:]
        positive = conditioning_set_values(positive, {"reference_motion": motion_latent})
        negative = conditioning_set_values(negative, {"reference_motion": motion_latent})

    latent = torch.zeros(
        [batch_size, 16, latent_t, height // 8, width // 8],
        device=management.intermediate_device(),
    )
    formats = importlib.import_module("dinkster_inference.latent_formats")
    control = formats.Wan21().process_out(torch.zeros_like(latent))
    if control_video is not None:
        video = materialize_image(control_video)
        video = utils.common_upscale(
            video[:length].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        encoded = vae.encode(video[:, :, :, :3])
        control[:, :, : encoded.shape[2]] = encoded
    return {
        "positive": conditioning_set_values(positive, {"control_video": control}),
        "negative": conditioning_set_values(negative, {"control_video": control}),
        "latent": {"samples": latent},
    }


class WanSoundImageToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanSoundImageToVideo",
            category="model/conditioning/wan/sound",
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                *_dimensions(77),
                InputSpec("audio_encoder_output", _AUDIO, required=False),
                InputSpec("ref_image", _IMAGE, required=False),
                InputSpec("control_video", _IMAGE, required=False),
                InputSpec("ref_motion", _IMAGE, required=False),
            ),
            outputs=_OUTPUTS,
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        vae: Any,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        audio_encoder_output: object = None,
        ref_image: object = None,
        control_video: object = None,
        ref_motion: object = None,
    ) -> Mapping[str, object]:
        return _sound_conditioning(
            positive,
            negative,
            vae,
            width,
            height,
            length,
            batch_size,
            audio_encoder_output=audio_encoder_output,
            ref_image=ref_image,
            control_video=control_video,
            ref_motion=ref_motion,
        )


class WanSoundImageToVideoExtend(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanSoundImageToVideoExtend",
            category="model/conditioning/wan/sound",
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                _length(77),
                InputSpec("video_latent", _LATENT),
                InputSpec("audio_encoder_output", _AUDIO, required=False),
                InputSpec("ref_image", _IMAGE, required=False),
                InputSpec("control_video", _IMAGE, required=False),
            ),
            outputs=_OUTPUTS,
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        vae: Any,
        length: int,
        video_latent: object,
        audio_encoder_output: object = None,
        ref_image: object = None,
        control_video: object = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        samples = materialize_latent(video_latent)["samples"]
        return _sound_conditioning(
            positive,
            negative,
            vae,
            samples.shape[-1] * 8,
            samples.shape[-2] * 8,
            length,
            samples.shape[0],
            frame_offset=samples.shape[-3] * 4,
            ref_image=ref_image,
            audio_encoder_output=audio_encoder_output,
            control_video=control_video,
            ref_motion_latent=samples,
        )


def _humo_window(audio: Any, length: int, torch: Any) -> Any:
    zero = torch.zeros(audio.shape[1:], dtype=audio.dtype, device=audio.device)
    prefix = torch.zeros((3, *audio.shape[1:]), dtype=audio.dtype, device=audio.device)
    windows: list[Any] = []
    for index in range(1 + (length - 1) // 4):
        start, end = (-2, 3) if index == 0 else (1 + 4 * (index - 1) - 2, 1 + 4 * index + 2)
        window = torch.stack(
            [audio[frame] if 0 <= frame < audio.shape[0] else zero for frame in range(start, end)]
        )
        windows.append(torch.cat((prefix, window)) if index == 0 else window)
    return torch.stack(windows)


class WanHuMoImageToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanHuMoImageToVideo",
            category="model/conditioning/wan/humo",
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                *_dimensions(97),
                InputSpec("audio_encoder_output", _AUDIO, required=False),
                InputSpec("ref_image", _IMAGE, required=False),
            ),
            outputs=_OUTPUTS,
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        vae: Any,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        audio_encoder_output: object = None,
        ref_image: object = None,
    ) -> Mapping[str, object]:
        torch, management, utils = _runtime()
        latent_t = (length - 1) // 4 + 1
        latent = torch.zeros(
            [batch_size, 16, latent_t, height // 8, width // 8],
            device=management.intermediate_device(),
        )
        if ref_image is not None:
            image = materialize_image(ref_image)
            image = utils.common_upscale(
                image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            reference = vae.encode(image[:, :, :, :3])
            reference_negative = torch.zeros_like(reference)
        else:
            reference = torch.zeros(
                [batch_size, 16, 1, height // 8, width // 8],
                device=management.intermediate_device(),
            )
            reference_negative = reference
        positive = _append_reference(positive, reference)
        negative = _append_reference(negative, reference_negative)
        if audio_encoder_output is not None:
            audio = materialize_latent(audio_encoder_output)
            layers = torch.stack(audio["encoded_audio_all_layers"], dim=2)
            layers = layers[:, : (audio["audio_samples"] // 640) * 2]
            features = [
                _interpolate(layers[:, :, start : start + 8].mean(dim=2), 50, 25, torch)
                for start in (0, 8, 16, 24)
            ]
            features.append(_interpolate(layers[:, :, 32], 50, 25, torch))
            embedding = _humo_window(torch.stack(features, dim=2)[0], length, torch).unsqueeze(0)
            embedding_negative = torch.zeros_like(embedding)
        else:
            embedding = torch.zeros(
                [batch_size, latent_t + 1, 8, 5, 1280],
                device=management.intermediate_device(),
            )
            embedding_negative = embedding
        return cls.outputs(
            positive=conditioning_set_values(positive, {"audio_embed": embedding}),
            negative=conditioning_set_values(negative, {"audio_embed": embedding_negative}),
            latent={"samples": latent},
        )
