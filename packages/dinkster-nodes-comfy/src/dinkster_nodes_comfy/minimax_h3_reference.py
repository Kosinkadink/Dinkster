"""Native MiniMax H3 reference image, video, and audio conditioning."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    AUDIO_TYPE,
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_COMBO,
    CORE_INT,
    CORE_STRING,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    ComboWidget,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    initialize_inference_process,
    materialize_audio,
    materialize_image,
)

from .conditioning import conditioning_set_values
from .minimax_h3_latent import empty_av_latent


def _resize(image: Any, width: int, height: int) -> Any:
    utils = importlib.import_module("dinkster_inference.utils")
    return utils.common_upscale(
        image[..., :3].movedim(-1, 1), width, height, "lanczos", "disabled"
    ).movedim(1, -1)


def _encode_audio(vae: Any, audio: object) -> tuple[Any, int]:
    source = materialize_audio(audio)
    waveform = cast("Any", source["waveform"])
    sample_rate = source["sample_rate"]
    vae_rate = getattr(vae, "audio_sample_rate", 32000)
    if sample_rate != vae_rate:
        waveform = importlib.import_module("dinkster_inference.audio").resample(
            waveform, sample_rate, vae_rate
        )
    latent = vae.encode(waveform[:1].movedim(1, -1))
    return latent, latent.shape[-1]


class MiniMaxH3ReferenceToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MiniMaxH3ReferenceToVideo",
            display_name="MiniMax H3 Reference to Video",
            description=(
                "<Picture i> / <Video k> / <Audio j> reference conditioning for MiniMax H3. "
                "Use the same tags when prompting."
            ),
            category="model/conditioning/minimax",
            inputs=(
                InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE), required=False),
                InputSpec("audio_vae", TypeExpr.concrete(VAE_TYPE), required=False),
                InputSpec(
                    "prompt",
                    TypeExpr.concrete(CORE_STRING),
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                InputSpec(
                    "width",
                    TypeExpr.concrete(CORE_INT),
                    default=1344,
                    widget=NumberWidget(min=32, max=16384, step=32),
                ),
                InputSpec(
                    "height",
                    TypeExpr.concrete(CORE_INT),
                    default=768,
                    widget=NumberWidget(min=32, max=16384, step=32),
                ),
                InputSpec(
                    "length",
                    TypeExpr.concrete(CORE_INT),
                    default=124,
                    widget=NumberWidget(min=5, max=3600, step=17),
                ),
                InputSpec(
                    "ref_image_size",
                    TypeExpr.concrete(CORE_COMBO),
                    default="match",
                    widget=ComboWidget(("match", "max")),
                ),
            ),
            input_families=tuple(
                InputFamilySpec(
                    family,
                    (InputSpec(member, TypeExpr.concrete(kind)),),
                    min_members=0,
                    max_members=limit,
                    member_prefix=member + "_",
                    required=False,
                )
                for family, member, kind, limit in (
                    ("ref_images", "ref_image", IMAGE_TYPE, 9),
                    ("ref_videos", "ref_video", IMAGE_TYPE, 3),
                    ("ref_video_audios", "ref_video_audio", AUDIO_TYPE, 3),
                    ("ref_audios", "ref_audio", AUDIO_TYPE, 3),
                )
            ),
            input_order=(
                "clip",
                "vae",
                "audio_vae",
                "prompt",
                "width",
                "height",
                "length",
                "ref_image_size",
                "ref_images",
                "ref_videos",
                "ref_video_audios",
                "ref_audios",
            ),
            outputs=(
                OutputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
                OutputSpec("LATENT", TypeExpr.concrete(LATENT_TYPE)),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        clip: Any,
        prompt: str,
        width: int,
        height: int,
        length: int,
        ref_image_size: str = "match",
        vae: Any = None,
        audio_vae: Any = None,
        ref_images: Mapping[str, object] | None = None,
        ref_videos: Mapping[str, object] | None = None,
        ref_video_audios: Mapping[str, object] | None = None,
        ref_audios: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        latent, frame_count = empty_av_latent(width, height, length)
        items: list[dict[str, Any]] = []
        blocks: list[dict[str, Any]] = []
        for image in (ref_images or {}).values():
            if image is None:
                continue
            pixels = materialize_image(image)
            h, w = pixels.shape[1:3]
            scale = (
                min(1.0, math.sqrt(width * height / (w * h)))
                if ref_image_size == "match"
                else min(1.0, 2048 / min(w, h))
            )
            tw, th = max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32)
            resized = _resize(pixels[:1], tw, th)
            items.append({"type": "image", "data": resized})
            if vae is not None:
                blocks.append(
                    {
                        "kind": "image",
                        "latent_h": th // 16,
                        "latent_w": tw // 16,
                        "latent": vae.encode(resized),
                    }
                )
        soundtracks = ref_video_audios or {}
        for name, frames in (ref_videos or {}).items():
            if frames is None:
                continue
            soundtrack = soundtracks.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
            pixels = materialize_image(frames)
            vh, vw = pixels.shape[1:3]
            ratio = vw / vh
            cw, ch = (768 * ratio, 768) if ratio >= 1 else (768, 768 / ratio)
            if cw * ch > 768 * 1344:
                scale = math.sqrt(768 * 1344 / (cw * ch))
                cw, ch = cw * scale, ch * scale
            cw, ch = max(32, round(cw / 32) * 32), max(32, round(ch / 32) * 32)
            if vw * vh < cw * ch:
                cw, ch = max(32, round(vw / 32) * 32), max(32, round(vh / 32) * 32)
            pixels = _resize(pixels, cw, ch)[:frame_count]
            count = pixels.shape[0]
            if count < 5:
                raise ValueError(
                    "MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)"
                )
            count -= (count - 5) % 17
            pixels = pixels[:count]
            if soundtrack is not None:
                items.append({"type": "audio"})
            indices = list(range(0, count, 12))
            items.append(
                {
                    "type": "video",
                    "data": pixels[indices],
                    "timestamps": [i / 2.0 for i in range(len(indices))],
                }
            )
            if vae is None:
                continue
            video_latent = vae.encode(pixels)
            audio_latent, audio_count = (
                _encode_audio(audio_vae, soundtrack)
                if soundtrack is not None and audio_vae is not None
                else (None, 0)
            )
            blocks.append(
                {
                    "kind": "video_audio" if audio_count else "video",
                    "latent_t": video_latent.shape[2],
                    "latent_h": ch // 16,
                    "latent_w": cw // 16,
                    "ref_audio_t": audio_count,
                    "latent": video_latent,
                    "audio_latent": audio_latent,
                }
            )
        for audio in (ref_audios or {}).values():
            if audio is None:
                continue
            items.append({"type": "audio"})
            if audio_vae is not None:
                audio_latent, audio_count = _encode_audio(audio_vae, audio)
                blocks.append(
                    {"kind": "audio", "ref_audio_t": audio_count, "audio_latent": audio_latent}
                )
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, minimax_ref_items=items))
        if blocks:
            positive = conditioning_set_values(positive, {"minimax_refs": blocks})
        return cls.outputs(positive=positive, LATENT=latent)
