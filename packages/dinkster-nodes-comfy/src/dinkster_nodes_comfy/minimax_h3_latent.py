"""Native MiniMax H3 joint latents and image-to-video conditioning."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    AUDIO_TYPE,
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_INT,
    CORE_STRING,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    LatentStream,
    MultiStreamLatent,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    initialize_inference_process,
    materialize_audio,
    materialize_image,
    materialize_latent,
)

from .conditioning import conditioning_set_values


def empty_av_latent(width: int, height: int, length: int) -> tuple[dict[str, object], int]:
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
    return {
        "samples": MultiStreamLatent((LatentStream("video", video), LatentStream("audio", audio)))
    }, frame_count


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
        latent, _ = empty_av_latent(width, height, length)
        return cls.outputs(LATENT=latent)


class MiniMaxH3ImageToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MiniMaxH3ImageToVideo",
            display_name="MiniMax H3 Image to Video",
            category="model/conditioning/minimax",
            inputs=(
                InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec(
                    "prompt",
                    TypeExpr.concrete(CORE_STRING),
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                *EmptyMiniMaxH3LatentAV.schema().inputs,
                InputSpec("first_frame", TypeExpr.concrete(IMAGE_TYPE), required=False),
                InputSpec("last_frame", TypeExpr.concrete(IMAGE_TYPE), required=False),
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
        vae: Any,
        prompt: str,
        width: int,
        height: int,
        length: int,
        first_frame: object = None,
        last_frame: object = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        utils = importlib.import_module("dinkster_inference.utils")
        latent, frame_count = empty_av_latent(width, height, length)
        images: list[Any] = []
        keyframes: list[dict[str, Any]] = []
        for image, index, crop in (
            (first_frame, 0, "disabled"),
            (last_frame, frame_count - 1, "center"),
        ):
            if image is not None:
                pixels = materialize_image(image)[:1, ..., :3].movedim(-1, 1)
                pixels = utils.common_upscale(pixels, width, height, "lanczos", crop).movedim(1, -1)
                images.append(pixels)
                keyframes.append({"resolved_frame_index": index, "image": pixels})
        tokens = clip.tokenize(prompt, images=images)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if keyframes:
            for keyframe in keyframes:
                keyframe["latent"] = vae.encode(keyframe.pop("image"))
            conditioning = conditioning_set_values(conditioning, {"minimax_keyframes": keyframes})
        return cls.outputs(positive=conditioning, LATENT=latent)


class MiniMaxH3AddGuide(Node):
    """Anchor image and/or audio guides at an arbitrary pixel frame of the target video."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MiniMaxH3AddGuide",
            display_name="Add Guide for MiniMax H3",
            category="model/conditioning/minimax",
            description=(
                "Anchor an image, a short clip, audio, or a clip with its soundtrack at any "
                "frame of a MiniMax H3 video. Chain several nodes to anchor several frames."
            ),
            inputs=(
                InputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec(
                    "vae",
                    TypeExpr.concrete(VAE_TYPE),
                    required=False,
                    doc="Video VAE, needed when an image is connected.",
                ),
                InputSpec(
                    "audio_vae",
                    TypeExpr.concrete(VAE_TYPE),
                    required=False,
                    doc="Audio VAE, needed when an audio is connected.",
                ),
                InputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),
                InputSpec(
                    "image",
                    TypeExpr.concrete(IMAGE_TYPE),
                    required=False,
                    doc=(
                        "Image or video frames to anchor. Multi-frame batches are anchored as a "
                        "clip and cropped down to the model's valid clip lengths: 5, 22, 39... "
                        "(17k + 5) frames. Batches shorter than 5 frames use only the first image."
                    ),
                ),
                InputSpec(
                    "audio",
                    TypeExpr.concrete(AUDIO_TYPE),
                    required=False,
                    doc=(
                        "Soundtrack to anchor starting at the same frame index, cropped to the "
                        "video's remaining duration."
                    ),
                ),
                InputSpec(
                    "frame_idx",
                    TypeExpr.concrete(CORE_INT),
                    default=0,
                    widget=NumberWidget(min=-9999, max=9999),
                    doc=(
                        "Frame index to anchor the image or the clip's first frame at. Negative "
                        "values are counted from the end of the video."
                    ),
                ),
            ),
            outputs=(OutputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Any,
        latent: object,
        frame_idx: int,
        vae: Any = None,
        audio_vae: Any = None,
        image: object = None,
        audio: object = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        samples = materialize_latent(latent)["samples"]
        if type(samples) is not MultiStreamLatent:
            raise ValueError("MiniMaxH3AddGuide expects a MiniMax H3 AV latent")
        streams = cast("MultiStreamLatent[Any]", samples)
        if set(streams.roles) != {"video", "audio"}:
            raise ValueError("MiniMaxH3AddGuide expects a MiniMax H3 AV latent")
        video = streams.by_role("video")
        audio_samples = streams.by_role("audio")
        if (
            video.ndim != 5
            or video.shape[1] != 24
            or audio_samples.ndim != 4
            or audio_samples.shape[1] != 32
            or audio_samples.shape[2] != 2
        ):
            raise ValueError("MiniMaxH3AddGuide expects a MiniMax H3 AV latent")
        if image is None and audio is None:
            raise ValueError("MiniMaxH3AddGuide needs an image or an audio to anchor")

        model = importlib.import_module("dinkster_inference.ldm.minimax.model")
        frame_count = sum(model.FRAME_PER_TOKEN[index % 5] for index in range(video.shape[2]))
        guide_frames = 1
        pixels = None
        if image is not None:
            if vae is None:
                raise ValueError("anchoring guide frames needs the vae input")
            pixels = materialize_image(image)
            guide_frames = pixels.shape[0]
            if guide_frames < 5:
                guide_frames = 1
            else:
                guide_frames -= (guide_frames - 5) % 17

        resolved_frame_index = frame_idx if frame_idx >= 0 else frame_count + frame_idx
        if resolved_frame_index < 0 or resolved_frame_index + guide_frames > frame_count:
            if guide_frames == 1:
                raise ValueError(
                    f"frame_idx {frame_idx} is outside the video's {frame_count} frames"
                )
            raise ValueError(
                f"a {guide_frames} frame guide clip at frame_idx {frame_idx} does not fit in "
                f"the video's {frame_count} frames"
            )

        keyframe: dict[str, Any] = {"resolved_frame_index": resolved_frame_index}
        if pixels is not None:
            utils = importlib.import_module("dinkster_inference.utils")
            frames = pixels[:guide_frames, ..., :3].movedim(-1, 1)
            frames = utils.common_upscale(
                frames, video.shape[4] * 16, video.shape[3] * 16, "lanczos", "center"
            ).movedim(1, -1)
            keyframe["latent"] = vae.encode(frames)

        if audio is not None:
            if audio_vae is None:
                raise ValueError("anchoring guide audio needs the audio_vae input")
            source = materialize_audio(audio)
            waveform = cast("Any", source["waveform"])
            sample_rate = source["sample_rate"]
            vae_rate = getattr(audio_vae, "audio_sample_rate", 32000)
            if sample_rate != vae_rate:
                waveform = importlib.import_module("dinkster_inference.audio").resample(
                    waveform, sample_rate, vae_rate
                )
            audio_latent = audio_vae.encode(waveform[:1].movedim(1, -1))
            max_audio_frames = math.floor(
                audio_samples.shape[-1] - model.FRAME_RESCALE * resolved_frame_index
            )
            if max_audio_frames < 1:
                raise ValueError(
                    f"frame_idx {frame_idx} is past the end of the video's audio track"
                )
            if audio_latent.shape[-1] > max_audio_frames:
                audio_latent = audio_latent[..., :max_audio_frames].clone()
            keyframe["audio_latent"] = audio_latent

        keyframes = list(positive[0][1].get("minimax_keyframes", []))
        keyframes.append(keyframe)
        return cls.outputs(
            positive=conditioning_set_values(positive, {"minimax_keyframes": keyframes})
        )
