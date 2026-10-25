"""Native MiniMax H3 joint latents and image-to-video conditioning."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
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
    materialize_image,
)

from .conditioning import conditioning_set_values


def _empty_av_latent(width: int, height: int, length: int) -> tuple[dict[str, object], int]:
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
        latent, _ = _empty_av_latent(width, height, length)
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
        latent, frame_count = _empty_av_latent(width, height, length)
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
