"""Wan SCAIL video conditioning."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CLIP_VISION_OUTPUT_TYPE,
    CONDITIONING_TYPE,
    CORE_BOOLEAN,
    CORE_FLOAT,
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
    materialize_clip_vision_output,
    materialize_image,
)

from .conditioning import conditioning_set_values

_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
_CLIP_VISION_OUTPUT = TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE)
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)
_CATEGORY = "model/conditioning/wan/scail"
_EPS = 1e-5


def _number(
    name: str,
    type_: TypeExpr,
    default: float | int,
    minimum: float | int,
    maximum: float | int,
    step: float | int | None = None,
) -> InputSpec:
    return InputSpec(
        name, type_, default=default, widget=NumberWidget(min=minimum, max=maximum, step=step)
    )


def _upscale(image: Any, width: int, height: int, mode: str) -> Any:
    utils = importlib.import_module("dinkster_inference.utils")
    return utils.common_upscale(image.movedim(-1, 1), width, height, mode, "center").movedim(1, -1)


def _append_values(
    conditioning: Sequence[Sequence[Any]], values: Mapping[str, Sequence[object]]
) -> list[list[Any]]:
    return [
        [
            text,
            {
                **metadata,
                **{key: [*(metadata.get(key) or ()), *value] for key, value in values.items()},
            },
        ]
        for text, metadata in conditioning
    ]


def _set_values_with_timestep_range(
    conditioning: Sequence[Sequence[Any]], values: Mapping[str, object], start: float, end: float
) -> list[list[Any]]:
    if start > end:
        return [[text, metadata] for text, metadata in conditioning]
    result: list[list[Any]] = []
    for text, metadata in conditioning:
        condition_start = metadata.get("start_percent", 0.0)
        condition_end = metadata.get("end_percent", 1.0)
        overlap_start = max(start, condition_start)
        overlap_end = min(end, condition_end)
        if overlap_start >= overlap_end:
            result.append([text, metadata])
            continue
        if overlap_start > condition_start:
            result.extend(
                conditioning_set_values(
                    [[text, metadata]],
                    {"start_percent": condition_start, "end_percent": overlap_start - _EPS},
                )
            )
        result.extend(
            conditioning_set_values(
                [[text, metadata]],
                {**values, "start_percent": overlap_start, "end_percent": overlap_end},
            )
        )
        if overlap_end < condition_end:
            result.extend(
                conditioning_set_values(
                    [[text, metadata]],
                    {"start_percent": overlap_end + _EPS, "end_percent": condition_end},
                )
            )
    return result


def _extract_mask_to_28ch(rgb_video: Any) -> Any:
    torch = importlib.import_module("torch")
    frames, height, width, _ = rgb_video.shape
    mask = rgb_video.movedim(-1, 1).float()
    threshold = 225.0 / 255.0
    red, green, blue = [
        (mask[:, channel : channel + 1] > threshold).float() for channel in range(3)
    ]
    not_red, not_green, not_blue = 1 - red, 1 - green, 1 - blue
    channels = torch.cat(
        (
            red * green * blue,
            red * not_green * not_blue,
            not_red * green * not_blue,
            not_red * not_green * blue,
            red * green * not_blue,
            red * not_green * blue,
            not_red * green * blue,
        ),
        dim=1,
    )
    latent_height, latent_width = height, width
    for _ in range(3):
        latent_height = (latent_height + 1) // 2
        latent_width = (latent_width + 1) // 2
    channels = torch.nn.functional.interpolate(
        channels, size=(latent_height, latent_width), mode="area"
    )
    latent_frames = (frames - 1) // 4 + 1
    padded = torch.cat((channels[:1].repeat(4, 1, 1, 1), channels[1:]), dim=0)
    return padded.view(latent_frames, 28, latent_height, latent_width).unsqueeze(0)


class WanSCAILToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanSCAILToVideo",
            category=_CATEGORY,
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", _VAE),
                _number("width", _INT, 512, 32, 16384, 32),
                _number("height", _INT, 896, 32, 16384, 32),
                _number("length", _INT, 81, 1, 16384, 4),
                _number("batch_size", _INT, 1, 1, 4096),
                InputSpec(
                    "pose_video",
                    _IMAGE,
                    required=False,
                    doc=(
                        "Video used for pose conditioning. Will be downscaled to half the "
                        "resolution of the main video."
                    ),
                ),
                InputSpec(
                    "pose_video_mask",
                    _IMAGE,
                    required=False,
                    doc=(
                        "SCAIL-2 only. Colored per-identity SAM3 mask video at the same resolution "
                        "as pose_video."
                    ),
                ),
                InputSpec(
                    "replacement_mode",
                    _BOOLEAN,
                    required=False,
                    default=False,
                    doc=(
                        "SCAIL-2 only. False = Animation Mode (pose_video_mask should have black "
                        "background). True = Replacement Mode (pose_video_mask should have white "
                        "background)."
                    ),
                ),
                InputSpec(
                    "pose_strength",
                    _FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                    doc="Strength of the pose latent.",
                ),
                InputSpec(
                    "pose_start",
                    _FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                    doc="Start step of the pose conditioning.",
                ),
                InputSpec(
                    "pose_end",
                    _FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                    doc="End step of the pose conditioning.",
                ),
                InputSpec(
                    "reference_image",
                    _IMAGE,
                    required=False,
                    doc=(
                        "Reference image. The first image is the primary reference (composite all "
                        "identities onto it). SCAIL-2: extra batch images are used as additional "
                        "views "
                        "(back view, close-up, occluded background), each needing a matching "
                        "reference_image_mask in that identity's color."
                    ),
                ),
                InputSpec(
                    "reference_image_mask",
                    _IMAGE,
                    required=False,
                    doc=(
                        "SCAIL-2 only. Colored reference mask, batch matching reference_image "
                        "(first = primary reference mask, rest = identity masks for the additional "
                        "reference_image)."
                    ),
                ),
                InputSpec(
                    "clip_vision_output",
                    _CLIP_VISION_OUTPUT,
                    required=False,
                    doc=(
                        "CLIP vision features for conditioning. Model is trained with stretch "
                        "resize to aspect ratio."
                    ),
                ),
                InputSpec(
                    "video_frame_offset",
                    _INT,
                    default=0,
                    widget=NumberWidget(min=0, max=16384, step=1),
                    doc=(
                        "Cumulative output frame this chunk begins at. Wire from the previous "
                        "chunk's video_frame_offset output."
                    ),
                ),
                InputSpec(
                    "previous_frame_count",
                    _INT,
                    default=5,
                    widget=NumberWidget(min=1, max=16384, step=4),
                    doc=(
                        "Tail frames of previous_frames to anchor. SCAIL-2 trained at 5 "
                        "(81-frame chunks, 76-frame step)."
                    ),
                ),
                InputSpec(
                    "previous_frames",
                    _IMAGE,
                    required=False,
                    doc=(
                        "SCAIL-2 only. Full decoded output of the previous chunk. Only the last "
                        "previous_frame_count are used as the extension anchor."
                    ),
                ),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING, display_name="positive"),
                OutputSpec("negative", _CONDITIONING, display_name="negative"),
                OutputSpec(
                    "latent",
                    _LATENT,
                    display_name="latent",
                    doc="Empty latent of the generation size.",
                ),
                OutputSpec(
                    "video_frame_offset",
                    _INT,
                    display_name="video_frame_offset",
                    doc="Adjusted offset + length. Wire into the next chunk.",
                ),
            ),
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
        pose_strength: float,
        pose_start: float,
        pose_end: float,
        video_frame_offset: int,
        previous_frame_count: int,
        replacement_mode: bool = False,
        reference_image: object | None = None,
        clip_vision_output: object | None = None,
        pose_video: object | None = None,
        pose_video_mask: object | None = None,
        reference_image_mask: object | None = None,
        previous_frames: object | None = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        latent = torch.zeros(
            (batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8),
            device=management.intermediate_device(),
        )
        positive = conditioning_set_values(positive, {"ref_mask_flag": not replacement_mode})
        negative = conditioning_set_values(negative, {"ref_mask_flag": not replacement_mode})
        previous: Any = None
        if previous_frames is not None:
            previous = materialize_image(previous_frames)
            if previous.shape[0] > 0:
                previous = previous[-previous_frame_count:]
                video_frame_offset = max(0, video_frame_offset - previous.shape[0])
            else:
                previous = None
        if reference_image is not None:
            references = _upscale(materialize_image(reference_image), width, height, "bicubic")
            if replacement_mode and reference_image_mask is not None:
                masks = _upscale(
                    materialize_image(reference_image_mask), width, height, "nearest-exact"
                )
                masks = masks[
                    [min(index, masks.shape[0] - 1) for index in range(references.shape[0])]
                ]
                references = references * (
                    masks[..., :3].max(dim=-1, keepdim=True).values > 0.1
                ).to(references.dtype)
            reference_latents = [
                vae.encode(references[index : index + 1, :, :, :3])
                for index in range(references.shape[0])
            ]
            positive = _append_values(positive, {"reference_latents": reference_latents})
            negative = _append_values(negative, {"reference_latents": reference_latents})
        if clip_vision_output is not None:
            values = {"clip_vision_output": materialize_clip_vision_output(clip_vision_output)}
            positive, negative = (
                conditioning_set_values(positive, values),
                conditioning_set_values(negative, values),
            )
        pose = materialize_image(pose_video) if pose_video is not None else None
        pose_mask = materialize_image(pose_video_mask) if pose_video_mask is not None else None
        pose = (
            None
            if pose is not None and pose.shape[0] <= video_frame_offset
            else (pose[video_frame_offset:] if pose is not None else None)
        )
        pose_mask = (
            None
            if pose_mask is not None and pose_mask.shape[0] <= video_frame_offset
            else (pose_mask[video_frame_offset:] if pose_mask is not None else None)
        )
        available = [item.shape[0] for item in (pose, pose_mask) if item is not None]
        if available:
            kept = ((min(min(available), length) - 1) // 4) * 4 + 1
            pose, pose_mask = (
                (pose[:kept] if pose is not None else None),
                (pose_mask[:kept] if pose_mask is not None else None),
            )
        if pose is not None:
            pose_latent = (
                vae.encode(_upscale(pose[:length], width // 2, height // 2, "area")[:, :, :, :3])
                * pose_strength
            )
            positive, negative = (
                _set_values_with_timestep_range(
                    positive, {"pose_video_latent": pose_latent}, pose_start, pose_end
                ),
                _set_values_with_timestep_range(
                    negative, {"pose_video_latent": pose_latent}, pose_start, pose_end
                ),
            )
        if pose_mask is not None:
            values = {
                "driving_mask_28ch": _extract_mask_to_28ch(
                    _upscale(pose_mask[:length], width // 2, height // 2, "area")
                )
            }
            positive, negative = (
                conditioning_set_values(positive, values),
                conditioning_set_values(negative, values),
            )
        if reference_image_mask is not None and reference_image is not None:
            masks = _upscale(
                materialize_image(reference_image_mask), width, height, "nearest-exact"
            )
            reference_count = materialize_image(reference_image).shape[0]
            additional = [
                _extract_mask_to_28ch(masks[min(index, masks.shape[0] - 1)][None])
                for index in range(1, reference_count)
            ]
            first = _extract_mask_to_28ch(masks[:1])
            zeros = torch.zeros(
                (1, latent.shape[2], 28, first.shape[-2], first.shape[-1]),
                device=first.device,
                dtype=first.dtype,
            )
            values = {"ref_mask_28ch": torch.cat([*additional, first, zeros], dim=1)}
            positive, negative = (
                conditioning_set_values(positive, values),
                conditioning_set_values(negative, values),
            )
        output: dict[str, Any] = {"samples": latent}
        if previous is not None:
            previous_latent = vae.encode(_upscale(previous, width, height, "bicubic")[:, :, :, :3])
            frames = min(previous_latent.shape[2], latent.shape[2])
            latent[:, :, :frames] = previous_latent[:, :, :frames].to(latent.dtype)
            noise_mask = torch.ones(
                (1, 1, latent.shape[2], latent.shape[-2], latent.shape[-1]),
                device=latent.device,
                dtype=latent.dtype,
            )
            noise_mask[:, :, :frames] = 0.0
            output["noise_mask"] = noise_mask
        return cls.outputs(
            positive=positive,
            negative=negative,
            latent=output,
            video_frame_offset=video_frame_offset + length,
        )
