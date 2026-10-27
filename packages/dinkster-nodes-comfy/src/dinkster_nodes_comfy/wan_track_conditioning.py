"""Native Wan motion-track conditioning node."""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CLIP_VISION_OUTPUT_TYPE,
    CONDITIONING_TYPE,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    initialize_inference_process,
    materialize_clip_vision_output,
    materialize_image,
)

from .conditioning import conditioning_set_values
from .wan_image_conditioning import WanImageToVideo

_CLIP_VISION_OUTPUT = TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE)
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_STRING = TypeExpr.concrete(CORE_STRING)
_VAE = TypeExpr.concrete(VAE_TYPE)
_FIXED_LENGTH = 121


def _integer(
    name: str, default: int, minimum: int, maximum: int, step: int | None = None
) -> InputSpec:
    return InputSpec(
        name,
        _INT,
        default=default,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


def parse_json_tracks(tracks: str | Sequence[str]) -> list[Any]:
    tracks_data: list[Any] = []
    try:
        if isinstance(tracks, str):
            tracks_data.extend(json.loads(tracks.replace("'", '"')))
        else:
            for track_string in tracks:
                tracks_data.append(json.loads(track_string.replace("'", '"')))
        if tracks_data and isinstance(tracks_data[0], dict) and "x" in tracks_data[0]:
            tracks_data = [tracks_data]
        elif (
            tracks_data
            and isinstance(tracks_data[0], list)
            and tracks_data[0]
            and isinstance(tracks_data[0][0], dict)
            and "x" in tracks_data[0][0]
        ):
            pass
    except json.JSONDecodeError:
        tracks_data = []
    return tracks_data


def process_tracks(tracks_np: Any, frame_size: tuple[int, int], num_frames: int) -> Any:
    torch = importlib.import_module("torch")
    tracks = torch.from_numpy(tracks_np).float()
    if tracks.shape[1] == _FIXED_LENGTH:
        tracks = torch.permute(tracks, (1, 0, 2, 3))
    tracks, visibles = tracks[..., :2], tracks[..., 2:3]
    short_edge = min(*frame_size)
    frame_center = torch.tensor([*frame_size]).type_as(tracks) / 2
    tracks = (tracks - frame_center) / short_edge * 2
    visibles = visibles * 2 - 1
    trange = torch.linspace(-1, 1, tracks.shape[0]).view(-1, 1, 1, 1).expand(*visibles.shape)
    output = torch.cat([trange, tracks, visibles], dim=-1).view(_FIXED_LENGTH, -1, 4)
    output_first = output[:1]
    output_rest = output[1:]
    divisor = math.gcd(120, num_frames)
    output_rest = torch.repeat_interleave(output_rest, num_frames // divisor, dim=0)[
        1 :: 120 // divisor
    ]
    return torch.cat([output_first, output_rest], dim=0)


def pad_pts(track: Sequence[Mapping[str, object]]) -> Any:
    numpy = importlib.import_module("numpy")
    points = numpy.array([[point["x"], point["y"], 1] for point in track], dtype=numpy.float32)
    if points.shape[0] < _FIXED_LENGTH:
        padding = numpy.zeros((_FIXED_LENGTH - points.shape[0], 3), dtype=numpy.float32)
        points = numpy.vstack((points, padding))
    else:
        points = points[:_FIXED_LENGTH]
    return points.reshape(_FIXED_LENGTH, 1, 3)


def ind_sel(target: Any, index: Any, dim: int = 1) -> Any:
    assert len(index.shape) > dim, (
        f"Index must have the target dim, but get dim: {dim}, ind shape: {index.shape}"
    )
    target = target.expand(
        *tuple(
            [
                index.shape[position] if target.shape[position] == 1 else -1
                for position in range(dim)
            ]
            + [-1] * (len(target.shape) - dim)
        )
    )
    padded_index = index
    if len(target.shape) > dim + 1:
        for _ in range(len(target.shape) - (dim + 1)):
            padded_index = padded_index.unsqueeze(-1)
        padded_index = padded_index.expand(*(-1,) * (dim + 1), *target.shape[(dim + 1) :])
    torch = importlib.import_module("torch")
    return torch.gather(target, dim=dim, index=padded_index)


def merge_final(vertex_attributes: Any, weight: Any, vertex_assignment: Any) -> Any:
    torch = importlib.import_module("torch")
    target_dim = len(vertex_assignment.shape) - 1
    if len(vertex_attributes.shape) == 2:
        assert vertex_attributes.shape[0] > vertex_assignment.max()
        tensor = vertex_attributes.reshape([1] * target_dim + list(vertex_attributes.shape))
        selected_attributes = ind_sel(tensor, vertex_assignment.type(torch.long), dim=target_dim)
    else:
        assert vertex_attributes.shape[1] > vertex_assignment.max()
        tensor = vertex_attributes.reshape(
            [vertex_attributes.shape[0]]
            + [1] * (target_dim - 1)
            + list(vertex_attributes.shape[1:])
        )
        selected_attributes = ind_sel(tensor, vertex_assignment.type(torch.long), dim=target_dim)
    return (selected_attributes * weight.unsqueeze(-1)).sum(dim=-2)


def _patch_motion_single(
    tracks: Any, video: Any, temperature: float, vae_divide: tuple[int, int], topk: int
) -> tuple[Any, Any]:
    torch = importlib.import_module("torch")
    _, time, height, width = video.shape
    points = tracks.shape[2]
    _, tracks_xy, visible = torch.split(tracks, [1, 2, 1], dim=-1)
    tracks_normalized = tracks_xy / torch.tensor(
        [width / min(height, width), height / min(height, width)], device=tracks_xy.device
    )
    tracks_normalized = tracks_normalized.clamp(-1, 1)
    visible = visible.clamp(0, 1)
    xx = torch.linspace(-width / min(height, width), width / min(height, width), width)
    yy = torch.linspace(-height / min(height, width), height / min(height, width), height)
    grid = torch.stack(torch.meshgrid(yy, xx, indexing="ij")[::-1], dim=-1).to(tracks_xy.device)
    tracks_padding = tracks_xy[:, 1:]
    visible_padding = visible[:, 1:]
    visible_aligned = visible_padding.view(time - 1, 4, *visible_padding.shape[2:]).sum(1)
    tracks_aligned = (tracks_padding * visible_padding).view(
        time - 1, 4, *tracks_padding.shape[2:]
    ).sum(1) / (visible_aligned + 1e-5)
    distance = (tracks_aligned[:, None, None] - grid[None, :, :, None]).pow(2).sum(-1)
    weight = torch.exp(-distance * temperature) * visible_aligned.clamp(0, 1).view(
        time - 1, 1, 1, points
    )
    vertex_weight, vertex_index = torch.topk(weight, k=min(topk, weight.shape[-1]), dim=-1)
    point_feature = torch.nn.functional.grid_sample(
        video.permute(1, 0, 2, 3)[:1],
        tracks_normalized[:, :1].type(video.dtype),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    point_feature = point_feature.squeeze(0).squeeze(1).permute(1, 0)
    output_feature = merge_final(point_feature, vertex_weight, vertex_index).permute(3, 0, 1, 2)
    output_weight = vertex_weight.sum(-1)
    mixed_feature = output_feature + video[:, 1:] * (1 - output_weight.clamp(0, 1))
    output_feature_full = torch.cat([video[:, :1], mixed_feature], dim=1)
    output_mask_full = torch.cat([torch.ones_like(output_weight[:1]), output_weight], dim=0)
    return output_mask_full[None].expand(vae_divide[0], -1, -1, -1), output_feature_full


def patch_motion(
    tracks: Sequence[Any],
    video: Any,
    temperature: float = 220.0,
    vae_divide: tuple[int, int] = (4, 16),
    topk: int = 2,
) -> tuple[Any, Any]:
    torch = importlib.import_module("torch")
    masks: list[Any] = []
    features: list[Any] = []
    for batch in range(len(tracks)):
        mask, feature = _patch_motion_single(
            tracks[batch], video[batch], temperature, vae_divide, topk
        )
        masks.append(mask)
        features.append(feature)
    return torch.stack(masks, dim=0), torch.stack(features, dim=0)


class WanTrackToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanTrackToVideo",
            search_terms=(
                "motion tracking",
                "trajectory video",
                "point tracking",
                "keypoint animation",
            ),
            category="model/conditioning/wan/move",
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", _VAE),
                InputSpec("tracks", _STRING, default="[]", widget=StringWidget(multiline=True)),
                _integer("width", 832, 16, 16384, 16),
                _integer("height", 480, 16, 16384, 16),
                _integer("length", 81, 1, 16384, 4),
                _integer("batch_size", 1, 1, 4096),
                InputSpec(
                    "temperature",
                    _FLOAT,
                    default=220.0,
                    advanced=True,
                    widget=NumberWidget(min=1.0, max=1000.0, step=0.1),
                ),
                InputSpec(
                    "topk", _INT, default=2, advanced=True, widget=NumberWidget(min=1, max=10)
                ),
                InputSpec("start_image", _IMAGE),
                InputSpec("clip_vision_output", _CLIP_VISION_OUTPUT, required=False),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING, display_name="positive"),
                OutputSpec("negative", _CONDITIONING, display_name="negative"),
                OutputSpec("latent", _LATENT, display_name="latent"),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        vae: Any,
        tracks: str | Sequence[str],
        width: int,
        height: int,
        length: int,
        batch_size: int,
        temperature: float,
        topk: int,
        start_image: object | None = None,
        clip_vision_output: object | None = None,
    ) -> Mapping[str, object]:
        tracks_data = parse_json_tracks(tracks)
        if not tracks_data:
            return WanImageToVideo.execute(
                positive=positive,
                negative=negative,
                vae=vae,
                width=width,
                height=height,
                length=length,
                batch_size=batch_size,
                start_image=start_image,
                clip_vision_output=clip_vision_output,
            )

        initialize_inference_process()
        numpy = importlib.import_module("numpy")
        torch = importlib.import_module("torch")
        latent_formats = importlib.import_module("dinkster_inference.latent_formats")
        management = importlib.import_module("dinkster_inference.model_management")
        utils = importlib.import_module("dinkster_inference.utils")
        latent = torch.zeros(
            (batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8),
            device=management.intermediate_device(),
        )
        if isinstance(tracks_data[0][0], dict):
            tracks_data = [tracks_data]
        processed_tracks: list[Any] = []
        for batch in tracks_data:
            points = [pad_pts(track) for track in batch]
            processed_tracks.append(
                process_tracks(numpy.stack(points, axis=0), (width, height), length - 1).unsqueeze(
                    0
                )
            )

        if start_image is not None:
            start = materialize_image(start_image)[:batch_size]
            start = utils.common_upscale(
                start.movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            videos = (
                torch.ones(
                    (start.shape[0], length, height, width, start.shape[-1]),
                    device=start.device,
                    dtype=start.dtype,
                )
                * 0.5
            )
            for index in range(start.shape[0]):
                videos[index, 0] = start[index]
            videos = utils.resize_to_batch_size(videos, batch_size)
            encoded = [vae.encode(videos[index, :, :, :, :3]) for index in range(batch_size)]
            latent_video = torch.cat(encoded, dim=0)
            latent_video = latent_formats.Wan21().process_in(latent_video)
            processed_tracks = utils.resize_list_to_batch_size(processed_tracks, batch_size)
            mask, concat_latent_image = patch_motion(
                processed_tracks,
                latent_video,
                temperature=temperature,
                topk=topk,
                vae_divide=(4, 16),
            )
            concat_latent_image = latent_formats.Wan21().process_out(concat_latent_image)
            values = {"concat_mask": -mask + 1.0, "concat_latent_image": concat_latent_image}
            positive = conditioning_set_values(positive, values)
            negative = conditioning_set_values(negative, values)

        if clip_vision_output is not None:
            values = {"clip_vision_output": materialize_clip_vision_output(clip_vision_output)}
            positive = conditioning_set_values(positive, values)
            negative = conditioning_set_values(negative, values)
        return cls.outputs(positive=positive, negative=negative, latent={"samples": latent})
