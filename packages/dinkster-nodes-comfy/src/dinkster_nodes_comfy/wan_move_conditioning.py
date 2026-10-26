"""Wan Move trajectory conditioning."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CLIP_VISION_OUTPUT_TYPE,
    CONDITIONING_TYPE,
    CORE_FLOAT,
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    TRACKS_TYPE,
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
    materialize_latent,
)

from .conditioning import conditioning_set_values

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_TRACKS = TypeExpr.concrete(TRACKS_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)


def _positions(tracks: Any, visibility: Any, ratios: list[int], height: int, width: int) -> Any:
    torch = importlib.import_module("torch")
    frames, count, _ = tracks.shape
    time, down_height, down_width = ratios
    positions = -torch.ones(count, (frames - 1) // time + 1, 2, dtype=torch.long)
    indices = torch.randperm(count)
    tracks, visibility = tracks[:, indices], visibility[:, indices]
    for frame in range(0, frames, time):
        current, visible = tracks[frame], visibility[frame]
        for track in range(count):
            if (
                not visible[track]
                or current[track][0] < 0
                or current[track][1] < 0
                or current[track][0] >= width
                or current[track][1] >= height
            ):
                continue
            x, y = current[track]
            positions[track, frame // time] = torch.stack((y // down_height, x // down_width))
    return positions


def _replace(feature: Any, positions: Any, strength: float) -> Any:
    torch = importlib.import_module("torch")
    batch, _, _, _, _ = feature.shape
    assert batch == positions.shape[0], "Batch size mismatch."
    positions = positions[:, torch.randperm(positions.shape[1])]
    current = positions[:, :, 1:]
    indices = ((current[..., 0] >= 0) & (current[..., 1] >= 0)).nonzero(as_tuple=False)
    if indices.shape[0] == 0:
        return feature
    batch_index, track_index, relative_time = indices[:, 0], indices[:, 1], indices[:, 2]
    target_time = relative_time + 1
    target_height = current[batch_index, track_index, relative_time, 0].long()
    target_width = current[batch_index, track_index, relative_time, 1].long()
    source_height = positions[batch_index, track_index, 0, 0].long()
    source_width = positions[batch_index, track_index, 0, 1].long()
    source = feature[batch_index, :, 0, source_height, source_width]
    destination = feature[batch_index, :, target_time, target_height, target_width]
    feature[batch_index, :, target_time, target_height, target_width] = (
        destination + (source - destination) * strength
    )
    return feature


class WanMoveTrackToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanMoveTrackToVideo",
            category="model/conditioning/wan/move",
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", _VAE),
                InputSpec("tracks", _TRACKS, required=False),
                InputSpec(
                    "strength",
                    _FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "width", _INT, default=832, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec(
                    "height", _INT, default=480, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec(
                    "length", _INT, default=81, widget=NumberWidget(min=1, max=16384, step=4)
                ),
                InputSpec("batch_size", _INT, default=1, widget=NumberWidget(min=1, max=4096)),
                InputSpec("start_image", _IMAGE, required=False),
                InputSpec(
                    "clip_vision_output", TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE), required=False
                ),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING),
                OutputSpec("negative", _CONDITIONING),
                OutputSpec("latent", _LATENT),
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
        strength: float,
        tracks: object | None = None,
        start_image: object | None = None,
        clip_vision_output: object | None = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        latent = torch.zeros(
            (batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8),
            device=management.intermediate_device(),
        )
        if start_image is not None:
            utils = importlib.import_module("dinkster_inference.utils")
            image = utils.common_upscale(
                materialize_image(start_image)[:length].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            ).movedim(1, -1)
            full = (
                torch.ones(
                    (length, height, width, image.shape[-1]), device=image.device, dtype=image.dtype
                )
                * 0.5
            )
            full[: image.shape[0]] = image
            encoded = vae.encode(full[:, :, :, :3])
            mask = torch.ones(
                (1, 1, latent.shape[2], encoded.shape[-2], encoded.shape[-1]),
                device=image.device,
                dtype=image.dtype,
            )
            mask[:, :, : ((image.shape[0] - 1) // 4) + 1] = 0.0
            positioned = encoded
            if tracks is not None and strength > 0.0:
                track_values = materialize_latent(tracks)
                path = track_values["track_path"][:length]
                visible = track_values.get(
                    "track_visibility",
                    torch.ones((length, path.shape[-2]), dtype=torch.bool, device=latent.device),
                )
                position = _positions(path, visible, [4, 8, 8], height, width).unsqueeze(0)
                position = utils.resize_to_batch_size(position, batch_size)
                positioned = _replace(encoded, position, strength)
            positive = conditioning_set_values(
                positive, {"concat_latent_image": positioned, "concat_mask": mask}
            )
            negative = conditioning_set_values(
                negative, {"concat_latent_image": encoded, "concat_mask": mask}
            )
        if clip_vision_output is not None:
            values = {"clip_vision_output": materialize_clip_vision_output(clip_vision_output)}
            positive, negative = (
                conditioning_set_values(positive, values),
                conditioning_set_values(negative, values),
            )
        return cls.outputs(positive=positive, negative=negative, latent={"samples": latent})
