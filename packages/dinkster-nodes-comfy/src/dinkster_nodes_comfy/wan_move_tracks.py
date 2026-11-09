"""Native Wan Move track construction."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    MASK_TYPE,
    TRACKS_TYPE,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
    materialize_latent,
)

_CATEGORY = "model/conditioning/wan/move"
_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_INT = TypeExpr.concrete(CORE_INT)
_MASK = TypeExpr.concrete(MASK_TYPE)
_STRING = TypeExpr.concrete(CORE_STRING)
_TRACKS = TypeExpr.concrete(TRACKS_TYPE)
_INTERPOLATIONS = ("linear", "ease_in", "ease_out", "ease_in_out", "constant")


def parse_json_tracks(tracks: str | Sequence[str]) -> list[Any]:
    tracks_data: list[Any] = []
    try:
        if isinstance(tracks, str):
            tracks_data.extend(json.loads(tracks.replace("'", '"')))
        else:
            for track in tracks:
                tracks_data.append(json.loads(track.replace("'", '"')))
        if tracks_data and isinstance(tracks_data[0], dict) and "x" in tracks_data[0]:
            tracks_data = [tracks_data]
    except json.JSONDecodeError:
        tracks_data = []
    return tracks_data


def _runtime() -> tuple[Any, Any]:
    initialize_inference_process()
    return (
        importlib.import_module("torch"),
        importlib.import_module("dinkster_inference.model_management"),
    )


def _visibility(track_mask: object, length: int, count: int, torch: Any, device: Any) -> Any:
    if track_mask is None:
        return torch.ones((length, count), dtype=torch.bool, device=device)
    mask = materialize_image(track_mask)
    return (mask > 0).any(dim=(1, 2)).unsqueeze(-1)


class WanMoveTracksFromCoords(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanMoveTracksFromCoords",
            category=_CATEGORY,
            inputs=(
                InputSpec(
                    "track_coords",
                    _STRING,
                    required=False,
                    default="[]",
                    force_input=True,
                    widget=StringWidget(multiline=False),
                ),
                InputSpec("track_mask", _MASK, required=False),
            ),
            outputs=(
                OutputSpec("TRACKS", _TRACKS),
                OutputSpec("track_length", _INT, display_name="track_length"),
            ),
        )

    @classmethod
    def execute(
        cls, *, track_coords: str | Sequence[str] = "[]", track_mask: object = None
    ) -> Mapping[str, object]:
        torch, management = _runtime()
        tracks_data = parse_json_tracks(track_coords)
        track_length = len(tracks_data[0])
        track_list = [
            [[track[frame]["x"], track[frame]["y"]] for track in tracks_data]
            for frame in range(track_length)
        ]
        tracks = torch.tensor(
            track_list, dtype=torch.float32, device=management.intermediate_device()
        )
        return cls.outputs(
            TRACKS={
                "track_path": tracks,
                "track_visibility": _visibility(
                    track_mask, track_length, tracks.shape[-2], torch, tracks.device
                ),
            },
            track_length=track_length,
        )


class GenerateTracks(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="GenerateTracks",
            display_name="Generate Video Tracks",
            category=_CATEGORY,
            search_terms=("motion paths", "camera movement", "trajectory"),
            inputs=(
                InputSpec(
                    "width", _INT, default=832, widget=NumberWidget(min=16, max=4096, step=16)
                ),
                InputSpec(
                    "height", _INT, default=480, widget=NumberWidget(min=16, max=4096, step=16)
                ),
                InputSpec(
                    "start_x",
                    _FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                    doc="Normalized X coordinate (0-1) for start position.",
                ),
                InputSpec(
                    "start_y",
                    _FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                    doc="Normalized Y coordinate (0-1) for start position.",
                ),
                InputSpec(
                    "end_x",
                    _FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                    doc="Normalized X coordinate (0-1) for end position.",
                ),
                InputSpec(
                    "end_y",
                    _FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                    doc="Normalized Y coordinate (0-1) for end position.",
                ),
                InputSpec("num_frames", _INT, default=81, widget=NumberWidget(min=1, max=1024)),
                InputSpec("num_tracks", _INT, default=5, widget=NumberWidget(min=1, max=100)),
                InputSpec(
                    "track_spread",
                    _FLOAT,
                    default=0.025,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                    doc=(
                        "Normalized distance between tracks. Tracks are spread perpendicular "
                        "to the motion direction."
                    ),
                ),
                InputSpec(
                    "bezier",
                    _BOOLEAN,
                    default=False,
                    doc="Enable Bezier curve path using the mid point as control point.",
                ),
                InputSpec(
                    "mid_x",
                    _FLOAT,
                    default=0.5,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                    doc=(
                        "Normalized X control point for Bezier curve. Only used when 'bezier' is "
                        "enabled."
                    ),
                ),
                InputSpec(
                    "mid_y",
                    _FLOAT,
                    default=0.5,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                    doc=(
                        "Normalized Y control point for Bezier curve. Only used when 'bezier' is "
                        "enabled."
                    ),
                ),
                InputSpec(
                    "interpolation",
                    _COMBO,
                    widget=ComboWidget(_INTERPOLATIONS),
                    doc="Controls the timing/speed of movement along the path.",
                ),
                InputSpec(
                    "track_mask",
                    _MASK,
                    required=False,
                    doc="Optional mask to indicate visible frames.",
                ),
            ),
            outputs=(
                OutputSpec("TRACKS", _TRACKS),
                OutputSpec("track_length", _INT, display_name="track_length"),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        width: int,
        height: int,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        num_frames: int,
        num_tracks: int,
        track_spread: float,
        bezier: bool = False,
        mid_x: float = 0.5,
        mid_y: float = 0.5,
        interpolation: Literal[
            "linear", "ease_in", "ease_out", "ease_in_out", "constant"
        ] = "linear",
        track_mask: object = None,
    ) -> Mapping[str, object]:
        torch, management = _runtime()
        start_x_px, start_y_px = start_x * width, start_y * height
        mid_x_px, mid_y_px = mid_x * width, mid_y * height
        end_x_px, end_y_px = end_x * width, end_y * height
        spread = track_spread * (width + height) / 2
        time = torch.linspace(0, 1, num_frames, device=management.intermediate_device())
        if interpolation == "constant":
            values = torch.zeros_like(time)
        elif interpolation == "linear":
            values = time
        elif interpolation == "ease_in":
            values = time**2
        elif interpolation == "ease_out":
            values = 1 - (1 - time) ** 2
        elif interpolation == "ease_in_out":
            values = time * time * (3 - 2 * time)

        if bezier:
            inverse = 1 - values
            x_positions = (
                inverse**2 * start_x_px + 2 * inverse * values * mid_x_px + values**2 * end_x_px
            )
            y_positions = (
                inverse**2 * start_y_px + 2 * inverse * values * mid_y_px + values**2 * end_y_px
            )
            tangent_x = 2 * inverse * (mid_x_px - start_x_px) + 2 * values * (end_x_px - mid_x_px)
            tangent_y = 2 * inverse * (mid_y_px - start_y_px) + 2 * values * (end_y_px - mid_y_px)
        else:
            x_positions = start_x_px + (end_x_px - start_x_px) * values
            y_positions = start_y_px + (end_y_px - start_y_px) * values
            tangent_x = torch.full_like(time, end_x_px - start_x_px)
            tangent_y = torch.full_like(time, end_y_px - start_y_px)

        tracks: list[list[list[float]]] = []
        for frame in range(num_frames):
            tx, ty = float(tangent_x[frame].item()), float(tangent_y[frame].item())
            length = (tx**2 + ty**2) ** 0.5
            perpendicular: tuple[float, float] = (
                (-ty / length, tx / length) if length > 0 else (1.0, 0.0)
            )
            tracks.append(
                [
                    [
                        float(x_positions[frame].item())
                        + perpendicular[0] * (track - (num_tracks - 1) / 2) * spread,
                        float(y_positions[frame].item())
                        + perpendicular[1] * (track - (num_tracks - 1) / 2) * spread,
                    ]
                    for track in range(num_tracks)
                ]
            )
        path = torch.tensor(tracks, dtype=torch.float32, device=management.intermediate_device())
        return cls.outputs(
            TRACKS={
                "track_path": path,
                "track_visibility": _visibility(
                    track_mask, num_frames, num_tracks, torch, path.device
                ),
            },
            track_length=num_frames,
        )


class WanMoveConcatTrack(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanMoveConcatTrack",
            category=_CATEGORY,
            inputs=(InputSpec("tracks_1", _TRACKS), InputSpec("tracks_2", _TRACKS, required=False)),
            outputs=(OutputSpec("TRACKS", _TRACKS),),
        )

    @classmethod
    def execute(cls, *, tracks_1: object, tracks_2: object = None) -> Mapping[str, object]:
        initialize_inference_process()
        if tracks_2 is None:
            return cls.outputs(TRACKS=tracks_1)
        torch = importlib.import_module("torch")
        first, second = materialize_latent(tracks_1), materialize_latent(tracks_2)
        return cls.outputs(
            TRACKS={
                "track_path": torch.cat((first["track_path"], second["track_path"]), dim=1),
                "track_visibility": torch.cat(
                    (first["track_visibility"], second["track_visibility"]), dim=-1
                ),
            }
        )
