"""Wan Move track visualization."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_INT,
    IMAGE_TYPE,
    TRACKS_TYPE,
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

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_TRACKS = TypeExpr.concrete(TRACKS_TYPE)


def _draw(
    video: Any,
    tracks: Any,
    visibility: Any,
    line_resolution: int,
    circle_size: int,
    opacity: float,
    line_width: int,
) -> Any:
    numpy = importlib.import_module("numpy")
    image = importlib.import_module("PIL.Image")
    drawing = importlib.import_module("PIL.ImageDraw")
    colors = ((102, 153, 255), (0, 255, 255), (255, 255, 0), (255, 102, 204), (0, 255, 0))
    frames, paths = video.byte().cpu().numpy(), tracks.long().detach().cpu().numpy()
    visible = visibility.detach().cpu().numpy()
    result: list[Any] = []
    for time, frame in enumerate(frames):
        overlay = image.new("RGBA", (frame.shape[1], frame.shape[0]), (0, 0, 0, 0))
        draw = drawing.Draw(overlay)
        polylines: list[Any] = []
        for track in range(paths.shape[1]):
            if not visible[time, track]:
                continue
            x, y = paths[time, track]
            color = colors[track % len(colors)] + (int(255 * opacity),)
            draw.ellipse(
                (x - circle_size, y - circle_size, x + circle_size, y + circle_size), fill=color
            )
            points = paths[max(time - line_resolution, 0) : time + 1, track][::-1]
            if len(points) > 1:
                polylines.append((points, color[:3]))
        layer = numpy.array(overlay)
        alpha = layer[:, :, 3:4] / 255.0
        frame_rgb = layer[:, :, :3] * alpha + frame.astype(numpy.float32) * (1 - alpha)
        if polylines:
            trail_overlay = image.new("RGBA", (frame.shape[1], frame.shape[0]), (0, 0, 0, 0))
            trail_draw = drawing.Draw(trail_overlay, "RGBA")
            for points, color in polylines:
                lengths = [
                    ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
                    for a, b in zip(points[:-1], points[1:], strict=False)
                ]
                length = sum(lengths)
                if not length:
                    continue
                walked = 0.0
                for (start, end), segment in zip(
                    zip(points[:-1], points[1:], strict=False), lengths, strict=True
                ):
                    steps = max(int(segment), 1)
                    for step in range(steps):
                        ratio = (walked + step / steps * segment) / length
                        x = int(start[0] + (end[0] - start[0]) * step / steps)
                        y = int(start[1] + (end[1] - start[1]) * step / steps)
                        trail_draw.line(
                            [(x, y), (x + 1, y)],
                            fill=color + (int(255 * (1 - ratio) * opacity),),
                            width=max(int(line_width * (1 - ratio)), 1),
                        )
                    walked += segment
            trail_layer = numpy.array(trail_overlay)
            alpha = trail_layer[:, :, 3:4] / 255.0
            frame_rgb = trail_layer[:, :, :3] * alpha + frame_rgb * (1 - alpha)
        result.append(image.fromarray(frame_rgb.astype(numpy.uint8)))
    return result


class WanMoveVisualizeTracks(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanMoveVisualizeTracks",
            category="model/conditioning/wan/move",
            inputs=(
                InputSpec("images", _IMAGE),
                InputSpec("tracks", _TRACKS, required=False),
                InputSpec(
                    "line_resolution",
                    TypeExpr.concrete(CORE_INT),
                    default=24,
                    widget=NumberWidget(min=1, max=1024),
                ),
                InputSpec(
                    "circle_size",
                    TypeExpr.concrete(CORE_INT),
                    default=12,
                    widget=NumberWidget(min=1, max=128),
                    advanced=True,
                ),
                InputSpec(
                    "opacity",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.75,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "line_width",
                    TypeExpr.concrete(CORE_INT),
                    default=16,
                    widget=NumberWidget(min=1, max=128),
                    advanced=True,
                ),
            ),
            outputs=(OutputSpec("IMAGE", _IMAGE),),
        )

    @classmethod
    def execute(
        cls,
        *,
        images: object,
        line_resolution: int,
        circle_size: int,
        opacity: float,
        line_width: int,
        tracks: object | None = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        image = materialize_image(images)
        if tracks is None:
            return cls.outputs(IMAGE=image)
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        values = materialize_latent(tracks)
        if image.shape[0] != values["track_path"].shape[0]:
            image = image.repeat(values["track_path"].shape[0] // image.shape[0], 1, 1, 1)
        frames = _draw(
            image * 255.0,
            values["track_path"],
            values["track_visibility"],
            line_resolution,
            circle_size,
            opacity,
            line_width,
        )
        numpy = importlib.import_module("numpy")
        return cls.outputs(
            IMAGE=torch.from_numpy(
                numpy.stack([frame_array for frame_array in map(numpy.array, frames)])
            )
            .movedim(-1, 1)
            .float()
            .movedim(1, -1)
            .to(management.intermediate_device())
            / 255.0
        )
