"""Native SAM3 tracked-object video preview."""

from __future__ import annotations

import importlib
import io
import uuid
from collections.abc import Mapping
from fractions import Fraction
from typing import Any, ClassVar, cast

from dinkster_api.v1 import (
    CORE_FLOAT,
    IMAGE_TYPE,
    SAM3_TRACK_DATA_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
    materialize_latent,
    publish_preview_artifact,
)


class SAM3_TrackPreview(Node):
    COLORS = (
        (0.12, 0.47, 0.71),
        (1.0, 0.5, 0.05),
        (0.17, 0.63, 0.17),
        (0.84, 0.15, 0.16),
        (0.58, 0.4, 0.74),
        (0.55, 0.34, 0.29),
        (0.89, 0.47, 0.76),
        (0.5, 0.5, 0.5),
        (0.74, 0.74, 0.13),
        (0.09, 0.75, 0.81),
        (0.94, 0.76, 0.06),
        (0.42, 0.68, 0.84),
    )
    _glyph_cache: ClassVar[dict[tuple[Any, int], Any]] = {}

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SAM3_TrackPreview",
            display_name="SAM3 Track Preview",
            category="image/detection",
            inputs=(
                InputSpec(
                    "track_data", TypeExpr.concrete(SAM3_TRACK_DATA_TYPE), display_name="track_data"
                ),
                InputSpec(
                    "images", TypeExpr.concrete(IMAGE_TYPE), display_name="images", required=False
                ),
                InputSpec(
                    "opacity",
                    TypeExpr.concrete(CORE_FLOAT),
                    display_name="opacity",
                    default=0.5,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.05),
                ),
                InputSpec(
                    "fps",
                    TypeExpr.concrete(CORE_FLOAT),
                    display_name="fps",
                    default=24.0,
                    widget=NumberWidget(min=1.0, max=120.0, step=1.0),
                ),
            ),
            output_node=True,
            emits_previews=True,
            idempotent=False,
        )

    @staticmethod
    def _get_glyphs(device: Any, scale: int = 3) -> Any:
        key = (device, scale)
        if key in SAM3_TrackPreview._glyph_cache:
            return SAM3_TrackPreview._glyph_cache[key]
        torch = importlib.import_module("torch")
        functional = importlib.import_module("torch.nn.functional")
        atlas = torch.tensor(
            [
                [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]],
                [[0, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 1]],
                [[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]],
                [[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
                [[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]],
                [[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]],
                [[1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
                [[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
            ],
            dtype=torch.bool,
        )
        glyphs: list[Any] = []
        outlines: list[Any] = []
        for digit in range(10):
            glyph = atlas[digit].repeat_interleave(scale, 0).repeat_interleave(scale, 1)
            padded = functional.pad(glyph.float().unsqueeze(0).unsqueeze(0), (1, 1, 1, 1))
            outline = functional.max_pool2d(padded, 3, stride=1, padding=1)[0, 0] > 0
            glyphs.append(glyph.to(device))
            outlines.append(outline.to(device))
        gh, gw = glyphs[0].shape
        oh, ow = outlines[0].shape
        result = (glyphs, outlines, gh, gw, oh, ow)
        SAM3_TrackPreview._glyph_cache[key] = result
        return result

    @staticmethod
    def _draw_number_gpu(
        frame: Any, number: int, cx: int, cy: int, color: Any, scale: int = 3
    ) -> None:
        torch = importlib.import_module("torch")
        height, width = frame.shape[:2]
        glyphs, outlines, gh, gw, oh, ow = SAM3_TrackPreview._get_glyphs(frame.device, scale)
        color_t = torch.tensor(color, device=frame.device, dtype=frame.dtype)
        digits = [int(digit) for digit in str(number)]
        total_w = len(digits) * (gw + scale) - scale
        x0 = cx - total_w // 2
        y0 = cy - gh // 2
        for index, digit in enumerate(digits):
            dx = x0 + index * (gw + scale)
            oy0, ox0 = y0 - 1, dx - 1
            osy1, osx1 = max(0, -oy0), max(0, -ox0)
            osy2, osx2 = min(oh, height - oy0), min(ow, width - ox0)
            if osy2 > osy1 and osx2 > osx1:
                fy1, fx1 = oy0 + osy1, ox0 + osx1
                frame[fy1 : fy1 + (osy2 - osy1), fx1 : fx1 + (osx2 - osx1)][
                    outlines[digit][osy1:osy2, osx1:osx2]
                ] = 0
            sy1, sx1 = max(0, -y0), max(0, -dx)
            sy2, sx2 = min(gh, height - y0), min(gw, width - dx)
            if sy2 > sy1 and sx2 > sx1:
                fy1, fx1 = y0 + sy1, dx + sx1
                frame[fy1 : fy1 + (sy2 - sy1), fx1 : fx1 + (sx2 - sx1)][
                    glyphs[digit][sy1:sy2, sx1:sx2]
                ] = color_t

    @classmethod
    def execute(
        cls, *, track_data: Any, images: Any = None, opacity: float = 0.5, fps: float = 24.0
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        functional = importlib.import_module("torch.nn.functional")
        av = importlib.import_module("av")
        management = importlib.import_module("dinkster_inference.model_management")
        tracker = importlib.import_module("dinkster_inference.ldm.sam3.tracker")
        track_data = materialize_latent(track_data)
        images = materialize_image(images) if images is not None else None
        packed: Any = track_data["packed_masks"]
        height, width = track_data["orig_size"]
        if images is not None:
            height, width = images.shape[1:3]
        if packed is None:
            count, objects = track_data["n_frames"], 0
        else:
            count, objects = packed.shape[:2]
        device = management.get_torch_device()
        encoded = io.BytesIO()
        with av.open(encoded, mode="w", format="mp4") as output:
            stream = output.add_stream("h264", rate=Fraction(round(fps * 1000), 1000))
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            frame_cpu = torch.empty(height, width, 3, dtype=torch.uint8)
            frame_np = frame_cpu.numpy()
            colors: Any = None
            grid_y: Any = None
            grid_x: Any = None
            if objects > 0:
                colors = torch.tensor(
                    [cls.COLORS[index % len(cls.COLORS)] for index in range(objects)],
                    device=device,
                    dtype=torch.float32,
                )
                grid_y = torch.arange(height, device=device).view(1, height, 1)
                grid_x = torch.arange(width, device=device).view(1, 1, width)
            for index in range(count):
                frame = (
                    images[index].clone()
                    if images is not None and index < images.shape[0]
                    else torch.zeros(height, width, 3)
                )
                if objects > 0:
                    binary = tracker.unpack_masks(cast(Any, packed)[index : index + 1].to(device))
                    masks = functional.interpolate(
                        binary.float(), size=(height, width), mode="nearest"
                    )[0]
                    frame_device = frame.to(device)
                    bool_masks = masks > 0.5
                    any_mask = bool_masks.any(dim=0)
                    if any_mask.any():
                        object_map = bool_masks.to(torch.uint8).argmax(dim=0)
                        frame_device = torch.where(
                            any_mask.unsqueeze(-1),
                            frame_device * (1 - opacity) + colors[object_map] * opacity,
                            frame_device,
                        )
                    area = bool_masks.sum(dim=(-1, -2)).clamp_(min=1)
                    cy = (bool_masks * grid_y).sum(dim=(-1, -2)) // area
                    cx = (bool_masks * grid_x).sum(dim=(-1, -2)) // area
                    has = area > 1
                    scores = track_data.get("scores", [])
                    label_scale = max(3, height // 240)
                    size_caps = (area.float().sqrt() / 15).clamp_(min=1).long().tolist()
                    for obj in range(objects):
                        if has[obj]:
                            color = cls.COLORS[obj % len(cls.COLORS)]
                            scale = min(label_scale, size_caps[obj])
                            score_scale = max(1, scale * 2 // 3)
                            cls._draw_number_gpu(
                                frame_device, obj, int(cx[obj]), int(cy[obj]), color, scale=scale
                            )
                            if obj < len(scores) and scores[obj] < 1.0:
                                cls._draw_number_gpu(
                                    frame_device,
                                    int(scores[obj] * 100),
                                    int(cx[obj]),
                                    int(cy[obj]) + 5 * scale + 3,
                                    color,
                                    scale=score_scale,
                                )
                    frame_cpu.copy_(frame_device.clamp_(0, 1).mul_(255).byte())
                else:
                    frame_cpu.copy_(frame.clamp_(0, 1).mul_(255).byte())
                video_frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
                output.mux(stream.encode(video_frame.reformat(format="yuv420p")))
            output.mux(stream.encode(None))
        publish_preview_artifact(
            encoded.getvalue(),
            name=f"sam3_track_preview_{uuid.uuid4().hex[:8]}.mp4",
            media_type="video/mp4",
        )
        return cls.outputs()
