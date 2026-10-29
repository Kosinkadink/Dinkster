"""Native SCAIL-2 colored SAM3 mask rendering."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_STRING,
    IMAGE_TYPE,
    MASK_TYPE,
    SAM3_TRACK_DATA_TYPE,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    StringWidget,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
    materialize_latent,
)

_CATEGORY = "model/conditioning/wan/scail"
_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_SAM3_TRACK_DATA_TYPE = cast(str, SAM3_TRACK_DATA_TYPE)
_REFERENCE_MASKS = TypeExpr.union(_SAM3_TRACK_DATA_TYPE, MASK_TYPE)
_SAM3_TRACK_DATA = TypeExpr.concrete(_SAM3_TRACK_DATA_TYPE)
_STRING = TypeExpr.concrete(CORE_STRING)
_PALETTE = (
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 0.0),
)
_SORT_OPTIONS = ("none", "left_to_right", "area")


def _runtime() -> tuple[Any, Any, Any]:
    initialize_inference_process()
    return (
        importlib.import_module("torch"),
        importlib.import_module("torch.nn.functional"),
        importlib.import_module("dinkster_inference.model_management"),
    )


def _unpack(track_data: Mapping[str, Any]) -> Any:
    packed = track_data["packed_masks"]
    if packed is None or packed.shape[1] == 0:
        return None
    tracker = importlib.import_module("dinkster_inference.ldm.sam3.tracker")
    return tracker.unpack_masks(packed)


def _first_appearance_cx_area(
    masks_bool: Any, torch: Any
) -> tuple[list[int], list[float], list[float]]:
    masks = masks_bool.float()
    frames, height, width = masks.shape[0], masks.shape[-2], masks.shape[-1]
    grid_x = torch.arange(width, device=masks.device, dtype=masks.dtype).view(1, 1, 1, width)
    area_per_frame = masks.sum(dim=(-1, -2))
    center_x_per_frame = (masks * grid_x).sum(dim=(-1, -2)) / area_per_frame.clamp(min=1)
    present = area_per_frame > 0
    frame_indices = torch.arange(frames, device=masks.device).unsqueeze(1)
    first_frame = torch.where(present, frame_indices, frames).amin(dim=0)
    selected = first_frame.clamp(max=frames - 1).unsqueeze(0)
    center_x = center_x_per_frame.gather(0, selected).squeeze(0)
    area = area_per_frame.gather(0, selected).squeeze(0)
    return first_frame.tolist(), (center_x / width).tolist(), (area / (height * width)).tolist()


def _subset_track_data(track_data: Mapping[str, Any], object_indices: list[int]) -> dict[str, Any]:
    result = dict(track_data)
    packed = track_data["packed_masks"]
    if packed is None or not object_indices:
        result["packed_masks"] = None
        if "scores" in result:
            result["scores"] = []
        return result
    result["packed_masks"] = packed[:, object_indices].contiguous()
    scores = track_data.get("scores")
    if scores is not None:
        result["scores"] = [scores[index] for index in object_indices if index < len(scores)]
    return result


def _render_colored_masks(
    track_data: Mapping[str, Any], background: str, torch: Any, functional: Any, management: Any
) -> Any:
    packed = track_data["packed_masks"]
    height, width = track_data["orig_size"]
    device = management.intermediate_device()
    dtype = management.intermediate_dtype()
    background_rgb = (1.0, 1.0, 1.0) if background.startswith("white") else (0.0, 0.0, 0.0)
    if packed is None or packed.shape[1] == 0:
        frames = track_data.get("n_frames", 1) if packed is None else packed.shape[0]
        output = torch.empty(frames, height, width, 3, device=device, dtype=dtype)
        output[..., 0], output[..., 1], output[..., 2] = background_rgb
        return output
    frames, objects = packed.shape[:2]
    colors = torch.tensor(
        [_PALETTE[index % len(_PALETTE)] for index in range(objects)], device=device, dtype=dtype
    )
    tracker = importlib.import_module("dinkster_inference.ldm.sam3.tracker")
    masks = tracker.unpack_masks(packed.to(device)).float()
    mask_height, mask_width = masks.shape[-2:]
    masks = (
        functional.interpolate(
            masks.view(frames * objects, 1, mask_height, mask_width),
            size=(height, width),
            mode="nearest",
        ).view(frames, objects, height, width)
        > 0.5
    )
    any_mask = masks.any(dim=1)
    overlay = colors[masks.to(torch.uint8).argmax(dim=1)]
    background_tensor = torch.tensor(background_rgb, device=device, dtype=overlay.dtype).view(
        1, 1, 1, 3
    )
    return torch.where(any_mask.unsqueeze(-1), overlay, background_tensor.expand_as(overlay))


def _render_mask_as_identity(mask: Any, background: str, torch: Any, management: Any) -> Any:
    device = management.intermediate_device()
    dtype = management.intermediate_dtype()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    mask = mask.to(device=device, dtype=dtype)
    batch, height, width = mask.shape
    background_rgb = (1.0, 1.0, 1.0) if background.startswith("white") else (0.0, 0.0, 0.0)
    color = torch.tensor(_PALETTE[0], device=device, dtype=dtype).view(1, 1, 1, 3)
    background_tensor = torch.tensor(background_rgb, device=device, dtype=dtype).view(1, 1, 1, 3)
    return torch.where(
        (mask > 0.5).unsqueeze(-1),
        color.expand(batch, height, width, 3),
        background_tensor.expand(batch, height, width, 3),
    )


class SCAIL2ColoredMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SCAIL2ColoredMask",
            display_name="Create SCAIL-2 Colored Mask",
            category=_CATEGORY,
            inputs=(
                InputSpec("driving_track_data", _SAM3_TRACK_DATA),
                InputSpec(
                    "ref_track_data",
                    _REFERENCE_MASKS,
                    required=False,
                    display_name="reference_masks",
                    doc=(
                        "SAM3 track of the reference image(s), or a plain MASK of the reference "
                        "subject rendered as a single identity."
                    ),
                ),
                InputSpec(
                    "object_indices",
                    _STRING,
                    default="",
                    widget=StringWidget(),
                ),
                InputSpec(
                    "sort_by",
                    _COMBO,
                    default="left_to_right",
                    widget=ComboWidget(options=_SORT_OPTIONS),
                ),
                InputSpec("replacement_mode", _BOOLEAN, default=False),
            ),
            outputs=(
                OutputSpec("pose_video_mask", _IMAGE),
                OutputSpec("reference_image_mask", _IMAGE),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        driving_track_data: object,
        object_indices: str,
        sort_by: str,
        replacement_mode: bool,
        ref_track_data: object = None,
    ) -> Mapping[str, object]:
        torch, functional, management = _runtime()

        def prepare(track_data: object) -> dict[str, Any]:
            data = materialize_latent(track_data)
            if not isinstance(data, Mapping):
                raise TypeError("track data must be a mapping")
            prepared: dict[str, Any] = dict(cast(Mapping[str, Any], data))
            masks = _unpack(prepared)
            if sort_by != "none" and masks is not None:
                first_frame, center_x, area = _first_appearance_cx_area(masks, torch)
                if sort_by == "left_to_right":
                    order = sorted(
                        range(len(center_x)),
                        key=lambda index: (first_frame[index], center_x[index]),
                    )
                else:
                    order = sorted(
                        range(len(area)), key=lambda index: (first_frame[index], -area[index])
                    )
                prepared = _subset_track_data(prepared, order)
            if object_indices.strip():
                indices = [
                    int(index.strip())
                    for index in object_indices.split(",")
                    if index.strip().isdigit()
                ]
                packed = prepared.get("packed_masks")
                objects = packed.shape[1] if packed is not None else 0
                prepared = _subset_track_data(
                    prepared, [index for index in indices if 0 <= index < objects]
                )
            return prepared

        driving = prepare(driving_track_data)
        pose_background = "white" if replacement_mode else "black"
        pose_video_mask = _render_colored_masks(
            driving, pose_background, torch, functional, management
        )
        reference_background = "black" if replacement_mode else "white"
        if ref_track_data is not None:
            if not isinstance(ref_track_data, Mapping):
                reference_image_mask = _render_mask_as_identity(
                    materialize_image(ref_track_data), reference_background, torch, management
                )
            else:
                reference_image_mask = _render_colored_masks(
                    prepare(cast(object, ref_track_data)),
                    reference_background,
                    torch,
                    functional,
                    management,
                )
        else:
            height, width = driving["orig_size"]
            fill_value = 1.0 if reference_background == "white" else 0.0
            reference_image_mask = torch.full(
                (1, height, width, 3),
                fill_value,
                device=management.intermediate_device(),
                dtype=management.intermediate_dtype(),
            )
        return cls.outputs(
            pose_video_mask=pose_video_mask,
            reference_image_mask=reference_image_mask,
        )
