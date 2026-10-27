"""Native SAM3 video tracking and packed-track mask selection."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    IMAGE_TYPE,
    MASK_TYPE,
    MODEL_TYPE,
    SAM3_TRACK_DATA_TYPE,
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


def _extract_text_prompts(conditioning: Any, device: Any, dtype: Any) -> list[Any]:
    torch = importlib.import_module("torch")
    metadata = conditioning[0][1]
    multiple = metadata.get("sam3_multi_cond")
    prompts: list[Any] = []
    if multiple is not None:
        for entry in multiple:
            embedding = entry["cond"].to(device=device, dtype=dtype)
            mask = (
                entry["attention_mask"].to(device) if entry["attention_mask"] is not None else None
            )
            if mask is None:
                mask = torch.ones(
                    embedding.shape[0], embedding.shape[1], dtype=torch.int64, device=device
                )
            prompts.append((embedding, mask, entry.get("max_detections", 1)))
    else:
        embedding = conditioning[0][0].to(device=device, dtype=dtype)
        mask = metadata.get("attention_mask")
        if mask is not None:
            mask = mask.to(device)
        else:
            mask = torch.ones(
                embedding.shape[0], embedding.shape[1], dtype=torch.int64, device=device
            )
        prompts.append((embedding, mask, 1))
    return prompts


class SAM3_VideoTrack(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SAM3_VideoTrack",
            display_name="Run SAM3 Video Track",
            category="image/detection",
            search_terms=("sam3", "video", "track", "propagate"),
            inputs=(
                InputSpec(
                    "images",
                    TypeExpr.concrete(IMAGE_TYPE),
                    display_name="images",
                    doc="Video frames as batched images",
                ),
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE), display_name="model"),
                InputSpec(
                    "initial_mask",
                    TypeExpr.concrete(MASK_TYPE),
                    display_name="initial_mask",
                    required=False,
                    doc="Mask(s) for the first frame to track (one per object)",
                ),
                InputSpec(
                    "conditioning",
                    TypeExpr.concrete(CONDITIONING_TYPE),
                    display_name="conditioning",
                    required=False,
                    doc="Text conditioning for detecting new objects during tracking",
                ),
                InputSpec(
                    "detection_threshold",
                    TypeExpr.concrete(CORE_FLOAT),
                    display_name="detection_threshold",
                    default=0.5,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                    doc="Score threshold for text-prompted detection.",
                ),
                InputSpec(
                    "max_objects",
                    TypeExpr.concrete(CORE_INT),
                    display_name="max_objects",
                    default=4,
                    widget=NumberWidget(min=0, max=64),
                    doc=(
                        "Max tracked objects. Initial masks count toward this limit. "
                        "0 uses the internal cap of 64."
                    ),
                ),
                InputSpec(
                    "detect_interval",
                    TypeExpr.concrete(CORE_INT),
                    display_name="detect_interval",
                    default=1,
                    widget=NumberWidget(min=1),
                    doc="Run detection every N frames (1=every frame). Higher values save compute.",
                ),
            ),
            outputs=(
                OutputSpec(
                    "track_data", TypeExpr.concrete(SAM3_TRACK_DATA_TYPE), display_name="track_data"
                ),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        images: Any,
        model: Any,
        initial_mask: Any = None,
        conditioning: Any = None,
        detection_threshold: float = 0.5,
        max_objects: int = 0,
        detect_interval: int = 1,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        management = importlib.import_module("dinkster_inference.model_management")
        utils = importlib.import_module("dinkster_inference.utils")
        images = materialize_image(images)
        count, height, width, _ = images.shape
        management.load_model_gpu(model)
        device = management.get_torch_device()
        dtype = model.model.get_dtype()
        tracker = model.model.diffusion_model
        frames = images[..., :3].movedim(-1, 1)
        masks = None
        if initial_mask is not None:
            masks = materialize_image(initial_mask).unsqueeze(1).to(device=device, dtype=dtype)
        progress = utils.ProgressBar(count)
        text_prompts = None
        if conditioning is not None and len(conditioning) > 0:
            conditioning = materialize_latent(conditioning)
            text_prompts = [
                (embedding, mask)
                for embedding, mask, _ in _extract_text_prompts(conditioning, device, dtype)
            ]
        elif initial_mask is None:
            raise ValueError("Either initial_mask or conditioning must be provided")
        result = tracker.forward_video(
            images=frames,
            initial_masks=masks,
            pbar=progress,
            text_prompts=text_prompts,
            new_det_thresh=detection_threshold,
            max_objects=max_objects,
            detect_interval=detect_interval,
            target_device=device,
            target_dtype=dtype,
        )
        result["orig_size"] = (height, width)
        return cls.outputs(track_data=result)


class SAM3_TrackToMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SAM3_TrackToMask",
            display_name="SAM3 Track to Mask",
            category="image/detection",
            inputs=(
                InputSpec(
                    "track_data", TypeExpr.concrete(SAM3_TRACK_DATA_TYPE), display_name="track_data"
                ),
                InputSpec(
                    "object_indices",
                    TypeExpr.concrete(CORE_STRING),
                    display_name="object_indices",
                    default="",
                    widget=StringWidget(),
                    doc=(
                        "Comma-separated object indices to include (e.g. '0,2,3'). "
                        "Empty = all objects."
                    ),
                ),
            ),
            outputs=(OutputSpec("masks", TypeExpr.concrete(MASK_TYPE), display_name="masks"),),
        )

    @classmethod
    def execute(cls, *, track_data: Any, object_indices: str = "") -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        functional = importlib.import_module("torch.nn.functional")
        tracker = importlib.import_module("dinkster_inference.ldm.sam3.tracker")
        management = importlib.import_module("dinkster_inference.model_management")
        track_data = materialize_latent(track_data)
        packed = track_data["packed_masks"]
        height, width = track_data["orig_size"]
        if packed is None:
            return cls.outputs(
                masks=torch.zeros(
                    track_data["n_frames"], height, width, device=management.intermediate_device()
                )
            )
        count, objects = packed.shape[:2]
        if object_indices.strip():
            indices = [
                int(index.strip()) for index in object_indices.split(",") if index.strip().isdigit()
            ]
            indices = [index for index in indices if 0 <= index < objects]
        else:
            indices = list(range(objects))
        if not indices:
            return cls.outputs(
                masks=torch.zeros(count, height, width, device=management.intermediate_device())
            )
        union_packed = packed[:, indices[0]].clone()
        for index in indices[1:]:
            union_packed |= packed[:, index]
        union = tracker.unpack_masks(union_packed).unsqueeze(1).float()
        return cls.outputs(
            masks=functional.interpolate(
                union, size=(height, width), mode="bilinear", align_corners=False
            )[:, 0]
        )


SAM3_TRACKING_NODES = (SAM3_VideoTrack, SAM3_TrackToMask)
