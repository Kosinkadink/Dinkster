"""Native SAM3 text, box and point detection with optional mask refinement."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    BOUNDING_BOX_TYPE,
    CONDITIONING_TYPE,
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    IMAGE_TYPE,
    MASK_TYPE,
    MODEL_TYPE,
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

from .sam3 import extract_text_prompts


def _refine_mask(
    model: Any,
    image: Any,
    coarse: Any,
    box: Any,
    height: int,
    width: int,
    device: Any,
    dtype: Any,
    iterations: int,
) -> Any:
    torch = importlib.import_module("torch")
    functional = importlib.import_module("torch.nn.functional")
    utils = importlib.import_module("dinkster_inference.utils")

    def fallback() -> Any:
        return (
            functional.interpolate(
                coarse.unsqueeze(0).unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0]
            > 0
        ).float()

    if iterations <= 0:
        return fallback()
    x1, y1, x2, y2 = box.tolist()
    box_width, box_height = x2 - x1, y2 - y1
    cx1, cy1 = max(0, int(x1 - box_width * 0.1)), max(0, int(y1 - box_height * 0.1))
    cx2 = min(width, int(x2 + box_width * 0.1))
    cy2 = min(height, int(y2 + box_height * 0.1))
    if cx2 <= cx1 or cy2 <= cy1:
        return fallback()
    crop = image[cy1:cy2, cx1:cx2, :3]
    crop = utils.common_upscale(
        crop.unsqueeze(0).movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled"
    ).to(device=device, dtype=dtype)
    mask_height, mask_width = coarse.shape[-2:]
    mx1, my1 = int(cx1 / width * mask_width), int(cy1 / height * mask_height)
    mx2, my2 = int(cx2 / width * mask_width), int(cy2 / height * mask_height)
    if mx2 <= mx1 or my2 <= my1:
        return fallback()
    logits = coarse[..., my1:my2, mx1:mx2].unsqueeze(0).unsqueeze(0)
    for _ in range(iterations):
        refined_input = functional.interpolate(
            logits, size=(1008, 1008), mode="bilinear", align_corners=False
        )
        logits = model.forward_segment(crop, mask_inputs=refined_input)
    refined = functional.interpolate(
        logits, size=(cy2 - cy1, cx2 - cx1), mode="bilinear", align_corners=False
    )
    full = torch.zeros(1, 1, height, width, device=device, dtype=dtype)
    full[:, :, cy1:cy2, cx1:cx2] = refined
    coarse_full = functional.interpolate(
        coarse.unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return ((full[0] > 0) | (coarse_full[0] > 0)).float()


class SAM3_Detect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SAM3_Detect",
            display_name="SAM3 Detect",
            category="image/detection",
            search_terms=(
                "sam3",
                "segment anything",
                "open vocabulary",
                "text detection",
                "segment",
            ),
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE), display_name="model"),
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE), display_name="image"),
                InputSpec(
                    "conditioning",
                    TypeExpr.concrete(CONDITIONING_TYPE),
                    display_name="conditioning",
                    required=False,
                    doc="Text conditioning from CLIPTextEncode",
                ),
                InputSpec(
                    "bboxes",
                    TypeExpr.concrete(BOUNDING_BOX_TYPE),
                    display_name="bboxes",
                    required=False,
                    force_input=True,
                    default={"x": 0, "y": 0, "width": 512, "height": 512},
                    doc="Bounding boxes to segment within",
                ),
                *(
                    InputSpec(
                        name,
                        TypeExpr.concrete(CORE_STRING),
                        display_name=name,
                        required=False,
                        force_input=True,
                        widget=StringWidget(multiline=False),
                        doc=(
                            f'{label} point prompts as JSON [{{"x": int, "y": int}}, ...] '
                            "(pixel coords)"
                        ),
                    )
                    for name, label in (
                        ("positive_coords", "Positive"),
                        ("negative_coords", "Negative"),
                    )
                ),
                InputSpec(
                    "threshold",
                    TypeExpr.concrete(CORE_FLOAT),
                    display_name="threshold",
                    default=0.5,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "refine_iterations",
                    TypeExpr.concrete(CORE_INT),
                    display_name="refine_iterations",
                    default=2,
                    widget=NumberWidget(min=0, max=5),
                    doc="SAM decoder refinement passes (0=use raw detector masks)",
                ),
                InputSpec(
                    "individual_masks",
                    TypeExpr.concrete(CORE_BOOLEAN),
                    display_name="individual_masks",
                    default=False,
                    doc="Output per-object masks instead of union",
                ),
            ),
            outputs=(
                OutputSpec("masks", TypeExpr.concrete(MASK_TYPE)),
                OutputSpec("bboxes", TypeExpr.concrete(BOUNDING_BOX_TYPE)),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        image: Any,
        conditioning: Any = None,
        bboxes: Any = None,
        positive_coords: str | None = None,
        negative_coords: str | None = None,
        threshold: float = 0.5,
        refine_iterations: int = 2,
        individual_masks: bool = False,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        functional = importlib.import_module("torch.nn.functional")
        management = importlib.import_module("dinkster_inference.model_management")
        utils = importlib.import_module("dinkster_inference.utils")
        image = materialize_image(image)
        batch, height, width, _ = image.shape
        image_in = utils.common_upscale(
            image[..., :3].movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled"
        )

        def box_tensor(boxes: Any) -> Any:
            return torch.tensor(
                [
                    [
                        [
                            (box["x"] + box["width"] / 2) / width,
                            (box["y"] + box["height"] / 2) / height,
                            box["width"] / width,
                            box["height"] / height,
                        ]
                        for box in boxes
                    ]
                ],
                dtype=torch.float32,
            )

        per_frame_boxes: list[Any] | None = None
        if isinstance(bboxes, dict):
            per_frame_boxes = [box_tensor([bboxes])] * batch
        elif isinstance(bboxes, list) and bboxes:
            if isinstance(bboxes[0], list):
                per_frame_boxes = [
                    box_tensor(boxes) if boxes else None for boxes in cast("list[Any]", bboxes)
                ]
                while len(per_frame_boxes) < batch:
                    per_frame_boxes.append(per_frame_boxes[-1] if per_frame_boxes else None)
            else:
                per_frame_boxes = [box_tensor(bboxes)] * batch
        positive: Any = json.loads(positive_coords) if positive_coords else []
        negative: Any = json.loads(negative_coords) if negative_coords else []
        management.load_model_gpu(model)
        device = management.get_torch_device()
        dtype = model.model.get_dtype()
        sam = model.model.diffusion_model
        points = None
        if positive or negative:
            coordinates = [
                [p["x"] / width * 1008, p["y"] / height * 1008] for p in positive + negative
            ]
            points = {
                "point_coords": torch.tensor([coordinates], dtype=dtype, device=device),
                "point_labels": torch.tensor(
                    [[1] * len(positive) + [0] * len(negative)], dtype=torch.int32, device=device
                ),
            }
        prompts = (
            extract_text_prompts(materialize_latent(conditioning), device, dtype)
            if conditioning is not None and len(conditioning) > 0
            else []
        )
        all_boxes: list[Any] = []
        all_masks: list[Any] = []
        progress = utils.ProgressBar(batch)
        for index in range(batch):
            frame = image_in[index : index + 1].to(device=device, dtype=dtype)
            boxes = None
            if per_frame_boxes is not None and per_frame_boxes[index] is not None:
                boxes = per_frame_boxes[index].to(device=device, dtype=dtype)
            frame_boxes: list[Any] = []
            frame_masks: list[Any] = []
            if points is not None:
                logits = sam.forward_segment(frame, point_inputs=points)
                for _ in range(max(0, refine_iterations - 1)):
                    logits = sam.forward_segment(frame, mask_inputs=logits)
                mask = functional.interpolate(
                    logits, size=(height, width), mode="bilinear", align_corners=False
                )
                frame_masks.append((mask[0] > 0).float())
            if boxes is not None and not prompts:
                for box in boxes[0]:
                    cx, cy, bw, bh = box.tolist()
                    corners = torch.tensor(
                        [
                            [
                                [(cx - bw / 2) * 1008, (cy - bh / 2) * 1008],
                                [(cx + bw / 2) * 1008, (cy + bh / 2) * 1008],
                            ]
                        ],
                        device=device,
                        dtype=dtype,
                    )
                    logits = sam.forward_segment(frame, box_inputs=corners)
                    for _ in range(max(0, refine_iterations - 1)):
                        logits = sam.forward_segment(frame, mask_inputs=logits)
                    mask = functional.interpolate(
                        logits, size=(height, width), mode="bilinear", align_corners=False
                    )
                    frame_masks.append((mask[0] > 0).float())
            for embedding, text_mask, maximum in prompts:
                result = sam(
                    frame,
                    text_embeddings=embedding,
                    text_mask=text_mask,
                    boxes=boxes,
                    threshold=threshold,
                    orig_size=(height, width),
                )
                probabilities = result["scores"][0].sigmoid()
                keep = probabilities > threshold
                kept_boxes = result["boxes"][0][keep].cpu()
                kept_scores = probabilities[keep].cpu()
                kept_masks = result["masks"][0][keep]
                order = kept_scores.argsort(descending=True)[:maximum]
                kept_boxes, kept_scores, kept_masks = (
                    kept_boxes[order],
                    kept_scores[order],
                    kept_masks[order],
                )
                for box, score in zip(kept_boxes, kept_scores, strict=False):
                    frame_boxes.append(
                        {
                            "x": float(box[0]),
                            "y": float(box[1]),
                            "width": float(box[2] - box[0]),
                            "height": float(box[3] - box[1]),
                            "score": float(score),
                        }
                    )
                for mask, box in zip(kept_masks, kept_boxes, strict=False):
                    frame_masks.append(
                        _refine_mask(
                            sam,
                            image[index],
                            mask,
                            box,
                            height,
                            width,
                            device,
                            dtype,
                            refine_iterations,
                        )
                    )
            all_boxes.append(frame_boxes)
            if frame_masks:
                combined = torch.cat(frame_masks, dim=0)
                all_masks.append(
                    combined if individual_masks else (combined > 0).any(dim=0).float()
                )
            else:
                shape = (0, height, width) if individual_masks else (height, width)
                all_masks.append(torch.zeros(*shape, device=management.intermediate_device()))
            progress.update(1)
        all_masks = [mask.to(management.intermediate_device()) for mask in all_masks]
        output = torch.cat(all_masks, dim=0) if individual_masks else torch.stack(all_masks)
        return cls.outputs(masks=output, bboxes=all_boxes)
