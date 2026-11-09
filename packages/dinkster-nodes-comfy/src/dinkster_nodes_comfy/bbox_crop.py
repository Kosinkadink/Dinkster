"""Per-frame bounding-box unions cropped to a common image size."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    BOUNDING_BOX_TYPE,
    CORE_COMBO,
    CORE_INT,
    IMAGE_TYPE,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)


class CropByBBoxes(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="CropByBBoxes",
            display_name="Crop By Bounding Boxes",
            category="image/transform",
            search_terms=("crop", "face crop", "bbox crop", "pose", "bounding box"),
            description=(
                "Crop and resize regions from the input image batch "
                "based on provided bounding boxes."
            ),
            inputs=(
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec(
                    "bboxes",
                    TypeExpr.concrete(BOUNDING_BOX_TYPE),
                    force_input=True,
                    default={"x": 0, "y": 0, "width": 512, "height": 512},
                ),
                InputSpec(
                    "output_width",
                    TypeExpr.concrete(CORE_INT),
                    default=512,
                    widget=NumberWidget(min=64, max=4096, step=8),
                    doc="Width each crop is resized to.",
                ),
                InputSpec(
                    "output_height",
                    TypeExpr.concrete(CORE_INT),
                    default=512,
                    widget=NumberWidget(min=64, max=4096, step=8),
                    doc="Height each crop is resized to.",
                ),
                InputSpec(
                    "padding",
                    TypeExpr.concrete(CORE_INT),
                    default=0,
                    widget=NumberWidget(min=0, max=1024, step=1),
                    doc="Extra padding in pixels added on each side of the bbox before cropping.",
                ),
                InputSpec(
                    "keep_aspect",
                    TypeExpr.concrete(CORE_COMBO),
                    default="stretch",
                    widget=ComboWidget(options=(ComboOption("stretch"), ComboOption("pad"))),
                    doc=(
                        "Whether to stretch the crop to fit the output size, or pad with black "
                        "pixels to preserve aspect ratio."
                    ),
                ),
            ),
            outputs=(
                OutputSpec(
                    "IMAGE",
                    TypeExpr.concrete(IMAGE_TYPE),
                    doc="All crops stacked into a single image batch.",
                ),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: Any,
        bboxes: Any,
        output_width: int,
        output_height: int,
        padding: int,
        keep_aspect: str = "stretch",
    ) -> Mapping[str, object]:
        image = materialize_image(image)
        if not isinstance(bboxes, list):
            bboxes = [[bboxes]]
        elif not bboxes:
            return cls.outputs(IMAGE=image)
        elif isinstance(bboxes[0], Mapping):
            bboxes = [bboxes]
        initialize_inference_process()
        torch = importlib.import_module("torch")
        utils = importlib.import_module("dinkster_inference.utils")
        total_frames, img_h, img_w, num_ch = image.shape
        frames = cast(list[list[dict[str, Any]]], bboxes)
        crops: list[Any] = []
        for frame_idx in range(total_frames):
            frame_bboxes = frames[min(frame_idx, len(frames) - 1)]
            if not frame_bboxes:
                continue
            frame_chw = image[frame_idx].permute(2, 0, 1).unsqueeze(0)
            x1 = min(int(b["x"]) for b in frame_bboxes)
            y1 = min(int(b["y"]) for b in frame_bboxes)
            x2 = max(int(b["x"] + b["width"]) for b in frame_bboxes)
            y2 = max(int(b["y"] + b["height"]) for b in frame_bboxes)
            if padding > 0:
                x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
                x2, y2 = min(img_w, x2 + padding), min(img_h, y2 + padding)
            x1, x2 = max(0, x1), min(img_w, x2)
            y1, y2 = max(0, y1), min(img_h, y2)
            if x2 <= x1 or y2 <= y1:
                fallback_size = int(min(img_h, img_w) * 0.3)
                x1 = max(0, (img_w - fallback_size) // 2)
                y1 = max(0, int(img_h * 0.1))
                x2, y2 = min(img_w, x1 + fallback_size), min(img_h, y1 + fallback_size)
                if x2 <= x1 or y2 <= y1:
                    crops.append(
                        torch.zeros(
                            1,
                            num_ch,
                            output_height,
                            output_width,
                            dtype=image.dtype,
                            device=image.device,
                        )
                    )
                    continue
            crop_chw = frame_chw[:, :, y1:y2, x1:x2]
            if keep_aspect == "pad":
                crop_h, crop_w = y2 - y1, x2 - x1
                scale = min(output_width / crop_w, output_height / crop_h)
                scaled_w, scaled_h = int(round(crop_w * scale)), int(round(crop_h * scale))
                scaled = utils.common_upscale(
                    crop_chw, scaled_w, scaled_h, upscale_method="area", crop="disabled"
                )
                pad_left, pad_top = (output_width - scaled_w) // 2, (output_height - scaled_h) // 2
                resized = torch.zeros(
                    1, num_ch, output_height, output_width, dtype=image.dtype, device=image.device
                )
                resized[:, :, pad_top : pad_top + scaled_h, pad_left : pad_left + scaled_w] = scaled
            else:
                resized = utils.common_upscale(
                    crop_chw, output_width, output_height, upscale_method="area", crop="disabled"
                )
            crops.append(resized)
        if not crops:
            return cls.outputs(IMAGE=image)
        return cls.outputs(IMAGE=torch.cat(crops, dim=0).permute(0, 2, 3, 1))
