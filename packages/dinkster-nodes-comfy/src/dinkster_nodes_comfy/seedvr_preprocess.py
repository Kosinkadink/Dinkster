"""Native SeedVR2 video image padding."""

from __future__ import annotations

import importlib
from collections.abc import Mapping

from dinkster_api.v1 import (
    IMAGE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)


class SeedVR2Preprocess(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        image = TypeExpr.concrete(IMAGE_TYPE)
        return NodeSchema(
            node_type="SeedVR2Preprocess",
            display_name="Pre-Process SeedVR2 Input",
            category="image/pre-processors",
            description=(
                "Pad a resized image for SeedVR2 model. Alpha channel is dropped. "
                "The node Post-Process SeedVR2 Output re-applies it from the "
                "original resized image."
            ),
            search_terms=("seedvr2", "upscale", "video upscale", "pad", "preprocess"),
            inputs=(InputSpec("resized_images", image, doc="The resized image to process."),),
            outputs=(OutputSpec("images", image, doc="The padded image for VAE encoding."),),
        )

    @classmethod
    def execute(cls, *, resized_images: object) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        images = materialize_image(resized_images)
        if images.ndim == 4:
            shorter_edge = min(images.shape[1], images.shape[2])
        elif images.ndim == 5:
            shorter_edge = min(images.shape[2], images.shape[3])
        else:
            raise ValueError(
                "SeedVR2Preprocess: expected 4-D or 5-D IMAGE tensor, "
                f"got shape {tuple(images.shape)}"
            )
        if shorter_edge < 2:
            raise ValueError(
                "SeedVR2Preprocess: input shorter edge must be at least 2 pixels; "
                f"got {shorter_edge}."
            )
        if images.shape[-1] > 3:
            images = images[..., :3]
        if images.ndim == 4:
            images = images.unsqueeze(0)
        images = images.permute(0, 1, 4, 2, 3)
        batch, frames, channels, height, width = images.shape
        images = images.reshape(batch * frames, channels, height, width).clamp(0.0, 1.0)
        images = torch.nn.functional.pad(images, (0, (-width) % 16, 0, (-height) % 16))
        images = images.reshape(batch, frames, channels, images.shape[-2], images.shape[-1])
        if frames < 1:
            raise ValueError("SeedVR2Preprocess expected at least one frame.")
        if frames > 1 and (frames - 1) % 4:
            padding = images[:, -1:].repeat(1, 4 - ((frames - 1) % 4), 1, 1, 1)
            images = torch.cat((images, padding), dim=1)
        return cls.outputs(images=images.permute(0, 1, 3, 4, 2).contiguous())
