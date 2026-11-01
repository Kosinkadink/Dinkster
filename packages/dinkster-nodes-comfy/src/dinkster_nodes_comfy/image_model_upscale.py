"""Native tiled image upscaling with a resident single-image model."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    IMAGE_TYPE,
    UPSCALE_MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_UPSCALE_MODEL = TypeExpr.concrete(UPSCALE_MODEL_TYPE)


class ImageUpscaleWithModel(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ImageUpscaleWithModel",
            display_name="Upscale Image (using Model)",
            category="image/upscaling",
            search_terms=(
                "upscale",
                "upscaler",
                "upsc",
                "enlarge image",
                "super resolution",
                "hires",
                "superres",
                "increase resolution",
            ),
            inputs=(InputSpec("upscale_model", _UPSCALE_MODEL), InputSpec("image", _IMAGE)),
            outputs=(OutputSpec("IMAGE", _IMAGE),),
        )

    @classmethod
    def execute(cls, *, upscale_model: Any, image: Any) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        utils = importlib.import_module("dinkster_inference.utils")
        image = materialize_image(image)
        device = upscale_model.patcher.load_device
        alpha = None
        if image.shape[-1] == 4:
            alpha = image[..., 3:4]
            image = image[..., :3]
        memory_required = (
            (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0
        )
        memory_required += image.nelement() * image.element_size()
        management.load_models_gpu(
            [upscale_model.patcher], memory_required=memory_required, force_full_load=True
        )
        in_img = image.movedim(-1, -3).to(device)
        tile, overlap = 512, 32
        output_device = management.intermediate_device()

        def upscale(value: Any) -> Any:
            return upscale_model(value.float())

        while True:
            try:
                steps = in_img.shape[0] * utils.get_tiled_scale_steps(
                    in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
                )
                scaled = utils.tiled_scale(
                    in_img,
                    upscale,
                    tile_x=tile,
                    tile_y=tile,
                    overlap=overlap,
                    upscale_amount=upscale_model.scale,
                    pbar=utils.ProgressBar(steps),
                    output_device=output_device,
                )
                break
            except Exception as error:
                management.raise_non_oom(error)
                tile //= 2
                if tile < 128:
                    raise
        scaled = torch.clamp(scaled.movedim(-3, -1), min=0, max=1.0).to(
            management.intermediate_dtype()
        )
        if alpha is not None:
            alpha = utils.common_upscale(
                alpha.movedim(-1, -3).to(scaled),
                scaled.shape[2],
                scaled.shape[1],
                "bilinear",
                "disabled",
            )
            scaled = torch.cat((scaled, alpha.movedim(-3, -1)), dim=-1)
        return cls.outputs(IMAGE=scaled)
