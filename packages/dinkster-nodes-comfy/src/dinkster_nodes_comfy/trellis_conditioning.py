"""Trellis image conditioning through the pinned DINO vision runtime."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CLIP_VISION_TYPE,
    CONDITIONING_TYPE,
    IMAGE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)


class Trellis2Conditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        conditioning = TypeExpr.concrete(CONDITIONING_TYPE)
        return NodeSchema(
            node_type="Trellis2Conditioning",
            display_name="Trellis2 Conditioning",
            category="model/conditioning/trellis",
            inputs=(
                InputSpec("clip_vision_model", TypeExpr.concrete(CLIP_VISION_TYPE)),
                InputSpec(
                    "image",
                    TypeExpr.concrete(IMAGE_TYPE),
                    doc="Preprocessed image from ImageCropToMask (pad_factor=1.0 for TRELLIS.2).",
                ),
            ),
            outputs=(OutputSpec("positive", conditioning), OutputSpec("negative", conditioning)),
        )

    @classmethod
    def execute(cls, *, clip_vision_model: Any, image: object) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        utils = importlib.import_module("dinkster_inference.utils")
        pixels = materialize_image(image)[..., :3]
        out_device = management.intermediate_device()
        management.load_model_gpu(clip_vision_model.patcher)
        device = management.get_torch_device()
        mean = torch.tensor(
            clip_vision_model.image_mean or [0.485, 0.456, 0.406], device=device
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            clip_vision_model.image_std or [0.229, 0.224, 0.225], device=device
        ).view(1, 3, 1, 1)
        tokens_512: list[Any] = []
        tokens_1024: list[Any] = []
        for index in range(pixels.shape[0]):
            item = pixels[index].movedim(-1, -3).unsqueeze(0).contiguous().float().clamp(0, 1)
            for size, tokens in ((512, tokens_512), (1024, tokens_1024)):
                resized = utils.common_upscale(item, size, size, "lanczos", "disabled").to(device)
                encoded = clip_vision_model.model(
                    (resized - mean) / std, skip_norm_elementwise=True
                )
                tokens.append(encoded[0].to(out_device))
        cond_512 = torch.cat(tokens_512, dim=0)
        cond_1024 = torch.cat(tokens_1024, dim=0)
        return cls.outputs(
            positive=[[cond_512, {"embeds": cond_1024}]],
            negative=[[torch.zeros_like(cond_512), {"embeds": torch.zeros_like(cond_1024)}]],
        )
