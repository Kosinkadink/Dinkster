"""Native autoregressive video sampler construction."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    MODEL_TYPE,
    SAMPLER_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)


class SamplerARVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SamplerARVideo",
            display_name="Sampler AR Video",
            category="model/sampling/samplers",
            inputs=(
                InputSpec(
                    "num_frame_per_block",
                    TypeExpr.concrete(CORE_INT),
                    default=1,
                    widget=NumberWidget(min=1, max=64),
                    doc=(
                        "Frames per autoregressive block. 1 = framewise, "
                        "3 = chunkwise. Must match the checkpoint's training mode."
                    ),
                ),
            ),
            outputs=(OutputSpec("SAMPLER", TypeExpr.concrete(SAMPLER_TYPE)),),
        )

    @classmethod
    def execute(cls, *, num_frame_per_block: int) -> Mapping[str, object]:
        initialize_inference_process()
        samplers = importlib.import_module("dinkster_inference.samplers")
        return cls.outputs(
            SAMPLER=samplers.ksampler("ar_video", {"num_frame_per_block": num_frame_per_block})
        )


class ARVideoI2V(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        integer = TypeExpr.concrete(CORE_INT)
        return NodeSchema(
            node_type="ARVideoI2V",
            category="model/conditioning/autoregressive",
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec("start_image", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec(
                    "width", integer, default=832, widget=NumberWidget(min=16, max=8192, step=16)
                ),
                InputSpec(
                    "height", integer, default=480, widget=NumberWidget(min=16, max=8192, step=16)
                ),
                InputSpec(
                    "length", integer, default=81, widget=NumberWidget(min=1, max=1024, step=4)
                ),
                InputSpec("batch_size", integer, default=1, widget=NumberWidget(min=1, max=64)),
            ),
            outputs=(
                OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),
                OutputSpec("LATENT", TypeExpr.concrete(LATENT_TYPE)),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        vae: Any,
        start_image: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        utils = importlib.import_module("dinkster_inference.utils")
        management = importlib.import_module("dinkster_inference.model_management")
        image = utils.common_upscale(
            materialize_image(start_image)[:1].movedim(-1, 1),
            width,
            height,
            "bilinear",
            "center",
        ).movedim(1, -1)
        initial_latent = vae.encode(image[:, :, :, :3])
        patched = model.clone()
        options = patched.model_options.setdefault("transformer_options", {})
        options.setdefault("ar_config", {})["initial_latent"] = initial_latent
        samples = torch.zeros(
            (batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8),
            device=management.intermediate_device(),
        )
        return cls.outputs(MODEL=patched, LATENT={"samples": samples})
