"""Stable Cascade latent creation, encoding and prior conditioning."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_image,
    materialize_latent,
)

_LATENT = TypeExpr.concrete(LATENT_TYPE)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_STAGES = (OutputSpec("stage_c", _LATENT), OutputSpec("stage_b", _LATENT))
_COMPRESSION = InputSpec(
    "compression",
    TypeExpr.concrete(CORE_INT),
    default=42,
    widget=NumberWidget(min=4, max=128, step=1),
    advanced=True,
)


class StableCascade_EmptyLatentImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="StableCascade_EmptyLatentImage",
            display_name="Empty Stable Cascade Latent Image",
            category="model/latent/stable cascade",
            inputs=(
                InputSpec(
                    "width",
                    TypeExpr.concrete(CORE_INT),
                    default=1024,
                    widget=NumberWidget(min=256, max=16384, step=8),
                ),
                InputSpec(
                    "height",
                    TypeExpr.concrete(CORE_INT),
                    default=1024,
                    widget=NumberWidget(min=256, max=16384, step=8),
                ),
                _COMPRESSION,
                InputSpec(
                    "batch_size",
                    TypeExpr.concrete(CORE_INT),
                    default=1,
                    widget=NumberWidget(min=1, max=4096),
                ),
            ),
            outputs=_STAGES,
        )

    @classmethod
    def execute(
        cls, *, width: int, height: int, compression: int, batch_size: int = 1
    ) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        return cls.outputs(
            stage_c={
                "samples": torch.zeros(
                    [batch_size, 16, height // compression, width // compression]
                )
            },
            stage_b={"samples": torch.zeros([batch_size, 4, height // 4, width // 4])},
        )


class StableCascade_StageC_VAEEncode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="StableCascade_StageC_VAEEncode",
            display_name="Stable Cascade Stage C VAE Encode",
            category="model/latent/stable cascade",
            inputs=(InputSpec("image", _IMAGE), InputSpec("vae", _VAE), _COMPRESSION),
            outputs=_STAGES,
        )

    @classmethod
    def execute(cls, *, image: Any, vae: Any, compression: int) -> Mapping[str, object]:
        image = materialize_image(image)
        torch = importlib.import_module("torch")
        utils = importlib.import_module("dinkster_inference.utils")
        width, height = image.shape[-2], image.shape[-3]
        resized = utils.common_upscale(
            image.movedim(-1, 1),
            (width // compression) * vae.downscale_ratio,
            (height // compression) * vae.downscale_ratio,
            "bicubic",
            "center",
        ).movedim(1, -1)
        encoded = vae.encode(resized[:, :, :, :3])
        return cls.outputs(
            stage_c={"samples": encoded},
            stage_b={
                "samples": torch.zeros([encoded.shape[0], 4, (height // 8) * 2, (width // 8) * 2])
            },
        )


class StableCascade_StageB_Conditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="StableCascade_StageB_Conditioning",
            category="model/conditioning/stable cascade",
            inputs=(InputSpec("conditioning", _CONDITIONING), InputSpec("stage_c", _LATENT)),
            outputs=(OutputSpec("CONDITIONING", _CONDITIONING),),
        )

    @classmethod
    def execute(
        cls, *, conditioning: Sequence[Sequence[Any]], stage_c: Mapping[str, Any]
    ) -> Mapping[str, object]:
        stage_c = materialize_latent(stage_c)
        prior = materialize_image(stage_c["samples"])
        return cls.outputs(
            CONDITIONING=[
                [text, {**metadata, "stable_cascade_prior": prior}]
                for text, metadata in conditioning
            ]
        )


class StableCascade_SuperResolutionControlnet(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="StableCascade_SuperResolutionControlnet",
            category="experimental/stable cascade",
            inputs=(InputSpec("image", _IMAGE), InputSpec("vae", _VAE)),
            outputs=(OutputSpec("controlnet_input", _IMAGE), *_STAGES),
        )

    @classmethod
    def execute(cls, *, image: Any, vae: Any) -> Mapping[str, object]:
        image = materialize_image(image)
        torch = importlib.import_module("torch")
        width, height, batch_size = image.shape[-2], image.shape[-3], image.shape[0]
        return cls.outputs(
            controlnet_input=vae.encode(image[:, :, :, :3]).movedim(1, -1),
            stage_c={"samples": torch.zeros([batch_size, 16, height // 16, width // 16])},
            stage_b={"samples": torch.zeros([batch_size, 4, height // 2, width // 2])},
        )


STABLE_CASCADE_NODES = (
    StableCascade_EmptyLatentImage,
    StableCascade_StageB_Conditioning,
    StableCascade_StageC_VAEEncode,
    StableCascade_SuperResolutionControlnet,
)
