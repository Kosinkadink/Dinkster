"""Cosmos image-to-video latent conditioning."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
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
    initialize_inference_process,
    materialize_image,
)

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)


def _integer(name: str, default: int, step: int) -> InputSpec:
    return InputSpec(
        name,
        _INT,
        default=default,
        widget=NumberWidget(min=16 if name != "length" else 1, max=16384, step=step),
    )


def vae_encode_with_padding(
    vae: Any, image: object, width: int, height: int, length: int, padding: int = 0
) -> Any:
    torch = importlib.import_module("torch")
    utils = importlib.import_module("dinkster_inference.utils")
    pixels = materialize_image(image)
    pixels = utils.common_upscale(
        pixels[..., :3].movedim(-1, 1), width, height, "bilinear", "center"
    ).movedim(1, -1)
    pixel_len = min(pixels.shape[0], length)
    padded_length = min(length, (((pixel_len - 1) // 8) + 1 + padding) * 8 - 7)
    padded_pixels = torch.ones((padded_length, height, width, 3)) * 0.5
    padded_pixels[:pixel_len] = pixels[:pixel_len]
    latent_len = ((pixel_len - 1) // 8) + 1
    latent_temp = vae.encode(padded_pixels)
    return latent_temp[:, :, :latent_len]


class _CosmosImageToVideoLatent(Node):
    width_default = 1280
    height_default = 704
    length_default = 121
    length_step = 8
    temporal_ratio = 8

    @classmethod
    def _process_out(cls, latent: Any, mask: Any) -> Any:
        return latent

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type=cls.__name__,
            category="model/conditioning/cosmos",
            inputs=(
                InputSpec("vae", _VAE),
                _integer("width", cls.width_default, 16),
                _integer("height", cls.height_default, 16),
                _integer("length", cls.length_default, cls.length_step),
                InputSpec("batch_size", _INT, default=1, widget=NumberWidget(min=1, max=4096)),
                InputSpec("start_image", _IMAGE, required=False),
                InputSpec("end_image", _IMAGE, required=False),
            ),
            outputs=(OutputSpec("LATENT", _LATENT),),
        )

    @classmethod
    def execute(
        cls,
        *,
        vae: Any,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        start_image: object | None = None,
        end_image: object | None = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        latent = torch.zeros(
            [
                1,
                16,
                ((length - 1) // cls.temporal_ratio) + 1,
                height // 8,
                width // 8,
            ],
            device=management.intermediate_device(),
        )
        if start_image is None and end_image is None:
            return cls.outputs(LATENT={"samples": latent})

        mask = torch.ones(
            [latent.shape[0], 1, latent.shape[-3], latent.shape[-2], latent.shape[-1]],
            device=management.intermediate_device(),
        )
        if start_image is not None:
            latent_temp = vae_encode_with_padding(
                vae, start_image, width, height, length, padding=1
            )
            latent[:, :, : latent_temp.shape[-3]] = latent_temp
            mask[:, :, : latent_temp.shape[-3]] *= 0.0
        if end_image is not None:
            latent_temp = vae_encode_with_padding(vae, end_image, width, height, length)
            latent[:, :, -latent_temp.shape[-3] :] = latent_temp
            mask[:, :, -latent_temp.shape[-3] :] *= 0.0
        latent = cls._process_out(latent, mask)
        return cls.outputs(
            LATENT={
                "samples": latent.repeat((batch_size,) + (1,) * (latent.ndim - 1)),
                "noise_mask": mask.repeat((batch_size,) + (1,) * (mask.ndim - 1)),
            }
        )


class CosmosImageToVideoLatent(_CosmosImageToVideoLatent):
    pass


class CosmosPredict2ImageToVideoLatent(_CosmosImageToVideoLatent):
    width_default = 848
    height_default = 480
    length_default = 93
    length_step = 4
    temporal_ratio = 4

    @classmethod
    def _process_out(cls, latent: Any, mask: Any) -> Any:
        latent_formats = importlib.import_module("dinkster_inference.latent_formats")
        return latent_formats.Wan21().process_out(latent) * mask + latent * (1.0 - mask)


COSMOS_CONDITIONING_NODES = (CosmosImageToVideoLatent, CosmosPredict2ImageToVideoLatent)
