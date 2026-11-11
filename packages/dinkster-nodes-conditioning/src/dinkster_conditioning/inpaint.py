"""Native ComfyUI VAE inpainting and conditioning nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_BOOLEAN,
    IMAGE_TYPE,
    LATENT_TYPE,
    MASK_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
    with_identity_replacement,
)
from dinkster_pack_helpers.conditioning import conditioning_set_values

_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)


_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)


_IMAGE = TypeExpr.concrete(IMAGE_TYPE)


_LATENT = TypeExpr.concrete(LATENT_TYPE)


_MASK = TypeExpr.concrete(MASK_TYPE)


_VAE = TypeExpr.concrete(VAE_TYPE)


def _mask_for_pixels(mask: Any, pixels: Any) -> Any:
    torch = importlib.import_module("torch")
    return torch.nn.functional.interpolate(
        mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
        size=(pixels.shape[1], pixels.shape[2]),
        mode="bilinear",
    )


def _crop(pixels: Any, mask: Any, ratio: int) -> tuple[Any, Any]:
    width = (pixels.shape[1] // ratio) * ratio
    height = (pixels.shape[2] // ratio) * ratio
    if pixels.shape[1] == width and pixels.shape[2] == height:
        return pixels, mask
    x_offset = (pixels.shape[1] % ratio) // 2
    y_offset = (pixels.shape[2] % ratio) // 2
    return (
        pixels[:, x_offset : width + x_offset, y_offset : height + y_offset, :],
        mask[:, :, x_offset : width + x_offset, y_offset : height + y_offset],
    )


def _masked_pixels(pixels: Any, mask: Any) -> Any:
    result = pixels.clone()
    inverse_mask = (1.0 - mask.round()).squeeze(1)
    for channel in range(3):
        result[:, :, :, channel] -= 0.5
        result[:, :, :, channel] *= inverse_mask
        result[:, :, :, channel] += 0.5
    return result


class InpaintModelConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            NodeSchema(
                node_type="dinkster.conditioning.inpaint_model_conditioning",
                display_name="InpaintModelConditioning",
                category="model/conditioning",
                inputs=(
                    InputSpec("positive", _CONDITIONING),
                    InputSpec("negative", _CONDITIONING),
                    InputSpec("vae", _VAE),
                    InputSpec("pixels", _IMAGE),
                    InputSpec("mask", _MASK),
                    InputSpec("noise_mask", _BOOLEAN, default=True),
                ),
                outputs=(
                    OutputSpec("positive", _CONDITIONING),
                    OutputSpec("negative", _CONDITIONING),
                    OutputSpec("latent", _LATENT),
                ),
            ),
            "InpaintModelConditioning",
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        vae: Any,
        pixels: object,
        mask: object,
        noise_mask: bool = True,
    ) -> Mapping[str, object]:
        image_value: Any = materialize_image(pixels)
        mask_value: Any = materialize_image(mask)
        mask_value = _mask_for_pixels(mask_value, image_value)
        original_pixels = image_value
        cropped_pixels, cropped_mask = _crop(image_value, mask_value, 8)
        concat_latent = vae.encode(_masked_pixels(cropped_pixels, cropped_mask))
        original_latent = vae.encode(original_pixels)
        values = {"concat_latent_image": concat_latent, "concat_mask": cropped_mask}
        latent: dict[str, object] = {"samples": original_latent}
        if noise_mask:
            latent["noise_mask"] = cropped_mask
        return cls.outputs(
            positive=conditioning_set_values(positive, values),
            negative=conditioning_set_values(negative, values),
            latent=latent,
        )
