"""Native ComfyUI VAE inpainting and conditioning nodes."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_BOOLEAN,
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    MASK_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_image,
)

_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
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


def _set_conditioning_values(
    conditioning: Sequence[Sequence[Any]], values: Mapping[str, object]
) -> list[list[Any]]:
    return [[entry[0], {**entry[1], **values}] for entry in conditioning]


class VAEEncodeForInpaint(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="VAEEncodeForInpaint",
            display_name="VAE Encode (for Inpainting)",
            category="model/latent",
            inputs=(
                InputSpec("pixels", _IMAGE),
                InputSpec("vae", _VAE),
                InputSpec("mask", _MASK),
                InputSpec(
                    "grow_mask_by",
                    _INT,
                    default=6,
                    widget=NumberWidget(min=0, max=64, step=1),
                ),
            ),
            outputs=(OutputSpec("LATENT", _LATENT),),
        )

    @classmethod
    def execute(
        cls, *, vae: Any, pixels: object, mask: object, grow_mask_by: int = 6
    ) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        image_value: Any = materialize_image(pixels)
        mask_value: Any = materialize_image(mask)
        mask_value = _mask_for_pixels(mask_value, image_value)
        image_value, mask_value = _crop(image_value, mask_value, vae.spacial_compression_encode())
        if grow_mask_by == 0:
            mask_erosion = mask_value
        else:
            kernel = torch.ones((1, 1, grow_mask_by, grow_mask_by))
            padding = math.ceil((grow_mask_by - 1) / 2)
            mask_erosion = torch.clamp(
                torch.nn.functional.conv2d(mask_value.round(), kernel, padding=padding), 0, 1
            )
        samples = vae.encode(_masked_pixels(image_value, mask_value))
        return cls.outputs(
            LATENT={
                "samples": samples,
                "noise_mask": mask_erosion[
                    :, :, : image_value.shape[1], : image_value.shape[2]
                ].round(),
            }
        )


class InpaintModelConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="InpaintModelConditioning",
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
            positive=_set_conditioning_values(positive, values),
            negative=_set_conditioning_values(negative, values),
            latent=latent,
        )


INPAINT_NODES = (VAEEncodeForInpaint, InpaintModelConditioning)
