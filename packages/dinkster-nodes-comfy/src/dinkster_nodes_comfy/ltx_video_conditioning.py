"""Native LTXV image, frame-rate and guide-tail conditioning nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_BOOLEAN,
    CORE_FLOAT,
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
    materialize_latent,
)

from .conditioning import conditioning_set_values

_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)
_CATEGORY = "model/conditioning/ltxv"


def _integer(
    name: str, default: int, minimum: int, maximum: int, step: int | None = None
) -> InputSpec:
    return InputSpec(
        name,
        _INT,
        default=default,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


def get_noise_mask(latent: Mapping[str, Any], torch: Any) -> Any:
    noise_mask = latent.get("noise_mask")
    if noise_mask is not None:
        return noise_mask.clone()
    samples = latent["samples"]
    return torch.ones(
        (samples.shape[0], 1, samples.shape[2], 1, 1),
        dtype=torch.float32,
        device=samples.device,
    )


class LTXVImgToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LTXVImgToVideo",
            category=_CATEGORY,
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", _VAE),
                InputSpec("image", _IMAGE),
                _integer("width", 768, 64, 16384, 32),
                _integer("height", 512, 64, 16384, 32),
                _integer("length", 97, 9, 16384, 8),
                _integer("batch_size", 1, 1, 4096),
                InputSpec("strength", _FLOAT, default=1.0, widget=NumberWidget(min=0.0, max=1.0)),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING, display_name="positive"),
                OutputSpec("negative", _CONDITIONING, display_name="negative"),
                OutputSpec("latent", _LATENT, display_name="latent"),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        image: object,
        vae: Any,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        strength: float,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        pixels = materialize_image(image)
        upscale = importlib.import_module("dinkster_inference.utils").common_upscale
        pixels = upscale(pixels.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        encoded = vae.encode(pixels[:, :, :, :3])
        samples = torch.zeros(
            [batch_size, 128, ((length - 1) // 8) + 1, height // 32, width // 32],
            device=management.intermediate_device(),
        )
        samples[:, :, : encoded.shape[2]] = encoded
        noise_mask = torch.ones(
            (batch_size, 1, samples.shape[2], 1, 1), dtype=torch.float32, device=samples.device
        )
        noise_mask[:, :, : encoded.shape[2]] = 1.0 - strength
        return cls.outputs(
            positive=positive,
            negative=negative,
            latent={"samples": samples, "noise_mask": noise_mask},
        )


class LTXVImgToVideoInplace(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LTXVImgToVideoInplace",
            category=_CATEGORY,
            inputs=(
                InputSpec("vae", _VAE),
                InputSpec("image", _IMAGE),
                InputSpec("latent", _LATENT),
                InputSpec("strength", _FLOAT, default=1.0, widget=NumberWidget(min=0.0, max=1.0)),
                InputSpec("bypass", _BOOLEAN, default=False, doc="Bypass the conditioning."),
            ),
            outputs=(OutputSpec("latent", _LATENT, display_name="latent"),),
        )

    @classmethod
    def execute(
        cls, *, vae: Any, image: object, latent: object, strength: float, bypass: bool = False
    ) -> Mapping[str, object]:
        if bypass:
            return cls.outputs(latent=latent)
        source = materialize_latent(latent)
        samples = source["samples"].clone()
        _, height_scale, width_scale = vae.downscale_index_formula
        height, width = samples.shape[-2] * height_scale, samples.shape[-1] * width_scale
        pixels = materialize_image(image)
        if pixels.shape[1] != height or pixels.shape[2] != width:
            upscale = importlib.import_module("dinkster_inference.utils").common_upscale
            pixels = upscale(pixels.movedim(-1, 1), width, height, "bilinear", "center").movedim(
                1, -1
            )
        encoded = vae.encode(pixels[:, :, :, :3])
        samples[:, :, : encoded.shape[2]] = encoded
        torch = importlib.import_module("torch")
        noise_mask = get_noise_mask(source, torch)
        noise_mask[:, :, : encoded.shape[2]] = 1.0 - strength
        return cls.outputs(latent={"samples": samples, "noise_mask": noise_mask})


class LTXVConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LTXVConditioning",
            category=_CATEGORY,
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec(
                    "frame_rate",
                    _FLOAT,
                    default=25.0,
                    widget=NumberWidget(min=0.0, max=1000.0, step=0.01),
                ),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING, display_name="positive"),
                OutputSpec("negative", _CONDITIONING, display_name="negative"),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        frame_rate: float,
    ) -> Mapping[str, object]:
        return cls.outputs(
            positive=conditioning_set_values(positive, {"frame_rate": frame_rate}),
            negative=conditioning_set_values(negative, {"frame_rate": frame_rate}),
        )


class LTXVCropGuides(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LTXVCropGuides",
            category=_CATEGORY,
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("latent", _LATENT),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING, display_name="positive"),
                OutputSpec("negative", _CONDITIONING, display_name="negative"),
                OutputSpec("latent", _LATENT, display_name="latent"),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        latent: object,
    ) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        source = materialize_latent(latent)
        samples = source["samples"].clone()
        noise_mask = get_noise_mask(source, torch)
        indices = next(
            (entry[1]["keyframe_idxs"] for entry in positive if "keyframe_idxs" in entry[1]), None
        )
        count = 0
        if indices is not None:
            if samples.ndim == 5:
                count = indices.shape[2] // (samples.shape[-2] * samples.shape[-1])
            else:
                entries = next(
                    (
                        entry[1]["guide_attention_entries"]
                        for entry in positive
                        if "guide_attention_entries" in entry[1]
                    ),
                    None,
                )
                count = (
                    sum(entry["latent_shape"][0] for entry in entries)
                    if entries
                    else torch.unique(indices[:, 0, :, 0]).shape[0]
                )
        if count:
            samples = samples[:, :, :-count]
            noise_mask = noise_mask[:, :, :-count]
            cleared = {"keyframe_idxs": None, "guide_attention_entries": None}
            positive = conditioning_set_values(positive, cleared)
            negative = conditioning_set_values(negative, cleared)
        return cls.outputs(
            positive=positive,
            negative=negative,
            latent={"samples": samples, "noise_mask": noise_mask},
        )


LTX_VIDEO_CONDITIONING_NODES = (
    LTXVImgToVideo,
    LTXVImgToVideoInplace,
    LTXVConditioning,
    LTXVCropGuides,
)
