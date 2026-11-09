"""VOID quadmask, inpainting conditioning and precomputed noise source."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    MASK_TYPE,
    NOISE_TYPE,
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

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_MASK = TypeExpr.concrete(MASK_TYPE)


def _valid_length(length: int) -> int:
    latent_t = ((length - 1) // 4) + 1
    if latent_t % 2 == 0:
        return length
    return (max(2, (latent_t // 2) * 2) - 1) * 4 + 1


class VOIDQuadmaskPreprocess(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="VOIDQuadmaskPreprocess",
            display_name="VOID Quadmask Preprocessor",
            category="image/mask",
            inputs=(
                InputSpec("mask", _MASK),
                InputSpec(
                    "dilate_width",
                    TypeExpr.concrete(CORE_INT),
                    default=0,
                    widget=NumberWidget(min=0, max=50, step=1),
                    doc="Dilation radius for the primary mask region (0 = no dilation)",
                ),
            ),
            outputs=(OutputSpec("quadmask", _MASK, display_name="quadmask"),),
        )

    @classmethod
    def execute(cls, *, mask: object, dilate_width: int = 0) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        m = materialize_image(mask).clone()
        if m.max() <= 1.0:
            m = m * 255.0
        if dilate_width > 0 and m.ndim >= 3:
            binary = (m < 128).float()
            if binary.ndim == 3:
                binary = binary.unsqueeze(1)
            dilated = torch.nn.functional.max_pool2d(
                binary, kernel_size=dilate_width * 2 + 1, stride=1, padding=dilate_width
            )
            if dilated.ndim == 4:
                dilated = dilated.squeeze(1)
            m = torch.where(dilated > 0.5, torch.zeros_like(m), m)
        m = torch.where(m <= 31, torch.zeros_like(m), m)
        m = torch.where((m > 31) & (m <= 95), torch.full_like(m, 63), m)
        m = torch.where((m > 95) & (m <= 191), torch.full_like(m, 127), m)
        m = torch.where(m > 191, torch.full_like(m, 255), m)
        return cls.outputs(quadmask=(255.0 - m) / 255.0)


class VOIDInpaintConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="VOIDInpaintConditioning",
            category="model/conditioning/void",
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec(
                    "video", TypeExpr.concrete(IMAGE_TYPE), doc="Source video frames [T, H, W, 3]"
                ),
                InputSpec(
                    "quadmask",
                    _MASK,
                    doc="Preprocessed quadmask from VOIDQuadmaskPreprocess [T, H, W]",
                ),
                InputSpec(
                    "width",
                    TypeExpr.concrete(CORE_INT),
                    default=672,
                    widget=NumberWidget(min=16, max=16384, step=8),
                ),
                InputSpec(
                    "height",
                    TypeExpr.concrete(CORE_INT),
                    default=384,
                    widget=NumberWidget(min=16, max=16384, step=8),
                ),
                InputSpec(
                    "length",
                    TypeExpr.concrete(CORE_INT),
                    default=45,
                    widget=NumberWidget(min=1, max=16384, step=1),
                    doc=(
                        "Number of pixel frames to process. For CogVideoX-Fun-V1.5 "
                        "(patch_size_t=2), latent_t must be even - lengths that produce "
                        "odd latent_t are rounded down (e.g. 49 -> 45)."
                    ),
                ),
                InputSpec(
                    "batch_size",
                    TypeExpr.concrete(CORE_INT),
                    default=1,
                    widget=NumberWidget(min=1, max=64),
                ),
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
        vae: Any,
        video: object,
        quadmask: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        utils = importlib.import_module("dinkster_inference.utils")
        management = importlib.import_module("dinkster_inference.model_management")
        adjusted = _valid_length(length)
        if adjusted != length:
            logging.warning(
                "VOIDInpaintConditioning: rounding length %d to %d for even latent_t.",
                length,
                adjusted,
            )
            length = adjusted
        latent_t = ((length - 1) // 4) + 1
        vid = materialize_image(video)[:length]
        vid = utils.common_upscale(vid.movedim(-1, 1), width, height, "bilinear", "center").movedim(
            1, -1
        )
        qm = materialize_image(quadmask)[:length]
        if qm.ndim == 3:
            qm = qm.unsqueeze(-1)
        qm = utils.common_upscale(qm.movedim(-1, 1), width, height, "bilinear", "center").movedim(
            1, -1
        )
        if qm.ndim == 4 and qm.shape[-1] == 1:
            qm = qm.squeeze(-1)
        mask = qm.unsqueeze(-1).expand(-1, -1, -1, 3) if qm.ndim == 3 else qm
        mask_latents = vae.encode(1.0 - mask)
        video_latents = vae.encode(vid[:, :, :, :3] * (1.0 - mask))
        matched: list[Any] = []
        for lat in (mask_latents, video_latents):
            if lat.shape[2] > latent_t:
                lat = lat[:, :, :latent_t]
            elif lat.shape[2] < latent_t:
                lat = torch.cat(
                    [lat, lat[:, :, -1:].repeat(1, 1, latent_t - lat.shape[2], 1, 1)], dim=2
                )
            matched.append(lat)
        values = {"concat_latent_image": torch.cat(matched, dim=1)}
        positive = conditioning_set_values(positive, values)
        negative = conditioning_set_values(negative, values)
        latent = torch.zeros(
            (batch_size, 16, latent_t, height // 8, width // 8),
            device=management.intermediate_device(),
        )
        return cls.outputs(positive=positive, negative=negative, latent={"samples": latent})


class _NoiseFromLatent:
    seed = 0

    def __init__(self, latent: object) -> None:
        self._samples = materialize_latent(latent)["samples"]

    def generate_noise(self, input_latent: object) -> Any:
        return self._samples.clone().cpu()


class VOIDWarpedNoiseSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="VOIDWarpedNoiseSource",
            category="model/latent/void",
            inputs=(
                InputSpec("warped_noise", _LATENT, doc="Warped noise latent from VOIDWarpedNoise"),
            ),
            outputs=(OutputSpec("NOISE", TypeExpr.concrete(NOISE_TYPE)),),
        )

    @classmethod
    def execute(cls, *, warped_noise: object) -> Mapping[str, object]:
        return cls.outputs(NOISE=_NoiseFromLatent(warped_noise))
