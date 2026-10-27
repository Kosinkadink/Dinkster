"""Native image preprocessing and gaussian decoding for TripoSplat."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    MASK_TYPE,
    SPLAT_TYPE,
    VAE_TYPE,
    InputSpec,
    MultiStreamLatent,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
    materialize_latent,
)


def _preprocess(image: Any, mask: Any, erode_radius: int, size: int) -> Any:
    torch = importlib.import_module("torch")
    functional = importlib.import_module("torch.nn.functional")
    utils = importlib.import_module("dinkster_inference.utils")
    rgb = image[..., :3].clamp(0, 1).movedim(-1, 0)
    alpha = mask.clamp(0, 1)[None]
    rgba = torch.cat([rgb, alpha], 0)[None]
    height, width = rgba.shape[-2:]
    scale = size / min(width, height)
    rgba = utils.common_upscale(
        rgba, max(1, round(width * scale)), max(1, round(height * scale)), "lanczos", "disabled"
    ).clamp(0, 1)
    alpha = rgba[:, 3:4]
    if erode_radius > 0:
        alpha = -functional.max_pool2d(-alpha, 2 * erode_radius + 1, stride=1, padding=erode_radius)
        rgba = torch.cat([rgba[:, :3], alpha], 1)
    ys, xs = torch.nonzero(alpha[0, 0] > 0, as_tuple=True)
    if xs.numel() == 0:
        raise ValueError("TripoSplatPreprocessImage: mask is empty (no foreground pixels).")
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2 * 1.2
    left, upper = int(center_x - half), int(center_y - half)
    right, lower = int(center_x + half), int(center_y + half)
    height, width = rgba.shape[-2:]
    crop = rgba.new_zeros((1, 4, lower - upper, right - left))
    sx0, sy0, sx1, sy1 = max(left, 0), max(upper, 0), min(right, width), min(lower, height)
    if sx1 > sx0 and sy1 > sy0:
        crop[:, :, sy0 - upper : sy1 - upper, sx0 - left : sx1 - left] = rgba[
            :, :, sy0:sy1, sx0:sx1
        ]
    crop = utils.common_upscale(crop, size, size, "lanczos", "disabled").clamp(0, 1)
    return (crop[:, :3] * crop[:, 3:4])[0].movedim(0, -1).unsqueeze(0)


class TripoSplatPreprocessImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="TripoSplatPreprocessImage",
            display_name="TripoSplat Preprocess Image",
            category="model/conditioning/triposplat",
            description=(
                "Crop center each image to a square canvas on a black background and add padding."
            ),
            inputs=(
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec("mask", TypeExpr.concrete(MASK_TYPE)),
                InputSpec(
                    "erode_radius",
                    TypeExpr.concrete(CORE_INT),
                    default=1,
                    widget=NumberWidget(min=0, max=16),
                    doc=(
                        "Erode the alpha matte by this pixel radius before cropping "
                        "(avoids border bleed)."
                    ),
                ),
                InputSpec(
                    "size",
                    TypeExpr.concrete(CORE_INT),
                    default=1024,
                    widget=NumberWidget(min=256, max=4096, step=16),
                    doc=(
                        "Square image size. The model is trained at 1024; "
                        "other sizes run but are off-distribution."
                    ),
                ),
            ),
            outputs=(OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),),
        )

    @classmethod
    def execute(
        cls, *, image: Any, mask: Any, erode_radius: int, size: int
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        functional = importlib.import_module("torch.nn.functional")
        utils = importlib.import_module("dinkster_inference.utils")
        image, mask = materialize_image(image), materialize_image(mask)
        size = max(16, (int(size) // 16) * 16)
        if mask.shape[0] != image.shape[0]:
            mask = utils.repeat_to_batch_size(mask, image.shape[0])
        if tuple(mask.shape[1:]) != tuple(image.shape[1:3]):
            mask = functional.interpolate(
                mask[:, None].float(),
                size=tuple(image.shape[1:3]),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        prepared = torch.cat(
            [_preprocess(image[i], mask[i], erode_radius, size) for i in range(image.shape[0])],
            dim=0,
        )
        return cls.outputs(image=prepared)


class VAEDecodeTripoSplat(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="VAEDecodeTripoSplat",
            display_name="TripoSplat Decode",
            category="model/latent/triposplat",
            description=(
                "Decode the sampled TripoSplat latent into a 3D gaussian splat. "
                "Modify the number of gaussians to vary the density."
            ),
            inputs=(
                InputSpec("samples", TypeExpr.concrete(LATENT_TYPE)),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE), doc="TripoSplat VAE decoder"),
                InputSpec(
                    "num_gaussians",
                    TypeExpr.concrete(CORE_INT),
                    default=262144,
                    widget=NumberWidget(min=32768, max=1048576, step=32),
                    doc=(
                        "Number of gaussians to produce (rounded to a multiple of 32). "
                        "262144 matches the octree's point density; higher oversamples "
                        "the same points (denser, but no new detail) and costs "
                        "proportionally more VRAM/time."
                    ),
                ),
                InputSpec(
                    "seed",
                    TypeExpr.concrete(CORE_INT),
                    default=0,
                    widget=NumberWidget(min=0, max=0xFFFFFFFFFFFFFFFF),
                    doc="Seeds the octree point sampler (global RNG) for deterministic decodes.",
                ),
            ),
            outputs=(OutputSpec("splat", TypeExpr.concrete(SPLAT_TYPE)),),
        )

    @classmethod
    def execute(
        cls, *, samples: object, vae: Any, num_gaussians: int, seed: int
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        streams = materialize_latent(samples)["samples"]
        if isinstance(streams, MultiStreamLatent):
            latent = cast("MultiStreamLatent[Any]", streams).streams[0].payload
        else:
            latent = streams.unbind()[0] if getattr(streams, "is_nested", False) else streams
        decoder = vae.first_stage_model
        per_point = decoder.gaussians_per_point
        count = max(32768, min(1048576, int(num_gaussians)))
        if count % per_point != 0:
            count = round(count / per_point) * per_point
        memory = (
            (latent.shape[1] * 4 + (count // per_point) * 10)
            * decoder.gs.model_channels
            * management.dtype_size(vae.vae_dtype)
        )
        management.load_models_gpu([vae.patcher], memory_required=memory)
        latent = latent.to(device=vae.device, dtype=vae.vae_dtype)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        parts = [
            gaussian.render_tensors()
            for gaussian in decoder.decode(latent, num_gaussians=count, generator=generator)
        ]
        return cls.outputs(
            splat=dict(
                zip(
                    ("positions", "scales", "rotations", "opacities", "sh"),
                    (torch.stack(tensors) for tensors in zip(*parts, strict=False)),
                    strict=True,
                )
            )
        )
