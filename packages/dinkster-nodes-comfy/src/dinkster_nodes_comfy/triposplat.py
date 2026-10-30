"""Native image preprocessing and vision conditioning for TripoSplat."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from io import BytesIO
from typing import Any

from dinkster_api.v1 import (
    CLIP_VISION_TYPE,
    CONDITIONING_TYPE,
    CORE_FLOAT,
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    MASK_TYPE,
    MODEL_TYPE,
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
    pack_logger,
    report_preview,
    report_progress,
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


class TripoSplatConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="TripoSplatConditioning",
            display_name="TripoSplat Conditioning",
            category="model/conditioning/triposplat",
            description=(
                "Encode the image with DINOv3 and the Flux2 VAE into TripoSplat "
                "positive/negative conditioning, and create the fixed size noise target "
                "(latent + camera) for the KSampler"
            ),
            inputs=(
                InputSpec(
                    "clip_vision",
                    TypeExpr.concrete(CLIP_VISION_TYPE),
                    doc="DINOv3 ViT-H/16+ image encoder",
                ),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE), doc="Flux2 VAE"),
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
            ),
            outputs=(
                OutputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
                OutputSpec("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
                OutputSpec(
                    "latent",
                    TypeExpr.concrete(LATENT_TYPE),
                    doc="The fixed size noise target (latent +camera).",
                ),
            ),
        )

    @classmethod
    def execute(cls, *, clip_vision: Any, vae: Any, image: object) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        functional = importlib.import_module("torch.nn.functional")
        management = importlib.import_module("dinkster_inference.model_management")
        image_tensor = materialize_image(image)
        management.load_model_gpu(clip_vision.patcher)
        device = clip_vision.load_device
        pixels = image_tensor.movedim(-1, 1).to(device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        pixels = (pixels - mean) / std
        sequence = clip_vision.model(pixel_values=pixels.float())[0]
        feature = functional.layer_norm(sequence.float(), sequence.shape[-1:]).to(
            management.intermediate_device()
        )
        reference = vae.encode(image_tensor).to(management.intermediate_device())
        batch = reference.shape[0]
        positive = [[feature, {"reference_latents": [reference]}]]
        negative = [
            [torch.zeros_like(feature), {"reference_latents": [torch.zeros_like(reference)]}]
        ]
        device = management.intermediate_device()
        samples = MultiStreamLatent[Any].from_pairs(
            (
                ("shape", torch.zeros([batch, 8192, 16], device=device)),
                ("camera", torch.zeros([batch, 1, 5], device=device)),
            )
        )
        return cls.outputs(positive=positive, negative=negative, latent={"samples": samples})


class TripoSplatSamplingPreview(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="TripoSplatSamplingPreview",
            display_name="TripoSplat Sampling Preview",
            category="model/latent/triposplat",
            description=(
                "Patch the TripoSplat model for the standard Ksampler node to show a live decoded "
                "gaussian splat preview at each step."
            ),
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE), doc="TripoSplat VAE decoder"),
                InputSpec(
                    "octree_level",
                    TypeExpr.concrete(CORE_INT),
                    default=5,
                    widget=NumberWidget(min=2, max=8),
                    advanced=True,
                    doc="Octree depth for the preview decode (lower = cheaper/coarser).",
                ),
                InputSpec(
                    "num_gaussians",
                    TypeExpr.concrete(CORE_INT),
                    default=16384,
                    widget=NumberWidget(min=1024, max=262144, step=32),
                    doc=(
                        "Number of gaussians to produce for the preview "
                        "(rounded to a multiple of 32)."
                    ),
                ),
                InputSpec(
                    "yaw",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=90.0,
                    widget=NumberWidget(min=-360.0, max=360.0, step=1.0),
                    advanced=True,
                    doc="Preview camera yaw in degrees.",
                ),
                InputSpec(
                    "pitch",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=15.0,
                    widget=NumberWidget(min=-89.0, max=89.0, step=1.0),
                    advanced=True,
                    doc="Preview camera pitch in degrees.",
                ),
                InputSpec(
                    "point_size",
                    TypeExpr.concrete(CORE_INT),
                    default=3,
                    widget=NumberWidget(min=1, max=16),
                    doc=(
                        "Maximum splat radius in pixels. Each gaussian is sized from its scale "
                        "and capped here; lower = finer/pointier, higher = chunkier."
                    ),
                ),
            ),
            outputs=(OutputSpec("model", TypeExpr.concrete(MODEL_TYPE), display_name="MODEL"),),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        vae: Any,
        octree_level: int = 5,
        num_gaussians: int = 16384,
        yaw: float = 90.0,
        pitch: float = 15.0,
        point_size: int = 3,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        management = importlib.import_module("dinkster_inference.model_management")
        patcher = importlib.import_module("dinkster_inference.patcher_extension")
        preview = importlib.import_module("dinkster_inference.ldm.triposplat.preview")
        image_ops = importlib.import_module("PIL.ImageOps")
        pil_image = importlib.import_module("PIL.Image")
        config = {
            "gaussians": num_gaussians,
            "level": octree_level,
            "yaw": yaw,
            "pitch": pitch,
            "point_size": point_size,
        }
        stage = vae.first_stage_model
        tokens = model.model.diffusion_model.q_token_length
        memory = (tokens * 4 + (num_gaussians // stage.gaussians_per_point) * 10) * (
            stage.gs.model_channels * management.dtype_size(vae.vae_dtype)
        )

        def outer_sample_wrapper(executor: Any, *args: Any, **kwargs: Any) -> Any:
            arguments = list(args)
            callback_index = 5
            original = (
                arguments[callback_index]
                if len(arguments) > callback_index
                else kwargs.get("callback")
            )
            enabled, loaded = True, False

            def callback(step: int, x0: Any, x: Any, total_steps: int) -> None:
                nonlocal enabled, loaded
                if original is not None:
                    original(step, x0, x, total_steps)
                if not enabled:
                    return
                try:
                    if not loaded:
                        models = management.loaded_models(only_currently_used=True)
                        models.append(vae.patcher)
                        management.load_models_gpu(models, memory_required=memory)
                        loaded = True
                    image = preview.decode_x0_to_image(vae, x0, config)
                    image = image_ops.contain(image, (512, 512), pil_image.Resampling.BILINEAR)
                    buffer = BytesIO()
                    image.save(buffer, format="JPEG", quality=95, compress_level=1)
                    report_progress(step + 1, total_steps)
                    report_preview(buffer.getvalue(), width=image.width, height=image.height)
                except Exception as error:
                    pack_logger("dinkster-nodes-comfy").warning(
                        "TripoSplatSamplingPreview: preview failed, disabling (%s)", error
                    )
                    enabled = False

            if len(arguments) > callback_index:
                arguments[callback_index] = callback
            else:
                kwargs["callback"] = callback
            return executor(*arguments, **kwargs)

        patched = model.clone()
        patched.add_wrapper_with_key(
            patcher.WrappersMP.OUTER_SAMPLE, "triposplat_sampling_preview", outer_sample_wrapper
        )
        return cls.outputs(model=patched)
