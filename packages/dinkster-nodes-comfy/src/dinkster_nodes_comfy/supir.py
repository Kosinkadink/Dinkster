"""SUPIR guidance registered through the inference model patcher."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_FLOAT,
    IMAGE_TYPE,
    MODEL_PATCH_TYPE,
    MODEL_TYPE,
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


class SUPIRApply(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SUPIRApply",
            display_name="Apply SUPIR Patch",
            category="model/patch/supir",
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("model_patch", TypeExpr.concrete(MODEL_PATCH_TYPE)),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec(
                    "strength_start",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                    doc="Control strength at the start of sampling (high sigma).",
                ),
                InputSpec(
                    "strength_end",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                    doc=(
                        "Control strength at the end of sampling (low sigma). "
                        "Linearly interpolated from start."
                    ),
                ),
                InputSpec(
                    "restore_cfg",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=4.0,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=20.0, step=0.1),
                    doc=(
                        "Pulls denoised output toward the input latent. "
                        "Higher = stronger fidelity to input. 0 to disable."
                    ),
                ),
                InputSpec(
                    "restore_cfg_s_tmin",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.05,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                    doc="Sigma threshold below which restore_cfg is disabled.",
                ),
            ),
            outputs=(OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def _encode_with_denoise_encoder(cls, vae: Any, model_patch: Any, image: Any) -> Any:
        weights = getattr(model_patch.model, "denoise_encoder_sd", None)
        if not weights:
            return vae.encode(image)
        original = vae.patcher
        vae.patcher = original.clone()
        try:
            vae.patcher.add_patches(
                {f"encoder.{key}": (value,) for key, value in weights.items()},
                strength_patch=1.0,
                strength_model=0.0,
            )
            return vae.encode(image)
        finally:
            vae.patcher = original

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        model_patch: Any,
        vae: Any,
        image: object,
        strength_start: float,
        strength_end: float,
        restore_cfg: float,
        restore_cfg_s_tmin: float,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        supir = importlib.import_module("dinkster_inference.ldm.supir.supir_patch")
        result = model.clone()
        latent_format = model.get_model_object("latent_format")
        hint = latent_format.process_in(
            cls._encode_with_denoise_encoder(vae, model_patch, materialize_image(image)[..., :3])
        )
        patch = supir.SUPIRPatch(
            model_patch, model_patch.model.project_modules, hint, strength_start, strength_end
        )
        patch.register(result)
        if restore_cfg > 0.0:
            decoded = vae.decode(latent_format.process_out(hint))
            center = latent_format.process_in(vae.encode(decoded[..., :3]))

            def restore(args: dict[str, Any]) -> Any:
                denoised, sigma = args["denoised"], args["sigma"]
                first_sigma = sigma[0].item() if sigma.dim() > 0 else sigma.item()
                if first_sigma > restore_cfg_s_tmin:
                    reference = center.to(device=denoised.device, dtype=denoised.dtype)
                    batch = denoised.shape[0]
                    if reference.shape[0] != batch:
                        reference = (
                            reference.expand(batch, -1, -1, -1)
                            if reference.shape[0] == 1
                            else reference.repeat(
                                (batch + reference.shape[0] - 1) // reference.shape[0], 1, 1, 1
                            )[:batch]
                        )
                    sigma_value = sigma.view(-1, 1, 1, 1) if sigma.dim() > 0 else sigma
                    denoised = denoised - (denoised - reference) * (
                        (sigma_value / 14.6146) ** restore_cfg
                    )
                return denoised

            result.set_model_sampler_post_cfg_function(restore)
        return cls.outputs(MODEL=result)
