"""Model guidance patches from ComfyUI's CFG extension nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_FLOAT,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
)

_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_MODEL = TypeExpr.concrete(MODEL_TYPE)


def optimized_scale(positive: Any, negative: Any) -> Any:
    torch = importlib.import_module("torch")
    positive_flat = positive.reshape(positive.shape[0], -1)
    negative_flat = negative.reshape(negative.shape[0], -1)
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
    return (dot_product / squared_norm).reshape([positive.shape[0]] + [1] * (positive.ndim - 1))


class CFGZeroStar(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="CFGZeroStar",
            category="advanced/guidance",
            inputs=(InputSpec("model", _MODEL),),
            outputs=(OutputSpec("MODEL", _MODEL, display_name="patched_model"),),
        )

    @classmethod
    def execute(cls, *, model: Any) -> Mapping[str, object]:
        patched = model.clone()

        def cfg_zero_star(args: dict[str, Any]) -> Any:
            guidance_scale = args["cond_scale"]
            x = args["input"]
            cond = args["cond_denoised"]
            uncond = args["uncond_denoised"]
            alpha = optimized_scale(x - cond, x - uncond)
            return (
                args["denoised"] + uncond * (alpha - 1.0) + guidance_scale * uncond * (1.0 - alpha)
            )

        patched.set_model_sampler_post_cfg_function(cfg_zero_star)
        return cls.outputs(MODEL=patched)


class CFGNorm(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="CFGNorm",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "strength",
                    _FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
                InputSpec(
                    "pre_cfg", TypeExpr.concrete(CORE_BOOLEAN), default=False, required=False
                ),
            ),
            outputs=(OutputSpec("MODEL", _MODEL, display_name="patched_model"),),
        )

    @classmethod
    def execute(cls, *, model: Any, strength: float, pre_cfg: bool = False) -> Mapping[str, object]:
        patched = model.clone()
        if pre_cfg:

            def cfg_norm_pre(args: dict[str, Any]) -> Any:
                torch = importlib.import_module("torch")
                cond = args["cond"]
                uncond = args["uncond"]
                combined = uncond + args["cond_scale"] * (cond - uncond)
                cond_norm = torch.linalg.vector_norm(cond, dim=1, keepdim=True)
                combined_norm = torch.linalg.vector_norm(combined, dim=1, keepdim=True)
                rescale = torch.where(
                    combined_norm > 0,
                    cond_norm / combined_norm.clamp_min(1e-12),
                    torch.ones_like(combined_norm),
                )
                rescaled = combined * rescale
                if strength != 1.0:
                    rescaled = strength * rescaled + (1.0 - strength) * combined
                return rescaled

            patched.set_model_sampler_cfg_function(cfg_norm_pre)
        else:

            def cfg_norm(args: dict[str, Any]) -> Any:
                torch = importlib.import_module("torch")
                cond = args["cond_denoised"]
                denoised = args["denoised"]
                scale = (
                    torch.norm(cond, dim=1, keepdim=True)
                    / (torch.norm(denoised, dim=1, keepdim=True) + 1e-8)
                ).clamp(0.0, 1.0)
                return denoised * scale * strength

            patched.set_model_sampler_post_cfg_function(cfg_norm)
        return cls.outputs(MODEL=patched)


def _project(value: Any, direction: Any) -> tuple[Any, Any]:
    torch = importlib.import_module("torch")
    direction = torch.nn.functional.normalize(direction, dim=[-1, -2, -3])
    parallel = (value * direction).sum(dim=[-1, -2, -3], keepdim=True) * direction
    return parallel, value - parallel


class APG(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="APG",
            display_name="Adaptive Projected Guidance",
            category="model/sampling/custom",
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "eta",
                    _FLOAT,
                    default=1.0,
                    advanced=True,
                    widget=NumberWidget(min=-10.0, max=10.0, step=0.01),
                ),
                InputSpec(
                    "norm_threshold",
                    _FLOAT,
                    default=5.0,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=50.0, step=0.1),
                ),
                InputSpec(
                    "momentum",
                    _FLOAT,
                    default=0.0,
                    advanced=True,
                    widget=NumberWidget(min=-5.0, max=1.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(
        cls, *, model: Any, eta: float, norm_threshold: float, momentum: float
    ) -> Mapping[str, object]:
        running_average: Any = 0
        previous_sigma: Any = None

        def pre_cfg(args: dict[str, Any]) -> Any:
            nonlocal previous_sigma, running_average
            outputs = args["conds_out"]
            if len(outputs) == 1:
                return outputs
            cond, uncond = outputs[:2]
            sigma = args["sigma"][0]
            if previous_sigma is not None and sigma > previous_sigma:
                running_average = 0
            previous_sigma = sigma
            guidance = cond - uncond
            torch = importlib.import_module("torch")
            if momentum != 0:
                running_average = (
                    guidance
                    if not torch.is_tensor(running_average)
                    else momentum * running_average + guidance
                )
                guidance = running_average
            if norm_threshold > 0:
                norm = guidance.norm(p=2, dim=[-1, -2, -3], keepdim=True)
                guidance = guidance * torch.minimum(torch.ones_like(norm), norm_threshold / norm)
            parallel, orthogonal = _project(guidance, cond)
            modified = (uncond + (orthogonal + eta * parallel)) + (cond - uncond) / args[
                "cond_scale"
            ]
            return [modified, uncond] + outputs[2:]

        patched = model.clone()
        patched.set_model_sampler_pre_cfg_function(pre_cfg)
        return cls.outputs(MODEL=patched)


def score_tangential_damping(cond_score: Any, uncond_score: Any) -> Any:
    torch = importlib.import_module("torch")
    batch = cond_score.shape[0]
    cond_flat = cond_score.reshape(batch, 1, -1).float()
    uncond_flat = uncond_score.reshape(batch, 1, -1).float()
    matrix = torch.cat((uncond_flat, cond_flat), dim=1)
    try:
        _, _, vectors = torch.linalg.svd(matrix, full_matrices=False)
    except RuntimeError:
        _, _, vectors = torch.linalg.svd(matrix.cpu(), full_matrices=False)
    principal = vectors[:, 0:1, :].to(uncond_flat.device)
    return (
        ((uncond_flat @ principal.transpose(-2, -1)) * principal)
        .reshape_as(uncond_score)
        .to(uncond_score.dtype)
    )


class TCFG(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="TCFG",
            display_name="Tangential Damping CFG",
            category="advanced/guidance",
            inputs=(InputSpec("model", _MODEL),),
            outputs=(OutputSpec("MODEL", _MODEL, display_name="patched_model"),),
        )

    @classmethod
    def execute(cls, *, model: Any) -> Mapping[str, object]:
        patched = model.clone()

        def tangential_damping(args: dict[str, Any]) -> Any:
            outputs = args["conds_out"]
            if len(outputs) <= 1 or None in args["conds"][:2]:
                return outputs
            x = args["input"]
            cond, uncond = outputs[:2]
            return [cond, x - score_tangential_damping(x - cond, x - uncond)] + outputs[2:]

        patched.set_model_sampler_pre_cfg_function(tangential_damping)
        return cls.outputs(MODEL=patched)


class Mahiro(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="Mahiro",
            display_name="Positive-Biased Guidance",
            category="experimental",
            inputs=(InputSpec("model", _MODEL),),
            outputs=(OutputSpec("MODEL", _MODEL, display_name="patched_model"),),
            search_terms=(
                "mahiro",
                "mahiro cfg",
                "similarity-adaptive guidance",
                "positive-biased cfg",
            ),
        )

    @classmethod
    def execute(cls, *, model: Any) -> Mapping[str, object]:
        patched = model.clone()

        def mahiro(args: dict[str, Any]) -> Any:
            torch = importlib.import_module("torch")
            scale = args["cond_scale"]
            leap = args["cond_denoised"] * scale
            denoised = args["denoised"]
            normalized_uncond = (
                torch.sqrt((args["uncond_denoised"] * scale).abs())
                * (args["uncond_denoised"] * scale).sign()
            )
            merged = (leap + denoised) / 2
            normalized_merged = torch.sqrt(merged.abs()) * merged.sign()
            similarity_scale = 2 * (
                torch.nn.functional.cosine_similarity(normalized_uncond, normalized_merged).mean()
                + 1
            )
            return (similarity_scale * denoised + (4 - similarity_scale) * leap) / 4

        patched.set_model_sampler_post_cfg_function(mahiro)
        return cls.outputs(MODEL=patched)


class NAGuidance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="NAGuidance",
            display_name="Normalized Attention Guidance",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "nag_scale",
                    _FLOAT,
                    default=5.0,
                    widget=NumberWidget(min=0.0, max=50.0, step=0.1),
                ),
                InputSpec(
                    "nag_alpha",
                    _FLOAT,
                    default=0.5,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "nag_tau",
                    _FLOAT,
                    default=1.5,
                    widget=NumberWidget(min=1.0, max=10.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(
        cls, *, model: Any, nag_scale: float, nag_alpha: float, nag_tau: float
    ) -> Mapping[str, object]:
        patched = model.clone()

        def nag(out: Any, extra_options: dict[str, Any]) -> Any:
            cond_or_uncond = extra_options.get("cond_or_uncond")
            if cond_or_uncond is None or not (1 in cond_or_uncond and 0 in cond_or_uncond):
                return out
            image_slice = extra_options.get("img_slice")
            target = out if image_slice is None else out[:, image_slice[0] : image_slice[1]]
            half = target.shape[0] // len(cond_or_uncond)
            negative_index = cond_or_uncond.index(1)
            positive_index = cond_or_uncond.index(0)
            positive = target[half * positive_index : half * (positive_index + 1)]
            negative = target[half * negative_index : half * (negative_index + 1)]
            guided = positive * nag_scale - negative * (nag_scale - 1.0)
            torch = importlib.import_module("torch")
            positive_norm = torch.norm(positive, p=1, dim=-1, keepdim=True).clamp_min(1e-6)
            guided_norm = torch.norm(guided, p=1, dim=-1, keepdim=True).clamp_min(1e-6)
            ratio = guided_norm / positive_norm
            normalized = guided * (torch.minimum(ratio, torch.full_like(ratio, nag_tau)) / ratio)
            result = normalized * nag_alpha + positive * (1.0 - nag_alpha)
            if image_slice is not None:
                out[
                    half * negative_index : half * (negative_index + 1),
                    image_slice[0] : image_slice[1],
                ] = result
                out[
                    half * positive_index : half * (positive_index + 1),
                    image_slice[0] : image_slice[1],
                ] = result
            else:
                target[half * positive_index : half * (positive_index + 1)] = result
            return out

        patched.set_model_attn1_output_patch(nag)
        patched.disable_model_cfg1_optimization()
        return cls.outputs(MODEL=patched)


CFG_GUIDANCE_NODES = (CFGZeroStar, CFGNorm, APG, TCFG, Mahiro, NAGuidance)
