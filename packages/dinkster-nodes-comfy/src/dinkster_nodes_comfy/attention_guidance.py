"""Attention-guidance model patches executed by the inference sampler."""

from __future__ import annotations

import importlib
import math
import re
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_STRING,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    initialize_inference_process,
)

_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_MODEL = TypeExpr.concrete(MODEL_TYPE)
_STRING = TypeExpr.concrete(CORE_STRING)


def _inference_modules() -> tuple[Any, Any, Any, Any]:
    initialize_inference_process()
    return (
        importlib.import_module("torch"),
        importlib.import_module("dinkster_inference.ldm.modules.attention"),
        importlib.import_module("dinkster_inference.model_patcher"),
        importlib.import_module("dinkster_inference.samplers"),
    )


def _attention_basic_with_sim(
    q: Any, k: Any, v: Any, heads: int, attn_precision: Any
) -> tuple[Any, Any]:
    torch, _, _, _ = _inference_modules()
    batch, _, dimension = q.shape
    dimension //= heads
    scale = dimension**-0.5
    q, k, v = (
        tensor.unsqueeze(3)
        .reshape(batch, -1, heads, dimension)
        .permute(0, 2, 1, 3)
        .reshape(batch * heads, -1, dimension)
        .contiguous()
        for tensor in (q, k, v)
    )
    if attn_precision == torch.float32:
        similarity = torch.einsum("b i d, b j d -> b i j", q.float(), k.float()) * scale
    else:
        similarity = torch.einsum("b i d, b j d -> b i j", q, k) * scale
    similarity = similarity.softmax(dim=-1)
    output = torch.einsum("b i j, b j d -> b i d", similarity.to(v.dtype), v)
    output = (
        output.unsqueeze(0)
        .reshape(batch, heads, -1, dimension)
        .permute(0, 2, 1, 3)
        .reshape(batch, -1, heads * dimension)
    )
    return output, similarity


def _gaussian_blur_2d(image: Any, kernel_size: int, sigma: float) -> Any:
    torch, _, _, _ = _inference_modules()
    half_size = (kernel_size - 1) * 0.5
    coordinate = torch.linspace(-half_size, half_size, steps=kernel_size)
    density = torch.exp(-0.5 * (coordinate / sigma).pow(2))
    kernel = density / density.sum()
    kernel = kernel.to(device=image.device, dtype=image.dtype)
    kernel = torch.mm(kernel[:, None], kernel[None, :])
    kernel = kernel.expand(image.shape[-3], 1, kernel.shape[0], kernel.shape[1])
    functional = importlib.import_module("torch.nn.functional")
    image = functional.pad(image, [kernel_size // 2] * 4, mode="reflect")
    return functional.conv2d(image, kernel, groups=image.shape[-3])


def _create_blur_map(x0: Any, attention: Any, sigma: float, threshold: float) -> Any:
    _, _, _, _ = _inference_modules()
    _, height_width_1, height_width_2 = attention.shape
    batch, _, latent_height, latent_width = x0.shape
    attention = attention.reshape(batch, -1, height_width_1, height_width_2)
    mask = attention.mean(1, keepdim=False).sum(1, keepdim=False) > threshold
    total = mask.shape[-1]
    x = round(math.sqrt((latent_height / latent_width) * total))
    for offset in range(math.floor(math.sqrt(total) / 2)):
        for candidate in (x + offset, max(1, x - offset)):
            if total % candidate == 0:
                x = candidate
                break
        else:
            continue
        break
    y = total // x
    mask = mask.reshape(batch, x, y).unsqueeze(1).type(attention.dtype)
    functional = importlib.import_module("torch.nn.functional")
    mask = functional.interpolate(mask, (latent_height, latent_width))
    blurred = _gaussian_blur_2d(x0, kernel_size=9, sigma=sigma)
    return blurred * mask + x0 * (1 - mask)


def _layers(value: str) -> list[int]:
    return [int(layer) for layer in re.findall(r"\d+", value)]


def _skip(args: Any, extra_args: Any) -> Any:
    return args


class SelfAttentionGuidance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SelfAttentionGuidance",
            display_name="Self-Attention Guidance",
            category="experimental",
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "scale", _FLOAT, default=0.5, widget=NumberWidget(min=-2.0, max=5.0, step=0.01)
                ),
                InputSpec(
                    "blur_sigma",
                    _FLOAT,
                    default=2.0,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.1),
                ),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(cls, *, model: Any, scale: float, blur_sigma: float) -> Mapping[str, object]:
        attention_scores: Any = None

        def record_attention(q: Any, k: Any, v: Any, options: dict[str, Any]) -> Any:
            nonlocal attention_scores
            _, attention, _, _ = _inference_modules()
            heads = options["n_heads"]
            cond_or_uncond = options["cond_or_uncond"]
            batch = q.shape[0] // len(cond_or_uncond)
            if 1 not in cond_or_uncond:
                return attention.optimized_attention_for_device(q.device)(
                    q, k, v, heads=heads, attn_precision=options["attn_precision"]
                )
            output, similarity = _attention_basic_with_sim(
                q, k, v, heads=heads, attn_precision=options["attn_precision"]
            )
            uncond_index = cond_or_uncond.index(1)
            slices = heads * batch
            attention_scores = similarity[slices * uncond_index : slices * (uncond_index + 1)]
            return output

        def post_cfg(args: dict[str, Any]) -> Any:
            _, _, _, samplers = _inference_modules()
            result = args["denoised"]
            if min(result.shape[2:]) <= 4:
                return result
            uncond = args["uncond"]
            uncond_prediction = args["uncond_denoised"]
            degraded = _create_blur_map(uncond_prediction, attention_scores, blur_sigma, 1.0)
            degraded_noised = degraded + args["input"] - uncond_prediction
            sag = samplers.calc_cond_batch(
                args["model"], [uncond], degraded_noised, args["sigma"], args["model_options"]
            )[0]
            return result + (degraded - sag) * scale

        patched = model.clone()
        patched.set_model_sampler_post_cfg_function(post_cfg, disable_cfg1_optimization=True)
        patched.set_model_attn1_replace(record_attention, "middle", 0, 0)
        return cls.outputs(MODEL=patched)


class PerturbedAttentionGuidance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="PerturbedAttentionGuidance",
            category="model/patch/unet",
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "scale",
                    _FLOAT,
                    default=3.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01, round=0.01),
                ),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(cls, *, model: Any, scale: float) -> Mapping[str, object]:
        def perturbed_attention(
            q: Any, k: Any, v: Any, options: dict[str, Any], mask: Any = None
        ) -> Any:
            return v

        def post_cfg(args: dict[str, Any]) -> Any:
            _, _, model_patcher, samplers = _inference_modules()
            result = args["denoised"]
            if scale == 0:
                return result
            options = model_patcher.set_model_options_patch_replace(
                args["model_options"].copy(), perturbed_attention, "attn1", "middle", 0
            )
            pag = samplers.calc_cond_batch(
                args["model"], [args["cond"]], args["input"], args["sigma"], options
            )[0]
            return result + (args["cond_denoised"] - pag) * scale

        patched = model.clone()
        patched.set_model_sampler_post_cfg_function(post_cfg)
        return cls.outputs(MODEL=patched)


class SkipLayerGuidanceDiT(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SkipLayerGuidanceDiT",
            category="advanced/guidance",
            description=(
                "Generic version of SkipLayerGuidance node that can be used on every DiT model."
            ),
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "double_layers",
                    _STRING,
                    default="7, 8, 9",
                    advanced=True,
                    widget=StringWidget(multiline=False),
                ),
                InputSpec(
                    "single_layers",
                    _STRING,
                    default="7, 8, 9",
                    advanced=True,
                    widget=StringWidget(multiline=False),
                ),
                InputSpec(
                    "scale", _FLOAT, default=3.0, widget=NumberWidget(min=0.0, max=10.0, step=0.1)
                ),
                InputSpec(
                    "start_percent",
                    _FLOAT,
                    default=0.01,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec(
                    "end_percent",
                    _FLOAT,
                    default=0.15,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec(
                    "rescaling_scale",
                    _FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        double_layers: str,
        single_layers: str,
        scale: float,
        start_percent: float,
        end_percent: float,
        rescaling_scale: float,
    ) -> Mapping[str, object]:
        double = _layers(double_layers)
        single = _layers(single_layers)
        if not double and not single:
            return cls.outputs(MODEL=model)
        sampling = model.get_model_object("model_sampling")
        sigma_start = sampling.percent_to_sigma(start_percent)
        sigma_end = sampling.percent_to_sigma(end_percent)

        def post_cfg(args: dict[str, Any]) -> Any:
            _, _, model_patcher, samplers = _inference_modules()
            result = args["denoised"]
            sigma = args["sigma"]
            if scale <= 0 or not sigma_end <= sigma[0].item() <= sigma_start:
                return result
            options = args["model_options"].copy()
            for layer in double:
                options = model_patcher.set_model_options_patch_replace(
                    options, _skip, "dit", "double_block", layer
                )
            for layer in single:
                options = model_patcher.set_model_options_patch_replace(
                    options, _skip, "dit", "single_block", layer
                )
            skipped = samplers.calc_cond_batch(
                args["model"], [args["cond"]], args["input"], sigma, options
            )[0]
            result = result + (args["cond_denoised"] - skipped) * scale
            if rescaling_scale != 0:
                factor = args["cond_denoised"].std() / result.std()
                result *= rescaling_scale * factor + (1 - rescaling_scale)
            return result

        patched = model.clone()
        patched.set_model_sampler_post_cfg_function(post_cfg)
        return cls.outputs(MODEL=patched)


class SkipLayerGuidanceDiTSimple(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SkipLayerGuidanceDiTSimple",
            category="advanced/guidance",
            description=(
                "Simple version of the SkipLayerGuidanceDiT node that only modifies the uncond "
                "pass."
            ),
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "double_layers",
                    _STRING,
                    default="7, 8, 9",
                    advanced=True,
                    widget=StringWidget(multiline=False),
                ),
                InputSpec(
                    "single_layers",
                    _STRING,
                    default="7, 8, 9",
                    advanced=True,
                    widget=StringWidget(multiline=False),
                ),
                InputSpec(
                    "start_percent",
                    _FLOAT,
                    default=0.0,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec(
                    "end_percent",
                    _FLOAT,
                    default=1.0,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        double_layers: str,
        single_layers: str,
        start_percent: float,
        end_percent: float,
    ) -> Mapping[str, object]:
        double = _layers(double_layers)
        single = _layers(single_layers)
        if not double and not single:
            return cls.outputs(MODEL=model)
        sampling = model.get_model_object("model_sampling")
        sigma_start = sampling.percent_to_sigma(start_percent)
        sigma_end = sampling.percent_to_sigma(end_percent)

        def calc_cond_batch(args: dict[str, Any]) -> Any:
            _, _, model_patcher, samplers = _inference_modules()
            cond, uncond = args["conds"]
            sigma = args["sigma"]
            options = args["model_options"]
            if uncond is None or not sigma_end <= sigma[0].item() <= sigma_start:
                return samplers.calc_cond_batch(
                    args["model"], args["conds"], args["input"], sigma, options
                )
            skipped_options = options.copy()
            for layer in double:
                skipped_options = model_patcher.set_model_options_patch_replace(
                    skipped_options, _skip, "dit", "double_block", layer
                )
            for layer in single:
                skipped_options = model_patcher.set_model_options_patch_replace(
                    skipped_options, _skip, "dit", "single_block", layer
                )
            cond_output = samplers.calc_cond_batch(
                args["model"], [cond, None], args["input"], sigma, options
            )[0]
            uncond_output = samplers.calc_cond_batch(
                args["model"], [None, uncond], args["input"], sigma, skipped_options
            )[1]
            return [cond_output, uncond_output]

        patched = model.clone()
        patched.set_model_sampler_calc_cond_batch_function(calc_cond_batch)
        return cls.outputs(MODEL=patched)


ATTENTION_GUIDANCE_NODES = (
    SelfAttentionGuidance,
    PerturbedAttentionGuidance,
    SkipLayerGuidanceDiT,
    SkipLayerGuidanceDiTSimple,
)
