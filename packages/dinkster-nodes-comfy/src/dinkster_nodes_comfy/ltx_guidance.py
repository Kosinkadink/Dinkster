"""LTXV reference identity, temporal and cross-modal guidance patches."""

from __future__ import annotations

import importlib
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

from dinkster_api.v1 import (
    AUDIO_TYPE,
    CONDITIONING_TYPE,
    CORE_FLOAT,
    CORE_STRING,
    MODEL_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    initialize_inference_process,
    materialize_audio,
)

from .conditioning import conditioning_set_values

_MODEL = TypeExpr.concrete(MODEL_TYPE)
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_WINDOW = (
    InputSpec(
        "start_percent",
        TypeExpr.concrete(CORE_FLOAT),
        default=0.0,
        widget=NumberWidget(min=0.0, max=1.0, step=0.001),
        advanced=True,
    ),
    InputSpec(
        "end_percent",
        TypeExpr.concrete(CORE_FLOAT),
        default=1.0,
        widget=NumberWidget(min=0.0, max=1.0, step=0.001),
        advanced=True,
    ),
)


class LTXVReferenceAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LTXVReferenceAudio",
            display_name="LTXV Reference Audio (ID-LoRA)",
            category="model/conditioning/ltxv",
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("reference_audio", TypeExpr.concrete(AUDIO_TYPE)),
                InputSpec("audio_vae", TypeExpr.concrete(VAE_TYPE), display_name="Audio VAE"),
                InputSpec(
                    "identity_guidance_scale",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=3.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01, round=0.01),
                ),
                *_WINDOW,
            ),
            outputs=(
                OutputSpec("MODEL", _MODEL),
                OutputSpec("positive", _CONDITIONING),
                OutputSpec("negative", _CONDITIONING),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        reference_audio: object,
        audio_vae: Any,
        identity_guidance_scale: float,
        start_percent: float,
        end_percent: float,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        audio = importlib.import_module("dinkster_inference.audio")
        samplers = importlib.import_module("dinkster_inference.samplers")
        source = cast("Mapping[str, Any]", materialize_audio(reference_audio))
        sample_rate = source["sample_rate"]
        target_rate = getattr(audio_vae, "audio_sample_rate", 44100)
        waveform = source["waveform"]
        if target_rate != sample_rate:
            waveform = audio.resample(waveform, sample_rate, target_rate)
        encoded = audio_vae.encode(waveform.movedim(1, -1))
        batch, channels, frames, frequencies = encoded.shape
        reference = {
            "tokens": encoded.permute(0, 2, 1, 3).reshape(batch, frames, channels * frequencies)
        }
        positive = conditioning_set_values(positive, {"ref_audio": reference})
        negative = conditioning_set_values(negative, {"ref_audio": reference})
        patched = model.clone()
        sampling = patched.get_model_object("model_sampling")
        sigma_start = sampling.percent_to_sigma(start_percent)
        sigma_end = sampling.percent_to_sigma(end_percent)

        def post_cfg(args: Mapping[str, Any]) -> Any:
            if identity_guidance_scale == 0:
                return args["denoised"]
            sigma = args["sigma"]
            value = sigma[0].item()
            if value > sigma_start or value < sigma_end:
                return args["denoised"]
            without_reference: list[dict[str, Any]] = []
            for entry in args["cond"]:
                copied = entry.copy()
                conditions = copied.get("model_conds", {}).copy()
                conditions.pop("ref_audio", None)
                copied["model_conds"] = conditions
                without_reference.append(copied)
            (prediction,) = samplers.calc_cond_batch(
                args["model"],
                [without_reference],
                args["input"],
                sigma,
                args["model_options"].copy(),
            )
            return args["denoised"] + (args["cond_denoised"] - prediction) * identity_guidance_scale

        patched.set_model_sampler_post_cfg_function(post_cfg)
        return cls.outputs(MODEL=patched, positive=positive, negative=negative)


class LTXVSpatioTemporalGuidance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LTXVSpatioTemporalGuidance",
            display_name="LTXV Spatio-Temporal Guidance (STG)",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "scale",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01, round=0.01),
                ),
                InputSpec(
                    "blocks",
                    TypeExpr.concrete(CORE_STRING),
                    default="29",
                    widget=StringWidget(multiline=False),
                ),
                *_WINDOW,
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        scale: float,
        blocks: str,
        start_percent: float,
        end_percent: float,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        samplers = importlib.import_module("dinkster_inference.samplers")
        block_set = frozenset(int(block) for block in re.findall(r"\d+", blocks))
        patched = model.clone()
        sampling = patched.get_model_object("model_sampling")
        sigma_start = sampling.percent_to_sigma(start_percent)
        sigma_end = sampling.percent_to_sigma(end_percent)

        def post_cfg(args: Mapping[str, Any]) -> Any:
            if scale == 0 or not block_set:
                return args["denoised"]
            value = args["sigma"][0].item()
            if value > sigma_start or value < sigma_end:
                return args["denoised"]
            options = args["model_options"].copy()
            transformer = options.get("transformer_options", {}).copy()
            transformer["stg_self_attn_blocks"] = block_set
            options["transformer_options"] = transformer
            (prediction,) = samplers.calc_cond_batch(
                args["model"],
                [args["cond"]],
                args["input"],
                args["sigma"],
                options,
            )
            return args["denoised"] + (args["cond_denoised"] - prediction) * scale

        patched.set_model_sampler_post_cfg_function(post_cfg)
        return cls.outputs(MODEL=patched)


class LTXVModalityGuidance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LTXVModalityGuidance",
            display_name="LTXV Modality Guidance (A/V coupling)",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "modality_scale",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=3.0,
                    widget=NumberWidget(min=1.0, max=100.0, step=0.1, round=0.01),
                ),
                *_WINDOW,
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        modality_scale: float,
        start_percent: float,
        end_percent: float,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        samplers = importlib.import_module("dinkster_inference.samplers")
        patched = model.clone()
        sampling = patched.get_model_object("model_sampling")
        sigma_start = sampling.percent_to_sigma(start_percent)
        sigma_end = sampling.percent_to_sigma(end_percent)

        def post_cfg(args: Mapping[str, Any]) -> Any:
            if math.isclose(modality_scale, 1.0):
                return args["denoised"]
            value = args["sigma"][0].item()
            if value > sigma_start or value < sigma_end:
                return args["denoised"]
            options = args["model_options"].copy()
            transformer = options.get("transformer_options", {}).copy()
            transformer["a2v_cross_attn"] = False
            transformer["v2a_cross_attn"] = False
            options["transformer_options"] = transformer
            (prediction,) = samplers.calc_cond_batch(
                args["model"],
                [args["cond"]],
                args["input"],
                args["sigma"],
                options,
            )
            return args["denoised"] + (args["cond_denoised"] - prediction) * (modality_scale - 1.0)

        patched.set_model_sampler_post_cfg_function(post_cfg)
        return cls.outputs(MODEL=patched)
