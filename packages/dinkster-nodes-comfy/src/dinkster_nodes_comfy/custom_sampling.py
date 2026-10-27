"""Native custom sampler nodes using the pinned inference sampling engine."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    GUIDER_TYPE,
    LATENT_TYPE,
    MODEL_TYPE,
    NOISE_TYPE,
    SAMPLER_TYPE,
    SIGMAS_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)

from .sampling_boundary import to_public, to_runtime

_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_GUIDER = TypeExpr.concrete(GUIDER_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_MODEL = TypeExpr.concrete(MODEL_TYPE)
_NOISE = TypeExpr.concrete(NOISE_TYPE)
_SAMPLER = TypeExpr.concrete(SAMPLER_TYPE)
_SIGMAS = TypeExpr.concrete(SIGMAS_TYPE)
_MAX_SEED = 0xFFFFFFFFFFFFFFFF


def _runtime() -> Any:
    initialize_inference_process()
    return importlib.import_module("dinkster_inference.sample")


def _capture_x0(output: dict[str, object]) -> Any:
    def callback(_step: int, x0: object, _samples: object, _total_steps: int) -> None:
        output["x0"] = x0

    return callback


def _prepare_latent(
    model: Any, latent_image: object
) -> tuple[dict[str, object], object, object, object | None]:
    materialized = materialize_latent(latent_image)
    if not isinstance(materialized, Mapping):
        raise TypeError("latent_image must be a mapping")
    source = dict(cast(Mapping[str, object], materialized))
    public_samples = source.get("samples")
    if public_samples is None:
        raise ValueError("latent_image is missing samples")
    sample = _runtime()
    samples = sample.fix_empty_latent_channels(
        model,
        to_runtime(public_samples),
        source.get("downscale_ratio_spacial"),
        source.get("downscale_ratio_temporal"),
    )
    source["samples"] = samples
    return source, public_samples, samples, to_runtime(source.get("noise_mask"))


def _outputs(
    model: Any,
    latent: dict[str, object],
    public_samples: object,
    samples: object,
    x0_output: Mapping[str, object],
) -> Mapping[str, object]:
    output = dict(latent)
    output.pop("downscale_ratio_spacial", None)
    output.pop("downscale_ratio_temporal", None)
    output["samples"] = to_public(samples, public_samples)
    if "x0" not in x0_output:
        return {"output": output, "denoised_output": output}
    denoised = dict(latent)
    x0 = cast(Any, x0_output["x0"])
    if getattr(samples, "is_nested", False) and not getattr(x0, "is_nested", False):
        nested_tensor = importlib.import_module("dinkster_inference.nested_tensor")
        utils = importlib.import_module("dinkster_inference.utils")
        nested_samples = cast(Any, samples)
        x0 = nested_tensor.NestedTensor(
            utils.unpack_latents(x0, [sample.shape for sample in nested_samples.unbind()])
        )
    denoised["samples"] = to_public(model.model.process_latent_out(x0.cpu()), public_samples)
    return {"output": output, "denoised_output": denoised}


def _schema(node_type: str, inputs: tuple[InputSpec, ...]) -> NodeSchema:
    return NodeSchema(
        node_type=node_type,
        category="model/sampling/custom",
        inputs=inputs,
        outputs=(OutputSpec("output", _LATENT), OutputSpec("denoised_output", _LATENT)),
    )


class SamplerCustom(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "SamplerCustom",
            (
                InputSpec("model", _MODEL),
                InputSpec("add_noise", _BOOLEAN, default=True, advanced=True),
                InputSpec(
                    "noise_seed",
                    _INT,
                    default=0,
                    widget=NumberWidget(min=0, max=_MAX_SEED, control_after_generate="fixed"),
                ),
                InputSpec(
                    "cfg",
                    _FLOAT,
                    default=8.0,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1, round=0.01),
                ),
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("sampler", _SAMPLER),
                InputSpec("sigmas", _SIGMAS),
                InputSpec("latent_image", _LATENT),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        add_noise: bool,
        noise_seed: int,
        cfg: float,
        positive: Any,
        negative: Any,
        sampler: Any,
        sigmas: Any,
        latent_image: object,
    ) -> Mapping[str, object]:
        sample = _runtime()
        latent, public_samples, samples, noise_mask = _prepare_latent(model, latent_image)
        noise = (
            sample.prepare_noise(samples, noise_seed, latent.get("batch_index"))
            if add_noise
            else sample.prepare_empty_noise(samples)
        )
        x0_output: dict[str, object] = {}
        callback = _capture_x0(x0_output)
        sampled = sample.sample_custom(
            model,
            noise,
            cfg,
            sampler,
            materialize_latent(sigmas),
            positive,
            negative,
            samples,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=True,
            seed=noise_seed,
        )
        return cls.outputs(**_outputs(model, latent, public_samples, sampled, x0_output))


class SamplerCustomAdvanced(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "SamplerCustomAdvanced",
            (
                InputSpec("noise", _NOISE),
                InputSpec("guider", _GUIDER),
                InputSpec("sampler", _SAMPLER),
                InputSpec("sigmas", _SIGMAS),
                InputSpec("latent_image", _LATENT),
            ),
        )

    @classmethod
    def execute(
        cls, *, noise: Any, guider: Any, sampler: Any, sigmas: Any, latent_image: object
    ) -> Mapping[str, object]:
        initialize_inference_process()
        latent, public_samples, samples, noise_mask = _prepare_latent(
            guider.model_patcher, latent_image
        )
        x0_output: dict[str, object] = {}
        callback = _capture_x0(x0_output)
        sampled = guider.sample(
            to_runtime(noise.generate_noise(latent)),
            samples,
            sampler,
            materialize_latent(sigmas),
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=True,
            seed=noise.seed,
        )
        management = importlib.import_module("dinkster_inference.model_management")
        sampled = sampled.to(management.intermediate_device())
        return cls.outputs(
            **_outputs(guider.model_patcher, latent, public_samples, sampled, x0_output)
        )


CUSTOM_SAMPLING_NODES = (SamplerCustom, SamplerCustomAdvanced)
