"""Native ComfyUI noise source and latent noising nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from dinkster_api.v1 import (
    CORE_INT,
    LATENT_TYPE,
    MODEL_TYPE,
    NOISE_TYPE,
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

_INT = TypeExpr.concrete(CORE_INT)
_MODEL = TypeExpr.concrete(MODEL_TYPE)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_NOISE = TypeExpr.concrete(NOISE_TYPE)
_SIGMAS = TypeExpr.concrete(SIGMAS_TYPE)
_CATEGORY = "model/sampling/noise"
_MAX_SEED = 0xFFFFFFFFFFFFFFFF


@dataclass(frozen=True)
class _RandomNoise:
    seed: int

    def generate_noise(self, latent: object) -> Any:
        materialized = materialize_latent(latent)
        if not isinstance(materialized, Mapping):
            raise TypeError("latent must be a mapping")
        source = cast(Mapping[str, object], materialized)
        samples = source.get("samples")
        if samples is None:
            raise ValueError("latent is missing samples")
        initialize_inference_process()
        sample = importlib.import_module("dinkster_inference.sample")
        return sample.prepare_noise(samples, self.seed, source.get("batch_index"))


@dataclass(frozen=True)
class _EmptyNoise:
    seed: int = 0

    def generate_noise(self, latent: object) -> Any:
        materialized = materialize_latent(latent)
        if not isinstance(materialized, Mapping):
            raise TypeError("latent must be a mapping")
        source = cast(Mapping[str, object], materialized)
        samples = source.get("samples")
        if samples is None:
            raise ValueError("latent is missing samples")
        initialize_inference_process()
        sample = importlib.import_module("dinkster_inference.sample")
        return sample.prepare_empty_noise(samples)


class RandomNoise(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="RandomNoise",
            category=_CATEGORY,
            inputs=(
                InputSpec(
                    "noise_seed",
                    _INT,
                    default=0,
                    widget=NumberWidget(min=0, max=_MAX_SEED, step=1),
                ),
            ),
            outputs=(OutputSpec("NOISE", _NOISE),),
        )

    @classmethod
    def execute(cls, *, noise_seed: int) -> Mapping[str, object]:
        return cls.outputs(NOISE=_RandomNoise(noise_seed))


class DisableNoise(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="DisableNoise",
            search_terms=("zero noise",),
            category=_CATEGORY,
            inputs=(),
            outputs=(OutputSpec("NOISE", _NOISE),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(NOISE=_EmptyNoise())


class AddNoise(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="AddNoise",
            category=_CATEGORY,
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec("noise", _NOISE),
                InputSpec("sigmas", _SIGMAS),
                InputSpec("latent_image", _LATENT),
            ),
            outputs=(OutputSpec("LATENT", _LATENT),),
        )

    @classmethod
    def execute(
        cls, *, model: Any, noise: Any, sigmas: object, latent_image: object
    ) -> Mapping[str, object]:
        materialized_sigmas: Any = materialize_latent(sigmas)
        if len(materialized_sigmas) == 0:
            return cls.outputs(LATENT=latent_image)
        materialized = materialize_latent(latent_image)
        if not isinstance(materialized, Mapping):
            raise TypeError("latent_image must be a mapping")
        source = cast(Mapping[str, object], materialized)
        samples = source.get("samples")
        if samples is None:
            raise ValueError("latent_image is missing samples")
        initialize_inference_process()
        torch = importlib.import_module("torch")
        with torch.inference_mode():
            noisy = noise.generate_noise(source)
            model_sampling = model.get_model_object("model_sampling")
            process_latent_out = model.get_model_object("process_latent_out")
            process_latent_in = model.get_model_object("process_latent_in")
            scale = (
                torch.abs(materialized_sigmas[0] - materialized_sigmas[-1])
                if len(materialized_sigmas) > 1
                else materialized_sigmas[0]
            )
            input_samples = samples
            if torch.count_nonzero(input_samples) > 0:
                input_samples = process_latent_in(input_samples)
            noisy = model_sampling.noise_scaling(scale, noisy, input_samples)
            noisy = torch.nan_to_num(process_latent_out(noisy), nan=0.0, posinf=0.0, neginf=0.0)
        output = dict[str, object](source)
        output["samples"] = noisy
        return cls.outputs(LATENT=output)


NOISE_NODES = (RandomNoise, DisableNoise, AddNoise)
