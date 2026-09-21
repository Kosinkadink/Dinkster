# pyright: basic
"""A new family needs registration data and a Denoiser adapter only."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from dinkster_inference import (
    FLOAT32,
    ComponentWiring,
    Conditioning,
    CustomSamplingRequest,
    DetectionEvidence,
    FamilyRegistry,
    FlowSigmas,
    GuidanceRole,
    LatentDescriptor,
    ModelFamily,
    Parameterization,
    SamplingDescriptor,
    TensorGeometry,
    WeightEntry,
    WeightSource,
)
from dinkster_inference_torch.denoise import prepare_noise
from dinkster_inference_torch.sampling_execution import (
    SamplingAdapterContext,
    SamplingDenoiserAdapter,
    SamplingDenoiserExecution,
    SamplingExecutionRegistration,
    SingleStreamLatentAdapter,
    build_sampling_schedule,
    resolve_sampling,
    sampling_execution,
)
from dinkster_inference_torch.sampling_runtime import SingleStreamSamplingRuntime
from dinkster_inference_torch.schedules import torch_scheduler_registry
from dinkster_inference_torch.solvers import torch_sampler_registry

TOY_FAMILY_ID = "test.toy-image"


@dataclass(frozen=True)
class ToyDetector:
    def detect(self, source: WeightSource) -> DetectionEvidence | None:
        key = "toy.denoiser.weight"
        if key not in source.keys():
            return None
        return DetectionEvidence(TOY_FAMILY_ID, (key,), {"channels": 4})


class HeaderSource:
    def __init__(self, keys: Sequence[str]) -> None:
        self._keys = tuple(keys)

    def keys(self) -> Sequence[str]:
        return self._keys

    def entry(self, key: str) -> WeightEntry:
        if key not in self._keys:
            raise KeyError(key)
        return WeightEntry(key, TensorGeometry((4, 4), FLOAT32), 0, 64)

    def metadata(self) -> Mapping[str, str]:
        return {}


SIGMA_SPACE = FlowSigmas()
TOY_FAMILY = ModelFamily(
    id=TOY_FAMILY_ID,
    display_name="Toy Image",
    detector=ToyDetector(),
    specificity=100,
    latent=LatentDescriptor(channels=4, scale_factor=1.0, shift_factor=0.0),
    sampling=SamplingDescriptor(
        Parameterization.FLOW,
        sigma_min=SIGMA_SPACE.sigma_min,
        sigma_max=SIGMA_SPACE.sigma_max,
    ),
    wiring=ComponentWiring(),
    supported_dtypes=frozenset({FLOAT32}),
)


class ToyDenoiserAdapter:
    evaluator_identity = "test.toy-image.denoiser.v1"

    @staticmethod
    def prepare_conditioning(value: object, _role: GuidanceRole) -> object:
        return value

    @staticmethod
    def evaluate_conditioning(value: torch.Tensor, _sigma: float, context: object) -> torch.Tensor:
        conditioning = cast("Conditioning[torch.Tensor]", context)
        return value * 0.25 + conditioning.embeddings.mean()

    @staticmethod
    def batchable(_conditions: tuple[object, ...]) -> bool:
        return True

    def evaluate_batch(
        self,
        value: torch.Tensor,
        sigma: float,
        conditions: tuple[object, ...],
        _context: object | None = None,
    ) -> tuple[torch.Tensor, ...]:
        return tuple(self.evaluate_conditioning(value, sigma, item) for item in conditions)

    evaluate_conditioning_batch = evaluate_batch


def toy_denoiser(
    _runtime: object,
    _dtype: torch.dtype,
    _context: SamplingAdapterContext,
) -> SamplingDenoiserExecution:
    return SamplingDenoiserExecution(cast("SamplingDenoiserAdapter", ToyDenoiserAdapter()))


class ToyRuntime(SingleStreamSamplingRuntime):
    sampling_execution_registration = SamplingExecutionRegistration(
        latent=SingleStreamLatentAdapter(lambda _latent: None),
        denoiser=toy_denoiser,
        device=lambda _runtime: torch.device("cpu"),
        compute_dtype=lambda _runtime: torch.float32,
        flow=True,
    )

    def __init__(self) -> None:
        self._samplers = torch_sampler_registry()
        self._schedulers = torch_scheduler_registry()
        self._guidance = None

    @property
    def family(self) -> ModelFamily:
        return TOY_FAMILY

    def _sampling_sigma_space(self, sampling_shift: float | None) -> FlowSigmas:
        assert sampling_shift is None
        return SIGMA_SPACE

    sample_custom = sampling_execution


def test_toy_family_detects_from_registration_data() -> None:
    registry = FamilyRegistry()
    registry.register(TOY_FAMILY)

    detected = registry.detect(HeaderSource(("toy.denoiser.weight",)))

    assert detected.best is not None
    assert detected.best.family_id == TOY_FAMILY_ID
    assert detected.best.fields == {"channels": 4}


def test_toy_family_runs_identically_through_both_sampler_surfaces_on_cpu() -> None:
    runtime = ToyRuntime()
    latent = torch.zeros((1, 4, 2, 3), dtype=torch.float32)
    conditioning = Conditioning(torch.arange(12, dtype=torch.float32).reshape(1, 3, 4))
    seed = 123
    sampler, scheduler = resolve_sampling(
        runtime._samplers,
        runtime._schedulers,
        "dinkster.euler",
        "dinkster.simple",
        error=ValueError,
    )
    schedule = build_sampling_schedule(
        scheduler,
        SIGMA_SPACE,
        sampler,
        3,
        denoise=1.0,
        flow=True,
    )
    request = CustomSamplingRequest(sampler, (), schedule.pre_offset)
    noise = prepare_noise(latent, seed)

    custom = runtime.sample_custom(
        latent,
        noise=noise,
        cond=conditioning,
        cfg=None,
        request=request,
        seed=seed,
    ).output
    ksampler = runtime.sample(
        latent,
        cond=conditioning,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=3,
        denoise=1.0,
        seed=seed,
    )

    assert torch.equal(custom, ksampler)
    assert torch.count_nonzero(ksampler)
    assert ksampler.device.type == "cpu"
