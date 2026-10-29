"""Exact pinned-source trajectory proofs for torch-native Euler ancestral."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_inference import Parameterization, SamplerInfo, SolverStateEvent, StepEvent
from dinkster_inference_torch.solvers import euler_ancestral, torch_sampler_registry
from golden_files import assert_reference_tensor, assert_reference_values, load_platform_golden

GOLDEN = load_platform_golden(
    Path(__file__).parent / "goldens/euler_ancestral_trajectory_goldens.json",
    allow_portable_fallback=True,
)


def _tensor(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 2, 2)


class _Model:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []
        self.sigmas: list[float] = []

    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        self.calls.append(x.detach().clone())
        self.sigmas.append(sigma)
        return x * (1.0 / (1.0 + sigma)) + x.square() * (0.05 / (1.0 + sigma))


class _Noise:
    def __init__(self, like: torch.Tensor) -> None:
        self.like = like
        self.bounds: list[tuple[float, float]] = []

    def __call__(self, sigma_from: float, sigma_to: float) -> torch.Tensor:
        self.bounds.append((sigma_from, sigma_to))
        values: list[float] = GOLDEN["noise_draws"][len(self.bounds) - 1]
        return _tensor(values).to(device=self.like.device, dtype=self.like.dtype)


@pytest.mark.parametrize(
    ("case_name", "parameterization"),
    [
        ("eps", Parameterization.EPS),
        ("flow", Parameterization.FLOW),
        ("flow", Parameterization.IMAGE_TO_IMAGE_FLOW),
    ],
)
def test_torch_euler_ancestral_matches_every_executed_reference_seam(
    case_name: str,
    parameterization: Parameterization,
) -> None:
    case: dict[str, Any] = GOLDEN["cases"][case_name]
    descriptor = torch_sampler_registry().get("dinkster.euler_ancestral")
    assert descriptor is not None
    model = _Model()
    initial = _tensor(GOLDEN["initial"])
    noise = _Noise(initial)
    events: list[StepEvent] = []
    states: list[SolverStateEvent[object]] = []
    result = descriptor.build()(
        model,
        initial,
        tuple(case["sigmas"]),
        SamplerInfo(parameterization, on_state=states.append),
        noise=noise,
        on_step=events.append,
    )

    assert_reference_values(model.sigmas, case["model_sigmas"])
    assert len(model.calls) == len(case["model_calls"])
    for actual, expected in zip(model.calls, case["model_calls"], strict=True):
        assert_reference_tensor(actual, _tensor(expected))
    assert len(noise.bounds) == len(case["noise_bounds"])
    assert_reference_values(
        [value for bounds in noise.bounds for value in bounds],
        [value for bounds in case["noise_bounds"] for value in bounds],
    )
    for actual, expected in zip(states, case["steps"], strict=True):
        assert type(actual.current) is torch.Tensor
        assert_reference_tensor(actual.current, _tensor(expected))
    assert_reference_tensor(result, _tensor(case["final"]))
    assert [(event.step, event.total) for event in events] == [
        (index, len(case["sigmas"]) - 1) for index in range(len(case["sigmas"]) - 1)
    ]
    assert_reference_values([event.sigma for event in events], case["sigmas"][:-1])


def test_euler_ancestral_golden_comes_from_the_pinned_reference() -> None:
    assert GOLDEN["_meta"]["reference_commit"] == ("b78cec879b9460d5cb25228a83a942fb78d2cd24")
    assert GOLDEN["_meta"]["source_path"] == "comfy/k_diffusion/sampling.py"


@pytest.mark.parametrize(
    ("parameterization", "eta", "s_noise"),
    [
        (Parameterization.EPS, 0.0, 1.0),
        (Parameterization.FLOW, 1.0, 0.0),
    ],
)
def test_euler_ancestral_does_not_require_zero_coefficient_noise(
    parameterization: Parameterization,
    eta: float,
    s_noise: float,
) -> None:
    result = euler_ancestral(eta=eta, s_noise=s_noise)(
        _Model(),
        _tensor(GOLDEN["initial"]),
        (1.0, 0.5, 0.0),
        SamplerInfo(parameterization),
    )

    assert torch.isfinite(result).all()


def test_euler_ancestral_applies_runtime_noise_scale_to_eps() -> None:
    initial = _tensor(GOLDEN["initial"])
    sigmas = (1.0, 0.5, 0.0)
    scaled = euler_ancestral()(
        _Model(),
        initial,
        sigmas,
        SamplerInfo(Parameterization.EPS, noise_scale=0.5),
        noise=_Noise(initial),
    )
    explicit = euler_ancestral(s_noise=0.5)(
        _Model(),
        initial,
        sigmas,
        SamplerInfo(Parameterization.EPS),
        noise=_Noise(initial),
    )

    assert torch.equal(scaled, explicit)
