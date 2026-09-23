"""Exact pinned-source trajectory proofs for torch-native ER-SDE."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_inference import Parameterization, SamplerInfo, SolverStateEvent, StepEvent
from dinkster_inference_torch.solvers import torch_sampler_registry
from golden_files import cpu_identity, load_platform_golden

GOLDEN = load_platform_golden(Path(__file__).parent / "goldens/er_sde_trajectory_goldens.json")


def _tensor(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 2, 2)


def _flow_scalar_diagnostics(sigmas: list[float]) -> list[dict[str, float]]:
    schedule = torch.tensor(sigmas, dtype=torch.float32)
    er_lambdas = schedule.logit().neg().neg().exp()
    result = []
    for sigma, er_lambda in zip(schedule[:-1], er_lambdas[:-1], strict=True):
        scale = er_lambda * ((er_lambda**0.3).exp() + 10.0)
        result.append(
            {
                "alpha": float(sigma / er_lambda),
                "lambda": float(er_lambda),
                "scale": float(scale),
                "sigma": float(sigma),
            }
        )
    return result


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
def test_torch_er_sde_matches_every_executed_reference_seam(
    case_name: str,
    parameterization: Parameterization,
) -> None:
    case: dict[str, Any] = GOLDEN["cases"][case_name]
    descriptor = torch_sampler_registry().get("dinkster.er_sde")
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

    assert model.sigmas == case["model_sigmas"]
    assert len(model.calls) == len(case["model_calls"])
    flow_scalars = _flow_scalar_diagnostics(case["sigmas"]) if case_name == "flow" else []
    for index, (actual, expected) in enumerate(
        zip(model.calls, case["model_calls"], strict=True)
    ):
        expected_tensor = _tensor(expected)
        assert torch.equal(actual, expected_tensor), (
            f"model_calls[{index}] differs; cpu={cpu_identity()}; torch={torch.__version__}; "
            f"actual={actual.flatten().tolist()}; expected={expected_tensor.flatten().tolist()}; "
            f"flow_scalars={flow_scalars}"
        )
    assert noise.bounds == [tuple(bounds) for bounds in case["noise_bounds"]]
    for index, (actual, expected) in enumerate(zip(states, case["steps"], strict=True)):
        assert type(actual.current) is torch.Tensor
        expected_tensor = _tensor(expected)
        assert torch.equal(actual.current, expected_tensor), (
            f"steps[{index}] differs; actual={actual.current.flatten().tolist()}; "
            f"expected={expected_tensor.flatten().tolist()}"
        )
    assert torch.equal(result, _tensor(case["final"]))
    assert [(event.step, event.total, event.sigma) for event in events] == [
        (index, len(case["sigmas"]) - 1, sigma) for index, sigma in enumerate(case["sigmas"][:-1])
    ]


def test_er_sde_golden_comes_from_the_pinned_reference() -> None:
    assert GOLDEN["_meta"]["reference_commit"] == ("b78cec879b9460d5cb25228a83a942fb78d2cd24")
    assert GOLDEN["_meta"]["source_path"] == "comfy/k_diffusion/sampling.py"
