"""Exact pinned-source trajectory proofs for torch-native res_multistep."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_inference import Parameterization, SamplerInfo, SolverStateEvent, StepEvent
from dinkster_inference_torch.solvers import torch_sampler_registry
from golden_files import assert_reference_tensor, assert_reference_values, load_platform_golden

GOLDEN = load_platform_golden(
    Path(__file__).parent / "goldens/res_multistep_trajectory_goldens.json",
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


@pytest.mark.parametrize(
    ("case_name", "parameterization"),
    [
        ("eps", Parameterization.EPS),
        ("flow", Parameterization.FLOW),
        ("flow", Parameterization.IMAGE_TO_IMAGE_FLOW),
    ],
)
def test_torch_res_multistep_matches_every_executed_reference_seam(
    case_name: str,
    parameterization: Parameterization,
) -> None:
    case: dict[str, Any] = GOLDEN["cases"][case_name]
    descriptor = torch_sampler_registry().get("dinkster.res_multistep")
    assert descriptor is not None
    model = _Model()
    initial = _tensor(GOLDEN["initial"])
    events: list[StepEvent] = []
    states: list[SolverStateEvent[object]] = []
    result = descriptor.build()(
        model,
        initial,
        tuple(case["sigmas"]),
        SamplerInfo(parameterization, on_state=states.append),
        on_step=events.append,
    )

    assert_reference_values(model.sigmas, case["model_sigmas"])
    assert len(model.calls) == len(case["model_calls"])
    for actual, expected in zip(model.calls, case["model_calls"], strict=True):
        assert_reference_tensor(actual, _tensor(expected))
    for actual, expected in zip(states, case["steps"], strict=True):
        assert type(actual.current) is torch.Tensor
        assert_reference_tensor(actual.current, _tensor(expected))
    assert_reference_tensor(result, _tensor(case["final"]))
    assert [(event.step, event.total) for event in events] == [
        (index, len(case["sigmas"]) - 1) for index in range(len(case["sigmas"]) - 1)
    ]
    assert_reference_values([event.sigma for event in events], case["sigmas"][:-1])


def test_res_multistep_golden_comes_from_the_pinned_reference() -> None:
    assert GOLDEN["_meta"]["reference_commit"] == ("b78cec879b9460d5cb25228a83a942fb78d2cd24")
    assert GOLDEN["_meta"]["source_path"] == "comfy/k_diffusion/sampling.py"
