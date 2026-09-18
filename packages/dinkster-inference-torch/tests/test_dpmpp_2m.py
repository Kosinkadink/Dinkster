"""Exact pinned-source trajectory proof for torch-native DPM++ 2M."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from dinkster_inference import Parameterization, SamplerInfo, SolverStateEvent, StepEvent
from dinkster_inference_torch.solvers import torch_sampler_registry
from golden_files import assert_reference_tensor, assert_reference_values, load_platform_golden

GOLDEN: dict[str, Any] = load_platform_golden(
    Path(__file__).parent / "goldens/dpmpp_2m_trajectory_goldens.json",
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


def test_torch_dpmpp_2m_matches_every_executed_reference_seam() -> None:
    descriptor = torch_sampler_registry().get("dinkster.dpmpp_2m")
    assert descriptor is not None
    model = _Model()
    events: list[StepEvent] = []
    states: list[SolverStateEvent[object]] = []
    result = descriptor.build()(
        model,
        _tensor(GOLDEN["initial"]),
        tuple(GOLDEN["sigmas"]),
        SamplerInfo(Parameterization.EPS, on_state=states.append),
        on_step=events.append,
    )

    assert_reference_values(model.sigmas, GOLDEN["model_sigmas"])
    for actual, expected in zip(model.calls, GOLDEN["model_calls"], strict=True):
        assert_reference_tensor(actual, _tensor(expected))
    for actual, expected in zip(states, GOLDEN["steps"], strict=True):
        assert type(actual.current) is torch.Tensor
        assert_reference_tensor(actual.current, _tensor(expected))
    assert_reference_tensor(result, _tensor(GOLDEN["final"]))
    assert [(event.step, event.total) for event in events] == [
        (index, len(GOLDEN["sigmas"]) - 1) for index in range(len(GOLDEN["sigmas"]) - 1)
    ]
    assert_reference_values([event.sigma for event in events], GOLDEN["sigmas"][:-1])


def test_dpmpp_2m_golden_comes_from_the_pinned_reference() -> None:
    assert GOLDEN["_meta"]["reference_commit"] == ("b78cec879b9460d5cb25228a83a942fb78d2cd24")
    assert GOLDEN["_meta"]["source_path"] == "comfy/k_diffusion/sampling.py"
