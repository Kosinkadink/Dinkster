"""Executed SDE pairing proofs for float32-exact table schedulers.

Brownian-tree noise streams decorrelate on one-float32-ulp sigma
differences, so a schedule that can feed an SDE sampler ships only with
executed golden coverage of the pairing. Each case replays the exact
reference run recorded by tools/gen_res4lyf_sde_pairing_goldens.py
(pinned ComfyUI sample_dpmpp_2m_sde over catalog beta/ddim and pinned
RES4LYF bong_tangent/beta57 sigmas with the solver's default seeded
BrownianTreeNoiseSampler)
and asserts the decorrelation-critical seams bit-exactly: schedule
sigmas, tree bounds, per-query noise draws, and the sigmas handed to
the denoiser. The latent trajectory is asserted value-close: Dinkster's
dpmpp_2m_sde computes its step coefficients in float64 scalar math
against the reference's float32 tensor math (the documented port
design proven by the root solver goldens), so per-step latents wobble
by float32 ULPs while the noise stream stays identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_inference import DiscreteSigmas, Parameterization, SamplerInfo, SolverStateEvent
from dinkster_inference_torch.brownian import BrownianTreeNoise
from dinkster_inference_torch.schedules import custom_beta_sigmas, torch_scheduler_registry
from dinkster_inference_torch.solvers import torch_sampler_registry
from golden_files import load_platform_golden

GOLDEN = load_platform_golden(Path(__file__).parent / "goldens/res4lyf_sde_pairing_goldens.json")

# Measured worst-case trajectory drift across the original four cases is
# 1.43e-6 abs (beta57, 8 steps); the catalog cases remain within that bound.
# Sigmas, tree bounds,
# and noise draws, pinning the residual to the solver's float64
# scalar coefficients against the reference's float32 tensor
# coefficients. 1e-5 gives ~7x headroom while staying four orders of
# magnitude under the 3.4e-2 brownian decorrelation this suite
# exists to catch.
TRAJECTORY_ATOL = 1e-5
PAIRING_CASES = (
    ("bong_tangent_4", "res4lyf.bong_tangent", 4, None),
    ("bong_tangent_8", "res4lyf.bong_tangent", 8, None),
    ("beta57_4", "res4lyf.beta57", 4, None),
    ("beta57_8", "res4lyf.beta57", 8, None),
    ("beta_4", "dinkster.beta", 4, None),
    ("beta_8", "dinkster.beta", 8, None),
    ("ddim_uniform_4", "dinkster.ddim_uniform", 4, None),
    ("ddim_uniform_8", "dinkster.ddim_uniform", 8, None),
    ("custom_beta_13_a2_b5", None, 13, (2.0, 5.0)),
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


class _RecordingBrownian:
    """The solver-default brownian sampler over the schedule's positive
    bounds, with every query and draw recorded."""

    def __init__(self, like: torch.Tensor, sigmas: tuple[float, ...]) -> None:
        positive = [sigma for sigma in sigmas if sigma > 0]
        self.bounds = (min(positive), max(sigmas))
        self.inner = BrownianTreeNoise(
            like, self.bounds[0], self.bounds[1], seed=GOLDEN["_meta"]["seed"], cpu=True
        )
        self.queries: list[tuple[float, float]] = []
        self.draws: list[torch.Tensor] = []

    def __call__(self, sigma_from: float, sigma_to: float) -> torch.Tensor:
        self.queries.append((sigma_from, sigma_to))
        draw = self.inner(sigma_from, sigma_to)
        self.draws.append(draw.detach().clone())
        return draw


@pytest.mark.parametrize(
    ("case_name", "scheduler_id", "steps", "custom_beta"),
    [case for case in PAIRING_CASES if case[0] in GOLDEN["cases"]],
)
def test_res4lyf_sde_pairing_matches_every_executed_reference_seam(
    case_name: str,
    scheduler_id: str | None,
    steps: int,
    custom_beta: tuple[float, float] | None,
) -> None:
    case: dict[str, Any] = GOLDEN["cases"][case_name]
    space = DiscreteSigmas.linear_beta()
    if custom_beta is None:
        assert scheduler_id is not None
        scheduler = torch_scheduler_registry().get(scheduler_id)
        assert scheduler is not None
        sigmas = scheduler.make_sigmas(steps, space)
    else:
        assert scheduler_id is None
        sigmas = custom_beta_sigmas(space, steps, *custom_beta)
    assert list(sigmas) == case["sigmas"]

    descriptor = torch_sampler_registry().get("dinkster.dpmpp_2m_sde")
    assert descriptor is not None
    model = _Model()
    initial = _tensor(GOLDEN["initial"])
    noise = _RecordingBrownian(initial, sigmas)
    assert list(noise.bounds) == case["tree_bounds"]
    states: list[SolverStateEvent[object]] = []
    result = descriptor.build()(
        model,
        initial,
        sigmas,
        SamplerInfo(Parameterization.EPS, on_state=states.append),
        noise=noise,
    )

    assert model.sigmas == case["model_sigmas"]
    assert [list(query) for query in noise.queries] == case["noise_queries"]
    for actual, expected in zip(noise.draws, case["noise_draws"], strict=True):
        assert torch.equal(actual, _tensor(expected))
    for actual, expected in zip(model.calls, case["model_calls"], strict=True):
        assert torch.allclose(actual, _tensor(expected), rtol=0.0, atol=TRAJECTORY_ATOL)
    for actual_state, expected_step in zip(states, case["steps"], strict=True):
        assert type(actual_state.current) is torch.Tensor
        assert torch.allclose(
            actual_state.current, _tensor(expected_step), rtol=0.0, atol=TRAJECTORY_ATOL
        )
    assert torch.allclose(result, _tensor(case["final"]), rtol=0.0, atol=TRAJECTORY_ATOL)


def test_res4lyf_sde_pairing_golden_comes_from_the_pinned_references() -> None:
    assert GOLDEN["_meta"]["comfyui_commit"] == "b78cec879b9460d5cb25228a83a942fb78d2cd24"
    assert GOLDEN["_meta"]["res4lyf_commit"] == "26036f647ca15d3048a193daf99a40cecfc3820d"
    assert GOLDEN["_meta"]["sampler"] == "sample_dpmpp_2m_sde"
    if "platform" not in GOLDEN["_meta"]:
        assert set(GOLDEN["cases"]) == {case[0] for case in PAIRING_CASES}
