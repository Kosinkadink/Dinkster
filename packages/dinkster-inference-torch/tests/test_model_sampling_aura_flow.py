"""AuraFlow multiplier-one schedules and executed source solver replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_inference import (
    FlowSigmas,
    Parameterization,
    SamplerInfo,
    SolverStateEvent,
    StepEvent,
    sampling_sigmas,
)
from dinkster_inference.sampling import run_step_begin_solver
from dinkster_inference_torch import BrownianTreeNoise
from dinkster_inference_torch.sampling_execution import (
    brownian_step_noise,
    build_custom_sampling_schedule,
)
from dinkster_inference_torch.schedules import torch_scheduler_registry
from dinkster_inference_torch.solvers import torch_sampler_registry
from golden_files import (
    assert_reference_schedule,
    assert_reference_tensor,
    assert_reference_values,
    load_platform_golden,
)

GOLDEN = load_platform_golden(
    Path(__file__).parent / "goldens" / "model_sampling_aura_flow.json",
    allow_portable_fallback=True,
)
CASES = GOLDEN["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: f"{case['shift']}-{case['steps']}")
@pytest.mark.parametrize("scheduler_id", ("simple", "normal", "beta", "ddim_uniform", "karras"))
@pytest.mark.parametrize("sampler_id", ("euler", "dpmpp_2m_sde"))
def test_aura_flow_matches_executed_source(
    case: dict[str, Any], scheduler_id: str, sampler_id: str
) -> None:
    space = FlowSigmas(shift=case["shift"], multiplier=1.0)
    expected = case["schedules"][scheduler_id]
    scheduler = torch_scheduler_registry().get(f"dinkster.{scheduler_id}")
    sampler = torch_sampler_registry().get(f"dinkster.{sampler_id}")
    assert scheduler is not None and sampler is not None
    sigmas = sampling_sigmas(scheduler, space, case["steps"])
    assert_reference_schedule(sigmas, expected["sigmas"])
    schedule = build_custom_sampling_schedule(sigmas, space, sampler, flow=True)
    if sampler.requires_snr_offset:
        # Torch 2.13 recomputation differs from the recorded float32 values by
        # at most 2.900e-8 relative across every AuraFlow and Flux case (#1538).
        assert_reference_schedule(schedule.sigmas, expected["snr_sigmas"], rel=1e-7)
    latent = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2) / 8
    noise = brownian_step_noise(sampler, schedule, latent, seed=23)
    if sampler_id == "dpmpp_2m_sde" and noise is None:
        # Unmoved schedules let the engine construct its own tree; this
        # direct solver replay supplies the same bounds explicitly.
        assert schedule.sigmas == schedule.pre_offset
        noise = BrownianTreeNoise(
            latent, min(sigma for sigma in sigmas if sigma > 0), max(sigmas), seed=23, cpu=True
        )
    actual = sampler.build()(
        lambda x, sigma: x * 0.25,
        latent,
        schedule.sigmas,
        SamplerInfo(parameterization=Parameterization.FLOW, seed=23),
        noise=noise,
    )
    assert_reference_tensor(actual, torch.tensor(expected["outputs"][sampler_id]))


@pytest.mark.parametrize(
    ("case", "sampler_id"),
    [
        (case, sampler_id)
        for case in GOLDEN["solver_cases"]
        for sampler_id in (
            "dpmpp_2m_sde",
            "dpmpp_2m_sde_gpu",
            "dpmpp_2m_sde_heun",
            "dpmpp_2m_sde_heun_gpu",
        )
        if "_heun" not in sampler_id or case["solver_type"] == "heun"
    ],
)
def test_dpmpp_2m_sde_trajectory(case: dict[str, Any], sampler_id: str) -> None:
    sampler = torch_sampler_registry().get(f"dinkster.{sampler_id}")
    assert sampler is not None
    options = {key: case[key] for key in ("eta", "s_noise")}
    if "_heun" not in sampler_id:
        options["solver_type"] = case["solver_type"]
    solver = sampler.build(**options)
    latent = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2) / 8
    noise_bounds: list[tuple[float, float]] = []
    states: list[SolverStateEvent[object]] = []
    steps: list[StepEvent] = []
    begins: list[int] = []

    def noise(sigma_from: float, sigma_to: float) -> torch.Tensor:
        noise_bounds.append((sigma_from, sigma_to))
        return torch.full_like(latent, 0.125 * len(noise_bounds))

    result = run_step_begin_solver(
        solver,
        lambda x, sigma: x * 0.25,
        latent,
        case["sigmas"],
        SamplerInfo(
            Parameterization.FLOW if case["flow"] else Parameterization.EPS,
            noise_scale=case["noise_scale"],
            on_state=states.append,
        ),
        noise=noise,
        on_step=steps.append,
        on_step_begin=begins.append,
    )
    assert len(noise_bounds) == len(case["noise_bounds"])
    assert_reference_values(
        [value for bounds in noise_bounds for value in bounds],
        [value for bounds in case["noise_bounds"] for value in bounds],
    )
    assert begins == list(range(len(case["sigmas"]) - 1))
    assert [(step.step, step.total) for step in steps] == [
        (index, len(begins)) for index in range(len(case["sigmas"]) - 1)
    ]
    assert_reference_values([step.sigma for step in steps], case["sigmas"][:-1])
    for state, expected in zip(states, case["states"], strict=True):
        assert isinstance(state.current, torch.Tensor)
        assert_reference_tensor(state.current, torch.tensor(expected))
    assert_reference_tensor(result, torch.tensor(case["output"]))


def test_aura_flow_golden_pins_executed_source() -> None:
    assert GOLDEN["comfyui_commit"] == "15eb748b3ec5f8a0a2d470b7fb280e2d7579f916"
    assert GOLDEN["_meta"]["cpu"]
