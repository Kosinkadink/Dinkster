"""CPU behavior for native sigma scheduler nodes against pinned inference."""

from __future__ import annotations

import sys
from typing import Any

import torch
from dinkster_nodes_comfy.sigma_schedulers import (
    BasicScheduler,
    BetaSamplingScheduler,
    ExponentialScheduler,
    KarrasScheduler,
    LaplaceScheduler,
    PolyexponentialScheduler,
    SDTurboScheduler,
    VPScheduler,
)


class ModelSampling:
    sigmas = torch.tensor((0.0, 0.125, 0.5, 1.0, 2.0, 4.0, 8.0), dtype=torch.float32)

    def sigma(self, timesteps: torch.Tensor) -> torch.Tensor:
        return timesteps.to(torch.float32) / 1000


class Model:
    def get_model_object(self, name: str) -> ModelSampling:
        assert name == "model_sampling"
        return ModelSampling()


def _enable_cpu_runtime() -> None:
    import dinkster_inference.options

    sys.argv = ["test-sigma-schedulers", "--cpu"]
    dinkster_inference.options.args_parsing = True


def test_parameterized_schedulers_match_pinned_runtime_outputs() -> None:
    _enable_cpu_runtime()
    cases: tuple[tuple[type[Any], dict[str, float | int], tuple[float, ...]], ...] = (
        (
            KarrasScheduler,
            {"steps": 4, "sigma_max": 3.7, "sigma_min": 0.13, "rho": 3.2},
            (3.7000005, 1.6964283, 0.60366714, 0.13000005, 0.0),
        ),
        (
            ExponentialScheduler,
            {"steps": 4, "sigma_max": 3.7, "sigma_min": 0.13},
            (3.6999998, 1.2118502, 0.3969138, 0.13000001, 0.0),
        ),
        (
            PolyexponentialScheduler,
            {"steps": 4, "sigma_max": 3.7, "sigma_min": 0.13, "rho": 1.7},
            (3.7000003, 0.69804287, 0.21807629, 0.13000001, 0.0),
        ),
        (
            LaplaceScheduler,
            {"steps": 4, "sigma_max": 3.7, "sigma_min": 0.13, "mu": 0.25, "beta": 0.8},
            (3.7, 1.7759936, 0.92833745, 0.13),
        ),
        (
            VPScheduler,
            {"steps": 4, "beta_d": 13.2, "beta_min": 0.37, "eps_s": 0.017},
            (32.607094, 4.9330225, 1.2199332, 0.09072536, 0.0),
        ),
    )
    for scheduler, inputs, expected in cases:
        sigmas = scheduler.execute(**inputs)["SIGMAS"]
        assert sigmas.dtype == torch.float32
        torch.testing.assert_close(sigmas, torch.tensor(expected, dtype=torch.float32))


def test_model_schedulers_match_denoise_boundaries_and_turbo_timesteps() -> None:
    _enable_cpu_runtime()
    model = Model()
    empty = BasicScheduler.execute(model=model, scheduler="simple", steps=3, denoise=0.0)["SIGMAS"]
    assert empty.dtype == torch.float32
    assert empty.numel() == 0
    full = BasicScheduler.execute(model=model, scheduler="simple", steps=3, denoise=1.0)["SIGMAS"]
    assert full.dtype == torch.float32
    torch.testing.assert_close(full, torch.tensor((8.0, 2.0, 0.5, 0.0)))
    fractional = BasicScheduler.execute(model=model, scheduler="simple", steps=3, denoise=0.6)[
        "SIGMAS"
    ]
    assert fractional.dtype == torch.float32
    torch.testing.assert_close(fractional, torch.tensor((2.0, 0.5, 0.125, 0.0)))
    beta = BetaSamplingScheduler.execute(model=model, steps=4, alpha=0.7, beta=1.3)["SIGMAS"]
    assert beta.dtype == torch.float32
    torch.testing.assert_close(beta, torch.tensor((8.0, 1.0, 0.5, 0.125, 0.0)))
    turbo = SDTurboScheduler.execute(model=model, steps=4, denoise=0.6)["SIGMAS"]
    torch.testing.assert_close(turbo, torch.tensor((0.599, 0.499, 0.399, 0.299, 0.0)))
    assert turbo.dtype == torch.float32
