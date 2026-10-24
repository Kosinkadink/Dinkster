"""CPU behavior for native sigma scheduler nodes."""

from __future__ import annotations

import sys
from types import ModuleType
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
    sigmas = torch.tensor((0.0, 0.1, 0.4, 1.0, 2.0, 4.0, 8.0), dtype=torch.float32)

    def sigma(self, timesteps: torch.Tensor) -> torch.Tensor:
        return timesteps.to(torch.float32) / 1000


class Model:
    def get_model_object(self, name: str) -> ModelSampling:
        assert name == "model_sampling"
        return ModelSampling()


def _install_runtime(monkeypatch: Any) -> None:
    runtime = ModuleType("dinkster_inference")
    samplers = ModuleType("dinkster_inference.samplers")
    sampling = ModuleType("dinkster_inference.k_diffusion.sampling")

    def calculate_sigmas(model_sampling: ModelSampling, scheduler: str, steps: int) -> torch.Tensor:
        assert model_sampling is not None
        assert scheduler == "normal"
        return torch.arange(steps + 1, dtype=torch.float32)

    def beta_scheduler(
        model_sampling: ModelSampling, steps: int, *, alpha: float, beta: float
    ) -> torch.Tensor:
        assert model_sampling is not None
        assert (alpha, beta) == (0.7, 1.3)
        return torch.linspace(steps, 0, steps + 1, dtype=torch.float32)

    samplers.calculate_sigmas = calculate_sigmas
    samplers.beta_scheduler = beta_scheduler
    sampling.get_sigmas_karras = lambda n, sigma_min, sigma_max, rho: torch.tensor(
        (sigma_max, rho, sigma_min, 0.0), dtype=torch.float32
    )
    sampling.get_sigmas_exponential = lambda n, sigma_min, sigma_max: torch.tensor(
        (sigma_max, sigma_min, 0.0), dtype=torch.float32
    )
    sampling.get_sigmas_polyexponential = lambda n, sigma_min, sigma_max, rho: torch.tensor(
        (sigma_max, rho, sigma_min, 0.0), dtype=torch.float32
    )
    sampling.get_sigmas_laplace = lambda n, sigma_min, sigma_max, mu, beta: torch.tensor(
        (sigma_max, mu, beta, sigma_min), dtype=torch.float32
    )
    sampling.get_sigmas_vp = lambda n, beta_d, beta_min, eps_s: torch.tensor(
        (beta_d, beta_min, eps_s, 0.0), dtype=torch.float32
    )
    monkeypatch.setitem(sys.modules, "dinkster_inference", runtime)
    monkeypatch.setitem(sys.modules, "dinkster_inference.samplers", samplers)
    monkeypatch.setitem(sys.modules, "dinkster_inference.k_diffusion", ModuleType("k_diffusion"))
    monkeypatch.setitem(sys.modules, "dinkster_inference.k_diffusion.sampling", sampling)


def test_schedulers_use_pinned_runtime_functions_and_float32_outputs(monkeypatch: Any) -> None:
    _install_runtime(monkeypatch)
    model = Model()
    expected = {
        KarrasScheduler: (
            dict(steps=3, sigma_max=4.2, sigma_min=0.2, rho=2.3),
            (4.2, 2.3, 0.2, 0.0),
        ),
        ExponentialScheduler: (dict(steps=3, sigma_max=4.2, sigma_min=0.2), (4.2, 0.2, 0.0)),
        PolyexponentialScheduler: (
            dict(steps=3, sigma_max=4.2, sigma_min=0.2, rho=1.7),
            (4.2, 1.7, 0.2, 0.0),
        ),
        LaplaceScheduler: (
            dict(steps=3, sigma_max=4.2, sigma_min=0.2, mu=-0.3, beta=0.7),
            (4.2, -0.3, 0.7, 0.2),
        ),
        VPScheduler: (
            dict(steps=3, beta_d=13.2, beta_min=0.3, eps_s=0.02),
            (13.2, 0.3, 0.02, 0.0),
        ),
    }
    for scheduler, (inputs, values) in expected.items():
        sigmas = scheduler.execute(**inputs)["SIGMAS"]
        assert sigmas.dtype == torch.float32
        torch.testing.assert_close(sigmas, torch.tensor(values, dtype=torch.float32))

    beta = BetaSamplingScheduler.execute(model=model, steps=4, alpha=0.7, beta=1.3)["SIGMAS"]
    torch.testing.assert_close(beta, torch.tensor((4.0, 3.0, 2.0, 1.0, 0.0)))


def test_basic_scheduler_denoise_boundaries_and_sd_turbo_terminal_zero(monkeypatch: Any) -> None:
    _install_runtime(monkeypatch)
    model = Model()
    empty = BasicScheduler.execute(model=model, scheduler="normal", steps=3, denoise=0.0)["SIGMAS"]
    assert empty.dtype == torch.float32
    assert empty.numel() == 0
    fractional = BasicScheduler.execute(model=model, scheduler="normal", steps=3, denoise=0.6)[
        "SIGMAS"
    ]
    torch.testing.assert_close(fractional, torch.tensor((2.0, 3.0, 4.0, 5.0)))
    turbo = SDTurboScheduler.execute(model=model, steps=4, denoise=0.6)["SIGMAS"]
    torch.testing.assert_close(turbo, torch.tensor((0.599, 0.499, 0.399, 0.299, 0.0)))
    assert turbo.dtype == torch.float32
