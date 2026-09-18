from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import pytest
from dinkster_native import native_arm as arm

torch = pytest.importorskip("torch")


def _sigmas(*values: float) -> object:
    return arm._CustomSigmasValue(values)


def _values(output: Mapping[str, object], name: str = "sigmas") -> tuple[float, ...]:
    value = output[name]
    assert type(value) is arm._CustomSigmasValue
    return value.values


def _append_zero(values: Any) -> Any:
    return torch.cat([values, values.new_zeros([1])])


def test_karras_and_exponential_schedulers_match_comfy_float32_kernels() -> None:
    steps = 7
    sigma_max = 14.614642
    sigma_min = 0.0291675
    rho = 7.0

    ramp = torch.linspace(0, 1, steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    expected_karras = _append_zero((max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho)
    actual_karras = arm.GenerationKarrasScheduler.execute(
        steps=steps,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        rho=rho,
    )
    assert _values(actual_karras) == tuple(expected_karras.tolist())

    expected_exponential = _append_zero(
        torch.linspace(math.log(sigma_max), math.log(sigma_min), steps).exp()
    )
    actual_exponential = arm.GenerationExponentialScheduler.execute(
        steps=steps,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
    )
    assert _values(actual_exponential) == tuple(expected_exponential.tolist())


def test_polyexponential_and_laplace_schedulers_match_comfy_float32_kernels() -> None:
    steps = 8
    sigma_max = 14.614642
    sigma_min = 0.0291675
    rho = 1.7
    ramp = torch.linspace(1, 0, steps) ** rho
    expected_polyexponential = _append_zero(
        torch.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min))
    )
    actual_polyexponential = arm.GenerationPolyexponentialScheduler.execute(
        steps=steps,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        rho=rho,
    )
    assert _values(actual_polyexponential) == tuple(expected_polyexponential.tolist())

    mu = 0.2
    beta = 0.7
    values = torch.linspace(0, 1, steps)
    transformed = mu - beta * torch.sign(0.5 - values) * torch.log(
        1 - 2 * torch.abs(0.5 - values) + 1e-5
    )
    expected_laplace = torch.clamp(torch.exp(transformed), min=sigma_min, max=sigma_max)
    actual_laplace = arm.GenerationLaplaceScheduler.execute(
        steps=steps,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        mu=mu,
        beta=beta,
    )
    assert _values(actual_laplace) == tuple(expected_laplace.tolist())
    assert len(_values(actual_laplace)) == steps


def test_vp_scheduler_matches_comfy_float32_kernel() -> None:
    steps = 9
    beta_d = 19.9
    beta_min = 0.1
    eps_s = 0.001
    values = torch.linspace(1, eps_s, steps)
    expected = _append_zero(
        torch.sqrt(torch.special.expm1(beta_d * values**2 / 2 + beta_min * values))
    )
    actual = arm.GenerationVPScheduler.execute(
        steps=steps,
        beta_d=beta_d,
        beta_min=beta_min,
        eps_s=eps_s,
    )
    assert _values(actual) == tuple(expected.tolist())


def _extend_reference(
    values: tuple[float, ...],
    steps: int,
    start_at_sigma: float,
    end_at_sigma: float,
    spacing: str,
) -> tuple[float, ...]:
    sigmas = torch.FloatTensor(values)
    if start_at_sigma < 0:
        start_at_sigma = float("inf")
    interpolators: dict[str, Callable[[Any], Any]] = {
        "linear": lambda value: value,
        "cosine": lambda value: torch.sin(value * math.pi / 2),
        "sine": lambda value: 1 - torch.cos(value * math.pi / 2),
    }
    x = torch.linspace(0, 1, steps + 1, device=sigmas.device)[1:-1]
    computed_spacing = interpolators[spacing](x)
    extended_sigmas = []
    for index in range(len(sigmas) - 1):
        sigma_current = sigmas[index]
        sigma_next = sigmas[index + 1]
        extended_sigmas.append(sigma_current)
        if end_at_sigma <= sigma_current <= start_at_sigma:
            interpolated = computed_spacing * (sigma_next - sigma_current) + sigma_current
            extended_sigmas.extend(interpolated.tolist())
    if len(sigmas) > 0:
        extended_sigmas.append(sigmas[-1])
    return tuple(torch.FloatTensor(extended_sigmas).tolist())


def test_manual_and_mutating_sigma_nodes_preserve_float32_behavior() -> None:
    manual = arm.GenerationManualSigmas.execute(sigmas="1, +0.1; -.25 and 2e-3")
    assert _values(manual) == tuple(torch.FloatTensor([1.0, 0.1, -0.25, 2.0, -3.0]).tolist())

    flipped = arm.GenerationFlipSigmas.execute(sigmas=_sigmas(3.0, 1.0, 0.0))
    expected_flip = torch.FloatTensor([3.0, 1.0, 0.0]).flip(0)
    expected_flip[0] = 0.0001
    assert _values(flipped) == tuple(expected_flip.tolist())
    assert _values(arm.GenerationFlipSigmas.execute(sigmas=_sigmas())) == ()

    replaced = arm.GenerationSetFirstSigma.execute(
        sigmas=_sigmas(1.0, 0.5, 0.0),
        sigma=136.123456789,
    )
    expected_replaced = torch.FloatTensor([1.0, 0.5, 0.0])
    expected_replaced[0] = 136.123456789
    assert _values(replaced) == tuple(expected_replaced.tolist())
    with pytest.raises(IndexError):
        arm.GenerationSetFirstSigma.execute(sigmas=_sigmas(), sigma=1.0)


def test_split_sigma_nodes_match_comfy_slice_edges() -> None:
    sigmas = _sigmas(4.0, 3.0, 2.0, 1.0, 0.0)
    split = arm.GenerationSplitSigmas.execute(sigmas=sigmas, step=2)
    assert _values(split, "high_sigmas") == (4.0, 3.0, 2.0)
    assert _values(split, "low_sigmas") == (2.0, 1.0, 0.0)

    denoised = arm.GenerationSplitSigmasDenoise.execute(sigmas=sigmas, denoise=0.5)
    assert _values(denoised, "high_sigmas") == (4.0, 3.0, 2.0)
    assert _values(denoised, "low_sigmas") == (2.0, 1.0, 0.0)

    zero = arm.GenerationSplitSigmasDenoise.execute(sigmas=sigmas, denoise=0.0)
    assert _values(zero, "high_sigmas") == ()
    assert _values(zero, "low_sigmas") == (0.0,)


@pytest.mark.parametrize("spacing", ["linear", "cosine", "sine"])
def test_extend_intermediate_sigmas_matches_comfy_float32_operations(spacing: str) -> None:
    values = (14.614642, 5.1234567, 1.25, 0.0)
    actual = arm.GenerationExtendIntermediateSigmas.execute(
        sigmas=arm._CustomSigmasValue(values),
        steps=4,
        start_at_sigma=-1.0,
        end_at_sigma=1.0,
        spacing=spacing,
    )
    assert _values(actual) == _extend_reference(values, 4, -1.0, 1.0, spacing)


@pytest.mark.parametrize(("steps", "width", "height"), [(20, 1024, 1024), (9, 1216, 832)])
def test_flux2_scheduler_matches_comfy_empirical_mu_kernel(
    steps: int, width: int, height: int
) -> None:
    from dinkster_inference import flux2_empirical_mu

    mu = flux2_empirical_mu(round(width * height / 256), steps)
    timesteps = torch.linspace(1, 0, steps + 1)
    expected = math.exp(mu) / (math.exp(mu) + (1 / timesteps - 1) ** 1.0)
    actual = _values(arm.GenerationFlux2Scheduler.execute(steps=steps, width=width, height=height))
    assert actual == tuple(expected.tolist())
    assert len(actual) == steps + 1
    assert actual[0] == 1.0
    assert actual[-1] == 0.0
