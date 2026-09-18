"""Executed golden replays for the RES4LYF beta RK engine.

Every case replays a reference run recorded by
tools/gen_res4lyf_rk_goldens.py (pinned RES4LYF sample_rk_beta over a
pinned ComfyUI sigma space with a deterministic mock denoiser and the
engine's own seeded gaussian streams) and asserts the reproducibility
seams structurally on ordinary hosts and at recorded values during CUDA
reference validation: prepared sigmas, per-step tableau coefficients, the
sigmas handed to the denoiser, stream seeds, and every noise draw's stream,
order, dtype, and float64 values. Both eta-free branch families are covered per
case space: the EPS cases take the variance-exploding SDE coefficients and
epsilon reconstruction, the CONST cases the variance-preserving flow branches.

The CUDA reference comparison splits by noise use. Cases without noise draws
(the _ode descriptors) replay bit-exactly: the reference ran the recorded
trajectory in float32 op-for-op like the port. Cases with noise swaps are
value-close only: the reference mixes its float64 stream draws into the
float32 state (promoting the swap arithmetic to float64) while the port casts
the normalized draw to float32 first, so per-step latents wobble by float32
ULPs while the draw values stay identical.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_inference import (
    Parameterization,
    SamplerInfo,
    SolverStateEvent,
    prepare_rk_sigmas,
    resolve_rk_tableau,
)
from dinkster_inference.res4lyf_rk import _sde_step  # pyright: ignore[reportPrivateUsage]
from dinkster_inference_torch.denoise import RES4LYFTwoStreamNoise, run_sampler_engine
from dinkster_inference_torch.solvers import torch_sampler_registry
from golden_files import (
    assert_reference_schedule,
    assert_reference_tensor,
    assert_reference_values,
    load_platform_golden,
)

GOLDEN = load_platform_golden(
    Path(__file__).parent / "goldens/res4lyf_rk_goldens.json",
    allow_portable_fallback=True,
)

# Measured worst-case trajectory drift across the 23 noise-swap cases
# is 2.384e-6 abs (res_3m/res_3s, 8 steps) with bit-exact prepared
# sigmas, tableau coefficients, model-call sigmas, and noise draws,
# pinning the residual to the swap arithmetic precision (the reference
# promotes it to float64 through the raw draws; the port casts the
# normalized draw to float32 first). 2e-5 gives ~8x headroom. The 16
# draw-free cases replayed bit-exactly and are asserted with
# torch.equal instead.
TRAJECTORY_ATOL = 2e-5


def _tensor(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 2, 2)


def _tensor64(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64).reshape(1, 1, 2, 2)


class _Model:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []
        self.sigmas: list[float] = []

    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        self.calls.append(x.detach().clone())
        self.sigmas.append(sigma)
        return x * (1.0 / (1.0 + sigma)) + x.square() * (0.05 / (1.0 + sigma))


class _RecordingStream:
    """Wraps one RES4LYFTwoStreamNoise stream, logging every raw
    standardized float64 draw with its stream tag (0 outer, 1 substep)
    into a shared ordered log."""

    def __init__(
        self,
        inner: Any,
        tag: int,
        log: list[tuple[int, torch.Tensor]],
    ) -> None:
        self._inner = inner
        self._tag = tag
        self._log = log

    def __call__(self) -> torch.Tensor:
        draw = self._inner()
        self._log.append((self._tag, draw.detach().clone()))
        return draw


def _case_setup(name: str) -> tuple[str, dict[str, object], str, Parameterization]:
    """(sampler id, build options, rk_type, parameterization) for one
    golden case, mirroring the generator's case construction."""
    parameterization = Parameterization.FLOW if "_const_" in name else Parameterization.EPS
    if name.startswith("rk_beta_"):
        rk_type = name[len("rk_beta_") :].rsplit("_", 1)[0]
        return "res4lyf.rk_beta", {"rk_type": rk_type}, rk_type, parameterization
    rk_type = name[:7] if name.startswith("deis") else name[:6]
    sampler = f"res4lyf.{rk_type}_ode" if "_ode_" in name else f"res4lyf.{rk_type}"
    return sampler, {}, rk_type, parameterization


@pytest.mark.parametrize("case_name", sorted(GOLDEN["cases"]))
def test_rk_engine_replays_every_recorded_reference_seam(case_name: str) -> None:
    case: dict[str, Any] = GOLDEN["cases"][case_name]
    sampler_id, options, rk_type, parameterization = _case_setup(case_name)
    vp = parameterization is Parameterization.FLOW
    exponential = rk_type.startswith("res")
    sigma_min: float = case["sigma_min"]
    sigma_max: float = case["sigma_max"]
    seed: int = GOLDEN["_meta"]["seed"]

    prep = prepare_rk_sigmas(case["schedule_sigmas"], sigma_min)
    assert_reference_schedule(prep, case["prepared_sigmas"])

    num_steps = len(prep) - 2 if prep[-1] == 0.0 else len(prep) - 1
    assert len(case["coeffs"]) == num_steps
    for step, entry in enumerate(case["coeffs"]):
        sigma = prep[step]
        sigma_next = prep[step + 1]
        _, sd_noeta, _ = _sde_step(sigma_next, 0.0, vp=vp, sigma_max=sigma_max)
        h = -math.log(sd_noeta / sigma) if exponential else sd_noeta - sigma
        a, b, ci, ms = resolve_rk_tableau(
            rk_type,
            h,
            step=step,
            sigmas=prep,
            sigma=sigma,
            sigma_next=sigma_next,
            sigma_down=sd_noeta,
        )
        assert entry["rk_type"] == rk_type
        assert entry["rows"] == len(a)
        assert len(a) == len(entry["a"])
        for actual_row, expected_row in zip(a, entry["a"], strict=True):
            assert_reference_values(actual_row, expected_row)
        assert len(b) == len(entry["b"])
        for actual_row, expected_row in zip(b, entry["b"], strict=True):
            assert_reference_values(actual_row, expected_row)
        assert_reference_values(ci, entry["c"])
        assert ms == entry["multistep_stages"]

    # The reference's own seeded path: outer seed + 1, substep + 10001.
    assert case["seeds"] == [seed + 1, seed + 10001]

    descriptor = torch_sampler_registry().get(sampler_id)
    assert descriptor is not None
    model = _Model()
    initial = _tensor(GOLDEN["initial"])
    noise = RES4LYFTwoStreamNoise(initial, seed=seed)
    draw_log: list[tuple[int, torch.Tensor]] = []
    noise.outer = _RecordingStream(noise.outer, 0, draw_log)
    noise.substep = _RecordingStream(noise.substep, 1, draw_log)
    states: list[SolverStateEvent[object]] = []
    result = descriptor.build(**options)(
        model,
        initial,
        case["schedule_sigmas"],
        SamplerInfo(
            parameterization,
            seed=seed,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            on_state=states.append,
        ),
        noise=noise,
    )

    assert_reference_values(model.sigmas, case["model_call_sigmas"])
    for (tag, draw), expected in zip(draw_log, case["noise_draws"], strict=True):
        assert tag == expected["stream"]
        assert str(draw.dtype) == expected["dtype"]
        assert_reference_tensor(draw, _tensor64(expected["values"]))

    def assert_close(actual: torch.Tensor, expected: list[float]) -> None:
        assert_reference_tensor(
            actual,
            _tensor(expected),
            atol=TRAJECTORY_ATOL if case["noise_draws"] else 0.0,
        )

    for actual, expected in zip(model.calls, case["model_calls"], strict=True):
        assert_close(actual, expected)
    step_entries = [entry for entry in case["steps"] if not entry["final"]]
    for state, entry in zip(states, step_entries, strict=True):
        assert state.phase == "post_update"
        assert state.step == entry["i"]
        assert state.total == num_steps
        assert type(state.current) is torch.Tensor
        assert_close(state.current, entry["x"])
    final_entries = [entry for entry in case["steps"] if entry["final"]]
    assert len(final_entries) == 1
    assert_close(result, final_entries[0]["x"])
    assert_close(result, case["final"])


def test_engine_dispatch_builds_the_two_stream_sampler_with_sigma_bounds() -> None:
    from dinkster_inference import NoiseKind

    captured: dict[str, object] = {}

    def spy_solver(
        denoiser: object,
        x: torch.Tensor,
        sigmas: object,
        info: SamplerInfo,
        *,
        noise: object = None,
        on_step: object = None,
    ) -> torch.Tensor:
        captured["info"] = info
        captured["noise"] = noise
        return x

    latent = torch.zeros(1, 1, 2, 2)
    run_sampler_engine(
        _Model(),
        spy_solver,
        latent=latent,
        noise=torch.zeros_like(latent),
        sigmas=[2.0, 1.0, 0.5],
        parameterization=Parameterization.EPS,
        sigma_max=14.6,
        sigma_min=0.03,
        process_in=lambda value: value,
        process_out=lambda value: value,
        seed=1234,
        noise_kind=NoiseKind.RES4LYF_GAUSSIAN,
    )
    info = captured["info"]
    assert isinstance(info, SamplerInfo)
    assert info.sigma_min == 0.03
    assert info.sigma_max == 14.6
    noise = captured["noise"]
    assert isinstance(noise, RES4LYFTwoStreamNoise)
    # Same seed mapping as a directly-constructed sampler.
    twin = RES4LYFTwoStreamNoise(latent, seed=1234)
    assert torch.equal(noise.step_noise(2.0, 1.0), twin.step_noise(2.0, 1.0))
    assert torch.equal(noise.substep_noise(2.0, 1.0), twin.substep_noise(2.0, 1.0))


def test_two_stream_sampler_refuses_plain_noise_calls() -> None:
    sampler = RES4LYFTwoStreamNoise(torch.zeros(1, 1, 2, 2), seed=0)
    with pytest.raises(TypeError, match="never as a plain NoiseSampler"):
        sampler(2.0, 1.0)


def test_rk_golden_comes_from_the_pinned_references() -> None:
    assert GOLDEN["_meta"]["comfyui_commit"] == "b78cec879b9460d5cb25228a83a942fb78d2cd24"
    assert GOLDEN["_meta"]["res4lyf_commit"] == "26036f647ca15d3048a193daf99a40cecfc3820d"
    assert GOLDEN["_meta"]["seed"] == 1234
