from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    Conditioning,
    CustomSamplingRequest,
    CustomSamplingRuntime,
    FlowSigmas,
    SamplingGuidance,
    sampling_sigmas,
)
from dinkster_inference_torch import (
    Chroma,
    ChromaDenoiser,
    ChromaDiffusionRuntime,
    ChromaRadiance,
    ChromaRadianceOptions,
    ChromaRuntimeError,
)
from dinkster_inference_torch.chroma_runtime import ChromaRadianceOptionWindow
from dinkster_inference_torch.sampling_execution import run_ksampler_as_custom
from dinkster_inference_torch.schedules import torch_scheduler_registry
from dinkster_inference_torch.solvers import torch_sampler_registry


class RecordingChroma(torch.nn.Module):
    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.value = value
        self.calls: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, object | None]
        ] = []

    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        guidance: torch.Tensor,
        *,
        options: object | None = None,
    ) -> torch.Tensor:
        self.calls.append((latent, timestep, context, guidance, options))
        return torch.full_like(latent, self.value)


class RecordingRadiance(ChromaRadiance):
    def __init__(self, value: float = 1.0) -> None:
        torch.nn.Module.__init__(self)
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.value = value
        self.calls: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, object | None]
        ] = []

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        guidance: torch.Tensor,
        *,
        options: ChromaRadianceOptions | None = None,
    ) -> torch.Tensor:
        self.calls.append((x, timesteps, context, guidance, options))
        return torch.full_like(x, self.value)


def runtime_for(
    model: RecordingChroma | RecordingRadiance,
    *,
    sampling_shift: float = 1.0,
    windows: tuple[ChromaRadianceOptionWindow, ...] = (),
) -> ChromaDiffusionRuntime:
    return ChromaDiffusionRuntime(
        cast("Chroma | ChromaRadiance", model),
        runtime_identity="native:dinkster.chroma:" + "0" * 64,
        compute_dtype=torch.float32,
        sampling_shift=sampling_shift,
        option_windows=windows,
        sampler_registry=torch_sampler_registry(),
        scheduler_registry=torch_scheduler_registry(),
    )


def request(sigmas: tuple[float, ...] = (1.0, 0.5, 0.0)) -> CustomSamplingRequest[torch.Tensor]:
    sampler = torch_sampler_registry().get("dinkster.euler")
    assert sampler is not None
    return CustomSamplingRequest(sampler, (), sigmas)


def condition(fill: float = 0.0) -> Conditioning[torch.Tensor]:
    return Conditioning(torch.full((1, 3, 4096), fill), None)


def test_denoiser_converts_flow_output_and_batches_cfg() -> None:
    model = RecordingChroma()
    evaluator = ChromaDenoiser(cast("Any", model), guidance=3.5, compute_dtype=torch.float32)
    latent = torch.full((2, 16, 2, 2), 4.0)
    outputs = evaluator.evaluate_conditioning_batch(
        latent,
        0.25,
        ((torch.zeros((1, 3, 4096)), "positive"), (torch.ones((1, 3, 4096)), "negative")),
    )
    assert len(outputs) == 2
    torch.testing.assert_close(outputs[0], torch.full_like(latent, 3.75))
    torch.testing.assert_close(outputs[1], torch.full_like(latent, 3.75))
    model_latent, timestep, context, guidance, options = model.calls[0]
    assert model_latent.shape == (4, 16, 2, 2)
    assert timestep.tolist() == [0.25] * 4
    assert context.shape == (4, 3, 4096)
    assert guidance.tolist() == [3.5] * 4
    assert options is None


def test_radiance_option_windows_are_inclusive_and_oldest_active_wins() -> None:
    first = ChromaRadianceOptions(nerf_tile_size=1)
    second = ChromaRadianceOptions(nerf_tile_size=2, force_sequential_txt_ids=True)
    windows = (
        ChromaRadianceOptionWindow(first, 1.0, 0.5),
        ChromaRadianceOptionWindow(second, 1.0, 0.0),
        ChromaRadianceOptionWindow(ChromaRadianceOptions(nerf_tile_size=3), 0.0, 1.0),
    )
    model = RecordingRadiance(value=0.0)
    evaluator = ChromaDenoiser(
        model,
        guidance=0.0,
        option_windows=windows,
        compute_dtype=torch.float32,
    )
    latent = torch.zeros((1, 3, 2, 2))
    prepared = (torch.zeros((1, 2, 4096)), "positive")
    evaluator.evaluate_conditioning(latent, 0.5, prepared)
    evaluator.evaluate_conditioning(latent, 0.25, prepared)
    assert model.calls[0][-1] is first
    assert model.calls[1][-1] is second


def test_runtime_requires_radiance_for_option_windows() -> None:
    window = ChromaRadianceOptionWindow(ChromaRadianceOptions(nerf_tile_size=1))
    with pytest.raises(ChromaRuntimeError, match="require a Chroma Radiance"):
        runtime_for(RecordingChroma(), windows=(window,))


def test_runtime_satisfies_custom_sampling_and_executes_flow() -> None:
    model = RecordingChroma(value=0.5)
    runtime = runtime_for(model)
    assert isinstance(runtime, CustomSamplingRuntime)
    latent = torch.zeros((1, 16, 2, 2))
    result = runtime.sample_custom(
        latent,
        noise=torch.ones_like(latent),
        cond=condition(),
        request=request(),
        seed=7,
        guidance=3.5,
        compute_dtype=torch.float32,
        device="cpu",
    )
    assert type(result.output) is torch.Tensor
    assert result.output.shape == latent.shape
    assert torch.isfinite(result.output).all()
    assert model.calls


def test_runtime_uses_configured_aura_flow_shift_for_all_schedule_surfaces() -> None:
    runtime = runtime_for(RecordingChroma(), sampling_shift=1.73)
    scheduler = torch_scheduler_registry().get("dinkster.simple")
    assert scheduler is not None
    space = FlowSigmas(shift=1.73, multiplier=1.0)
    assert runtime.custom_sampling_sigmas("dinkster.simple", 4, 1.0) == sampling_sigmas(
        scheduler, space, 4
    )
    assert runtime.custom_sampling_percent_to_sigma(0.5, return_actual_sigma=True) == pytest.approx(
        space.percent_to_sigma(0.5)
    )


@pytest.mark.parametrize("shift", (0.0, -1.0, 1, float("inf"), float("nan")))
def test_runtime_refuses_invalid_aura_flow_shift(shift: object) -> None:
    with pytest.raises(ChromaRuntimeError, match="sampling shift"):
        runtime_for(RecordingChroma(), sampling_shift=cast("Any", shift))
    runtime = runtime_for(RecordingChroma())
    with pytest.raises(ChromaRuntimeError, match="sampling shift"):
        runtime.custom_sampling_sigmas("dinkster.simple", 3, 1.0, sampling_shift=cast("Any", shift))
    sampler = torch_sampler_registry().get("dinkster.euler")
    assert sampler is not None
    latent = torch.zeros((1, 16, 2, 2))
    with pytest.raises(ChromaRuntimeError, match="sampling shift"):
        runtime.sample_custom(
            latent,
            noise=torch.ones_like(latent),
            cond=condition(),
            request=CustomSamplingRequest(sampler, (), (1.0, 0.5, 0.0)),
            sampling_shift=cast("Any", shift),
        )


def test_ksampler_sugar_calls_the_one_custom_sampling_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = runtime_for(RecordingChroma(value=0.0))
    calls = 0
    original = runtime.sample_custom

    def record(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "sample_custom", record)
    latent = torch.zeros((1, 16, 2, 2))
    result = run_ksampler_as_custom(
        runtime,
        latent,
        samplers=torch_sampler_registry(),
        schedulers=torch_scheduler_registry(),
        space=FlowSigmas(shift=1.0),
        flow=True,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        denoise=1.0,
        seed=11,
        cond=condition(1.0),
        cfg=SamplingGuidance(condition(0.0), 3.0),
        guidance=3.5,
        error=ChromaRuntimeError,
    )
    assert calls == 1
    assert type(result.output) is torch.Tensor
    assert result.output.shape == latent.shape


def test_custom_sampling_refuses_wrong_shapes_and_unsupported_modes() -> None:
    runtime = runtime_for(RecordingChroma(value=0.0))
    latent = torch.zeros((1, 16, 2, 2))

    def sample(**overrides: Any) -> Any:
        arguments: dict[str, Any] = {
            "noise": torch.zeros_like(latent),
            "cond": condition(),
            "request": request(),
        }
        arguments.update(overrides)
        return runtime.sample_custom(arguments.pop("latent", latent), **arguments)

    with pytest.raises(ChromaRuntimeError, match="shape"):
        sample(latent=torch.zeros((1, 4, 2, 2)), noise=torch.zeros((1, 4, 2, 2)))
    with pytest.raises(ChromaRuntimeError, match="guidance"):
        sample(guidance=101.0)
    with pytest.raises(ChromaRuntimeError, match="inpaint"):
        sample(inpaint=cast("Any", object()))
    with pytest.raises(ChromaRuntimeError, match="context windows"):
        sample(context_windows=cast("Any", object()))


@pytest.mark.parametrize("guidance", (True, [1.0], {"scale": 1.0}, float("nan"), float("inf")))
def test_custom_sampling_refuses_malformed_guidance_before_evaluation(guidance: object) -> None:
    model = RecordingChroma(value=0.0)
    runtime = runtime_for(model)
    latent = torch.zeros((1, 16, 2, 2))

    with pytest.raises(ChromaRuntimeError, match="None, 'disabled', or a finite float"):
        runtime.sample_custom(
            latent,
            noise=torch.zeros_like(latent),
            cond=condition(),
            request=request(),
            guidance=cast("Any", guidance),
        )

    assert model.calls == []


def test_denoiser_refuses_malformed_conditioning() -> None:
    evaluator = ChromaDenoiser(
        cast("Any", RecordingChroma()), guidance=0.0, compute_dtype=torch.float32
    )
    with pytest.raises(ChromaRuntimeError, match="Conditioning"):
        evaluator.prepare_conditioning(torch.zeros((1, 2, 4096)))
    with pytest.raises(ChromaRuntimeError, match="pooled"):
        evaluator.prepare_conditioning(Conditioning(torch.zeros((1, 2, 4096)), torch.zeros((1, 1))))
    with pytest.raises(ChromaRuntimeError, match="4096"):
        evaluator.prepare_conditioning(Conditioning(torch.zeros((1, 2, 1024)), None))
    with pytest.raises(ChromaRuntimeError, match="lane"):
        evaluator.prepare_conditioning(condition(), lane_id="middle")


def test_explicit_sampling_shift_matches_the_configured_sigma_space() -> None:
    configured = runtime_for(RecordingChroma(), sampling_shift=1.73)
    runtime = runtime_for(RecordingChroma())
    sigmas = configured.custom_sampling_sigmas("dinkster.simple", 3, 1.0)
    assert runtime.custom_sampling_sigmas("dinkster.simple", 3, 1.0, sampling_shift=1.73) == sigmas
    sampler = torch_sampler_registry().get("dinkster.euler")
    assert sampler is not None
    request = CustomSamplingRequest(sampler, (), sigmas)
    latent = torch.zeros((1, 16, 2, 2))
    noise = torch.ones_like(latent)
    expected = configured.sample_custom(latent, noise=noise, cond=condition(), request=request)
    actual = runtime.sample_custom(
        latent, noise=noise, cond=condition(), request=request, sampling_shift=1.73
    )
    assert torch.equal(actual.output, expected.output)
    assert actual.denoised_output is not None and expected.denoised_output is not None
    assert torch.equal(actual.denoised_output, expected.denoised_output)
    assert runtime.sampling_sigma_space() == FlowSigmas(shift=1.0, multiplier=1.0)
