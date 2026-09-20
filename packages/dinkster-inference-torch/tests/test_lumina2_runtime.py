from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, cast

import dinkster_inference_torch.lumina2_runtime as runtime_mod
import pytest
import torch
from dinkster_inference import (
    LUMINA2,
    LUMINA2_SIGMAS,
    Conditioning,
    CustomSamplingRequest,
    CustomSamplingRuntime,
    FlowSigmas,
    PreparedMultiStreamConditioning,
    SamplingGuidance,
    sampling_sigmas,
)
from dinkster_inference_torch import (
    AssembledLumina2,
    Lumina2Denoiser,
    Lumina2DiffusionRuntime,
    Lumina2Runtime,
    Lumina2RuntimeError,
    ZImage,
)
from dinkster_inference_torch.sampling_execution import run_ksampler_as_custom
from dinkster_inference_torch.schedules import (
    custom_beta_sigmas,
    custom_percent_to_sigma,
    torch_scheduler_registry,
)
from dinkster_inference_torch.solvers import torch_sampler_registry


class RecordingLumina2(torch.nn.Module):
    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.value = value
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def forward(
        self, latent: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        self.calls.append((latent, timestep, context))
        return torch.full_like(latent, self.value)


class ArithmeticLumina2(RecordingLumina2):
    def forward(
        self, latent: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        self.calls.append((latent, timestep, context))
        scale = context.float().mean(dim=(1, 2)).reshape(-1, 1, 1, 1)
        return latent * 0.5 + timestep.reshape(-1, 1, 1, 1) * 0.25 + scale


def diffusion_runtime(model: RecordingLumina2) -> Lumina2DiffusionRuntime:
    return Lumina2DiffusionRuntime(
        cast("ZImage", model),
        runtime_identity="native:dinkster.lumina2:" + "0" * 64,
        compute_dtype=torch.float32,
        sampler_registry=torch_sampler_registry(),
        scheduler_registry=torch_scheduler_registry(),
    )


def conditioning(fill: float = 0.0) -> Conditioning[torch.Tensor]:
    return Conditioning(torch.full((1, 4, 2304), fill), None)


def custom_request(
    sampler_id: str = "dinkster.euler",
    sigmas: tuple[float, ...] = (1.0, 0.5, 0.0),
) -> CustomSamplingRequest[torch.Tensor]:
    sampler = torch_sampler_registry().get(sampler_id)
    assert sampler is not None
    return CustomSamplingRequest(sampler, (), sigmas)


def test_complete_runtime_reuses_sampling_and_delegates_text_and_codec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True)
    class Codec:
        compute_dtype: torch.dtype | None = None

        def encode(self, value: torch.Tensor) -> torch.Tensor:
            return value + 2

        def decode(self, value: torch.Tensor) -> torch.Tensor:
            return value - 2

    model = RecordingLumina2()
    text = object()
    vae = object()
    encoded = conditioning()

    def encode(prompt: str) -> Conditioning[torch.Tensor]:
        assert prompt == "fox"
        return encoded

    def text_encoder(module: object) -> SimpleNamespace:
        assert module is text
        return SimpleNamespace(encode_text=encode)

    def codec_plugin(module: object) -> Codec:
        assert module is vae
        return Codec()

    monkeypatch.setattr(runtime_mod, "Lumina2TextRuntime", text_encoder)
    monkeypatch.setattr(runtime_mod, "kl_codec_plugin", codec_plugin)
    assembled = AssembledLumina2(
        family=LUMINA2,
        diffusion=cast("Any", model),
        gemma2_2b=cast("Any", text),
        vae=cast("Any", vae),
        _component_compute_dtypes={"diffusion": torch.float32, "vae": torch.float32},
    )
    runtime = Lumina2Runtime(assembled, runtime_identity="complete-lumina2")
    assert runtime.assembled is assembled
    assert runtime.family is LUMINA2
    assert runtime.runtime_identity == "complete-lumina2"
    assert runtime.codec.compute_dtype is torch.float32
    assert runtime.encode_text("fox") is encoded
    assert Lumina2Runtime.sample_custom is Lumina2DiffusionRuntime.sample_custom
    value = torch.zeros((1, 16, 2, 2))
    torch.testing.assert_close(runtime.encode_content(value), value + 2)
    torch.testing.assert_close(runtime.decode_latent(value), value - 2)
    arguments: dict[str, Any] = dict(
        noise=torch.ones_like(value), cond=encoded, request=custom_request(), seed=17
    )
    actual = runtime.sample_custom(value, **arguments)
    expected = diffusion_runtime(model).sample_custom(value, **arguments)
    torch.testing.assert_close(actual.output, expected.output, rtol=0, atol=0)
    space = FlowSigmas(shift=3.1, multiplier=1.0)
    derived = runtime.with_sampling_space(space)
    assert type(derived) is Lumina2Runtime
    assert derived.assembled is runtime.assembled and derived.codec is runtime.codec
    assert derived.runtime_identity == runtime.runtime_identity
    assert derived.sampling_sigma_space() is space
    assert runtime.sampling_sigma_space() == LUMINA2_SIGMAS
    assert torch.equal(derived.sample_custom(value, **arguments).output, actual.output)


def test_denoiser_batches_cfg_and_converts_flow_output() -> None:
    model = RecordingLumina2()
    evaluator = Lumina2Denoiser(cast("Any", model), compute_dtype=torch.float32)
    latent = torch.full((2, 16, 2, 2), 4.0)
    outputs = evaluator.evaluate_conditioning_batch(
        latent,
        0.25,
        (
            (torch.zeros((1, 3, 2304)), "positive"),
            (torch.ones((1, 3, 2304)), "negative"),
        ),
    )
    assert len(outputs) == 2
    torch.testing.assert_close(outputs[0], torch.full_like(latent, 3.75))
    torch.testing.assert_close(outputs[1], torch.full_like(latent, 3.75))
    model_latent, timestep, context = model.calls[0]
    assert model_latent.shape == (4, 16, 2, 2)
    assert timestep.tolist() == [0.25] * 4
    assert context.shape == (4, 3, 2304)


def test_denoiser_accepts_single_frame_rank_five() -> None:
    model = RecordingLumina2()
    evaluator = Lumina2Denoiser(cast("Any", model), compute_dtype=torch.float32)
    latent = torch.zeros((1, 16, 1, 2, 2))
    result = evaluator.evaluate_conditioning(latent, 0.5, (torch.zeros((1, 2, 2304)), "positive"))
    assert result.shape == latent.shape
    assert model.calls[0][0].shape == (1, 16, 2, 2)


def test_denoiser_refuses_invalid_conditioning_and_latents() -> None:
    evaluator = Lumina2Denoiser(cast("Any", RecordingLumina2()))
    with pytest.raises(Lumina2RuntimeError, match="Conditioning value"):
        evaluator.prepare_conditioning(torch.zeros((1, 2, 2304)))
    with pytest.raises(Lumina2RuntimeError, match="pooled"):
        evaluator.prepare_conditioning(Conditioning(torch.zeros((1, 2, 2304)), torch.zeros((1, 1))))
    with pytest.raises(Lumina2RuntimeError, match="2304"):
        evaluator.prepare_conditioning(Conditioning(torch.zeros((1, 2, 2560)), None))
    with pytest.raises(Lumina2RuntimeError, match="lane id"):
        evaluator.prepare_conditioning(conditioning(), lane_id="reference")
    with pytest.raises(Lumina2RuntimeError, match="empty or incompatible"):
        evaluator.evaluate_conditioning_batch(torch.zeros((1, 16, 2, 2)), 0.5, ())
    with pytest.raises(Lumina2RuntimeError, match="image or single-frame"):
        evaluator.evaluate_conditioning_batch(
            torch.zeros((1, 16, 2, 2, 2)),
            0.5,
            ((torch.zeros((1, 2, 2304)), "positive"),),
        )


def test_runtime_uses_default_and_explicit_shift(monkeypatch: pytest.MonkeyPatch) -> None:
    from dinkster_inference_torch import sampling_execution

    runtime = diffusion_runtime(RecordingLumina2(value=0.0))
    shifts: list[float] = []
    original = sampling_execution.build_sampling_schedule

    def capture(*args: Any, **kwargs: Any) -> Any:
        shifts.append(args[1].shift)
        return original(*args, **kwargs)

    monkeypatch.setattr(sampling_execution, "build_sampling_schedule", capture)
    latent = torch.zeros((1, 16, 2, 2))
    for shift in (None, 4.0):
        result = runtime.sample(
            latent,
            cond=conditioning(),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
            compute_dtype=torch.float32,
            sampling_shift=shift,
        )
        assert result.shape == latent.shape
    assert shifts == [6.0, 4.0]


@pytest.mark.parametrize("value", (0.0, -1.0, float("inf"), 6, True))
def test_runtime_refuses_invalid_sampling_shift(value: object) -> None:
    runtime = diffusion_runtime(RecordingLumina2(value=0.0))
    with pytest.raises(Lumina2RuntimeError, match="sampling_shift"):
        runtime.sample(
            torch.zeros((1, 16, 2, 2)),
            cond=conditioning(),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
            sampling_shift=cast("Any", value),
        )


def test_runtime_satisfies_custom_sampling_protocol() -> None:
    runtime = diffusion_runtime(RecordingLumina2())
    assert isinstance(runtime, CustomSamplingRuntime)
    assert runtime.streamed_residency_components == frozenset()


def test_lumina_explicit_space_reaches_schedule_and_execution() -> None:
    from dinkster_inference_torch.denoise import prepare_noise

    runtime = diffusion_runtime(ArithmeticLumina2())
    space = FlowSigmas(shift=3.1, multiplier=1.0)
    derived = runtime.with_sampling_space(space)
    assert derived is not runtime and derived.assembled is runtime.assembled
    sigmas = derived.custom_sampling_sigmas("dinkster.simple", 3, 1.0)
    descriptor = torch_scheduler_registry().get("dinkster.simple")
    assert descriptor is not None
    assert sigmas == sampling_sigmas(descriptor, space, 3)
    latent = torch.zeros((1, 16, 2, 2))
    result = derived.sample(
        latent,
        cond=conditioning(),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=3,
        denoise=1.0,
        seed=7,
        compute_dtype=torch.float32,
    )
    custom = derived.sample_custom(
        latent,
        noise=prepare_noise(latent, 7),
        cond=conditioning(),
        request=custom_request(sigmas=sigmas),
        seed=7,
        compute_dtype=torch.float32,
    )
    assert torch.equal(result, custom.output)
    assert runtime.sampling_sigma_space() == LUMINA2_SIGMAS
    with pytest.raises(Lumina2RuntimeError, match="cannot combine"):
        derived.sampling_sigma_space(4.0)


@pytest.mark.parametrize(
    ("sampler_id", "cfg_scale"),
    (("dinkster.euler", None), ("dinkster.res_multistep", 3.5), ("dinkster.dpmpp_sde", None)),
)
def test_ksampler_is_bit_equal_sugar_over_custom_sampling(
    sampler_id: str, cfg_scale: float | None
) -> None:
    runtime = diffusion_runtime(ArithmeticLumina2())
    positive = conditioning(2.0)
    negative = None if cfg_scale is None else conditioning(5.0)
    latent = torch.rand((1, 16, 2, 2), generator=torch.Generator().manual_seed(11))
    guidance = SamplingGuidance(negative, 1.0 if cfg_scale is None else cfg_scale)
    expected = runtime.sample(
        latent,
        cond=positive,
        cfg=guidance,
        sampler_id=sampler_id,
        scheduler_id="dinkster.simple",
        steps=3,
        denoise=1.0,
        seed=185,
        compute_dtype=torch.float32,
    )
    result = run_ksampler_as_custom(
        runtime,
        latent,
        samplers=cast("Any", runtime)._samplers,
        schedulers=cast("Any", runtime)._schedulers,
        space=LUMINA2_SIGMAS,
        flow=True,
        sampler_id=sampler_id,
        scheduler_id="dinkster.simple",
        steps=3,
        denoise=1.0,
        seed=185,
        cond=positive,
        cfg=guidance,
        error=Lumina2RuntimeError,
    )
    assert type(result.output) is torch.Tensor
    assert torch.equal(result.output, expected)


def test_custom_sigma_surfaces_follow_shifted_flow_space() -> None:
    runtime = diffusion_runtime(RecordingLumina2())
    scheduler = torch_scheduler_registry().get("dinkster.simple")
    assert scheduler is not None
    assert runtime.custom_sampling_sigmas("dinkster.simple", 4, 0.5) == sampling_sigmas(
        scheduler, LUMINA2_SIGMAS, 4, denoise=0.5
    )
    shifted = FlowSigmas(shift=4.0)
    assert runtime.custom_sampling_sigmas(
        "dinkster.simple", 4, 0.5, sampling_shift=4.0
    ) == sampling_sigmas(scheduler, shifted, 4, denoise=0.5)
    assert runtime.custom_sampling_beta_sigmas(4, 0.6, 0.6) == custom_beta_sigmas(
        LUMINA2_SIGMAS, 4, 0.6, 0.6
    )
    assert runtime.custom_sampling_percent_to_sigma(
        0.3, return_actual_sigma=True
    ) == custom_percent_to_sigma(
        LUMINA2_SIGMAS,
        LUMINA2_SIGMAS.percent_to_sigma,
        0.3,
        return_actual_sigma=True,
    )
    with pytest.raises(Lumina2RuntimeError, match="unknown scheduler"):
        runtime.custom_sampling_sigmas("test.missing", 4, 1.0)


def test_custom_sampling_refuses_wrong_shapes_and_unsupported_modes() -> None:
    model = RecordingLumina2(value=0.0)
    runtime = diffusion_runtime(model)
    latent = torch.zeros((1, 16, 2, 2))
    request = custom_request()

    def sample(**overrides: Any) -> Any:
        arguments: dict[str, Any] = {
            "noise": torch.zeros_like(latent),
            "cond": conditioning(),
            "request": request,
        }
        arguments.update(overrides)
        return runtime.sample_custom(arguments.pop("latent", latent), **arguments)

    with pytest.raises(Lumina2RuntimeError, match="prepared multi-stream payload"):
        sample(cond=PreparedMultiStreamConditioning("native:test", object()))
    with pytest.raises(Lumina2RuntimeError, match="Lumina2 latent must be"):
        sample(latent=torch.zeros((1, 4, 2, 2)), noise=torch.zeros((1, 4, 2, 2)))
    with pytest.raises(Lumina2RuntimeError, match="distilled-guidance"):
        sample(guidance=4.0)
    with pytest.raises(Lumina2RuntimeError, match="inpaint"):
        sample(inpaint=cast("Any", object()))
    with pytest.raises(Lumina2RuntimeError, match="context windows"):
        sample(context_windows=cast("Any", object()))
    euler = torch_sampler_registry().get("dinkster.euler")
    assert euler is not None
    unknown = replace(euler, id="test.missing", aliases=())
    with pytest.raises(Lumina2RuntimeError, match="unknown sampler"):
        sample(request=CustomSamplingRequest(unknown, (), (1.0, 0.0)))
    with pytest.raises(Lumina2RuntimeError, match="adapter options: bogus_option"):
        sample(bogus_option=True)
    with pytest.raises(Lumina2RuntimeError, match="adapter options: bogus_option"):
        runtime.sample(
            latent,
            cond=conditioning(),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
            bogus_option=True,
        )
    assert not model.calls


def test_diffusion_component_carries_no_text_encoder_or_codec() -> None:
    runtime = diffusion_runtime(RecordingLumina2())
    with pytest.raises(Lumina2RuntimeError, match="carries no text encoder"):
        runtime.encode_text("a red fox")
    with pytest.raises(Lumina2RuntimeError, match="carries no VAE codec"):
        runtime.decode_latent(torch.zeros((1, 16, 2, 2)))
    with pytest.raises(Lumina2RuntimeError, match="carries no VAE codec"):
        runtime.encode_content(torch.zeros((1, 3, 16, 16)))
