"""The native Flux2 runtime: context padding, flow sampling, and codec wiring.

The denoiser and runtime are proven over recording fakes (the real
Flux transformer math is proven in test_flux.py and the text towers
against executed goldens in test_flux2_text.py).

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    FLUX2_DEV,
    KLEIN_QWEN3_8B_CONFIG,
    MISTRAL3_24B_PRUNED_CONFIG,
    Conditioning,
    ConditioningSet,
    CustomSamplingRequest,
    CustomSamplingResult,
    CustomSamplingRuntime,
    FluxFlowSigmas,
    InpaintConditioning,
    KLConfig,
    NoiseKind,
    PayloadReference,
    QwenTextConfig,
    Registry,
    SamplerDescriptor,
    SamplingSegment,
    SamplingStateEvent,
    builtin_sampler_registry,
    load_flux2_tekken_bpe,
    make_conditioning_carrier,
    offset_first_sigma_for_snr,
    sampling_sigmas,
    tokenize_flux2_dev_prompt,
)
from dinkster_inference_torch import (
    FLUX2_REFERENCE_LATENTS_KEY,
    FLUX_GUIDANCE_DISABLED,
    Flux2Conditioning,
    Flux2Denoiser,
    Flux2DevTextEncoder,
    Flux2DiffusionRuntime,
    Flux2KleinTextEncoder,
    Flux2Runtime,
    Flux2RuntimeError,
    basic_conditioning_to_carrier,
    materialize_flux2_conditioning,
    prepare_noise,
    tensor_to_payload_binding,
)
from dinkster_inference_torch import sampling_execution as sampling_execution_mod
from dinkster_inference_torch.schedules import (
    _endpoints,  # pyright: ignore[reportPrivateUsage]
    custom_beta_sigmas,
    torch_scheduler_registry,
)
from dinkster_inference_torch.solvers import torch_sampler_registry


class RecordingFlux(torch.nn.Module):
    def __init__(self, value: float = 0.0, *, distilled: bool = True) -> None:
        super().__init__()
        self.value = value
        self.guidance_in = object() if distilled else None
        self.vector_in = None
        self.config = SimpleNamespace(vec_in_dim=None, context_in_dim=8)
        # sample_custom resolves device and dtype from the module's parameters.
        self.device_probe = torch.nn.Parameter(torch.zeros(()))
        self.calls: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, object, torch.Tensor | None]
        ] = []
        self.reference_calls: list[tuple[torch.Tensor, ...]] = []

    def forward(
        self,
        xc: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        y: object,
        guidance: torch.Tensor | None,
        *,
        ref_latents: tuple[torch.Tensor, ...] = (),
    ) -> torch.Tensor:
        self.calls.append((xc, timesteps, context, y, guidance))
        self.reference_calls.append(ref_latents)
        return torch.full_like(xc, self.value)


def _float32_compute_dtype(_role: str) -> torch.dtype:
    return torch.float32


def _bare_runtime(model: RecordingFlux) -> Flux2Runtime:
    assembled = type(
        "Assembled",
        (),
        {
            "family": FLUX2_DEV,
            "diffusion": model,
            "compute_dtype": staticmethod(_float32_compute_dtype),
        },
    )()
    runtime = object.__new__(Flux2Runtime)
    runtime.assembled = cast("Any", assembled)
    runtime._samplers = torch_sampler_registry()  # pyright: ignore[reportPrivateUsage]
    runtime._schedulers = torch_scheduler_registry()  # pyright: ignore[reportPrivateUsage]
    runtime._guidance = None  # pyright: ignore[reportPrivateUsage]
    return runtime


def _flux2_carrier(*references: torch.Tensor) -> Any:
    carrier = basic_conditioning_to_carrier(Conditioning(torch.ones((1, 3, 8)), None))
    bindings = [
        tensor_to_payload_binding(
            f"reference-{index}",
            reference,
            space="flux2-reference-latent",
        )
        for index, reference in enumerate(references)
    ]
    record = replace(
        carrier.conditioning.records[0],
        extension_metadata=(
            (
                FLUX2_REFERENCE_LATENTS_KEY,
                tuple(PayloadReference(binding.reference_id) for binding in bindings),
            ),
        ),
    )
    return make_conditioning_carrier(
        ConditioningSet((record,)),
        (*carrier.bindings, *bindings),
    )


def test_flux2_denoiser_left_pads_short_context_to_512_rows() -> None:
    evaluator = Flux2Denoiser(cast("Any", RecordingFlux()), compute_dtype=torch.float32)
    short = Conditioning(torch.ones((1, 3, 8)), None)
    context, pooled = cast(
        "tuple[torch.Tensor, torch.Tensor | None]",
        evaluator.prepare_conditioning(short),
    )
    assert pooled is None
    assert context.shape == (1, 512, 8)
    assert torch.count_nonzero(context[:, :509]).item() == 0
    assert torch.equal(context[:, 509:], short.embeddings)
    exact = torch.ones((1, 512, 8))
    assert torch.equal(evaluator.prepare_conditioning(Conditioning(exact, None))[0], exact)
    longer = torch.ones((1, 600, 8))
    assert torch.equal(evaluator.prepare_conditioning(Conditioning(longer, None))[0], longer)


def test_flux2_reference_carrier_materializes_ordered_owned_latents() -> None:
    first = torch.arange(128 * 2 * 3, dtype=torch.float32).reshape(1, 128, 2, 3)
    second = torch.full((1, 128, 3, 2), 4.0)

    materialized = materialize_flux2_conditioning(
        _flux2_carrier(first, second),
        device="cpu",
    )

    assert torch.equal(materialized.embeddings, torch.ones((1, 3, 8)))
    assert len(materialized.reference_latents) == 2
    assert torch.equal(materialized.reference_latents[0], first)
    assert torch.equal(materialized.reference_latents[1], second)
    assert materialized.reference_latents[0].data_ptr() != first.data_ptr()


def test_flux2_reference_carrier_refuses_malformed_metadata_and_geometry() -> None:
    malformed = _flux2_carrier(torch.zeros((1, 128, 2, 2)))
    reference = dict(malformed.conditioning.records[0].extension_metadata)[
        FLUX2_REFERENCE_LATENTS_KEY
    ]
    assert isinstance(reference, tuple)
    record = replace(
        malformed.conditioning.records[0],
        extension_metadata=((FLUX2_REFERENCE_LATENTS_KEY, reference[0]),),
    )
    with pytest.raises(Flux2RuntimeError, match="ordered payload tuple"):
        materialize_flux2_conditioning(
            make_conditioning_carrier(
                ConditioningSet((record,)),
                malformed.bindings,
            ),
            device="cpu",
        )
    with pytest.raises(Flux2RuntimeError, match=r"floating \[batch,128"):
        materialize_flux2_conditioning(
            _flux2_carrier(torch.zeros((1, 127, 2, 2))),
            device="cpu",
        )


def test_flux2_denoiser_delivers_reference_latents_and_disabled_guidance() -> None:
    model = RecordingFlux(value=1.0)
    evaluator = Flux2Denoiser(
        cast("Any", model),
        guidance=FLUX_GUIDANCE_DISABLED,
        compute_dtype=torch.float32,
    )
    reference = torch.full((1, 128, 2, 3), 5.0)
    prepared = evaluator.prepare_conditioning(
        Flux2Conditioning(torch.ones((1, 3, 8)), None, (reference,))
    )

    output = evaluator.evaluate_conditioning(
        torch.zeros((1, 128, 2, 2)),
        0.5,
        prepared,
    )

    assert output.shape == (1, 128, 2, 2)
    assert model.calls[0][4] is None
    assert len(model.reference_calls[0]) == 1
    assert torch.equal(model.reference_calls[0][0], reference)


def test_flux2_denoiser_batches_only_matching_reference_geometry() -> None:
    evaluator = Flux2Denoiser(cast("Any", RecordingFlux()), compute_dtype=torch.float32)
    text = Conditioning(torch.ones((1, 3, 8)), None)
    first = evaluator.prepare_conditioning(
        Flux2Conditioning(text.embeddings, text.pooled, (torch.zeros((1, 128, 2, 3)),))
    )
    same = evaluator.prepare_conditioning(
        Flux2Conditioning(text.embeddings, text.pooled, (torch.zeros((2, 128, 2, 3)),))
    )
    different = evaluator.prepare_conditioning(
        Flux2Conditioning(text.embeddings, text.pooled, (torch.zeros((1, 128, 3, 2)),))
    )

    assert evaluator.batchable((first, same))
    assert not evaluator.batchable((first, different))


def test_flux2_denoiser_refuses_empty_batch_and_wrong_context_width() -> None:
    evaluator = Flux2Denoiser(cast("Any", RecordingFlux()), compute_dtype=torch.float32)
    with pytest.raises(Flux2RuntimeError, match="tokens x 8"):
        evaluator.prepare_conditioning(Conditioning(torch.ones((1, 3, 12)), None))
    with pytest.raises(Flux2RuntimeError, match="tokens x 8"):
        evaluator.prepare_conditioning(Conditioning(torch.ones((0, 3, 8)), None))


def test_flux2_denoiser_padding_batches_mixed_lengths_with_dev_guidance() -> None:
    model = RecordingFlux(value=1.0)
    evaluator = Flux2Denoiser(cast("Any", model), compute_dtype=torch.float32)
    first = evaluator.prepare_conditioning(Conditioning(torch.ones((1, 3, 8)), None))
    second = evaluator.prepare_conditioning(Conditioning(torch.ones((1, 40, 8)), None))
    assert evaluator.batchable((first, second))
    latent = torch.full((1, 128, 2, 2), 4.0)
    outputs = evaluator.evaluate_conditioning_batch(latent, 0.25, (first, second))
    assert len(outputs) == 2
    torch.testing.assert_close(outputs[0], torch.full_like(latent, 3.75))
    torch.testing.assert_close(outputs[1], torch.full_like(latent, 3.75))
    xc, timesteps, context, y, guidance = model.calls[0]
    assert xc.shape == (2, 128, 2, 2)
    assert timesteps.tolist() == [0.25] * 2
    assert context.shape == (2, 512, 8)
    assert y is None
    assert guidance is not None
    assert guidance.tolist() == [3.5] * 2


def test_flux2_runtime_samples_in_family_flow_space_with_shift_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = RecordingFlux(value=0.0)
    runtime = _bare_runtime(model)
    shifts: list[float] = []
    original = sampling_execution_mod.build_sampling_schedule

    def capture(*args: Any, **kwargs: Any) -> Any:
        shifts.append(args[1].shift)
        return original(*args, **kwargs)

    monkeypatch.setattr(sampling_execution_mod, "build_sampling_schedule", capture)
    latent = torch.zeros((1, 128, 2, 2))
    cond = Conditioning(torch.zeros((1, 3, 8)), None)
    result = runtime.sample(
        latent,
        cond=cond,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=1,
        seed=4,
        compute_dtype=torch.float32,
    )
    assert result.shape == latent.shape
    assert model.calls[0][4] is not None
    assert model.calls[0][4].tolist() == [3.5]
    runtime.sample(
        latent,
        cond=cond,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=1,
        seed=4,
        compute_dtype=torch.float32,
        sampling_shift=1.18452766,
    )
    assert shifts == [2.02, 1.18452766]


def test_flux2_sampling_paths_reject_unknown_adapter_options() -> None:
    model = RecordingFlux(value=0.0)
    runtime = _bare_runtime(model)
    latent = torch.zeros((1, 128, 2, 2))
    condition = Conditioning(torch.zeros((1, 3, 8)), None)
    sampler = runtime._samplers.get("dinkster.euler")  # pyright: ignore[reportPrivateUsage]
    assert sampler is not None

    with pytest.raises(Flux2RuntimeError, match="adapter options: bogus_option"):
        runtime.sample_custom(
            latent,
            noise=torch.zeros_like(latent),
            cond=condition,
            request=CustomSamplingRequest(sampler, (), (1.0, 0.0)),
            bogus_option=True,
        )
    with pytest.raises(Flux2RuntimeError, match="adapter options: bogus_option"):
        runtime.sample(
            latent,
            cond=condition,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
            bogus_option=True,
        )
    assert not model.calls


def test_flux2_runtime_sample_delegates_ksampler_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference_torch import sampling_runtime

    runtime = _bare_runtime(RecordingFlux())
    latent = torch.zeros((1, 128, 2, 2))
    cond = Conditioning(torch.zeros((1, 3, 8)), None)
    output = torch.ones_like(latent)
    segment = SamplingSegment(4, 1, 3, False, True)
    denoise_mask = torch.ones_like(latent)
    inpaint = cast("InpaintConditioning[torch.Tensor]", object())
    steps: list[object] = []
    states: list[object] = []
    calls: list[tuple[object, torch.Tensor, dict[str, object]]] = []

    def run_ksampler(
        runtime_value: object, value: torch.Tensor, **kwargs: object
    ) -> CustomSamplingResult[torch.Tensor]:
        calls.append((runtime_value, value, dict(kwargs)))
        return CustomSamplingResult(output, None)

    monkeypatch.setattr(sampling_runtime, "run_ksampler_as_custom", run_ksampler)
    result = runtime.sample(
        latent,
        cond=cond,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        denoise=0.8,
        seed=23,
        guidance=3.5,
        segment=segment,
        denoise_mask=denoise_mask,
        inpaint=inpaint,
        noise_inds=(4,),
        on_step=steps.append,
        on_state=states.append,
        compute_dtype=torch.float32,
        device="cpu",
        sampling_shift=1.18452766,
    )

    assert result is output
    assert len(calls) == 1
    runtime_value, value, kwargs = calls[0]
    assert runtime_value is runtime
    assert value is latent
    assert kwargs["samplers"] is cast("Any", runtime)._samplers
    assert kwargs["schedulers"] is cast("Any", runtime)._schedulers
    assert cast("FluxFlowSigmas", kwargs["space"]).shift == 1.18452766
    assert kwargs["flow"] is True
    assert kwargs["cond"] is cond
    assert kwargs["sampler_id"] == "dinkster.euler"
    assert kwargs["scheduler_id"] == "dinkster.simple"
    assert kwargs["steps"] == 2
    assert kwargs["denoise"] == 0.8
    assert kwargs["seed"] == 23
    assert kwargs["guidance"] == 3.5
    assert kwargs["segment"] is segment
    assert kwargs["denoise_mask"] is denoise_mask
    assert kwargs["inpaint"] is inpaint
    assert kwargs["noise_inds"] == (4,)
    assert kwargs["on_step"] == steps.append
    assert kwargs["on_state"] == states.append
    assert kwargs["sample_custom_kwargs"] == {
        "sampling_shift": 1.18452766,
        "compute_dtype": torch.float32,
        "device": "cpu",
    }
    assert kwargs["error"] is Flux2RuntimeError


@pytest.mark.parametrize("runtime_type", (Flux2Runtime, Flux2DiffusionRuntime))
@pytest.mark.parametrize("private_argument", ("_compute_dtype", "_device"))
def test_flux2_ksampler_refuses_private_compute_placement(
    runtime_type: type[Flux2Runtime] | type[Flux2DiffusionRuntime], private_argument: str
) -> None:
    runtime = object.__new__(runtime_type)
    with pytest.raises(TypeError, match="private compute placement"):
        runtime.sample(
            torch.zeros((1, 128, 2, 2)),
            cond=Conditioning(torch.zeros((1, 3, 8))),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
            **cast(
                "Any",
                {
                    private_argument: torch.float64
                    if private_argument == "_compute_dtype"
                    else "meta"
                },
            ),
        )


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), True, 2])
def test_flux2_runtime_refuses_non_positive_finite_float_sampling_shift(value: object) -> None:
    model = RecordingFlux(value=0.0)
    runtime = _bare_runtime(model)
    with pytest.raises(Flux2RuntimeError, match="positive finite float"):
        runtime.sample(
            torch.zeros((1, 128, 2, 2)),
            cond=Conditioning(torch.zeros((1, 3, 8)), None),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
            sampling_shift=cast("Any", value),
        )
    assert not model.calls


def test_flux2_runtime_refuses_nonlatent_and_unsupported_inputs() -> None:
    model = RecordingFlux(value=0.0)
    runtime = _bare_runtime(model)
    condition = Conditioning(torch.zeros((1, 3, 8)), None)
    for latent in (torch.zeros((1, 16, 2, 2)), torch.zeros((128, 2, 2))):
        with pytest.raises(Flux2RuntimeError, match="128"):
            runtime.sample(
                latent,
                cond=condition,
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.simple",
                steps=1,
            )
    latent = torch.zeros((1, 128, 2, 2))
    with pytest.raises(Flux2RuntimeError, match="inpaint"):
        runtime.sample(
            latent,
            cond=condition,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
            inpaint=cast("Any", object()),
        )
    klein = RecordingFlux(value=0.0, distilled=False)
    runtime = _bare_runtime(klein)
    with pytest.raises(Flux2RuntimeError, match="distilled-guidance"):
        runtime.sample(
            latent,
            cond=condition,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
            guidance=4.0,
        )
    runtime.sample(
        latent,
        cond=condition,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=1,
        guidance=FLUX_GUIDANCE_DISABLED,
    )
    assert not model.calls
    assert klein.calls[0][4] is None


def test_flux2_runtime_satisfies_the_custom_sampling_protocol() -> None:
    runtime = _bare_runtime(RecordingFlux())
    assert isinstance(runtime, CustomSamplingRuntime)
    diffusion = Flux2DiffusionRuntime(
        cast("Any", RecordingFlux()), FLUX2_DEV, runtime_identity="native:test"
    )
    assert isinstance(diffusion, CustomSamplingRuntime)


def test_flux2_custom_sampling_uses_exact_sigmas_noise_options_and_denoised_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built_options: list[dict[str, object]] = []

    def make(options: Mapping[str, object]) -> Any:
        built_options.append(dict(options))

        def solver(*_args: object, **_kwargs: object) -> torch.Tensor:
            raise AssertionError("the captured run_denoise must not execute the solver")

        return solver

    builtin = builtin_sampler_registry().get("dinkster.euler")
    assert builtin is not None
    sampler = replace(
        builtin,
        id="test.custom",
        aliases=(),
        make=make,
        noise=NoiseKind.BROWNIAN_GPU,
        requires_snr_offset=True,
    )
    samplers: Registry[SamplerDescriptor[Any]] = Registry()
    samplers.register(sampler)
    model = RecordingFlux()
    runtime = _bare_runtime(model)
    runtime._samplers = samplers  # pyright: ignore[reportPrivateUsage]
    latent = torch.zeros((1, 128, 2, 2))
    noise = torch.full_like(latent, 0.25)
    pre_offset = (1.0, 0.7, 0.0)
    request = CustomSamplingRequest(
        sampler,
        (("s_churn", 0.5),),
        pre_offset,
    )
    captured: dict[str, object] = {}
    denoised = torch.full_like(latent, 0.4)
    output = torch.full_like(latent, 0.6)
    user_events: list[SamplingStateEvent[object]] = []
    step_noise = object()

    def capture_step_noise(
        resolved_sampler: SamplerDescriptor[Any],
        schedule: object,
        like: torch.Tensor,
        *,
        seed: int,
        device: torch.device,
    ) -> Any:
        assert resolved_sampler is sampler
        assert schedule is not None
        assert like is latent
        assert seed == 9
        assert device == next(model.parameters()).device
        return step_noise

    def capture_run(_denoiser: object, _solver: object, **kwargs: object) -> torch.Tensor:
        captured.update(kwargs)
        callback = cast("Any", kwargs["on_state"])
        callback(
            SamplingStateEvent(
                step=0,
                total=2,
                sigma=0.7,
                phase="pre_update",
                current=latent,
                denoised=denoised,
            )
        )
        return output

    monkeypatch.setattr(sampling_execution_mod, "brownian_step_noise", capture_step_noise)
    monkeypatch.setattr(sampling_execution_mod, "run_denoise", capture_run)
    result = runtime.sample_custom(
        latent,
        noise=noise,
        cond=Conditioning(torch.zeros((1, 3, 8)), None),
        cfg=None,
        request=request,
        seed=9,
        on_state=user_events.append,
    )

    expected_sigmas = offset_first_sigma_for_snr(
        pre_offset,
        FluxFlowSigmas(shift=FLUX2_DEV.sampling.shift),
        flow=True,
    )
    assert captured["noise"] is noise
    assert captured["sigmas"] == expected_sigmas
    assert captured["initial_sigma"] == pre_offset[0]
    assert captured["noise_sampler"] is step_noise
    assert built_options == [
        {
            "s_churn": 0.5,
            "s_tmin": 0.0,
            "s_tmax": float("inf"),
            "s_noise": 1.0,
        }
    ]
    assert result.output is output
    # The Flux2 packed latent has identity process_out.
    assert result.denoised_output is not None
    assert torch.equal(result.denoised_output, denoised)
    assert len(user_events) == 1


def test_flux2_custom_sampling_sigmas_are_basic_scheduler_space() -> None:
    runtime = _bare_runtime(RecordingFlux())
    scheduler = torch_scheduler_registry().get("dinkster.normal")
    assert scheduler is not None
    space = FluxFlowSigmas(shift=FLUX2_DEV.sampling.shift)
    expected = sampling_sigmas(scheduler, space, 4, denoise=0.5)
    assert runtime.custom_sampling_sigmas("dinkster.normal", 4, 0.5) == expected
    with pytest.raises(Flux2RuntimeError, match="unknown scheduler"):
        runtime.custom_sampling_sigmas("nope", 4, 0.5)


def test_flux2_custom_sampling_model_sigma_queries_use_the_family_space() -> None:
    runtime = _bare_runtime(RecordingFlux())
    space = FluxFlowSigmas(shift=FLUX2_DEV.sampling.shift)
    shifted_space = FluxFlowSigmas(shift=1.18452766)
    assert runtime.custom_sampling_beta_sigmas(4, 0.6, 0.6) == custom_beta_sigmas(
        space, 4, 0.6, 0.6
    )
    assert runtime.custom_sampling_beta_sigmas(
        4, 0.6, 0.6, sampling_shift=1.18452766
    ) == custom_beta_sigmas(shifted_space, 4, 0.6, 0.6)
    # return_actual_sigma clamps to the reference's float32 table endpoints.
    sigma_min, sigma_max = _endpoints(space)  # pyright: ignore[reportPrivateUsage]
    assert runtime.custom_sampling_percent_to_sigma(0.0, return_actual_sigma=True) == sigma_max
    assert runtime.custom_sampling_percent_to_sigma(1.0, return_actual_sigma=True) == sigma_min
    assert runtime.custom_sampling_percent_to_sigma(
        0.3, return_actual_sigma=False
    ) == space.percent_to_sigma(0.3)
    assert runtime.custom_sampling_percent_to_sigma(
        0.3, return_actual_sigma=False, sampling_shift=1.18452766
    ) == shifted_space.percent_to_sigma(0.3)
    with pytest.raises(ValueError, match="require a discrete sigma space"):
        runtime.custom_sampling_sd_turbo_sigmas(4, 0.65)


def test_flux2_custom_sampling_matches_the_standard_flux2_path() -> None:
    runtime = _bare_runtime(RecordingFlux(value=0.5))
    latent = torch.zeros((1, 128, 2, 2))
    cond = Conditioning(torch.zeros((1, 3, 8)), None)
    seed = 23
    sampling_shift = 1.18452766
    sigmas = runtime.custom_sampling_sigmas(
        "dinkster.normal", 2, 1.0, sampling_shift=sampling_shift
    )
    sampler = builtin_sampler_registry().get("dinkster.euler")
    assert sampler is not None

    expected = runtime.sample(
        latent,
        cond=cond,
        sampler_id=sampler.id,
        scheduler_id="dinkster.normal",
        steps=2,
        seed=seed,
        compute_dtype=torch.float32,
        sampling_shift=sampling_shift,
    )
    result = runtime.sample_custom(
        latent,
        noise=prepare_noise(latent, seed),
        cond=cond,
        cfg=None,
        request=CustomSamplingRequest(sampler, (), sigmas),
        seed=seed,
        sampling_shift=sampling_shift,
    )

    assert torch.equal(result.output, expected)


def test_flux2_check_custom_sampling_refuses_unsupported_modes() -> None:
    sampler = builtin_sampler_registry().get("dinkster.euler")
    assert sampler is not None
    request = CustomSamplingRequest(sampler, (), (1.0, 0.5, 0.0))
    runtime = _bare_runtime(RecordingFlux())
    runtime.check_custom_sampling(
        request, has_denoise_mask=True, has_inpaint=False, has_context_windows=False
    )
    with pytest.raises(Flux2RuntimeError, match="inpaint"):
        runtime.check_custom_sampling(
            request, has_denoise_mask=False, has_inpaint=True, has_context_windows=False
        )
    with pytest.raises(Flux2RuntimeError, match="does not support context windows"):
        runtime.check_custom_sampling(
            request, has_denoise_mask=False, has_inpaint=False, has_context_windows=True
        )
    unknown = replace(sampler, id="test.unregistered", aliases=())
    with pytest.raises(Flux2RuntimeError, match="unknown sampler"):
        runtime.check_custom_sampling(
            CustomSamplingRequest(unknown, (), (1.0, 0.0)),
            has_denoise_mask=False,
            has_inpaint=False,
            has_context_windows=False,
        )
    klein = _bare_runtime(RecordingFlux(distilled=False))
    with pytest.raises(Flux2RuntimeError, match="distilled-guidance"):
        klein.check_custom_sampling(
            request,
            has_denoise_mask=False,
            has_inpaint=False,
            has_context_windows=False,
            guidance=4.0,
        )


@pytest.mark.parametrize(
    "guidance",
    ([1.0], {"scale": 1.0}, torch.tensor((1.0, 2.0)), True, float("nan"), float("inf")),
)
def test_flux2_check_custom_sampling_refuses_malformed_guidance(guidance: object) -> None:
    sampler = builtin_sampler_registry().get("dinkster.euler")
    assert sampler is not None
    runtime = _bare_runtime(RecordingFlux())

    with pytest.raises(Flux2RuntimeError, match="None, 'disabled', or a finite float"):
        runtime.check_custom_sampling(
            CustomSamplingRequest(sampler, (), (1.0, 0.0)),
            has_denoise_mask=False,
            has_inpaint=False,
            has_context_windows=False,
            guidance=cast("Any", guidance),
        )


def test_flux2_diffusion_runtime_delegates_custom_sampling() -> None:
    model = RecordingFlux(value=0.5)
    diffusion = Flux2DiffusionRuntime(cast("Any", model), FLUX2_DEV, runtime_identity="native:d")
    reference = _bare_runtime(RecordingFlux(value=0.5))
    sigmas = diffusion.custom_sampling_sigmas("dinkster.normal", 2, 1.0)
    assert sigmas == reference.custom_sampling_sigmas("dinkster.normal", 2, 1.0)
    assert diffusion.custom_sampling_beta_sigmas(
        4, 0.6, 0.6
    ) == reference.custom_sampling_beta_sigmas(4, 0.6, 0.6)
    assert diffusion.custom_sampling_percent_to_sigma(
        0.3, return_actual_sigma=False
    ) == reference.custom_sampling_percent_to_sigma(0.3, return_actual_sigma=False)
    sampler = builtin_sampler_registry().get("dinkster.euler")
    assert sampler is not None
    latent = torch.zeros((1, 128, 2, 2))
    cond = Conditioning(torch.zeros((1, 3, 8)), None)
    seed = 7
    result = diffusion.sample_custom(
        latent,
        noise=prepare_noise(latent, seed),
        cond=cond,
        cfg=None,
        request=CustomSamplingRequest(sampler, (), sigmas),
        seed=seed,
    )
    expected = reference.sample_custom(
        latent,
        noise=prepare_noise(latent, seed),
        cond=cond,
        cfg=None,
        request=CustomSamplingRequest(sampler, (), sigmas),
        seed=seed,
    )
    assert torch.equal(result.output, expected.output)


class _RecordingStackModel:
    """Fake tower returning a (batch, captures, tokens, hidden) stack."""

    hidden = 2

    def __init__(self, config: QwenTextConfig) -> None:
        self.config = config
        self.embed_tokens = type("Embedding", (), {"weight": torch.empty(0)})()
        self.ids: torch.Tensor | None = None

    def __call__(
        self, ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.ids = ids
        captures = len(self.config.output_hidden_layers or ())
        count = captures * ids.shape[1] * self.hidden
        return torch.arange(count, dtype=torch.float32).reshape(
            1, captures, ids.shape[1], self.hidden
        )


class _InitAssembled:
    family = FLUX2_DEV

    def __init__(self, text_encoder: _RecordingStackModel) -> None:
        self.diffusion = RecordingFlux()
        self.text_encoder = text_encoder
        self.vae = SimpleNamespace(
            config=KLConfig(
                in_channels=3,
                out_channels=3,
                ch=4,
                decoder_ch=4,
                ch_mult=(1, 2),
                num_res_blocks=1,
                z_channels=4,
                embed_dim=4,
                quant_convs=True,
                batch_norm_latent=True,
            )
        )
        self.attention_status = "test"

    def compute_dtype(self, _component: str) -> torch.dtype:
        return torch.bfloat16


def test_flux2_runtime_init_selects_encoder_by_architecture_and_binds_codec() -> None:
    dev_tower = _RecordingStackModel(MISTRAL3_24B_PRUNED_CONFIG)
    runtime = Flux2Runtime(cast("Any", _InitAssembled(dev_tower)), runtime_identity="native:test")
    encoder = runtime._text_encoder  # pyright: ignore[reportPrivateUsage]
    assert isinstance(encoder, Flux2DevTextEncoder)
    assert encoder.model is cast("Any", dev_tower)
    conditioning = runtime.encode_text("Hello, world!")
    expected = tokenize_flux2_dev_prompt("Hello, world!", tokenizer=load_flux2_tekken_bpe())
    assert dev_tower.ids is not None
    assert dev_tower.ids.tolist() == [list(expected.ids)]
    assert conditioning.pooled is None
    assert runtime.codec.compute_dtype == torch.bfloat16
    assert runtime.codec.descriptor.id == "dinkster.autoencoder_kl"
    assert runtime.codec.encoder is cast("Any", runtime.assembled.vae)
    assert runtime.runtime_identity == "native:test"
    assert runtime.family is FLUX2_DEV

    klein_tower = _RecordingStackModel(KLEIN_QWEN3_8B_CONFIG)
    runtime = Flux2Runtime(cast("Any", _InitAssembled(klein_tower)), runtime_identity="native:test")
    klein_encoder = runtime._text_encoder  # pyright: ignore[reportPrivateUsage]
    assert isinstance(klein_encoder, Flux2KleinTextEncoder)
    assert klein_encoder.model is cast("Any", klein_tower)


def test_flux2_runtime_codec_delegates_decode_and_encode() -> None:
    runtime = object.__new__(Flux2Runtime)
    decoded = torch.zeros((1, 3, 8, 8))
    encoded = torch.zeros((1, 128, 1, 1))
    calls: list[tuple[str, torch.Tensor]] = []

    def decode(latent: torch.Tensor) -> torch.Tensor:
        calls.append(("decode", latent))
        return decoded

    def encode(content: torch.Tensor) -> torch.Tensor:
        calls.append(("encode", content))
        return encoded

    runtime.codec = cast("Any", SimpleNamespace(decode=decode, encode=encode))
    latent = torch.full((1, 128, 1, 1), 0.5)
    content = torch.full((1, 3, 8, 8), 0.5)
    assert runtime.decode_latent(latent) is decoded
    assert runtime.encode_content(content) is encoded
    assert [(name, tensor.shape) for name, tensor in calls] == [
        ("decode", latent.shape),
        ("encode", content.shape),
    ]


def test_flux2_diffusion_runtime_assembly_enrolls_for_residency() -> None:
    from dinkster_inference_torch import CastOperations, enroll_assembled

    class _Routable(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = CastOperations(torch.float32).linear(2, 2)

    diffusion = Flux2DiffusionRuntime(
        cast("Any", _Routable()), FLUX2_DEV, runtime_identity="native:d"
    )
    enrolled = enroll_assembled(
        cast("Any", diffusion.assembled),
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )
    assert tuple(enrolled) == ("diffusion",)
    assert not enrolled.storage_dtype_report.enabled
    from dinkster_inference.runtime import FamilyRuntime

    assert isinstance(diffusion, FamilyRuntime)
    with pytest.raises(Flux2RuntimeError, match="no text encoder"):
        diffusion.encode_text("prompt")
    with pytest.raises(Flux2RuntimeError, match="no VAE codec"):
        diffusion.decode_latent(torch.zeros((1, 128, 1, 1)))
    with pytest.raises(Flux2RuntimeError, match="no VAE codec"):
        diffusion.encode_content(torch.zeros((1, 3, 8, 8)))
