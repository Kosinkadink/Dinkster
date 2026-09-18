from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    KREA2_SIGMAS,
    Conditioning,
    CustomSamplingRequest,
    CustomSamplingRuntime,
    PerpNegSamplingGuidance,
    PreparedMultiStreamConditioning,
    ReconstructionRecipe,
    RuntimeKnobs,
    SamplingGuidance,
    SamplingSegment,
    WeightSourceBinding,
    WeightSourceRef,
    sampling_sigmas,
)
from dinkster_inference_torch import (
    Krea2DiffusionRuntime,
    Krea2DiT,
    Krea2RuntimeError,
    Krea2TextRuntime,
    enroll_assembled,
    krea2_language_model,
)
from dinkster_inference_torch import krea2_runtime as runtime_mod
from dinkster_inference_torch.sampling_execution import run_ksampler_as_custom
from dinkster_inference_torch.schedules import (
    custom_beta_sigmas,
    custom_percent_to_sigma,
    torch_scheduler_registry,
)
from dinkster_inference_torch.solvers import torch_sampler_registry
from test_krea2_dit import CASES, build_model


class RecordingKrea2(torch.nn.Module):
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


def _diffusion_runtime(model: RecordingKrea2) -> Krea2DiffusionRuntime:
    return Krea2DiffusionRuntime(
        cast("Krea2DiT", model),
        runtime_identity="native:dinkster.krea2:" + "0" * 64,
        compute_dtype=torch.float32,
        sampler_registry=torch_sampler_registry(),
        scheduler_registry=torch_scheduler_registry(),
    )


def test_krea2_denoiser_converts_native_flow_output_and_batches_cfg() -> None:
    model = RecordingKrea2()
    evaluator = runtime_mod._Krea2Denoiser(  # pyright: ignore[reportPrivateUsage]
        cast("Any", model), compute_dtype=torch.float32
    )
    latent = torch.full((2, 16, 1, 2, 2), 4.0)
    first = torch.zeros((1, 3, 30720))
    second = torch.ones((1, 3, 30720))
    outputs = evaluator.evaluate_conditioning_batch(latent, 0.25, (first, second))
    assert len(outputs) == 2
    torch.testing.assert_close(outputs[0], torch.full_like(latent, 3.75))
    torch.testing.assert_close(outputs[1], torch.full_like(latent, 3.75))
    model_latent, timestep, context = model.calls[0]
    assert model_latent.shape == (4, 16, 1, 2, 2)
    assert timestep.tolist() == [0.25] * 4
    assert context.shape == (4, 3, 30720)


def test_krea2_denoiser_refuses_invalid_conditioning() -> None:
    evaluator = runtime_mod._Krea2Denoiser(  # pyright: ignore[reportPrivateUsage]
        cast("Any", RecordingKrea2())
    )
    with pytest.raises(Krea2RuntimeError, match="Conditioning value"):
        evaluator.prepare_conditioning(torch.zeros((1, 2, 30720)))
    with pytest.raises(Krea2RuntimeError, match="pooled"):
        evaluator.prepare_conditioning(
            Conditioning(torch.zeros((1, 2, 30720)), torch.zeros((1, 1)))
        )
    with pytest.raises(Krea2RuntimeError, match="30720"):
        evaluator.prepare_conditioning(Conditioning(torch.zeros((1, 2, 2560)), None))
    with pytest.raises(Krea2RuntimeError, match="empty or incompatible"):
        evaluator.evaluate_conditioning_batch(torch.zeros((1, 16, 1, 2, 2)), 0.5, ())
    with pytest.raises(Krea2RuntimeError, match="one or match"):
        evaluator.evaluate_conditioning_batch(
            torch.zeros((2, 16, 1, 2, 2)),
            0.5,
            (torch.zeros((3, 2, 30720)),),
        )


def test_krea2_diffusion_runtime_streams_no_components() -> None:
    assert Krea2DiffusionRuntime.streamed_residency_components == frozenset()


def test_krea2_runtime_samples_in_shift_flow_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference_torch import sampling_execution

    model = RecordingKrea2(value=0.0)
    runtime = _diffusion_runtime(model)
    shifts: list[float] = []
    original = sampling_execution.build_sampling_schedule

    def capture(*args: Any, **kwargs: Any) -> Any:
        shifts.append(args[1].shift)
        return original(*args, **kwargs)

    monkeypatch.setattr(sampling_execution, "build_sampling_schedule", capture)
    latent = torch.zeros((1, 16, 1, 2, 2))
    result = runtime.sample(
        latent,
        cond=Conditioning(torch.zeros((1, 3, 30720)), None),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=1,
        seed=4,
        compute_dtype=torch.float32,
    )
    assert shifts == [1.15]
    assert result.shape == latent.shape
    assert model.calls


def test_krea2_runtime_refuses_nonlatent_and_unsupported_inputs() -> None:
    model = RecordingKrea2(value=0.0)
    runtime = _diffusion_runtime(model)
    condition = Conditioning(torch.zeros((1, 2, 30720)), None)
    with pytest.raises(Krea2RuntimeError, match="16"):
        runtime.sample(
            torch.zeros((1, 16, 2, 2)),
            cond=condition,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
        )
    with pytest.raises(Krea2RuntimeError, match="16"):
        runtime.sample(
            torch.zeros((1, 16, 2, 2, 2)),
            cond=condition,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
        )
    latent = torch.zeros((1, 16, 1, 2, 2))
    with pytest.raises(Krea2RuntimeError, match="distilled-guidance"):
        runtime.sample(
            latent,
            cond=condition,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
            guidance=3.5,
        )
    with pytest.raises(Krea2RuntimeError, match="inpaint"):
        runtime.sample(
            latent,
            cond=condition,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
            inpaint=cast("Any", object()),
        )
    assert not model.calls


class ArithmeticKrea2(RecordingKrea2):
    """Input-dependent fake: the output couples latent, timestep, and
    context so schedule, noise, and CFG-lane parity are all visible."""

    def forward(
        self, latent: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        self.calls.append((latent, timestep, context))
        scale = context.float().mean(dim=(1, 2)).reshape(-1, 1, 1, 1, 1)
        return latent * 0.5 + timestep.reshape(-1, 1, 1, 1, 1) * 0.25 + scale


def _filled_conditioning(fill: float) -> Conditioning[torch.Tensor]:
    return Conditioning(torch.full((1, 4, 30720), fill), None)


def _custom_request(
    sampler_id: str = "dinkster.euler",
    sigmas: tuple[float, ...] = (1.0, 0.5, 0.0),
) -> CustomSamplingRequest[torch.Tensor]:
    descriptor = torch_sampler_registry().get(sampler_id)
    assert descriptor is not None
    return CustomSamplingRequest(descriptor, (), sigmas)


def test_krea2_runtime_satisfies_the_custom_sampling_protocol() -> None:
    assert isinstance(_diffusion_runtime(RecordingKrea2()), CustomSamplingRuntime)


@pytest.mark.parametrize(
    ("sampler_id", "scheduler_id", "steps", "cfg_scale", "segment"),
    [
        ("dinkster.euler", "dinkster.simple", 2, None, None),
        ("dinkster.res_multistep", "dinkster.simple", 2, 2.0, None),
        ("dinkster.dpmpp_sde", "dinkster.simple", 3, None, None),
        (
            "dinkster.euler",
            "dinkster.simple",
            3,
            None,
            SamplingSegment(
                steps=3,
                start_step=1,
                end_step=3,
                add_noise=False,
                return_with_leftover_noise=False,
            ),
        ),
    ],
    ids=("euler", "cfg", "brownian", "segment"),
)
def test_ksampler_surface_is_bit_equal_sugar_over_sample_custom(
    sampler_id: str,
    scheduler_id: str,
    steps: int,
    cfg_scale: float | None,
    segment: SamplingSegment | None,
) -> None:
    runtime = _diffusion_runtime(ArithmeticKrea2())
    positive = _filled_conditioning(2.0)
    negative = None if cfg_scale is None else _filled_conditioning(5.0)
    generator = torch.Generator().manual_seed(11)
    latent = torch.rand((1, 16, 1, 2, 2), generator=generator)

    expected = runtime.sample(
        latent,
        cond=positive,
        cfg=SamplingGuidance(negative, 1.0 if cfg_scale is None else cfg_scale),
        sampler_id=sampler_id,
        scheduler_id=scheduler_id,
        steps=steps,
        denoise=1.0,
        seed=185,
        segment=segment,
        compute_dtype=torch.float32,
    )
    result = run_ksampler_as_custom(
        runtime,
        latent,
        samplers=cast("Any", runtime)._samplers,
        schedulers=cast("Any", runtime)._schedulers,
        space=KREA2_SIGMAS,
        flow=True,
        sampler_id=sampler_id,
        scheduler_id=scheduler_id,
        steps=steps,
        denoise=1.0,
        seed=185,
        cond=positive,
        cfg=SamplingGuidance(negative, 1.0 if cfg_scale is None else cfg_scale),
        segment=segment,
        error=Krea2RuntimeError,
    )

    output = result.output
    assert type(output) is torch.Tensor
    assert torch.equal(output, expected)


def test_custom_sampling_sigma_surfaces_match_krea2_flow_space() -> None:
    runtime = _diffusion_runtime(RecordingKrea2())
    space = KREA2_SIGMAS
    scheduler = torch_scheduler_registry().get("dinkster.simple")
    assert scheduler is not None
    assert runtime.custom_sampling_sigmas("dinkster.simple", 4, 0.5) == sampling_sigmas(
        scheduler, space, 4, denoise=0.5
    )
    with pytest.raises(Krea2RuntimeError, match="unknown scheduler"):
        runtime.custom_sampling_sigmas("test.missing", 4, 1.0)
    assert runtime.custom_sampling_beta_sigmas(4, 0.6, 0.6) == custom_beta_sigmas(
        space, 4, 0.6, 0.6
    )
    with pytest.raises(ValueError, match="discrete sigma space"):
        runtime.custom_sampling_sd_turbo_sigmas(2, 1.0)
    assert runtime.custom_sampling_percent_to_sigma(
        0.5, return_actual_sigma=False
    ) == custom_percent_to_sigma(space, space.percent_to_sigma, 0.5, return_actual_sigma=False)
    assert runtime.custom_sampling_percent_to_sigma(
        0.3, return_actual_sigma=True
    ) == custom_percent_to_sigma(space, space.percent_to_sigma, 0.3, return_actual_sigma=True)


def test_sample_custom_refuses_multistream_shapes_and_unsupported_modes() -> None:
    model = RecordingKrea2(value=0.0)
    runtime = _diffusion_runtime(model)
    condition = Conditioning(torch.zeros((1, 3, 30720)), None)
    latent = torch.zeros((1, 16, 1, 2, 2))
    noise = torch.zeros_like(latent)
    request = _custom_request()

    def sample(**overrides: Any) -> Any:
        arguments: dict[str, Any] = {"noise": noise, "cond": condition, "request": request}
        arguments.update(overrides)
        return runtime.sample_custom(arguments.pop("latent", latent), **arguments)

    with pytest.raises(Krea2RuntimeError, match="prepared multi-stream payload"):
        sample(cond=PreparedMultiStreamConditioning("native:test", object()))
    with pytest.raises(Krea2RuntimeError, match="perp-neg"):
        sample(cfg=PerpNegSamplingGuidance(condition, condition, 3.0, 1.0))
    with pytest.raises(Krea2RuntimeError, match="Krea 2 latent must have shape"):
        sample(latent=torch.zeros((1, 16, 2, 2)), noise=torch.zeros((1, 16, 2, 2)))
    with pytest.raises(Krea2RuntimeError, match="Krea 2 latent must have shape"):
        sample(latent=torch.zeros((1, 4, 1, 2, 2)), noise=torch.zeros((1, 4, 1, 2, 2)))
    with pytest.raises(Krea2RuntimeError, match="distilled-guidance"):
        sample(guidance=4.0)
    with pytest.raises(Krea2RuntimeError, match="inpaint"):
        sample(inpaint=cast("Any", object()))
    with pytest.raises(Krea2RuntimeError, match="context windows"):
        sample(context_windows=cast("Any", object()))
    euler = torch_sampler_registry().get("dinkster.euler")
    assert euler is not None
    unknown = replace(euler, id="test.missing", aliases=())
    with pytest.raises(Krea2RuntimeError, match="unknown sampler"):
        sample(request=CustomSamplingRequest(unknown, (), (1.0, 0.0)))
    assert not model.calls


def test_sample_custom_captures_denoised_output() -> None:
    runtime = _diffusion_runtime(RecordingKrea2(value=0.0))
    latent = torch.zeros((1, 16, 1, 2, 2))
    result = runtime.sample_custom(
        latent,
        noise=torch.zeros_like(latent),
        cond=Conditioning(torch.zeros((1, 3, 30720)), None),
        request=_custom_request(),
        seed=9,
    )
    assert result.output.shape == latent.shape
    assert result.denoised_output is not None
    assert result.denoised_output.shape == latent.shape


def test_krea2_diffusion_component_carries_no_text_encoder_or_codec() -> None:
    runtime = _diffusion_runtime(RecordingKrea2())
    with pytest.raises(Krea2RuntimeError, match="carries no text encoder"):
        runtime.encode_text("a red fox")
    with pytest.raises(Krea2RuntimeError, match="carries no VAE codec"):
        runtime.decode_latent(torch.zeros((1, 16, 1, 2, 2)))
    with pytest.raises(Krea2RuntimeError, match="carries no VAE codec"):
        runtime.encode_content(torch.zeros((1, 3, 4, 4)))


class _RecordingKrea2LanguageModel:
    def __init__(self, shape: Any) -> None:
        self.shape = shape
        self.embed_tokens = type("Embedding", (), {"weight": torch.empty(0)})()
        self.ids: torch.Tensor | None = None

    def validate_sequence_length(self, length: int) -> None:
        if length > self.shape.max_position_embeddings:
            raise ValueError

    def tapped_states(
        self,
        ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        tap_layers: tuple[int, ...],
    ) -> torch.Tensor:
        del attention_mask
        self.ids = ids
        length = ids.shape[1]
        positions = torch.arange(length, dtype=torch.float32).reshape(1, 1, length, 1)
        return positions.expand(1, len(tap_layers), length, 3).clone()


def test_krea2_text_runtime_encodes_over_its_resident_component() -> None:
    with torch.device("meta"):
        shape = krea2_language_model().shape
    model = _RecordingKrea2LanguageModel(shape)
    runtime = Krea2TextRuntime(cast("Any", model))
    got = runtime.encode_text("a red fox")
    assert runtime.text is cast("Any", model)
    assert model.ids is not None
    assert got.pooled is None
    assert got.embeddings.ndim == 3
    assert got.embeddings.shape[0] == 1


def test_split_diffusion_assembly_enrolls_in_native_residency() -> None:
    runtime = Krea2DiffusionRuntime(
        build_model(CASES[0]),
        runtime_identity="native:dinkster.krea2:" + "3" * 64,
        compute_dtype=torch.float32,
    )

    enrolled = enroll_assembled(
        cast("Any", runtime.assembled),
        load_device="cpu",
        offload_device="cpu",
    )

    assert tuple(enrolled) == ("diffusion",)
    assert runtime.streamed_residency_components <= enrolled.keys()
    assert runtime.assembled.compute_dtype("diffusion") is torch.float32


def _native_residency_module() -> Any:
    root = Path(__file__).resolve().parents[3]
    for source in sorted((root / "packages").glob("*/src")):
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    return importlib.import_module("dinkster_compat_comfy.native_residency")


def _krea2_diffusion_recipe() -> ReconstructionRecipe:
    return ReconstructionRecipe(
        sources=(
            WeightSourceBinding(
                "diffusion",
                WeightSourceRef("blake3:" + "0" * 64, "krea2-diffusion.safetensors", 0),
            ),
        ),
        family_id="dinkster.krea2",
        component_identity=("family=dinkster.krea2",),
        knobs=RuntimeKnobs(
            diffusion_dtype="float32",
            text_dtype="float32",
            vae_dtype="float32",
            fp8_matmul=False,
        ),
    )


def _cpu_coordinator(native_residency: Any) -> Any:
    from dinkster_inference_torch.memory import DeviceMemory, MemoryPolicy
    from dinkster_inference_torch.residency import ResidencyManager

    manager = ResidencyManager(
        policy=MemoryPolicy(
            inference_reserve=0,
            physical_headroom=0,
            min_weight_memory_ratio=1.0,
            load_inflation=1.0,
        ),
        free_memory=lambda _device: DeviceMemory(free_total=1 << 40, free_torch=0),
    )
    return native_residency.NativeResidencyCoordinator(manager)


def test_native_runtime_handle_constructs_over_split_diffusion_runtime() -> None:
    """The production handle builder accepts the diffusion-only runtime.

    Guards the streamed-declaration/enrollment contract at the real
    NativeRuntimeHandle constructor: a streamed component the assembly does
    not enroll must fail construction, and the shipped runtime must not
    carry one.
    """
    native_residency = _native_residency_module()
    recipe = _krea2_diffusion_recipe()
    runtime = Krea2DiffusionRuntime(
        build_model(CASES[0]),
        runtime_identity=recipe.runtime_identity,
        compute_dtype=torch.float32,
    )

    handle = native_residency.NativeRuntimeHandle(
        runtime,
        "cpu",
        recipe=recipe,
        coordinator=_cpu_coordinator(native_residency),
    )

    assert len(handle.mechanisms) == 1
    with handle.stage("diffusion"):
        assert handle.mechanisms[0].loaded_bytes() > 0


def test_native_runtime_handle_rejects_streamed_component_not_enrolled() -> None:
    class _MisdeclaredRuntime(Krea2DiffusionRuntime):
        streamed_residency_components = frozenset({"qwen3vl_4b"})

    native_residency = _native_residency_module()
    recipe = _krea2_diffusion_recipe()
    runtime = _MisdeclaredRuntime(
        build_model(CASES[0]),
        runtime_identity=recipe.runtime_identity,
        compute_dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="streams unknown components: qwen3vl_4b"):
        native_residency.NativeRuntimeHandle(
            runtime,
            "cpu",
            recipe=recipe,
            coordinator=_cpu_coordinator(native_residency),
        )
