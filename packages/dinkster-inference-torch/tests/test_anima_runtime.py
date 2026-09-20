"""The native Anima split runtimes: flow denoising and text encoding.

The denoiser and runtime are proven over recording fakes (the real
AnimaModel math is proven against executed goldens in
test_anima_model.py); the text-encode path runs a real tiny Qwen tower
and tiny AnimaModel over the deterministic hash fill so the baked
adapter context is exercised end to end. Anima has no combined
runtime: the VAE codec rides the canonical shared Wan handle.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    AnimaConfig,
    Conditioning,
    ConditioningCarrier,
    ConditioningSet,
    CustomSamplingRequest,
    CustomSamplingRuntime,
    FlowSigmas,
    GuidanceContribution,
    GuidancePostCFGDescriptor,
    PerpNegSamplingGuidance,
    PreparedMultiStreamConditioning,
    QwenTextConfig,
    ReconstructionRecipe,
    RuntimeKnobs,
    SamplingGuidance,
    SamplingSegment,
    WeightSourceBinding,
    WeightSourceRef,
    sampling_sigmas,
    tokenize_anima_prompt,
)
from dinkster_inference_torch import (
    AnimaConditioning,
    AnimaDenoiser,
    AnimaDiffusionRuntime,
    AnimaModel,
    AnimaRuntimeError,
    AnimaTextRuntime,
    QwenTextModel,
    anima_conditioning_to_carrier,
    enroll_assembled,
    materialize_anima_conditioning,
)
from dinkster_inference_torch import sampling_execution as execution_mod
from dinkster_inference_torch.sampling_execution import run_ksampler_as_custom
from dinkster_inference_torch.schedules import (
    custom_beta_sigmas,
    custom_percent_to_sigma,
    torch_scheduler_registry,
)
from dinkster_inference_torch.solvers import torch_sampler_registry
from unet_fill import fill_state_dict


class RecordingAnima(torch.nn.Module):
    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.value = value
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def preprocess_text_embeds(
        self, hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        del ids, weights
        return torch.zeros((hidden.shape[0], 512, 1024))

    def forward(
        self, latent: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        self.calls.append((latent, timestep, context))
        return torch.full_like(latent, self.value)


def _diffusion_runtime(model: RecordingAnima) -> AnimaDiffusionRuntime:
    return AnimaDiffusionRuntime(
        cast("AnimaModel", model),
        runtime_identity="native:dinkster.anima:" + "0" * 64,
        compute_dtype=torch.float32,
        sampler_registry=torch_sampler_registry(),
        scheduler_registry=torch_scheduler_registry(),
    )


def _raw_conditioning() -> AnimaConditioning:
    return AnimaConditioning(
        torch.zeros((1, 4, 12)),
        None,
        torch.ones((1, 4), dtype=torch.int64),
        torch.ones((1, 4, 1)),
    )


def test_anima_denoiser_converts_native_flow_output_and_batches_cfg() -> None:
    model = RecordingAnima()
    evaluator = AnimaDenoiser(cast("Any", model), compute_dtype=torch.float32)
    latent = torch.full((2, 16, 1, 2, 2), 4.0)
    first = torch.zeros((1, 512, 1024))
    second = torch.ones((1, 512, 1024))
    outputs = evaluator.evaluate_conditioning_batch(
        latent, 0.25, ((first, "positive"), (second, "negative"))
    )
    assert len(outputs) == 2
    torch.testing.assert_close(outputs[0], torch.full_like(latent, 3.75))
    torch.testing.assert_close(outputs[1], torch.full_like(latent, 3.75))
    model_latent, timestep, context = model.calls[0]
    assert model_latent.shape == (4, 16, 1, 2, 2)
    assert timestep.tolist() == [0.25] * 4
    assert context.shape == (4, 512, 1024)


def test_anima_denoiser_refuses_malformed_conditioning() -> None:
    evaluator = AnimaDenoiser(cast("Any", RecordingAnima()))
    with pytest.raises(AnimaRuntimeError, match="pooled"):
        evaluator.prepare_conditioning(
            Conditioning(torch.zeros((1, 512, 1024)), torch.zeros((1, 1)))
        )
    with pytest.raises(AnimaRuntimeError, match="1024"):
        evaluator.prepare_conditioning(Conditioning(torch.zeros((1, 512, 1000)), None))
    with pytest.raises(AnimaRuntimeError, match="512"):
        evaluator.prepare_conditioning(Conditioning(torch.zeros((1, 511, 1024)), None))
    with pytest.raises(AnimaRuntimeError, match="lane"):
        evaluator.prepare_conditioning(
            Conditioning(torch.zeros((1, 512, 1024)), None), lane_id="middle"
        )


def test_anima_runtime_samples_in_shift_three_flow_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference_torch import sampling_execution

    model = RecordingAnima(value=0.0)
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
        cond=_raw_conditioning(),
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=1,
        seed=4,
        compute_dtype=torch.float32,
    )
    assert shifts == [3.0]
    assert result.shape == latent.shape
    assert model.calls


def test_anima_runtime_refuses_nonlatent_and_unsupported_inputs() -> None:
    model = RecordingAnima(value=0.0)
    runtime = _diffusion_runtime(model)
    condition = _raw_conditioning()
    with pytest.raises(AnimaRuntimeError, match="exact AnimaConditioning"):
        runtime.sample(
            torch.zeros((1, 16, 1, 2, 2)),
            cond=Conditioning(torch.zeros((1, 512, 1024)), None),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
        )
    for latent in (torch.zeros((1, 16, 2, 2)), torch.zeros((1, 4, 1, 2, 2))):
        with pytest.raises(AnimaRuntimeError, match="Anima input must have shape"):
            runtime.sample(
                latent,
                cond=condition,
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.simple",
                steps=1,
            )
    latent = torch.zeros((1, 16, 1, 2, 2))
    with pytest.raises(AnimaRuntimeError, match="distilled-guidance"):
        runtime.sample(
            latent,
            cond=condition,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
            guidance=4.0,
        )
    with pytest.raises(AnimaRuntimeError, match="inpaint"):
        runtime.sample(
            latent,
            cond=condition,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
            inpaint=cast("Any", object()),
        )
    assert not model.calls


class ArithmeticAnima(RecordingAnima):
    """Input-dependent fake: the output couples latent, timestep, and
    context so schedule, noise, and CFG-lane parity are all visible."""

    def preprocess_text_embeds(
        self, hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        del ids, weights
        return torch.full((hidden.shape[0], 512, 1024), float(hidden.float().mean()))

    def forward(
        self, latent: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        self.calls.append((latent, timestep, context))
        scale = context.float().mean(dim=(1, 2)).reshape(-1, 1, 1, 1, 1)
        return latent * 0.5 + timestep.reshape(-1, 1, 1, 1, 1) * 0.25 + scale


def _filled_conditioning(fill: float) -> AnimaConditioning:
    return AnimaConditioning(
        torch.full((1, 4, 12), fill),
        None,
        torch.ones((1, 4), dtype=torch.int64),
        torch.ones((1, 4, 1)),
    )


def _custom_request(
    sampler_id: str = "dinkster.euler",
    sigmas: tuple[float, ...] = (1.0, 0.5, 0.0),
) -> CustomSamplingRequest[torch.Tensor]:
    descriptor = torch_sampler_registry().get(sampler_id)
    assert descriptor is not None
    return CustomSamplingRequest(descriptor, (), sigmas)


def test_anima_runtime_satisfies_the_custom_sampling_protocol() -> None:
    assert isinstance(_diffusion_runtime(RecordingAnima()), CustomSamplingRuntime)


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
    runtime = _diffusion_runtime(ArithmeticAnima())
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
        space=FlowSigmas(shift=3.0),
        flow=True,
        sampler_id=sampler_id,
        scheduler_id=scheduler_id,
        steps=steps,
        denoise=1.0,
        seed=185,
        cond=positive,
        cfg=SamplingGuidance(negative, 1.0 if cfg_scale is None else cfg_scale),
        segment=segment,
        error=AnimaRuntimeError,
    )

    output = result.output
    assert type(output) is torch.Tensor
    assert torch.equal(output, expected)


def test_custom_sampling_sigma_surfaces_match_shift_three_flow_space() -> None:
    runtime = _diffusion_runtime(RecordingAnima())
    space = FlowSigmas(shift=3.0)
    scheduler = torch_scheduler_registry().get("dinkster.simple")
    assert scheduler is not None
    assert runtime.custom_sampling_sigmas("dinkster.simple", 4, 0.5) == sampling_sigmas(
        scheduler, space, 4, denoise=0.5
    )
    with pytest.raises(AnimaRuntimeError, match="unknown scheduler"):
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
    model = RecordingAnima(value=0.0)
    runtime = _diffusion_runtime(model)
    condition = _raw_conditioning()
    latent = torch.zeros((1, 16, 1, 2, 2))
    noise = torch.zeros_like(latent)
    request = _custom_request()

    def sample(**overrides: Any) -> Any:
        arguments: dict[str, Any] = {"noise": noise, "cond": condition, "request": request}
        arguments.update(overrides)
        return runtime.sample_custom(arguments.pop("latent", latent), **arguments)

    with pytest.raises(AnimaRuntimeError, match="prepared multi-stream payload"):
        sample(cond=PreparedMultiStreamConditioning("native:test", object()))
    with pytest.raises(AnimaRuntimeError, match="perp-neg"):
        sample(cfg=PerpNegSamplingGuidance(condition, condition, 3.0, 1.0))
    with pytest.raises(AnimaRuntimeError, match="Anima input must have shape"):
        sample(latent=torch.zeros((1, 16, 2, 2)), noise=torch.zeros((1, 16, 2, 2)))
    with pytest.raises(AnimaRuntimeError, match="Anima input must have shape"):
        sample(latent=torch.zeros((1, 4, 1, 2, 2)), noise=torch.zeros((1, 4, 1, 2, 2)))
    with pytest.raises(AnimaRuntimeError, match="distilled-guidance"):
        sample(guidance=4.0)
    with pytest.raises(AnimaRuntimeError, match="inpaint"):
        sample(inpaint=cast("Any", object()))
    euler = torch_sampler_registry().get("dinkster.euler")
    assert euler is not None
    unknown = replace(euler, id="test.missing", aliases=())
    with pytest.raises(AnimaRuntimeError, match="unknown sampler"):
        sample(request=CustomSamplingRequest(unknown, (), (1.0, 0.0)))
    with pytest.raises(AnimaRuntimeError, match="exact AnimaConditioning"):
        sample(cond=Conditioning(torch.zeros((1, 512, 1024)), None))
    with pytest.raises(AnimaRuntimeError, match="adapter options: bogus_option"):
        sample(bogus_option=True)
    with pytest.raises(AnimaRuntimeError, match="adapter options: bogus_option"):
        runtime.sample(
            latent,
            cond=condition,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
            bogus_option=True,
        )
    assert not model.calls


def test_sample_custom_captures_denoised_output() -> None:
    runtime = _diffusion_runtime(RecordingAnima(value=0.0))
    latent = torch.zeros((1, 16, 1, 2, 2))
    result = runtime.sample_custom(
        latent,
        noise=torch.zeros_like(latent),
        cond=_raw_conditioning(),
        request=_custom_request(),
        seed=9,
    )
    assert result.output.shape == latent.shape
    assert result.denoised_output is not None
    assert result.denoised_output.shape == latent.shape


def _tiny_qwen_config() -> QwenTextConfig:
    return QwenTextConfig(
        architecture="anima_qwen3_06b",
        vocab_size=151936,
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        qkv_bias=False,
        qk_norm=True,
        prompt_template="{}",
        min_tokens=1,
        pad_token_id=151643,
        zero_masked=False,
        layer_norm_hidden_state=False,
    )


def _tiny_anima_config() -> AnimaConfig:
    """The golden proof architecture, adapter source width matched to
    the tiny Qwen tower (the frozen AnimaConfig only represents the
    full 2B profile, so the tiny architecture is duck-typed)."""
    return cast(
        "AnimaConfig",
        SimpleNamespace(
            blocks=1,
            hidden_width=48,
            attention_heads=2,
            attention_head_dim=24,
            context_width=16,
            adaln_lora_dim=8,
            patchified_input_channels=20,
            output_latent_channels=4,
            patch=(1, 2, 2),
            latent_channels=4,
            adapter_blocks=1,
            adapter_width=16,
            adapter_heads=2,
            adapter_head_dim=8,
            adapter_source_width=12,
            adapter_vocabulary=32128,
        ),
    )


def _hash_filled(model: torch.nn.Module) -> torch.nn.Module:
    entries = [(key, list(value.shape)) for key, value in model.state_dict().items()]
    model.load_state_dict(fill_state_dict(entries), strict=True)
    return model


def test_split_text_and_diffusion_bake_padded_adapter_context() -> None:
    encoder = cast("QwenTextModel", _hash_filled(QwenTextModel(_tiny_qwen_config())))
    diffusion = cast("AnimaModel", _hash_filled(AnimaModel(_tiny_anima_config())))
    split_text = AnimaTextRuntime(encoder)
    split_diffusion = AnimaDiffusionRuntime(
        diffusion,
        runtime_identity="native:dinkster.anima:" + "1" * 64,
        compute_dtype=torch.float32,
    )

    with torch.no_grad():
        raw = split_text.encode_text("a watercolor cat")
        actual = split_diffusion._bake_conditioning(raw)  # pyright: ignore[reportPrivateUsage]

    tokens = tokenize_anima_prompt("a watercolor cat")
    assert raw.embeddings.shape == (1, len(tokens.qwen_ids), 12)
    assert raw.t5xxl_ids.shape[0] == 1
    assert raw.t5xxl_weights.shape == (*raw.t5xxl_ids.shape, 1)
    assert actual.pooled is None
    context = actual.embeddings
    assert context.shape == (1, 512, 16)
    assert context.dtype == torch.float32
    assert torch.isfinite(context).all()
    assert torch.any(context[:, : len(tokens.t5xxl_ids)] != 0)
    assert torch.all(context[:, len(tokens.t5xxl_ids) :] == 0)


def test_split_diffusion_sample_forwards_per_run_guidance_transforms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = cast("QwenTextModel", _hash_filled(QwenTextModel(_tiny_qwen_config())))
    diffusion = cast("AnimaModel", _hash_filled(AnimaModel(_tiny_anima_config())))
    split_text = AnimaTextRuntime(encoder)
    runtime = AnimaDiffusionRuntime(
        diffusion,
        runtime_identity="native:dinkster.anima:" + "5" * 64,
        compute_dtype=torch.float32,
    )
    captured: list[SamplingGuidance[Conditioning[torch.Tensor]] | None] = []
    real_plan = execution_mod.compile_guidance_plan

    def capture_plan(*args: Any, **kwargs: Any) -> Any:
        captured.append(args[1])
        return real_plan(*args, **kwargs)

    def fake_denoiser(*_args: Any, **_kwargs: Any) -> object:
        return object()

    def fake_run_denoise(*_args: Any, **kwargs: Any) -> Any:
        return kwargs["latent"]

    monkeypatch.setattr(execution_mod, "compile_guidance_plan", capture_plan)
    monkeypatch.setattr(execution_mod, "guided_denoiser", fake_denoiser)
    monkeypatch.setattr(execution_mod, "run_denoise", fake_run_denoise)
    transforms = (
        (
            "ext",
            GuidanceContribution(
                post_cfg=(GuidancePostCFGDescriptor("ext.noop", lambda context: context.reduced),)
            ),
        ),
    )
    with torch.no_grad():
        raw = split_text.encode_text("a cat")
        runtime.sample(
            torch.zeros(1, 16, 1, 2, 2),
            cond=raw,
            cfg=SamplingGuidance(raw, 3.0, transforms),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=1,
        )
    baked = captured[0]
    assert baked is not None
    assert baked.transforms == transforms
    assert baked.uncond is not None
    assert not isinstance(baked.uncond, AnimaConditioning)


def test_split_diffusion_assembly_enrolls_in_native_residency() -> None:
    diffusion = cast("AnimaModel", _hash_filled(AnimaModel(_tiny_anima_config())))
    runtime = AnimaDiffusionRuntime(
        diffusion,
        runtime_identity="native:dinkster.anima:" + "3" * 64,
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


def test_split_conditioning_carrier_round_trips_all_adapter_inputs() -> None:
    value = AnimaConditioning(
        torch.arange(24, dtype=torch.float32).reshape(1, 2, 12),
        None,
        torch.tensor([[4, 8, 15]], dtype=torch.int64),
        torch.tensor([[[0.5], [1.0], [1.5]]], dtype=torch.float32),
    )

    carrier = anima_conditioning_to_carrier(value)
    materialized = materialize_anima_conditioning(carrier, device="cpu")

    torch.testing.assert_close(materialized.embeddings, value.embeddings)
    torch.testing.assert_close(materialized.t5xxl_ids, value.t5xxl_ids)
    torch.testing.assert_close(materialized.t5xxl_weights, value.t5xxl_weights)
    assert materialized.pooled is None


def test_split_conditioning_refuses_malformed_adapter_metadata() -> None:
    value = AnimaConditioning(
        torch.zeros((1, 2, 12)),
        None,
        torch.ones((1, 2), dtype=torch.int64),
        torch.ones((1, 2, 1)),
    )
    carrier = anima_conditioning_to_carrier(value)
    record = carrier.conditioning.records[0]
    metadata = dict(record.extension_metadata)
    weights = metadata.pop("dinkster-anima/t5xxl-weights")
    metadata["foreign/weights"] = weights
    malformed = ConditioningCarrier(
        ConditioningSet((replace(record, extension_metadata=tuple(metadata.items())),)),
        carrier.bindings,
    )

    with pytest.raises(AnimaRuntimeError, match="requires T5 ids and weights"):
        materialize_anima_conditioning(malformed, device="cpu")


def test_split_conditioning_refuses_wrong_tensor_contracts() -> None:
    with pytest.raises(AnimaRuntimeError, match="raw context"):
        AnimaConditioning(
            torch.zeros((2, 12)),
            None,
            torch.ones((1, 2), dtype=torch.int64),
            torch.ones((1, 2, 1)),
        )
    with pytest.raises(AnimaRuntimeError, match="T5 ids"):
        AnimaConditioning(
            torch.zeros((1, 2, 12)),
            None,
            torch.ones((1, 2), dtype=torch.float32),
            torch.ones((1, 2, 1)),
        )
    with pytest.raises(AnimaRuntimeError, match="T5 weights"):
        AnimaConditioning(
            torch.zeros((1, 2, 12)),
            None,
            torch.ones((1, 2), dtype=torch.int64),
            torch.ones((1, 2)),
        )


def _native_residency_module() -> Any:
    root = Path(__file__).resolve().parents[3]
    for source in sorted((root / "packages").glob("*/src")):
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    return importlib.import_module("dinkster_compat_comfy.native_residency")


def _anima_diffusion_recipe() -> ReconstructionRecipe:
    return ReconstructionRecipe(
        sources=(
            WeightSourceBinding(
                "diffusion",
                WeightSourceRef("blake3:" + "0" * 64, "anima-diffusion.safetensors", 0),
            ),
        ),
        family_id="dinkster.anima",
        component_identity=("family=dinkster.anima",),
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
    diffusion = cast("AnimaModel", _hash_filled(AnimaModel(_tiny_anima_config())))
    recipe = _anima_diffusion_recipe()
    runtime = AnimaDiffusionRuntime(
        diffusion,
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
    class _MisdeclaredRuntime(AnimaDiffusionRuntime):
        streamed_residency_components = frozenset({"qwen3_06b"})

    native_residency = _native_residency_module()
    diffusion = cast("AnimaModel", _hash_filled(AnimaModel(_tiny_anima_config())))
    recipe = _anima_diffusion_recipe()
    runtime = _MisdeclaredRuntime(
        diffusion,
        runtime_identity=recipe.runtime_identity,
        compute_dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="streams unknown components: qwen3_06b"):
        native_residency.NativeRuntimeHandle(
            runtime,
            "cpu",
            recipe=recipe,
            coordinator=_cpu_coordinator(native_residency),
        )
