from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    QWEN_IMAGE,
    QWEN_IMAGE_CONFIG,
    QWEN_IMAGE_LAYERED_CONFIG,
    ComponentBinding,
    Conditioning,
    ConditioningChannel,
    ConditioningSet,
    ControlApplication,
    CustomSamplingRequest,
    CustomSamplingRuntime,
    FlowSigmas,
    FluxFlowSigmas,
    PayloadReference,
    PercentRange,
    PerpNegSamplingGuidance,
    PreparedMultiStreamConditioning,
    ReconstructionRecipe,
    RuntimeKnobs,
    SamplingDescriptor,
    SamplingGuidance,
    SamplingSegment,
    SamplingSpaceOverrideRuntime,
    WeightSourceBinding,
    WeightSourceRef,
    bind_component_conditioning,
    sampling_sigmas,
)
from dinkster_inference.sampling_wire import SigmaSchedule
from dinkster_inference_torch import basic_conditioning_to_carrier, materialize_basic_conditioning
from dinkster_inference_torch import qwen_image_control as control_module
from dinkster_inference_torch import qwen_image_runtime as runtime
from dinkster_inference_torch import sampling_execution as sampling_execution_module
from dinkster_inference_torch.denoise import prepare_noise
from dinkster_inference_torch.operations import InitlessOperations
from dinkster_inference_torch.qwen_image import QwenImage
from dinkster_inference_torch.qwen_image_assembly import (
    AssembledQwenImage,
)
from dinkster_inference_torch.qwen_image_control import (
    QwenImageControlConditioning,
    QwenImageDiffSynthConditioning,
    QwenImageDiffSynthExecution,
    QwenImageDiffSynthPatch,
    QwenImageFunControlNet,
    qwen_image_control_hint_digest,
    qwen_image_control_resource_digest,
    qwen_image_diffsynth_resource_digest,
)
from dinkster_inference_torch.qwen_image_runtime import (
    QwenImageConditioning,
    QwenImageRuntime,
    QwenImageRuntimeError,
    WanVAECodecRuntime,
    materialize_qwen_image_conditioning,
    qwen_image_conditioning_to_carrier,
)
from dinkster_inference_torch.qwen_image_text import QwenImageTextModel
from dinkster_inference_torch.sampling_execution import (
    SamplingSchedule,
    build_sampling_schedule,
    run_ksampler_as_custom,
)
from dinkster_inference_torch.schedules import (
    custom_beta_sigmas,
    custom_percent_to_sigma,
    torch_scheduler_registry,
)
from dinkster_inference_torch.solvers import torch_sampler_registry
from dinkster_inference_torch.wan21_vae import WanVAE


def test_independent_wan_codec_component_encodes_and_decodes() -> None:
    class VAE:
        def encode(self, value: torch.Tensor) -> torch.Tensor:
            return value[:, :1]

        def process_in(self, value: torch.Tensor) -> torch.Tensor:
            return value + 2

        def process_out(self, value: torch.Tensor) -> torch.Tensor:
            return value - 2

        def decode(self, value: torch.Tensor) -> torch.Tensor:
            return value.repeat(1, 3, 1, 1, 1)

    codec = WanVAECodecRuntime(VAE())
    pixels = torch.full((1, 3, 2, 2), 0.75)
    latent = codec.encode_content(pixels)

    assert latent.shape == (1, 1, 1, 2, 2)
    assert torch.equal(codec.decode_latent(latent), pixels)
    layered = WanVAECodecRuntime(VAE(), layered=True).decode_latent(
        torch.full((1, 1, 4, 2, 2), 2.5)
    )
    assert layered.shape == (1, 3, 4, 2, 2)


def test_qwen_text_runtime_places_ids_on_the_bound_compute_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[torch.device] = []

    class Text(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = SimpleNamespace(embed_tokens=torch.nn.Embedding(2, 2))

        def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, None]:
            seen.append(ids.device)
            return torch.empty((1, ids.shape[1], 3584), device=ids.device), None

    text = Text()

    def bound_device(module: torch.nn.Module) -> torch.device:
        assert module is text.model.embed_tokens
        return torch.device("meta")

    def encode(_text: str) -> list[int]:
        return [1]

    monkeypatch.setattr(runtime, "bound_compute_device", bound_device)
    monkeypatch.setattr(
        runtime,
        "load_qwen_bpe",
        lambda: SimpleNamespace(encode=encode),
    )

    encoded = runtime.QwenImageTextRuntime(text).encode_text("prompt")

    assert seen == [torch.device("meta")]
    assert encoded.embeddings.device.type == "meta"


def test_qwen_conditioning_carrier_preserves_basic_and_variant_payloads() -> None:
    basic = QwenImageConditioning(
        torch.arange(12, dtype=torch.float32).reshape(1, 3, 4),
        torch.arange(4, dtype=torch.float32).reshape(1, 4),
    )
    basic_carrier = basic_conditioning_to_carrier(basic)
    generic_basic = materialize_basic_conditioning(basic_carrier, device="cpu")
    qwen_basic = materialize_qwen_image_conditioning(basic_carrier, device="cpu")
    assert torch.equal(qwen_basic.embeddings, generic_basic.embeddings)
    assert qwen_basic.pooled is not None
    assert generic_basic.pooled is not None
    assert torch.equal(qwen_basic.pooled, generic_basic.pooled)

    basic_round_trip = materialize_qwen_image_conditioning(
        qwen_image_conditioning_to_carrier(basic), device="cpu"
    )
    assert torch.equal(basic_round_trip.embeddings, basic.embeddings)
    assert basic_round_trip.pooled is not None
    assert basic.pooled is not None
    assert torch.equal(basic_round_trip.pooled, basic.pooled)
    assert basic_round_trip.attention_mask is None
    assert basic_round_trip.reference_latents == ()

    rich = QwenImageConditioning(
        basic.embeddings,
        None,
        torch.tensor([[1, 1, 0]], dtype=torch.int64),
        (
            torch.ones((1, 16, 1, 2, 3), dtype=torch.bfloat16),
            torch.full((1, 16, 1, 1, 2), 2.0, dtype=torch.bfloat16),
        ),
        torch.tensor([0.25], dtype=torch.float32),
    )
    rich_round_trip = materialize_qwen_image_conditioning(
        qwen_image_conditioning_to_carrier(rich), device="cpu"
    )
    assert torch.equal(rich_round_trip.embeddings, rich.embeddings)
    assert rich_round_trip.attention_mask is not None
    assert rich.attention_mask is not None
    assert torch.equal(rich_round_trip.attention_mask, rich.attention_mask)
    assert len(rich_round_trip.reference_latents) == 2
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            rich_round_trip.reference_latents, rich.reference_latents, strict=True
        )
    )
    assert rich_round_trip.additional_t_cond is not None
    assert rich.additional_t_cond is not None
    assert torch.equal(rich_round_trip.additional_t_cond, rich.additional_t_cond)

    record = basic_carrier.conditioning.records[0]
    channels = dict(record.channels)
    malformed_text = replace(channels[ConditioningChannel.TEXT], shape=(999,))
    malformed_record = replace(
        record,
        channels=((ConditioningChannel.TEXT, malformed_text), *record.channels[1:]),
    )
    malformed = replace(
        basic_carrier,
        conditioning=ConditioningSet((malformed_record,)),
    )
    with pytest.raises(ValueError, match="conditioning-wire:descriptor-mismatch"):
        materialize_qwen_image_conditioning(malformed, device="cpu")


def test_qwen_runtime_materializes_conditioning_on_the_bound_input_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diffusion = torch.nn.Module()
    diffusion.img_in = torch.nn.Linear(2, 2)
    assembled = AssembledQwenImage(
        diffusion=cast(QwenImage, diffusion),
        text=cast(QwenImageTextModel, torch.nn.Identity()),
        vae=cast(WanVAE, torch.nn.Identity()),
        family=QWEN_IMAGE,
    )
    native = QwenImageRuntime(assembled, runtime_identity="test.qwen-image-bound-device")
    carrier = qwen_image_conditioning_to_carrier(QwenImageConditioning(torch.ones((1, 3, 4))))

    def bound_device(module: torch.nn.Module) -> torch.device:
        assert module is diffusion.img_in
        return torch.device("meta")

    monkeypatch.setattr(runtime, "bound_compute_device", bound_device)

    prepared = native.prepare_single_stream_conditioning(carrier)

    assert prepared.embeddings.device.type == "meta"


class _Diffusion(torch.nn.Module):
    def __init__(
        self,
        events: list[str],
        *,
        fail: bool = False,
        config: object = QWEN_IMAGE_CONFIG,
    ) -> None:
        super().__init__()
        self.events = events
        self.fail = fail
        self.config = config
        self.output = torch.nn.Parameter(torch.tensor(0.25))
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
        self.variant_calls: list[tuple[tuple[torch.Tensor, ...], torch.Tensor | None]] = []

    def forward(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        ref_latents: tuple[torch.Tensor, ...] = (),
        additional_t_cond: torch.Tensor | None = None,
        control_residuals: tuple[torch.Tensor | None, ...] | None = None,
        block_patches: tuple[QwenImageDiffSynthExecution, ...] = (),
    ) -> torch.Tensor:
        self.events.append("diffusion")
        self.calls.append((timesteps.detach().clone(), context.detach().clone(), attention_mask))
        self.variant_calls.append((ref_latents, additional_t_cond))
        self.control_residuals = control_residuals
        self.block_patches = block_patches
        if self.fail:
            raise RuntimeError("diffusion failed")
        return self.output.to(latent).expand_as(latent)


class _VAE(torch.nn.Module):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def process_out(self, latent: torch.Tensor) -> torch.Tensor:
        self.events.append("process_out")
        return latent + 0.125

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.events.append("decode")
        return latent[:, :3].repeat_interleave(8, dim=3).repeat_interleave(8, dim=4)


def test_native_runtime_encodes_edit_plus_vision_and_reference_latents() -> None:
    class Text(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.empty(0))
            self.patches: tuple[torch.Tensor, ...] = ()
            self.grids: tuple[torch.Tensor, ...] = ()

        def forward(
            self,
            ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            *,
            image_patches: tuple[torch.Tensor, ...],
            image_grid_thw: tuple[torch.Tensor, ...],
        ) -> tuple[torch.Tensor, None]:
            self.patches = image_patches
            self.grids = image_grid_thw
            return torch.zeros(1, ids.shape[1], 3584), None

    class VAE(torch.nn.Module):
        def encode(self, content: torch.Tensor) -> torch.Tensor:
            return torch.zeros(1, 16, 1, content.shape[3] // 8, content.shape[4] // 8)

        def process_in(self, latent: torch.Tensor) -> torch.Tensor:
            return latent + 1

    text = Text()
    assembled = AssembledQwenImage(
        diffusion=cast(QwenImage, torch.nn.Identity()),
        text=cast(QwenImageTextModel, text),
        vae=cast(WanVAE, VAE()),
        family=QWEN_IMAGE,
    )
    native = QwenImageRuntime(assembled, runtime_identity="test.qwen-image")
    contents = (
        torch.full((1, 3, 48, 64), 0.25),
        torch.full((1, 3, 64, 48), 0.75),
    )
    conditioning = native.encode_edit("combine them", contents)
    assert len(text.patches) == len(text.grids) == 2
    assert [grid.tolist() for grid in text.grids] == [[[1, 24, 32]], [[1, 32, 24]]]
    assert len(conditioning.reference_latents) == 2
    assert [latent.shape for latent in conditioning.reference_latents] == [
        (1, 16, 1, 111, 148),
        (1, 16, 1, 148, 111),
    ]
    assert all(bool(torch.all(latent == 1)) for latent in conditioning.reference_latents)


def test_layered_runtime_samples_and_decodes_the_node_geometry() -> None:
    diffusion = _Diffusion([], config=QWEN_IMAGE_LAYERED_CONFIG)
    assembled = AssembledQwenImage(
        diffusion=cast(QwenImage, diffusion),
        text=cast(QwenImageTextModel, torch.nn.Identity()),
        vae=cast(WanVAE, _VAE([])),
        family=QWEN_IMAGE,
    )
    native = QwenImageRuntime(assembled, runtime_identity="test.qwen-image-layered")
    latent = torch.zeros((2, 16, 4, 2, 3), dtype=torch.float32)
    cond = QwenImageConditioning(
        torch.zeros((2, 3, 3584)),
        reference_latents=(torch.zeros((2, 16, 1, 2, 3)),),
        additional_t_cond=torch.zeros((2,), dtype=torch.long),
    )
    uncond = QwenImageConditioning(
        torch.ones((2, 3, 3584)),
        reference_latents=(torch.ones((2, 16, 1, 2, 3)),),
        additional_t_cond=torch.ones((2,), dtype=torch.long),
    )

    sampled = native.sample(
        latent,
        cond=cond,
        cfg=SamplingGuidance(uncond, 2.0),
        sampler_id="euler",
        scheduler_id="simple",
        steps=1,
    )
    decoded = native.decode_latent(sampled)

    assert sampled.shape == latent.shape
    assert decoded.shape == (2, 3, 4, 16, 24)
    assert len(diffusion.calls) == 2
    variants = {
        (
            float(references[0].mean()),
            int(cast("torch.Tensor", additional).sum()),
        )
        for references, additional in diffusion.variant_calls
    }
    assert variants == {(0.0, 0), (1.0, 2)}

    base = QwenImageRuntime(
        AssembledQwenImage(
            diffusion=cast(QwenImage, _Diffusion([])),
            text=cast(QwenImageTextModel, torch.nn.Identity()),
            vae=cast(WanVAE, _VAE([])),
            family=QWEN_IMAGE,
        ),
        runtime_identity="test.qwen-image-base",
    )
    with pytest.raises(QwenImageRuntimeError, match="Layered profile"):
        base.sample(
            latent,
            cond=cond,
            sampler_id="euler",
            scheduler_id="simple",
            steps=1,
        )


def test_runtime_forwards_pre_offset_brownian_sampling_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assembled = AssembledQwenImage(
        diffusion=cast(QwenImage, _Diffusion([])),
        text=cast(QwenImageTextModel, torch.nn.Identity()),
        vae=cast(WanVAE, _VAE([])),
        family=QWEN_IMAGE,
    )
    native = QwenImageRuntime(assembled, runtime_identity="test.qwen-image-brownian")
    space = FlowSigmas(shift=3.1, multiplier=1.0, timesteps=1000)
    native = native.with_sampling_space(space)
    latent = torch.zeros((1, 16, 1, 2, 3), dtype=torch.float32)
    sentinel = object()

    def capture_noise(
        _sampler: object,
        schedule: SamplingSchedule,
        _latent: torch.Tensor,
        *,
        seed: int,
        device: torch.device | str | None,
    ) -> object:
        assert schedule.initial_sigma == 1.0
        assert schedule.sigmas[0] != 1.0
        assert seed == 7
        return sentinel

    def capture_engine(
        _denoiser: object,
        _solver: object,
        **kwargs: object,
    ) -> torch.Tensor:
        assert kwargs["initial_sigma"] == 1.0
        assert kwargs["noise_sampler"] is sentinel
        sigmas = kwargs["sigmas"]
        assert isinstance(sigmas, tuple)
        assert sigmas[0] != 1.0
        sampling = cast("SamplingDescriptor", kwargs["sampling"])
        assert sampling.sigma_min == space.sigma_min
        assert sampling.sigma_max == space.sigma_max
        return latent

    monkeypatch.setattr(sampling_execution_module, "brownian_step_noise", capture_noise)
    monkeypatch.setattr(sampling_execution_module, "run_denoise", capture_engine)

    output = native.sample(
        latent,
        cond=QwenImageConditioning(torch.zeros((1, 2, 3584))),
        sampler_id="dpmpp_sde",
        scheduler_id="simple",
        steps=2,
        seed=7,
    )

    assert output is latent


def test_variant_denoiser_preserves_integer_timestep_condition() -> None:
    diffusion = _Diffusion([])
    latent = torch.zeros((1, 16, 1, 2, 2))
    additional_t_cond = torch.ones((1,), dtype=torch.long)
    denoiser = runtime._QwenImageDenoiser(  # pyright: ignore[reportPrivateUsage]
        cast(QwenImage, diffusion),
        lambda: False,
        torch.float16,
    )

    denoiser.evaluate_conditioning(
        latent,
        0.5,
        QwenImageConditioning(
            torch.zeros((1, 2, 3584)),
            reference_latents=(latent,),
            additional_t_cond=additional_t_cond,
        ),
    )

    refs, received_additional = diffusion.variant_calls[-1]
    assert refs[0].dtype is torch.float16
    assert received_additional is not None
    assert received_additional.dtype is torch.long


def test_fun_control_uses_upstream_square_root_strength(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with torch.device("meta"):
        control = QwenImageFunControlNet(QWEN_IMAGE_CONFIG)
    digest = qwen_image_control_resource_digest("blake3:" + "0" * 64, "fun", torch.float32)
    control_module._bind_qwen_image_control_resource(  # pyright: ignore[reportPrivateUsage]
        control, digest
    )
    hint = torch.zeros((1, 16, 2, 2))
    hint_digest = qwen_image_control_hint_digest(hint)
    conditioning = QwenImageControlConditioning(
        ControlApplication("qwen-fun", PayloadReference(hint_digest), 2.25, PercentRange(0.0, 1.0)),
        control,
        "fun",
        hint,
        digest,
        hint_digest,
    )

    def control_forward(*_args: object, **_kwargs: object) -> tuple[torch.Tensor, ...]:
        return (torch.ones((1, 1, 1)),)

    monkeypatch.setattr(QwenImageFunControlNet, "forward", control_forward)
    diffusion = _Diffusion([])
    denoiser = runtime._QwenImageDenoiser(  # pyright: ignore[reportPrivateUsage]
        cast(QwenImage, diffusion), lambda: False, torch.float32, control=conditioning
    )
    denoiser.set_control_strength(2.25)
    latent = torch.zeros((1, 16, 1, 2, 2))
    condition = QwenImageConditioning(torch.zeros((1, 2, 3584)))
    assert not denoiser.batchable((condition, condition))
    denoiser.evaluate_conditioning(latent, 0.5, condition)
    assert diffusion.control_residuals is not None
    torch.testing.assert_close(diffusion.control_residuals[0], torch.full((1, 1, 1), 1.5))


def test_qwen_image_conditioning_batch_matches_separate_evaluation() -> None:
    denoiser = runtime._QwenImageDenoiser(  # pyright: ignore[reportPrivateUsage]
        cast(QwenImage, _ArithmeticDiffusion()), lambda: False, torch.float32
    )
    latent = torch.arange(64, dtype=torch.float32).reshape(2, 16, 1, 1, 2)
    conditions = (
        QwenImageConditioning(
            torch.full((1, 3, 3584), 1.0),
            attention_mask=torch.tensor(((1, 1, 0),)),
        ),
        QwenImageConditioning(
            torch.full((1, 3, 3584), -2.0),
            attention_mask=torch.tensor(((1, 0, 0),)),
        ),
    )

    expected = tuple(
        denoiser.evaluate_conditioning(latent, 0.5, condition) for condition in conditions
    )
    actual = denoiser.evaluate_conditioning_batch(latent, 0.5, conditions)

    assert len(actual) == 2
    for fused, separate in zip(actual, expected, strict=True):
        torch.testing.assert_close(fused, separate)


def test_qwen_image_conditioning_batch_rejects_variant_inputs() -> None:
    denoiser = runtime._QwenImageDenoiser(  # pyright: ignore[reportPrivateUsage]
        cast(QwenImage, _ArithmeticDiffusion()), lambda: False, torch.float32
    )
    condition = QwenImageConditioning(torch.zeros((1, 2, 3584)))
    latent = torch.zeros((1, 16, 1, 2, 2))

    assert denoiser.batchable((condition, condition))
    assert not denoiser.batchable((condition, QwenImageConditioning(torch.zeros((1, 3, 3584)))))
    malformed_mask = QwenImageConditioning(
        torch.zeros((1, 2, 3584)), attention_mask=torch.zeros((1, 3))
    )
    assert not denoiser.batchable((malformed_mask, malformed_mask))
    assert not denoiser.batchable(
        (
            condition,
            malformed_mask,
        )
    )
    assert not denoiser.batchable(
        (
            condition,
            QwenImageConditioning(torch.zeros((1, 2, 3584)), reference_latents=(latent,)),
        )
    )
    assert not denoiser.batchable(
        (
            condition,
            QwenImageConditioning(torch.zeros((1, 2, 3584)), additional_t_cond=torch.ones(1)),
        )
    )


def test_diffsynth_runtime_prepares_wan_latent_and_inverted_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class VAE(torch.nn.Module):
        config = SimpleNamespace(spatial_ratio=8)

        def __init__(self) -> None:
            super().__init__()
            self.encoded_content: torch.Tensor | None = None

        def encode(self, content: torch.Tensor) -> torch.Tensor:
            self.encoded_content = content
            return torch.full((1, 16, 1, content.shape[-2] // 8, content.shape[-1] // 8), 2.0)

        def process_in(self, latent: torch.Tensor) -> torch.Tensor:
            return latent + 3.0

    with torch.device("meta"):
        patch = QwenImageDiffSynthPatch(input_features=68)
    digest = qwen_image_diffsynth_resource_digest(
        "blake3:" + "3" * 64, "diffsynth_inpaint", torch.float32
    )
    control_module._bind_qwen_image_diffsynth_resource(  # pyright: ignore[reportPrivateUsage]
        patch, digest
    )
    prepared: list[torch.Tensor] = []

    def prepare(_self: QwenImageDiffSynthPatch, value: torch.Tensor) -> torch.Tensor:
        prepared.append(value)
        tokens = ((value.shape[-2] + 1) // 2) * ((value.shape[-1] + 1) // 2)
        return torch.zeros((value.shape[0], tokens, 3072))

    monkeypatch.setattr(QwenImageDiffSynthPatch, "prepare_condition", prepare)
    content = torch.full((1, 3, 8, 8), 0.75)
    mask = torch.tensor([[[0.0, 0.5], [1.0, 0.25]]])
    conditioning = QwenImageDiffSynthConditioning(
        patch,
        "diffsynth_inpaint",
        content,
        mask,
        -1.25,
        digest,
        qwen_image_control_hint_digest(content),
        qwen_image_control_hint_digest(mask),
    )
    vae = VAE()
    assembled = AssembledQwenImage(
        diffusion=cast(QwenImage, torch.nn.Identity()),
        text=cast(QwenImageTextModel, torch.nn.Identity()),
        vae=cast(WanVAE, vae),
        family=QWEN_IMAGE,
    )
    native = QwenImageRuntime(assembled, runtime_identity="test.qwen-image")
    execution = native._prepare_diffsynth(  # pyright: ignore[reportPrivateUsage]
        torch.zeros((1, 16, 1, 2, 3)), conditioning, torch.float32
    )
    assert vae.encoded_content is not None
    assert vae.encoded_content.shape == (1, 3, 1, 16, 24)
    assert bool(torch.all(vae.encoded_content == 0.5))
    assert execution.strength == -1.25
    assert prepared[0].shape == (1, 17, 1, 2, 3)
    assert bool(torch.all(prepared[0][:, :16] == 5.0))
    expected_mask = 1.0 - torch.nn.functional.interpolate(
        mask.unsqueeze(1), size=(2, 3), mode="bilinear", align_corners=False
    )
    torch.testing.assert_close(prepared[0][:, 16, 0], expected_mask[:, 0])


def test_diffsynth_denoiser_threads_ordered_block_patches() -> None:
    with torch.device("meta"):
        patch = QwenImageDiffSynthPatch()
    digest = qwen_image_diffsynth_resource_digest("blake3:" + "4" * 64, "diffsynth", torch.float32)
    control_module._bind_qwen_image_diffsynth_resource(  # pyright: ignore[reportPrivateUsage]
        patch, digest
    )
    executions = (
        QwenImageDiffSynthExecution(patch, torch.zeros((1, 1, 3072)), 0.5, digest),
        QwenImageDiffSynthExecution(patch, torch.zeros((1, 1, 3072)), -0.25, digest),
    )
    diffusion = _Diffusion([])
    denoiser = runtime._QwenImageDenoiser(  # pyright: ignore[reportPrivateUsage]
        cast(QwenImage, diffusion), lambda: False, torch.float32, diffsynth=executions
    )
    latent = torch.zeros((1, 16, 1, 2, 2))
    denoiser.evaluate_conditioning(latent, 0.5, QwenImageConditioning(torch.zeros((1, 2, 3584))))
    assert diffusion.block_patches == executions


def test_runtime_admits_prepared_diffsynth_execution_without_vae(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with torch.device("meta"):
        patch = QwenImageDiffSynthPatch()
    digest = qwen_image_diffsynth_resource_digest("blake3:" + "5" * 64, "diffsynth", torch.float32)
    control_module._bind_qwen_image_diffsynth_resource(  # pyright: ignore[reportPrivateUsage]
        patch, digest
    )
    condition = torch.zeros((1, 1, 3072))
    execution = QwenImageDiffSynthExecution(patch, condition, 0.75, digest)
    diffusion = _Diffusion([])
    native = runtime.QwenImageDiffusionRuntime(
        cast(QwenImage, diffusion),
        QWEN_IMAGE,
        runtime_identity="test.qwen-image-prepared-diffsynth",
    )

    def apply(
        _self: QwenImageDiffSynthExecution,
        image: torch.Tensor,
        _block_index: int,
    ) -> torch.Tensor:
        return image

    monkeypatch.setattr(QwenImageDiffSynthExecution, "__call__", apply)
    native.sample(
        torch.zeros((1, 16, 1, 2, 2)),
        cond=QwenImageConditioning(torch.zeros((1, 2, 3584))),
        sampler_id="euler",
        scheduler_id="simple",
        steps=1,
        diffsynth=(execution,),
    )

    assert diffusion.block_patches is not None
    admitted = diffusion.block_patches[0]
    assert admitted.model is patch
    assert admitted.condition is not condition
    torch.testing.assert_close(admitted.condition, condition)
    assert admitted.strength == 0.75


def test_qwen_image_diffusion_runtime_assembly_enrolls_for_residency() -> None:
    from dinkster_inference_torch import CastOperations, enroll_assembled

    class _Routable(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = CastOperations(torch.float32).linear(2, 2)

    native = runtime.QwenImageDiffusionRuntime(
        cast(QwenImage, _Routable()),
        QWEN_IMAGE,
        runtime_identity="test.qwen-image-residency",
    )
    enrolled = enroll_assembled(
        cast("Any", native.assembled),
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )
    assert tuple(enrolled) == ("diffusion",)
    assert not enrolled.storage_dtype_report.enabled
    from dinkster_inference.runtime import FamilyRuntime

    assert isinstance(native, FamilyRuntime)
    with pytest.raises(QwenImageRuntimeError, match="no text encoder"):
        native.encode_text("prompt")
    with pytest.raises(QwenImageRuntimeError, match="no VAE codec"):
        native.decode_latent(torch.zeros((1, 16, 1, 2, 2)))
    with pytest.raises(QwenImageRuntimeError, match="no VAE codec"):
        native.encode_content(torch.zeros((1, 3, 8, 8)))


class _ArithmeticDiffusion(torch.nn.Module):
    """Input-dependent fake: the output couples latent, timesteps, context,
    and control residuals so schedule, noise, CFG-lane, and control-window
    parity are all visible."""

    def __init__(self) -> None:
        super().__init__()
        self.img_in = InitlessOperations().linear(1, 1, bias=False)
        torch.nn.init.zeros_(self.img_in.weight)
        self.config = QWEN_IMAGE_CONFIG

    def forward(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        ref_latents: tuple[torch.Tensor, ...] = (),
        additional_t_cond: torch.Tensor | None = None,
        control_residuals: tuple[torch.Tensor | None, ...] | None = None,
        block_patches: tuple[QwenImageDiffSynthExecution, ...] = (),
    ) -> torch.Tensor:
        del attention_mask, ref_latents, additional_t_cond, block_patches
        scale = context.to(latent.dtype).mean(dim=(1, 2)).reshape(-1, 1, 1, 1, 1)
        value = latent * 0.5 + timesteps.to(latent.dtype).reshape(-1, 1, 1, 1, 1) * 0.25 + scale
        if control_residuals is not None:
            for residual in control_residuals:
                if residual is not None:
                    value = value + residual.to(latent.dtype).mean()
        return value


def _custom_runtime(diffusion: torch.nn.Module) -> QwenImageRuntime:
    assembled = AssembledQwenImage(
        diffusion=cast(QwenImage, diffusion),
        text=cast(QwenImageTextModel, torch.nn.Identity()),
        vae=cast(WanVAE, _VAE([])),
        family=QWEN_IMAGE,
    )
    return QwenImageRuntime(assembled, runtime_identity="test.qwen-image-custom")


def _filled_qwen_conditioning(fill: float) -> QwenImageConditioning:
    return QwenImageConditioning(torch.full((1, 3, 3584), fill))


def _qwen_custom_request(
    sampler_id: str = "dinkster.euler",
    sigmas: tuple[float, ...] = (1.0, 0.5, 0.0),
) -> CustomSamplingRequest[torch.Tensor]:
    descriptor = torch_sampler_registry().get(sampler_id)
    assert descriptor is not None
    return CustomSamplingRequest(descriptor, (), sigmas)


def test_qwen_image_runtimes_satisfy_the_custom_sampling_protocol() -> None:
    assert isinstance(_custom_runtime(_Diffusion([])), CustomSamplingRuntime)
    diffusion_only = runtime.QwenImageDiffusionRuntime(
        cast(QwenImage, _Diffusion([])),
        QWEN_IMAGE,
        runtime_identity="test.qwen-image-custom-protocol",
    )
    assert isinstance(diffusion_only, CustomSamplingRuntime)


@pytest.mark.parametrize("component", (False, True), ids=("complete", "component"))
@pytest.mark.parametrize("sampler", ("euler", "dpmpp_2m_sde"))
@pytest.mark.parametrize(
    "device",
    (
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
        ),
    ),
)
def test_native_sampling_nodes_use_runtime_device(
    component: bool, sampler: str, device: str
) -> None:
    from dinkster_compat_comfy import native_arm as arm
    from dinkster_compat_comfy.native_residency import (
        NativeResidencyCoordinator,
        NativeRuntimeHandle,
    )
    from dinkster_inference_torch.memory import DeviceMemory, MemoryPolicy
    from dinkster_inference_torch.residency import ResidencyManager

    recipe = ReconstructionRecipe(
        sources=(
            WeightSourceBinding(
                "diffusion" if component else "checkpoint",
                WeightSourceRef("blake3:" + "1" * 64, "qwen.safetensors", 1),
            ),
        ),
        family_id=QWEN_IMAGE.id,
        component_identity=("family=dinkster.qwen_image",),
        knobs=RuntimeKnobs(
            diffusion_dtype="float32", text_dtype="unloaded", vae_dtype="unloaded", fp8_matmul=False
        ),
    )
    diffusion = cast(QwenImage, _ArithmeticDiffusion())
    native = (
        runtime.QwenImageDiffusionRuntime(
            diffusion, QWEN_IMAGE, runtime_identity=recipe.runtime_identity
        )
        if component
        else QwenImageRuntime(
            AssembledQwenImage(
                diffusion=diffusion,
                text=cast(QwenImageTextModel, torch.nn.Identity()),
                vae=cast(WanVAE, torch.nn.Identity()),
                family=QWEN_IMAGE,
            ),
            runtime_identity=recipe.runtime_identity,
        )
    )
    manager = ResidencyManager(
        policy=MemoryPolicy(inference_reserve=0, physical_headroom=0),
        free_memory=lambda _device: DeviceMemory(free_total=1 << 40, free_torch=0),
    )
    handle = NativeRuntimeHandle(
        native, device, recipe=recipe, coordinator=NativeResidencyCoordinator(cast(Any, manager))
    )
    forward_devices: list[str] = []

    def record_device(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        forward_devices.append(args[0].device.type)
        assert all(value.device.type == device for value in args if isinstance(value, torch.Tensor))

    hook = diffusion.register_forward_pre_hook(record_device)
    positive = replace(_filled_qwen_conditioning(2.0), attention_mask=torch.ones((1, 3)))
    negative = replace(_filled_qwen_conditioning(0.5), attention_mask=torch.ones((1, 3)))
    rows: object = qwen_image_conditioning_to_carrier(positive)
    negative_rows: object = qwen_image_conditioning_to_carrier(negative)
    if component:
        text_recipe = replace(
            recipe,
            sources=(
                WeightSourceBinding(
                    "qwen2_5_vl_7b", WeightSourceRef("blake3:" + "2" * 64, "text.safetensors", 1)
                ),
            ),
            knobs=replace(recipe.knobs, diffusion_dtype="unloaded", text_dtype="float32"),
        )
        binding = ComponentBinding("qwen2_5_vl_7b", QWEN_IMAGE.id, text_recipe.runtime_identity)
        rows = bind_component_conditioning(qwen_image_conditioning_to_carrier(positive), binding)
        negative_rows = bind_component_conditioning(
            qwen_image_conditioning_to_carrier(negative), binding
        )
    try:
        sigmas = arm.GenerationBasicScheduler.execute(
            model=handle, scheduler="simple", steps=3, denoise=1.0
        )["sigmas"]
        latent = {"samples": torch.zeros((1, 16, 2, 2))}
        common: dict[str, Any] = dict(
            model=handle, cfg=2.5, positive=rows, negative=negative_rows, latent_image=latent
        )
        selected = arm.GenerationKSamplerSelect.execute(sampler_name=sampler)["sampler"]
        custom_latent = arm.GenerationSamplerCustom.execute(
            **common, add_noise=True, noise_seed=7, sampler=selected, sigmas=sigmas
        )["output"]
        assert isinstance(custom_latent, dict)
        custom = custom_latent["samples"]
        assert isinstance(custom, torch.Tensor) and custom.device.type == "cpu"
        if not component:
            key = arm._NATIVE_PREPARED_CONDITIONING_KEY  # pyright: ignore[reportPrivateUsage]
            common.update(
                positive=[[positive.embeddings, {key: positive}]],
                negative=[[negative.embeddings, {key: negative}]],
            )
        composed_latent = arm.GenerationKSampler.execute(
            **common, seed=7, steps=3, sampler_name=sampler, scheduler="simple", denoise=1.0
        )["latent"]
        assert isinstance(composed_latent, dict)
        composed = composed_latent["samples"]
        assert isinstance(composed, torch.Tensor)
        assert torch.equal(composed.cpu(), custom)
        assert torch.isfinite(custom).all()
        assert forward_devices and set(forward_devices) == {device}
        assert latent["samples"].device.type == "cpu"
        assert not torch.count_nonzero(latent["samples"])
        assert positive.embeddings.device.type == negative.embeddings.device.type == "cpu"
    finally:
        hook.remove()
        handle.terminal_release()


@pytest.mark.parametrize(
    "space",
    (
        object(),
        FlowSigmas(shift=3.1),
        FlowSigmas(shift=0.0, multiplier=1.0),
        FlowSigmas(shift=-1.0, multiplier=1.0),
        FlowSigmas(shift=float("nan"), multiplier=1.0),
        FluxFlowSigmas(shift=float("inf")),
        FluxFlowSigmas(shift=1000.0),
        FluxFlowSigmas(shift=-1000.0),
    ),
)
def test_qwen_rejects_unsupported_sampling_spaces_without_mutation(space: Any) -> None:
    native = _custom_runtime(_ArithmeticDiffusion())
    original = native.sampling_sigma_space()
    with pytest.raises(QwenImageRuntimeError, match="sampling override"):
        native.with_sampling_space(space)
    assert native.sampling_sigma_space() == original


@pytest.mark.parametrize("component", (False, True), ids=("complete", "component"))
@pytest.mark.parametrize("scheduler", ("simple", "normal", "beta", "ddim_uniform", "karras"))
@pytest.mark.parametrize("sampler", ("euler", "dpmpp_2m_sde"))
@pytest.mark.parametrize("use_override", (False, True), ids=("default", "auraflow"))
@pytest.mark.parametrize(
    "device",
    (
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
        ),
    ),
)
def test_qwen_sampling_space_reaches_both_native_sampling_nodes(
    component: bool,
    scheduler: str,
    sampler: str,
    use_override: bool,
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_compat_comfy import native_arm as arm
    from dinkster_compat_comfy.native_residency import (
        NativeResidencyCoordinator,
        NativeRuntimeHandle,
    )
    from dinkster_inference_torch.memory import DeviceMemory, MemoryPolicy
    from dinkster_inference_torch.residency import ResidencyManager

    recipe = ReconstructionRecipe(
        sources=(
            WeightSourceBinding(
                "diffusion" if component else "checkpoint",
                WeightSourceRef("blake3:" + "1" * 64, "qwen.safetensors", 1),
            ),
        ),
        family_id=QWEN_IMAGE.id,
        component_identity=("family=dinkster.qwen_image",),
        knobs=RuntimeKnobs(
            diffusion_dtype="float32", text_dtype="unloaded", vae_dtype="unloaded", fp8_matmul=False
        ),
    )
    diffusion = cast(QwenImage, _ArithmeticDiffusion())
    native = (
        runtime.QwenImageDiffusionRuntime(
            diffusion, QWEN_IMAGE, runtime_identity=recipe.runtime_identity
        )
        if component
        else QwenImageRuntime(
            AssembledQwenImage(
                diffusion=diffusion,
                text=cast(QwenImageTextModel, torch.nn.Identity()),
                vae=cast(WanVAE, torch.nn.Identity()),
                family=QWEN_IMAGE,
            ),
            runtime_identity=recipe.runtime_identity,
        )
    )
    assert isinstance(native, SamplingSpaceOverrideRuntime)
    original_space = native.sampling_sigma_space()
    space = FlowSigmas(shift=3.1, multiplier=1.0, timesteps=1000)
    derived = native.with_sampling_space(space)
    assert derived is not native and type(derived) is type(native)
    assert derived.assembled is native.assembled
    assert derived.runtime_identity == native.runtime_identity
    assert derived.conditioning_identity == native.conditioning_identity
    assert derived.sampling_sigma_space() is space
    manager = ResidencyManager(
        policy=MemoryPolicy(inference_reserve=0, physical_headroom=0),
        free_memory=lambda _device: DeviceMemory(free_total=1 << 40, free_torch=0),
    )
    handle = NativeRuntimeHandle(
        native, device, recipe=recipe, coordinator=NativeResidencyCoordinator(cast(Any, manager))
    )
    forward_devices: list[str] = []

    def record_device(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        forward_devices.append(args[0].device.type)

    hook = diffusion.register_forward_pre_hook(record_device)
    if use_override:
        model = arm.GenerationModelSamplingAuraFlow.execute(model=handle, shift=3.1)["model"]
        assert isinstance(model, arm._NativeModelOverlay)  # pyright: ignore[reportPrivateUsage]
        assert model.sampling_space == space and model.sampling_shift is None
        expected_runtime = derived
    else:
        model = handle
        expected_runtime = native
    positive = _filled_qwen_conditioning(2.0)
    negative = _filled_qwen_conditioning(0.5)
    key = arm._NATIVE_PREPARED_CONDITIONING_KEY  # pyright: ignore[reportPrivateUsage]
    rows: object = [[positive.embeddings, {key: positive}]]
    negative_rows: object = [[negative.embeddings, {key: negative}]]
    if component:
        text_recipe = replace(
            recipe,
            sources=(
                WeightSourceBinding(
                    "qwen2_5_vl_7b", WeightSourceRef("blake3:" + "2" * 64, "text.safetensors", 1)
                ),
            ),
            knobs=replace(recipe.knobs, diffusion_dtype="unloaded", text_dtype="float32"),
        )
        binding = ComponentBinding("qwen2_5_vl_7b", QWEN_IMAGE.id, text_recipe.runtime_identity)
        rows = bind_component_conditioning(qwen_image_conditioning_to_carrier(positive), binding)
        negative_rows = bind_component_conditioning(
            qwen_image_conditioning_to_carrier(negative), binding
        )
    executed_spaces: list[object] = []
    build_schedule = sampling_execution_module.build_custom_sampling_schedule

    def capture(*args: Any, **kwargs: Any) -> Any:
        executed_spaces.append(args[1])
        return build_schedule(*args, **kwargs)

    monkeypatch.setattr(sampling_execution_module, "build_custom_sampling_schedule", capture)
    try:
        sigmas = arm.GenerationBasicScheduler.execute(
            model=model, scheduler=scheduler, steps=3, denoise=1.0
        )["sigmas"]
        assert isinstance(sigmas, SigmaSchedule)
        assert sigmas.values == expected_runtime.custom_sampling_sigmas(
            f"dinkster.{scheduler}", 3, 1.0
        )
        if use_override:
            assert sigmas.values != native.custom_sampling_sigmas(f"dinkster.{scheduler}", 3, 1.0)
        latent = {"samples": torch.zeros((1, 16, 2, 2))}
        common: dict[str, Any] = dict(
            model=model, cfg=2.5, positive=rows, negative=negative_rows, latent_image=latent
        )
        selected = arm.GenerationKSamplerSelect.execute(sampler_name=sampler)["sampler"]
        composed_latent = arm.GenerationKSampler.execute(
            **common, seed=7, steps=3, sampler_name=sampler, scheduler=scheduler, denoise=1.0
        )["latent"]
        assert isinstance(composed_latent, dict)
        composed = composed_latent["samples"]
        assert isinstance(composed, torch.Tensor)
        if not component:
            common.update(
                positive=qwen_image_conditioning_to_carrier(positive),
                negative=qwen_image_conditioning_to_carrier(negative),
            )
        custom_latent = arm.GenerationSamplerCustom.execute(
            **common, add_noise=True, noise_seed=7, sampler=selected, sigmas=sigmas
        )["output"]
        assert isinstance(custom_latent, dict)
        custom = custom_latent["samples"]
        assert isinstance(custom, torch.Tensor)
        assert custom.device.type == "cpu"
        assert torch.equal(composed.cpu(), custom)
        assert latent["samples"].device.type == "cpu"
        assert not torch.count_nonzero(latent["samples"])
        assert torch.isfinite(custom).all()
        assert executed_spaces == [space if use_override else original_space] * 2
        assert forward_devices and set(forward_devices) == {device}
        assert native.sampling_sigma_space() == original_space
    finally:
        hook.remove()
        handle.terminal_release()


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
    native = _custom_runtime(_ArithmeticDiffusion())
    positive = _filled_qwen_conditioning(2.0)
    negative = None if cfg_scale is None else _filled_qwen_conditioning(5.0)
    generator = torch.Generator().manual_seed(11)
    latent = torch.rand((1, 16, 1, 2, 2), generator=generator)

    expected = native.sample(
        latent,
        cond=positive,
        cfg=SamplingGuidance(negative, 1.0 if cfg_scale is None else cfg_scale),
        sampler_id=sampler_id,
        scheduler_id=scheduler_id,
        steps=steps,
        denoise=1.0,
        seed=185,
        segment=segment,
    )
    result = run_ksampler_as_custom(
        native,
        latent,
        samplers=cast("Any", native)._samplers,
        schedulers=cast("Any", native)._schedulers,
        space=FluxFlowSigmas(shift=1.15, timesteps=10000),
        flow=True,
        sampler_id=sampler_id,
        scheduler_id=scheduler_id,
        steps=steps,
        denoise=1.0,
        seed=185,
        cond=positive,
        cfg=SamplingGuidance(negative, 1.0 if cfg_scale is None else cfg_scale),
        segment=segment,
        error=QwenImageRuntimeError,
    )

    output = result.output
    assert type(output) is torch.Tensor
    assert torch.equal(output, expected)


def test_control_windowed_ksampler_is_bit_equal_sugar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with torch.device("meta"):
        control = QwenImageFunControlNet(QWEN_IMAGE_CONFIG)
    digest = qwen_image_control_resource_digest("blake3:" + "1" * 64, "fun", torch.float32)
    control_module._bind_qwen_image_control_resource(  # pyright: ignore[reportPrivateUsage]
        control, digest
    )
    hint = torch.zeros((1, 16, 2, 2))
    hint_digest = qwen_image_control_hint_digest(hint)
    conditioning = QwenImageControlConditioning(
        ControlApplication(
            "qwen-fun", PayloadReference(hint_digest), 2.25, PercentRange(0.25, 0.75)
        ),
        control,
        "fun",
        hint,
        digest,
        hint_digest,
    )

    def control_forward(*_args: object, **_kwargs: object) -> tuple[torch.Tensor, ...]:
        return (torch.ones((1, 1, 1)),)

    monkeypatch.setattr(QwenImageFunControlNet, "forward", control_forward)
    native = _custom_runtime(_ArithmeticDiffusion())
    generator = torch.Generator().manual_seed(3)
    latent = torch.rand((1, 16, 1, 2, 2), generator=generator)
    positive = _filled_qwen_conditioning(2.0)

    baseline = native.sample(
        latent,
        cond=positive,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=4,
        denoise=1.0,
        seed=21,
    )
    expected = native.sample(
        latent,
        cond=positive,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=4,
        denoise=1.0,
        seed=21,
        control=conditioning,
    )
    assert not torch.equal(expected, baseline)

    sampler = torch_sampler_registry().get("dinkster.euler")
    scheduler = torch_scheduler_registry().get("dinkster.simple")
    assert sampler is not None
    assert scheduler is not None
    schedule = build_sampling_schedule(
        scheduler,
        FluxFlowSigmas(shift=1.15, timesteps=10000),
        sampler,
        4,
        denoise=1.0,
        flow=True,
    )
    result = native.sample_custom(
        latent,
        noise=prepare_noise(latent, 21),
        cond=positive,
        request=CustomSamplingRequest(sampler, (), schedule.pre_offset),
        seed=21,
        control=conditioning,
    )
    assert torch.equal(result.output, expected)


def test_custom_sampling_sigma_surfaces_match_family_flow_space() -> None:
    native = _custom_runtime(_Diffusion([]))
    space = FluxFlowSigmas(shift=1.15, timesteps=10000)
    scheduler = torch_scheduler_registry().get("dinkster.simple")
    assert scheduler is not None
    assert native.custom_sampling_sigmas("dinkster.simple", 4, 0.5) == sampling_sigmas(
        scheduler, space, 4, denoise=0.5
    )
    with pytest.raises(QwenImageRuntimeError, match="unknown scheduler"):
        native.custom_sampling_sigmas("test.missing", 4, 1.0)
    assert native.custom_sampling_beta_sigmas(4, 0.6, 0.6) == custom_beta_sigmas(space, 4, 0.6, 0.6)
    with pytest.raises(ValueError, match="discrete sigma space"):
        native.custom_sampling_sd_turbo_sigmas(2, 1.0)
    assert native.custom_sampling_percent_to_sigma(
        0.5, return_actual_sigma=False
    ) == custom_percent_to_sigma(space, space.percent_to_sigma, 0.5, return_actual_sigma=False)
    assert native.custom_sampling_percent_to_sigma(
        0.3, return_actual_sigma=True
    ) == custom_percent_to_sigma(space, space.percent_to_sigma, 0.3, return_actual_sigma=True)


def test_sample_custom_refuses_multistream_shapes_and_unsupported_modes() -> None:
    diffusion = _Diffusion([])
    native = _custom_runtime(diffusion)
    condition = QwenImageConditioning(torch.zeros((1, 2, 3584)))
    latent = torch.zeros((1, 16, 1, 2, 2))
    noise = torch.zeros_like(latent)
    request = _qwen_custom_request()

    def sample(**overrides: Any) -> Any:
        arguments: dict[str, Any] = {"noise": noise, "cond": condition, "request": request}
        arguments.update(overrides)
        return native.sample_custom(arguments.pop("latent", latent), **arguments)

    with pytest.raises(QwenImageRuntimeError, match="prepared multi-stream payload"):
        sample(cond=PreparedMultiStreamConditioning("native:test", object()))
    with pytest.raises(QwenImageRuntimeError, match="perp-neg"):
        sample(cfg=PerpNegSamplingGuidance(condition, condition, 3.0, 1.0))
    with pytest.raises(QwenImageRuntimeError, match="must have shape"):
        sample(latent=torch.zeros((1, 16, 2, 2)), noise=torch.zeros((1, 16, 2, 2)))
    with pytest.raises(QwenImageRuntimeError, match="must have shape"):
        sample(latent=torch.zeros((1, 4, 1, 2, 2)), noise=torch.zeros((1, 4, 1, 2, 2)))
    with pytest.raises(QwenImageRuntimeError, match="must have shape"):
        sample(latent=torch.zeros((1, 16, 2, 2, 2)), noise=torch.zeros((1, 16, 2, 2, 2)))
    with pytest.raises(QwenImageRuntimeError, match="distilled-guidance"):
        sample(guidance=4.0)
    with pytest.raises(QwenImageRuntimeError, match="inpaint"):
        sample(inpaint=cast("Any", object()))
    euler = torch_sampler_registry().get("dinkster.euler")
    assert euler is not None
    unknown = replace(euler, id="test.missing", aliases=())
    with pytest.raises(QwenImageRuntimeError, match="unknown sampler"):
        sample(request=CustomSamplingRequest(unknown, (), (1.0, 0.0)))
    with pytest.raises(QwenImageRuntimeError, match="exact QwenImageConditioning"):
        sample(cond=Conditioning(torch.zeros((1, 2, 3584)), None))
    with pytest.raises(QwenImageRuntimeError, match="adapter options: bogus_option"):
        sample(bogus_option=True)
    with pytest.raises(QwenImageRuntimeError, match="adapter options: bogus_option"):
        native.sample(
            latent,
            cond=condition,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
            bogus_option=True,
        )
    assert not diffusion.calls


def test_sample_custom_captures_denoised_output() -> None:
    native = _custom_runtime(_Diffusion([]))
    latent = torch.zeros((1, 16, 1, 2, 2))
    result = native.sample_custom(
        latent,
        noise=torch.zeros_like(latent),
        cond=QwenImageConditioning(torch.zeros((1, 2, 3584))),
        request=_qwen_custom_request(),
        seed=9,
    )
    assert result.output.shape == latent.shape
    assert result.denoised_output is not None
    assert result.denoised_output.shape == latent.shape
