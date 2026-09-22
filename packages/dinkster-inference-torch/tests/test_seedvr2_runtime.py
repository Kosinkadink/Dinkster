from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    SEEDVR2_SIGMAS,
    Conditioning,
    CustomSamplingRequest,
    CustomSamplingRuntime,
    SamplingGuidance,
    sampling_sigmas,
)
from dinkster_inference_torch import (
    SeedVR2CodecRuntime,
    SeedVR2Conditioning,
    SeedVR2Denoiser,
    SeedVR2DiffusionRuntime,
    SeedVR2RuntimeError,
    seedvr2_conditioning,
)
from dinkster_inference_torch.sampling_execution import run_ksampler_as_custom
from dinkster_inference_torch.schedules import torch_scheduler_registry
from dinkster_inference_torch.solvers import torch_sampler_registry

IDENTITY = "native:dinkster.seedvr2:" + "1" * 64


class RecordingSeedVR2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.contexts = {
            "positive": torch.full((58, 5120), 2.0),
            "negative": torch.full((64, 5120), -3.0),
        }
        self.calls: list[dict[str, object]] = []

    @contextmanager
    def materialized_text_conditioning(
        self, branch: str, *, device: torch.device, dtype: torch.dtype
    ) -> Generator[torch.Tensor, None, None]:
        yield self.contexts[branch].to(device=device, dtype=dtype)

    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "latent": latent,
                "timestep": timestep,
                "context": context,
                **kwargs,
            }
        )
        condition = cast("torch.Tensor", kwargs["condition"])
        context_mean = context.float().mean(dim=(1, 2)).reshape(-1, 1, 1, 1, 1)
        return latent * 0.25 + condition[:, :16] * 0.125 + context_mean


def _runtime(model: RecordingSeedVR2) -> SeedVR2DiffusionRuntime:
    return SeedVR2DiffusionRuntime(
        cast("Any", model),
        runtime_identity=IDENTITY,
        compute_dtype=torch.float32,
        sampler_registry=torch_sampler_registry(),
        scheduler_registry=torch_scheduler_registry(),
    )


def _request(
    sigmas: tuple[float, ...] = (1.0, 0.5, 0.0),
) -> CustomSamplingRequest[torch.Tensor]:
    sampler = torch_sampler_registry().get("dinkster.euler")
    assert sampler is not None
    return CustomSamplingRequest(sampler, (), sigmas)


def _conditioning(value: float = 0.5) -> tuple[SeedVR2Conditioning, SeedVR2Conditioning]:
    return seedvr2_conditioning(torch.full((1, 16, 2, 3, 4), value), component_identity=IDENTITY)


def test_sampling_memory_matches_comfyui_accelerated_attention_estimate() -> None:
    runtime = SeedVR2DiffusionRuntime(
        cast("Any", RecordingSeedVR2()),
        runtime_identity=IDENTITY,
        compute_dtype=torch.bfloat16,
    )
    assert runtime.sampling_memory_requirements((1, 16, 2, 32, 48)) == (
        257_698_036,
        128_849_018,
    )


def test_seedvr2_conditioning_appends_exact_mask_and_branches() -> None:
    latent = torch.arange(16 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 16, 2, 3, 4)
    positive, negative = seedvr2_conditioning(latent, component_identity=IDENTITY)
    assert positive.branch == "positive"
    assert negative.branch == "negative"
    assert positive.component_identity == negative.component_identity == IDENTITY
    assert positive.embeddings is negative.embeddings
    assert torch.equal(positive.embeddings[:, :16], latent)
    assert torch.equal(positive.embeddings[:, 16:], torch.ones((1, 1, 2, 3, 4)))


def test_seedvr2_denoiser_uses_branch_context_condition_latent_and_flow_math() -> None:
    model = RecordingSeedVR2()
    evaluator = SeedVR2Denoiser(
        cast("Any", model), runtime_identity=IDENTITY, compute_dtype=torch.float32
    )
    positive, negative = _conditioning()
    latent = torch.full((1, 16, 2, 3, 4), 4.0)

    positive_output = evaluator.evaluate_conditioning(
        latent, 0.25, evaluator.prepare_conditioning(positive)
    )
    negative_output = evaluator.evaluate_conditioning(
        latent, 0.25, evaluator.prepare_conditioning(negative)
    )

    expected_positive_model = 4.0 * 0.25 + 0.5 * 0.125 + 2.0
    expected_negative_model = 4.0 * 0.25 + 0.5 * 0.125 - 3.0
    torch.testing.assert_close(
        positive_output, torch.full_like(latent, 4.0 - 0.25 * expected_positive_model)
    )
    torch.testing.assert_close(
        negative_output, torch.full_like(latent, 4.0 - 0.25 * expected_negative_model)
    )
    assert cast("torch.Tensor", model.calls[0]["context"]).shape == (1, 58, 5120)
    assert cast("torch.Tensor", model.calls[1]["context"]).shape == (1, 64, 5120)
    assert torch.equal(cast("torch.Tensor", model.calls[0]["timestep"]), torch.tensor([250.0]))
    assert torch.equal(cast("torch.Tensor", model.calls[1]["timestep"]), torch.tensor([250.0]))
    assert cast("torch.Tensor", model.calls[0]["condition"]).shape == (1, 17, 2, 3, 4)
    assert model.calls[0]["transformer_options"] == {"cond_or_uncond": ["positive"]}


def test_seedvr2_denoiser_refuses_wrong_identity_and_incompatible_batch() -> None:
    evaluator = SeedVR2Denoiser(
        cast("Any", RecordingSeedVR2()), runtime_identity=IDENTITY, compute_dtype=torch.float32
    )
    wrong = SeedVR2Conditioning(torch.zeros((1, 17, 1, 2, 2)), component_identity="native:other")
    with pytest.raises(SeedVR2RuntimeError, match="different diffusion"):
        evaluator.prepare_conditioning(wrong)
    positive, negative = _conditioning()
    prepared = tuple(map(evaluator.prepare_conditioning, (positive, negative)))
    assert not evaluator.batchable(prepared)
    with pytest.raises(SeedVR2RuntimeError, match="empty or incompatible"):
        evaluator.evaluate_conditioning_batch(torch.zeros((1, 16, 2, 3, 4)), 0.5, prepared)


def test_seedvr2_runtime_is_custom_sampling_and_uses_discrete_flow_space() -> None:
    runtime = _runtime(RecordingSeedVR2())
    assert isinstance(runtime, CustomSamplingRuntime)
    scheduler = torch_scheduler_registry().get("dinkster.simple")
    assert scheduler is not None
    assert runtime.custom_sampling_sigmas("dinkster.simple", 4, 0.75) == sampling_sigmas(
        scheduler, SEEDVR2_SIGMAS, 4, denoise=0.75
    )
    assert SEEDVR2_SIGMAS.shift == 1.0
    assert SEEDVR2_SIGMAS.sigma_min == 0.001


def test_ksampler_is_sugar_over_seedvr2_custom_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(RecordingSeedVR2())
    positive, negative = _conditioning()
    latent = torch.rand((1, 16, 2, 3, 4), generator=torch.Generator().manual_seed(9))
    guidance: SamplingGuidance[Conditioning[torch.Tensor]] = SamplingGuidance(negative, 2.0)
    captured: list[bool] = []
    sample_custom = runtime.sample_custom

    def record_capture(*args: object, **kwargs: object) -> object:
        captured.append(cast("bool", kwargs.get("capture_denoised", True)))
        return cast("Any", sample_custom)(*args, **kwargs)

    monkeypatch.setattr(runtime, "sample_custom", record_capture)
    expected = runtime.sample(
        latent,
        cond=positive,
        cfg=guidance,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        denoise=1.0,
        seed=123,
    )
    assert captured == [True]
    monkeypatch.setattr(runtime, "sample_custom", sample_custom)
    result = run_ksampler_as_custom(
        runtime,
        latent,
        samplers=cast("Any", runtime)._samplers,
        schedulers=cast("Any", runtime)._schedulers,
        space=SEEDVR2_SIGMAS,
        flow=True,
        sampler_id="dinkster.euler",
        scheduler_id="dinkster.simple",
        steps=2,
        denoise=1.0,
        seed=123,
        cond=positive,
        cfg=guidance,
        error=SeedVR2RuntimeError,
    )
    assert type(result.output) is torch.Tensor
    assert torch.equal(result.output, expected)


def test_seedvr2_runtime_refuses_partial_and_wrong_conditioning_surfaces() -> None:
    runtime = _runtime(RecordingSeedVR2())
    positive, _ = _conditioning()
    latent = torch.zeros((1, 16, 2, 3, 4))
    request = _request()
    with pytest.raises(SeedVR2RuntimeError, match="shape"):
        runtime.sample_custom(
            torch.zeros((1, 16, 3, 4)),
            noise=torch.zeros((1, 16, 3, 4)),
            cond=positive,
            request=request,
        )
    with pytest.raises(SeedVR2RuntimeError, match="inpaint"):
        runtime.sample_custom(
            latent,
            noise=torch.zeros_like(latent),
            cond=positive,
            request=request,
            inpaint=cast("Any", object()),
        )
    with pytest.raises(SeedVR2RuntimeError, match="context windows"):
        runtime.sample_custom(
            latent,
            noise=torch.zeros_like(latent),
            cond=positive,
            request=request,
            context_windows=cast("Any", object()),
        )
    with pytest.raises(SeedVR2RuntimeError, match="adapter options: bogus_option"):
        runtime.sample_custom(
            latent,
            noise=torch.zeros_like(latent),
            cond=positive,
            request=request,
            bogus_option=True,
        )
    with pytest.raises(SeedVR2RuntimeError, match="adapter options: bogus_option"):
        runtime.sample(
            latent,
            cond=positive,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
            bogus_option=True,
        )
    assert not cast("RecordingSeedVR2", runtime.assembled.diffusion).calls


class RecordingVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        self.encode_inputs: list[torch.Tensor] = []
        self.decode_inputs: list[tuple[torch.Tensor, object]] = []
        self.tiled_encode: list[dict[str, object]] = []

    def encode(self, content: torch.Tensor) -> torch.Tensor:
        self.encode_inputs.append(content)
        return torch.full((content.shape[0], 16, 1, 2, 2), 4.0, dtype=content.dtype)

    @staticmethod
    def comfy_format_encoded(latent: torch.Tensor) -> torch.Tensor:
        return latent * 0.9152

    def decode(self, latent: torch.Tensor, seedvr2_tiling: object = None) -> torch.Tensor:
        self.decode_inputs.append((latent, seedvr2_tiling))
        return torch.full((latent.shape[0], 3, 1, 16, 16), -0.25, dtype=latent.dtype)

    def encode_tiled(self, content: torch.Tensor, **kwargs: object) -> torch.Tensor:
        self.tiled_encode.append({"content": content, **kwargs})
        return self.encode(content)


def test_seedvr2_codec_preserves_normalization_scaling_and_video_layout() -> None:
    vae = RecordingVAE()
    runtime = SeedVR2CodecRuntime(cast("Any", vae), compute_dtype=torch.float16)
    content = torch.full((2, 3, 1, 16, 16), 0.75)
    latent = runtime.encode_content(content)
    assert latent.shape == (2, 16, 1, 2, 2)
    assert latent.dtype is torch.float32
    assert torch.all(vae.encode_inputs[0] == torch.tensor(0.5, dtype=torch.float16))
    expected = torch.tensor(4.0, dtype=torch.float16).float() * 0.9152
    torch.testing.assert_close(latent, torch.full_like(latent, expected.item()))

    decoded = runtime.decode_latent(latent)
    assert decoded.shape == (2, 3, 1, 16, 16)
    assert decoded.dtype is torch.float32
    torch.testing.assert_close(decoded, torch.full_like(decoded, 0.375))
    assert vae.decode_inputs[0][0].dtype is torch.float16


def test_seedvr2_codec_adapts_ordinary_image_batches() -> None:
    vae = RecordingVAE()
    runtime = SeedVR2CodecRuntime(cast("Any", vae), compute_dtype=torch.float32)
    images = torch.full((2, 3, 16, 16), 0.75)

    latent = runtime.encode_content(images)
    decoded = runtime.decode_latent(latent)

    assert vae.encode_inputs[0].shape == (2, 3, 1, 16, 16)
    assert latent.shape == (2, 16, 2, 2)
    assert vae.decode_inputs[0][0].shape == (2, 16, 1, 2, 2)
    assert decoded.shape == (2, 3, 16, 16)


def test_seedvr2_codec_tiled_paths_use_upstream_spatial_units() -> None:
    vae = RecordingVAE()
    runtime = SeedVR2CodecRuntime(cast("Any", vae), compute_dtype=torch.float32)
    content = torch.zeros((1, 3, 1, 32, 48))
    runtime.encode_content_tiled(
        content,
        tile=(999, 256, 384),
        overlap=(0, 32, 32),
    )
    assert vae.tiled_encode[0]["tile_y"] == 256
    assert vae.tiled_encode[0]["tile_x"] == 384
    assert vae.tiled_encode[0]["overlap"] == 32

    latent = torch.zeros((1, 16, 1, 4, 6))
    runtime.decode_latent_tiled(
        latent,
        tile=(999, 32, 48),
        overlap=(0, 4, 6),
    )
    options = cast("dict[str, object]", vae.decode_inputs[-1][1])
    assert options["tile_size"] == (256, 384)
    assert options["tile_overlap"] == (32, 48)


def test_seedvr2_tiled_codec_adapts_ordinary_image_batches() -> None:
    vae = RecordingVAE()
    runtime = SeedVR2CodecRuntime(cast("Any", vae), compute_dtype=torch.float32)

    latent = runtime.encode_content_tiled(
        torch.zeros((2, 3, 32, 48)),
        tile=(999, 256, 384),
        overlap=(0, 32, 32),
    )
    decoded = runtime.decode_latent_tiled(
        latent,
        tile=(999, 32, 48),
        overlap=(0, 4, 6),
    )

    assert cast("torch.Tensor", vae.tiled_encode[0]["content"]).shape == (2, 3, 1, 32, 48)
    assert latent.shape == (2, 16, 2, 2)
    assert vae.decode_inputs[-1][0].shape == (2, 16, 1, 2, 2)
    assert decoded.shape == (2, 3, 16, 16)
