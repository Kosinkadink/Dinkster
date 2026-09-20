"""LTXV node legs execute bit-identically to the Python sampling API.

The KSampler node leg (the NativeKSampler multi-stream branch) must match
runtime.sample_multistream, and the decomposed node leg
(GenerationSamplerCustom over KSamplerSelect/BasicScheduler wire values)
must match runtime.sample_custom, via torch.equal on the sampled video
stream. The harness is the production path end to end: an independently
loaded diffusion component, the production
NativeRuntimeHandle with real CPU enrollment, and the shipped node
executors. The registered CLIP output also passes through LTXV Conditioning
and both sampling node surfaces, including frame-rate retiming.
"""

from __future__ import annotations

import importlib
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    LTXAV_19B_CONFIG,
    LTXV_2B_V09_CONFIG,
    ComponentBinding,
    Conditioning,
    ConditioningCarrier,
    CustomSamplingRequest,
    CustomSamplingResult,
    MultiStreamLatent,
    PreparedMultiStreamConditioning,
    ReconstructionRecipe,
    RuntimeKnobs,
    SamplingGuidance,
    WeightSourceBinding,
    WeightSourceRef,
    bind_component_conditioning,
)
from dinkster_inference_torch.denoise import prepare_multistream_noise
from dinkster_inference_torch.ltxav_runtime import LTXAVDiffusionRuntime, LTXAVTextRuntime
from dinkster_inference_torch.ltxv_runtime import (
    LTXVDiffusionRuntime,
    LTXVPreparedConditioning,
    ltxv_text_conditioning_to_carrier,
)
from dinkster_inference_torch.operations import InitlessOperations

ROOT = Path(__file__).resolve().parents[3]
for source in sorted((ROOT / "packages").glob("*/src")):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

native_arm = importlib.import_module("dinkster_compat_comfy.native_arm")
native_residency = importlib.import_module("dinkster_compat_comfy.native_residency")


class _Diffusion(torch.nn.Module):
    """Deterministic stand-in with the LTXV model config surface."""

    def __init__(self) -> None:
        super().__init__()
        self.config = LTXV_2B_V09_CONFIG
        self.patchify_proj = InitlessOperations().linear(128, 128, bias=False)
        self.frame_rates: list[float] = []

    def forward(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        frame_rate: float,
    ) -> torch.Tensor:
        del timesteps, attention_mask
        self.frame_rates.append(frame_rate)
        velocity = context.mean(dim=(1, 2), keepdim=True).reshape(-1, 1, 1, 1, 1)
        return torch.ones_like(latent) * velocity


def _recipe() -> ReconstructionRecipe:
    return ReconstructionRecipe(
        sources=(
            WeightSourceBinding(
                "diffusion",
                WeightSourceRef("blake3:" + "1" * 64, "ltxv-diffusion.safetensors", 1),
            ),
        ),
        family_id="dinkster.ltxv",
        component_identity=("family=dinkster.ltxv", "role=diffusion"),
        knobs=RuntimeKnobs(
            diffusion_dtype="float32",
            text_dtype="unloaded",
            vae_dtype="unloaded",
            fp8_matmul=False,
        ),
    )


def _text_recipe() -> ReconstructionRecipe:
    return ReconstructionRecipe(
        sources=(
            WeightSourceBinding(
                "t5xxl",
                WeightSourceRef("blake3:" + "2" * 64, "ltxv-t5xxl.safetensors", 1),
            ),
        ),
        family_id="dinkster.ltxv",
        component_identity=("family=dinkster.ltxv", "role=t5xxl"),
        knobs=RuntimeKnobs(
            diffusion_dtype="unloaded",
            text_dtype="float32",
            vae_dtype="unloaded",
            fp8_matmul=False,
        ),
    )


def _cpu_coordinator() -> Any:
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


def _runtime_and_handle() -> tuple[LTXVDiffusionRuntime, _Diffusion, Any, str]:
    diffusion = _Diffusion()
    recipe = _recipe()
    runtime = LTXVDiffusionRuntime(
        cast("Any", diffusion),
        runtime_identity=recipe.runtime_identity,
        compute_dtype=torch.float32,
    )
    handle = native_residency.NativeRuntimeHandle(
        runtime,
        "cpu",
        recipe=recipe,
        coordinator=_cpu_coordinator(),
    )
    return runtime, diffusion, handle, _text_recipe().runtime_identity


def _text_handle(text_identity: str) -> Any:
    from dinkster_inference_torch.residency import ResidentWeights

    recipe = _text_recipe()
    assert recipe.runtime_identity == text_identity
    return native_residency.NativeComponentHandle(
        object(),
        ResidentWeights({}, load_device="cpu", offload_device="cpu"),
        torch.device("cpu"),
        resource_identity=text_identity,
        coordinator=_cpu_coordinator(),
        recipe=recipe,
    )


def _prepared(frame_rate: float = 25.0) -> LTXVPreparedConditioning:
    return LTXVPreparedConditioning(torch.ones((1, 4, 4096)), 3, frame_rate)


def _wire(runtime: LTXVDiffusionRuntime, payload: LTXVPreparedConditioning) -> Any:
    return PreparedMultiStreamConditioning(runtime.conditioning_identity, payload)


def _lane(runtime: LTXVDiffusionRuntime, payload: LTXVPreparedConditioning) -> list[list[object]]:
    return [[_wire(runtime, payload), {}]]


def _streams() -> MultiStreamLatent[torch.Tensor]:
    return MultiStreamLatent.from_pairs((("video", torch.zeros((1, 128, 1, 2, 2))),))


def test_ksampler_node_leg_matches_sample_multistream() -> None:
    runtime, _, handle, _ = _runtime_and_handle()
    positive = _prepared()
    negative = LTXVPreparedConditioning(torch.full((1, 4, 4096), 0.5), 2, 25.0)

    output = native_arm.NativeKSampler.execute(
        model=handle,
        seed=7,
        steps=2,
        cfg=5.0,
        sampler_name="euler",
        scheduler="simple",
        positive=_lane(runtime, positive),
        negative=_lane(runtime, negative),
        latent_image={"samples": _streams()},
        denoise=1.0,
    )
    node_result = output["latent"]["samples"]
    assert type(node_result) is MultiStreamLatent
    assert node_result.roles == ("video",)

    direct = runtime.sample_multistream(
        _streams(),
        conditioning=positive,
        cfg=SamplingGuidance(negative, 5.0),
        sampler_id="euler",
        scheduler_id="simple",
        steps=2,
        denoise=1.0,
        seed=7,
    )

    assert torch.equal(node_result.by_role("video"), direct.by_role("video"))


def test_decomposed_node_leg_matches_sample_custom() -> None:
    runtime, _, handle, _ = _runtime_and_handle()
    positive = _prepared()
    negative = LTXVPreparedConditioning(torch.full((1, 4, 4096), 0.5), 2, 25.0)

    sampler_wire = native_arm.GenerationKSamplerSelect.execute(sampler_name="euler")["sampler"]
    sigmas_wire = native_arm.GenerationBasicScheduler.execute(
        model=handle,
        scheduler="simple",
        steps=2,
        denoise=1.0,
    )["sigmas"]

    output = native_arm.GenerationSamplerCustom.execute(
        model=handle,
        add_noise=True,
        noise_seed=7,
        cfg=5.0,
        positive=_lane(runtime, positive),
        negative=_lane(runtime, negative),
        sampler=sampler_wire,
        sigmas=sigmas_wire,
        latent_image={"samples": _streams()},
    )
    node_result = output["output"]["samples"]
    assert type(node_result) is MultiStreamLatent
    assert node_result.roles == ("video",)

    streams = _streams()
    request = CustomSamplingRequest(
        sampler_wire.descriptor,
        sampler_wire.options,
        sigmas_wire.values,
    )
    direct = runtime.sample_custom(
        streams,
        noise=prepare_multistream_noise(streams, 7),
        cond=_wire(runtime, positive),
        cfg=SamplingGuidance(_wire(runtime, negative), 5.0),
        request=request,
        seed=7,
    )
    assert type(direct) is CustomSamplingResult

    assert torch.equal(
        node_result.by_role("video"),
        cast("Any", direct.output).by_role("video"),
    )


def test_ltxv_conditioning_node_retimes_real_payload_through_decomposed_leg() -> None:
    runtime, diffusion, handle, _ = _runtime_and_handle()

    retimed = native_arm.GenerationLTXVConditioning.execute(
        positive=_lane(runtime, _prepared()),
        negative=_lane(
            runtime,
            LTXVPreparedConditioning(torch.full((1, 4, 4096), 0.5), 2, 25.0),
        ),
        frame_rate=50.0,
    )
    positive_payload = retimed["positive"][0][0].payload
    assert type(positive_payload) is LTXVPreparedConditioning
    assert positive_payload.frame_rate == 50.0

    sampler_wire = native_arm.GenerationKSamplerSelect.execute(sampler_name="euler")["sampler"]
    sigmas_wire = native_arm.GenerationBasicScheduler.execute(
        model=handle,
        scheduler="simple",
        steps=1,
        denoise=1.0,
    )["sigmas"]
    native_arm.GenerationSamplerCustom.execute(
        model=handle,
        add_noise=True,
        noise_seed=7,
        cfg=5.0,
        positive=retimed["positive"],
        negative=retimed["negative"],
        sampler=sampler_wire,
        sigmas=sigmas_wire,
        latent_image={"samples": _streams()},
    )

    assert diffusion.frame_rates
    assert all(rate == 50.0 for rate in diffusion.frame_rates)


def test_registered_clip_output_reaches_both_sampling_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, diffusion, handle, text_identity = _runtime_and_handle()

    class _TextRuntime:
        def __init__(self, _component: object) -> None:
            pass

        def encode_text(self, text: str, **_kwargs: object) -> Conditioning[torch.Tensor]:
            value = 1.0 if text == "positive" else 0.5
            return Conditioning(torch.full((1, 4, 4096), value))

    monkeypatch.setattr(
        importlib.import_module("dinkster_inference_torch"), "LTXVTextRuntime", _TextRuntime
    )
    clip = _text_handle(text_identity)
    positive = native_arm.GenerationClipTextEncode.execute(text="positive", clip=clip)[
        "conditioning"
    ]
    negative = native_arm.GenerationClipTextEncode.execute(text="negative", clip=clip)[
        "conditioning"
    ]
    assert type(positive) is ConditioningCarrier
    assert type(negative) is ConditioningCarrier

    retimed = native_arm.GenerationLTXVConditioning.execute(
        positive=positive,
        negative=negative,
        frame_rate=50.0,
    )
    assert type(retimed["positive"]) is ConditioningCarrier
    assert type(retimed["negative"]) is ConditioningCarrier

    ksampler = native_arm.GenerationKSampler.execute(
        model=handle,
        seed=7,
        steps=1,
        cfg=5.0,
        sampler_name="euler",
        scheduler="simple",
        positive=retimed["positive"],
        negative=retimed["negative"],
        latent_image={"samples": _streams()},
        denoise=1.0,
    )["latent"]["samples"]
    assert type(ksampler) is MultiStreamLatent
    assert ksampler.roles == ("video",)
    assert diffusion.frame_rates
    assert all(rate == 50.0 for rate in diffusion.frame_rates)

    diffusion.frame_rates.clear()
    sampler_wire = native_arm.GenerationKSamplerSelect.execute(sampler_name="euler")["sampler"]
    sigmas_wire = native_arm.GenerationBasicScheduler.execute(
        model=handle,
        scheduler="simple",
        steps=1,
        denoise=1.0,
    )["sigmas"]
    custom = native_arm.GenerationSamplerCustom.execute(
        model=handle,
        add_noise=True,
        noise_seed=7,
        cfg=5.0,
        positive=retimed["positive"],
        negative=retimed["negative"],
        sampler=sampler_wire,
        sigmas=sigmas_wire,
        latent_image={"samples": _streams()},
    )["output"]["samples"]
    assert type(custom) is MultiStreamLatent
    assert custom.roles == ("video",)
    assert diffusion.frame_rates
    assert all(rate == 50.0 for rate in diffusion.frame_rates)


def test_split_components_reach_both_sampling_nodes_through_one_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, diffusion, handle, text_identity = _runtime_and_handle()
    text_binding = ComponentBinding("t5xxl", "dinkster.ltxv", text_identity)

    def conditioning(value: float) -> ConditioningCarrier:
        carrier = ltxv_text_conditioning_to_carrier(Conditioning(torch.full((1, 4, 4096), value)))
        return bind_component_conditioning(carrier, text_binding)

    retimed = native_arm.GenerationLTXVConditioning.execute(
        positive=conditioning(1.0),
        negative=conditioning(0.5),
        frame_rate=50.0,
    )
    sample_custom_calls: list[str] = []
    original_sample_custom = cast("Any", LTXVDiffusionRuntime.sample_custom)

    def sample_custom(self: LTXVDiffusionRuntime, *args: Any, **kwargs: Any) -> Any:
        sample_custom_calls.append(self.runtime_identity)
        return original_sample_custom(self, *args, **kwargs)

    monkeypatch.setattr(LTXVDiffusionRuntime, "sample_custom", sample_custom)

    ksampler = native_arm.GenerationKSampler.execute(
        model=handle,
        seed=7,
        steps=1,
        cfg=5.0,
        sampler_name="euler",
        scheduler="simple",
        positive=retimed["positive"],
        negative=retimed["negative"],
        latent_image={"samples": _streams()},
        denoise=1.0,
    )["latent"]["samples"]
    assert type(ksampler) is MultiStreamLatent
    assert ksampler.roles == ("video",)
    assert sample_custom_calls and len(sample_custom_calls) == 1
    assert diffusion.frame_rates and all(rate == 50.0 for rate in diffusion.frame_rates)

    diffusion.frame_rates.clear()
    sampler_wire = native_arm.GenerationKSamplerSelect.execute(sampler_name="euler")["sampler"]
    sigmas_wire = native_arm.GenerationBasicScheduler.execute(
        model=handle,
        scheduler="simple",
        steps=1,
        denoise=1.0,
    )["sigmas"]
    custom = native_arm.GenerationSamplerCustom.execute(
        model=handle,
        add_noise=True,
        noise_seed=7,
        cfg=5.0,
        positive=retimed["positive"],
        negative=retimed["negative"],
        sampler=sampler_wire,
        sigmas=sigmas_wire,
        latent_image={"samples": _streams()},
    )["output"]["samples"]
    assert type(custom) is MultiStreamLatent
    assert custom.roles == ("video",)
    assert len(sample_custom_calls) == 2
    assert diffusion.frame_rates and all(rate == 50.0 for rate in diffusion.frame_rates)


class _LTXAVDiffusion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = LTXAV_19B_CONFIG
        self.patchify_proj = InitlessOperations().linear(128, 128, bias=False)
        self.frame_rates: list[float] = []

    @staticmethod
    def preprocess_text_embeds(context: torch.Tensor) -> torch.Tensor:
        return context

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        timesteps: torch.Tensor,
        audio_timesteps: torch.Tensor,
        context: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        frame_rate: float,
        denoise_mask: torch.Tensor | None = None,
        ref_audio_tokens: torch.Tensor | None = None,
        stg_self_attn_blocks: frozenset[int] = frozenset(),
        a2v_cross_attention: bool = True,
        v2a_cross_attention: bool = True,
        generated_keyframes: object | None = None,
        context_preprocessed: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert context_preprocessed is True
        del (
            timesteps,
            audio_timesteps,
            attention_mask,
            denoise_mask,
            ref_audio_tokens,
            stg_self_attn_blocks,
            a2v_cross_attention,
            v2a_cross_attention,
            generated_keyframes,
        )
        self.frame_rates.append(frame_rate)
        velocity = context.mean(dim=(1, 2))
        return (
            torch.ones_like(video) * velocity.reshape(-1, 1, 1, 1, 1),
            torch.ones_like(audio) * velocity.reshape(-1, 1, 1, 1),
        )


def _ltxav_recipe() -> ReconstructionRecipe:
    return ReconstructionRecipe(
        sources=(
            WeightSourceBinding(
                "diffusion",
                WeightSourceRef("blake3:" + "3" * 64, "ltxav-diffusion.safetensors", 1),
            ),
        ),
        family_id="dinkster.ltxav",
        component_identity=("family=dinkster.ltxav", "role=diffusion"),
        knobs=RuntimeKnobs(
            diffusion_dtype="float32",
            text_dtype="unloaded",
            vae_dtype="unloaded",
            fp8_matmul=False,
        ),
    )


def _ltxav_runtime_and_handle() -> tuple[LTXAVDiffusionRuntime, _LTXAVDiffusion, Any]:
    diffusion = _LTXAVDiffusion()
    recipe = _ltxav_recipe()
    runtime = LTXAVDiffusionRuntime(
        cast("Any", diffusion),
        runtime_identity=recipe.runtime_identity,
        compute_dtype=torch.float32,
    )
    handle = native_residency.NativeRuntimeHandle(
        runtime,
        "cpu",
        recipe=recipe,
        coordinator=_cpu_coordinator(),
    )
    return runtime, diffusion, handle


def _ltxav_text_handle() -> Any:
    class TextEncoder:
        @staticmethod
        def encode(text: str) -> Conditioning[torch.Tensor]:
            value = 1.0 if text == "positive" else 0.5
            return Conditioning(torch.full((1, 4, 7680), value))

    runtime = object.__new__(LTXAVTextRuntime)
    raw_runtime = cast("Any", runtime)
    raw_runtime._encoder = TextEncoder()
    raw_runtime._text_dim = 7680
    raw_runtime._text_stream = "gemma3_12b"
    handle = object.__new__(native_arm._LTXAVTextHandle)
    handle._runtime = runtime
    handle._handles = (
        SimpleNamespace(
            load_device=torch.device("cpu"),
            require_active=lambda: None,
            stage=nullcontext,
        ),
    )
    handle.resource_identity = "native:dinkster.ltxav:" + "4" * 64
    handle.load_device = torch.device("cpu")
    return handle


def _ltxav_streams() -> MultiStreamLatent[torch.Tensor]:
    return MultiStreamLatent.from_pairs(
        (
            ("video", torch.zeros((1, 128, 1, 2, 2))),
            ("audio", torch.zeros((1, 8, 1, 16))),
        )
    )


def test_ltxv_latent_upsampler_normalizes_and_preserves_latent_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference_torch = importlib.import_module("dinkster_inference_torch")
    stages: list[str] = []

    class Upscaler(torch.nn.Module):
        @staticmethod
        def forward(value: torch.Tensor) -> torch.Tensor:
            return value.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4) * 2

    class Statistics:
        @staticmethod
        def un_normalize(value: torch.Tensor) -> torch.Tensor:
            return value + 10

        @staticmethod
        def normalize(value: torch.Tensor) -> torch.Tensor:
            return value - 3

    class VideoVAE(torch.nn.Module):
        per_channel_statistics = Statistics()

    class Coordinator:
        @staticmethod
        def locked() -> Any:
            return nullcontext()

    coordinator = Coordinator()

    def stage(name: str) -> Any:
        class Stage:
            def __enter__(self) -> None:
                stages.append(f"{name}:enter")

            def __exit__(self, *_args: object) -> None:
                stages.append(f"{name}:exit")

        return Stage()

    upscaler = SimpleNamespace(
        component=Upscaler(),
        coordinator=coordinator,
        load_device=torch.device("cpu"),
        recipe=SimpleNamespace(knobs=SimpleNamespace(vae_dtype="float32")),
        stage=lambda: stage("upscaler"),
    )
    vae = SimpleNamespace(
        component=VideoVAE(),
        coordinator=coordinator,
        stage=lambda: stage("vae"),
    )
    monkeypatch.setattr(inference_torch, "LTXLatentUpsampler", Upscaler)
    monkeypatch.setattr(inference_torch, "LTXVideoVAE", VideoVAE)
    monkeypatch.setattr(inference_torch, "LTXDiffusionVideoVAE", VideoVAE)

    def component_handle(value: object, *_args: object) -> object:
        return upscaler if value == "upscaler" else vae

    monkeypatch.setattr(native_arm, "load_registered_component", component_handle)
    latent = torch.arange(128 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 128, 2, 3, 4)

    output = native_arm.GenerationLTXVLatentUpsampler.execute(
        samples={"samples": latent, "noise_mask": torch.ones(1), "custom": "preserved"},
        upscale_model="upscaler",
        vae="vae",
    )["latent"]

    expected = (latent + 10).repeat_interleave(2, dim=3).repeat_interleave(2, dim=4) * 2 - 3
    assert torch.equal(output["samples"], expected)
    assert output["samples"].shape == (1, 128, 2, 6, 8)
    assert output["custom"] == "preserved"
    assert "noise_mask" not in output
    assert stages == ["vae:enter", "upscaler:enter", "upscaler:exit", "vae:exit"]


def test_registered_ltxav_clip_output_reaches_both_sampling_nodes_through_one_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, diffusion, handle = _ltxav_runtime_and_handle()
    clip = _ltxav_text_handle()
    positive = native_arm.GenerationClipTextEncode.execute(text="positive", clip=clip)[
        "conditioning"
    ]
    negative = native_arm.GenerationClipTextEncode.execute(text="negative", clip=clip)[
        "conditioning"
    ]
    assert type(positive) is ConditioningCarrier
    assert type(negative) is ConditioningCarrier

    retimed = native_arm.GenerationLTXAVConditioning.execute(
        positive=positive,
        negative=negative,
        frame_rate=50.0,
    )
    sample_custom_calls: list[str] = []
    original_sample_custom = cast("Any", LTXAVDiffusionRuntime.sample_custom)

    def sample_custom(self: LTXAVDiffusionRuntime, *args: Any, **kwargs: Any) -> Any:
        sample_custom_calls.append(self.runtime_identity)
        return original_sample_custom(self, *args, **kwargs)

    monkeypatch.setattr(LTXAVDiffusionRuntime, "sample_custom", sample_custom)

    ksampler = native_arm.GenerationKSampler.execute(
        model=handle,
        seed=7,
        steps=1,
        cfg=5.0,
        sampler_name="euler",
        scheduler="simple",
        positive=retimed["positive"],
        negative=retimed["negative"],
        latent_image={"samples": _ltxav_streams()},
        denoise=1.0,
    )["latent"]["samples"]
    assert type(ksampler) is MultiStreamLatent
    assert ksampler.roles == ("video", "audio")
    assert len(sample_custom_calls) == 1
    assert diffusion.frame_rates and all(rate == 50.0 for rate in diffusion.frame_rates)

    diffusion.frame_rates.clear()
    sampler_wire = native_arm.GenerationKSamplerSelect.execute(sampler_name="euler")["sampler"]
    sigmas_wire = native_arm.GenerationBasicScheduler.execute(
        model=handle,
        scheduler="simple",
        steps=1,
        denoise=1.0,
    )["sigmas"]
    custom = native_arm.GenerationSamplerCustom.execute(
        model=handle,
        add_noise=True,
        noise_seed=7,
        cfg=5.0,
        positive=retimed["positive"],
        negative=retimed["negative"],
        sampler=sampler_wire,
        sigmas=sigmas_wire,
        latent_image={"samples": _ltxav_streams()},
    )["output"]["samples"]
    assert type(custom) is MultiStreamLatent
    assert custom.roles == ("video", "audio")
    assert len(sample_custom_calls) == 2
    assert diffusion.frame_rates and all(rate == 50.0 for rate in diffusion.frame_rates)
