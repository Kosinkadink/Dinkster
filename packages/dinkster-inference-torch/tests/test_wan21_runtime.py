"""Wan 2.1 T2V runtime contracts over the shared sampling engine."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    WAN21,
    WAN21_ANIMATE2_SETTINGS_KEY,
    WAN21_CAUSAL_AR_1_3B,
    WAN21_CODEC,
    WAN21_FLOW_RVS_CODEC,
    WAN21_HUMO_17B,
    WAN21_I2V_14B,
    WAN21_MULTITALK,
    WAN21_SCAIL_REPLACEMENT_KEY,
    WAN21_SIGMAS,
    WAN21_T2V_14B,
    WAN22,
    WAN22_BERNINI_14B,
    WAN22_CODEC,
    WAN22_DANCER_SETTINGS_KEY,
    WAN22_FUN_CONTROL_5B,
    WAN22_FUN_INPAINT_5B,
    WAN22_S2V_14B,
    WAN22_SIGMAS,
    WAN22_TI2V_5B,
    WAN22_WANDANCER_14B,
    CodecDescriptor,
    Conditioning,
    ConditioningBatching,
    ConditioningBatchingMode,
    ConditioningChannel,
    ConditioningRecord,
    ConditioningSet,
    ContextFuseMethod,
    ContextWindowSchedule,
    ContextWindowsSpec,
    CustomSamplingRequest,
    CustomSamplingRuntime,
    DualSamplingGuidance,
    FlowSigmas,
    MultiStreamConditioningRuntime,
    MultiStreamLatent,
    PayloadDescriptor,
    PayloadReference,
    PercentRange,
    PreparedMultiStreamConditioning,
    SamplingCancelled,
    SamplingGuidance,
    SamplingSegment,
    SamplingStateEvent,
    StepEvent,
    TokenLayoutDescriptor,
    TokenSegmentDescriptor,
    Wan21Animate2Settings,
    Wan21Config,
    Wan21PoseBlockCacheDevice,
    Wan21PoseBlockCacheSettings,
    Wan21PoseBlockCacheStorage,
    Wan22DancerSettings,
    make_conditioning_carrier,
    sampling_sigmas,
    select_builtin_sampler,
)
from dinkster_inference_torch import sampling_execution as sampling_engine
from dinkster_inference_torch import wan21_multitalk as wan21_multitalk_module
from dinkster_inference_torch import wan21_runtime as wan21_runtime_module
from dinkster_inference_torch._conditioning_layout import (
    DeclaredConditioning,
    declare_text_conditioning,
)
from dinkster_inference_torch.conditioning_adapters import basic_conditioning_to_carrier
from dinkster_inference_torch.context_windows import apply_freenoise
from dinkster_inference_torch.denoise import prepare_noise
from dinkster_inference_torch.guidance_transforms import epsilon_scaling
from dinkster_inference_torch.module_residency import enroll_assembled
from dinkster_inference_torch.payloads import tensor_to_payload_binding
from dinkster_inference_torch.schedules import torch_scheduler_registry
from dinkster_inference_torch.solvers import torch_sampler_registry
from dinkster_inference_torch.wan21_causal import Wan21CausalModel
from dinkster_inference_torch.wan21_multitalk import (
    Wan21MultiTalk,
    Wan21MultiTalkExecution,
    wan21_multitalk_resource_digest,
    wan21_multitalk_tensor_digest,
)
from dinkster_inference_torch.wan21_runtime import (
    Wan21CausalDiffusionRuntime,
    Wan21DiffusionRuntime,
    Wan21InfiniteTalkExecution,
    Wan21PreparedConditioning,
    Wan21Runtime,
    Wan21RuntimeError,
    _normalize_concat_latent,  # pyright: ignore[reportPrivateUsage]
    compose_wan21_animate2_conditioning,
    compose_wan21_animate_conditioning,
    compose_wan21_humo_conditioning,
    compose_wan21_scail_conditioning,
    compose_wan22_dancer_conditioning,
    compose_wan22_s2v_conditioning,
)

FUSE_CFG_LANES = ConditioningBatching(ConditioningBatchingMode.MAX_FUSED_LANES, 2)
FUSE_DUAL_CFG_LANES = ConditioningBatching(ConditioningBatchingMode.MAX_FUSED_LANES, 3)


def test_wan_runtime_retains_text_offload_storage() -> None:
    assert Wan21Runtime.retained_offload_storage_components == frozenset({"umt5xxl"})


def test_diffusion_only_assembly_enrolls_with_explicit_storage_policy() -> None:
    diffusion = torch.nn.Identity()
    assembled = wan21_runtime_module._Wan21DiffusionAssembly(  # pyright: ignore[reportPrivateUsage]
        cast("Any", diffusion), cast("Any", SimpleNamespace(id="dinkster.wan21"))
    )

    enrolled = enroll_assembled(
        cast("Any", assembled),
        load_device="cpu",
        offload_device="cpu",
    )

    assert tuple(enrolled) == ("diffusion",)
    assert not enrolled.storage_dtype_report.enabled


def test_diffusion_only_assembly_uses_bound_compute_dtype() -> None:
    linear = torch.nn.Identity()
    linear.__dict__["compute_dtype"] = torch.bfloat16
    diffusion = SimpleNamespace(time_embedding=(linear,))
    assembled = wan21_runtime_module._Wan21DiffusionAssembly(  # pyright: ignore[reportPrivateUsage]
        cast("Any", diffusion), cast("Any", SimpleNamespace(id="dinkster.wan21"))
    )

    assert assembled.compute_dtype("diffusion") is torch.bfloat16
    assert assembled.compute_dtype("vae") is None


@pytest.mark.parametrize("causal", (False, True))
def test_component_runtime_binds_selected_dtype_and_public_identity(causal: bool) -> None:
    from dinkster_inference.component_catalog import default_component_registry
    from dinkster_inference.component_registry import build_component_runtime

    if causal:
        model = Wan21CausalModel.__new__(Wan21CausalModel)
        model.config = WAN21_CAUSAL_AR_1_3B
    else:
        model = wan21_runtime_module.Wan21Model.__new__(wan21_runtime_module.Wan21Model)
        model.config = WAN21_T2V_14B
    torch.nn.Module.__init__(model)
    model.__dict__["time_embedding"] = (SimpleNamespace(),)
    descriptor = default_component_registry().get(WAN21.id)
    assert descriptor is not None
    identity = "native:dinkster.wan21:" + "1" * 64

    runtime = build_component_runtime(
        descriptor,
        SimpleNamespace(module=model),
        identity,
        torch.bfloat16,
    )

    assert runtime.runtime_identity == identity
    assert runtime.assembled.compute_dtype("diffusion") is torch.bfloat16


@pytest.mark.parametrize("compute_dtype", (torch.float32, torch.bfloat16, torch.float16))
@pytest.mark.parametrize("cast_storage", (False, True))
def test_diffusion_only_assembly_resolves_loader_selected_dtype(
    compute_dtype: torch.dtype, cast_storage: bool
) -> None:
    from dinkster_inference_torch.assemble import (
        _pick_operations,  # pyright: ignore[reportPrivateUsage]
    )
    from dinkster_inference_torch.operations import INITLESS, bound_compute_dtype

    storage_dtype = (
        (torch.float16 if compute_dtype is torch.float32 else torch.float32)
        if cast_storage
        else compute_dtype
    )
    weight = torch.ones((2, 2), dtype=storage_dtype)
    operations = _pick_operations({"weight": weight}, frozenset(), compute_dtype)
    linear = operations.linear(2, 2).to(dtype=storage_dtype)
    assert (operations is INITLESS) is not cast_storage
    assert bound_compute_dtype(linear) is (compute_dtype if cast_storage else None)
    assembled = wan21_runtime_module._Wan21DiffusionAssembly(  # pyright: ignore[reportPrivateUsage]
        cast("Any", SimpleNamespace(time_embedding=(linear,))), WAN21
    )

    assert assembled.compute_dtype("diffusion") is compute_dtype


class _Diffusion(torch.nn.Module):
    def __init__(self, model_type: str = "t2v") -> None:
        super().__init__()
        in_channels = {
            "t2v": 16,
            "i2v": 36,
            "flf": 36,
            "i2v22": 36,
            "ti2v": 48,
            "vace": 16,
            "fun22": 52,
            "camera21": 32,
            "camera22": 36,
            "animate": 36,
            "animate2": 36,
            "scail": 20,
            "scail2": 20,
            "flow_rvs": 16,
            "bernini": 16,
            "s2v": 16,
            "humo": 36,
        }[model_type]
        out_channels = 48 if model_type == "ti2v" else 16
        semantic_type = (
            "t2v"
            if model_type
            in ("i2v22", "vace", "fun22", "camera22", "flow_rvs", "bernini", "s2v", "humo")
            else "i2v"
            if model_type in ("flf", "camera21", "animate", "animate2", "scail", "scail2")
            else model_type
        )
        self.config: Any = SimpleNamespace(
            text_dim=8,
            model_type=semantic_type,
            model_variant=(
                model_type
                if model_type
                in (
                    "animate",
                    "animate2",
                    "scail",
                    "scail2",
                    "flow_rvs",
                    "bernini",
                    "s2v",
                    "humo",
                )
                else "base"
            ),
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_size=16,
            ffn_hidden_size=32,
            num_heads=2,
            num_layers=1,
            flf_pos_embed_token_number=514 if model_type == "flf" else None,
            reference_channels=16 if model_type == "fun22" else None,
            vace_layers=1 if model_type == "vace" else None,
            camera_channels=24 if model_type.startswith("camera") else None,
        )
        self.patch_embedding = torch.nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
        self.reference_calls: list[torch.Tensor] = []
        self.vace_calls: list[tuple[torch.Tensor, tuple[float, ...]]] = []
        self.camera_calls: list[torch.Tensor] = []
        self.temporal_reference_calls: list[torch.Tensor] = []
        self.animate_calls: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []
        self.animate2_calls: list[
            tuple[
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
                float,
                float,
                object | None,
            ]
        ] = []
        self.scail_calls: list[
            tuple[
                torch.Tensor,
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
                bool,
            ]
        ] = []
        self.uni3c_calls: list[tuple[object, torch.Tensor]] = []
        self.context_calls: list[tuple[torch.Tensor, ...]] = []
        self.s2v_calls: list[
            tuple[
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
            ]
        ] = []
        self.humo_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.dancer_calls: list[
            tuple[
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
                float,
                float,
            ]
        ] = []
        self.multitalk_calls: list[object] = []

    def forward(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        vision: torch.Tensor | None = None,
        *,
        temporal_reference: torch.Tensor | None = None,
        reference_latent: torch.Tensor | None = None,
        vace_context: torch.Tensor | None = None,
        vace_strength: tuple[float, ...] | None = None,
        camera_conditions: torch.Tensor | None = None,
        pose_latents: torch.Tensor | None = None,
        face_pixel_values: torch.Tensor | None = None,
        pose_context: torch.Tensor | None = None,
        pose_vision: torch.Tensor | None = None,
        pose_strength: float = 1.0,
        reference_strength: float = 1.0,
        pose_cache: object | None = None,
        reference_mask: torch.Tensor | None = None,
        driving_mask: torch.Tensor | None = None,
        replacement: bool = False,
        uni3c: object | None = None,
        uni3c_input: torch.Tensor | None = None,
        context_latents: tuple[torch.Tensor, ...] = (),
        audio_embed: torch.Tensor | None = None,
        control_video: torch.Tensor | None = None,
        reference_motion: torch.Tensor | None = None,
        reference_vision: torch.Tensor | None = None,
        fps: float = 30.0,
        audio_inject_scale: float = 1.0,
        multitalk: object | None = None,
    ) -> torch.Tensor:
        self.calls.append(
            (
                latent.detach().clone(),
                timesteps.detach().clone(),
                context.detach().clone(),
                None if vision is None else vision.detach().clone(),
            )
        )
        if reference_latent is not None:
            self.reference_calls.append(reference_latent.detach().clone())
        if vace_context is not None and vace_strength is not None:
            self.vace_calls.append((vace_context.detach().clone(), vace_strength))
        if camera_conditions is not None:
            self.camera_calls.append(camera_conditions.detach().clone())
        if temporal_reference is not None:
            self.temporal_reference_calls.append(temporal_reference.detach().clone())
        if self.config.model_variant == "animate":
            self.animate_calls.append(
                (
                    None if pose_latents is None else pose_latents.detach().clone(),
                    None if face_pixel_values is None else face_pixel_values.detach().clone(),
                )
            )
        if self.config.model_variant == "animate2":
            self.animate2_calls.append(
                (
                    None if pose_latents is None else pose_latents.detach().clone(),
                    None if pose_context is None else pose_context.detach().clone(),
                    None if pose_vision is None else pose_vision.detach().clone(),
                    pose_strength,
                    reference_strength,
                    pose_cache,
                )
            )
        if self.config.model_variant in ("scail", "scail2"):
            assert reference_latent is not None
            self.scail_calls.append(
                (
                    reference_latent.detach().clone(),
                    None if pose_latents is None else pose_latents.detach().clone(),
                    None if reference_mask is None else reference_mask.detach().clone(),
                    None if driving_mask is None else driving_mask.detach().clone(),
                    replacement,
                )
            )
        if uni3c is not None:
            assert uni3c_input is not None
            self.uni3c_calls.append((uni3c, uni3c_input.detach().clone()))
        if context_latents:
            self.context_calls.append(context_latents)
        if self.config.model_variant == "s2v":
            self.s2v_calls.append(
                (
                    None if audio_embed is None else audio_embed.detach().clone(),
                    None if reference_latent is None else reference_latent.detach().clone(),
                    None if reference_motion is None else reference_motion.detach().clone(),
                    None if control_video is None else control_video.detach().clone(),
                )
            )
        if self.config.model_variant == "humo":
            assert audio_embed is not None and reference_latent is not None
            self.humo_calls.append(
                (audio_embed.detach().clone(), reference_latent.detach().clone())
            )
        if self.config.model_variant == "wandancer":
            self.dancer_calls.append(
                (
                    None if vision is None else vision.detach().clone(),
                    None if reference_vision is None else reference_vision.detach().clone(),
                    None if audio_embed is None else audio_embed.detach().clone(),
                    fps,
                    audio_inject_scale,
                )
            )
        if multitalk is not None:
            self.multitalk_calls.append(multitalk)
        velocity = context.mean(dim=(1, 2), keepdim=True).reshape(-1, 1, 1, 1, 1)
        return torch.ones_like(latent[:, : self.config.out_channels]) * velocity


class _VAE:
    def __init__(self) -> None:
        self.processed_in = 0
        self.processed_out = 0

    def process_in(self, latent: torch.Tensor) -> torch.Tensor:
        self.processed_in += 1
        return latent + 20.0

    def process_out(self, latent: torch.Tensor) -> torch.Tensor:
        self.processed_out += 1
        return latent + 10.0


class _Codec:
    def __init__(self) -> None:
        self.encoded: list[torch.Tensor] = []
        self.decoded: list[torch.Tensor] = []

    def encode(self, content: torch.Tensor) -> torch.Tensor:
        if content.ndim != 5:
            raise ValueError("content must be rank 5")
        self.encoded.append(content)
        return content[:, :1] + 2.0

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 5:
            raise ValueError("latent must be rank 5")
        self.decoded.append(latent)
        return latent[:, :1] + 3.0


class _Tokenizer:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def tokenize(self, text: str) -> tuple[str, ...]:
        self.texts.append(text)
        return (text,)


class _TextEncoder:
    def __init__(self) -> None:
        self.spans: list[object] = []

    def encode(self, spans: object) -> Conditioning[torch.Tensor]:
        self.spans.append(spans)
        return declare_text_conditioning(Conditioning(torch.ones((1, 4, 8)), None), 4)


@pytest.fixture(params=("dinkster.wan21", "example.renamed-video"))
def family_id(request: pytest.FixtureRequest) -> str:
    return cast("str", request.param)


def _runtime(
    model_type: str = "t2v", *, family_id: str | None = None
) -> tuple[Wan21Runtime, _Diffusion, _VAE, _Codec]:
    diffusion = _Diffusion(model_type)
    vae = _VAE()
    codec = _Codec()

    def compute_dtype(_component: object) -> torch.dtype:
        return torch.float32

    assembled = SimpleNamespace(
        family=(
            replace(WAN22 if model_type == "ti2v" else WAN21, id=family_id)
            if family_id is not None
            else WAN22
            if model_type == "ti2v"
            else WAN21
        ),
        diffusion=diffusion,
        vae=vae,
        clip_vision=(
            object()
            if model_type in ("i2v", "flf", "animate", "animate2", "scail", "scail2")
            else None
        ),
        compute_dtype=compute_dtype,
    )
    runtime = object.__new__(Wan21Runtime)
    raw = cast("Any", runtime)
    raw._assembled = assembled
    raw._runtime_identity = "test:wan21"
    raw.codec = codec
    raw._samplers = torch_sampler_registry()
    raw._schedulers = torch_scheduler_registry()
    raw._tokenizer = _Tokenizer()
    raw._text_encoder = _TextEncoder()
    raw._latent_process_in = vae.process_in
    raw._latent_process_out = vae.process_out
    raw._pose_cache_settings = None
    return runtime, diffusion, vae, codec


def _video(shape: tuple[int, ...] = (1, 16, 1, 2, 2)) -> torch.Tensor:
    return torch.zeros(shape)


def _wire(
    runtime: Wan21Runtime,
    conditioning: Wan21PreparedConditioning,
) -> PreparedMultiStreamConditioning:
    return PreparedMultiStreamConditioning(runtime.conditioning_identity, conditioning)


def _sample(
    runtime: Wan21Runtime,
    latent: object,
    conditioning: object,
    *,
    cfg: SamplingGuidance[object] | DualSamplingGuidance[object] | None = None,
    **kwargs: object,
) -> torch.Tensor:
    prepared = (
        runtime.prepare_text_conditioning(cast("Any", conditioning))
        if type(conditioning) in (Conditioning, DeclaredConditioning)
        else conditioning
    )
    prepared_cfg = cfg
    if cfg is not None and type(cfg.uncond) in (Conditioning, DeclaredConditioning):
        prepared_cfg = SamplingGuidance(
            runtime.prepare_text_conditioning(cast("Any", cfg.uncond)), cfg.scale
        )
    streams = (
        latent
        if type(latent) is MultiStreamLatent
        else MultiStreamLatent.from_pairs((("video", cast("Any", latent)),))
    )
    result = runtime.sample_multistream(
        cast("Any", streams),
        conditioning=prepared,
        cfg=cast("Any", prepared_cfg),
        sampler_id=cast("str", kwargs.pop("sampler_id", "euler")),
        scheduler_id=cast("str", kwargs.pop("scheduler_id", "simple")),
        steps=cast("int", kwargs.pop("steps", 2)),
        denoise=cast("float", kwargs.pop("denoise", 1.0)),
        seed=cast("int", kwargs.pop("seed", 123)),
        **cast("Any", kwargs),
    )
    return result.by_role("video")


def _custom_request(sampler_id: str, **options: object) -> CustomSamplingRequest[torch.Tensor]:
    selection = select_builtin_sampler(sampler_id, **options)
    descriptor = torch_sampler_registry().get(selection.sampler_id)
    assert descriptor is not None
    return CustomSamplingRequest(descriptor, selection.options, (1.0, 0.5, 0.0))


def test_multistream_facade_delegates_once_to_custom_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _, _, _ = _runtime()
    latent = MultiStreamLatent.from_pairs((("video", _video()),))
    prepared = runtime.prepare_text_conditioning(Conditioning(torch.ones((1, 2, 8))))
    original = runtime.sample_custom
    calls: list[dict[str, object]] = []

    def capture(latent_value: object, **kwargs: object) -> object:
        calls.append(kwargs)
        return original(cast("Any", latent_value), **cast("Any", kwargs))

    monkeypatch.setattr(runtime, "sample_custom", capture)
    result = runtime.sample_multistream(
        latent,
        conditioning=prepared,
        sampler_id="euler",
        scheduler_id="simple",
        steps=2,
        denoise=1.0,
        seed=17,
    )

    assert result.roles == ("video",)
    assert len(calls) == 1
    assert type(calls[0]["request"]) is CustomSamplingRequest
    assert type(calls[0]["noise"]) is MultiStreamLatent
    assert type(calls[0]["cond"]) is PreparedMultiStreamConditioning
    assert calls[0]["noise_inds"] is None


def test_standard_wan_custom_sampling_matches_multistream_facade() -> None:
    runtime, _, _, _ = _runtime()
    latent = MultiStreamLatent.from_pairs((("video", _video()),))
    prepared = runtime.prepare_text_conditioning(Conditioning(torch.ones((1, 2, 8))))
    direct = runtime.sample_custom(
        latent,
        noise=MultiStreamLatent.from_pairs((("video", prepare_noise(_video(), 23)),)),
        cond=_wire(runtime, prepared),
        cfg=None,
        request=replace(
            _custom_request("euler"),
            sigmas=runtime.custom_sampling_sigmas("dinkster.simple", 2, 1.0),
        ),
        seed=23,
    )
    facade = runtime.sample_multistream(
        latent,
        conditioning=prepared,
        sampler_id="euler",
        scheduler_id="simple",
        steps=2,
        denoise=1.0,
        seed=23,
    )

    assert type(direct.output) is MultiStreamLatent
    assert torch.equal(direct.output.by_role("video"), facade.by_role("video"))
    assert type(direct.denoised_output) is MultiStreamLatent


def test_s2v_conditioning_materializes_batches_normalizes_and_dispatches(
    monkeypatch: pytest.MonkeyPatch,
    family_id: str,
) -> None:
    runtime, model, vae, _codec = _runtime("s2v", family_id=family_id)
    model.config = WAN22_S2V_14B
    monkeypatch.setattr(wan21_runtime_module, "Wan22S2VModel", _Diffusion)
    text = runtime.text_conditioning_carrier(Conditioning(torch.ones((1, 2, 4096)), None))
    audio = torch.ones((1, 25, 1024, 8))
    reference = torch.full((1, 16, 1, 2, 2), 2.0)
    motion = torch.full((1, 16, 3, 2, 2), 3.0)
    control = torch.full((1, 16, 2, 2, 2), 4.0)
    positive = compose_wan22_s2v_conditioning(
        text,
        audio_embed=audio,
        reference_latent=reference,
        reference_motion=motion,
        control_video=control,
    )
    negative = compose_wan22_s2v_conditioning(
        text,
        audio_embed=torch.zeros_like(audio),
        reference_latent=reference,
        reference_motion=motion,
        control_video=control,
    )
    prepared = runtime.prepare_conditioning(positive)
    prepared_negative = runtime.prepare_conditioning(negative)

    result = _sample(
        runtime,
        torch.zeros((2, 16, 2, 2, 2)),
        prepared,
        cfg=SamplingGuidance(prepared_negative, 2.0, batching=FUSE_CFG_LANES),
        steps=1,
    )

    assert result.shape == (2, 16, 2, 2, 2)
    assert len(model.s2v_calls) == 1
    dispatched_audio, dispatched_reference, dispatched_motion, dispatched_control = model.s2v_calls[
        0
    ]
    assert dispatched_audio is not None and dispatched_audio.shape == (4, 25, 1024, 8)
    assert not torch.count_nonzero(dispatched_audio[:2])
    torch.testing.assert_close(dispatched_audio[2:], torch.ones((2, 25, 1024, 8)))
    assert dispatched_reference is not None and dispatched_reference.shape == (4, 16, 1, 2, 2)
    assert dispatched_motion is not None and dispatched_motion.shape == (4, 16, 3, 2, 2)
    assert dispatched_control is not None and dispatched_control.shape == (4, 16, 2, 2, 2)
    torch.testing.assert_close(dispatched_reference, torch.full_like(dispatched_reference, 22.0))
    torch.testing.assert_close(dispatched_motion, torch.full_like(dispatched_motion, 23.0))
    torch.testing.assert_close(dispatched_control, torch.full_like(dispatched_control, 24.0))
    assert vae.processed_in == 6


def test_humo_conditioning_builds_model_space_streams_and_preserves_cfg_lanes(
    monkeypatch: pytest.MonkeyPatch,
    family_id: str,
) -> None:
    runtime, model, vae, _codec = _runtime("humo", family_id=family_id)
    model.config = WAN21_HUMO_17B
    monkeypatch.setattr(wan21_runtime_module, "Wan21HumoModel", _Diffusion)
    text = runtime.text_conditioning_carrier(Conditioning(torch.ones((1, 2, 4096)), None))
    positive = compose_wan21_humo_conditioning(
        text,
        audio_embed=torch.ones((1, 3, 8, 5, 1280)),
        reference_latent=torch.full((1, 16, 2, 2, 2), 2.0),
    )
    negative = compose_wan21_humo_conditioning(
        text,
        audio_embed=torch.zeros((1, 3, 8, 5, 1280)),
        reference_latent=torch.full((1, 16, 2, 2, 2), 3.0),
    )
    prepared = runtime.prepare_conditioning(positive)
    prepared_negative = runtime.prepare_conditioning(negative)

    result = _sample(
        runtime,
        torch.zeros((2, 16, 3, 2, 2)),
        prepared,
        cfg=SamplingGuidance(prepared_negative, 2.0, batching=FUSE_CFG_LANES),
        steps=1,
    )

    assert result.shape == (2, 16, 3, 2, 2)
    assert len(model.humo_calls) == 1
    audio, reference = model.humo_calls[0]
    assert audio.shape == (4, 3, 8, 5, 1280)
    assert not torch.count_nonzero(audio[:2])
    torch.testing.assert_close(audio[2:], torch.ones_like(audio[2:]))
    assert reference.shape == (4, 36, 2, 2, 2)
    assert not torch.count_nonzero(reference[:, :16])
    torch.testing.assert_close(reference[:, 16:20], torch.ones_like(reference[:, 16:20]))
    torch.testing.assert_close(reference[:2, 20:], torch.full_like(reference[:2, 20:], 23.0))
    torch.testing.assert_close(reference[2:, 20:], torch.full_like(reference[2:, 20:], 22.0))
    model_input = model.calls[0][0]
    assert model_input.shape == (4, 36, 3, 2, 2)
    assert not torch.count_nonzero(model_input[:, 16:20])
    expected_first = torch.tensor(wan21_runtime_module._HUMO_ZERO_FIRST)  # pyright: ignore[reportPrivateUsage]
    expected_second = torch.tensor(wan21_runtime_module._HUMO_ZERO_SECOND)  # pyright: ignore[reportPrivateUsage]
    expected_later = torch.tensor(wan21_runtime_module._HUMO_ZERO_LATER)  # pyright: ignore[reportPrivateUsage]
    torch.testing.assert_close(model_input[0, 20:, 0, 0, 0], expected_first)
    torch.testing.assert_close(model_input[0, 20:, 1, 0, 0], expected_second)
    torch.testing.assert_close(model_input[0, 20:, 2, 0, 0], expected_later)
    assert vae.processed_in == 2


def test_dancer_conditioning_materializes_and_dispatches_batched_cfg(
    monkeypatch: pytest.MonkeyPatch,
    family_id: str,
) -> None:
    runtime, model, vae, _codec = _runtime("i2v", family_id=family_id)
    model.config = WAN22_WANDANCER_14B
    monkeypatch.setattr(wan21_runtime_module, "Wan22DancerModel", _Diffusion)
    text = runtime.text_conditioning_carrier(Conditioning(torch.ones((1, 2, 4096)), None))
    positive_concat = torch.full((1, 20, 2, 2, 2), 2.0)
    positive_concat[:, :4] = 0.25
    negative_concat = torch.full((1, 20, 2, 2, 2), 4.0)
    negative_concat[:, :4] = 0.5
    settings = Wan22DancerSettings(24.0, 0.75)
    positive = compose_wan22_dancer_conditioning(
        text,
        concat_latent=positive_concat,
        vision=torch.ones((1, 257, 1280)),
        reference_vision=torch.full((1, 257, 1280), 2.0),
        audio_embed=torch.full((1, 3, 35), 3.0),
        settings=settings,
    )
    negative = compose_wan22_dancer_conditioning(
        text,
        concat_latent=negative_concat,
        vision=torch.full((1, 257, 1280), 5.0),
        reference_vision=torch.full((1, 257, 1280), 6.0),
        audio_embed=torch.full((1, 3, 35), 7.0),
        settings=settings,
    )
    prepared = runtime.prepare_conditioning(positive)
    prepared_negative = runtime.prepare_conditioning(negative)

    assert prepared.dancer_settings == settings
    assert torch.equal(cast("torch.Tensor", prepared.concat_latent), positive_concat)
    assert torch.equal(cast("torch.Tensor", prepared.vision), torch.ones((1, 257, 1280)))
    assert torch.equal(
        cast("torch.Tensor", prepared.dancer_reference_vision),
        torch.full((1, 257, 1280), 2.0),
    )
    assert torch.equal(
        cast("torch.Tensor", prepared.dancer_audio_embed), torch.full((1, 3, 35), 3.0)
    )

    result = _sample(
        runtime,
        torch.zeros((2, 16, 2, 2, 2)),
        prepared,
        cfg=SamplingGuidance(prepared_negative, 2.0, batching=FUSE_CFG_LANES),
        steps=1,
    )

    assert result.shape == (2, 16, 2, 2, 2)
    assert len(model.dancer_calls) == 1
    vision, reference_vision, audio, fps, audio_scale = model.dancer_calls[0]
    assert vision is not None and vision.shape == (4, 257, 1280)
    assert reference_vision is not None and reference_vision.shape == (4, 257, 1280)
    assert audio is not None and audio.shape == (4, 3, 35)
    torch.testing.assert_close(vision[:2], torch.full_like(vision[:2], 5.0))
    torch.testing.assert_close(vision[2:], torch.ones_like(vision[2:]))
    torch.testing.assert_close(reference_vision[:2], torch.full_like(reference_vision[:2], 6.0))
    torch.testing.assert_close(reference_vision[2:], torch.full_like(reference_vision[2:], 2.0))
    torch.testing.assert_close(audio[:2], torch.full_like(audio[:2], 7.0))
    torch.testing.assert_close(audio[2:], torch.full_like(audio[2:], 3.0))
    assert (fps, audio_scale) == (24.0, 0.75)
    model_input = model.calls[0][0]
    assert model_input.shape == (4, 36, 2, 2, 2)
    torch.testing.assert_close(model_input[:2, 16:20], torch.full_like(model_input[:2, 16:20], 0.5))
    torch.testing.assert_close(
        model_input[2:, 16:20], torch.full_like(model_input[2:, 16:20], 0.25)
    )
    torch.testing.assert_close(model_input[:2, 20:], torch.full_like(model_input[:2, 20:], 24.0))
    torch.testing.assert_close(model_input[2:, 20:], torch.full_like(model_input[2:, 20:], 22.0))
    assert vae.processed_in == 2


def test_dancer_uses_zero_concat_and_default_settings_without_optional_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, model, _vae, _codec = _runtime("i2v")
    model.config = WAN22_WANDANCER_14B
    monkeypatch.setattr(wan21_runtime_module, "Wan22DancerModel", _Diffusion)
    text = basic_conditioning_to_carrier(Conditioning(torch.ones((1, 2, 4096)), None))
    prepared = runtime.prepare_conditioning(compose_wan22_dancer_conditioning(text))

    _sample(runtime, torch.zeros((2, 16, 2, 2, 2)), prepared, steps=1)

    model_input = model.calls[0][0]
    assert model_input.shape == (2, 36, 2, 2, 2)
    assert not torch.count_nonzero(model_input[:, 16:])
    assert model.dancer_calls == [(None, None, None, 30.0, 1.0)]


def test_dancer_conditioning_refuses_mutated_settings_and_channels() -> None:
    runtime, model, _vae, _codec = _runtime("i2v")
    model.config = WAN22_WANDANCER_14B
    text = basic_conditioning_to_carrier(Conditioning(torch.ones((1, 2, 4096)), None))
    carrier = compose_wan22_dancer_conditioning(text)
    record = carrier.conditioning.records[0]
    invalid_settings = make_conditioning_carrier(
        ConditioningSet(
            (
                replace(
                    record,
                    extension_metadata=((WAN22_DANCER_SETTINGS_KEY, {"fps": 30.0}),),
                ),
            )
        ),
        carrier.bindings,
    )
    invalid_channels = make_conditioning_carrier(
        ConditioningSet(
            (
                replace(
                    record,
                    channels=(
                        *record.channels,
                        (ConditioningChannel.REFERENCE_LATENT, record.channels[0][1]),
                    ),
                ),
            )
        ),
        carrier.bindings,
    )

    with pytest.raises(Wan21RuntimeError, match="settings metadata is invalid"):
        runtime.prepare_conditioning(invalid_settings)
    with pytest.raises(Wan21RuntimeError, match="does not consume.*reference_latent"):
        runtime.prepare_conditioning(invalid_channels)


def test_infinite_talk_replaces_motion_for_cfg_and_restores_continuation_overlap(
    monkeypatch: pytest.MonkeyPatch,
    family_id: str,
) -> None:
    runtime, model, _, _ = _runtime("i2v", family_id=family_id)
    model.config = WAN21_I2V_14B
    monkeypatch.setattr(wan21_runtime_module, "Wan21Model", _Diffusion)
    patch = Wan21MultiTalk.__new__(Wan21MultiTalk)
    torch.nn.Module.__init__(patch)
    patch.config = WAN21_MULTITALK
    model_digest = wan21_multitalk_resource_digest("blake3:" + "7" * 64, torch.float32)
    wan21_multitalk_module._bind_wan21_multitalk_resource(  # pyright: ignore[reportPrivateUsage]
        patch, model_digest
    )
    audio = torch.full((1, 3, 32, 768), 5.0)
    patch_execution = Wan21MultiTalkExecution(
        patch,
        audio,
        None,
        0.75,
        model_digest,
        wan21_multitalk_tensor_digest(audio),
        None,
    )
    motion = torch.full((1, 16, 1, 2, 2), 3.0)
    execution = Wan21InfiniteTalkExecution(
        patch_execution,
        motion,
        True,
        wan21_multitalk_tensor_digest(motion),
    )
    text = runtime.prepare_text_conditioning(Conditioning(torch.ones((1, 2, 4096)), None))
    concat = torch.zeros((1, 20, 3, 2, 2))
    prepared = runtime.prepare_i2v_conditioning(text, concat)

    result = _sample(
        runtime,
        torch.zeros((1, 16, 3, 2, 2)),
        prepared,
        cfg=SamplingGuidance(prepared, 2.0, batching=FUSE_CFG_LANES),
        steps=1,
        multitalk=execution,
    )

    assert model.calls
    for model_input, _, _, _ in model.calls:
        assert model_input.shape == (2, 36, 3, 2, 2)
        assert torch.equal(model_input[:, :16, :1], torch.full((2, 16, 1, 2, 2), 23.0))
    assert len(model.multitalk_calls) == len(model.calls)
    for admitted in model.multitalk_calls:
        assert type(admitted) is Wan21MultiTalkExecution
        assert admitted.model is patch
        assert admitted.audio_context.shape == (1, 3, 32, 768)
        assert admitted.strength == 0.75
    assert torch.equal(result[:, :, :1], motion)


def test_humo_requires_one_audio_window_per_target_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, model, _vae, _codec = _runtime("humo")
    model.config = WAN21_HUMO_17B
    monkeypatch.setattr(wan21_runtime_module, "Wan21HumoModel", _Diffusion)
    prepared = Wan21PreparedConditioning(
        torch.ones((1, 2, 4096)),
        humo_audio_embed=torch.zeros((1, 4, 8, 5, 1280)),
        humo_reference_latent=torch.zeros((1, 16, 1, 2, 2)),
    )

    with pytest.raises(Wan21RuntimeError, match="one window per target latent frame"):
        _sample(runtime, torch.zeros((1, 16, 3, 2, 2)), prepared, steps=1)


def test_non_s2v_runtime_refuses_manually_prepared_s2v_channels() -> None:
    runtime, _model, _vae, _codec = _runtime()
    prepared = Wan21PreparedConditioning(
        torch.ones((1, 2, 8)),
        s2v_control_video=torch.zeros((1, 16, 2, 2, 2)),
    )

    with pytest.raises(Wan21RuntimeError, match="contains S2V conditioning"):
        _sample(runtime, torch.zeros((1, 16, 2, 2, 2)), prepared, steps=1)


def test_causal_runtime_requires_matching_sampler_and_refuses_ordinary_sampling() -> None:
    ordinary, _, _, _ = _runtime()
    ar_request = _custom_request("ar_video", num_frame_per_block=2)
    with pytest.raises(Wan21RuntimeError, match="requires the Wan CausalAR profile"):
        ordinary.check_custom_sampling(
            ar_request,
            has_denoise_mask=False,
            has_inpaint=False,
            has_context_windows=False,
        )

    model = Wan21CausalModel.__new__(Wan21CausalModel)
    torch.nn.Module.__init__(model)
    model.config = WAN21_CAUSAL_AR_1_3B
    causal, _, _, _ = _runtime()
    cast("Any", causal).assembled.diffusion = model
    with pytest.raises(Wan21RuntimeError, match="requires the ar_video sampler"):
        causal.check_custom_sampling(
            _custom_request("euler"),
            has_denoise_mask=False,
            has_inpaint=False,
            has_context_windows=False,
        )
    with pytest.raises(Wan21RuntimeError, match="does not support masks"):
        causal.check_custom_sampling(
            ar_request,
            has_denoise_mask=True,
            has_inpaint=False,
            has_context_windows=False,
        )
    with pytest.raises(Wan21RuntimeError, match="distilled-guidance"):
        causal.check_custom_sampling(
            ar_request,
            has_denoise_mask=False,
            has_inpaint=False,
            has_context_windows=False,
            guidance=3.5,
        )
    with pytest.raises(Wan21RuntimeError, match="does not support context windows"):
        causal.check_custom_sampling(
            ar_request,
            has_denoise_mask=False,
            has_inpaint=False,
            has_context_windows=True,
        )
    with pytest.raises(Wan21RuntimeError, match="CFG exactly 1.0"):
        causal.sample_custom(
            torch.zeros((1, 16, 2, 2, 2)),
            noise=torch.zeros((1, 16, 2, 2, 2)),
            cond=Conditioning(torch.ones((1, 2, 8))),
            cfg=SamplingGuidance(Conditioning(torch.ones((1, 2, 8))), 2.0),
            request=ar_request,
        )
    with pytest.raises(Wan21RuntimeError, match="does not support guidance transforms"):
        causal.sample_custom(
            torch.zeros((1, 16, 2, 2, 2)),
            noise=torch.zeros((1, 16, 2, 2, 2)),
            cond=Conditioning(torch.ones((1, 2, 8))),
            cfg=SamplingGuidance(
                Conditioning(torch.ones((1, 2, 8))),
                1.0,
                transforms=(("dinkster.epsilon_scaling:0", epsilon_scaling(1.005)),),
            ),
            request=ar_request,
        )
    with pytest.raises(Wan21RuntimeError, match="initial latent must match"):
        causal.sample_custom(
            torch.zeros((1, 16, 2, 2, 2)),
            noise=torch.zeros((1, 16, 2, 2, 2)),
            cond=Conditioning(torch.ones((1, 2, WAN21_CAUSAL_AR_1_3B.text_dim))),
            cfg=None,
            request=ar_request,
            initial_latent=torch.zeros((1, 16, 1, 3, 2)),
        )
    with pytest.raises(Wan21RuntimeError, match="requires the ar_video sampler"):
        causal.sample_multistream(
            MultiStreamLatent.from_pairs((("video", torch.zeros((1, 16, 2, 2, 2))),)),
            conditioning=object(),
            sampler_id="euler",
            scheduler_id="simple",
            steps=2,
            denoise=1.0,
            seed=0,
        )


def test_wan_runtimes_expose_their_custom_schedule_contracts(family_id: str) -> None:
    ordinary, _, _, _ = _runtime(family_id=family_id)
    scheduler = torch_scheduler_registry().get("dinkster.simple")
    assert scheduler is not None
    ordinary_expected = sampling_sigmas(scheduler, WAN21_SIGMAS, 2, denoise=1.0)
    assert ordinary.custom_sampling_sigmas("dinkster.simple", 2, 1.0) == ordinary_expected
    shifted_expected = sampling_sigmas(scheduler, FlowSigmas(shift=3.0), 2, denoise=1.0)
    assert (
        ordinary.custom_sampling_sigmas("dinkster.simple", 2, 1.0, sampling_shift=3.0)
        == shifted_expected
    )

    model = Wan21CausalModel.__new__(Wan21CausalModel)
    torch.nn.Module.__init__(model)
    causal, _, _, _ = _runtime(family_id=family_id)
    cast("Any", causal).assembled.diffusion = model
    assert isinstance(causal, CustomSamplingRuntime)
    expected = sampling_sigmas(scheduler, FlowSigmas(shift=5.0), 4, denoise=0.5)
    assert causal.custom_sampling_sigmas("dinkster.simple", 4, 0.5) == expected
    assert (
        causal.custom_sampling_percent_to_sigma(0.0, return_actual_sigma=True)
        == FlowSigmas(shift=5.0).sigma_max
    )
    with pytest.raises(Wan21RuntimeError, match="fixed at 5.0"):
        causal.custom_sampling_sigmas("dinkster.simple", 2, 1.0, sampling_shift=3.0)

    diffusion = Wan21CausalDiffusionRuntime(
        model,
        causal.family,
        runtime_identity="native:dinkster.wan21:" + "1" * 64,
    )
    assert isinstance(diffusion, CustomSamplingRuntime)
    assert diffusion.custom_sampling_sigmas("dinkster.simple", 4, 0.5) == expected


@pytest.mark.parametrize(
    "model_class, config",
    (
        (wan21_runtime_module.Wan21Model, WAN21_T2V_14B),
        (wan21_runtime_module.Wan21HumoModel, WAN21_HUMO_17B),
        (wan21_runtime_module.Wan22S2VModel, WAN22_S2V_14B),
        (wan21_runtime_module.Wan21Model, WAN22_TI2V_5B),
        (wan21_runtime_module.Wan22DancerModel, WAN22_WANDANCER_14B),
    ),
)
def test_diffusion_runtime_admits_config_independently_of_family_label(
    model_class: Any, config: Wan21Config, family_id: str
) -> None:
    model = model_class.__new__(model_class)
    torch.nn.Module.__init__(model)
    model.config = config
    family = replace(WAN21, id=family_id)
    runtime = Wan21DiffusionRuntime(model, family, runtime_identity="test:diffusion")
    assert runtime.family is family
    assert runtime.assembled.diffusion.config is config
    with pytest.raises(ValueError, match="identity must be nonempty"):
        Wan21DiffusionRuntime(model, family, runtime_identity="")
    model.config = WAN21_I2V_14B
    with pytest.raises(ValueError, match="requires exact"):
        Wan21DiffusionRuntime(model, family, runtime_identity="test:diffusion")
    with pytest.raises(ValueError, match="exact CausalAR model"):
        Wan21CausalDiffusionRuntime(model, family, runtime_identity="test:causal")


@pytest.mark.parametrize("entry", ("custom", "ksampler"))
@pytest.mark.parametrize("masked", (False, True))
@pytest.mark.parametrize("compute_dtype", (None, torch.float32))
def test_diffusion_only_runtime_executes_the_standard_sampling_engine(
    monkeypatch: pytest.MonkeyPatch, entry: str, masked: bool, compute_dtype: torch.dtype | None
) -> None:
    from test_wan21_model import _model  # pyright: ignore[reportPrivateUsage]

    model = _model("t2v_reduced")
    monkeypatch.setattr(wan21_runtime_module, "WAN21_T2V_14B", model.config)
    runtime = Wan21DiffusionRuntime(model, WAN21, runtime_identity="test:diffusion")
    reference = object.__new__(Wan21Runtime)
    for name, value in runtime.__dict__.items():
        setattr(reference, name, value)
    cond = runtime.prepare_text_conditioning(Conditioning(torch.ones(1, 2, model.config.text_dim)))
    latent = MultiStreamLatent.from_pairs((("video", torch.zeros(1, 16, 1, 2, 2)),))
    mask = torch.tensor([0.0, 1.0]).expand(1, 16, 1, 2, 2) if masked else None

    def sample(active: Any, selected_dtype: torch.dtype | None) -> torch.Tensor:
        dtype_kwargs = {} if selected_dtype is None else {"compute_dtype": selected_dtype}
        if entry == "ksampler":
            return active.sample_multistream(
                latent,
                conditioning=cond,
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.simple",
                steps=2,
                denoise=1.0,
                seed=123,
                denoise_mask=mask,
                **dtype_kwargs,
            ).by_role("video")
        sampler = torch_sampler_registry().get("dinkster.euler")
        assert sampler is not None
        result = active.sample_custom(
            latent,
            noise=MultiStreamLatent.from_pairs((("video", torch.ones(1, 16, 1, 2, 2)),)),
            cond=PreparedMultiStreamConditioning(runtime.conditioning_identity, cond),
            cfg=None,
            request=CustomSamplingRequest(sampler, (), (1.0, 0.5, 0.0)),
            denoise_mask=mask,
            **dtype_kwargs,
        )
        return result.output.by_role("video")

    assert torch.equal(sample(runtime, compute_dtype), sample(reference, torch.float32))


@pytest.mark.parametrize("invalid_id", ("", " ", None, 17))
def test_runtime_family_identity_must_remain_well_formed(invalid_id: object) -> None:
    family = replace(WAN21, id=cast("Any", invalid_id))
    model_class = wan21_runtime_module.Wan21Model
    model = model_class.__new__(model_class)
    torch.nn.Module.__init__(model)
    model.config = WAN21_T2V_14B
    causal = Wan21CausalModel.__new__(Wan21CausalModel)
    torch.nn.Module.__init__(causal)
    with pytest.raises(ValueError, match="family identity must be a nonempty string"):
        Wan21Runtime(cast("Any", SimpleNamespace(family=family)), runtime_identity="test:full")
    with pytest.raises(ValueError, match="family identity must be a nonempty string"):
        Wan21DiffusionRuntime(model, family, runtime_identity="test:diffusion")
    with pytest.raises(ValueError, match="family identity must be a nonempty string"):
        Wan21CausalDiffusionRuntime(causal, family, runtime_identity="test:causal")


@pytest.mark.parametrize("structural", (False, True))
def test_causal_runtime_processes_custom_denoised_output_with_wan_normalization(
    monkeypatch: pytest.MonkeyPatch,
    structural: bool,
) -> None:
    model = Wan21CausalModel.__new__(Wan21CausalModel)
    torch.nn.Module.__init__(model)
    model.config = WAN21_CAUSAL_AR_1_3B
    model.patch_embedding = torch.nn.Conv3d(16, 16, 1)
    runtime, _, _, _ = _runtime()
    raw = cast("Any", runtime)
    raw.assembled.diffusion = model
    raw.assembled.family = WAN21

    def process_out(value: torch.Tensor) -> torch.Tensor:
        return value + 100.0

    raw._latent_process_out = process_out
    latent = torch.zeros((1, 16, 2, 2, 2))

    def fake_run_sampler_engine(
        _denoiser: object, _solver: object, **kwargs: object
    ) -> torch.Tensor:
        report_state = cast("Any", kwargs["on_state"])
        process_out = cast("Any", kwargs["process_out"])
        unpack_state = cast("Any", kwargs["unpack_state"])
        if unpack_state is not None:
            current = unpack_state(latent)
            denoised = unpack_state(torch.full_like(latent, 4.0))
        else:
            current = process_out(latent)
            denoised = process_out(torch.full_like(latent, 4.0))
        report_state(
            SamplingStateEvent(
                step=0,
                total=1,
                sigma=1.0,
                phase="pre_update",
                current=current,
                denoised=denoised,
            )
        )
        return process_out(torch.full_like(latent, 7.0))

    monkeypatch.setattr(sampling_engine, "run_denoise", fake_run_sampler_engine)

    text = torch.ones((1, 2, WAN21_CAUSAL_AR_1_3B.text_dim))
    events: list[SamplingStateEvent[object]] = []
    result = runtime.sample_custom(
        MultiStreamLatent.from_pairs((("video", latent),)) if structural else latent,
        noise=(
            MultiStreamLatent.from_pairs((("video", torch.zeros_like(latent)),))
            if structural
            else torch.zeros_like(latent)
        ),
        cond=(
            PreparedMultiStreamConditioning(
                runtime.conditioning_identity, Wan21PreparedConditioning(text)
            )
            if structural
            else Conditioning(text)
        ),
        cfg=None,
        request=_custom_request("ar_video", num_frame_per_block=1),
        on_state=events.append,
    )

    output = result.output
    denoised = result.denoised_output
    assert len(events) == 1
    if structural:
        assert type(output) is MultiStreamLatent
        assert type(denoised) is MultiStreamLatent
        assert type(events[0].current) is MultiStreamLatent
        assert type(events[0].denoised) is MultiStreamLatent
        output, denoised = output.by_role("video"), denoised.by_role("video")
    assert torch.equal(cast("torch.Tensor", output), torch.full_like(latent, 107.0))
    assert result.denoised_output is not None
    assert torch.equal(cast("torch.Tensor", denoised), torch.full_like(latent, 104.0))


@pytest.mark.parametrize(
    ("noise_video", "error", "match"),
    (
        (
            torch.zeros((1, 16, 2, 2, 2), dtype=torch.int64),
            TypeError,
            "exact strided floating torch.Tensor",
        ),
        (
            torch.zeros((1, 16, 3, 2, 2)),
            Wan21RuntimeError,
            "noise must match the video latent shape",
        ),
    ),
)
def test_causal_runtime_refuses_invalid_structural_noise(
    noise_video: torch.Tensor,
    error: type[Exception],
    match: str,
) -> None:
    model = Wan21CausalModel.__new__(Wan21CausalModel)
    torch.nn.Module.__init__(model)
    model.config = WAN21_CAUSAL_AR_1_3B
    model.patch_embedding = torch.nn.Conv3d(16, 16, 1)
    runtime, _, _, _ = _runtime()
    raw = cast("Any", runtime)
    raw.assembled.diffusion = model
    raw.assembled.family = WAN21
    latent = torch.zeros((1, 16, 2, 2, 2))
    text = torch.ones((1, 2, WAN21_CAUSAL_AR_1_3B.text_dim))

    with pytest.raises(error, match=match):
        runtime.sample_custom(
            MultiStreamLatent.from_pairs((("video", latent),)),
            noise=MultiStreamLatent.from_pairs((("video", noise_video),)),
            cond=PreparedMultiStreamConditioning(
                runtime.conditioning_identity, Wan21PreparedConditioning(text)
            ),
            cfg=None,
            request=_custom_request("ar_video", num_frame_per_block=1),
        )


def test_causal_runtime_checks_cancellation_during_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Wan21CausalModel.__new__(Wan21CausalModel)
    torch.nn.Module.__init__(model)
    model.config = WAN21_CAUSAL_AR_1_3B
    model.patch_embedding = torch.nn.Conv3d(16, 16, 1)
    runtime, _, _, _ = _runtime()
    raw = cast("Any", runtime)
    raw.assembled.diffusion = model
    raw.assembled.family = WAN21
    latent = torch.zeros((1, 16, 2, 2, 2))
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return checks > 1

    def fake_run_sampler_engine(
        _denoiser: object, _solver: object, **kwargs: object
    ) -> torch.Tensor:
        report_step = cast("Any", kwargs["on_step"])
        report_step(StepEvent(0, 1, 1.0))
        raise AssertionError("cancellation must stop the sampler step")

    monkeypatch.setattr(sampling_engine, "run_denoise", fake_run_sampler_engine)

    with pytest.raises(SamplingCancelled):
        runtime.sample_custom(
            latent,
            noise=torch.zeros_like(latent),
            cond=Conditioning(torch.ones((1, 2, WAN21_CAUSAL_AR_1_3B.text_dim))),
            cfg=None,
            request=_custom_request("ar_video", num_frame_per_block=1),
            cancelled=cancelled,
        )
    assert checks == 2


def test_encode_text_uses_wan_owned_tokenizer_and_umt5_encoder() -> None:
    runtime, _, _, _ = _runtime()

    result = runtime.encode_text("a fox running")
    latent = _video()
    sampled = _sample(runtime, latent, result, denoise=0.0)

    raw = cast("Any", runtime)
    assert raw._tokenizer.texts == ["a fox running"]
    assert raw._text_encoder.spans == [("a fox running",)]
    assert result.pooled is None
    assert result.embeddings.shape == (1, 4, 8)
    assert sampled is latent


def test_encode_vision_forwards_the_crop_mode() -> None:
    runtime, _, _, _ = _runtime("i2v")

    class VisionEncoder:
        def __init__(self) -> None:
            self.crop_modes: list[bool] = []

        def __call__(self, image: torch.Tensor, *, crop: bool = True) -> torch.Tensor:
            self.crop_modes.append(crop)
            return image

    encoder = VisionEncoder()
    cast("Any", runtime.assembled).clip_vision = encoder
    image = torch.zeros((1, 8, 12, 3))

    runtime.encode_vision(image)
    runtime.encode_vision(image, crop=False)

    assert encoder.crop_modes == [True, False]


def test_conditioning_identity_tracks_profile_not_runtime_recipe() -> None:
    first, _, _, _ = _runtime("t2v")
    second, _, _, _ = _runtime("t2v")
    cast("Any", second)._runtime_identity = "test:wan21:patched"
    image, _, _, _ = _runtime("i2v")
    flf, _, _, _ = _runtime("flf")

    assert first.conditioning_identity == second.conditioning_identity
    assert first.conditioning_identity != image.conditioning_identity
    assert image.conditioning_identity != flf.conditioning_identity


def test_runtime_satisfies_the_conditioning_preparation_protocol() -> None:
    runtime, _, _, _ = _runtime()

    assert isinstance(runtime, MultiStreamConditioningRuntime)


@pytest.mark.parametrize("config", (WAN22_TI2V_5B, WAN22_FUN_CONTROL_5B, WAN22_FUN_INPAINT_5B))
@pytest.mark.parametrize("family_id", ("dinkster.wan22", "dinkster.wan21", "example.renamed-video"))
def test_wan22_codec_center_crops_direct_and_tiled_encode(
    monkeypatch: pytest.MonkeyPatch,
    config: Wan21Config,
    family_id: str,
) -> None:
    from dinkster_inference_torch import wan21_runtime

    class Codec:
        def __init__(self) -> None:
            self.encoded: list[torch.Tensor] = []

        def encode(self, content: torch.Tensor) -> torch.Tensor:
            self.encoded.append(content.detach().clone())
            return torch.zeros(
                (
                    content.shape[0],
                    48,
                    1 + (content.shape[2] - 1) // 4,
                    content.shape[3] // 16,
                    content.shape[4] // 16,
                )
            )

        def decode(self, _latent: torch.Tensor) -> torch.Tensor:
            raise AssertionError("encode test must not decode")

        def process_in(self, latent: torch.Tensor) -> torch.Tensor:
            return latent

        def process_out(self, latent: torch.Tensor) -> torch.Tensor:
            return latent

    codec = Codec()
    diffusion = _Diffusion("ti2v")
    diffusion.config = config

    def compute_dtype(_component: object) -> torch.dtype:
        return torch.float32

    def tokenizer(_model: object) -> SimpleNamespace:
        def encode(_word: str) -> tuple[int, ...]:
            return (1,)

        return SimpleNamespace(encode=encode)

    def text_encoder(_model: object) -> _TextEncoder:
        return _TextEncoder()

    assembled = SimpleNamespace(
        family=replace(WAN22, id=family_id),
        diffusion=diffusion,
        vae=codec,
        clip_vision=None,
        tokenizer_model=object(),
        umt5xxl=object(),
        compute_dtype=compute_dtype,
    )
    monkeypatch.setattr(wan21_runtime, "Umt5SentencePieceTokenizer", tokenizer)
    monkeypatch.setattr(wan21_runtime, "T5TextEncoder", text_encoder)
    runtime = Wan21Runtime(cast("Any", assembled), runtime_identity="test:wan22")
    assert runtime.codec.descriptor is WAN22_CODEC
    assert runtime.sampling_sigma_space() is WAN22_SIGMAS
    content = torch.arange(3 * 5 * 19 * 34, dtype=torch.float32).reshape(1, 3, 5, 19, 34)
    expected = content[:, :, :, 1:17, 1:33] * 2.0 - 1.0

    direct = runtime.encode_content(content)
    tiled = runtime.codec.encode_tiled(content, output_device="cpu", dtype=torch.float32)

    assert direct.shape == tiled.shape == (1, 48, 2, 1, 2)
    assert len(codec.encoded) == 2
    torch.testing.assert_close(codec.encoded[0], expected)
    torch.testing.assert_close(codec.encoded[1], expected)

    with pytest.raises(ValueError, match="smaller than one 16x"):
        runtime.encode_content(torch.zeros((1, 3, 1, 15, 32)))


def test_runtime_executes_cfg_on_shift8_flow_with_video_stream_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, diffusion, vae, _ = _runtime()
    latent = _video()
    positive = Conditioning(torch.ones((1, 3, 8)), None)
    negative = Conditioning(torch.full((1, 3, 8), 3.0), None)
    planned: list[tuple[int, int, int]] = []
    states: list[SamplingStateEvent[object]] = []
    from dinkster_inference_torch import wan21_runtime

    real_plan = wan21_runtime.plan_wan21_token_layout

    def plan(geometry: object):
        raw_geometry = cast("Any", geometry)
        planned.append((raw_geometry.temporal, raw_geometry.height, raw_geometry.width))
        return real_plan(raw_geometry)

    monkeypatch.setattr(wan21_runtime, "plan_wan21_token_layout", plan)

    def capture_state(event: SamplingStateEvent[object]) -> None:
        states.append(event)
        assert type(event.current) is MultiStreamLatent
        event.current.by_role("video").fill_(99)

    result = _sample(
        runtime,
        latent,
        positive,
        cfg=SamplingGuidance(negative, 2.0, batching=FUSE_CFG_LANES),
        on_state=capture_state,
    )

    assert runtime.family.id == "dinkster.wan21"
    expected = prepare_noise(latent, 123) + 11.0
    torch.testing.assert_close(result, expected, rtol=0, atol=1e-6)
    assert planned == [(1, 2, 2)]
    assert len(diffusion.calls) == 2
    assert [call[0].shape for call in diffusion.calls] == [(2, 16, 1, 2, 2)] * 2
    assert [call[1][0].item() for call in diffusion.calls] == pytest.approx(
        [1000.0, 888.8888888888888]
    )
    assert [sorted(call[2].mean(dim=(1, 2)).tolist()) for call in diffusion.calls] == [
        [1.0, 3.0],
        [1.0, 3.0],
    ]
    assert all(call[3] is None for call in diffusion.calls)
    assert vae.processed_in == 0
    assert vae.processed_out == 1
    assert states
    assert all(type(event.current) is MultiStreamLatent for event in states)


@pytest.mark.parametrize(
    "model_type, descriptor", (("t2v", WAN21_CODEC), ("flow_rvs", WAN21_FLOW_RVS_CODEC))
)
@pytest.mark.parametrize("family_id", ("dinkster.wan21", "dinkster.wan22", "example.renamed-video"))
def test_runtime_selects_rgb_or_mask_codec_from_config(
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
    descriptor: CodecDescriptor,
    family_id: str,
) -> None:
    from dinkster_inference_torch import wan21_runtime

    class Codec(_Codec):
        def process_in(self, latent: torch.Tensor) -> torch.Tensor:
            return latent

        def process_out(self, latent: torch.Tensor) -> torch.Tensor:
            return latent

    def tokenizer(_model: object) -> SimpleNamespace:
        def encode(_word: str) -> tuple[int]:
            return (1,)

        return SimpleNamespace(encode=encode)

    def compute_dtype(_component: str) -> torch.dtype:
        return torch.float32

    def text_encoder(_model: object) -> _TextEncoder:
        return _TextEncoder()

    codec = Codec()
    diffusion = _Diffusion(model_type)
    assembled = SimpleNamespace(
        family=replace(WAN21, id=family_id),
        diffusion=diffusion,
        vae=codec,
        clip_vision=None,
        tokenizer_model=object(),
        umt5xxl=object(),
        compute_dtype=compute_dtype,
    )
    monkeypatch.setattr(wan21_runtime, "Umt5SentencePieceTokenizer", tokenizer)
    monkeypatch.setattr(wan21_runtime, "T5TextEncoder", text_encoder)

    runtime = Wan21Runtime(cast("Any", assembled), runtime_identity="test:flow-rvs")

    assert runtime.codec.descriptor is descriptor
    assert runtime.sampling_sigma_space() is WAN21_SIGMAS


def test_flow_rvs_sampling_uses_raw_output_ignores_noise_and_inverts_the_result() -> None:
    runtime, diffusion, vae, _ = _runtime("flow_rvs")
    latent = torch.full((1, 16, 1, 2, 2), 2.0)
    conditioning = Conditioning(torch.ones((1, 3, 8)), None)

    first = _sample(runtime, latent, conditioning, seed=123)
    first_calls = tuple(diffusion.calls)
    diffusion.calls.clear()
    second = _sample(runtime, latent, conditioning, seed=999)

    assert torch.equal(first, second)
    assert torch.equal(first, torch.full_like(latent, 10.0))
    assert len(first_calls) == len(diffusion.calls) == 2
    assert torch.equal(first_calls[0][0], torch.full_like(latent, 22.0))
    assert torch.equal(diffusion.calls[0][0], first_calls[0][0])
    assert vae.processed_in == 2
    assert vae.processed_out == 2


def test_wan22_ti2v_uses_shift8_and_frame_mask_timesteps() -> None:
    runtime, diffusion, vae, _ = _runtime("ti2v")
    latent = torch.zeros((1, 48, 3, 2, 2))
    latent[:, :, 0] = 2.0
    mask = torch.ones((1, 1, 3, 2, 2))
    mask[:, :, 0] = 0.0

    result = _sample(
        runtime,
        latent,
        Conditioning(torch.ones((1, 3, 8)), None),
        denoise_mask=mask,
    )

    assert runtime.family.id == "dinkster.wan22"
    assert result.shape == latent.shape
    assert len(diffusion.calls) == 2
    assert all(call[0].shape == (1, 48, 3, 2, 2) for call in diffusion.calls)
    assert all(call[1].shape == (1, 3) for call in diffusion.calls)
    assert all(call[1][0, 0].item() == 0.0 for call in diffusion.calls)
    assert all(
        torch.equal(call[0][:, :, 0], torch.full_like(call[0][:, :, 0], 22.0))
        for call in diffusion.calls
    )
    assert diffusion.calls[0][1][0, 1:].tolist() == pytest.approx([1000.0, 1000.0])
    assert vae.processed_in == 1
    assert vae.processed_out == 1


def test_wan22_ti2v_preserves_per_channel_denoise_masks() -> None:
    runtime, _, _, _ = _runtime("ti2v")
    latent = torch.zeros((1, 48, 1, 2, 2))
    mask = torch.ones_like(latent)
    mask[:, :24] = 0.0

    result = _sample(
        runtime,
        latent,
        Conditioning(torch.ones((1, 3, 8)), None),
        denoise_mask=mask,
    )

    assert torch.equal(result[:, :24], torch.full_like(result[:, :24], 10.0))
    assert not torch.equal(result[:, 24:], torch.full_like(result[:, 24:], 10.0))


def test_wan22_ti2v_empty_latent_skips_normalization_without_a_frame_mask() -> None:
    runtime, diffusion, vae, _ = _runtime("ti2v")
    latent = torch.zeros((1, 48, 2, 2, 2))

    result = _sample(
        runtime,
        latent,
        Conditioning(torch.ones((1, 3, 8)), None),
    )

    assert result.shape == latent.shape
    assert len(diffusion.calls) == 2
    assert all(call[1].shape == (1,) for call in diffusion.calls)
    assert vae.processed_in == 0
    assert vae.processed_out == 1


def test_wan22_14b_high_low_noise_models_compose_at_step_ten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    high, high_diffusion, high_vae, _ = _runtime()
    low, low_diffusion, low_vae, _ = _runtime()
    latent = _video()
    positive = Conditioning(torch.ones((1, 3, 8)), None)
    negative = Conditioning(torch.full((1, 3, 8), 3.0), None)
    cfg = SamplingGuidance(cast("object", negative), 3.5, batching=FUSE_CFG_LANES)
    noise_inputs: list[torch.Tensor] = []
    from dinkster_inference_torch import wan21_runtime

    real_prepare_noise = wan21_runtime.prepare_noise

    def capture_noise(value: torch.Tensor, seed: int) -> torch.Tensor:
        noise_inputs.append(value)
        return real_prepare_noise(value, seed)

    monkeypatch.setattr(wan21_runtime, "prepare_noise", capture_noise)

    high_output = _sample(
        high,
        latent,
        positive,
        cfg=cfg,
        steps=20,
        segment=SamplingSegment(20, 0, 10, True, True),
        sampling_shift=5.0,
    )
    output = _sample(
        low,
        high_output,
        positive,
        cfg=cfg,
        steps=20,
        segment=SamplingSegment(20, 10, 20, False, False),
        sampling_shift=5.0,
    )

    assert output.shape == latent.shape
    assert noise_inputs == [latent]
    assert len(high_diffusion.calls) == 10
    assert len(low_diffusion.calls) == 10
    assert high_diffusion.calls[0][1][0].item() == pytest.approx(1000.0)
    assert high_diffusion.calls[1][1][0].item() == pytest.approx(989.5833333333334)
    assert low_diffusion.calls[0][1][0].item() == pytest.approx(833.3333333333334)
    assert high_vae.processed_in == 0
    assert high_vae.processed_out == 1
    assert low_vae.processed_in == 1
    assert low_vae.processed_out == 1


@pytest.mark.parametrize(
    ("latent", "error", "message"),
    [
        (object(), TypeError, "exact strided floating torch.Tensor"),
        (
            torch.sparse_coo_tensor(
                torch.empty((5, 0), dtype=torch.int64),
                [],
                (1, 16, 1, 2, 2),
                check_invariants=True,
            ),
            TypeError,
            "exact strided floating torch.Tensor",
        ),
        (_video((1, 15, 1, 2, 2)), Wan21RuntimeError, r"\[B,16,T,H,W\]"),
        (_video((1, 16, 2, 2)), Wan21RuntimeError, r"\[B,16,T,H,W\]"),
    ],
)
def test_runtime_refuses_non_wan_video_latents(
    latent: object, error: type[Exception], message: str
) -> None:
    runtime, _, _, _ = _runtime()
    with pytest.raises(error, match=message):
        _sample(runtime, latent, Conditioning(torch.ones((1, 2, 8)), None))


def test_runtime_requires_exact_video_stream_role() -> None:
    runtime, _, _, _ = _runtime()
    conditioning = runtime.prepare_text_conditioning(Conditioning(torch.ones((1, 2, 8)), None))
    with pytest.raises(Wan21RuntimeError, match="exact latent stream role 'video'"):
        _sample(
            runtime,
            MultiStreamLatent.from_pairs((("image", _video()),)),
            conditioning,
            denoise=0.0,
        )


@pytest.mark.parametrize(
    ("conditioning", "error", "message"),
    [
        ({"text": torch.ones((1, 2, 8))}, TypeError, "exact Conditioning"),
        (
            Conditioning(torch.ones((1, 2, 8)), torch.ones((1, 8))),
            Wan21RuntimeError,
            "pooled",
        ),
        (Conditioning(torch.ones((1, 2, 7)), None), Wan21RuntimeError, r"\[B,tokens,8\]"),
        (Conditioning(torch.ones((1, 0, 8)), None), Wan21RuntimeError, "nonempty"),
        (
            Conditioning(
                torch.sparse_coo_tensor(
                    torch.empty((3, 0), dtype=torch.int64),
                    [],
                    (1, 2, 8),
                    check_invariants=True,
                ),
                None,
            ),
            TypeError,
            "exact strided floating torch.Tensor",
        ),
    ],
)
def test_runtime_prepares_only_valid_text_conditioning(
    conditioning: object, error: type[Exception], message: str
) -> None:
    runtime, _, _, _ = _runtime()
    with pytest.raises(error, match=message):
        runtime.prepare_text_conditioning(cast("Any", conditioning))


def test_text_conditioning_carrier_round_trips_through_prepare_conditioning() -> None:
    runtime, _, _, _ = _runtime()
    conditioning = Conditioning(torch.full((1, 2, 8), 0.5), None)
    carrier = runtime.text_conditioning_carrier(conditioning)
    prepared = runtime.prepare_conditioning(carrier)
    expected = runtime.prepare_text_conditioning(conditioning)
    torch.testing.assert_close(prepared.text, expected.text)
    assert prepared.concat_latent is None
    assert prepared.vision is None


def test_text_conditioning_carrier_refuses_invalid_text() -> None:
    runtime, _, _, _ = _runtime()
    with pytest.raises(Wan21RuntimeError, match="pooled"):
        runtime.text_conditioning_carrier(Conditioning(torch.ones((1, 2, 8)), torch.ones((1, 8))))
    with pytest.raises(Wan21RuntimeError, match=r"\[B,tokens,8\]"):
        runtime.text_conditioning_carrier(Conditioning(torch.ones((1, 2, 7)), None))


def test_runtime_accepts_masks_and_refuses_unknown_catalog_entries() -> None:
    runtime, _, vae, _ = _runtime()
    conditioning = Conditioning(torch.ones((1, 2, 8)), None)
    latent = torch.ones_like(_video())
    masked = _sample(runtime, latent, conditioning, denoise_mask=torch.zeros((1, 1, 1, 2, 2)))
    assert torch.equal(masked, vae.process_out(vae.process_in(latent)))
    with pytest.raises(Wan21RuntimeError, match="does not accept per-batch noise indices"):
        _sample(runtime, _video(), conditioning, noise_inds=(0,))
    with pytest.raises(Wan21RuntimeError, match="unknown sampler"):
        _sample(runtime, _video(), conditioning, sampler_id="not-a-sampler")
    with pytest.raises(Wan21RuntimeError, match="unknown scheduler"):
        _sample(runtime, _video(), conditioning, scheduler_id="not-a-scheduler")


def test_runtime_honors_cancellation_before_sampling() -> None:
    runtime, _, _, _ = _runtime()
    with pytest.raises(SamplingCancelled):
        _sample(
            runtime,
            _video(),
            Conditioning(torch.ones((1, 2, 8)), None),
            cancelled=lambda: True,
        )


def test_family_adapter_refuses_channels_not_consumed_by_t2v() -> None:
    runtime, _, _, _ = _runtime()
    text = torch.ones((1, 2, 8))
    binding = tensor_to_payload_binding("concat", text, space="conditioning-text")
    descriptor = PayloadDescriptor(
        PayloadReference(binding.reference_id), binding.shape, binding.dtype, binding.space
    )
    carrier = make_conditioning_carrier(
        ConditioningSet(
            (
                ConditioningRecord(
                    channels=((ConditioningChannel.CONCAT_LATENT, descriptor),),
                    token_layout=TokenLayoutDescriptor(
                        "dinkster.wan21",
                        1,
                        ("umt5",),
                        (TokenSegmentDescriptor("umt5", "umt5", 0, 2),),
                    ),
                ),
            )
        ),
        (binding,),
    )

    with pytest.raises(Wan21RuntimeError, match="does not consume.*concat_latent"):
        runtime.prepare_conditioning(carrier)


@pytest.mark.parametrize(
    ("source_batch", "target_batch", "expected"),
    (
        (5, 3, [0.0, 2.0, 4.0]),
        (2, 5, [0.0, 0.0, 1.0, 1.0, 1.0]),
        (3, 1, [0.0]),
    ),
)
def test_i2v_concat_batch_resize_matches_comfyui_selection(
    source_batch: int, target_batch: int, expected: list[float]
) -> None:
    from dinkster_inference_torch import wan21_runtime

    source = torch.arange(source_batch, dtype=torch.float32).reshape(source_batch, 1, 1, 1, 1)

    resized = wan21_runtime._resize_concat_batch(  # pyright: ignore[reportPrivateUsage]
        source, target_batch
    )

    assert resized[:, 0, 0, 0, 0].tolist() == expected


def test_i2v_family_adapter_materializes_all_channels_and_builds_exact_model_input() -> None:
    runtime, diffusion, vae, _ = _runtime("i2v")
    text = runtime.prepare_text_conditioning(Conditioning(torch.ones((1, 2, 8)), None))
    concat = torch.zeros((1, 20, 1, 2, 2))
    concat[:, :4] = 1.0
    concat[:, 4:] = 2.0
    vision = torch.full((1, 257, 1280), 3.0)
    prepared = runtime.prepare_i2v_conditioning(text, concat, vision)

    result = _sample(
        runtime,
        _video(),
        prepared,
        cfg=SamplingGuidance(prepared, 2.0, batching=FUSE_CFG_LANES),
    )

    assert result.shape == (1, 16, 1, 2, 2)
    assert len(diffusion.calls) == 2
    for model_input, _, context, model_vision in diffusion.calls:
        assert model_input.shape == (2, 36, 1, 2, 2)
        torch.testing.assert_close(model_input[:, 16:20], torch.ones_like(model_input[:, 16:20]))
        torch.testing.assert_close(model_input[:, 20:], torch.full_like(model_input[:, 20:], 22.0))
        assert context.shape == (2, 2, 8)
        assert model_vision is not None
        assert model_vision.shape == (2, 257, 1280)
    assert vae.processed_in == 2
    assert vae.processed_out == 1


@pytest.mark.parametrize("with_vision", (False, True))
def test_i2v_adapter_allows_optional_vision_and_normalizes_conditioning(
    with_vision: bool,
) -> None:
    runtime, diffusion, vae, _ = _runtime("i2v")
    text = runtime.prepare_text_conditioning(Conditioning(torch.ones((1, 2, 8)), None))
    concat = torch.zeros((1, 20, 1, 2, 2))
    concat[:, :4] = 1.0
    concat[:, 4:] = 2.0
    vision = torch.full((1, 257, 1280), 3.0) if with_vision else None
    prepared = runtime.prepare_i2v_conditioning(text, concat, vision)

    _sample(runtime, _video(), prepared)

    assert prepared.concat_mask_index == 0
    assert diffusion.calls
    model_input, _, _, model_vision = diffusion.calls[0]
    assert model_input.shape == (1, 36, 1, 2, 2)
    torch.testing.assert_close(model_input[:, 16:20], torch.ones_like(model_input[:, 16:20]))
    torch.testing.assert_close(model_input[:, 20:], torch.full_like(model_input[:, 20:], 22.0))
    assert (model_vision is not None) is with_vision
    assert vae.processed_in == 1


def test_wan22_i2v_builds_reference_input_without_vision_conditioning() -> None:
    runtime, diffusion, vae, _ = _runtime("i2v22")
    text = runtime.prepare_text_conditioning(Conditioning(torch.ones((1, 2, 8)), None))
    concat = torch.zeros((1, 20, 1, 2, 2))
    concat[:, :4] = 1.0
    concat[:, 4:] = 2.0
    prepared = runtime.prepare_i2v_conditioning(text, concat)

    result = _sample(runtime, _video(), prepared)

    assert result.shape == (1, 16, 1, 2, 2)
    assert len(diffusion.calls) == 2
    for model_input, _, context, model_vision in diffusion.calls:
        assert model_input.shape == (1, 36, 1, 2, 2)
        torch.testing.assert_close(model_input[:, 16:20], torch.ones_like(model_input[:, 16:20]))
        torch.testing.assert_close(model_input[:, 20:], torch.full_like(model_input[:, 20:], 22.0))
        assert context.shape == (1, 2, 8)
        assert model_vision is None
    assert vae.processed_in == 1


def test_animate_batches_shared_pose_and_distinct_positive_negative_face_inputs() -> None:
    runtime, diffusion, vae, _ = _runtime("animate")
    positive_text = runtime.prepare_text_conditioning(Conditioning(torch.ones((1, 2, 8)), None))
    negative_text = runtime.prepare_text_conditioning(Conditioning(torch.zeros((1, 2, 8)), None))
    concat = torch.zeros((1, 20, 2, 2, 2))
    concat[:, :4] = 1.0
    concat[:, 4:] = 2.0
    pose = torch.full((1, 16, 1, 2, 2), 4.0)
    positive_face = torch.full((1, 3, 4, 512, 512), 0.5)
    negative_face = torch.full_like(positive_face, -1.0)
    positive = runtime.prepare_animate_conditioning(
        positive_text,
        concat,
        pose_latents=pose,
        face_pixel_values=positive_face,
    )
    negative = runtime.prepare_animate_conditioning(
        negative_text,
        concat,
        pose_latents=pose,
        face_pixel_values=negative_face,
    )

    result = _sample(
        runtime,
        _video((1, 16, 2, 2, 2)),
        positive,
        cfg=SamplingGuidance(negative, 2.0, batching=FUSE_CFG_LANES),
    )

    assert result.shape == (1, 16, 2, 2, 2)
    assert diffusion.animate_calls
    model_pose, model_face = diffusion.animate_calls[0]
    assert model_pose is not None and model_face is not None
    assert model_pose.shape == (2, 16, 1, 2, 2)
    assert torch.equal(model_pose, torch.full_like(model_pose, 24.0))
    assert model_face.shape == (2, 3, 4, 512, 512)
    assert sorted(model_face[:, 0, 0, 0, 0].tolist()) == [-1.0, 0.5]
    assert vae.processed_in == 4


def test_animate_carrier_builder_materializes_all_optional_channels(family_id: str) -> None:
    runtime, _, _, _ = _runtime("animate", family_id=family_id)
    text = runtime.text_conditioning_carrier(Conditioning(torch.ones((1, 2, 8)), None))
    concat = torch.zeros((1, 20, 2, 2, 2))
    vision = torch.ones((1, 257, 1280))
    pose = torch.full((1, 16, 1, 2, 2), 2.0)
    face = torch.full((1, 3, 4, 512, 512), 0.5)

    carrier = compose_wan21_animate_conditioning(
        text,
        concat,
        vision=vision,
        pose_latents=pose,
        face_pixel_values=face,
    )
    record = carrier.conditioning.records[0]
    assert [channel for channel, _ in record.channels] == [
        ConditioningChannel.TEXT,
        ConditioningChannel.CONCAT_LATENT,
        ConditioningChannel.VISION_EMBEDDING,
        ConditioningChannel.POSE_LATENT,
        ConditioningChannel.FACE_PIXELS,
    ]
    assert [descriptor.space for _, descriptor in record.channels] == [
        "conditioning-text",
        "conditioning-concat-latent",
        "conditioning-vision-embedding",
        "conditioning-pose-latent",
        "conditioning-face-pixels",
    ]
    prepared = runtime.prepare_conditioning(carrier)
    assert torch.equal(prepared.text, torch.ones((1, 2, 8)))
    assert prepared.concat_latent is not None and torch.equal(prepared.concat_latent, concat)
    assert prepared.vision is not None and torch.equal(prepared.vision, vision)
    assert prepared.pose_latents is not None and torch.equal(prepared.pose_latents, pose)
    assert prepared.face_pixel_values is not None and torch.equal(prepared.face_pixel_values, face)
    assert prepared.concat_mask_index == 0


@pytest.mark.parametrize("animate", (False, True))
def test_renamed_text_layout_preserves_identity_width_and_stream_checks(animate: bool) -> None:
    runtime, _, _, _ = _runtime("animate" if animate else "t2v", family_id="example.renamed-video")
    text = runtime.text_conditioning_carrier(Conditioning(torch.ones((1, 2, 8))))

    def compose(carrier: object):
        if animate:
            return compose_wan21_animate_conditioning(
                cast("Any", carrier), torch.zeros((1, 20, 1, 2, 2))
            )
        return wan21_runtime_module._basic_wan_text_payload(  # pyright: ignore[reportPrivateUsage]
            carrier, name="test text"
        )

    record = text.conditioning.records[0]
    layout = record.token_layout
    assert layout is not None
    compose(text)
    for invalid_layout in (
        replace(layout, version=2),
        replace(
            layout,
            text_streams=("other",),
            segments=(TokenSegmentDescriptor("other", "other", 0, 2),),
        ),
    ):
        invalid = make_conditioning_carrier(
            ConditioningSet((replace(record, token_layout=invalid_layout),)), text.bindings
        )
        with pytest.raises(Wan21RuntimeError, match="incompatible token layout"):
            compose(invalid)

    other, _, _, _ = _runtime("animate" if animate else "t2v", family_id="example.other-video")
    carrier = compose(text) if animate else text
    with pytest.raises(Wan21RuntimeError, match="token layout is unsupported"):
        other.prepare_conditioning(cast("Any", carrier))
    with pytest.raises(Wan21RuntimeError, match="8"):
        runtime.text_conditioning_carrier(Conditioning(torch.ones((1, 2, 7))))


def test_animate_carrier_builder_keeps_optional_channels_and_other_profiles_refuse() -> None:
    text = basic_conditioning_to_carrier(Conditioning(torch.ones((1, 2, 8)), None))
    concat = torch.zeros((1, 20, 2, 2, 2))
    animate, _, _, _ = _runtime("animate")
    minimal = compose_wan21_animate_conditioning(text, concat)
    prepared = animate.prepare_conditioning(minimal)
    assert prepared.pose_latents is None
    assert prepared.face_pixel_values is None

    pose = torch.zeros((1, 16, 1, 2, 2))
    carrier = compose_wan21_animate_conditioning(text, concat, pose_latents=pose)
    base, _, _, _ = _runtime("i2v")
    with pytest.raises(Wan21RuntimeError, match="does not consume.*pose_latent"):
        base.prepare_conditioning(carrier)


def test_animate_refuses_generic_adapters_and_misaligned_pose_latents() -> None:
    runtime, _, _, _ = _runtime("animate")
    text = runtime.prepare_text_conditioning(Conditioning(torch.ones((1, 2, 8)), None))
    concat = torch.zeros((1, 20, 2, 2, 2))

    with pytest.raises(Wan21RuntimeError, match="typed native adapter"):
        runtime.prepare_i2v_conditioning(text, concat)
    with pytest.raises(Wan21RuntimeError, match="leave the leading reference frame"):
        _sample(
            runtime,
            _video((1, 16, 2, 2, 2)),
            runtime.prepare_animate_conditioning(
                text,
                concat,
                pose_latents=torch.zeros((1, 16, 2, 2, 2)),
            ),
        )

    base, _, _, _ = _runtime("i2v")
    base_text = base.prepare_text_conditioning(Conditioning(torch.ones((1, 2, 8)), None))
    with pytest.raises(Wan21RuntimeError, match="does not consume Animate"):
        base.prepare_animate_conditioning(base_text, concat)


def test_animate2_carrier_materializes_two_strict_records() -> None:
    runtime, _, _, _ = _runtime("animate2")
    text = basic_conditioning_to_carrier(Conditioning(torch.ones((1, 2, 8)), None))
    pose_text = basic_conditioning_to_carrier(Conditioning(torch.full((1, 3, 8), 2.0), None))
    concat = torch.zeros((1, 20, 3, 2, 2))
    vision = torch.ones((1, 257, 1280))
    pose_vision = torch.full((1, 257, 1280), 3.0)
    pose = torch.full((1, 16, 2, 2, 2), 4.0)
    schedule = PercentRange(0.2, 0.8)
    settings = Wan21Animate2Settings(0.625, 0.75)

    carrier = compose_wan21_animate2_conditioning(
        text,
        concat,
        vision=vision,
        pose_text=pose_text,
        pose_vision=pose_vision,
        pose_latents=pose,
        pose_schedule=schedule,
        settings=settings,
    )

    main, pose_record = carrier.conditioning.records
    assert [channel for channel, _ in main.channels] == [
        ConditioningChannel.TEXT,
        ConditioningChannel.CONCAT_LATENT,
        ConditioningChannel.VISION_EMBEDDING,
    ]
    assert [channel for channel, _ in pose_record.channels] == [
        ConditioningChannel.POSE_TEXT,
        ConditioningChannel.POSE_VISION_EMBEDDING,
        ConditioningChannel.POSE_LATENT,
    ]
    assert [descriptor.space for _, descriptor in pose_record.channels] == [
        "conditioning-pose-text",
        "conditioning-pose-vision-embedding",
        "conditioning-pose-latent",
    ]
    assert pose_record.schedule == schedule
    assert dict(pose_record.extension_metadata) == {
        WAN21_ANIMATE2_SETTINGS_KEY: {
            "pose_strength": 0.625,
            "reference_strength": 0.75,
        }
    }
    prepared = runtime.prepare_conditioning(carrier)
    assert torch.equal(prepared.text, torch.ones((1, 2, 8)))
    assert prepared.concat_latent is not None and torch.equal(prepared.concat_latent, concat)
    assert prepared.vision is not None and torch.equal(prepared.vision, vision)
    assert prepared.pose_text is not None and torch.equal(
        prepared.pose_text, torch.full((1, 3, 8), 2.0)
    )
    assert prepared.pose_vision is not None and torch.equal(prepared.pose_vision, pose_vision)
    assert prepared.pose_latents is not None and torch.equal(prepared.pose_latents, pose)
    assert prepared.pose_schedule == schedule
    assert prepared.animate2_settings == settings


@pytest.mark.parametrize(
    "metadata",
    (
        ((WAN21_ANIMATE2_SETTINGS_KEY, {"pose_strength": 1.0}),),
        (
            (
                WAN21_ANIMATE2_SETTINGS_KEY,
                {"pose_strength": 1.0, "reference_strength": 1.0, "other": 1.0},
            ),
        ),
        (("dinkster.wan21/other", {"pose_strength": 1.0, "reference_strength": 1.0}),),
    ),
)
def test_animate2_refuses_noncanonical_settings_metadata(
    metadata: tuple[tuple[str, object], ...],
) -> None:
    runtime, _, _, _ = _runtime("animate2")
    text = basic_conditioning_to_carrier(Conditioning(torch.ones((1, 2, 8)), None))
    carrier = compose_wan21_animate2_conditioning(
        text,
        torch.zeros((1, 20, 2, 2, 2)),
    )
    main, pose = carrier.conditioning.records
    invalid = make_conditioning_carrier(
        ConditioningSet((main, replace(pose, extension_metadata=metadata))),
        carrier.bindings,
    )

    with pytest.raises(Wan21RuntimeError, match="metadata"):
        runtime.prepare_conditioning(invalid)


def test_animate2_carrier_is_refused_by_other_wan_profiles() -> None:
    text = basic_conditioning_to_carrier(Conditioning(torch.ones((1, 2, 8)), None))
    carrier = compose_wan21_animate2_conditioning(
        text,
        torch.zeros((1, 20, 2, 2, 2)),
        pose_latents=torch.zeros((1, 16, 1, 2, 2)),
    )
    for profile in ("i2v", "animate"):
        runtime, _, _, _ = _runtime(profile)
        with pytest.raises(Wan21RuntimeError):
            runtime.prepare_conditioning(carrier)


def test_animate2_runtime_honors_pose_window_strengths_default_shift_and_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, diffusion, vae, _ = _runtime("animate2")
    text = basic_conditioning_to_carrier(Conditioning(torch.ones((1, 2, 8)), None))
    pose_text = basic_conditioning_to_carrier(Conditioning(torch.full((1, 3, 8), 2.0), None))
    carrier = compose_wan21_animate2_conditioning(
        text,
        torch.zeros((1, 20, 3, 2, 2)),
        vision=torch.ones((1, 257, 1280)),
        pose_text=pose_text,
        pose_vision=torch.full((1, 257, 1280), 3.0),
        pose_latents=torch.full((1, 16, 2, 2, 2), 4.0),
        pose_schedule=PercentRange(0.0, 0.4),
        settings=Wan21Animate2Settings(0.625, 0.75),
    )
    prepared = runtime.prepare_conditioning(carrier)
    raw = cast("Any", runtime)
    raw._pose_cache_settings = Wan21PoseBlockCacheSettings(
        Wan21PoseBlockCacheDevice.CPU,
        1234,
        Wan21PoseBlockCacheStorage.INT8,
    )
    captured_shifts: list[float] = []
    from dinkster_inference_torch import sampling_execution, wan21_runtime

    real_build_schedule = sampling_execution.build_sampling_schedule

    def capture_schedule(*args: Any, **kwargs: Any) -> object:
        captured_shifts.append(args[1].shift)
        return real_build_schedule(*args, **kwargs)

    class FakePoseCache:
        instances: list[FakePoseCache] = []

        def __init__(
            self,
            device: torch.device,
            storage: str,
            memory_limit_bytes: int | None,
        ) -> None:
            self.args = (device, storage, memory_limit_bytes)
            self.freed = False
            self.instances.append(self)

        def free(self) -> None:
            self.freed = True

    monkeypatch.setattr(sampling_execution, "build_sampling_schedule", capture_schedule)
    monkeypatch.setattr(wan21_runtime, "PoseBranchCache", FakePoseCache)

    result = _sample(
        runtime,
        _video((1, 16, 3, 2, 2)),
        prepared,
        steps=4,
    )

    assert result.shape == (1, 16, 3, 2, 2)
    assert captured_shifts == [5.0]
    assert len(FakePoseCache.instances) == 1
    cache = FakePoseCache.instances[0]
    assert cache.args == (torch.device("cpu"), "int8", 1234)
    assert cache.freed
    active = [call for call in diffusion.animate2_calls if call[0] is not None]
    inactive = [call for call in diffusion.animate2_calls if call[0] is None]
    assert active and inactive
    for pose, pose_context, pose_vision, pose_strength, reference_strength, call_cache in active:
        assert pose is not None and torch.equal(pose, torch.full_like(pose, 24.0))
        assert pose_context is not None and torch.equal(
            pose_context, torch.full_like(pose_context, 2.0)
        )
        assert pose_vision is not None and torch.equal(
            pose_vision, torch.full_like(pose_vision, 3.0)
        )
        assert pose_strength == 0.625
        assert reference_strength == 0.75
        assert call_cache is cache
    assert all(call[5] is None and call[4] == 0.75 for call in inactive)
    assert vae.processed_in == 2


def test_animate2_runtime_frees_pose_cache_when_sampling_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _, _, _ = _runtime("animate2")
    text = basic_conditioning_to_carrier(Conditioning(torch.ones((1, 2, 8)), None))
    prepared = runtime.prepare_conditioning(
        compose_wan21_animate2_conditioning(
            text,
            torch.zeros((1, 20, 3, 2, 2)),
        )
    )
    raw = cast("Any", runtime)
    raw._pose_cache_settings = Wan21PoseBlockCacheSettings()
    from dinkster_inference_torch import wan21_runtime

    class FakePoseCache:
        instances: list[FakePoseCache] = []

        def __init__(self, *_args: object) -> None:
            self.freed = False
            self.instances.append(self)

        def free(self) -> None:
            self.freed = True

    def fail_sampling(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError("sampler failed")

    monkeypatch.setattr(wan21_runtime, "PoseBranchCache", FakePoseCache)
    monkeypatch.setattr(sampling_engine, "run_denoise", fail_sampling)

    with pytest.raises(RuntimeError, match="sampler failed"):
        _sample(runtime, _video((1, 16, 3, 2, 2)), prepared)

    assert len(FakePoseCache.instances) == 1
    assert FakePoseCache.instances[0].freed


def test_scail_carrier_orders_references_and_materializes_strict_records() -> None:
    runtime, _, _, _ = _runtime("scail2")
    text = basic_conditioning_to_carrier(Conditioning(torch.ones((1, 2, 8)), None))
    primary = torch.ones((1, 16, 2, 2, 2))
    additional = torch.full((1, 16, 1, 2, 2), 2.0)
    pose = torch.full((1, 16, 2, 1, 1), 3.0)
    reference_mask = torch.full((1, 28, 5, 2, 2), 4.0)
    driving_mask = torch.full((1, 28, 2, 1, 1), 5.0)
    schedule = PercentRange(0.25, 0.75)

    carrier = compose_wan21_scail_conditioning(
        text,
        (primary, additional),
        vision=torch.ones((1, 257, 1280)),
        pose_latents=pose,
        reference_mask=reference_mask,
        driving_mask=driving_mask,
        pose_schedule=schedule,
        replacement=True,
    )

    main, pose_record = carrier.conditioning.records
    assert [channel for channel, _ in main.channels] == [
        ConditioningChannel.TEXT,
        ConditioningChannel.SCAIL_REFERENCE_LATENT,
        ConditioningChannel.VISION_EMBEDDING,
        ConditioningChannel.SCAIL_REFERENCE_MASK,
    ]
    assert dict(main.extension_metadata) == {WAN21_SCAIL_REPLACEMENT_KEY: True}
    assert [channel for channel, _ in pose_record.channels] == [
        ConditioningChannel.POSE_LATENT,
        ConditioningChannel.SCAIL_DRIVING_MASK,
    ]
    assert pose_record.schedule == schedule

    prepared = runtime.prepare_conditioning(carrier)
    assert prepared.scail_reference_latent is not None
    assert torch.equal(
        prepared.scail_reference_latent,
        torch.cat((additional, primary), dim=2),
    )
    assert prepared.pose_latents is not None and torch.equal(prepared.pose_latents, pose)
    assert prepared.scail_reference_mask is not None and torch.equal(
        prepared.scail_reference_mask, reference_mask
    )
    assert prepared.scail_driving_mask is not None and torch.equal(
        prepared.scail_driving_mask, driving_mask
    )
    assert prepared.pose_schedule == schedule
    assert prepared.scail_replacement is True


def test_scail_refuses_scail2_masks() -> None:
    runtime, _, _, _ = _runtime("scail")
    text = basic_conditioning_to_carrier(Conditioning(torch.ones((1, 2, 8)), None))
    carrier = compose_wan21_scail_conditioning(
        text,
        (torch.ones((1, 16, 1, 2, 2)),),
        reference_mask=torch.ones((1, 28, 2, 2, 2)),
    )

    with pytest.raises(Wan21RuntimeError, match="does not consume.*scail_reference_mask"):
        runtime.prepare_conditioning(carrier)


@pytest.mark.parametrize("model_type", ("scail", "scail2"))
def test_scail_runtime_normalizes_streams_and_uses_scalar_timesteps(model_type: str) -> None:
    runtime, diffusion, vae, _ = _runtime(model_type)
    assert runtime.supports_denoise_mask
    text = basic_conditioning_to_carrier(Conditioning(torch.ones((1, 2, 8)), None))
    primary = torch.ones((1, 16, 2, 2, 2))
    additional = torch.full((1, 16, 1, 2, 2), 2.0)
    pose = torch.full((1, 16, 2, 1, 1), 3.0)
    reference_mask = torch.full((1, 28, 5, 2, 2), 4.0) if model_type == "scail2" else None
    driving_mask = torch.full((1, 28, 2, 1, 1), 5.0) if model_type == "scail2" else None
    carrier = compose_wan21_scail_conditioning(
        text,
        (primary, additional),
        pose_latents=pose,
        reference_mask=reference_mask,
        driving_mask=driving_mask,
        replacement=True,
    )

    result = _sample(
        runtime,
        _video((1, 16, 2, 2, 2)),
        runtime.prepare_conditioning(carrier),
    )

    assert result.shape == (1, 16, 2, 2, 2)
    assert diffusion.scail_calls
    reference, model_pose, model_reference_mask, model_driving_mask, replacement = (
        diffusion.scail_calls[0]
    )
    assert reference.shape == (1, 20, 3, 2, 2)
    assert torch.equal(reference[:, :16, 0], torch.full_like(reference[:, :16, 0], 22.0))
    assert torch.equal(reference[:, :16, 1:], torch.full_like(reference[:, :16, 1:], 21.0))
    assert torch.equal(reference[:, 16:], torch.ones_like(reference[:, 16:]))
    assert model_pose is not None and model_pose.shape == (1, 20, 2, 1, 1)
    assert torch.equal(model_pose[:, :16], torch.full_like(model_pose[:, :16], 23.0))
    assert torch.equal(model_pose[:, 16:], torch.ones_like(model_pose[:, 16:]))
    assert replacement is True
    assert vae.processed_in == 2
    assert all(call[0].shape == (1, 20, 2, 2, 2) for call in diffusion.calls)
    assert all(call[1].shape == (1,) for call in diffusion.calls)
    assert all(torch.count_nonzero(call[0][:, 16:]) == 0 for call in diffusion.calls)
    if model_type == "scail2":
        assert reference_mask is not None and driving_mask is not None
        assert model_reference_mask is not None and torch.equal(
            model_reference_mask, reference_mask
        )
        assert model_driving_mask is not None and torch.equal(model_driving_mask, driving_mask)
    else:
        assert model_reference_mask is None and model_driving_mask is None


def test_scail2_denoise_mask_builds_history_and_keeps_fixed_frames() -> None:
    runtime, diffusion, _, _ = _runtime("scail2")
    text = basic_conditioning_to_carrier(Conditioning(torch.ones((1, 2, 8)), None))
    carrier = compose_wan21_scail_conditioning(
        text,
        (torch.ones((1, 16, 1, 2, 2)),),
    )
    latent = torch.zeros((1, 16, 2, 2, 2))
    latent[:, :, 0] = 2.0
    mask = torch.ones((1, 4, 2, 2, 2))
    mask[:, :, 0] = 0.0

    result = _sample(
        runtime,
        latent,
        runtime.prepare_conditioning(carrier),
        denoise_mask=mask,
    )

    assert all(call[1].shape == (1,) for call in diffusion.calls)
    assert all(
        torch.equal(call[0][:, 16:, 0], torch.ones_like(call[0][:, 16:, 0]))
        for call in diffusion.calls
    )
    assert all(torch.count_nonzero(call[0][:, 16:, 1:]) == 0 for call in diffusion.calls)
    assert torch.equal(result[:, :, 0], torch.full_like(result[:, :, 0], 32.0))


def test_scail2_four_channel_mask_preserves_history_and_fixed_blend() -> None:
    runtime, diffusion, _, _ = _runtime("scail2")
    text = basic_conditioning_to_carrier(Conditioning(torch.ones((1, 2, 8)), None))
    carrier = compose_wan21_scail_conditioning(
        text,
        (torch.ones((1, 16, 1, 2, 2)),),
    )
    latent = torch.zeros((1, 16, 1, 2, 2))
    mask = torch.ones((1, 4, 1, 2, 2))
    mask[:, 0] = 0.0

    result = _sample(
        runtime,
        latent,
        runtime.prepare_conditioning(carrier),
        denoise_mask=mask,
    )

    fixed_channels = (0, 4, 8, 12)
    generated_channels = tuple(index for index in range(16) if index not in fixed_channels)
    assert torch.equal(result[:, fixed_channels], torch.full_like(result[:, fixed_channels], 10.0))
    assert not torch.equal(
        result[:, generated_channels],
        torch.full_like(result[:, generated_channels], 10.0),
    )
    assert all(
        torch.equal(call[0][:, 16], torch.ones_like(call[0][:, 16]))
        and torch.count_nonzero(call[0][:, 17:]) == 0
        for call in diffusion.calls
    )


def test_phantom_runtime_normalizes_and_batches_real_and_empty_references(family_id: str) -> None:
    runtime, diffusion, vae, _ = _runtime("t2v", family_id=family_id)
    positive_text = runtime.prepare_text_conditioning(Conditioning(torch.ones((1, 2, 8)), None))
    negative_text = runtime.prepare_text_conditioning(Conditioning(torch.zeros((1, 2, 8)), None))
    reference = torch.full((1, 16, 2, 2, 2), 2.0)
    empty_reference = vae.process_out(torch.zeros_like(reference))
    positive = runtime.prepare_phantom_conditioning(positive_text, reference)
    negative = runtime.prepare_phantom_conditioning(negative_text, empty_reference)

    result = _sample(
        runtime,
        _video(),
        positive,
        cfg=SamplingGuidance(negative, 2.0, batching=FUSE_CFG_LANES),
    )

    assert result.shape == (1, 16, 1, 2, 2)
    assert len(diffusion.temporal_reference_calls) == 2
    assert [
        (
            tuple(temporal_reference.shape),
            temporal_reference[:, 0, 0, 0, 0].tolist(),
        )
        for temporal_reference in diffusion.temporal_reference_calls
    ] == [
        ((2, 16, 2, 2, 2), [30.0, 22.0]),
        ((2, 16, 2, 2, 2), [30.0, 22.0]),
    ]
    assert vae.processed_in == 2
    assert vae.processed_out == 2


def test_phantom_runtime_batches_three_lane_dual_cfg_in_reference_order() -> None:
    runtime, diffusion, vae, _ = _runtime("t2v")
    positive_text = runtime.prepare_text_conditioning(
        Conditioning(torch.full((1, 2, 8), 4.0), None)
    )
    middle_text = runtime.prepare_text_conditioning(Conditioning(torch.full((1, 2, 8), 2.0), None))
    negative_text = runtime.prepare_text_conditioning(Conditioning(torch.zeros((1, 2, 8)), None))
    reference = torch.full((1, 16, 2, 2, 2), 2.0)
    empty_reference = vae.process_out(torch.zeros_like(reference))
    positive = runtime.prepare_phantom_conditioning(positive_text, reference)
    middle = runtime.prepare_phantom_conditioning(middle_text, reference)
    negative = runtime.prepare_phantom_conditioning(negative_text, empty_reference)

    result = _sample(
        runtime,
        _video(),
        positive,
        cfg=DualSamplingGuidance(middle, negative, 7.5, 5.0, batching=FUSE_DUAL_CFG_LANES),
    )

    assert result.shape == (1, 16, 1, 2, 2)
    assert len(diffusion.calls) == 2
    assert [call[2].mean(dim=(1, 2)).tolist() for call in diffusion.calls] == [
        [4.0, 2.0, 0.0],
        [4.0, 2.0, 0.0],
    ]
    assert [
        (
            tuple(temporal_reference.shape),
            temporal_reference[:, 0, 0, 0, 0].tolist(),
        )
        for temporal_reference in diffusion.temporal_reference_calls
    ] == [
        ((3, 16, 2, 2, 2), [22.0, 22.0, 30.0]),
        ((3, 16, 2, 2, 2), [22.0, 22.0, 30.0]),
    ]


def test_phantom_runtime_rejects_non_t2v_profile_and_bad_reference(family_id: str) -> None:
    i2v, _, _, _ = _runtime("i2v", family_id=family_id)
    text = i2v.prepare_text_conditioning(Conditioning(torch.ones((1, 2, 8)), None))
    with pytest.raises(Wan21RuntimeError, match="does not consume Phantom"):
        i2v.prepare_phantom_conditioning(text, torch.zeros((1, 16, 1, 2, 2)))

    t2v, _, _, _ = _runtime("t2v", family_id=family_id)
    text = t2v.prepare_text_conditioning(Conditioning(torch.ones((1, 2, 8)), None))
    with pytest.raises(Wan21RuntimeError, match=r"\[B,16,T,H,W\]"):
        t2v.prepare_phantom_conditioning(text, torch.zeros((1, 15, 1, 2, 2)))


def test_flow_rvs_sampling_rejects_manually_prepared_phantom_reference() -> None:
    runtime, _, _, _ = _runtime("flow_rvs")
    prepared = Wan21PreparedConditioning(
        torch.ones((1, 2, 8)),
        temporal_reference=torch.zeros((1, 16, 1, 2, 2)),
    )

    with pytest.raises(Wan21RuntimeError, match="does not consume Phantom"):
        _sample(runtime, _video(), prepared)


def _admit_test_bernini(monkeypatch: pytest.MonkeyPatch) -> tuple[Wan21Runtime, _Diffusion, _VAE]:
    monkeypatch.setattr(wan21_runtime_module, "Wan21Model", _Diffusion)
    runtime, diffusion, vae, _ = _runtime("bernini")
    cast("Any", diffusion).config = replace(WAN22_BERNINI_14B)
    assert diffusion.config is not WAN22_BERNINI_14B
    return runtime, diffusion, vae


@pytest.mark.parametrize("operation", ("prepare", "sample"))
def test_bernini_rejects_non_wan_diffusion_before_context_tensor_work(operation: str) -> None:
    runtime, diffusion, _, _ = _runtime("bernini")
    cast("Any", diffusion).config = replace(WAN22_BERNINI_14B)
    text = Wan21PreparedConditioning(torch.ones((1, 2, 4096)))
    context = torch.zeros((1, 16, 1, 2, 2))

    with pytest.raises(Wan21RuntimeError, match="exact native Wan 2.2 14B T2V"):
        if operation == "prepare":
            runtime.prepare_bernini_conditioning(text, (context,))
        else:
            _sample(runtime, _video(), replace(text, context_latents=(context,)))


def test_bernini_normalizes_context_once_and_batches_cfg_lanes(
    monkeypatch: pytest.MonkeyPatch,
    family_id: str,
) -> None:
    runtime, diffusion, vae = _admit_test_bernini(monkeypatch)
    cast("Any", runtime.assembled).family = replace(WAN21, id=family_id)
    positive_text = Wan21PreparedConditioning(torch.ones((1, 2, 4096)))
    negative_text = Wan21PreparedConditioning(torch.zeros((1, 2, 4096)))
    first = torch.full((1, 16, 1, 2, 2), 2.0)
    second = torch.full((1, 16, 2, 1, 1), 3.0)
    positive = runtime.prepare_bernini_conditioning(positive_text, (first, second))
    negative = runtime.prepare_bernini_conditioning(negative_text, (first, second))

    _sample(
        runtime,
        torch.cat((_video(), _video()), dim=0),
        positive,
        cfg=SamplingGuidance(negative, 2.0, batching=FUSE_CFG_LANES),
    )

    assert vae.processed_in == 4
    assert diffusion.context_calls
    assert len(diffusion.context_calls) == len(diffusion.calls)
    for context_latents in diffusion.context_calls:
        assert [tuple(value.shape) for value in context_latents] == [
            (4, 16, 1, 2, 2),
            (4, 16, 2, 1, 1),
        ]
        assert torch.equal(context_latents[0], torch.full_like(context_latents[0], 22.0))
        assert torch.equal(context_latents[1], torch.full_like(context_latents[1], 23.0))


def test_bernini_batch_expansion_is_one_reused_noncopy_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, diffusion, vae = _admit_test_bernini(monkeypatch)
    text = Wan21PreparedConditioning(torch.ones((1, 2, 4096)))
    context = torch.full((1, 16, 1, 2, 2), 2.0)
    prepared = runtime.prepare_bernini_conditioning(text, (context,))

    _sample(runtime, torch.cat((_video(), _video()), dim=0), prepared)

    assert vae.processed_in == 1
    assert diffusion.context_calls
    views = [call[0] for call in diffusion.context_calls]
    assert all(value.shape == (2, 16, 1, 2, 2) for value in views)
    assert all(value.stride(0) == 0 and value._base is not None for value in views)
    assert all(value is views[0] for value in views)


def test_bernini_sampling_refuses_manual_context_on_other_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wan21_runtime_module, "Wan21Model", _Diffusion)
    runtime, _, _, _ = _runtime("t2v")
    prepared = Wan21PreparedConditioning(
        torch.ones((1, 2, 8)),
        context_latents=(torch.zeros((1, 16, 1, 2, 2)),),
    )

    with pytest.raises(Wan21RuntimeError, match="require exact native Wan 2.2 14B T2V"):
        _sample(runtime, _video(), prepared)


def test_bernini_refuses_context_combined_with_another_intervention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _, _ = _admit_test_bernini(monkeypatch)
    prepared = Wan21PreparedConditioning(
        torch.ones((1, 2, 4096)),
        temporal_reference=torch.zeros((1, 16, 1, 2, 2)),
        context_latents=(torch.zeros((1, 16, 1, 2, 2)),),
    )

    with pytest.raises(Wan21RuntimeError, match="conditioning for another Wan variant"):
        _sample(runtime, _video(), prepared)


class _Uni3CExecution:
    def __init__(
        self,
        render_latent: torch.Tensor,
        *,
        strength: float = 1.0,
        window: PercentRange | None = None,
    ) -> None:
        self.render_latent = render_latent
        self.strength = strength
        self.window = window or PercentRange(0.0, 1.0)
        self.model_digest = "model-digest"
        self.render_digest = "render-digest"


def _admit_test_uni3c(monkeypatch: pytest.MonkeyPatch) -> None:
    def snapshot(value: object) -> object:
        return value

    monkeypatch.setattr(wan21_runtime_module, "Wan21Model", _Diffusion)
    monkeypatch.setattr(wan21_runtime_module, "Wan21Uni3CExecution", _Uni3CExecution)
    monkeypatch.setattr(wan21_runtime_module, "snapshot_wan21_uni3c_execution", snapshot)


def test_uni3c_t2v_zero_pads_control_and_reuses_one_lane_for_cfg(
    monkeypatch: pytest.MonkeyPatch,
    family_id: str,
) -> None:
    _admit_test_uni3c(monkeypatch)
    runtime, diffusion, _, _ = _runtime("t2v", family_id=family_id)
    cast("Any", diffusion).config = WAN21_T2V_14B
    positive = Wan21PreparedConditioning(torch.ones((1, 2, 4096)))
    negative = Wan21PreparedConditioning(torch.zeros((1, 2, 4096)))
    render = torch.full((1, 16, 1, 2, 2), 7.0)
    execution = _Uni3CExecution(render)

    _sample(
        runtime,
        _video(),
        positive,
        cfg=SamplingGuidance(negative, 2.0, batching=FUSE_CFG_LANES),
        uni3c=execution,
    )

    assert diffusion.uni3c_calls
    assert len(diffusion.uni3c_calls) == len(diffusion.calls)
    for (seen_execution, control), model_call in zip(
        diffusion.uni3c_calls, diffusion.calls, strict=True
    ):
        assert seen_execution is execution
        assert control.shape == (1, 36, 1, 2, 2)
        assert torch.equal(control[:, :16], model_call[0][:1, :16])
        assert torch.count_nonzero(control[:, 16:20]) == 0
        assert torch.equal(control[:, 20:], render)
        assert model_call[0].shape[0] == 2


def test_uni3c_expands_batch_once_before_denoiser_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _admit_test_uni3c(monkeypatch)
    runtime, diffusion, _, _ = _runtime("t2v")
    cast("Any", diffusion).config = WAN21_T2V_14B
    render_shape = (1, 16, 1, 2, 2)
    original_to_batch = wan21_runtime_module.to_batch

    def refuse_render_repeat(tensor: torch.Tensor, batch: int) -> torch.Tensor:
        if tuple(tensor.shape) == render_shape and batch == 2:
            raise AssertionError("Uni3C render must not repeat inside the denoiser")
        return original_to_batch(tensor, batch)

    monkeypatch.setattr(wan21_runtime_module, "to_batch", refuse_render_repeat)
    render = torch.full(render_shape, 7.0)
    video = torch.cat((_video(), _video()), dim=0)

    _sample(
        runtime,
        video,
        Wan21PreparedConditioning(torch.ones((1, 2, 4096))),
        uni3c=_Uni3CExecution(render),
    )

    assert diffusion.uni3c_calls
    for _execution, control in diffusion.uni3c_calls:
        assert control.shape == (2, 36, 1, 2, 2)
        assert torch.equal(control[:, 20:], render.expand(2, -1, -1, -1, -1))


def test_uni3c_i2v_uses_noise_and_four_mask_channels_from_shared_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _admit_test_uni3c(monkeypatch)
    runtime, diffusion, _, _ = _runtime("i2v")
    cast("Any", diffusion).config = WAN21_I2V_14B
    concat = torch.zeros((1, 20, 1, 2, 2))
    concat[:, :4] = torch.arange(1, 5, dtype=torch.float32).view(1, 4, 1, 1, 1)
    concat[:, 4:] = 9.0
    positive = Wan21PreparedConditioning(
        torch.ones((1, 2, 4096)),
        concat_latent=concat,
        concat_mask_index=0,
    )
    negative = Wan21PreparedConditioning(
        torch.zeros((1, 2, 4096)),
        concat_latent=concat,
        concat_mask_index=0,
    )
    render = torch.full((1, 16, 1, 2, 2), 6.0)

    _sample(
        runtime,
        _video(),
        positive,
        cfg=SamplingGuidance(negative, 2.0, batching=FUSE_CFG_LANES),
        uni3c=_Uni3CExecution(render),
    )

    assert diffusion.uni3c_calls
    for (_execution, control), model_call in zip(
        diffusion.uni3c_calls, diffusion.calls, strict=True
    ):
        assert control.shape == (1, 36, 1, 2, 2)
        assert torch.equal(control[:, :20], model_call[0][:1, :20])
        assert torch.equal(control[:, 16:20], concat[:, :4])
        assert torch.equal(control[:, 20:], render)
        assert model_call[0].shape[0] == 2


@pytest.mark.parametrize(
    "execution",
    (
        _Uni3CExecution(torch.zeros((1, 16, 1, 2, 2)), strength=0.0),
        _Uni3CExecution(
            torch.zeros((1, 16, 1, 2, 2)),
            window=PercentRange(1.0, 1.0),
        ),
    ),
)
def test_uni3c_skips_zero_strength_and_inactive_windows(
    monkeypatch: pytest.MonkeyPatch,
    execution: _Uni3CExecution,
) -> None:
    _admit_test_uni3c(monkeypatch)
    runtime, diffusion, _, _ = _runtime("t2v")
    cast("Any", diffusion).config = WAN21_T2V_14B

    _sample(
        runtime,
        _video(),
        Wan21PreparedConditioning(torch.ones((1, 2, 4096))),
        uni3c=execution,
    )

    assert not diffusion.uni3c_calls


def test_uni3c_refuses_phantom_and_non_exact_wan_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _admit_test_uni3c(monkeypatch)
    runtime, diffusion, _, _ = _runtime("t2v")
    cast("Any", diffusion).config = WAN21_T2V_14B
    prepared = Wan21PreparedConditioning(
        torch.ones((1, 2, 4096)),
        temporal_reference=torch.zeros((1, 16, 1, 2, 2)),
    )
    execution = _Uni3CExecution(torch.zeros((1, 16, 1, 2, 2)))
    with pytest.raises(Wan21RuntimeError, match="cannot be combined with Phantom"):
        _sample(runtime, _video(), prepared, uni3c=execution)

    other_runtime, _, _, _ = _runtime("t2v")
    with pytest.raises(Wan21RuntimeError, match="only exact native base"):
        _sample(
            other_runtime,
            _video(),
            Wan21PreparedConditioning(torch.ones((1, 2, 8))),
            uni3c=execution,
        )


@pytest.mark.parametrize(
    ("model_type", "expected_extra", "vision"),
    (
        ("camera21", 16, torch.zeros((1, 257, 1280))),
        ("camera22", 20, None),
    ),
)
def test_camera_runtime_normalizes_reference_and_propagates_cfg_trajectory(
    model_type: str,
    expected_extra: int,
    vision: torch.Tensor | None,
) -> None:
    runtime, diffusion, _, _ = _runtime(model_type)
    text = runtime.prepare_text_conditioning(Conditioning(torch.ones((1, 2, 8)), None))
    concat = torch.full((1, expected_extra, 1, 2, 2), 2.0)
    if model_type == "camera22":
        concat[:, :4] = 1.0
    camera = torch.arange(24 * 16 * 16, dtype=torch.float32).reshape(1, 24, 1, 16, 16)
    prepared = runtime.prepare_camera_conditioning(text, concat, camera, vision)

    _sample(
        runtime,
        _video(),
        prepared,
        cfg=SamplingGuidance(prepared, 2.0, batching=FUSE_CFG_LANES),
    )

    assert diffusion.calls
    model_input = diffusion.calls[0][0]
    assert model_input.shape[1] == 16 + expected_extra
    if model_type == "camera21":
        assert torch.equal(model_input[:, 16:], torch.full((2, 16, 1, 2, 2), 22.0))
    else:
        assert torch.equal(model_input[:, 16:20], torch.ones((2, 4, 1, 2, 2)))
        assert torch.equal(model_input[:, 20:], torch.full((2, 16, 1, 2, 2), 22.0))
    assert diffusion.camera_calls
    assert torch.equal(diffusion.camera_calls[0], camera.expand(2, -1, -1, -1, -1))


@pytest.mark.parametrize("model_type", ("camera21", "camera22"))
def test_camera_runtime_uses_zero_reference_when_start_image_is_absent(model_type: str) -> None:
    runtime, diffusion, vae, _ = _runtime(model_type)
    text = runtime.prepare_text_conditioning(Conditioning(torch.ones((1, 2, 8)), None))
    camera = torch.zeros((1, 24, 1, 16, 16))
    prepared = runtime.prepare_camera_conditioning(text, None, camera)

    _sample(runtime, _video(), prepared)

    assert diffusion.calls
    model_input = diffusion.calls[0][0]
    assert model_input.shape[1] == diffusion.config.in_channels
    assert torch.count_nonzero(model_input[:, 16:]) == 0
    assert vae.processed_in == 0


def test_camera_runtime_fails_closed_on_wrong_profile_and_geometry() -> None:
    t2v, _, _, _ = _runtime()
    text = t2v.prepare_text_conditioning(Conditioning(torch.ones((1, 2, 8)), None))
    concat = torch.zeros((1, 16, 1, 2, 2))
    camera = torch.zeros((1, 24, 1, 16, 16))
    with pytest.raises(Wan21RuntimeError, match="does not consume camera"):
        t2v.prepare_camera_conditioning(text, concat, camera)

    camera_runtime, _, _, _ = _runtime("camera21")
    camera_text = camera_runtime.prepare_text_conditioning(
        Conditioning(torch.ones((1, 2, 8)), None)
    )
    vision = torch.zeros((1, 257, 1280))
    with pytest.raises(Wan21RuntimeError, match="geometry must match"):
        camera_runtime.prepare_camera_conditioning(
            camera_text,
            concat,
            torch.zeros((1, 24, 1, 8, 16)),
            vision,
        )
    with pytest.raises(Wan21RuntimeError, match="B,16"):
        camera_runtime.prepare_camera_conditioning(
            camera_text,
            torch.zeros((1, 20, 1, 2, 2)),
            camera,
            vision,
        )


@pytest.mark.parametrize(
    ("latent_channels", "mask_index", "concat_channels"),
    (
        (16, None, 32),
        (16, 0, 20),
        (16, 16, 36),
        (48, 0, 52),
        (48, 48, 100),
    ),
)
def test_fun_concat_normalization_preserves_mask_channels(
    latent_channels: int,
    mask_index: int | None,
    concat_channels: int,
) -> None:
    value = torch.arange(concat_channels, dtype=torch.float32).reshape(1, concat_channels, 1, 1, 1)
    actual = _normalize_concat_latent(
        value,
        latent_channels=latent_channels,
        mask_index=mask_index,
        process_in=lambda part: part + 100.0,
    )
    expected = value + 100.0
    if mask_index is not None:
        expected[:, mask_index : mask_index + 4] = value[:, mask_index : mask_index + 4]

    assert torch.equal(actual, expected)


def test_fun_control_builds_wide_concat_and_full_reference_input() -> None:
    runtime, diffusion, vae, _ = _runtime("fun22")
    text = runtime.prepare_text_conditioning(Conditioning(torch.ones((1, 2, 8)), None))
    concat = torch.zeros((1, 36, 1, 2, 2))
    concat[:, :16] = 1.0
    concat[:, 16:20] = 2.0
    concat[:, 20:] = 3.0
    reference = torch.full((1, 16, 1, 2, 2), 4.0)
    prepared = runtime.prepare_fun_conditioning(
        text,
        concat,
        concat_mask_index=16,
        reference_latent=reference,
    )

    result = _sample(runtime, _video(), prepared)

    assert result.shape == (1, 16, 1, 2, 2)
    assert len(diffusion.calls) == 2
    for model_input, _, _, model_vision in diffusion.calls:
        assert model_input.shape == (1, 52, 1, 2, 2)
        assert torch.equal(model_input[:, 16:32], torch.full((1, 16, 1, 2, 2), 21.0))
        assert torch.equal(model_input[:, 32:36], torch.full((1, 4, 1, 2, 2), 2.0))
        assert torch.equal(model_input[:, 36:], torch.full((1, 16, 1, 2, 2), 23.0))
        assert model_vision is None
    assert len(diffusion.reference_calls) == 2
    assert all(
        torch.equal(call, torch.full((1, 16, 2, 2), 24.0)) for call in diffusion.reference_calls
    )
    assert vae.processed_in == 3


def test_wan22_fun_inpaint_14b_accepts_indistinguishable_i2v_profile() -> None:
    runtime, _, _, _ = _runtime("i2v22")
    text = runtime.prepare_text_conditioning(Conditioning(torch.ones((1, 2, 8)), None))

    prepared = runtime.prepare_fun_conditioning(
        text,
        torch.zeros((1, 20, 1, 2, 2)),
        concat_mask_index=0,
    )

    assert prepared.concat_latent is not None
    assert prepared.concat_latent.shape == (1, 20, 1, 2, 2)
    assert prepared.concat_mask_index == 0


def test_i2v_refuses_missing_mismatched_and_t2v_only_conditioning() -> None:
    runtime, _, _, _ = _runtime("i2v")
    text = runtime.prepare_text_conditioning(Conditioning(torch.ones((1, 2, 8)), None))
    concat = torch.zeros((1, 20, 1, 2, 2))
    vision = torch.zeros((1, 257, 1280))

    with pytest.raises(Wan21RuntimeError, match="requires CONCAT_LATENT"):
        _sample(runtime, _video(), text)
    with pytest.raises(Wan21RuntimeError, match="temporal and spatial shape"):
        _sample(
            runtime,
            _video(),
            runtime.prepare_i2v_conditioning(text, torch.zeros((1, 20, 2, 2, 2)), vision),
        )
    with pytest.raises(Wan21RuntimeError, match=r"shape \[B,257,1280\]"):
        runtime.prepare_i2v_conditioning(text, concat, torch.zeros((1, 256, 1280)))
    with pytest.raises(Wan21RuntimeError, match=r"shape \[B,257,1280\]"):
        runtime.prepare_i2v_conditioning(text, concat, torch.zeros((0, 257, 1280)))
    with pytest.raises(Wan21RuntimeError, match=r"shape \[B,257,1280\]"):
        runtime.prepare_i2v_conditioning(text, concat, torch.zeros((1, 514, 1280)))
    with pytest.raises(TypeError, match="VISION_EMBEDDING"):
        runtime.prepare_i2v_conditioning(
            text, concat, torch.zeros((1, 257, 1280), dtype=torch.int64)
        )
    with pytest.raises(TypeError, match="CONCAT_LATENT"):
        runtime.prepare_i2v_conditioning(
            text, torch.zeros((1, 20, 1, 2, 2), dtype=torch.int64), vision
        )

    t2v, _, _, _ = _runtime()
    t2v_text = t2v.prepare_text_conditioning(Conditioning(torch.ones((1, 2, 8)), None))
    with pytest.raises(Wan21RuntimeError, match="does not consume reference-latent"):
        t2v.prepare_i2v_conditioning(t2v_text, concat, vision)


@pytest.mark.parametrize("rows", (257, 514))
def test_flf_profile_accepts_one_or_two_clip_vision_encodes(rows: int) -> None:
    runtime, _, _, _ = _runtime("flf")
    text = runtime.prepare_text_conditioning(Conditioning(torch.ones((1, 2, 8)), None))
    prepared = runtime.prepare_i2v_conditioning(
        text,
        torch.zeros((1, 20, 1, 2, 2)),
        torch.zeros((1, rows, 1280)),
    )

    assert prepared.vision is not None
    assert prepared.vision.shape == (1, rows, 1280)


def test_vace_runtime_normalizes_and_propagates_ordered_controls() -> None:
    runtime, diffusion, _, _ = _runtime("vace")
    prepared = runtime.prepare_text_conditioning(Conditioning(torch.ones((1, 2, 8)), None))
    first_frames = torch.full((1, 32, 2, 2, 2), 2.0, dtype=torch.float64)
    first_masks = torch.zeros((1, 64, 2, 2, 2), dtype=torch.float64)
    first_masks[:, :, 1] = 0.25
    second_frames = torch.full((1, 32, 2, 2, 2), 4.0)
    second_masks = torch.full((1, 64, 2, 2, 2), 0.75)
    prepared = runtime.prepare_vace_conditioning(prepared, first_frames, first_masks, 0.25)
    prepared = runtime.prepare_vace_conditioning(prepared, second_frames, second_masks, 1.5)

    result = _sample(
        runtime,
        _video((1, 16, 2, 2, 2)),
        prepared,
        cfg=SamplingGuidance(prepared, 2.0, batching=FUSE_CFG_LANES),
    )

    assert result.shape == (1, 16, 2, 2, 2)
    assert len(diffusion.vace_calls) == 2
    for context, strengths in diffusion.vace_calls:
        assert context.shape == (2, 2, 96, 2, 2, 2)
        assert context.dtype is torch.float32
        assert strengths == (0.25, 1.5)
        torch.testing.assert_close(context[:, 0, :32], torch.full_like(context[:, 0, :32], 22.0))
        torch.testing.assert_close(context[:, 0, 32:, 0], torch.zeros_like(context[:, 0, 32:, 0]))
        torch.testing.assert_close(
            context[:, 0, 32:, 1], torch.full_like(context[:, 0, 32:, 1], 0.25)
        )
        torch.testing.assert_close(context[:, 1, :32], torch.full_like(context[:, 1, :32], 24.0))
        torch.testing.assert_close(context[:, 1, 32:], torch.full_like(context[:, 1, 32:], 0.75))


def test_vace_runtime_fails_closed_on_missing_or_invalid_payloads() -> None:
    runtime, _, _, _ = _runtime("vace")
    text = runtime.prepare_text_conditioning(Conditioning(torch.ones((1, 2, 8)), None))
    frame = torch.zeros((1, 32, 2, 2, 2))
    mask = torch.ones((1, 64, 2, 2, 2))

    with pytest.raises(Wan21RuntimeError, match="requires VACE conditioning"):
        _sample(runtime, _video((1, 16, 2, 2, 2)), text)
    with pytest.raises(ValueError, match=r"\[B,32,T,H,W\]"):
        runtime.prepare_vace_conditioning(text, frame[:, :31], mask, 1.0)
    with pytest.raises(ValueError, match=r"\[B,64,T,H,W\]"):
        runtime.prepare_vace_conditioning(text, frame, mask[:, :63], 1.0)
    with pytest.raises(ValueError, match="frame and mask geometry must match"):
        runtime.prepare_vace_conditioning(text, frame, mask[:, :, :1], 1.0)
    with pytest.raises(ValueError, match="finite non-negative"):
        runtime.prepare_vace_conditioning(text, frame, mask, float("nan"))
    with pytest.raises(ValueError, match="temporal and spatial conditioning shape"):
        _sample(
            runtime,
            _video((1, 16, 1, 2, 2)),
            runtime.prepare_vace_conditioning(text, frame, mask, 1.0),
        )

    t2v, _, _, _ = _runtime()
    t2v_text = t2v.prepare_text_conditioning(Conditioning(torch.ones((1, 2, 8)), None))
    with pytest.raises(Wan21RuntimeError, match="does not consume VACE"):
        t2v.prepare_vace_conditioning(t2v_text, frame, mask, 1.0)


def test_vace_prepared_conditioning_rejects_inconsistent_tuples() -> None:
    text = torch.ones((1, 2, 8))
    frame = torch.zeros((1, 32, 1, 2, 2))
    mask = torch.ones((1, 64, 1, 2, 2))

    with pytest.raises(ValueError, match="matching exact tuples"):
        Wan21PreparedConditioning(text, vace_frames=(frame,), vace_masks=(), vace_strengths=(1.0,))
    with pytest.raises(ValueError, match="finite non-negative floats"):
        Wan21PreparedConditioning(
            text,
            vace_frames=(frame,),
            vace_masks=(mask,),
            vace_strengths=(-1.0,),
        )


def test_sample_requires_family_owned_prepared_conditioning() -> None:
    runtime, _, _, _ = _runtime()
    streams = MultiStreamLatent.from_pairs((("video", _video()),))
    with pytest.raises(TypeError, match="exact Wan21PreparedConditioning"):
        runtime.sample_multistream(
            streams,
            conditioning=Conditioning(torch.ones((1, 2, 8)), None),
            sampler_id="euler",
            scheduler_id="simple",
            steps=2,
            denoise=1.0,
            seed=123,
        )
    with pytest.raises(Wan21RuntimeError, match=r"\[B,tokens,8\]"):
        runtime.sample_multistream(
            streams,
            conditioning=Wan21PreparedConditioning(torch.ones((1, 2, 7))),
            sampler_id="euler",
            scheduler_id="simple",
            steps=2,
            denoise=0.0,
            seed=123,
        )
    prepared = Wan21PreparedConditioning(torch.ones((1, 2, 8)))
    assert prepared.text.shape == (1, 2, 8)


def test_codec_entry_points_preserve_rank5_video_geometry() -> None:
    runtime, _, _, codec = _runtime()
    content = torch.ones((2, 3, 5, 8, 8), dtype=torch.bfloat16)
    latent = torch.ones((2, 16, 2, 1, 1), dtype=torch.bfloat16)

    encoded = runtime.encode_content(content)
    decoded = runtime.decode_latent(latent)

    assert codec.encoded == [content]
    assert codec.decoded == [latent]
    assert encoded.shape == (2, 1, 5, 8, 8)
    assert decoded.shape == (2, 1, 2, 1, 1)
    assert encoded.dtype is torch.float32
    assert decoded.dtype is torch.float32
    with pytest.raises(ValueError, match="rank 5"):
        runtime.encode_content(content[:, :, 0])
    with pytest.raises(ValueError, match="rank 5"):
        runtime.decode_latent(latent[:, :, 0])


def _windows_spec(**overrides: Any) -> ContextWindowsSpec:
    values: dict[str, Any] = {
        "schedule": ContextWindowSchedule.STATIC_STANDARD,
        "fuse_method": ContextFuseMethod.PYRAMID,
        "length": 3,
        "overlap": 1,
        "dim": 2,
    }
    values.update(overrides)
    return ContextWindowsSpec(**values)


def test_context_windows_split_model_applications_and_fuse_to_full_length() -> None:
    video = _video((1, 16, 7, 2, 2))
    cond = Conditioning(torch.ones((1, 4, 8)))

    baseline_runtime, baseline_diffusion, _, _ = _runtime()
    baseline = _sample(baseline_runtime, video.clone(), cond, steps=1)
    assert [call[0].shape[2] for call in baseline_diffusion.calls] == [7]

    windowed_runtime, windowed_diffusion, _, _ = _runtime()
    windowed = _sample(
        windowed_runtime,
        video.clone(),
        cond,
        steps=1,
        context_windows=_windows_spec(),
    )

    # length 3 with overlap 1 tiles 7 frames as [0..2], [2..4], [4..6].
    assert [call[0].shape[2] for call in windowed_diffusion.calls] == [3, 3, 3]
    # The fake model's velocity depends only on the text context, so the
    # fused windowed result must reconstruct the unwindowed output.
    torch.testing.assert_close(windowed, baseline, rtol=1e-6, atol=1e-6)


def test_context_windows_batch_lanes_window_per_invocation() -> None:
    video = _video((1, 16, 7, 2, 2))
    positive = Conditioning(torch.ones((1, 4, 8)))
    negative = Wan21PreparedConditioning(torch.zeros((1, 4, 8)))

    runtime, diffusion, _, _ = _runtime()
    _sample(
        runtime,
        video,
        positive,
        cfg=SamplingGuidance(cast("Any", negative), 2.0, batching=FUSE_CFG_LANES),
        steps=1,
        context_windows=_windows_spec(),
    )

    # Three windows, each stacking both guidance lanes into one model row pair.
    assert [tuple(call[0].shape[:3]) for call in diffusion.calls] == [(2, 16, 3)] * 3


def test_context_windows_freenoise_shuffles_initial_noise() -> None:
    video = _video((1, 16, 7, 2, 2))
    cond = Conditioning(torch.ones((1, 4, 8)))
    seed = 123

    plain_runtime, _, _, _ = _runtime()
    plain = _sample(plain_runtime, video.clone(), cond, steps=1, seed=seed)

    freenoise_runtime, _, _, _ = _runtime()
    shuffled = _sample(
        freenoise_runtime,
        video.clone(),
        cond,
        steps=1,
        seed=seed,
        context_windows=_windows_spec(freenoise=True),
    )

    # The euler step is linear in the initial noise, so the output delta is
    # exactly the FreeNoise shuffle delta and the first window is untouched.
    base_noise = prepare_noise(video, seed)
    expected_delta = apply_freenoise(base_noise, 2, 3, 1, seed) - base_noise
    assert not torch.equal(shuffled, plain)
    torch.testing.assert_close(shuffled - plain, expected_delta, rtol=1e-5, atol=1e-5)
    assert torch.equal(expected_delta[:, :, :3], torch.zeros_like(expected_delta[:, :, :3]))


def test_context_windows_admission_refuses_unsupported_profiles_and_extras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _windows_spec()
    text_only = Wan21PreparedConditioning(torch.ones((1, 2, 8)))

    ti2v_runtime, _, _, _ = _runtime("ti2v")
    with pytest.raises(Wan21RuntimeError, match="base text-to-video"):
        _sample(ti2v_runtime, _video((1, 48, 1, 2, 2)), text_only, context_windows=spec)

    vace_runtime, _, _, _ = _runtime("vace")
    vace_text = vace_runtime.prepare_text_conditioning(Conditioning(torch.ones((1, 2, 8)), None))
    vace_prepared = vace_runtime.prepare_vace_conditioning(
        vace_text, torch.zeros((1, 32, 1, 2, 2)), torch.ones((1, 64, 1, 2, 2)), 1.0
    )
    with pytest.raises(Wan21RuntimeError, match="do not support VACE"):
        _sample(vace_runtime, _video(), vace_prepared, context_windows=spec)

    masked_runtime, _, vae, _ = _runtime()
    latent = torch.ones_like(_video())
    masked = _sample(
        masked_runtime,
        latent,
        text_only,
        denoise_mask=torch.zeros((1, 1, 1, 2, 2)),
        context_windows=spec,
    )
    assert torch.equal(masked, vae.process_out(vae.process_in(latent)))

    _admit_test_uni3c(monkeypatch)
    uni3c_runtime, uni3c_diffusion, _, _ = _runtime("t2v")
    cast("Any", uni3c_diffusion).config = WAN21_T2V_14B
    with pytest.raises(Wan21RuntimeError, match="Uni3C or InfiniteTalk"):
        _sample(
            uni3c_runtime,
            _video(),
            Wan21PreparedConditioning(torch.ones((1, 2, 4096))),
            uni3c=_Uni3CExecution(torch.zeros((1, 16, 1, 2, 2))),
            context_windows=spec,
        )


def test_context_windows_admission_refuses_structural_conditioning_lanes() -> None:
    spec = _windows_spec()
    structural = Wan21PreparedConditioning(
        torch.ones((1, 2, 8)),
        temporal_reference=torch.zeros((1, 16, 1, 2, 2)),
    )

    runtime, _, _, _ = _runtime()
    with pytest.raises(Wan21RuntimeError, match="text-only conditioning"):
        _sample(runtime, _video(), structural, context_windows=spec)

    uncond_runtime, _, _, _ = _runtime()
    with pytest.raises(Wan21RuntimeError, match="text-only unconditional"):
        _sample(
            uncond_runtime,
            _video(),
            Wan21PreparedConditioning(torch.ones((1, 2, 8))),
            cfg=SamplingGuidance(cast("Any", structural), 2.0),
            context_windows=spec,
        )
