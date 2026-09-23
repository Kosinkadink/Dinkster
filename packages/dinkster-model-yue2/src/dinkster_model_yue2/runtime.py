"""YuE2 text generation, codec, and shared sampling-engine adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import partial
from types import MappingProxyType
from typing import Any, cast

import torch
from dinkster_inference import (
    Conditioning,
    ConditioningCarrier,
    ConditioningSet,
    FlowSigmas,
    GuidanceRole,
    ModelFamily,
    Registry,
    SamplerDescriptor,
    SchedulerDescriptor,
    SigmaSpace,
    make_conditioning_carrier,
)
from dinkster_inference.assembly import ComponentPlan
from dinkster_inference_torch.assemble import _load_component  # pyright: ignore[reportPrivateUsage]
from dinkster_inference_torch.attention import AttentionKernel, AttentionRole
from dinkster_inference_torch.conditioning_adapters import (
    basic_conditioning_to_carrier,
    materialize_basic_conditioning,
)
from dinkster_inference_torch.guidance import GuidanceExecutor
from dinkster_inference_torch.operations import module_compute_device
from dinkster_inference_torch.sampling_execution import (
    SamplingAdapterContext,
    SamplingDenoiserAdapter,
    SamplingDenoiserExecution,
    SamplingExecutionRegistration,
    SingleStreamLatentAdapter,
    sampling_execution,
)
from dinkster_inference_torch.sampling_runtime import SingleStreamSamplingRuntime
from dinkster_inference_torch.schedules import torch_scheduler_registry
from dinkster_inference_torch.solvers import torch_sampler_registry
from dinkster_inference_torch.sources import load_tensors
from tokenizers import Tokenizer

from .codec import AudioOobleckVAE
from .declarations import FRAMES_PER_SECOND, LATENT_CHANNELS, YUE2_FAMILY
from .model import YuE2AcousticModel
from .text import ABC_END, CONTEXT, MUSIC_START, YuE2TextModel, prompt_tokens, tokenizer_from_bytes

_CHUNKS_METADATA = "dinkster-model-yue2/chunks"
_FRAMES_METADATA = "dinkster-model-yue2/frames"


@dataclass(frozen=True)
class YuE2Conditioning(Conditioning[torch.Tensor]):
    chunks: tuple[tuple[int, int, int, int], ...] = ()
    frames: int = 0


def yue2_conditioning_to_carrier(value: YuE2Conditioning) -> ConditioningCarrier:
    carrier = basic_conditioning_to_carrier(value)
    (record,) = carrier.conditioning.records
    record = replace(
        record,
        extension_metadata=(
            (_CHUNKS_METADATA, tuple(tuple(item) for item in value.chunks)),
            (_FRAMES_METADATA, value.frames),
        ),
    )
    return make_conditioning_carrier(ConditioningSet((record,)), carrier.bindings)


def materialize_yue2_conditioning(
    carrier: ConditioningCarrier,
    *,
    device: torch.device | str,
) -> YuE2Conditioning:
    (record,) = carrier.conditioning.records
    metadata = dict(record.extension_metadata)
    chunks_raw = metadata.get(_CHUNKS_METADATA)
    frames = metadata.get(_FRAMES_METADATA)
    if not isinstance(chunks_raw, tuple) or type(frames) is not int:
        raise ValueError("YuE2 conditioning is missing chunk metadata")
    if any(
        not isinstance(item, tuple) or len(item) != 4 or any(type(part) is not int for part in item)
        for item in chunks_raw
    ):
        raise ValueError("YuE2 conditioning chunk metadata is invalid")
    chunks = cast("tuple[tuple[int, int, int, int], ...]", chunks_raw)
    if frames < 1:
        raise ValueError("YuE2 conditioning frame metadata is invalid")
    basic = make_conditioning_carrier(
        ConditioningSet((replace(record, extension_metadata=()),)), carrier.bindings
    )
    value = materialize_basic_conditioning(basic, device=device)
    return YuE2Conditioning(value.embeddings, value.pooled, chunks, frames)


class YuE2TextRuntime:
    def __init__(self, model: YuE2TextModel, tokenizer: Tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def generate_abc(
        self,
        *,
        style: Any,
        lyrics: Any,
        seed: Any,
        mode: Any,
        max_abc_tokens: Any,
        temperature: Any,
        top_p: Any,
        top_k: Any,
        repetition_penalty: Any,
        penalty_window: Any,
    ) -> str:
        prompt = prompt_tokens(self.tokenizer, str(style), str(lyrics), str(mode))
        ids, _truncated = self.model.generate(
            prompt.prefix,
            seed=int(seed),
            max_tokens=int(max_abc_tokens),
            phase="abc",
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
            penalty_window=int(penalty_window),
            min_tokens=min(32, int(max_abc_tokens)),
        )
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def generate_music(
        self,
        *,
        style: Any,
        lyrics: Any,
        abc: Any,
        seed: Any,
        mode: Any,
        max_duration: Any,
        temperature: Any,
        top_p: Any,
        top_k: Any,
        repetition_penalty: Any,
        cfg_scale: Any = None,
    ) -> YuE2Conditioning:
        abc_text = str(abc)
        selected_mode = str(mode) if abc_text.strip() else "off"
        prompt = prompt_tokens(self.tokenizer, str(style), str(lyrics), selected_mode, abc_text)
        abc_ids = () if selected_mode == "off" else prompt.abc_ids
        guidance = 1.01 if cfg_scale is None and selected_mode == "off" else 1.0
        if cfg_scale is not None:
            guidance = float(cfg_scale)
        prefix = (*prompt.prefix, *abc_ids, ABC_END, MUSIC_START)
        negative = (
            (*prompt.negative, MUSIC_START)
            if selected_mode == "off"
            else (*prompt.negative, 151_847, *abc_ids, ABC_END, MUSIC_START)
        )
        requested = max(1, round(float(max_duration) * FRAMES_PER_SECOND))
        budget = min(requested, CONTEXT - max(len(prefix), len(negative)))
        if budget < 1 or len(prefix) + 5 > CONTEXT:
            raise ValueError("YuE2 prompt leaves no room for music")
        semantic, _truncated = self.model.generate(
            prefix,
            seed=int(seed),
            max_tokens=budget,
            phase="semantic",
            negative=negative,
            cfg_scale=guidance,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
            penalty_window=50,
            min_tokens=min(200, budget),
            legacy_off=selected_mode == "off",
        )
        context, chunks = self.model.acoustic_conditioning(prefix, semantic)
        return YuE2Conditioning(context, None, chunks, len(semantic))

    def encode_text(self, text: str) -> YuE2Conditioning:
        return self.generate_music(
            style=text,
            lyrics="",
            abc="",
            seed=0,
            mode="off",
            max_duration=120.0,
            temperature=1.0,
            top_p=0.95,
            top_k=100,
            repetition_penalty=1.2,
            cfg_scale=1.01,
        )

    @staticmethod
    def text_conditioning_carrier(value: YuE2Conditioning) -> ConditioningCarrier:
        return yue2_conditioning_to_carrier(value)


class YuE2CodecRuntime:
    def __init__(self, model: AudioOobleckVAE) -> None:
        self.model = model

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.model.decode(latent)

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        return self.model.encode(audio)


def realize_yue2_component(
    plan: ComponentPlan[object],
    *,
    compute_dtype: torch.dtype,
    fp8_matmul: bool,
    attention_kernels: Mapping[AttentionRole, AttentionKernel],
) -> torch.nn.Module:
    builders: dict[str, Any] = {
        "diffusion": partial(
            YuE2AcousticModel,
            attention_kernel=attention_kernels["qwen"],
        ),
        "text": partial(
            YuE2TextModel,
            attention_kernel=attention_kernels["qwen"],
        ),
        "vae": AudioOobleckVAE,
    }
    module = _load_component(
        plan,
        builders[plan.component],
        compute_dtype=compute_dtype,
        fp8_matmul=fp8_matmul,
    )
    if plan.component == "text":
        source = load_tensors(plan.path, {"text_encoders.yue2_tokenizer_json"})
        encoded = source["text_encoders.yue2_tokenizer_json"]
        tokenizer = tokenizer_from_bytes(encoded.numpy().tobytes())
        module._dinkster_yue2_tokenizer = tokenizer
    return module


load_yue2_component = realize_yue2_component


def checkpoint_text_runtime(assembled: Any) -> YuE2TextRuntime:
    model = assembled.components["text"]
    tokenizer = getattr(model, "_dinkster_yue2_tokenizer", None)
    if not isinstance(model, YuE2TextModel) or not isinstance(tokenizer, Tokenizer):
        raise TypeError("YuE2 checkpoint has no native text model and tokenizer")
    return YuE2TextRuntime(model, tokenizer)


def checkpoint_codec(assembled: Any) -> YuE2CodecRuntime:
    model = assembled.components["vae"]
    if not isinstance(model, AudioOobleckVAE):
        raise TypeError("YuE2 checkpoint has no native AudioOobleck codec")
    return YuE2CodecRuntime(model)


class YuE2Denoiser:
    evaluator_identity = "dinkster.yue2.conditioning.v1"

    def __init__(self, model: YuE2AcousticModel, compute_dtype: torch.dtype) -> None:
        self.model = model
        self.compute_dtype = compute_dtype

    def prepare_conditioning(self, value: object, _role: GuidanceRole) -> YuE2Conditioning:
        if not isinstance(value, YuE2Conditioning):
            raise TypeError("YuE2 sampling requires YuE2 conditioning")
        return value

    @staticmethod
    def batchable(conditions: tuple[YuE2Conditioning, ...]) -> bool:
        return len(conditions) == 1

    def evaluate_conditioning(
        self, latent: torch.Tensor, sigma: float, condition: YuE2Conditioning
    ) -> torch.Tensor:
        if latent.shape[-1] != condition.frames:
            raise ValueError("YuE2 latent frames must match generated conditioning frames")
        context = condition.embeddings.to(latent.device, self.compute_dtype)
        if context.shape[0] == 1 and latent.shape[0] > 1:
            context = context.expand(latent.shape[0], -1, -1)
        timestep = latent.new_full((latent.shape[0],), 1.0 - sigma)
        velocity = self.model(
            latent.to(self.compute_dtype), timestep, context, condition.chunks
        ).float()
        return latent.float() - velocity * sigma

    def evaluate_conditioning_batch(
        self,
        latent: torch.Tensor,
        sigma: float,
        conditions: tuple[YuE2Conditioning, ...],
    ) -> tuple[torch.Tensor, ...]:
        return tuple(self.evaluate_conditioning(latent, sigma, item) for item in conditions)


@dataclass(frozen=True)
class _YuE2DiffusionAssembly:
    diffusion: YuE2AcousticModel
    family: ModelFamily = YUE2_FAMILY
    attention_status: Mapping[object, object] = field(default_factory=dict)
    compute: torch.dtype = torch.bfloat16
    _storage_dtype_follows_compute: bool = field(default=False, repr=False, compare=False)
    _component_compute_dtypes: Mapping[str, torch.dtype] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False
    )

    def compute_dtype(self, component: str) -> torch.dtype | None:
        return self.compute if component == "diffusion" else None


def _validate_latent(latent: torch.Tensor) -> None:
    if latent.ndim != 3 or latent.shape[1] != LATENT_CHANNELS:
        raise ValueError(f"YuE2 latent must be [batch,{LATENT_CHANNELS},frames]")


def yue2_denoiser(
    runtime: object,
    compute_dtype: torch.dtype,
    context: SamplingAdapterContext,
) -> SamplingDenoiserExecution:
    if context.options:
        raise ValueError("YuE2 sampling does not accept adapter options")
    owner = cast("YuE2DiffusionRuntime", runtime)
    return SamplingDenoiserExecution(
        cast(
            "SamplingDenoiserAdapter",
            YuE2Denoiser(owner.assembled.diffusion, compute_dtype),
        )
    )


def _device(runtime: object) -> torch.device:
    owner = cast("YuE2DiffusionRuntime", runtime)
    return module_compute_device(owner.assembled.diffusion)


def _compute_dtype(runtime: object) -> torch.dtype:
    owner = cast("YuE2DiffusionRuntime", runtime)
    return owner.assembled.compute


class YuE2DiffusionRuntime(SingleStreamSamplingRuntime):
    streamed_residency_components = frozenset()
    sampling_error = ValueError
    sampling_compute_dtype = torch.bfloat16
    supports_denoised_capture = True
    sampling_execution_registration = SamplingExecutionRegistration(
        latent=SingleStreamLatentAdapter(_validate_latent),
        denoiser=yue2_denoiser,
        device=_device,
        compute_dtype=_compute_dtype,
        flow=True,
    )

    def __init__(
        self,
        diffusion: YuE2AcousticModel,
        *,
        runtime_identity: str,
        compute_dtype: torch.dtype = torch.bfloat16,
        sampler_registry: Registry[SamplerDescriptor[Any]] | None = None,
        scheduler_registry: Registry[SchedulerDescriptor] | None = None,
        guidance_executor: GuidanceExecutor | None = None,
        **_options: object,
    ) -> None:
        self.assembled = _YuE2DiffusionAssembly(diffusion, compute=compute_dtype)
        self.attention_status = self.assembled.attention_status
        self._runtime_identity = runtime_identity
        self._samplers = torch_sampler_registry(sampler_registry)
        self._schedulers = (
            torch_scheduler_registry() if scheduler_registry is None else scheduler_registry
        )
        self._guidance = guidance_executor

    @property
    def family(self) -> ModelFamily:
        return self.assembled.family

    @property
    def runtime_identity(self) -> str:
        return self._runtime_identity

    def _sampling_sigma_space(self, sampling_shift: float | None) -> SigmaSpace:
        del sampling_shift
        return FlowSigmas()

    sample_custom = sampling_execution


__all__ = [
    "YuE2Conditioning",
    "YuE2DiffusionRuntime",
    "YuE2TextRuntime",
    "checkpoint_codec",
    "checkpoint_text_runtime",
    "materialize_yue2_conditioning",
    "realize_yue2_component",
    "yue2_conditioning_to_carrier",
    "yue2_denoiser",
]
