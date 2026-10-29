"""Wan node execution over the public inference resource seams."""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
import torch.nn.functional as functional
from dinkster_api.v1 import GIBIBYTE, AssetRef
from dinkster_inference import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    WAN21_CAUSAL_INITIAL_LATENT_KEY,
    WAN21_CODEC,
    WAN21_I2V_14B,
    WAN21_T2V_14B,
    WAN22_WANDANCER_14B,
    WAV2VEC2_CHINESE_BASE,
    WAV2VEC2_COMPONENT_FAMILY_ID,
    WHISPER_LARGE_V3_COMPONENT_FAMILY_ID,
    ApplicationChain,
    ComponentApplication,
    ComponentBinding,
    ConditioningCarrier,
    InferenceCodecHandle,
    InferenceComponentHandle,
    InferenceRuntimeHandle,
    MultiStreamLatent,
    PercentRange,
    SizedTensor,
    Wan21Animate2Settings,
    Wan22DancerSettings,
    bind_component_conditioning,
    build_runtime_identity_from_facts,
    extend_runtime_identity,
    load_safetensors_header,
    plan_wan21_uni3c,
    plan_wav2vec2_component,
    plan_whisper_large_v3_component,
    require_inference_codec_handle,
    require_inference_component_handle,
    require_inference_runtime_handle,
    runtime_component_identity,
    select_builtin_sampler,
    split_component_conditioning,
    wav2vec2_component_runtime_identity,
    whisper_large_v3_component_runtime_identity,
)
from dinkster_inference_torch import (
    Wan21CausalDiffusionRuntime,
    Wan21InfiniteTalkExecution,
    Wan21MultiTalk,
    Wan21MultiTalkExecution,
    Wan21Runtime,
    Wan21Uni3C,
    Wan21Uni3CExecution,
    Wan22DancerModel,
    Wav2Vec2Model,
    WhisperLargeV3Model,
    assemble_wan21_uni3c,
    component_publisher,
    compose_wan21_animate2_conditioning,
    compose_wan21_animate_conditioning,
    compose_wan21_humo_conditioning,
    compose_wan21_i2v_conditioning,
    compose_wan21_scail_conditioning,
    compose_wan22_dancer_conditioning,
    compose_wan22_s2v_conditioning,
    load_wav2vec2_component,
    load_whisper_large_v3_component,
    ltx_audio_resample,
    wan21_multitalk_tensor_digest,
    wan21_uni3c_tensor_digest,
)
from dinkster_inference_torch.wan21_model import Wan21Model
from dinkster_inference_torch.wan21_vae import LATENTS_MEAN

from .wandancer_audio import (
    RESAMPLE_SAMPLE_RATE,
    encode_wandancer_audio_features,
    plan_wandancer_keyframe_list,
    plan_wandancer_keyframes,
)


class _AssetRef(Protocol):
    digest: str
    size: int

    def local_path(self) -> Path: ...


@dataclass(frozen=True, slots=True)
class WanS2VAudioOutput:
    """One validated Wav2Vec2 layer stack for Wan S2V conditioning."""

    layers: tuple[torch.Tensor, ...]
    audio_samples: int

    def __post_init__(self) -> None:
        if type(self.layers) is not tuple or len(self.layers) != 25:
            raise ValueError("Wan S2V audio output must contain exactly 25 layer tensors")
        first_shape: tuple[int, ...] | None = None
        for layer in self.layers:
            if (
                type(layer) is not torch.Tensor
                or not layer.is_floating_point()
                or layer.ndim != 3
                or layer.shape[0] != 1
                or layer.shape[2] != 1024
                or any(size <= 0 for size in layer.shape)
            ):
                raise ValueError("Wan S2V audio layers must be nonempty [1,time,1024] tensors")
            if layer.device.type != "cpu":
                raise ValueError("Wan S2V audio layers must use CPU-owned storage")
            if first_shape is None:
                first_shape = tuple(layer.shape)
            elif tuple(layer.shape) != first_shape:
                raise ValueError("Wan S2V audio layers must share one shape")
        if type(self.audio_samples) is not int or self.audio_samples <= 0:
            raise ValueError("Wan S2V audio output must record a positive sample count")


@dataclass(frozen=True, slots=True)
class WanHumoAudioOutput:
    """One validated Whisper Large v3 layer stack for Wan HuMo conditioning."""

    layers: tuple[torch.Tensor, ...]
    audio_samples: int

    def __post_init__(self) -> None:
        if type(self.layers) is not tuple or len(self.layers) != 33:
            raise ValueError("Wan HuMo audio output must contain exactly 33 layer tensors")
        first_shape: tuple[int, ...] | None = None
        for layer in self.layers:
            if (
                type(layer) is not torch.Tensor
                or not layer.is_floating_point()
                or layer.ndim != 3
                or layer.shape[0] != 1
                or layer.shape[2] != 1280
                or any(size <= 0 for size in layer.shape)
            ):
                raise ValueError("Wan HuMo audio layers must be nonempty [1,time,1280] tensors")
            if layer.device.type != "cpu":
                raise ValueError("Wan HuMo audio layers must use CPU-owned storage")
            if first_shape is None:
                first_shape = tuple(layer.shape)
            elif tuple(layer.shape) != first_shape:
                raise ValueError("Wan HuMo audio layers must share one shape")
        if type(self.audio_samples) is not int or self.audio_samples <= 0:
            raise ValueError("Wan HuMo audio output must record a positive sample count")


@dataclass(frozen=True, slots=True)
class WanInfiniteTalkAudioOutput:
    """One validated Wav2Vec2 Chinese base layer stack for InfiniteTalk."""

    layers: tuple[torch.Tensor, ...]
    audio_samples: int

    def __post_init__(self) -> None:
        if type(self.layers) is not tuple or len(self.layers) != 13:
            raise ValueError("Wan InfiniteTalk audio output must contain exactly 13 layer tensors")
        first_shape: tuple[int, ...] | None = None
        for layer in self.layers:
            if (
                type(layer) is not torch.Tensor
                or not layer.is_floating_point()
                or layer.ndim != 3
                or layer.shape[0] != 1
                or layer.shape[2] != 768
                or any(size <= 0 for size in layer.shape)
            ):
                raise ValueError(
                    "Wan InfiniteTalk audio layers must be nonempty [1,time,768] tensors"
                )
            if layer.device.type != "cpu":
                raise ValueError("Wan InfiniteTalk audio layers must use CPU-owned storage")
            if first_shape is None:
                first_shape = tuple(layer.shape)
            elif tuple(layer.shape) != first_shape:
                raise ValueError("Wan InfiniteTalk audio layers must share one shape")
        if type(self.audio_samples) is not int or self.audio_samples <= 0:
            raise ValueError("Wan InfiniteTalk audio output must record a positive sample count")


@dataclass(frozen=True, slots=True)
class WanDancerAudioOutput:
    """One CPU-owned WanDancer music feature sequence and its runtime settings."""

    audio_feature: torch.Tensor
    settings: Wan22DancerSettings

    def __post_init__(self) -> None:
        feature = self.audio_feature
        if (
            type(feature) is not torch.Tensor
            or not feature.is_floating_point()
            or feature.layout != torch.strided
            or feature.ndim != 3
            or feature.shape[0] != 1
            or feature.shape[1] <= 0
            or feature.shape[2] != 35
        ):
            raise ValueError("WanDancer audio feature must be nonempty floating [1,time,35]")
        if feature.device.type != "cpu" or not feature.is_contiguous():
            raise ValueError("WanDancer audio feature must use contiguous CPU-owned storage")
        if not bool(torch.isfinite(feature).all()):
            raise ValueError("WanDancer audio feature must contain only finite values")
        if type(self.settings) is not Wan22DancerSettings:
            raise TypeError("WanDancer audio settings must be exact Wan22DancerSettings")


def _integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _nonnegative_float(value: object, name: str, *, maximum: float) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative float")
    if value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _image(value: object, name: str) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    elif type(value) is torch.Tensor:
        tensor = value.detach()
    else:
        raise TypeError(f"{name} must be a numpy array or exact torch.Tensor")
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if (
        tensor.ndim != 4
        or tensor.shape[-1] not in (1, 3, 4)
        or any(size <= 0 for size in tensor.shape)
    ):
        raise ValueError(f"{name} must be nonempty HWC or BHWC with 1, 3, or 4 channels")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must contain floating-point values")
    tensor = tensor.to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(tensor).all()) or bool(torch.any((tensor < 0.0) | (tensor > 1.0))):
        raise ValueError(f"{name} values must be finite and in [0, 1]")
    if tensor.shape[-1] == 1:
        return tensor.expand(*tensor.shape[:-1], 3).contiguous()
    return tensor[..., :3].contiguous()


def _mask(value: object, name: str) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    elif type(value) is torch.Tensor:
        tensor = value.detach()
    else:
        raise TypeError(f"{name} must be a numpy array or exact torch.Tensor")
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or any(size <= 0 for size in tensor.shape):
        raise ValueError(f"{name} must be nonempty HW or BHW")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must contain floating-point values")
    tensor = tensor.to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(tensor).all()) or bool(torch.any((tensor < 0.0) | (tensor > 1.0))):
        raise ValueError(f"{name} values must be finite and in [0, 1]")
    return tensor


def _common_upscale(
    samples: torch.Tensor,
    width: int,
    height: int,
    mode: str,
) -> torch.Tensor:
    """ComfyUI common_upscale's center-crop and flattening behavior."""

    original_shape: tuple[int, ...] = tuple(samples.shape)
    if len(original_shape) < 4:
        raise ValueError("upscale input must have at least four dimensions")
    original_batch = original_shape[0]
    original_channels = original_shape[1]
    if len(original_shape) > 4:
        samples = samples.reshape(
            samples.shape[0], samples.shape[1], -1, samples.shape[-2], samples.shape[-1]
        )
        samples = samples.movedim(2, 1)
        samples = samples.reshape(-1, original_shape[1], original_shape[-2], original_shape[-1])
    old_width = samples.shape[-1]
    old_height = samples.shape[-2]
    old_aspect = old_width / old_height
    new_aspect = width / height
    x = 0
    y = 0
    if old_aspect > new_aspect:
        x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
    cropped = samples.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
    output = functional.interpolate(cropped, size=(height, width), mode=mode)
    if len(original_shape) == 4:
        return output
    output = output.reshape((original_batch, -1, original_channels, height, width))
    return output.movedim(2, 1).reshape(original_shape[:-2] + (height, width))


def _resize_image(value: torch.Tensor, width: int, height: int) -> torch.Tensor:
    return _common_upscale(value.movedim(-1, 1), width, height, "area").movedim(1, -1)


def _codec_content(frames: torch.Tensor, device: object) -> torch.Tensor:
    return frames[..., :3].permute(3, 0, 1, 2).unsqueeze(0).to(device=cast("Any", device))


def _encoded_tensor(codec: InferenceCodecHandle[SizedTensor], frames: torch.Tensor) -> torch.Tensor:
    encoded = codec.encode_content(_codec_content(frames, codec.load_device))
    if type(encoded) is not torch.Tensor:
        raise TypeError("vae encode_content must return an exact torch.Tensor")
    return encoded


def execute_empty_ar_video_latent(
    *, width: int, height: int, length: int, batch_size: int
) -> Mapping[str, object]:
    width = _integer(width, "width", minimum=16, maximum=8192)
    height = _integer(height, "height", minimum=16, maximum=8192)
    length = _integer(length, "length", minimum=1, maximum=1024)
    batch_size = _integer(batch_size, "batch_size", minimum=1, maximum=64)
    if width % 16 or height % 16 or (length - 1) % 4:
        raise ValueError(
            "width/height must be multiples of 16 and length must be 1 plus a multiple of 4"
        )
    latent = torch.zeros((batch_size, 16, (length - 1) // 4 + 1, height // 8, width // 8))
    return {"latent": {"samples": latent}}


def execute_sampler_ar_video(*, num_frame_per_block: int) -> Mapping[str, object]:
    count = _integer(
        num_frame_per_block,
        "num_frame_per_block",
        minimum=1,
        maximum=64,
    )
    return {"sampler": select_builtin_sampler("dinkster.ar_video", num_frame_per_block=count)}


def execute_ar_video_i2v(
    *,
    model: object,
    vae: object,
    start_image: object,
    width: int,
    height: int,
    length: int,
    batch_size: int,
) -> Mapping[str, object]:
    model_handle = require_inference_runtime_handle(model, "model")
    runtime = model_handle.runtime
    if not isinstance(runtime, (Wan21Runtime, Wan21CausalDiffusionRuntime)) or (
        runtime.assembled.diffusion.config.model_variant != "causal_ar"
    ):
        raise ValueError("model must expose a Wan 2.1 CausalAR runtime")
    latent_output = execute_empty_ar_video_latent(
        width=width,
        height=height,
        length=length,
        batch_size=batch_size,
    )["latent"]
    codec = require_inference_codec_handle(vae, "vae")
    if codec.descriptor != WAN21_CODEC:
        raise ValueError("vae must expose the Wan 2.1 codec descriptor")
    image = _image(start_image, "start_image")[:1]
    resized = _common_upscale(
        image.movedim(-1, 1),
        width,
        height,
        "bilinear",
    ).movedim(1, -1)
    with codec.stage(), torch.inference_mode():
        initial = _encoded_tensor(codec, resized)
    if initial.ndim != 5 or initial.shape[0:3] != (1, 16, 1):
        raise ValueError("vae start image output must have shape [1,16,1,H,W]")
    expected_spatial = (height // 8, width // 8)
    if initial.shape[-2:] != expected_spatial:
        raise ValueError(f"vae start image output must have spatial shape {expected_spatial}")
    latent = dict(cast("Mapping[str, object]", latent_output))
    latent[WAN21_CAUSAL_INITIAL_LATENT_KEY] = initial.detach().to(device="cpu").contiguous()
    return {"model": model_handle, "latent": latent}


def _animate_runtime(handle: InferenceRuntimeHandle) -> Wan21Runtime:
    runtime = handle.runtime
    if not isinstance(runtime, Wan21Runtime):
        raise TypeError("model runtime must be a Wan21Runtime")
    config = runtime.assembled.diffusion.config
    if config.model_variant != "animate":
        raise ValueError("model runtime must be the Wan 2.2 Animate profile")
    return runtime


def _animate2_runtime(handle: InferenceRuntimeHandle) -> Wan21Runtime:
    runtime = handle.runtime
    if not isinstance(runtime, Wan21Runtime):
        raise TypeError("model runtime must be a Wan21Runtime")
    config = runtime.assembled.diffusion.config
    if config.model_variant != "animate2":
        raise ValueError("model runtime must be the Wan 2.1 Animate2 profile")
    return runtime


def _scail_runtime(handle: InferenceRuntimeHandle) -> Wan21Runtime:
    runtime = handle.runtime
    if not isinstance(runtime, Wan21Runtime):
        raise TypeError("model runtime must be a Wan21Runtime")
    config = runtime.assembled.diffusion.config
    if config.model_variant not in ("scail", "scail2"):
        raise ValueError("model runtime must be a Wan 2.1 SCAIL or SCAIL2 profile")
    return runtime


def _asset(value: object, name: str) -> _AssetRef:
    digest = getattr(value, "digest", None)
    size = getattr(value, "size", None)
    local_path = getattr(value, "local_path", None)
    if type(digest) is not str or type(size) is not int or not callable(local_path):
        raise TypeError(f"{name} must be a resolved asset reference")
    return cast("_AssetRef", value)


def execute_load_wav2vec2_audio_encoder(*, audio_encoder: object) -> Mapping[str, object]:
    if type(audio_encoder) is not AssetRef:
        raise TypeError("audio_encoder must be an exact AssetRef")
    path = audio_encoder.local_path()
    source = load_safetensors_header(
        path,
        asset_digest=audio_encoder.digest,
        asset_size=audio_encoder.size,
    )
    source_keys = set(source.keys())
    is_wav2vec2 = "wav2vec2.feature_extractor.conv_layers.0.conv.weight" in source_keys
    is_whisper = "model.encoder.conv1.weight" in source_keys
    if is_wav2vec2 == is_whisper:
        raise ValueError("audio_encoder must be an exact supported Wav2Vec2 or Whisper Large v3")
    if is_wav2vec2:
        plan = plan_wav2vec2_component(source, path=path)
        identity = wav2vec2_component_runtime_identity(plan, FLOAT16)
        loaded = load_wav2vec2_component(
            path,
            asset=audio_encoder,
            expected_identity=identity,
            compute_dtype=torch.float16,
        )
    else:
        plan = plan_whisper_large_v3_component(source, path=path)
        identity = whisper_large_v3_component_runtime_identity(plan, FLOAT32)
        loaded = load_whisper_large_v3_component(
            path,
            asset=audio_encoder,
            expected_identity=identity,
            compute_dtype=torch.float32,
        )
    handle = component_publisher().publish(loaded.module, resource_identity=identity)
    return {"audio_encoder": handle}


def _wan_audio_component(value: object) -> tuple[InferenceComponentHandle, str]:
    handle = require_inference_component_handle(value, "audio_encoder")
    for role, family_id in (
        ("wav2vec2-large", WAV2VEC2_COMPONENT_FAMILY_ID),
        ("whisper-large-v3", WHISPER_LARGE_V3_COMPONENT_FAMILY_ID),
    ):
        try:
            ComponentBinding(role, family_id, handle.resource_identity)
        except (TypeError, ValueError):
            continue
        return handle, family_id
    raise TypeError("audio_encoder must be native Wav2Vec2 or Whisper Large v3")


def _audio_input(value: object) -> tuple[torch.Tensor, int]:
    if not isinstance(value, Mapping) or set(value) != {"waveform", "sample_rate"}:
        raise TypeError("audio must contain exactly waveform and sample_rate")
    waveform = value["waveform"]
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    elif type(waveform) is torch.Tensor:
        waveform = waveform.detach()
    else:
        raise TypeError("audio waveform must be a numpy array or exact torch.Tensor")
    if (
        waveform.ndim != 3
        or waveform.shape[0] != 1
        or any(size <= 0 for size in waveform.shape)
        or not waveform.is_floating_point()
    ):
        raise ValueError("audio waveform must be nonempty floating [1,channels,samples]")
    if not bool(torch.isfinite(waveform).all()):
        raise ValueError("audio waveform must contain only finite values")
    sample_rate = value["sample_rate"]
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("audio sample_rate must be a positive integer")
    return waveform.to(device="cpu", dtype=torch.float32).contiguous(), sample_rate


def execute_encode_wav2vec2_audio(*, audio_encoder: object, audio: object) -> Mapping[str, object]:
    handle, family_id = _wan_audio_component(audio_encoder)
    waveform, sample_rate = _audio_input(audio)
    if sample_rate != 16_000:
        waveform = ltx_audio_resample(waveform, sample_rate, 16_000)
    with handle.stage(), torch.inference_mode():
        model = handle.component
        if family_id == WAV2VEC2_COMPONENT_FAMILY_ID and type(model) is Wav2Vec2Model:
            _encoded, layers = model(
                waveform.to(device=cast("Any", handle.load_device), dtype=torch.float16)
            )
            if model.config is WAV2VEC2_CHINESE_BASE:
                output_type = WanInfiniteTalkAudioOutput
            else:
                output_type = WanS2VAudioOutput
        elif (
            family_id == WHISPER_LARGE_V3_COMPONENT_FAMILY_ID and type(model) is WhisperLargeV3Model
        ):
            _encoded, layers = model(
                waveform.to(device=cast("Any", handle.load_device), dtype=torch.float32)
            )
            output_type = WanHumoAudioOutput
        else:
            raise TypeError("audio_encoder does not contain native Wav2Vec2 or Whisper Large v3")
    cpu_layers = tuple(layer.detach().to(device="cpu").contiguous() for layer in layers)
    return {"audio_encoder_output": output_type(cpu_layers, int(waveform.shape[2]))}


def _wan21_uni3c_resource_identity(component_plan: Any, model_digest: str) -> str:
    return extend_runtime_identity(
        build_runtime_identity_from_facts(
            "dinkster.wan21",
            runtime_component_identity("dinkster.wan21", (component_plan,)),
            diffusion_dtype=BFLOAT16.name,
            text_dtype="unloaded",
            vae_dtype="unloaded",
            fp8_matmul=False,
            runtime_facts=component_plan.runtime_facts,
        ),
        (f"resource={model_digest}",),
    )


def execute_load_wan21_uni3c(*, model_patch: object) -> Mapping[str, object]:
    asset = _asset(model_patch, "model_patch")
    source = load_safetensors_header(
        asset.local_path(),
        asset_digest=asset.digest,
        asset_size=asset.size,
    )
    plan = plan_wan21_uni3c(source, asset_digest=asset.digest)
    assembled = assemble_wan21_uni3c(plan, compute_dtype=torch.bfloat16)
    identity = _wan21_uni3c_resource_identity(plan.patch, assembled.resource_digest)
    handle = component_publisher().publish(assembled.patch, resource_identity=identity)
    return {"patch": handle}


def _wan21_uni3c_component(value: object) -> InferenceComponentHandle:
    handle = require_inference_component_handle(value, "patch")
    try:
        ComponentBinding("wan21_uni3c", "dinkster.wan21", handle.resource_identity)
    except (TypeError, ValueError) as error:
        raise TypeError("patch must be a native Wan 2.1 Uni3C component") from error
    return handle


def _wan21_uni3c_runtime(value: object) -> Wan21Runtime:
    if isinstance(value, ApplicationChain):
        raise ValueError("Uni3C cannot be combined with another model intervention")
    handle = require_inference_runtime_handle(value, "model")
    runtime = handle.runtime
    if type(runtime) is not Wan21Runtime:
        raise TypeError("model runtime must be the exact native Wan21Runtime")
    diffusion = runtime.assembled.diffusion
    if type(diffusion) is not Wan21Model or (
        diffusion.config is not WAN21_T2V_14B and diffusion.config is not WAN21_I2V_14B
    ):
        raise ValueError("Uni3C supports only exact base Wan 2.1 T2V or I2V 14B")
    return runtime


def _finite_float(value: object, name: str, *, minimum: float, maximum: float) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a number")
    resolved = float(cast("int | float", value))
    if not math.isfinite(resolved) or not minimum <= resolved <= maximum:
        raise ValueError(f"{name} must be finite and within [{minimum}, {maximum}]")
    return resolved


def _rgb_video(value: object) -> torch.Tensor:
    raw_shape = getattr(value, "shape", None)
    if not isinstance(raw_shape, tuple):
        raise ValueError("render_video must be RGB HWC or BHWC")
    shape = cast("tuple[int, ...]", raw_shape)
    if len(shape) not in (3, 4) or shape[-1] != 3:
        raise ValueError("render_video must be RGB HWC or BHWC")
    return _image(value, "render_video").clone()


def _append_application(model: object, application: ComponentApplication) -> ApplicationChain:
    return (
        model.append(application)
        if isinstance(model, ApplicationChain)
        else ApplicationChain(model, (application,))
    )


def _prepare_uni3c_render(
    runtime: Wan21Runtime,
    codec: InferenceCodecHandle[SizedTensor],
    render_video: torch.Tensor,
    video: torch.Tensor,
) -> torch.Tensor:
    target_frames = (video.shape[2] - 1) * 4 + 1
    frames = render_video[:target_frames]
    if frames.shape[0] < target_frames:
        frames = torch.cat(
            (frames, frames[-1:].expand(target_frames - frames.shape[0], -1, -1, -1)),
            dim=0,
        )
    frames = _common_upscale(
        frames.movedim(-1, 1),
        video.shape[4] * 8,
        video.shape[3] * 8,
        "bilinear",
    ).movedim(1, -1)
    with codec.stage(), torch.inference_mode():
        encoded = _encoded_tensor(codec, frames)
    if encoded.ndim != 5 or encoded.shape[1] != 16:
        raise ValueError("vae render output must be [batch,16,T,H,W]")
    normalized = runtime.assembled.vae.process_in(encoded)
    expected = (1, 16, *video.shape[2:])
    if tuple(normalized.shape) != expected:
        raise ValueError(f"vae render output must have shape {expected}")
    return normalized.detach().to(device="cpu").contiguous()


def execute_apply_wan21_uni3c(
    *,
    model: object,
    patch: object,
    vae: object,
    render_video: object,
    strength: float,
    start_percent: float,
    end_percent: float,
) -> Mapping[str, object]:
    _wan21_uni3c_runtime(model)
    handle = _wan21_uni3c_component(patch)
    codec = require_inference_codec_handle(vae, "vae")
    if codec.descriptor != WAN21_CODEC:
        raise ValueError("vae must expose the Wan 2.1 codec descriptor")
    render = _rgb_video(render_video)
    strength = _finite_float(strength, "strength", minimum=-10.0, maximum=10.0)
    start = _finite_float(start_percent, "start_percent", minimum=0.0, maximum=1.0)
    end = _finite_float(end_percent, "end_percent", minimum=0.0, maximum=1.0)
    window = PercentRange(start, end)
    render_source_digest = wan21_uni3c_tensor_digest(render)

    def materialize(
        runtime_value: object,
        live_component: object,
        latent_value: object,
    ) -> Mapping[str, object]:
        if type(runtime_value) is not Wan21Runtime:
            raise TypeError("Uni3C sampling runtime must be the exact native Wan21Runtime")
        diffusion = runtime_value.assembled.diffusion
        if type(diffusion) is not Wan21Model or (
            diffusion.config is not WAN21_T2V_14B and diffusion.config is not WAN21_I2V_14B
        ):
            raise ValueError("Uni3C supports only exact base Wan 2.1 T2V or I2V 14B")
        if type(live_component) is not Wan21Uni3C:
            raise TypeError("patch must contain the exact maintained Wan 2.1 Uni3C model")
        if type(latent_value) is not MultiStreamLatent:
            raise TypeError("Uni3C sampling latent must be an exact MultiStreamLatent")
        streams = cast("MultiStreamLatent[object]", latent_value)
        if streams.roles != ("video",):
            raise ValueError("Uni3C requires the exact latent stream role 'video'")
        video = streams.by_role("video")
        if (
            type(video) is not torch.Tensor
            or not video.is_floating_point()
            or video.ndim != 5
            or video.shape[1] != 16
            or any(size < 1 for size in video.shape)
        ):
            raise ValueError("Uni3C target video must be floating [batch,16,T,H,W]")
        render_latent = _prepare_uni3c_render(runtime_value, codec, render, video)
        model_digest = live_component.resource_digest
        if model_digest is None:
            raise RuntimeError("Wan 2.1 Uni3C model has no assembly provenance")
        render_digest = wan21_uni3c_tensor_digest(render_latent)
        return {
            "uni3c": Wan21Uni3CExecution(
                live_component,
                render_latent,
                strength,
                window,
                model_digest,
                render_digest,
            )
        }

    identity = extend_runtime_identity(
        handle.resource_identity,
        (
            "role=wan21_uni3c",
            f"render={render_source_digest}",
            f"codec={codec.resource_identity}",
            f"strength={strength.hex()}",
            f"start={start.hex()}",
            f"end={end.hex()}",
        ),
    )
    application = ComponentApplication(
        "dinkster.wan21",
        "diffusion",
        handle,
        identity,
        materialize,
        resident_dependencies=(codec,),
    )
    return {"model": _append_application(model, application)}


def _scail_mask(value: torch.Tensor) -> torch.Tensor:
    """Convert an RGB identity mask into SCAIL2's 28-channel latent layout."""

    frames, height, width, _channels = value.shape
    threshold = 225.0 / 255.0
    mask = value.movedim(-1, 1).float()
    red = (mask[:, 0:1] > threshold).float()
    green = (mask[:, 1:2] > threshold).float()
    blue = (mask[:, 2:3] > threshold).float()
    not_red = 1.0 - red
    not_green = 1.0 - green
    not_blue = 1.0 - blue
    binary = torch.cat(
        (
            red * green * blue,
            red * not_green * not_blue,
            not_red * green * not_blue,
            not_red * not_green * blue,
            red * green * not_blue,
            red * not_green * blue,
            not_red * green * blue,
        ),
        dim=1,
    )
    latent_height = height
    latent_width = width
    for _ in range(3):
        latent_height = (latent_height + 1) // 2
        latent_width = (latent_width + 1) // 2
    binary = functional.interpolate(binary, size=(latent_height, latent_width), mode="area")
    latent_frames = ((frames - 1) // 4) + 1
    padded = torch.cat((binary[:1].repeat(4, 1, 1, 1), binary[1:]), dim=0)
    return padded.view(latent_frames, 28, latent_height, latent_width).movedim(0, 1).unsqueeze(0)


def _s2v_audio_embedding(
    output: WanS2VAudioOutput | None,
    *,
    latent_frames: int,
    frame_offset: int,
) -> torch.Tensor | None:
    if output is None:
        return None
    if type(output) is not WanS2VAudioOutput:
        raise TypeError("audio_encoder_output must be an exact WanS2VAudioOutput")
    features = torch.cat(output.layers, dim=0)
    output_frames = int(features.shape[1] / 50.0 * 30.0)
    if output_frames <= 0:
        raise ValueError("audio_encoder_output is too short for Wan S2V")
    features = functional.interpolate(
        features.transpose(1, 2),
        size=output_frames,
        mode="linear",
        align_corners=True,
    ).transpose(1, 2)
    batch_frames = latent_frames * 4
    minimum_batches = int(output_frames / (batch_frames * (30.0 / 16.0))) + 1
    bucket_frames = minimum_batches * batch_frames
    padded_frames = math.ceil(bucket_frames / 16.0 * 30.0)
    sample_times = np.linspace(0.0, bucket_frames / 16.0, bucket_frames, endpoint=False)
    indices = np.round(sample_times * 30.0).astype(np.int64)
    indices = np.clip(indices, 0, padded_frames - 1)
    valid = torch.from_numpy(indices < output_frames)
    gather = torch.from_numpy(np.minimum(indices, output_frames - 1))
    bucket = features.index_select(1, gather)
    bucket[:, ~valid] = 0.0
    embedding = bucket.permute(0, 2, 1).unsqueeze(0)
    embedding = embedding[..., frame_offset : frame_offset + batch_frames]
    if embedding.shape[3] == 0:
        return None
    if embedding.shape[3] < batch_frames:
        embedding = functional.pad(embedding, (0, batch_frames - embedding.shape[3]))
    return embedding


def _humo_audio_embedding(
    output: WanHumoAudioOutput | None,
    *,
    latent_frames: int,
) -> torch.Tensor:
    if output is None:
        return torch.zeros((1, latent_frames, 8, 5, 1280), dtype=torch.float32)
    if type(output) is not WanHumoAudioOutput:
        raise TypeError("audio_encoder_output must be an exact WanHumoAudioOutput")
    features = torch.stack(output.layers, dim=2)
    usable_frames = (output.audio_samples // 640) * 2
    features = features[:, :usable_frames]
    if features.shape[1] == 0:
        raise ValueError("audio_encoder_output is too short for Wan HuMo")
    groups = (
        features[:, :, 0:8].mean(dim=2),
        features[:, :, 8:16].mean(dim=2),
        features[:, :, 16:24].mean(dim=2),
        features[:, :, 24:32].mean(dim=2),
        features[:, :, 32],
    )
    output_frames = int(features.shape[1] / 50.0 * 25.0)
    if output_frames == 0:
        raise ValueError("audio_encoder_output is too short for Wan HuMo")
    interpolated = tuple(
        functional.interpolate(
            group.transpose(1, 2),
            size=output_frames,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        for group in groups
    )
    audio = torch.stack(interpolated, dim=2)[0]
    zero = audio.new_zeros((5, 1280))
    prefix = audio.new_zeros((3, 5, 1280))
    windows: list[torch.Tensor] = []
    for latent_index in range(latent_frames):
        if latent_index == 0:
            indices = range(-2, 3)
            window = torch.stack(
                [audio[index] if 0 <= index < audio.shape[0] else zero for index in indices]
            )
            window = torch.cat((prefix, window), dim=0)
        else:
            start = 1 + 4 * (latent_index - 1) - 2
            stop = 1 + 4 * latent_index + 2
            window = torch.stack(
                [
                    audio[index] if 0 <= index < audio.shape[0] else zero
                    for index in range(start, stop)
                ]
            )
        windows.append(window)
    return torch.stack(windows).unsqueeze(0).contiguous()


def execute_wan21_humo(
    *,
    positive: object,
    negative: object,
    vae: object,
    width: int,
    height: int,
    length: int,
    batch_size: int,
    audio_encoder_output: object | None = None,
    ref_image: object | None = None,
) -> Mapping[str, object]:
    width = _integer(width, "width", minimum=16, maximum=16384)
    height = _integer(height, "height", minimum=16, maximum=16384)
    length = _integer(length, "length", minimum=1, maximum=16384)
    batch_size = _integer(batch_size, "batch_size", minimum=1, maximum=4096)
    if width % 16 or height % 16:
        raise ValueError("width and height must be multiples of 16")
    if (length - 1) % 4:
        raise ValueError("length must equal 1 plus a multiple of 4")
    if type(positive) is not ConditioningCarrier or type(negative) is not ConditioningCarrier:
        raise TypeError("positive and negative must be exact ConditioningCarrier values")
    positive_text, positive_binding = split_component_conditioning(positive)
    negative_text, negative_binding = split_component_conditioning(negative)
    if positive_binding != negative_binding:
        raise ValueError("Wan HuMo conditioning lanes must share one component binding")
    if positive_binding is not None and positive_binding.role != "umt5xxl":
        raise ValueError("Wan HuMo conditioning must come from a Wan UMT5-XXL component")

    codec = require_inference_codec_handle(vae, "vae")
    if codec.descriptor != WAN21_CODEC:
        raise ValueError("vae must expose the Wan 2.1 codec descriptor")
    latent_frames = ((length - 1) // 4) + 1
    latent_height = height // 8
    latent_width = width // 8
    reference = torch.zeros((1, 16, 1, latent_height, latent_width), dtype=torch.float32)
    if ref_image is not None:
        pixels = _image(ref_image, "ref_image")[:1]
        with codec.stage(), torch.inference_mode():
            reference = _s2v_encode_media(
                codec,
                pixels,
                width=width,
                height=height,
                name="reference",
            )
        del pixels
        if reference.shape != (1, 16, 1, latent_height, latent_width):
            raise ValueError(
                f"vae reference output must have shape {(1, 16, 1, latent_height, latent_width)}"
            )

    audio = _humo_audio_embedding(
        None if audio_encoder_output is None else cast("WanHumoAudioOutput", audio_encoder_output),
        latent_frames=latent_frames,
    )
    positive_output = compose_wan21_humo_conditioning(
        positive_text,
        audio_embed=audio,
        reference_latent=reference,
    )
    negative_output = compose_wan21_humo_conditioning(
        negative_text,
        audio_embed=torch.zeros_like(audio),
        reference_latent=torch.zeros_like(reference),
    )
    if positive_binding is not None:
        positive_output = bind_component_conditioning(positive_output, positive_binding)
        negative_output = bind_component_conditioning(negative_output, positive_binding)
    latent = torch.zeros(
        (batch_size, 16, latent_frames, latent_height, latent_width),
        dtype=torch.float32,
    )
    return {
        "positive": positive_output,
        "negative": negative_output,
        "latent": {"samples": latent},
    }


def _infinite_talk_audio_features(
    output: object,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    if type(output) is not WanInfiniteTalkAudioOutput:
        raise TypeError(f"{name} must be an exact WanInfiniteTalkAudioOutput")
    features = torch.stack(output.layers, dim=0).squeeze(1)[1:]
    output_frames = int(features.shape[1] / 50.0 * 25.0)
    if output_frames <= 0:
        raise ValueError(f"{name} is too short for Wan InfiniteTalk")
    return (
        functional.interpolate(
            features.to(device=device, dtype=torch.float16).transpose(1, 2),
            size=output_frames,
            mode="linear",
            align_corners=True,
        )
        .transpose(1, 2)
        .movedim(0, 1)
        .contiguous()
    )


def _infinite_talk_audio_streams(
    first: object,
    second: object | None,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    streams = [_infinite_talk_audio_features(first, "audio_encoder_output_1", device)]
    if second is None:
        return tuple(streams)
    streams.append(_infinite_talk_audio_features(second, "audio_encoder_output_2", device))
    total_frames = sum(stream.shape[0] for stream in streams)
    combined: list[torch.Tensor] = []
    offset = 0
    for stream in streams:
        result = stream.new_zeros((total_frames, *stream.shape[1:]))
        result[offset : offset + stream.shape[0]] = stream
        combined.append(result)
        offset += stream.shape[0]
    return tuple(combined)


def _infinite_talk_runtime(model: object) -> tuple[InferenceRuntimeHandle, Wan21Runtime]:
    if isinstance(model, ApplicationChain):
        raise ValueError("Wan InfiniteTalk cannot be combined with another model intervention")
    handle = require_inference_runtime_handle(model, "model")
    runtime = handle.runtime
    if (
        type(runtime) is not Wan21Runtime
        or type(runtime.assembled.diffusion) is not Wan21Model
        or runtime.assembled.diffusion.config is not WAN21_I2V_14B
    ):
        raise ValueError("Wan InfiniteTalk supports only exact native base Wan 2.1 I2V 14B")
    return handle, runtime


def _infinite_talk_patch(value: object) -> InferenceComponentHandle:
    handle = require_inference_component_handle(value, "model_patch")
    try:
        ComponentBinding("wan21_multitalk", "dinkster.wan21", handle.resource_identity)
    except (TypeError, ValueError) as error:
        raise TypeError("model_patch must be a native Wan InfiniteTalk component") from error
    return handle


def execute_wan_infinite_talk_to_video(
    *,
    mode: str,
    model: object,
    model_patch: object,
    positive: object,
    negative: object,
    vae: object,
    width: int,
    height: int,
    length: int,
    audio_encoder_output_1: object,
    motion_frame_count: int,
    audio_scale: float,
    start_image: object | None = None,
    previous_frames: object | None = None,
    audio_encoder_output_2: object | None = None,
    mask_1: object | None = None,
    mask_2: object | None = None,
) -> Mapping[str, object]:
    if mode not in ("single_speaker", "two_speakers"):
        raise ValueError("mode must be single_speaker or two_speakers")
    width = _integer(width, "width", minimum=16, maximum=16384)
    height = _integer(height, "height", minimum=16, maximum=16384)
    length = _integer(length, "length", minimum=1, maximum=16384)
    motion_frame_count = _integer(
        motion_frame_count,
        "motion_frame_count",
        minimum=1,
        maximum=33,
    )
    audio_scale = _finite_float(audio_scale, "audio_scale", minimum=-10.0, maximum=10.0)
    if width % 16 or height % 16:
        raise ValueError("width and height must be multiples of 16")
    if (length - 1) % 4:
        raise ValueError("length must equal 1 plus a multiple of 4")
    if start_image is None and previous_frames is None:
        raise ValueError("Wan InfiniteTalk requires start_image or previous_frames")
    if type(positive) is not ConditioningCarrier or type(negative) is not ConditioningCarrier:
        raise TypeError("positive and negative must be exact ConditioningCarrier values")
    positive_text, positive_binding = split_component_conditioning(positive)
    negative_text, negative_binding = split_component_conditioning(negative)
    if positive_binding != negative_binding:
        raise ValueError("Wan InfiniteTalk conditioning lanes must share one component binding")
    if positive_binding is not None and positive_binding.role != "umt5xxl":
        raise ValueError("Wan InfiniteTalk conditioning must come from a Wan UMT5-XXL component")

    if mode == "single_speaker":
        if any(value is not None for value in (audio_encoder_output_2, mask_1, mask_2)):
            raise ValueError("single_speaker mode does not accept a second audio stream or masks")
    elif audio_encoder_output_2 is None or mask_1 is None or mask_2 is None:
        raise ValueError("two_speakers mode requires a second audio stream and both masks")

    model_handle, runtime = _infinite_talk_runtime(model)
    patch_handle = _infinite_talk_patch(model_patch)
    codec = require_inference_codec_handle(vae, "vae")
    if codec.descriptor != WAN21_CODEC:
        raise ValueError("vae must expose the Wan 2.1 codec descriptor")
    latent_frames = ((length - 1) // 4) + 1
    latent_height = height // 8
    latent_width = width // 8

    start_pixels = None if start_image is None else _image(start_image, "start_image")[:length]
    previous_pixels = (
        None if previous_frames is None else _image(previous_frames, "previous_frames")
    )
    if previous_pixels is not None and previous_pixels.shape[0] < motion_frame_count:
        raise ValueError("previous_frames does not contain motion_frame_count frames")

    concat_latent = None
    motion_latent = None
    with codec.stage(), torch.inference_mode():
        if start_pixels is not None:
            padded = torch.full((length, height, width, 3), 0.5, dtype=torch.float32)
            resized_start = _common_upscale(
                start_pixels.movedim(-1, 1),
                width,
                height,
                "bilinear",
            ).movedim(1, -1)
            padded[: resized_start.shape[0]] = resized_start
            concat_latent = _s2v_encode_media(
                codec,
                padded,
                width=width,
                height=height,
                name="start image",
            )
            del padded, resized_start
        if previous_pixels is not None:
            resized_motion = _common_upscale(
                previous_pixels[-motion_frame_count:].movedim(-1, 1),
                width,
                height,
                "bilinear",
            ).movedim(1, -1)
            motion_latent = _s2v_encode_media(
                codec,
                resized_motion,
                width=width,
                height=height,
                name="previous frames",
            )
            del resized_motion

    mean = torch.tensor(LATENTS_MEAN, dtype=torch.float32).view(1, 16, 1, 1, 1)
    if concat_latent is None:
        concat_latent = mean.expand(1, 16, latent_frames, latent_height, latent_width).clone()
        known_frames = 0
    else:
        if concat_latent.shape != (1, 16, latent_frames, latent_height, latent_width):
            raise ValueError("vae start image output has incompatible geometry")
        assert start_pixels is not None
        known_frames = ((start_pixels.shape[0] - 1) // 4) + 1
    known_mask = torch.zeros(
        (1, 4, latent_frames, latent_height, latent_width),
        dtype=concat_latent.dtype,
    )
    known_mask[:, :, :known_frames] = 1.0
    combined_concat = torch.cat((known_mask, concat_latent), dim=1)

    extending = motion_latent is not None
    if motion_latent is None:
        motion_latent = concat_latent[:, :, :1].clone()
        audio_start = 0
        trim_image = 0
    else:
        if (
            motion_latent.shape[0:2] != (1, 16)
            or motion_latent.shape[2] > latent_frames
            or motion_latent.shape[-2:] != (latent_height, latent_width)
        ):
            raise ValueError("vae previous frames output has incompatible geometry")
        assert previous_pixels is not None
        audio_start = previous_pixels.shape[0] - motion_frame_count
        trim_image = motion_frame_count
    motion_latent = motion_latent.detach().to(device="cpu").contiguous()

    vision = None
    if start_pixels is not None and runtime.assembled.clip_vision is not None:
        with model_handle.stage("vision"), torch.inference_mode():
            vision = (
                runtime.encode_vision(
                    start_pixels[:1].to(device=cast("Any", model_handle.load_device))
                )
                .detach()
                .to(device="cpu")
                .contiguous()
            )
    positive_output = compose_wan21_i2v_conditioning(
        positive_text,
        concat_latent=combined_concat,
        vision=vision,
    )
    negative_output = compose_wan21_i2v_conditioning(
        negative_text,
        concat_latent=combined_concat,
        vision=vision,
    )
    if positive_binding is not None:
        positive_output = bind_component_conditioning(positive_output, positive_binding)
        negative_output = bind_component_conditioning(negative_output, positive_binding)

    target_masks = None
    if mask_1 is not None and mask_2 is not None:
        first_mask = _mask(mask_1, "mask_1")
        second_mask = _mask(mask_2, "mask_2")
        if first_mask.shape[0] != 1 or second_mask.shape[0] != 1:
            raise ValueError("Wan InfiniteTalk speaker masks must each contain one frame")
        target_masks = functional.interpolate(
            torch.cat((first_mask, second_mask), dim=0).unsqueeze(0),
            size=(latent_height // 2, latent_width // 2),
            mode="nearest",
        )[0]
        target_masks = (target_masks > 0.0).reshape(2, -1).contiguous()

    audio_end = audio_start + length
    with patch_handle.stage(), torch.inference_mode():
        live_patch = patch_handle.component
        if type(live_patch) is not Wan21MultiTalk:
            raise TypeError("model_patch must contain the exact maintained Wan InfiniteTalk patch")
        patch_digest = live_patch.resource_digest
        if patch_digest is None:
            raise RuntimeError("Wan InfiniteTalk patch has no assembly provenance")
        streams = _infinite_talk_audio_streams(
            audio_encoder_output_1,
            audio_encoder_output_2,
            cast("Any", patch_handle.load_device),
        )
        projected_audio = (
            live_patch.project_audio(
                streams,
                audio_start,
                audio_end,
            )
            .detach()
            .to(device="cpu")
            .contiguous()
        )

    audio_digest = wan21_multitalk_tensor_digest(projected_audio)
    masks_digest = None if target_masks is None else wan21_multitalk_tensor_digest(target_masks)
    motion_digest = wan21_multitalk_tensor_digest(motion_latent)

    def materialize(
        runtime_value: object,
        live_component: object,
        latent_value: object,
    ) -> Mapping[str, object]:
        if (
            type(runtime_value) is not Wan21Runtime
            or type(runtime_value.assembled.diffusion) is not Wan21Model
            or runtime_value.assembled.diffusion.config is not WAN21_I2V_14B
        ):
            raise ValueError("Wan InfiniteTalk supports only exact native base Wan 2.1 I2V 14B")
        if type(live_component) is not Wan21MultiTalk:
            raise TypeError("model_patch must contain the exact maintained Wan InfiniteTalk patch")
        if live_component.resource_digest != patch_digest:
            raise ValueError("Wan InfiniteTalk patch provenance changed")
        if type(latent_value) is not MultiStreamLatent:
            raise TypeError("Wan InfiniteTalk sampling latent must be an exact MultiStreamLatent")
        streams_value = cast("MultiStreamLatent[object]", latent_value)
        if streams_value.roles != ("video",):
            raise ValueError("Wan InfiniteTalk requires the exact latent stream role 'video'")
        video = streams_value.by_role("video")
        if (
            type(video) is not torch.Tensor
            or not video.is_floating_point()
            or video.shape != (1, 16, latent_frames, latent_height, latent_width)
        ):
            raise ValueError("Wan InfiniteTalk target video has incompatible geometry")
        patch_execution = Wan21MultiTalkExecution(
            live_component,
            projected_audio,
            target_masks,
            audio_scale,
            patch_digest,
            audio_digest,
            masks_digest,
        )
        return {
            "multitalk": Wan21InfiniteTalkExecution(
                patch_execution,
                motion_latent,
                extending,
                motion_digest,
            )
        }

    application_identity = extend_runtime_identity(
        patch_handle.resource_identity,
        (
            "role=wan21_multitalk",
            f"audio={audio_digest}",
            f"masks={masks_digest or 'none'}",
            f"motion={motion_digest}",
            f"extend={int(extending)}",
            f"strength={audio_scale.hex()}",
        ),
    )
    application = ComponentApplication(
        "dinkster.wan21",
        "diffusion",
        patch_handle,
        application_identity,
        materialize,
    )
    latent = torch.zeros(
        (1, 16, latent_frames, latent_height, latent_width),
        dtype=torch.float32,
    )
    return {
        "model": _append_application(model, application),
        "positive": positive_output,
        "negative": negative_output,
        "latent": {"samples": latent},
        "trim_image": trim_image,
    }


def _s2v_encode_media(
    codec: InferenceCodecHandle[SizedTensor],
    frames: torch.Tensor,
    *,
    width: int,
    height: int,
    name: str,
) -> torch.Tensor:
    resized = _common_upscale(
        frames.movedim(-1, 1),
        width,
        height,
        "bilinear",
    ).movedim(1, -1)
    encoded = _encoded_tensor(codec, resized).detach().to(device="cpu").contiguous()
    del resized
    if encoded.ndim != 5 or encoded.shape[:2] != (1, 16):
        raise ValueError(f"vae {name} output must have shape [1,16,T,H,W]")
    if encoded.shape[-2:] != (height // 8, width // 8):
        raise ValueError(f"vae {name} output has incompatible spatial geometry")
    return encoded


def execute_encode_wandancer_audio(
    *,
    audio: object,
    video_frames: int,
    audio_inject_scale: float,
) -> Mapping[str, object]:
    video_frames = _integer(video_frames, "video_frames", minimum=1, maximum=16384)
    audio_inject_scale = _nonnegative_float(
        audio_inject_scale,
        "audio_inject_scale",
        maximum=10.0,
    )
    waveform, sample_rate = _audio_input(audio)
    resampled = (
        waveform
        if sample_rate == RESAMPLE_SAMPLE_RATE
        else ltx_audio_resample(waveform, sample_rate, RESAMPLE_SAMPLE_RATE)
    )
    features = encode_wandancer_audio_features(
        waveform.numpy(),
        sample_rate,
        resampled.numpy(),
        video_frames,
        audio_inject_scale,
    )
    feature = torch.from_numpy(np.array(features.audio_feature, copy=True)).contiguous()
    output = WanDancerAudioOutput(
        feature,
        Wan22DancerSettings(features.fps, features.audio_inject_scale),
    )
    fps_string = (
        ", \u5e27\u7387\u662f30fps\u3002"
        if int(features.fps + 0.5) == 30
        else f" \u5e27\u7387\u662f{features.fps:.4f}"
    )
    return {"audio_encoder_output": output, "fps_string": fps_string}


def _dancer_runtime(model: object) -> tuple[InferenceRuntimeHandle, Wan21Runtime]:
    if isinstance(model, ApplicationChain):
        raise ValueError("WanDancer cannot be combined with another model intervention")
    handle = require_inference_runtime_handle(model, "model")
    runtime = handle.runtime
    if (
        type(runtime) is not Wan21Runtime
        or type(runtime.assembled.diffusion) is not Wan22DancerModel
        or runtime.assembled.diffusion.config is not WAN22_WANDANCER_14B
    ):
        raise ValueError("model must expose the exact native WanDancer 14B runtime")
    return handle, runtime


def execute_wan22_dancer_video(
    *,
    model: object,
    vae: object,
    positive: object,
    negative: object,
    width: int,
    height: int,
    length: int,
    batch_size: int,
    start_image: object | None = None,
    mask: object | None = None,
    reference_image: object | None = None,
    audio_encoder_output: object | None = None,
) -> Mapping[str, object]:
    width = _integer(width, "width", minimum=16, maximum=16384)
    height = _integer(height, "height", minimum=16, maximum=16384)
    length = _integer(length, "length", minimum=1, maximum=16384)
    batch_size = _integer(batch_size, "batch_size", minimum=1, maximum=4096)
    if width % 16 or height % 16:
        raise ValueError("width and height must be multiples of 16")
    if (length - 1) % 4:
        raise ValueError("length must equal 1 plus a multiple of 4")
    if type(positive) is not ConditioningCarrier or type(negative) is not ConditioningCarrier:
        raise TypeError("positive and negative must be exact ConditioningCarrier values")
    if mask is not None and start_image is None:
        raise ValueError("mask requires start_image")
    if model is vae:
        raise ValueError("model and vae must be distinct resources")
    model_handle, runtime = _dancer_runtime(model)
    codec = require_inference_codec_handle(vae, "vae")
    if codec.descriptor != WAN21_CODEC:
        raise ValueError("vae must expose the Wan 2.1 codec descriptor")
    positive_text, positive_binding = split_component_conditioning(positive)
    negative_text, negative_binding = split_component_conditioning(negative)
    if positive_binding != negative_binding:
        raise ValueError("WanDancer conditioning lanes must share one component binding")
    if positive_binding is not None and positive_binding.role != "umt5xxl":
        raise ValueError("WanDancer conditioning must come from a Wan UMT5-XXL component")

    latent_frames = ((length - 1) // 4) + 1
    latent_height = height // 8
    latent_width = width // 8
    start_pixels = None if start_image is None else _image(start_image, "start_image")[:length]
    reference_pixels = (
        None if reference_image is None else _image(reference_image, "reference_image")[:1]
    )
    concat_latent = None
    if start_pixels is not None:
        with codec.stage(), torch.inference_mode():
            resized = _common_upscale(
                start_pixels.movedim(-1, 1),
                width,
                height,
                "bilinear",
            ).movedim(1, -1)
            canvas = torch.zeros((length, height, width, 3), dtype=resized.dtype)
            canvas[: resized.shape[0]] = resized
            concat_latent = _s2v_encode_media(
                codec,
                canvas,
                width=width,
                height=height,
                name="start image",
            )
    if concat_latent is not None:
        if concat_latent.shape != (1, 16, latent_frames, latent_height, latent_width):
            raise ValueError("vae start image output has incompatible geometry")
        if mask is None:
            assert start_pixels is not None
            known_frames = ((start_pixels.shape[0] - 1) // 4) + 1
            concat_mask = torch.ones(
                (1, 4, latent_frames, latent_height, latent_width),
                dtype=concat_latent.dtype,
            )
            concat_mask[:, :, :known_frames] = 0.0
        else:
            raw_mask = _mask(mask, "mask")
            if raw_mask.shape[0] == 1:
                raw_mask = raw_mask.expand(length, -1, -1)
            elif raw_mask.shape[0] < length:
                raise ValueError("mask must contain one frame or at least length frames")
            raw_mask = functional.interpolate(
                raw_mask[:length].unsqueeze(1),
                size=(latent_height, latent_width),
                mode="nearest-exact",
            )
            expanded_mask = torch.cat(((1.0 - raw_mask[:1]).repeat(4, 1, 1, 1), 1.0 - raw_mask[1:]))
            concat_mask = expanded_mask.view(
                1,
                latent_frames,
                4,
                latent_height,
                latent_width,
            ).transpose(1, 2)
        concat_latent = torch.cat((concat_mask, concat_latent), dim=1).contiguous()
    vision = None
    reference_vision = None
    if start_pixels is not None or reference_pixels is not None:
        with model_handle.stage("vision"), torch.inference_mode():
            if start_pixels is not None:
                vision = (
                    runtime.encode_vision(
                        start_pixels[:1].to(device=cast("Any", model_handle.load_device))
                    )
                    .detach()
                    .to(device="cpu")
                    .contiguous()
                )
            if reference_pixels is not None:
                reference_vision = (
                    runtime.encode_vision(
                        reference_pixels.to(device=cast("Any", model_handle.load_device))
                    )
                    .detach()
                    .to(device="cpu")
                    .contiguous()
                )

    audio = None
    settings = Wan22DancerSettings()
    if audio_encoder_output is not None:
        if type(audio_encoder_output) is not WanDancerAudioOutput:
            raise TypeError("audio_encoder_output must be an exact WanDancerAudioOutput")
        audio = audio_encoder_output.audio_feature
        settings = audio_encoder_output.settings
    positive_output = compose_wan22_dancer_conditioning(
        positive_text,
        concat_latent=concat_latent,
        vision=vision,
        reference_vision=reference_vision,
        audio_embed=audio,
        settings=settings,
    )
    negative_output = compose_wan22_dancer_conditioning(
        negative_text,
        concat_latent=concat_latent,
        vision=vision,
        reference_vision=reference_vision,
        audio_embed=audio,
        settings=settings,
    )
    if positive_binding is not None:
        positive_output = bind_component_conditioning(positive_output, positive_binding)
        negative_output = bind_component_conditioning(negative_output, positive_binding)
    latent = torch.zeros(
        (batch_size, 16, latent_frames, latent_height, latent_width),
        dtype=torch.float32,
    )
    return {
        "positive": positive_output,
        "negative": negative_output,
        "latent": {"samples": latent},
    }


def _keyframe_output(segment: object) -> Mapping[str, object]:
    from .wandancer_audio import WanDancerKeyframeSegment

    if type(segment) is not WanDancerKeyframeSegment:
        raise TypeError("segment must be exact WanDancerKeyframeSegment")
    return {
        "keyframes_sequence": torch.from_numpy(np.array(segment.keyframes, copy=True)),
        "keyframes_mask": torch.from_numpy(np.array(segment.mask, copy=True)),
        "audio_segment": {
            "waveform": torch.from_numpy(np.array(segment.audio_waveform, copy=True)),
            "sample_rate": segment.sample_rate,
        },
    }


def execute_wandancer_pad_keyframes(
    *,
    images: object,
    segment_length: int,
    segment_index: int,
    audio: object,
) -> Mapping[str, object]:
    segment_length = _integer(segment_length, "segment_length", minimum=1, maximum=10000)
    segment_index = _integer(segment_index, "segment_index", minimum=0, maximum=100)
    image = _image(images, "images")
    waveform, sample_rate = _audio_input(audio)
    return _keyframe_output(
        plan_wandancer_keyframes(
            image.numpy(),
            segment_length,
            segment_index,
            waveform.numpy(),
            sample_rate,
        )
    )


def execute_wandancer_pad_keyframe_list(
    *,
    images: object,
    segment_length: int,
    num_segments: int,
    audio: object,
) -> Mapping[str, object]:
    segment_length = _integer(segment_length, "segment_length", minimum=1, maximum=10000)
    num_segments = _integer(num_segments, "num_segments", minimum=1, maximum=100)
    image = _image(images, "images")
    waveform, sample_rate = _audio_input(audio)
    outputs = tuple(
        _keyframe_output(segment)
        for segment in plan_wandancer_keyframe_list(
            image.numpy(),
            segment_length,
            num_segments,
            waveform.numpy(),
            sample_rate,
        )
    )
    return {
        "keyframes_sequence": [output["keyframes_sequence"] for output in outputs],
        "keyframes_mask": [output["keyframes_mask"] for output in outputs],
        "audio_segment": [output["audio_segment"] for output in outputs],
    }


def _wan22_s2v_conditioning(
    *,
    positive: object,
    negative: object,
    vae: object,
    width: int,
    height: int,
    length: int,
    batch_size: int,
    frame_offset: int,
    audio_encoder_output: object | None,
    ref_image: object | None,
    control_video: object | None,
    ref_motion: object | None,
    ref_motion_latent: torch.Tensor | None,
) -> Mapping[str, object]:
    width = _integer(width, "width", minimum=16, maximum=16384)
    height = _integer(height, "height", minimum=16, maximum=16384)
    length = _integer(length, "length", minimum=1, maximum=16384)
    batch_size = _integer(batch_size, "batch_size", minimum=1, maximum=4096)
    frame_offset = _integer(frame_offset, "frame_offset", minimum=0, maximum=GIBIBYTE)
    if width % 16 or height % 16:
        raise ValueError("width and height must be multiples of 16")
    if (length - 1) % 4:
        raise ValueError("length must equal 1 plus a multiple of 4")
    if type(positive) is not ConditioningCarrier or type(negative) is not ConditioningCarrier:
        raise TypeError("positive and negative must be exact ConditioningCarrier values")
    codec = require_inference_codec_handle(vae, "vae")
    if codec.descriptor != WAN21_CODEC:
        raise ValueError("vae must expose the Wan 2.1 codec descriptor")

    latent_frames = ((length - 1) // 4) + 1
    latent_height = height // 8
    latent_width = width // 8
    audio = _s2v_audio_embedding(
        None if audio_encoder_output is None else cast("WanS2VAudioOutput", audio_encoder_output),
        latent_frames=latent_frames,
        frame_offset=frame_offset,
    )
    if audio is not None and audio.shape[3] == 0:
        audio = None

    reference_pixels = None if ref_image is None else _image(ref_image, "ref_image")[:1]
    motion_pixels = None if ref_motion is None else _image(ref_motion, "ref_motion")[-73:]
    control_pixels = (
        None if control_video is None else _image(control_video, "control_video")[:length]
    )
    reference_latent = None
    motion_latent = ref_motion_latent
    encoded_control = None
    stage = (
        codec.stage()
        if any(value is not None for value in (reference_pixels, motion_pixels, control_pixels))
        else nullcontext()
    )
    with torch.inference_mode(), stage:
        if reference_pixels is not None:
            reference_latent = _s2v_encode_media(
                codec,
                reference_pixels,
                width=width,
                height=height,
                name="reference",
            )
            del reference_pixels
        if motion_pixels is not None:
            padded_motion = torch.full((73, *motion_pixels.shape[1:]), 0.5, dtype=torch.float32)
            padded_motion[-motion_pixels.shape[0] :] = motion_pixels
            del motion_pixels
            motion_latent = _s2v_encode_media(
                codec,
                padded_motion,
                width=width,
                height=height,
                name="motion",
            )
            del padded_motion
        if control_pixels is not None:
            encoded_control = _s2v_encode_media(
                codec,
                control_pixels,
                width=width,
                height=height,
                name="control",
            )
            del control_pixels

    if reference_latent is not None and reference_latent.shape[2] != 1:
        raise ValueError("vae reference output must contain one latent frame")
    if motion_latent is not None:
        if (
            type(motion_latent) is not torch.Tensor
            or not motion_latent.is_floating_point()
            or motion_latent.ndim != 5
            or motion_latent.shape[0] not in (1, batch_size)
            or motion_latent.shape[1] != 16
            or motion_latent.shape[-2:] != (latent_height, latent_width)
        ):
            raise ValueError(
                "reference motion latent must match the target batch and spatial shape"
            )
        motion_latent = motion_latent[:, :, -19:].detach().to(device="cpu").contiguous()
    mean = torch.tensor(LATENTS_MEAN, dtype=torch.float32).view(1, 16, 1, 1, 1)
    control_latent = mean.expand(1, 16, latent_frames, latent_height, latent_width).clone()
    if encoded_control is not None:
        if encoded_control.shape[2] > latent_frames:
            raise ValueError("vae control output exceeds the target temporal geometry")
        control_latent[:, :, : encoded_control.shape[2]] = encoded_control

    positive_output = compose_wan22_s2v_conditioning(
        positive,
        audio_embed=audio,
        reference_latent=reference_latent,
        reference_motion=motion_latent,
        control_video=control_latent,
    )
    negative_output = compose_wan22_s2v_conditioning(
        negative,
        audio_embed=None if audio is None else torch.zeros_like(audio),
        reference_latent=reference_latent,
        reference_motion=motion_latent,
        control_video=control_latent,
    )
    latent = torch.zeros(
        (batch_size, 16, latent_frames, latent_height, latent_width),
        dtype=torch.float32,
    )
    return {
        "positive": positive_output,
        "negative": negative_output,
        "latent": {"samples": latent},
    }


def execute_wan22_s2v(
    *,
    positive: object,
    negative: object,
    vae: object,
    width: int,
    height: int,
    length: int,
    batch_size: int,
    audio_encoder_output: object | None = None,
    ref_image: object | None = None,
    control_video: object | None = None,
    ref_motion: object | None = None,
) -> Mapping[str, object]:
    return _wan22_s2v_conditioning(
        positive=positive,
        negative=negative,
        vae=vae,
        width=width,
        height=height,
        length=length,
        batch_size=batch_size,
        frame_offset=0,
        audio_encoder_output=audio_encoder_output,
        ref_image=ref_image,
        control_video=control_video,
        ref_motion=ref_motion,
        ref_motion_latent=None,
    )


def execute_wan22_s2v_extend(
    *,
    positive: object,
    negative: object,
    vae: object,
    length: int,
    video_latent: object,
    audio_encoder_output: object | None = None,
    ref_image: object | None = None,
    control_video: object | None = None,
) -> Mapping[str, object]:
    if not isinstance(video_latent, Mapping) or set(video_latent) != {"samples"}:
        raise TypeError("video_latent must contain exactly samples")
    samples = video_latent["samples"]
    if (
        type(samples) is not torch.Tensor
        or not samples.is_floating_point()
        or samples.ndim != 5
        or samples.shape[1] != 16
        or any(size <= 0 for size in samples.shape)
    ):
        raise ValueError("video_latent samples must be nonempty floating [B,16,T,H,W]")
    return _wan22_s2v_conditioning(
        positive=positive,
        negative=negative,
        vae=vae,
        width=samples.shape[4] * 8,
        height=samples.shape[3] * 8,
        length=length,
        batch_size=samples.shape[0],
        frame_offset=samples.shape[2] * 4,
        audio_encoder_output=audio_encoder_output,
        ref_image=ref_image,
        control_video=control_video,
        ref_motion=None,
        ref_motion_latent=samples,
    )


def execute_wan22_animate_to_video(
    *,
    positive: object,
    negative: object,
    model: object,
    vae: object,
    width: int,
    height: int,
    length: int,
    batch_size: int,
    continue_motion_max_frames: int,
    video_frame_offset: int,
    reference_image: object | None = None,
    face_video: object | None = None,
    pose_video: object | None = None,
    background_video: object | None = None,
    character_mask: object | None = None,
    continue_motion: object | None = None,
) -> Mapping[str, object]:
    """Construct Animate conditioning and its raw output latent."""

    width = _integer(width, "width", minimum=16, maximum=16384)
    height = _integer(height, "height", minimum=16, maximum=16384)
    length = _integer(length, "length", minimum=1, maximum=16384)
    batch_size = _integer(batch_size, "batch_size", minimum=1, maximum=4096)
    continue_motion_max_frames = _integer(
        continue_motion_max_frames,
        "continue_motion_max_frames",
        minimum=1,
        maximum=16384,
    )
    video_frame_offset = _integer(
        video_frame_offset,
        "video_frame_offset",
        minimum=0,
        maximum=16384,
    )
    if width % 16 or height % 16:
        raise ValueError("width and height must be multiples of 16")
    if (length - 1) % 4:
        raise ValueError("length must equal 1 plus a multiple of 4")
    if (continue_motion_max_frames - 1) % 4:
        raise ValueError("continue_motion_max_frames must equal 1 plus a multiple of 4")
    if type(positive) is not ConditioningCarrier or type(negative) is not ConditioningCarrier:
        raise TypeError("positive and negative must be exact ConditioningCarrier values")
    if model is vae:
        raise ValueError("model and vae must be distinct resources")
    model_handle = require_inference_runtime_handle(model, "model")
    codec = require_inference_codec_handle(vae, "vae")
    if codec.descriptor != WAN21_CODEC:
        raise ValueError("vae must expose the Wan 2.1 codec descriptor")
    runtime = _animate_runtime(model_handle)

    latent_length = ((length - 1) // 4) + 1
    latent_width = width // 8
    latent_height = height // 8
    real_reference = reference_image is not None
    reference = (
        torch.zeros((1, height, width, 3), dtype=torch.float32)
        if reference_image is None
        else _image(reference_image, "reference_image")
    )
    resized_reference = _resize_image(reference[:length], width, height)

    motion = None if continue_motion is None else _image(continue_motion, "continue_motion")
    reference_motion_latent_length = 0
    if motion is None:
        motion_frames = torch.ones((length, height, width, 3), dtype=torch.float32) * 0.5
    else:
        motion = motion[-continue_motion_max_frames:]
        video_frame_offset = max(0, video_frame_offset - motion.shape[0])
        motion = _resize_image(motion[-length:], width, height)
        motion_frames = (
            torch.ones(
                (length, height, width, motion.shape[-1]),
                device=motion.device,
                dtype=motion.dtype,
            )
            * 0.5
        )
        motion_frames[: motion.shape[0]] = motion
        reference_motion_latent_length = ((motion.shape[0] - 1) // 4) + 1

    pose = None if pose_video is None else _image(pose_video, "pose_video")
    if pose is not None:
        pose = None if pose.shape[0] <= video_frame_offset else pose[video_frame_offset:]
    if pose is not None:
        pose = _resize_image(pose[:length], width, height)
        if pose.shape[0] < length:
            pose = torch.cat((pose,) + (pose[-1:],) * (length - pose.shape[0]), dim=0)

    face = None if face_video is None else _image(face_video, "face_video")
    if face is not None:
        face = None if face.shape[0] <= video_frame_offset else face[video_frame_offset:]
    face_pixels = None
    if face is not None:
        face_pixels = _common_upscale(face[:length].movedim(-1, 1), 512, 512, "area") * 2.0 - 1.0
        face_pixels = face_pixels.movedim(0, 1).unsqueeze(0)

    background = None if background_video is None else _image(background_video, "background_video")
    reference_images = max(0, reference_motion_latent_length * 4 - 3)
    if background is not None and background.shape[0] > video_frame_offset:
        background = _resize_image(background[video_frame_offset:][:length], width, height)
        if background.shape[0] > reference_images:
            motion_frames[reference_images : background.shape[0]] = background[reference_images:]

    mask_input = None if character_mask is None else _mask(character_mask, "character_mask")

    with torch.inference_mode(), codec.stage():
        reference_latent = _encoded_tensor(codec, resized_reference)
        pose_latent = None if pose is None else _encoded_tensor(codec, pose)
        motion_latent = _encoded_tensor(codec, motion_frames)

    if reference_latent.ndim != 5 or reference_latent.shape[1] != 16:
        raise ValueError("vae reference output must have shape [B,16,T,H,W]")
    if reference_latent.shape[-2:] != (latent_height, latent_width):
        raise ValueError("vae reference output has incompatible spatial geometry")
    trim_latent = reference_latent.shape[2]
    concat_latent = torch.cat((reference_latent, motion_latent), dim=2)
    mask = torch.zeros(
        (
            1,
            4,
            trim_latent,
            reference_latent.shape[-2],
            reference_latent.shape[-1],
        ),
        device=reference_latent.device,
        dtype=reference_latent.dtype,
    )
    motion_mask = torch.ones(
        (1, 1, latent_length * 4, latent_height, latent_width),
        device=mask.device,
        dtype=mask.dtype,
    )
    if motion is not None:
        motion_mask[:, :, : reference_motion_latent_length * 4] = 0.0
    if mask_input is not None and (
        mask_input.shape[0] > video_frame_offset or mask_input.shape[0] == 1
    ):
        if mask_input.shape[0] == 1:
            mask_input = mask_input.repeat((length, 1, 1))
        else:
            mask_input = mask_input[video_frame_offset:]
        mask_input = mask_input.unsqueeze(1).movedim(0, 1).unsqueeze(1)
        mask_input = _common_upscale(
            mask_input[:, :, :length], latent_width, latent_height, "nearest-exact"
        ).to(device=motion_mask.device, dtype=motion_mask.dtype)
        if mask_input.shape[2] > reference_images:
            motion_mask[:, :, reference_images : mask_input.shape[2]] = mask_input[
                :, :, reference_images:
            ]
    motion_mask = motion_mask.view(
        1, motion_mask.shape[2] // 4, 4, motion_mask.shape[3], motion_mask.shape[4]
    ).transpose(1, 2)
    mask = 1.0 - torch.cat((mask, motion_mask), dim=2)
    combined_concat = torch.cat((mask, concat_latent), dim=1)

    vision = None
    if real_reference and runtime.assembled.clip_vision is not None:
        with torch.inference_mode(), model_handle.stage("vision"):
            vision = runtime.encode_vision(
                reference[:1].to(device=cast("Any", model_handle.load_device))
            )

    positive_output = compose_wan21_animate_conditioning(
        positive,
        combined_concat,
        vision=vision,
        pose_latents=pose_latent,
        face_pixel_values=face_pixels,
    )
    negative_output = compose_wan21_animate_conditioning(
        negative,
        combined_concat,
        vision=vision,
        pose_latents=pose_latent,
        face_pixel_values=None if face_pixels is None else face_pixels * 0.0 - 1.0,
    )
    latent = torch.zeros(
        (batch_size, 16, latent_length + trim_latent, latent_height, latent_width),
        dtype=torch.float32,
    )
    return {
        "positive": positive_output,
        "negative": negative_output,
        "latent": {"samples": latent},
        "trim_latent": trim_latent,
        "trim_image": reference_images,
        "video_frame_offset": video_frame_offset + length,
    }


def execute_wan21_animate2_to_video(
    *,
    positive: object,
    negative: object,
    model: object,
    vae: object,
    width: int,
    height: int,
    length: int,
    batch_size: int,
    video_frame_offset: int,
    pose_strength: float,
    pose_start_percent: float,
    pose_end_percent: float,
    reference_image_strength: float,
    reference_image: object | None = None,
    pose_video: object | None = None,
    positive_pose: object | None = None,
    continue_motion: object | None = None,
) -> Mapping[str, object]:
    """Construct Animate2 conditioning and its raw output latent."""

    width = _integer(width, "width", minimum=16, maximum=16384)
    height = _integer(height, "height", minimum=16, maximum=16384)
    length = _integer(length, "length", minimum=1, maximum=16384)
    batch_size = _integer(batch_size, "batch_size", minimum=1, maximum=4096)
    video_frame_offset = _integer(
        video_frame_offset,
        "video_frame_offset",
        minimum=0,
        maximum=16384,
    )
    pose_strength = _nonnegative_float(pose_strength, "pose_strength", maximum=10.0)
    reference_image_strength = _nonnegative_float(
        reference_image_strength,
        "reference_image_strength",
        maximum=10.0,
    )
    if type(pose_start_percent) is not float or type(pose_end_percent) is not float:
        raise TypeError("pose start and end percents must be floats")
    pose_schedule = PercentRange(pose_start_percent, pose_end_percent)
    settings = Wan21Animate2Settings(pose_strength, reference_image_strength)
    if width % 16 or height % 16:
        raise ValueError("width and height must be multiples of 16")
    if (length - 1) % 4:
        raise ValueError("length must equal 1 plus a multiple of 4")
    if type(positive) is not ConditioningCarrier or type(negative) is not ConditioningCarrier:
        raise TypeError("positive and negative must be exact ConditioningCarrier values")
    pose_conditioning = positive if positive_pose is None else positive_pose
    if type(pose_conditioning) is not ConditioningCarrier:
        raise TypeError("positive_pose must be an exact ConditioningCarrier")
    if model is vae:
        raise ValueError("model and vae must be distinct resources")
    model_handle = require_inference_runtime_handle(model, "model")
    codec = require_inference_codec_handle(vae, "vae")
    if codec.descriptor != WAN21_CODEC:
        raise ValueError("vae must expose the Wan 2.1 codec descriptor")
    runtime = _animate2_runtime(model_handle)

    latent_length = ((length - 1) // 4) + 1
    latent_width = width // 8
    latent_height = height // 8
    real_reference = reference_image is not None
    reference = (
        torch.zeros((1, height, width, 3), dtype=torch.float32)
        if reference_image is None
        else _image(reference_image, "reference_image")
    )
    resized_reference = _resize_image(reference[:1], width, height)

    motion = None if continue_motion is None else _image(continue_motion, "continue_motion")
    reference_motion_latent_length = 0
    if motion is None:
        motion_frames = torch.ones((length, height, width, 3), dtype=torch.float32) * 0.5
    else:
        motion = motion[-1:]
        video_frame_offset = max(0, video_frame_offset - motion.shape[0])
        motion = _resize_image(motion, width, height)
        motion_frames = (
            torch.ones(
                (length, height, width, motion.shape[-1]),
                device=motion.device,
                dtype=motion.dtype,
            )
            * 0.5
        )
        motion_frames[: motion.shape[0]] = motion
        reference_motion_latent_length = 1

    pose = None if pose_video is None else _image(pose_video, "pose_video")
    pose_vision_image = None
    if pose is not None:
        if pose.shape[0] <= video_frame_offset:
            raise ValueError(
                f"pose_video has {pose.shape[0]} frames but video_frame_offset is "
                f"{video_frame_offset}; nothing remains"
            )
        pose_vision_image = pose[:1].clone()
        pose = _resize_image(pose[video_frame_offset:][:length], width, height)
        if pose.shape[0] < length:
            pose = torch.cat((pose,) + (pose[-1:],) * (length - pose.shape[0]), dim=0)

    with torch.inference_mode(), codec.stage():
        reference_latent = _encoded_tensor(codec, resized_reference)
        motion_latent = _encoded_tensor(codec, motion_frames)
        pose_latent = None if pose is None else _encoded_tensor(codec, pose)
    for name, latent in (
        ("reference", reference_latent),
        ("motion", motion_latent),
        ("pose", pose_latent),
    ):
        if latent is None:
            continue
        if latent.ndim != 5 or latent.shape[1] != 16:
            raise ValueError(f"vae {name} output must have shape [B,16,T,H,W]")
        if latent.shape[-2:] != (latent_height, latent_width):
            raise ValueError(f"vae {name} output has incompatible spatial geometry")
    trim_latent = reference_latent.shape[2]
    concat_latent = torch.cat((reference_latent, motion_latent), dim=2)
    if concat_latent.shape[2] != trim_latent + latent_length:
        raise ValueError("vae motion output has incompatible temporal geometry")
    if pose_latent is not None and pose_latent.shape[2] != latent_length:
        raise ValueError("vae pose output has incompatible temporal geometry")
    mask = torch.ones(
        (1, 4, trim_latent + latent_length, latent_height, latent_width),
        device=concat_latent.device,
        dtype=concat_latent.dtype,
    )
    mask[:, :, : trim_latent + reference_motion_latent_length] = 0.0
    combined_concat = torch.cat((1.0 - mask, concat_latent), dim=1)

    vision = None
    pose_vision = None
    if runtime.assembled.clip_vision is not None and (
        real_reference or pose_vision_image is not None
    ):
        with torch.inference_mode(), model_handle.stage("vision"):
            if real_reference:
                vision = runtime.encode_vision(
                    reference[:1].to(device=cast("Any", model_handle.load_device))
                )
            if pose_vision_image is not None:
                pose_vision = runtime.encode_vision(
                    pose_vision_image.to(device=cast("Any", model_handle.load_device))
                )
    if pose_vision is None:
        pose_vision = vision

    positive_output = compose_wan21_animate2_conditioning(
        positive,
        combined_concat,
        vision=vision,
        pose_text=pose_conditioning,
        pose_vision=pose_vision,
        pose_latents=pose_latent,
        pose_schedule=pose_schedule,
        settings=settings,
    )
    negative_output = compose_wan21_animate2_conditioning(
        negative,
        combined_concat,
        vision=vision,
        pose_text=pose_conditioning,
        pose_vision=pose_vision,
        pose_latents=pose_latent,
        pose_schedule=pose_schedule,
        settings=settings,
    )
    latent = torch.zeros(
        (batch_size, 16, latent_length + trim_latent, latent_height, latent_width),
        dtype=torch.float32,
    )
    return {
        "positive": positive_output,
        "negative": negative_output,
        "latent": {"samples": latent},
        "trim_latent": trim_latent,
        "trim_image": max(0, reference_motion_latent_length * 4 - 3),
        "video_frame_offset": video_frame_offset + length,
    }


def execute_wan21_scail_to_video(
    *,
    positive: object,
    negative: object,
    model: object,
    vae: object,
    width: int,
    height: int,
    length: int,
    batch_size: int,
    pose_strength: float,
    pose_start_percent: float,
    pose_end_percent: float,
    video_frame_offset: int,
    previous_frame_count: int,
    replacement_mode: bool,
    reference_image: object | None,
    pose_video: object | None = None,
    pose_video_mask: object | None = None,
    reference_image_mask: object | None = None,
    previous_frames: object | None = None,
) -> Mapping[str, object]:
    """Construct SCAIL or SCAIL2 conditioning and its raw output latent."""

    width = _integer(width, "width", minimum=32, maximum=16384)
    height = _integer(height, "height", minimum=32, maximum=16384)
    length = _integer(length, "length", minimum=1, maximum=16384)
    batch_size = _integer(batch_size, "batch_size", minimum=1, maximum=4096)
    video_frame_offset = _integer(
        video_frame_offset,
        "video_frame_offset",
        minimum=0,
        maximum=16384,
    )
    previous_frame_count = _integer(
        previous_frame_count,
        "previous_frame_count",
        minimum=1,
        maximum=16384,
    )
    pose_strength = _nonnegative_float(pose_strength, "pose_strength", maximum=10.0)
    if type(pose_start_percent) is not float or type(pose_end_percent) is not float:
        raise TypeError("pose start and end percents must be floats")
    pose_schedule = PercentRange(pose_start_percent, pose_end_percent)
    if type(replacement_mode) is not bool:
        raise TypeError("replacement_mode must be an exact bool")
    if width % 32 or height % 32:
        raise ValueError("width and height must be multiples of 32")
    if (length - 1) % 4:
        raise ValueError("length must equal 1 plus a multiple of 4")
    if (previous_frame_count - 1) % 4:
        raise ValueError("previous_frame_count must equal 1 plus a multiple of 4")
    if type(positive) is not ConditioningCarrier or type(negative) is not ConditioningCarrier:
        raise TypeError("positive and negative must be exact ConditioningCarrier values")
    if model is vae:
        raise ValueError("model and vae must be distinct resources")
    model_handle = require_inference_runtime_handle(model, "model")
    codec = require_inference_codec_handle(vae, "vae")
    if codec.descriptor != WAN21_CODEC:
        raise ValueError("vae must expose the Wan 2.1 codec descriptor")
    runtime = _scail_runtime(model_handle)
    variant = runtime.assembled.diffusion.config.model_variant
    if variant == "scail" and (pose_video_mask is not None or reference_image_mask is not None):
        raise ValueError("colored identity masks require a Wan 2.1 SCAIL2 profile")
    if variant == "scail" and previous_frames is not None:
        raise ValueError("previous_frames requires a Wan 2.1 SCAIL2 profile")

    reference = None if reference_image is None else _image(reference_image, "reference_image")
    original_reference = reference
    if reference is not None:
        reference = _common_upscale(reference.movedim(-1, 1), width, height, "bicubic").movedim(
            1, -1
        )
    reference_mask_pixels = (
        None
        if reference_image_mask is None
        else _image(reference_image_mask, "reference_image_mask")
    )
    if reference_mask_pixels is not None and reference is None:
        raise ValueError("reference_image_mask requires reference_image")
    resized_reference_mask = None
    if reference_mask_pixels is not None:
        assert reference is not None
        resized_reference_mask = _common_upscale(
            reference_mask_pixels.movedim(-1, 1), width, height, "nearest-exact"
        ).movedim(1, -1)
        selected = resized_reference_mask[
            [min(index, resized_reference_mask.shape[0] - 1) for index in range(reference.shape[0])]
        ]
        if replacement_mode:
            foreground = (selected[..., :3].amax(dim=-1, keepdim=True) > 0.1).to(reference)
            reference = reference * foreground

    previous = None if previous_frames is None else _image(previous_frames, "previous_frames")
    if previous is not None:
        previous = previous[-previous_frame_count:]
        video_frame_offset = max(0, video_frame_offset - previous.shape[0])

    pose = None if pose_video is None else _image(pose_video, "pose_video")
    driving_mask_pixels = (
        None if pose_video_mask is None else _image(pose_video_mask, "pose_video_mask")
    )
    if pose is not None:
        pose = None if pose.shape[0] <= video_frame_offset else pose[video_frame_offset:]
    if driving_mask_pixels is not None:
        driving_mask_pixels = (
            None
            if driving_mask_pixels.shape[0] <= video_frame_offset
            else driving_mask_pixels[video_frame_offset:]
        )
    available_frames = tuple(
        value.shape[0] for value in (pose, driving_mask_pixels) if value is not None
    )
    if available_frames:
        kept_frames = ((min(min(available_frames), length) - 1) // 4) * 4 + 1
        if pose is not None:
            pose = pose[:kept_frames]
        if driving_mask_pixels is not None:
            driving_mask_pixels = driving_mask_pixels[:kept_frames]
    if driving_mask_pixels is not None and pose is None:
        raise ValueError("pose_video_mask requires pose_video")
    if pose is not None:
        pose = _resize_image(pose, width // 2, height // 2)
    if driving_mask_pixels is not None:
        driving_mask_pixels = _resize_image(
            driving_mask_pixels,
            width // 2,
            height // 2,
        )

    with torch.inference_mode(), codec.stage():
        reference_latents = (
            ()
            if reference is None
            else tuple(
                _encoded_tensor(codec, reference[index : index + 1])
                for index in range(reference.shape[0])
            )
        )
        pose_latent = None if pose is None else _encoded_tensor(codec, pose) * pose_strength
        previous_latent = (
            None
            if previous is None
            else _encoded_tensor(
                codec,
                _common_upscale(previous.movedim(-1, 1), width, height, "bicubic").movedim(1, -1),
            )
        )

    latent_height = height // 8
    latent_width = width // 8
    for index, value in enumerate(reference_latents):
        if value.ndim != 5 or value.shape[:3] != (1, 16, 1):
            raise ValueError(f"vae reference output {index} must have shape [1,16,1,H,W]")
        if value.shape[-2:] != (latent_height, latent_width):
            raise ValueError(f"vae reference output {index} has incompatible spatial geometry")
    latent_frames = ((length - 1) // 4) + 1
    if pose_latent is not None:
        if pose_latent.ndim != 5 or pose_latent.shape[:2] != (1, 16):
            raise ValueError("vae pose output must have shape [1,16,T,H,W]")
        if pose_latent.shape[-2:] != (latent_height // 2, latent_width // 2):
            raise ValueError("vae pose output has incompatible spatial geometry")

    driving_mask = None if driving_mask_pixels is None else _scail_mask(driving_mask_pixels)
    reference_mask = None
    if resized_reference_mask is not None:
        assert reference is not None
        mask_count = resized_reference_mask.shape[0]
        additional_masks = tuple(
            _scail_mask(
                resized_reference_mask[min(index, mask_count - 1) : min(index, mask_count - 1) + 1]
            )
            for index in range(1, reference.shape[0])
        )
        primary_mask = _scail_mask(resized_reference_mask[:1])
        zeros = torch.zeros(
            (1, 28, latent_frames, latent_height, latent_width),
            device=primary_mask.device,
            dtype=primary_mask.dtype,
        )
        reference_mask = torch.cat((*additional_masks, primary_mask, zeros), dim=2)

    vision = None
    if runtime.assembled.clip_vision is not None and original_reference is not None:
        with torch.inference_mode(), model_handle.stage("vision"):
            vision = runtime.encode_vision(
                original_reference[:1].to(device=cast("Any", model_handle.load_device)),
                crop=False,
            )

    positive_output = compose_wan21_scail_conditioning(
        positive,
        reference_latents,
        vision=vision,
        pose_latents=pose_latent,
        reference_mask=reference_mask,
        driving_mask=driving_mask,
        pose_schedule=pose_schedule,
        replacement=replacement_mode,
    )
    negative_output = compose_wan21_scail_conditioning(
        negative,
        reference_latents,
        vision=vision,
        pose_latents=pose_latent,
        reference_mask=reference_mask,
        driving_mask=driving_mask,
        pose_schedule=pose_schedule,
        replacement=replacement_mode,
    )
    latent = torch.zeros(
        (batch_size, 16, latent_frames, latent_height, latent_width),
        dtype=torch.float32,
    )
    latent_output: dict[str, torch.Tensor] = {"samples": latent}
    if previous_latent is not None:
        if previous_latent.ndim != 5 or previous_latent.shape[:2] != (1, 16):
            raise ValueError("vae previous_frames output must have shape [1,16,T,H,W]")
        if previous_latent.shape[-2:] != (latent_height, latent_width):
            raise ValueError("vae previous_frames output has incompatible spatial geometry")
        fixed_frames = min(previous_latent.shape[2], latent_frames)
        latent[:, :, :fixed_frames] = previous_latent[:, :, :fixed_frames].to(latent)
        noise_mask = torch.ones(
            (batch_size, 1, latent_frames, latent_height, latent_width),
            dtype=latent.dtype,
        )
        noise_mask[:, :, :fixed_frames] = 0.0
        latent_output["noise_mask"] = noise_mask
    return {
        "positive": positive_output,
        "negative": negative_output,
        "latent": latent_output,
        "video_frame_offset": video_frame_offset + length,
    }


__all__ = [
    "WanDancerAudioOutput",
    "WanHumoAudioOutput",
    "WanInfiniteTalkAudioOutput",
    "WanS2VAudioOutput",
    "execute_ar_video_i2v",
    "execute_apply_wan21_uni3c",
    "execute_encode_wandancer_audio",
    "execute_encode_wav2vec2_audio",
    "execute_empty_ar_video_latent",
    "execute_load_wav2vec2_audio_encoder",
    "execute_load_wan21_uni3c",
    "execute_sampler_ar_video",
    "execute_wan21_animate2_to_video",
    "execute_wan21_humo",
    "execute_wan21_scail_to_video",
    "execute_wan22_animate_to_video",
    "execute_wan22_dancer_video",
    "execute_wan22_s2v",
    "execute_wan22_s2v_extend",
    "execute_wan_infinite_talk_to_video",
    "execute_wandancer_pad_keyframe_list",
    "execute_wandancer_pad_keyframes",
]
