"""Qwen Image node execution over public inference resource seams."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from dinkster_inference import (
    BFLOAT16,
    QWEN_IMAGE_CONFIG,
    WAN21_CODEC,
    ApplicationChain,
    ComponentApplication,
    ComponentBinding,
    ControlApplication,
    InferenceComponentHandle,
    InferenceRuntimeHandle,
    PayloadReference,
    PercentRange,
    bind_component_conditioning,
    build_runtime_identity_from_facts,
    extend_runtime_identity,
    load_safetensors_header,
    plan_qwen_image_control,
    plan_qwen_image_diffsynth,
    require_inference_codec_handle,
    require_inference_component_handle,
    require_inference_runtime_handle,
    runtime_component_identity,
)
from dinkster_inference_torch import (
    QwenImageConditioning,
    QwenImageControlConditioning,
    QwenImageDiffSynthExecution,
    QwenImageDiffSynthPatch,
    QwenImageFunControlNet,
    QwenImageInstantXControlNet,
    QwenImageRuntime,
    QwenImageTextRuntime,
    WanVAECodecRuntime,
    assemble_qwen_image_control,
    assemble_qwen_image_diffsynth,
    component_publisher,
    qwen_image_conditioning_to_carrier,
    qwen_image_control_hint_digest,
    resize_qwen_image_content,
)


class _AssetRef(Protocol):
    digest: str
    size: int

    def local_path(self) -> Path: ...


def _asset(value: object, name: str) -> _AssetRef:
    digest = getattr(value, "digest", None)
    size = getattr(value, "size", None)
    local_path = getattr(value, "local_path", None)
    if type(digest) is not str or type(size) is not int or not callable(local_path):
        raise TypeError(f"{name} must be a resolved asset reference")
    return cast("_AssetRef", value)


def _control_resource_identity(component_plan: Any) -> str:
    return build_runtime_identity_from_facts(
        QWEN_IMAGE_CONFIG.family_id,
        runtime_component_identity(QWEN_IMAGE_CONFIG.family_id, (component_plan,)),
        diffusion_dtype=BFLOAT16.name,
        text_dtype="unloaded",
        vae_dtype="unloaded",
        fp8_matmul=False,
        runtime_facts=component_plan.runtime_facts,
    )


def execute_load_qwen_image_control(*, control_net: object) -> Mapping[str, object]:
    asset = _asset(control_net, "control_net")
    source = load_safetensors_header(
        asset.local_path(),
        asset_digest=asset.digest,
        asset_size=asset.size,
    )
    plan = plan_qwen_image_control(source, asset_digest=asset.digest)
    assembled = assemble_qwen_image_control(plan, compute_dtype=torch.bfloat16)
    identity = _control_resource_identity(plan.control)
    handle = component_publisher().publish(assembled.control, resource_identity=identity)
    return {"control": handle}


def execute_load_qwen_image_diffsynth(*, model_patch: object) -> Mapping[str, object]:
    asset = _asset(model_patch, "model_patch")
    source = load_safetensors_header(
        asset.local_path(),
        asset_digest=asset.digest,
        asset_size=asset.size,
    )
    plan = plan_qwen_image_diffsynth(source, asset_digest=asset.digest)
    assembled = assemble_qwen_image_diffsynth(plan, compute_dtype=torch.bfloat16)
    identity = _control_resource_identity(plan.patch)
    handle = component_publisher().publish(assembled.patch, resource_identity=identity)
    return {"patch": handle}


def _latent(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a latent mapping")
    samples = cast("Mapping[object, object]", value).get("samples")
    if type(samples) is not torch.Tensor:
        raise TypeError(f"{name} samples must be an exact torch.Tensor")
    tensor = samples
    if (
        tensor.ndim != 5
        or tensor.shape[0] < 1
        or tensor.shape[1] < 1
        or tensor.shape[2] != 1
        or min(tensor.shape[-2:]) < 1
        or not tensor.is_floating_point()
    ):
        raise ValueError(f"{name} samples must be floating [batch,channels,1,height,width]")
    return tensor.detach().clone()


def _finite(value: object, name: str, *, minimum: float, maximum: float) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a number")
    resolved = float(cast("int | float", value))
    if not math.isfinite(resolved) or not minimum <= resolved <= maximum:
        raise ValueError(f"{name} must be finite and within [{minimum}, {maximum}]")
    return resolved


def _append_application(model: object, application: ComponentApplication) -> ApplicationChain:
    return (
        model.append(application)
        if isinstance(model, ApplicationChain)
        else ApplicationChain(model, (application,))
    )


def execute_apply_qwen_image_control(
    *,
    model: object,
    control: object,
    hint: object,
    strength: float,
    start_percent: float,
    end_percent: float,
) -> Mapping[str, object]:
    handle = _qwen_component(control, "control", "qwen_image_control")
    control_hint = _latent(hint, "hint")
    strength = _finite(strength, "strength", minimum=0.0, maximum=10.0)
    start = _finite(start_percent, "start_percent", minimum=0.0, maximum=1.0)
    end = _finite(end_percent, "end_percent", minimum=0.0, maximum=1.0)
    window = PercentRange(start, end)
    hint_digest = qwen_image_control_hint_digest(control_hint)

    def materialize(
        _runtime: object,
        live_component: object,
        _latent_value: object,
    ) -> Mapping[str, object]:
        if type(live_component) is QwenImageFunControlNet:
            kind = "fun"
        elif type(live_component) is QwenImageInstantXControlNet:
            kind = (
                "instantx_inpaint"
                if live_component.controlnet_x_embedder.in_features == 68
                else "instantx"
            )
        else:
            raise TypeError("control must contain a maintained Qwen Image ControlNet")
        model_digest = live_component.resource_digest
        if model_digest is None:
            raise RuntimeError("Qwen Image ControlNet has no assembly provenance")
        return {
            "control": QwenImageControlConditioning(
                ControlApplication(
                    f"qwen-image-{kind}",
                    PayloadReference(hint_digest),
                    strength,
                    window,
                ),
                live_component,
                kind,
                control_hint,
                model_digest,
                hint_digest,
            )
        }

    identity = extend_runtime_identity(
        handle.resource_identity,
        (
            "role=qwen_image_control",
            f"hint={hint_digest}",
            f"strength={strength.hex()}",
            f"start={start.hex()}",
            f"end={end.hex()}",
        ),
    )
    application = ComponentApplication(
        QWEN_IMAGE_CONFIG.family_id,
        "diffusion",
        handle,
        identity,
        materialize,
    )
    return {"model": _append_application(model, application)}


def execute_apply_qwen_image_diffsynth(
    *,
    model: object,
    patch: object,
    hint: object,
    strength: float,
) -> Mapping[str, object]:
    handle = _qwen_component(patch, "patch", "qwen_image_diffsynth")
    control_hint = _latent(hint, "hint")
    strength = _finite(strength, "strength", minimum=-10.0, maximum=10.0)
    hint_digest = qwen_image_control_hint_digest(control_hint)

    def materialize(
        _runtime: object,
        live_component: object,
        latent_value: object,
    ) -> Mapping[str, object]:
        if type(live_component) is not QwenImageDiffSynthPatch:
            raise TypeError("patch must contain a maintained Qwen Image DiffSynth model")
        if type(latent_value) is not torch.Tensor:
            raise TypeError("Qwen Image sampling latent must be an exact torch.Tensor")
        if control_hint.shape[0] != latent_value.shape[0]:
            raise ValueError("Qwen Image DiffSynth hint batch must match the sampling latent")
        parameter = next(live_component.parameters())
        condition = live_component.prepare_condition(
            control_hint.to(device=parameter.device, dtype=parameter.dtype)
        )
        model_digest = live_component.resource_digest
        if model_digest is None:
            raise RuntimeError("Qwen Image DiffSynth model has no assembly provenance")
        return {
            "diffsynth": (
                QwenImageDiffSynthExecution(
                    live_component,
                    condition,
                    strength,
                    model_digest,
                ),
            )
        }

    identity = extend_runtime_identity(
        handle.resource_identity,
        (
            "role=qwen_image_diffsynth",
            f"hint={hint_digest}",
            f"strength={strength.hex()}",
        ),
    )
    application = ComponentApplication(
        QWEN_IMAGE_CONFIG.family_id,
        "diffusion",
        handle,
        identity,
        materialize,
    )
    return {"model": _append_application(model, application)}


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
        tensor = tensor.expand(*tensor.shape[:-1], 3)
    return tensor[..., :3].permute(0, 3, 1, 2).contiguous()


def _qwen_runtime(handle: InferenceRuntimeHandle) -> QwenImageRuntime:
    runtime = handle.runtime
    if not isinstance(runtime, QwenImageRuntime):
        raise TypeError("clip runtime must be a QwenImageRuntime")
    return runtime


def _qwen_component(value: object, name: str, role: str) -> InferenceComponentHandle:
    handle = require_inference_component_handle(value, name)
    try:
        ComponentBinding(role, QWEN_IMAGE_CONFIG.family_id, handle.resource_identity)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a native Qwen Image {role} component") from error
    return handle


def _encoded_reference(codec: Any, content: torch.Tensor, load_device: object) -> torch.Tensor:
    encoded = codec.encode_content(content.to(device=cast("Any", load_device)))
    if type(encoded) is not torch.Tensor:
        raise TypeError("vae encode_content must return an exact torch.Tensor")
    return encoded


def execute_qwen_image_edit_encode(
    *,
    clip: object,
    vae: object | None,
    prompt: str,
    image: object | None,
) -> Mapping[str, object]:
    return _execute_qwen_image_edit_encode(
        clip=clip,
        vae=vae,
        prompt=prompt,
        images=(image,),
        edit_plus=False,
    )


def execute_qwen_image_edit_plus_encode(
    *,
    clip: object,
    vae: object | None,
    prompt: str,
    images: Sequence[object | None],
) -> Mapping[str, object]:
    return _execute_qwen_image_edit_encode(
        clip=clip,
        vae=vae,
        prompt=prompt,
        images=images,
        edit_plus=True,
    )


def _execute_qwen_image_edit_encode(
    *,
    clip: object,
    vae: object | None,
    prompt: str,
    images: Sequence[object | None],
    edit_plus: bool,
) -> Mapping[str, object]:
    present = tuple(
        (index, _image(image, f"image{index}"))
        for index, image in enumerate(images, start=1)
        if image is not None
    )
    reference_slots = tuple(index for index, _image_value in present)
    references = tuple(image_value for _index, image_value in present)
    if isinstance(clip, InferenceRuntimeHandle):
        clip_handle = require_inference_runtime_handle(clip, "clip")
        runtime = _qwen_runtime(clip_handle)
        text_identity = ""
        component_conditioning = False
    else:
        clip_handle = _qwen_component(clip, "clip", "qwen2_5_vl_7b")
        runtime = None
        text_identity = clip_handle.resource_identity
        component_conditioning = True
    text_references = tuple(
        image.to(device=cast("Any", clip_handle.load_device)) for image in references
    )
    clip_stage = (
        cast("InferenceComponentHandle", clip_handle).stage()
        if component_conditioning
        else cast("Any", clip_handle).stage("text")
    )
    with clip_stage:
        with torch.inference_mode():
            if component_conditioning:
                runtime = QwenImageTextRuntime(
                    cast("InferenceComponentHandle", clip_handle).component
                )
            assert runtime is not None
            if references or edit_plus:
                context, attention_mask = runtime.encode_edit_text(
                    prompt,
                    text_references,
                    edit_plus=edit_plus,
                    image_slots=reference_slots,
                )
                conditioning = QwenImageConditioning(context, None, attention_mask)
            else:
                conditioning = runtime.encode_text(prompt)
    if references and vae is not None:
        if isinstance(vae, InferenceComponentHandle):
            codec_handle = _qwen_component(vae, "vae", "vae")
            codec = None
        else:
            codec = require_inference_codec_handle(vae, "vae")
            if codec.descriptor != WAN21_CODEC:
                raise ValueError("vae must expose the Wan 2.1 codec descriptor")
            codec_handle = codec
        multiple = 8 if edit_plus else 1
        resized = tuple(
            resize_qwen_image_content(image, target_pixels=1024 * 1024, multiple=multiple)
            for image in references
        )
        with codec_handle.stage():
            with torch.inference_mode():
                if codec is None:
                    codec = WanVAECodecRuntime(
                        cast("InferenceComponentHandle", codec_handle).component
                    )
                latents = tuple(
                    _encoded_reference(codec, image, codec_handle.load_device) for image in resized
                )
        conditioning = QwenImageConditioning(
            conditioning.embeddings,
            conditioning.pooled,
            conditioning.attention_mask,
            latents,
        )
    carrier = qwen_image_conditioning_to_carrier(conditioning)
    if component_conditioning:
        carrier = bind_component_conditioning(
            carrier,
            ComponentBinding("qwen2_5_vl_7b", QWEN_IMAGE_CONFIG.family_id, text_identity),
        )
    return {"conditioning": carrier}


def _integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def execute_empty_qwen_image_layered_latent(
    *, width: int, height: int, layers: int, batch_size: int
) -> Mapping[str, object]:
    width = _integer(width, "width", minimum=16, maximum=16384)
    height = _integer(height, "height", minimum=16, maximum=16384)
    layers = _integer(layers, "layers", minimum=0, maximum=4096)
    batch_size = _integer(batch_size, "batch_size", minimum=1, maximum=4096)
    if width % 16 or height % 16:
        raise ValueError("width and height must be multiples of 16")
    samples = torch.zeros(
        (batch_size, 16, layers + 1, height // 8, width // 8),
        dtype=torch.float32,
    )
    return {"latent": {"samples": samples}}


__all__ = [
    "execute_apply_qwen_image_control",
    "execute_apply_qwen_image_diffsynth",
    "execute_empty_qwen_image_layered_latent",
    "execute_load_qwen_image_control",
    "execute_load_qwen_image_diffsynth",
    "execute_qwen_image_edit_encode",
    "execute_qwen_image_edit_plus_encode",
]
