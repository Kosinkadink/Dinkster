"""Native loading and application for standard SD1.5 IP-Adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from dinkster_inference import (
    FLOAT16,
    FLOAT32,
    SD15,
    SD15_IPADAPTER_SITES,
    ApplicationChain,
    ComponentApplication,
    ComponentPlan,
    InferenceComponentHandle,
    PayloadReference,
    PercentRange,
    SD15AttentionContribution,
    build_runtime_identity_from_facts,
    extend_runtime_identity,
    load_safetensors_header,
    plan_sd15_ipadapter,
    runtime_component_identity,
)
from dinkster_inference_torch import (
    SD15IPAdapter,
    SD15IPAdapterClipVisionEncoder,
    SD15IPAdapterConditioning,
    assemble_sd15_ipadapter,
    component_publisher,
    sd15_ipadapter_identity_facts,
    sd15_ipadapter_tensor_digest,
)

_ADAPTER_ASSET_DIGEST = "blake3:7f0a43a48969f0e17963995676df5dab9849bf1051a50c24685348f5a8960f33"
_ADAPTER_ASSET_SIZE = 44_642_768
_CLIP_VISION_ASSET_DIGEST = (
    "blake3:4649ee2cccf3b579a716035ba57d199aaad1b090217516632ccc1df004e0291a"
)
_CLIP_VISION_ASSET_SIZE = 2_528_373_448


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


def _official_asset(
    value: object,
    name: str,
    *,
    digest: str,
    size: int,
) -> _AssetRef:
    asset = _asset(value, name)
    if asset.digest != digest or asset.size != size:
        raise ValueError(f"{name} must be the pinned official standard SD1.5 IP-Adapter artifact")
    return asset


def _component_identity(
    component: ComponentPlan[Any],
    *,
    role: str,
    asset_digest: str,
    compute_dtype: str,
) -> str:
    return build_runtime_identity_from_facts(
        SD15.id,
        runtime_component_identity(SD15.id, (component,)),
        diffusion_dtype=compute_dtype if role == "ipadapter" else "unloaded",
        text_dtype=compute_dtype if role == "ipadapter_clip_vision" else "unloaded",
        vae_dtype="unloaded",
        fp8_matmul=False,
        runtime_facts=(f"role={role}", f"asset={asset_digest}"),
    )


@dataclass(frozen=True, slots=True)
class SD15IPAdapterResource:
    adapter: InferenceComponentHandle
    clip_vision: InferenceComponentHandle
    adapter_asset_digest: str
    clip_vision_asset_digest: str
    identity: str

    def __post_init__(self) -> None:
        if self.adapter_asset_digest != _ADAPTER_ASSET_DIGEST:
            raise ValueError("IP-Adapter resource must bind the pinned official adapter")
        if self.clip_vision_asset_digest != _CLIP_VISION_ASSET_DIGEST:
            raise ValueError("IP-Adapter resource must bind the pinned official CLIP vision model")
        expected = extend_runtime_identity(
            self.adapter.resource_identity,
            (f"clip_vision={self.clip_vision.resource_identity}",),
        )
        if self.identity != expected:
            raise ValueError("IP-Adapter resource identity does not bind its CLIP vision model")

    @property
    def _dinkster_resident_owner(self) -> object:
        return self.adapter

    @property
    def _dinkster_resident_refs(self) -> tuple[object, ...]:
        return (self.clip_vision,)

    @property
    def _dinkster_resident_fingerprint(self) -> str:
        return self.identity


def execute_load_sd15_ipadapter(*, adapter: object, clip_vision: object) -> Mapping[str, object]:
    adapter_asset = _official_asset(
        adapter,
        "adapter",
        digest=_ADAPTER_ASSET_DIGEST,
        size=_ADAPTER_ASSET_SIZE,
    )
    vision_asset = _official_asset(
        clip_vision,
        "clip_vision",
        digest=_CLIP_VISION_ASSET_DIGEST,
        size=_CLIP_VISION_ASSET_SIZE,
    )
    adapter_source = load_safetensors_header(
        adapter_asset.local_path(),
        asset_digest=adapter_asset.digest,
        asset_size=adapter_asset.size,
    )
    vision_source = load_safetensors_header(
        vision_asset.local_path(),
        asset_digest=vision_asset.digest,
        asset_size=vision_asset.size,
    )
    plan = plan_sd15_ipadapter(
        adapter_source,
        vision_source,
        adapter_asset_digest=adapter_asset.digest,
        clip_vision_asset_digest=vision_asset.digest,
    )
    assembled = assemble_sd15_ipadapter(plan, adapter_dtype=torch.float16)
    adapter_identity = _component_identity(
        plan.adapter,
        role="ipadapter",
        asset_digest=adapter_asset.digest,
        compute_dtype=FLOAT16.name,
    )
    vision_identity = _component_identity(
        plan.clip_vision,
        role="ipadapter_clip_vision",
        asset_digest=vision_asset.digest,
        compute_dtype=FLOAT32.name,
    )
    publisher = component_publisher()
    adapter_handle = publisher.publish(assembled.adapter, resource_identity=adapter_identity)
    vision_handle = publisher.publish(assembled.clip_vision, resource_identity=vision_identity)
    identity = extend_runtime_identity(
        adapter_identity,
        (f"clip_vision={vision_identity}",),
    )
    return {
        "ipadapter": SD15IPAdapterResource(
            adapter_handle,
            vision_handle,
            adapter_asset.digest,
            vision_asset.digest,
            identity,
        )
    }


def _image(value: object) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    elif type(value) is torch.Tensor:
        tensor = value.detach()
    else:
        raise TypeError("image must be a numpy array or exact torch.Tensor")
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if (
        tensor.ndim != 4
        or tensor.shape[0] != 1
        or tensor.shape[-1] not in (1, 3, 4)
        or any(size <= 0 for size in tensor.shape)
    ):
        raise ValueError("standard SD1.5 IP-Adapter image must be one nonempty HWC image")
    if not tensor.is_floating_point():
        raise TypeError("image must contain floating-point values")
    tensor = tensor.to(device="cpu", dtype=torch.float32, copy=True)
    if not bool(torch.isfinite(tensor).all()) or bool(torch.any((tensor < 0.0) | (tensor > 1.0))):
        raise ValueError("image values must be finite and in [0, 1]")
    if tensor.shape[-1] == 1:
        tensor = tensor.expand(*tensor.shape[:-1], 3)
    return tensor[..., :3].contiguous()


def _mask(value: object | None) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    elif type(value) is torch.Tensor:
        tensor = value.detach()
    else:
        raise TypeError("mask must be a numpy array or exact torch.Tensor")
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or any(size <= 0 for size in tensor.shape):
        raise ValueError("mask must be nonempty [batch x H x W]")
    if not tensor.is_floating_point():
        raise TypeError("mask must contain floating-point values")
    tensor = tensor.to(device="cpu", dtype=torch.float32, copy=True).contiguous()
    if not bool(torch.isfinite(tensor).all()) or bool(torch.any((tensor < 0.0) | (tensor > 1.0))):
        raise ValueError("mask values must be finite and in [0, 1]")
    return tensor


def _finite(value: object, name: str, *, minimum: float, maximum: float) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a number")
    resolved = float(cast("int | float", value))
    if not math.isfinite(resolved) or not minimum <= resolved <= maximum:
        raise ValueError(f"{name} must be finite and within [{minimum}, {maximum}]")
    return resolved


def _prepare_conditioning(
    resource: SD15IPAdapterResource,
    reference: torch.Tensor,
    output_mask: torch.Tensor | None,
    window: PercentRange,
    strength: float,
) -> tuple[
    SD15AttentionContribution,
    torch.Tensor,
    torch.Tensor,
    str,
    str,
    str | None,
    tuple[str, ...],
]:
    mask_digest = (
        None
        if output_mask is None
        else sd15_ipadapter_tensor_digest(output_mask, role="output-mask")
    )
    with ExitStack() as stages, torch.inference_mode():
        stages.enter_context(resource.clip_vision.stage())
        stages.enter_context(resource.adapter.stage())
        vision = resource.clip_vision.component
        adapter_model = resource.adapter.component
        if type(vision) is not SD15IPAdapterClipVisionEncoder:
            raise TypeError("IP-Adapter resource does not contain its native CLIP vision encoder")
        if type(adapter_model) is not SD15IPAdapter:
            raise TypeError("IP-Adapter resource does not contain a native standard adapter")
        embedding = vision(
            reference.to(device=cast("torch.device", resource.clip_vision.load_device))
        )
        cond, uncond = adapter_model.project_image_embedding(embedding)
        cond = cond.detach().to(device="cpu", copy=True)
        uncond = uncond.detach().to(device="cpu", copy=True)
        model_digest = adapter_model.resource_digest
        if model_digest is None:
            raise RuntimeError("IP-Adapter has no assembly provenance")
        token_digest = sd15_ipadapter_tensor_digest(
            torch.cat((cond, uncond), dim=0), role="projected-tokens"
        )
        declaration = SD15AttentionContribution(
            PayloadReference(model_digest),
            PayloadReference(token_digest),
            window,
            strength,
            mask=None if mask_digest is None else PayloadReference(mask_digest),
        )
        proven = SD15IPAdapterConditioning(
            declaration,
            adapter_model,
            cond,
            uncond,
            model_digest,
            token_digest,
            output_mask,
            mask_digest,
        )
        facts = (
            f"adapter_asset={resource.adapter_asset_digest}",
            f"clip_vision_asset={resource.clip_vision_asset_digest}",
            f"clip_vision={resource.clip_vision.resource_identity}",
            f"canonical_sites={','.join(site.id for site in SD15_IPADAPTER_SITES)}",
            *sd15_ipadapter_identity_facts((proven,)),
        )
    return declaration, cond, uncond, model_digest, token_digest, mask_digest, facts


def execute_apply_sd15_ipadapter(
    *,
    model: object,
    ipadapter: object,
    image: object,
    strength: float,
    start_percent: float,
    end_percent: float,
    mask: object | None = None,
) -> Mapping[str, object]:
    if type(ipadapter) is not SD15IPAdapterResource:
        raise TypeError("ipadapter must come from Load SD1.5 IP-Adapter")
    resource = ipadapter
    strength = _finite(strength, "strength", minimum=-1.0, maximum=3.0)
    start = _finite(start_percent, "start_percent", minimum=0.0, maximum=1.0)
    end = _finite(end_percent, "end_percent", minimum=0.0, maximum=1.0)
    window = PercentRange(start, end)
    reference = _image(image)
    output_mask = _mask(mask)
    declaration, cond, uncond, model_digest, token_digest, mask_digest, facts = (
        _prepare_conditioning(resource, reference, output_mask, window, strength)
    )

    def materialize(
        _runtime: object,
        live_component: object,
        _latent: object,
    ) -> Mapping[str, object]:
        if type(live_component) is not SD15IPAdapter:
            raise TypeError("IP-Adapter application requires its native adapter component")
        return {
            "sd15_attention_contributions": (
                SD15IPAdapterConditioning(
                    declaration,
                    live_component,
                    cond,
                    uncond,
                    model_digest,
                    token_digest,
                    output_mask,
                    mask_digest,
                ),
            )
        }

    application = ComponentApplication(
        SD15.id,
        "diffusion",
        resource.adapter,
        extend_runtime_identity(resource.adapter.resource_identity, facts),
        materialize,
    )
    applied = (
        model.append(application)
        if isinstance(model, ApplicationChain)
        else ApplicationChain(model, (application,))
    )
    return {"model": applied}


__all__ = [
    "SD15IPAdapterResource",
    "execute_apply_sd15_ipadapter",
    "execute_load_sd15_ipadapter",
]
