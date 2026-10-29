"""Classic SD1.5 ControlNet module and invocation-local hint normalization."""

from __future__ import annotations

import ctypes
import sys
import weakref
from collections.abc import Generator, MutableMapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import torch
import torch.nn.functional as F
from blake3 import blake3
from dinkster_inference import (
    CompiledEffectMaskField,
    ContributionGain,
    EffectMaskInput,
    MaskMediaPlacement,
    ModelTokenLayout,
    ModelTokenSegment,
    TokenGridTransform,
    TokenRowTable,
    compile_effect_mask_field,
)
from dinkster_inference.controlnet import (
    SD15_CONTROL_RESIDUAL_SITES,
    SD_CONTROL_MODE_INDEX,
    SDXL_CONTROL_RESIDUAL_SITES,
    ControlApplication,
    ControlNetSourceLayout,
    SD15ControlNetConfig,
    SDControlMode,
    SDXLControlLoRAConfig,
    SDXLControlNetConfig,
    SDXLControlNetUnionConfig,
)

from .attention import AttentionKernel
from .model_prefetch import (
    close_prefetch_queue,
    make_prefetch_queue,
    prefetch_queue_pop,
)
from .operations import INITLESS, InitlessOperations, Operations, ResidencyRouted
from .unet import Downsample, ResBlock, SpatialTransformer, UNetModel, timestep_embedding

if TYPE_CHECKING:
    from .t2i_adapter import SD15T2IAdapter

SDControlProvider: TypeAlias = (
    "SD15ControlNet | SD15T2IAdapter | SDXLControlLoRA | SDXLControlNet | SDXLControlNetUnion"
)

__all__ = [
    "ControlResourceBindingError",
    "SDControlConditioning",
    "SDControlProvider",
    "SDEffectMaskField",
    "SDEffectMaskSource",
    "SD15ControlNet",
    "SDXLControlLoRA",
    "SDXLControlNet",
    "SDXLControlNetUnion",
    "SDControlResiduals",
    "compile_sd_effect_mask",
    "normalize_control_hint",
    "sd15_controlnet_resource_digest",
    "sdxl_control_lora_resource_digest",
    "sdxl_controlnet_resource_digest",
    "sdxl_controlnet_union_resource_digest",
    "sd_effect_mask_source_digest",
    "sd_control_hint_digest",
]


_DOWN_CHANNELS = (320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280)
_DOWN_SCALES = (1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8)
_SDXL_DOWN_CHANNELS = (320, 320, 320, 320, 640, 640, 640, 1280, 1280)
_SDXL_DOWN_SCALES = (1, 1, 1, 2, 2, 2, 4, 4, 4)
_EFFECT_MASK_TRANSFORM = "sd15.control-effect-mask.bilinear-align-corners-false.v1"
_HintCacheKey = tuple[int, int, int, int, torch.device, torch.dtype, int]
_SD15_CONFIG = SD15ControlNetConfig()
_SDXL_CONTROL_LORA_CONFIG = SDXLControlLoRAConfig()
_SDXL_CONTROLNET_CONFIG = SDXLControlNetConfig()
_SDXL_CONTROLNET_UNION_CONFIG = SDXLControlNetUnionConfig()


class ControlResourceBindingError(ValueError):
    """A declared ControlNet resource does not match its materialized value."""


@dataclass(frozen=True, slots=True)
class _ResourceTensorSeal:
    name: str
    tensor: torch.Tensor
    version: int
    dtype: torch.dtype
    shape: tuple[int, ...]
    storage: torch.UntypedStorage | None = None
    stride: tuple[int, ...] | None = None
    storage_offset: int | None = None
    conjugated: bool | None = None
    negated: bool | None = None


@dataclass(frozen=True, slots=True)
class _ControlNetResourceSeal:
    """Assembly provenance plus mutation evidence checked at admission.

    Resources detect unauthorized tensor, storage, and in-place mutation while
    accepting residency-owned replacements between declared devices.
    """

    digest: str
    tensors: tuple[_ResourceTensorSeal, ...]


def _require_blake3(name: str, digest: object) -> str:
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{name} must be a lowercase BLAKE3 digest")
    return digest


def _require_asset_digest(name: str, digest: object) -> str:
    if (
        type(digest) is not str
        or not digest.startswith("blake3:")
        or len(digest) != 71
        or any(character not in "0123456789abcdef" for character in digest[7:])
    ):
        raise ValueError(f"{name} must be a canonical blake3 asset digest")
    return digest


def sd_control_hint_digest(hint: torch.Tensor) -> str:
    """Content digest for one materialized SD1.5 ControlNet hint."""
    if type(hint) is not torch.Tensor:
        raise TypeError("control hint must be an exact torch.Tensor")
    if hint.layout is not torch.strided:
        raise TypeError("control hint must use strided tensor storage")
    value = hint.detach().resolve_conj().resolve_neg().contiguous().cpu()
    raw: bytes | bytearray = ctypes.string_at(
        value.data_ptr(), value.numel() * value.element_size()
    )
    width = value.element_size()
    if sys.byteorder == "big" and width > 1:
        raw = bytearray(raw)
        for start in range(0, len(raw), width):
            raw[start : start + width] = reversed(raw[start : start + width])
    hasher = blake3()
    hasher.update(b"dinkster.sd15-control-hint.v2\n")
    hasher.update(f"shape={','.join(str(dim) for dim in value.shape)}\n".encode("ascii"))
    hasher.update(f"dtype={str(value.dtype).removeprefix('torch.')}\n".encode("ascii"))
    hasher.update(b"byte_order=little\n\n")
    hasher.update(raw)
    return hasher.hexdigest()


def sd_effect_mask_source_digest(mask: torch.Tensor) -> str:
    """Canonical BLAKE3 asset identity for one float32 B/H/W mask."""
    if type(mask) is not torch.Tensor:
        raise TypeError("effect mask must be an exact torch.Tensor")
    if mask.layout is not torch.strided or mask.dtype is not torch.float32 or mask.ndim != 3:
        raise TypeError("effect mask must be a strided float32 [batch x H x W] tensor")
    value = mask.detach().resolve_conj().resolve_neg().contiguous().cpu()
    raw: bytes | bytearray = ctypes.string_at(value.data_ptr(), value.numel() * 4)
    if sys.byteorder == "big":
        raw = bytearray(raw)
        for start in range(0, len(raw), 4):
            raw[start : start + 4] = reversed(raw[start : start + 4])
    hasher = blake3()
    hasher.update(b"dinkster.sd15-effect-mask-source.v1\n")
    hasher.update(f"shape={','.join(str(dim) for dim in value.shape)}\n".encode("ascii"))
    hasher.update(b"dtype=float32\nbyte_order=little\n\n")
    hasher.update(raw)
    return "blake3:" + hasher.hexdigest()


def sd15_controlnet_resource_digest(
    asset_digest: str,
    source_layout: ControlNetSourceLayout,
    compute_dtype: torch.dtype,
) -> str:
    """Identity for loaded ControlNet weights and load-time behavior knobs."""
    _require_asset_digest("ControlNet source digest", asset_digest)
    if source_layout not in ("canonical", "diffusers"):
        raise ValueError("ControlNet source layout must be canonical or diffusers")
    if not compute_dtype.is_floating_point:
        raise TypeError("ControlNet compute dtype must be floating")
    hasher = blake3()
    hasher.update(b"dinkster.sd15-controlnet-resource.v2\n")
    hasher.update(f"source.asset_digest={asset_digest}\n".encode("ascii"))
    hasher.update(f"source_layout={source_layout}\n".encode("ascii"))
    hasher.update(f"compute_dtype={str(compute_dtype).removeprefix('torch.')}\n".encode("ascii"))
    return hasher.hexdigest()


def sdxl_control_lora_resource_digest(
    asset_digest: str,
    base_asset_digest: str,
    compute_dtype: torch.dtype,
) -> str:
    """Identity for one Control-LoRA artifact bound to an SDXL base model."""
    _require_asset_digest("Control-LoRA source digest", asset_digest)
    _require_asset_digest("Control-LoRA base model digest", base_asset_digest)
    if not compute_dtype.is_floating_point:
        raise TypeError("Control-LoRA compute dtype must be floating")
    hasher = blake3()
    hasher.update(b"dinkster.sdxl-control-lora-resource.v1\n")
    hasher.update(f"source.asset_digest={asset_digest}\n".encode("ascii"))
    hasher.update(f"base.asset_digest={base_asset_digest}\n".encode("ascii"))
    hasher.update(f"compute_dtype={str(compute_dtype).removeprefix('torch.')}\n".encode("ascii"))
    return hasher.hexdigest()


def sdxl_controlnet_resource_digest(
    asset_digest: str,
    config: SDXLControlNetConfig,
    compute_dtype: torch.dtype,
) -> str:
    """Identity for one assembled classic SDXL ControlNet artifact."""
    _require_asset_digest("SDXL ControlNet source digest", asset_digest)
    if type(config) is not SDXLControlNetConfig:
        raise TypeError("SDXL ControlNet config must be exact")
    if not compute_dtype.is_floating_point:
        raise TypeError("SDXL ControlNet compute dtype must be floating")
    hasher = blake3()
    hasher.update(b"dinkster.sdxl-controlnet-resource.v1\n")
    hasher.update(f"source.asset_digest={asset_digest}\n".encode("ascii"))
    hasher.update(f"hint_channels={config.hint_channels}\n".encode("ascii"))
    hasher.update(f"compute_dtype={str(compute_dtype).removeprefix('torch.')}\n".encode("ascii"))
    return hasher.hexdigest()


def sdxl_controlnet_union_resource_digest(
    asset_digest: str,
    config: SDXLControlNetUnionConfig,
    compute_dtype: torch.dtype,
) -> str:
    """Identity for one assembled SDXL ControlNet Union artifact."""
    _require_asset_digest("SDXL ControlNet Union source digest", asset_digest)
    if type(config) is not SDXLControlNetUnionConfig:
        raise TypeError("Union config must be an exact SDXLControlNetUnionConfig")
    if not compute_dtype.is_floating_point:
        raise TypeError("Union compute dtype must be floating")
    hasher = blake3()
    hasher.update(b"dinkster.sdxl-controlnet-union-resource.v1\n")
    hasher.update(f"source.asset_digest={asset_digest}\n".encode("ascii"))
    hasher.update(f"mode_capacity={config.mode_capacity}\n".encode("ascii"))
    hasher.update(f"compute_dtype={str(compute_dtype).removeprefix('torch.')}\n".encode("ascii"))
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class SDEffectMaskSource:
    """One resolved effect-mask artifact before family compilation."""

    declaration: EffectMaskInput
    mask: torch.Tensor

    def __post_init__(self) -> None:
        if type(self.declaration) is not EffectMaskInput:
            raise TypeError("effect-mask declaration must be an exact EffectMaskInput")
        if type(self.mask) is not torch.Tensor:
            raise TypeError("effect mask must be an exact torch.Tensor")
        if self.mask.ndim != 3 or self.mask.dtype is not torch.float32:
            raise TypeError("SD1.5 effect mask must be float32 [batch x H x W]")
        if tuple(self.mask.shape) != self.declaration.payload.shape:
            raise ControlResourceBindingError(
                "control-resource-binding-mismatch: effect-mask geometry does not match declaration"
            )
        if self.declaration.payload.dtype != "float32" or self.declaration.payload.space != "mask":
            raise ValueError("SD1.5 effect-mask payload must declare float32 mask space")
        if self.declaration.axis_identity != ("batch", "height", "width"):
            raise ValueError("SD1.5 effect-mask axes must be batch, height, width")
        if self.declaration.media_placement is not MaskMediaPlacement.FULL_DOMAIN:
            raise ValueError("SD1.5 effect masks require full-domain media placement")
        if self.declaration.target_segment not in (
            *SD15_CONTROL_RESIDUAL_SITES,
            *SDXL_CONTROL_RESIDUAL_SITES,
        ):
            raise ValueError("SD effect mask targets an unsupported residual site")
        if self.declaration.transform != _EFFECT_MASK_TRANSFORM:
            raise ValueError("SD1.5 effect mask requires the family bilinear transform")
        if self.mask.shape[0] < 1 or self.mask.shape[1] < 1 or self.mask.shape[2] < 1:
            raise ValueError("SD1.5 effect-mask geometry must be positive")
        if not bool(torch.isfinite(self.mask).all()) or not bool(
            ((self.mask >= 0.0) & (self.mask <= 1.0)).all()
        ):
            raise ValueError("SD1.5 effect-mask values must be finite and in [0, 1]")
        if sd_effect_mask_source_digest(self.mask) != self.declaration.source_digest:
            raise ControlResourceBindingError(
                "control-resource-binding-mismatch: effect-mask source digest does not match tensor"
            )


@dataclass(frozen=True, slots=True, init=False)
class SDEffectMaskField:
    """An SD residual-site mask and its generic compiled row field."""

    compiled: CompiledEffectMaskField
    mask: torch.Tensor
    shape: tuple[int, int, int, int]

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("SD effect-mask fields are created by compile_sd_effect_mask")


def compile_sd_effect_mask(
    source: SDEffectMaskSource,
    *,
    latent_height: int,
    latent_width: int,
) -> SDEffectMaskField:
    """Compile one full-domain source into its declared residual site."""
    if type(source) is not SDEffectMaskSource:
        raise TypeError("source must be an exact SDEffectMaskSource")
    if type(latent_height) is not int or type(latent_width) is not int:
        raise TypeError("latent geometry must use exact ints")
    if latent_height < 1 or latent_width < 1 or latent_height % 8 or latent_width % 8:
        raise ValueError("SD1.5 effect-mask latent geometry must be positive and divisible by 8")
    if source.declaration.target_segment in SD15_CONTROL_RESIDUAL_SITES:
        sites = SD15_CONTROL_RESIDUAL_SITES
        scales = _DOWN_SCALES
        middle_scale = 8
    else:
        sites = SDXL_CONTROL_RESIDUAL_SITES
        scales = _SDXL_DOWN_SCALES
        middle_scale = 4
    site_index = sites.index(source.declaration.target_segment)
    scale = scales[site_index] if site_index < len(scales) else middle_scale
    height, width = latent_height // scale, latent_width // scale
    normalized = F.interpolate(
        source.mask.unsqueeze(1),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).detach()
    layout = ModelTokenLayout(
        (
            ModelTokenSegment(
                source.declaration.target_segment,
                "image",
                "control-residual",
                0,
                height * width,
                (height, width),
            ),
        ),
        0,
    )
    transform = TokenGridTransform(
        _EFFECT_MASK_TRANSFORM,
        "image",
        source.declaration.target_segment,
        tuple(source.mask.shape[1:]),
        None,
    )
    row_values = tuple(
        tuple(float(value) for value in row)
        for row in normalized[:, 0].permute(1, 2, 0).reshape(height * width, source.mask.shape[0])
    )
    compiled = compile_effect_mask_field(
        source.declaration, layout, transform, TokenRowTable(row_values)
    )
    field = object.__new__(SDEffectMaskField)
    object.__setattr__(field, "compiled", compiled)
    object.__setattr__(field, "mask", normalized)
    object.__setattr__(field, "shape", tuple(normalized.shape))
    return field


def _snapshot_sd_effect_mask_field(  # pyright: ignore[reportUnusedFunction]
    field: SDEffectMaskField,
) -> SDEffectMaskField:
    """Own and validate an SD effect field at denoiser admission."""
    if type(field) is not SDEffectMaskField:
        raise TypeError("effect-mask field must be an exact SDEffectMaskField")
    mask = field.mask.detach().clone()
    if mask.layout is not torch.strided or mask.dtype is not torch.float32:
        raise ControlResourceBindingError(
            "control-resource-binding-mismatch: compiled effect-mask dtype or layout changed"
        )
    if tuple(mask.shape) != field.shape or mask.ndim != 4 or mask.shape[1] != 1:
        raise ControlResourceBindingError(
            "control-resource-binding-mismatch: compiled effect-mask tensor shape changed"
        )
    batch, _, height, width = field.shape
    expected = torch.tensor(
        field.compiled.table.values,
        device=mask.device,
        dtype=torch.float32,
    ).reshape(height, width, batch)
    expected = expected.permute(2, 0, 1).unsqueeze(1)
    if not torch.equal(mask, expected):
        raise ControlResourceBindingError(
            "control-resource-binding-mismatch: compiled effect-mask values changed after"
            " compilation"
        )
    snapshot = object.__new__(SDEffectMaskField)
    object.__setattr__(snapshot, "compiled", field.compiled)
    object.__setattr__(snapshot, "mask", mask)
    object.__setattr__(snapshot, "shape", field.shape)
    return snapshot


@dataclass(frozen=True, slots=True)
class SDControlConditioning:
    """One resolved SD1.5 ControlNet application for a sampling run.

    The application hint reference id is the materialized hint's content
    digest.
    """

    application: ControlApplication
    model: SDControlProvider
    hint: torch.Tensor
    model_digest: str
    hint_digest: str
    gain: ContributionGain | None = None
    previous: SDControlConditioning | None = None
    effect_masks: tuple[SDEffectMaskSource, ...] = ()

    def __post_init__(self) -> None:
        if type(self.application) is not ControlApplication:
            raise TypeError("control application must be an exact ControlApplication")
        from .t2i_adapter import SD15T2IAdapter, validate_sd15_t2i_adapter_resource

        if type(self.model) not in (
            SD15ControlNet,
            SD15T2IAdapter,
            SDXLControlLoRA,
            SDXLControlNet,
            SDXLControlNetUnion,
        ):
            raise TypeError("control model must be an exact SD control provider")
        if type(self.hint) is not torch.Tensor:
            raise TypeError("control hint must be an exact torch.Tensor")
        if type(self.model) is SD15T2IAdapter:
            hint_channels = 1
        elif type(self.model) is SDXLControlNet:
            hint_channels = self.model.config.hint_channels
        else:
            hint_channels = 3
        allowed_channels = (1, 3) if type(self.model) is SD15T2IAdapter else (hint_channels,)
        if self.hint.ndim != 4 or self.hint.shape[1] not in allowed_channels:
            raise ValueError(
                f"control hint must be [batch x {hint_channels} x H x W],"
                f" got {tuple(self.hint.shape)}"
            )
        if not self.hint.is_floating_point():
            raise TypeError("control hint must have a floating dtype")
        if self.hint.shape[0] < 1 or self.hint.shape[2] < 1 or self.hint.shape[3] < 1:
            raise ValueError("control hint batch and spatial geometry must be positive")
        model_digest = _require_blake3("control model digest", self.model_digest)
        hint_digest = _require_blake3("control hint digest", self.hint_digest)
        if type(self.model) is SD15ControlNet:
            _validate_sd15_controlnet_resource(self.model, model_digest)
        elif type(self.model) is SD15T2IAdapter:
            validate_sd15_t2i_adapter_resource(self.model, model_digest)
        elif type(self.model) is SDXLControlLoRA:
            _validate_sdxl_control_lora_resource(self.model, model_digest)
        elif type(self.model) is SDXLControlNet:
            _validate_sdxl_controlnet_resource(self.model, model_digest)
        else:
            _validate_sdxl_controlnet_union_resource(
                cast("SDXLControlNetUnion", self.model), model_digest
            )
        mode = self.application.mode
        if type(self.model) is SDXLControlNetUnion:
            if mode is not None and (
                type(mode) is not SDControlMode or mode.provider != "sdxl-controlnet-union"
            ):
                raise ValueError("SDXL ControlNet Union requires a Union control mode or auto")
            if mode is not None and (
                SD_CONTROL_MODE_INDEX[mode.token] >= self.model.config.mode_capacity
            ):
                raise ValueError(
                    "SDXL ControlNet Union mode exceeds the detected artifact capacity"
                )
        elif mode is not None:
            raise ValueError("non-Union SD control providers do not accept a control mode")
        if self.application.hint.id != hint_digest:
            raise ControlResourceBindingError(
                "control-resource-binding-mismatch: control hint reference does not resolve to"
                " the declared hint digest"
            )
        if sd_control_hint_digest(self.hint) != hint_digest:
            raise ControlResourceBindingError(
                "control-resource-binding-mismatch: declared hint digest does not match the"
                " materialized hint tensor"
            )
        if self.gain is not None and type(self.gain) is not ContributionGain:
            raise TypeError("control gain must be an exact ContributionGain or None")
        if type(self.effect_masks) is not tuple or any(
            type(mask) is not SDEffectMaskSource for mask in self.effect_masks
        ):
            raise TypeError("control effect masks must be exact SDEffectMaskSource values")
        mask_digests = tuple(mask.declaration.digest for mask in self.effect_masks)
        if len(mask_digests) != len(set(mask_digests)):
            raise ValueError("control effect-mask declaration digests must be unique")
        if self.previous is not None and type(self.previous) is not SDControlConditioning:
            raise TypeError("previous control must be an exact SDControlConditioning or None")
        previous_application = None if self.previous is None else self.previous.application
        if self.application.previous != previous_application:
            raise ControlResourceBindingError(
                "control-resource-binding-mismatch: control application chain does not match"
                " the resolved control resources"
            )


@dataclass(frozen=True)
class SDControlResiduals:
    """One exact SD1.5 or SDXL input/middle residual layout."""

    down: tuple[torch.Tensor, ...]
    middle: torch.Tensor
    down_channels: tuple[int, ...] = _DOWN_CHANNELS
    down_scales: tuple[int, ...] = _DOWN_SCALES

    def __post_init__(self) -> None:
        if (self.down_channels, self.down_scales) not in (
            (_DOWN_CHANNELS, _DOWN_SCALES),
            (_SDXL_DOWN_CHANNELS, _SDXL_DOWN_SCALES),
        ):
            raise ValueError("unsupported SD control residual site layout")
        if not isinstance(cast("object", self.down), tuple) or len(self.down) != len(
            self.down_channels
        ):
            raise ValueError("SD control residual count does not match its declared sites")
        if not all(isinstance(cast("object", value), torch.Tensor) for value in self.down):
            raise TypeError("SD down residuals must be tensors")
        if not isinstance(cast("object", self.middle), torch.Tensor):
            raise TypeError("SD middle residual must be a tensor")
        first = self.down[0]
        if not first.is_floating_point():
            raise TypeError("SD control residuals require a floating dtype")
        if first.ndim != 4:
            raise ValueError("SD control residuals must be rank-4 NCHW tensors")
        batch, _, height, width = first.shape
        if batch < 1 or height < 1 or width < 1 or height % 8 or width % 8:
            raise ValueError(
                "SD control residual base geometry must be positive and divisible by 8"
            )
        for index, (value, channels, scale) in enumerate(
            zip(self.down, self.down_channels, self.down_scales, strict=True)
        ):
            expected = (batch, channels, height // scale, width // scale)
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"down residual {index} shape {tuple(value.shape)} does not match {expected}"
                )
            if value.device != first.device:
                raise ValueError(f"down residual {index} device does not match residual 0")
            if value.dtype != first.dtype:
                raise ValueError(f"down residual {index} dtype does not match residual 0")
        middle_shape = (
            batch,
            self.down_channels[-1],
            height // self.down_scales[-1],
            width // self.down_scales[-1],
        )
        if tuple(self.middle.shape) != middle_shape:
            raise ValueError(
                f"middle residual shape {tuple(self.middle.shape)} does not match {middle_shape}"
            )
        if self.middle.device != first.device:
            raise ValueError("middle residual device does not match down residuals")
        if self.middle.dtype != first.dtype:
            raise ValueError("middle residual dtype does not match down residuals")


def normalize_control_hint(
    hint: torch.Tensor,
    *,
    latent_height: int,
    latent_width: int,
    batch: int,
    device: torch.device,
    compute_dtype: torch.dtype,
    expected_channels: int = 3,
    cache: MutableMapping[_HintCacheKey, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Center-crop, nearest-exact resize, batch, and convert one hint.

    A caller that normalizes repeatedly in one invocation supplies the
    invocation-owned cache and clears it on every exit. Without one, this call
    owns a temporary cache and clears it before returning.
    """
    allowed_channels = (1, 3) if expected_channels == 1 else (expected_channels,)
    if hint.ndim != 4 or hint.shape[1] not in allowed_channels:
        raise ValueError(
            f"control hint must be [batch x {expected_channels} x H x W], got {tuple(hint.shape)}"
        )
    if not hint.is_floating_point():
        raise TypeError("control hint must have a floating dtype")
    if not compute_dtype.is_floating_point:
        raise TypeError("control hint compute dtype must be floating")
    if latent_height < 1 or latent_width < 1 or batch < 1:
        raise ValueError("latent geometry and batch must be positive")
    source_batch, _, source_height, source_width = hint.shape
    if source_batch < 1 or source_height < 1 or source_width < 1:
        raise ValueError("control hint batch and spatial geometry must be positive")
    if source_batch > batch:
        raise ValueError(
            f"control hint batch {source_batch} cannot unambiguously map to latent batch {batch}"
        )

    target_height = latent_height * 8
    target_width = latent_width * 8
    key = (id(hint), target_height, target_width, batch, device, compute_dtype, expected_channels)
    owned_cache: MutableMapping[_HintCacheKey, torch.Tensor]
    owned_cache = {} if cache is None else cache
    try:
        cached = owned_cache.get(key)
        if cached is not None:
            return cached

        normalized = hint
        if source_height != target_height or source_width != target_width:
            old_aspect = source_width / source_height
            new_aspect = target_width / target_height
            crop_x = 0
            crop_y = 0
            if old_aspect > new_aspect:
                crop_x = round((source_width - source_width * (new_aspect / old_aspect)) / 2)
            elif old_aspect < new_aspect:
                crop_y = round((source_height - source_height * (old_aspect / new_aspect)) / 2)
            normalized = hint.narrow(-2, crop_y, source_height - crop_y * 2).narrow(
                -1, crop_x, source_width - crop_x * 2
            )
            normalized = F.interpolate(
                normalized,
                size=(target_height, target_width),
                mode="nearest-exact",
            )
        if expected_channels == 1 and normalized.shape[1] == 3:
            normalized = normalized.float().to(device).mean(dim=1, keepdim=True)
        if source_batch < batch:
            repeats = -(batch // -source_batch)
            normalized = normalized.repeat(repeats, *([1] * (normalized.ndim - 1))).narrow(
                0, 0, batch
            )
        normalized = normalized.to(device=device, dtype=compute_dtype)
        owned_cache[key] = normalized
        return normalized
    finally:
        if cache is None:
            owned_cache.clear()


class _Layers(torch.nn.Sequential):
    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class _UnionAttention(torch.nn.Module):
    _attention_kernel: AttentionKernel

    def __init__(
        self,
        width: int,
        heads: int,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.width = width
        self.heads = heads
        object.__setattr__(self, "_attention_kernel", attention_kernel)
        self.in_proj = operations.linear(width, width * 3)
        self.out_proj = operations.linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        batch, tokens, width = q.shape
        head_width = width // self.heads

        def heads(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch, tokens, self.heads, head_width).transpose(1, 2)

        attended = self._attention_kernel(heads(q), heads(k), heads(v))
        return self.out_proj(attended.transpose(1, 2).reshape(batch, tokens, width))


class _QuickGELU(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class _UnionMLP(torch.nn.Module):
    def __init__(self, width: int, operations: Operations) -> None:
        super().__init__()
        self.c_fc = operations.linear(width, width * 4)
        self.gelu = _QuickGELU()
        self.c_proj = operations.linear(width * 4, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class _UnionTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.attn = _UnionAttention(width, heads, operations, attention_kernel)
        self.ln_1 = operations.layer_norm(width)
        self.mlp = _UnionMLP(width, operations)
        self.ln_2 = operations.layer_norm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class _ControlAddEmbedding(torch.nn.Module):
    def __init__(self, capacity: int, output_width: int, operations: Operations) -> None:
        super().__init__()
        self.capacity = capacity
        self.linear_1 = operations.linear(256 * capacity, output_width)
        self.linear_2 = operations.linear(output_width, output_width)

    def forward(
        self, mode_index: int | None, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        control_type = torch.zeros(self.capacity, device=device)
        if mode_index is not None:
            control_type[mode_index] = 1.0
        embedding = timestep_embedding(control_type, 256).to(dtype).reshape(1, -1)
        return self.linear_2(F.silu(self.linear_1(embedding)))


class SDXLControlNetUnion(ResidencyRouted, torch.nn.Module):
    """xinsir SDXL ControlNet Union with automatic or explicit semantic mode selection."""

    def __init__(
        self,
        config: SDXLControlNetUnionConfig = _SDXL_CONTROLNET_UNION_CONFIG,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.config = config
        base = config.base
        channels = base.model_channels
        embedding_channels = base.time_embed_dim
        self.time_embed = torch.nn.Sequential(
            operations.linear(channels, embedding_channels),
            torch.nn.SiLU(),
            operations.linear(embedding_channels, embedding_channels),
        )
        assert base.adm_in_channels is not None
        self.label_emb = torch.nn.Sequential(
            torch.nn.Sequential(
                operations.linear(base.adm_in_channels, embedding_channels),
                torch.nn.SiLU(),
                operations.linear(embedding_channels, embedding_channels),
            )
        )

        def transformer(width: int, depth: int) -> SpatialTransformer:
            heads, head_dim = base.heads_for(width)
            return SpatialTransformer(
                width,
                heads,
                head_dim,
                depth,
                base.context_dim,
                base.use_linear_in_transformer,
                operations=operations,
                attention_kernel=attention_kernel,
            )

        self.input_blocks = torch.nn.ModuleList(
            [_Layers(operations.conv2d(base.in_channels, channels, 3, padding=1))]
        )
        self.zero_convs = torch.nn.ModuleList([self._zero_conv(channels, operations)])
        self.input_hint_block = _Layers(
            operations.conv2d(config.hint_channels, 16, 3, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(16, 16, 3, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(16, 32, 3, stride=2, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(32, 32, 3, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(32, 96, 3, stride=2, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(96, 96, 3, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(96, 256, 3, stride=2, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(256, channels, 3, padding=1),
        )
        depths = list(base.transformer_depth)
        for level, multiplier in enumerate(base.channel_mult):
            for _ in range(base.num_res_blocks[level]):
                out_channels = multiplier * base.model_channels
                layers: list[torch.nn.Module] = [
                    ResBlock(channels, embedding_channels, out_channels, operations=operations)
                ]
                channels = out_channels
                depth = depths.pop(0)
                if depth:
                    layers.append(transformer(channels, depth))
                self.input_blocks.append(_Layers(*layers))
                self.zero_convs.append(self._zero_conv(channels, operations))
            if level != len(base.channel_mult) - 1:
                self.input_blocks.append(_Layers(Downsample(channels, operations=operations)))
                self.zero_convs.append(self._zero_conv(channels, operations))
        self.middle_block = _Layers(
            ResBlock(channels, embedding_channels, channels, operations=operations),
            transformer(channels, base.transformer_depth_middle),
            ResBlock(channels, embedding_channels, channels, operations=operations),
        )
        self.middle_block_out = self._zero_conv(channels, operations)
        self.task_embedding = torch.nn.Parameter(torch.empty(config.mode_capacity, 320))
        self.transformer_layes = torch.nn.Sequential(
            _UnionTransformerBlock(320, 8, operations, attention_kernel)
        )
        self.spatial_ch_projs = operations.linear(320, 320)
        self.control_add_embedding = _ControlAddEmbedding(
            config.mode_capacity, embedding_channels, operations
        )

    @staticmethod
    def _zero_conv(channels: int, operations: Operations) -> torch.nn.Sequential:
        return torch.nn.Sequential(operations.conv2d(channels, channels, 1))

    def forward(
        self,
        x: torch.Tensor,
        hint: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        y: torch.Tensor,
        mode_index: int | None,
    ) -> SDControlResiduals:
        self._validate_inputs(x, hint, timesteps, context, y, mode_index)
        hint_cache: dict[_HintCacheKey, torch.Tensor] = {}
        try:
            hint = normalize_control_hint(
                hint,
                latent_height=x.shape[2],
                latent_width=x.shape[3],
                batch=x.shape[0],
                device=x.device,
                compute_dtype=x.dtype,
                cache=hint_cache,
            )
            embedding = self.time_embed(
                timestep_embedding(timesteps, self.config.base.model_channels).to(x.dtype)
            )
            embedding = embedding + self.control_add_embedding(mode_index, x.dtype, x.device)
            guided_hint = self.input_hint_block(hint, embedding, context)
            if mode_index is not None:
                pooled = guided_hint.mean(dim=(2, 3))
                binding = self._offloaded_residency()
                if binding is None:
                    pooled = pooled + self.task_embedding[mode_index].to(pooled)
                else:
                    with binding.lease() as lease:
                        pooled = (
                            pooled + lease.get("task_embedding", dtype=pooled.dtype)[mode_index]
                        )
                transformed = self.transformer_layes(pooled.unsqueeze(1))[:, 0]
                guided_hint = guided_hint + self.spatial_ch_projs(transformed)[..., None, None]
            embedding = embedding + self.label_emb(y)
            return self._run_control(x, embedding, guided_hint, context)
        finally:
            hint_cache.clear()

    def _run_control(
        self,
        x: torch.Tensor,
        embedding: torch.Tensor,
        guided_hint: torch.Tensor,
        context: torch.Tensor,
    ) -> SDControlResiduals:
        h = x
        down: list[torch.Tensor] = []
        prefetch = make_prefetch_queue(self.input_blocks)
        try:
            for index, (module, zero_conv) in enumerate(
                zip(self.input_blocks, self.zero_convs, strict=True)
            ):
                prefetch_queue_pop(prefetch, module)
                h = module(h, embedding, context)
                if index == 0:
                    h = h + guided_hint
                down.append(zero_conv(h))
            prefetch_queue_pop(prefetch, None)
        finally:
            close_prefetch_queue(prefetch)
        h = self.middle_block(h, embedding, context)
        return SDControlResiduals(
            tuple(down),
            self.middle_block_out(h),
            _SDXL_DOWN_CHANNELS,
            _SDXL_DOWN_SCALES,
        )

    def _validate_inputs(
        self,
        x: torch.Tensor,
        hint: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        y: torch.Tensor,
        mode_index: int | None,
    ) -> None:
        base = self.config.base
        if x.ndim != 4 or x.shape[1] != base.in_channels or x.shape[2] % 8 or x.shape[3] % 8:
            raise ValueError("SDXL ControlNet Union x has the wrong geometry")
        if x.shape[0] < 1 or x.shape[2] < 8 or x.shape[3] < 8 or not x.is_floating_point():
            raise ValueError("SDXL ControlNet Union x must have positive floating geometry")
        if hint.ndim != 4 or hint.shape[1] != self.config.hint_channels:
            raise ValueError("SDXL ControlNet Union hint must be [batch x 3 x H x W]")
        if not hint.is_floating_point():
            raise TypeError("SDXL ControlNet Union hint must have a floating dtype")
        if timesteps.ndim != 1 or timesteps.shape[0] != x.shape[0]:
            raise ValueError("SDXL ControlNet Union timesteps must match the latent batch")
        if context.ndim != 3 or tuple(context.shape[::2]) != (x.shape[0], base.context_dim):
            raise ValueError("SDXL ControlNet Union context has the wrong geometry")
        if y.ndim != 2 or y.shape != (x.shape[0], base.adm_in_channels):
            raise ValueError("SDXL ControlNet Union ADM conditioning has the wrong geometry")
        if any(value.device != x.device for value in (timesteps, context, y)):
            raise ValueError("SDXL ControlNet Union conditioning devices must match x")
        if context.dtype != x.dtype or y.dtype != x.dtype:
            raise ValueError("SDXL ControlNet Union context and ADM dtypes must match x")
        if mode_index is not None and (
            type(mode_index) is not int or not 0 <= mode_index < self.config.mode_capacity
        ):
            raise ValueError("SDXL ControlNet Union mode index exceeds artifact capacity")

    @property
    def resource_digest(self) -> str | None:
        seal = _CONTROL_UNION_RESOURCE_SEALS.get(self)
        return None if seal is None else seal.digest


class SDXLControlNet(torch.nn.Module):
    """Classic SDXL ControlNet with nine down residuals and one middle residual."""

    def __init__(
        self,
        config: SDXLControlNetConfig = _SDXL_CONTROLNET_CONFIG,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.config = config
        base = config.base
        channels = base.model_channels
        embedding_channels = base.time_embed_dim
        self.time_embed = torch.nn.Sequential(
            operations.linear(channels, embedding_channels),
            torch.nn.SiLU(),
            operations.linear(embedding_channels, embedding_channels),
        )
        assert base.adm_in_channels is not None
        self.label_emb = torch.nn.Sequential(
            torch.nn.Sequential(
                operations.linear(base.adm_in_channels, embedding_channels),
                torch.nn.SiLU(),
                operations.linear(embedding_channels, embedding_channels),
            )
        )

        def transformer(width: int, depth: int) -> SpatialTransformer:
            heads, head_dim = base.heads_for(width)
            return SpatialTransformer(
                width,
                heads,
                head_dim,
                depth,
                base.context_dim,
                base.use_linear_in_transformer,
                operations=operations,
            )

        self.input_blocks = torch.nn.ModuleList(
            [_Layers(operations.conv2d(base.in_channels, channels, 3, padding=1))]
        )
        self.zero_convs = torch.nn.ModuleList([self._zero_conv(channels, operations)])
        self.input_hint_block = _Layers(
            operations.conv2d(config.hint_channels, 16, 3, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(16, 16, 3, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(16, 32, 3, stride=2, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(32, 32, 3, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(32, 96, 3, stride=2, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(96, 96, 3, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(96, 256, 3, stride=2, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(256, channels, 3, padding=1),
        )
        depths = list(base.transformer_depth)
        for level, multiplier in enumerate(base.channel_mult):
            for _ in range(base.num_res_blocks[level]):
                out_channels = multiplier * base.model_channels
                layers: list[torch.nn.Module] = [
                    ResBlock(channels, embedding_channels, out_channels, operations=operations)
                ]
                channels = out_channels
                depth = depths.pop(0)
                if depth:
                    layers.append(transformer(channels, depth))
                self.input_blocks.append(_Layers(*layers))
                self.zero_convs.append(self._zero_conv(channels, operations))
            if level != len(base.channel_mult) - 1:
                self.input_blocks.append(_Layers(Downsample(channels, operations=operations)))
                self.zero_convs.append(self._zero_conv(channels, operations))
        self.middle_block = _Layers(
            ResBlock(channels, embedding_channels, channels, operations=operations),
            transformer(channels, base.transformer_depth_middle),
            ResBlock(channels, embedding_channels, channels, operations=operations),
        )
        self.middle_block_out = self._zero_conv(channels, operations)

    @staticmethod
    def _zero_conv(channels: int, operations: Operations) -> torch.nn.Sequential:
        return torch.nn.Sequential(operations.conv2d(channels, channels, 1))

    def forward(
        self,
        x: torch.Tensor,
        hint: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        y: torch.Tensor,
    ) -> SDControlResiduals:
        self._validate_inputs(x, hint, timesteps, context, y)
        hint_cache: dict[_HintCacheKey, torch.Tensor] = {}
        try:
            hint = normalize_control_hint(
                hint,
                latent_height=x.shape[2],
                latent_width=x.shape[3],
                batch=x.shape[0],
                device=x.device,
                compute_dtype=x.dtype,
                expected_channels=self.config.hint_channels,
                cache=hint_cache,
            )
            embedding = self.time_embed(
                timestep_embedding(timesteps, self.config.base.model_channels).to(x.dtype)
            )
            guided_hint = self.input_hint_block(hint, embedding, context)
            embedding = embedding + self.label_emb(y)
            h = x
            down: list[torch.Tensor] = []
            prefetch = make_prefetch_queue(self.input_blocks)
            try:
                for index, (module, zero_conv) in enumerate(
                    zip(self.input_blocks, self.zero_convs, strict=True)
                ):
                    prefetch_queue_pop(prefetch, module)
                    h = module(h, embedding, context)
                    if index == 0:
                        h = h + guided_hint
                    down.append(zero_conv(h))
                prefetch_queue_pop(prefetch, None)
            finally:
                close_prefetch_queue(prefetch)
            h = self.middle_block(h, embedding, context)
            return SDControlResiduals(
                tuple(down),
                self.middle_block_out(h),
                _SDXL_DOWN_CHANNELS,
                _SDXL_DOWN_SCALES,
            )
        finally:
            hint_cache.clear()

    def _validate_inputs(
        self,
        x: torch.Tensor,
        hint: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        base = self.config.base
        if x.ndim != 4 or x.shape[1] != base.in_channels or x.shape[2] % 8 or x.shape[3] % 8:
            raise ValueError("SDXL ControlNet x has the wrong geometry")
        if x.shape[0] < 1 or x.shape[2] < 8 or x.shape[3] < 8 or not x.is_floating_point():
            raise ValueError("SDXL ControlNet x must have positive floating geometry")
        if hint.ndim != 4 or hint.shape[1] != self.config.hint_channels:
            raise ValueError(
                f"SDXL ControlNet hint must be [batch x {self.config.hint_channels} x H x W]"
            )
        if not hint.is_floating_point():
            raise TypeError("SDXL ControlNet hint must have a floating dtype")
        if timesteps.ndim != 1 or timesteps.shape[0] != x.shape[0]:
            raise ValueError("SDXL ControlNet timesteps must match the latent batch")
        if context.ndim != 3 or tuple(context.shape[::2]) != (x.shape[0], base.context_dim):
            raise ValueError("SDXL ControlNet context has the wrong geometry")
        if y.ndim != 2 or y.shape != (x.shape[0], base.adm_in_channels):
            raise ValueError("SDXL ControlNet ADM conditioning has the wrong geometry")
        if any(value.device != x.device for value in (timesteps, context, y)):
            raise ValueError("SDXL ControlNet conditioning devices must match x")
        if context.dtype != x.dtype or y.dtype != x.dtype:
            raise ValueError("SDXL ControlNet context and ADM dtypes must match x")

    @property
    def resource_digest(self) -> str | None:
        seal = _SDXL_CONTROLNET_RESOURCE_SEALS.get(self)
        return None if seal is None else seal.digest


class _ControlLoRAWeights(ResidencyRouted):
    up: torch.nn.Parameter | None
    down: torch.nn.Parameter | None

    @contextmanager
    def _control_weights(
        self, input: torch.Tensor
    ) -> Generator[tuple[torch.Tensor, torch.Tensor | None]]:
        layer = cast("torch.nn.Linear | torch.nn.Conv2d", self)
        binding = self._offloaded_residency()
        with ExitStack() as stack:
            lease = None if binding is None else stack.enter_context(binding.lease())

            def materialize(name: str, value: torch.Tensor) -> torch.Tensor:
                if lease is not None:
                    return lease.get(name, dtype=input.dtype)
                return value.to(device=input.device, dtype=input.dtype)

            weight = materialize("weight", layer.weight)
            bias = None if layer.bias is None else materialize("bias", layer.bias)
            if self.up is not None and self.down is not None:
                up = materialize("up", self.up)
                down = materialize("down", self.down)
                weight = weight + torch.mm(
                    up.flatten(start_dim=1), down.flatten(start_dim=1)
                ).reshape(weight.shape)
            yield weight, bias


class _ControlLoRALinear(_ControlLoRAWeights, torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, *, bias: bool = True) -> None:
        torch.nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.empty(out_features)) if bias else None
        self.up = None
        self.down = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        with self._control_weights(input) as (weight, bias):
            return F.linear(input, weight, bias)


class _ControlLoRAConv2d(_ControlLoRAWeights, torch.nn.Conv2d):
    def reset_parameters(self) -> None:
        return None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.up = None
        self.down = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        with self._control_weights(input) as (weight, bias):
            return self._conv_forward(input, weight, bias)


class _ControlLoRAOperations(InitlessOperations):
    def linear(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
    ) -> torch.nn.Linear:
        return _ControlLoRALinear(in_features, out_features, bias=bias)

    def conv2d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
    ) -> torch.nn.Conv2d:
        return _ControlLoRAConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )


class SDXLControlLoRA(torch.nn.Module):
    """SDXL Control-LoRA over an assembly-bound base UNet."""

    def __init__(self, config: SDXLControlLoRAConfig = _SDXL_CONTROL_LORA_CONFIG) -> None:
        super().__init__()
        self.config = config
        base = config.base
        operations = _ControlLoRAOperations()
        channels = base.model_channels
        embedding_channels = base.time_embed_dim
        self.time_embed = torch.nn.Sequential(
            operations.linear(channels, embedding_channels),
            torch.nn.SiLU(),
            operations.linear(embedding_channels, embedding_channels),
        )
        assert base.adm_in_channels is not None
        self.label_emb = torch.nn.Sequential(
            torch.nn.Sequential(
                operations.linear(base.adm_in_channels, embedding_channels),
                torch.nn.SiLU(),
                operations.linear(embedding_channels, embedding_channels),
            )
        )

        def transformer(width: int, depth: int) -> SpatialTransformer:
            heads, head_dim = base.heads_for(width)
            return SpatialTransformer(
                width,
                heads,
                head_dim,
                depth,
                base.context_dim,
                base.use_linear_in_transformer,
                operations=operations,
            )

        self.input_blocks = torch.nn.ModuleList(
            [_Layers(operations.conv2d(base.in_channels, channels, 3, padding=1))]
        )
        self.zero_convs = torch.nn.ModuleList([self._zero_conv(channels, operations)])
        self.input_hint_block = _Layers(
            operations.conv2d(config.hint_channels, 16, 3, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(16, 16, 3, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(16, 32, 3, stride=2, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(32, 32, 3, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(32, 96, 3, stride=2, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(96, 96, 3, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(96, 256, 3, stride=2, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(256, channels, 3, padding=1),
        )
        depths = list(base.transformer_depth)
        for level, multiplier in enumerate(base.channel_mult):
            for _ in range(base.num_res_blocks[level]):
                out_channels = multiplier * base.model_channels
                layers: list[torch.nn.Module] = [
                    ResBlock(channels, embedding_channels, out_channels, operations=operations)
                ]
                channels = out_channels
                depth = depths.pop(0)
                if depth:
                    layers.append(transformer(channels, depth))
                self.input_blocks.append(_Layers(*layers))
                self.zero_convs.append(self._zero_conv(channels, operations))
            if level != len(base.channel_mult) - 1:
                self.input_blocks.append(_Layers(Downsample(channels, operations=operations)))
                self.zero_convs.append(self._zero_conv(channels, operations))
        self.middle_block = _Layers(
            ResBlock(channels, embedding_channels, channels, operations=operations),
            transformer(channels, base.transformer_depth_middle),
            ResBlock(channels, embedding_channels, channels, operations=operations),
        )
        self.middle_block_out = self._zero_conv(channels, operations)

    @staticmethod
    def _zero_conv(channels: int, operations: Operations) -> torch.nn.Sequential:
        return torch.nn.Sequential(operations.conv2d(channels, channels, 1))

    def forward(
        self,
        x: torch.Tensor,
        hint: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        y: torch.Tensor,
    ) -> SDControlResiduals:
        self._validate_inputs(x, hint, timesteps, context, y)
        hint_cache: dict[_HintCacheKey, torch.Tensor] = {}
        try:
            guided_hint = normalize_control_hint(
                hint,
                latent_height=x.shape[2],
                latent_width=x.shape[3],
                batch=x.shape[0],
                device=x.device,
                compute_dtype=x.dtype,
                cache=hint_cache,
            )
            guided_hint = self.input_hint_block(
                guided_hint,
                torch.empty(0, device=x.device, dtype=x.dtype),
                context,
            )
            embedding = self.time_embed(
                timestep_embedding(timesteps, self.config.base.model_channels).to(x.dtype)
            )
            embedding = embedding + self.label_emb(y)
            h = x
            down: list[torch.Tensor] = []
            prefetch = make_prefetch_queue(self.input_blocks)
            try:
                for index, (module, zero_conv) in enumerate(
                    zip(self.input_blocks, self.zero_convs, strict=True)
                ):
                    prefetch_queue_pop(prefetch, module)
                    h = module(h, embedding, context)
                    if index == 0:
                        h = h + guided_hint
                    down.append(zero_conv(h))
                prefetch_queue_pop(prefetch, None)
            finally:
                close_prefetch_queue(prefetch)
            h = self.middle_block(h, embedding, context)
            return SDControlResiduals(
                tuple(down),
                self.middle_block_out(h),
                _SDXL_DOWN_CHANNELS,
                _SDXL_DOWN_SCALES,
            )
        finally:
            hint_cache.clear()

    def _validate_inputs(
        self,
        x: torch.Tensor,
        hint: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        base = self.config.base
        if x.ndim != 4 or x.shape[1] != base.in_channels:
            raise ValueError(f"SDXL Control-LoRA x must be [batch x {base.in_channels} x H x W]")
        if x.shape[0] < 1 or x.shape[2] < 8 or x.shape[3] < 8:
            raise ValueError("SDXL Control-LoRA x geometry must be positive")
        if x.shape[2] % 8 or x.shape[3] % 8:
            raise ValueError("SDXL Control-LoRA latent geometry must be divisible by 8")
        if not x.is_floating_point():
            raise TypeError("SDXL Control-LoRA x must have a floating dtype")
        if hint.ndim != 4 or hint.shape[1] != self.config.hint_channels:
            raise ValueError("SDXL Control-LoRA hint must be [batch x 3 x H x W]")
        if not hint.is_floating_point():
            raise TypeError("SDXL Control-LoRA hint must have a floating dtype")
        if timesteps.ndim != 1 or timesteps.shape[0] != x.shape[0]:
            raise ValueError("SDXL Control-LoRA timesteps must match the latent batch")
        if not timesteps.is_floating_point():
            raise TypeError("SDXL Control-LoRA timesteps must have a floating dtype")
        if context.ndim != 3 or tuple(context.shape[::2]) != (x.shape[0], base.context_dim):
            raise ValueError("SDXL Control-LoRA context has the wrong geometry")
        if y.ndim != 2 or y.shape != (x.shape[0], base.adm_in_channels):
            raise ValueError("SDXL Control-LoRA ADM conditioning has the wrong shape")
        if any(value.device != x.device for value in (timesteps, context, y)):
            raise ValueError("SDXL Control-LoRA conditioning devices must match x")
        if context.dtype != x.dtype or y.dtype != x.dtype:
            raise ValueError("SDXL Control-LoRA context and ADM dtypes must match x")

    @property
    def resource_digest(self) -> str | None:
        seal = _CONTROL_LORA_RESOURCE_SEALS.get(self)
        return None if seal is None else seal.digest


class SD15ControlNet(torch.nn.Module):
    """The copied SD1.5 UNet input/middle path with ControlNet outputs."""

    def __init__(
        self,
        config: SD15ControlNetConfig = _SD15_CONFIG,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.config = config
        m = config.model_channels
        emb_channels = m * 4

        self.time_embed = torch.nn.Sequential(
            operations.linear(m, emb_channels),
            torch.nn.SiLU(),
            operations.linear(emb_channels, emb_channels),
        )
        self.input_blocks = torch.nn.ModuleList(
            [_Layers(operations.conv2d(config.in_channels, m, 3, padding=1))]
        )
        self.zero_convs = torch.nn.ModuleList([self._zero_conv(m, operations)])
        self.input_hint_block = _Layers(
            operations.conv2d(config.hint_channels, 16, 3, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(16, 16, 3, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(16, 32, 3, stride=2, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(32, 32, 3, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(32, 96, 3, stride=2, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(96, 96, 3, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(96, 256, 3, stride=2, padding=1),
            torch.nn.SiLU(),
            operations.conv2d(256, m, 3, padding=1),
        )

        def transformer(channels: int, depth: int) -> SpatialTransformer:
            return SpatialTransformer(
                channels,
                config.num_heads,
                channels // config.num_heads,
                depth,
                config.context_dim,
                False,
                operations=operations,
            )

        depths = list(config.transformer_depth)
        channels = m
        for level, multiplier in enumerate(config.channel_mult):
            for _ in range(config.num_res_blocks[level]):
                out_channels = multiplier * m
                layers: list[torch.nn.Module] = [
                    ResBlock(
                        channels,
                        emb_channels,
                        out_channels,
                        operations=operations,
                    )
                ]
                channels = out_channels
                depth = depths.pop(0)
                if depth > 0:
                    layers.append(transformer(channels, depth))
                self.input_blocks.append(_Layers(*layers))
                self.zero_convs.append(self._zero_conv(channels, operations))
            if level != len(config.channel_mult) - 1:
                self.input_blocks.append(_Layers(Downsample(channels, operations=operations)))
                self.zero_convs.append(self._zero_conv(channels, operations))

        self.middle_block = _Layers(
            ResBlock(channels, emb_channels, channels, operations=operations),
            transformer(channels, config.transformer_depth_middle),
            ResBlock(channels, emb_channels, channels, operations=operations),
        )
        self.middle_block_out = self._zero_conv(channels, operations)

    @staticmethod
    def _zero_conv(channels: int, operations: Operations) -> torch.nn.Sequential:
        return torch.nn.Sequential(operations.conv2d(channels, channels, 1))

    def forward(
        self,
        x: torch.Tensor,
        hint: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> SDControlResiduals:
        self._validate_inputs(x, hint, timesteps, context)
        hint_cache: dict[_HintCacheKey, torch.Tensor] = {}
        try:
            guided_hint = normalize_control_hint(
                hint,
                latent_height=x.shape[2],
                latent_width=x.shape[3],
                batch=x.shape[0],
                device=x.device,
                compute_dtype=x.dtype,
                cache=hint_cache,
            )
            guided_hint = self.input_hint_block(
                guided_hint,
                torch.empty(0, device=x.device, dtype=x.dtype),
                context,
            )
            embedding = timestep_embedding(timesteps, self.config.model_channels).to(x.dtype)
            embedding = self.time_embed(embedding)

            h = x
            down: list[torch.Tensor] = []
            prefetch = make_prefetch_queue(self.input_blocks)
            try:
                for index, (module, zero_conv) in enumerate(
                    zip(self.input_blocks, self.zero_convs, strict=True)
                ):
                    prefetch_queue_pop(prefetch, module)
                    h = module(h, embedding, context)
                    if index == 0:
                        h = h + guided_hint
                    down.append(zero_conv(h))
                prefetch_queue_pop(prefetch, None)
            finally:
                close_prefetch_queue(prefetch)
            h = self.middle_block(h, embedding, context)
            return SDControlResiduals(tuple(down), self.middle_block_out(h))
        finally:
            hint_cache.clear()

    def _validate_inputs(
        self,
        x: torch.Tensor,
        hint: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> None:
        if x.ndim != 4 or x.shape[1] != self.config.in_channels:
            raise ValueError(
                f"ControlNet x must be [batch x {self.config.in_channels} x H x W],"
                f" got {tuple(x.shape)}"
            )
        if x.shape[0] < 1 or x.shape[2] < 8 or x.shape[3] < 8:
            raise ValueError("ControlNet x batch and spatial geometry must be positive")
        if x.shape[2] % 8 or x.shape[3] % 8:
            raise ValueError("ControlNet latent height and width must be divisible by 8")
        if not x.is_floating_point():
            raise TypeError("ControlNet x must have a floating dtype")
        if hint.ndim != 4 or hint.shape[1] != self.config.hint_channels:
            raise ValueError(
                f"ControlNet hint must be [batch x {self.config.hint_channels} x H x W],"
                f" got {tuple(hint.shape)}"
            )
        if not hint.is_floating_point():
            raise TypeError("ControlNet hint must have a floating dtype")
        if timesteps.ndim != 1 or timesteps.shape[0] != x.shape[0]:
            raise ValueError("ControlNet timesteps must be rank 1 with the latent batch")
        if not timesteps.is_floating_point():
            raise TypeError("ControlNet timesteps must have a floating dtype")
        if context.ndim != 3 or tuple(context.shape[::2]) != (
            x.shape[0],
            self.config.context_dim,
        ):
            raise ValueError(
                f"ControlNet context must be [latent batch x tokens x {self.config.context_dim}]"
            )
        if timesteps.device != x.device:
            raise ValueError("ControlNet timesteps device must match x")
        if context.device != x.device:
            raise ValueError("ControlNet context device must match x")
        if context.dtype != x.dtype:
            raise ValueError("ControlNet context dtype must match x")

    @property
    def resource_digest(self) -> str | None:
        """The assembly-authoritative resource digest, if one is bound."""
        seal = _CONTROLNET_RESOURCE_SEALS.get(self)
        return None if seal is None else seal.digest


_CONTROLNET_RESOURCE_SEALS: weakref.WeakKeyDictionary[SD15ControlNet, _ControlNetResourceSeal] = (
    weakref.WeakKeyDictionary()
)
_CONTROL_LORA_RESOURCE_SEALS: weakref.WeakKeyDictionary[
    SDXLControlLoRA, _ControlNetResourceSeal
] = weakref.WeakKeyDictionary()
_SDXL_CONTROLNET_RESOURCE_SEALS: weakref.WeakKeyDictionary[
    SDXLControlNet, _ControlNetResourceSeal
] = weakref.WeakKeyDictionary()
_CONTROL_UNION_RESOURCE_SEALS: weakref.WeakKeyDictionary[
    SDXLControlNetUnion, _ControlNetResourceSeal
] = weakref.WeakKeyDictionary()
_SDXL_BASE_RESOURCE_SEALS: weakref.WeakKeyDictionary[UNetModel, _ControlNetResourceSeal] = (
    weakref.WeakKeyDictionary()
)


def _resource_tensors(model: torch.nn.Module) -> tuple[tuple[str, torch.Tensor], ...]:
    return (
        *((f"parameter:{name}", tensor) for name, tensor in model.named_parameters()),
        *((f"buffer:{name}", tensor) for name, tensor in model.named_buffers()),
    )


def _bind_sdxl_base_resource(  # pyright: ignore[reportUnusedFunction]
    model: UNetModel, asset_digest: str
) -> None:
    if type(model) is not UNetModel or model.config != _SDXL_CONTROL_LORA_CONFIG.base:
        raise TypeError("SDXL base resource binding requires the exact SDXL base UNet")
    asset_digest = _require_asset_digest("SDXL base asset digest", asset_digest)
    if model in _SDXL_BASE_RESOURCE_SEALS:
        raise ValueError("SDXL base resource is already bound")
    _SDXL_BASE_RESOURCE_SEALS[model] = _ControlNetResourceSeal(
        asset_digest,
        tuple(_resource_tensor_seal(name, tensor) for name, tensor in _resource_tensors(model)),
    )


def _validate_sdxl_base_resource(  # pyright: ignore[reportUnusedFunction]
    model: UNetModel, asset_digest: str
) -> None:
    seal = _SDXL_BASE_RESOURCE_SEALS.get(model)
    current = _resource_tensors(model)
    if (
        seal is None
        or seal.digest != asset_digest
        or len(current) != len(seal.tensors)
        or any(
            not _matches_sealed_resource_tensor(model, name, tensor, expected)
            for (name, tensor), expected in zip(current, seal.tensors, strict=False)
        )
    ):
        raise ControlResourceBindingError(
            "control-resource-binding-mismatch: SDXL base assembly provenance is absent,"
            " changed, or does not match the declared asset digest"
        )


def _tensor_version(tensor: torch.Tensor) -> int:
    try:
        return int(tensor._version)
    except RuntimeError as error:
        raise ValueError("ControlNet resource tensors must track mutation versions") from error


def _resource_tensor_seal(name: str, tensor: torch.Tensor) -> _ResourceTensorSeal:
    return _ResourceTensorSeal(
        name=name,
        tensor=tensor,
        version=_tensor_version(tensor),
        dtype=tensor.dtype,
        shape=tuple(tensor.shape),
        storage=tensor.untyped_storage(),
        stride=tuple(tensor.stride()),
        storage_offset=int(tensor.storage_offset()),
        conjugated=tensor.is_conj(),
        negated=tensor.is_neg(),
    )


def _matches_sealed_resource_tensor(
    model: torch.nn.Module,
    name: str,
    tensor: torch.Tensor,
    expected: _ResourceTensorSeal,
) -> bool:
    from .module_residency import (
        _residency_assignment_generation,  # pyright: ignore[reportPrivateUsage]
        _residency_assignment_version,  # pyright: ignore[reportPrivateUsage]
    )

    if (
        name != expected.name
        or tensor.dtype != expected.dtype
        or tuple(tensor.shape) != expected.shape
        or tuple(tensor.stride()) != expected.stride
        or tensor.storage_offset() != expected.storage_offset
        or tensor.is_conj() != expected.conjugated
        or tensor.is_neg() != expected.negated
    ):
        return False
    if (
        tensor is expected.tensor
        and tensor.untyped_storage() is expected.storage
        and tensor.dtype == expected.dtype
        and _tensor_version(tensor) == expected.version
    ):
        return True
    key = name.partition(":")[2]
    if _residency_assignment_generation(model, key, tensor) is None:
        return False
    authorized_version = _residency_assignment_version(model, key, tensor)
    try:
        current_version = _tensor_version(tensor)
    except ValueError:
        return tensor.is_inference() and authorized_version is None
    return authorized_version == current_version


def _bind_sd15_controlnet_resource(  # pyright: ignore[reportUnusedFunction]
    model: SD15ControlNet, digest: str
) -> None:
    """Bind provenance after the proving loader has populated model state."""
    if type(model) is not SD15ControlNet:
        raise TypeError("ControlNet resource binding requires an exact SD15ControlNet")
    digest = _require_blake3("ControlNet resource digest", digest)
    if model in _CONTROLNET_RESOURCE_SEALS:
        raise ValueError("ControlNet resource is already bound")
    _CONTROLNET_RESOURCE_SEALS[model] = _ControlNetResourceSeal(
        digest,
        tuple(_resource_tensor_seal(name, tensor) for name, tensor in _resource_tensors(model)),
    )


def _validate_sd15_controlnet_resource(model: SD15ControlNet, digest: str) -> None:
    seal = _CONTROLNET_RESOURCE_SEALS.get(model)
    if seal is None:
        raise ControlResourceBindingError(
            "control-resource-binding-mismatch: ControlNet has no assembly provenance; load it"
            " through assemble_sd15_controlnet"
        )
    if seal.digest != digest:
        raise ControlResourceBindingError(
            "control-resource-binding-mismatch: declared model digest does not match the loaded"
            " ControlNet"
        )
    current = _resource_tensors(model)
    if len(current) != len(seal.tensors) or any(
        not _matches_sealed_resource_tensor(model, name, tensor, expected)
        for (name, tensor), expected in zip(current, seal.tensors, strict=False)
    ):
        raise ControlResourceBindingError(
            "control-resource-binding-mismatch: loaded ControlNet state changed after assembly"
        )


def _bind_sdxl_control_lora_resource(  # pyright: ignore[reportUnusedFunction]
    model: SDXLControlLoRA, digest: str
) -> None:
    if type(model) is not SDXLControlLoRA:
        raise TypeError("Control-LoRA resource binding requires an exact SDXLControlLoRA")
    digest = _require_blake3("Control-LoRA resource digest", digest)
    if model in _CONTROL_LORA_RESOURCE_SEALS:
        raise ValueError("Control-LoRA resource is already bound")
    _CONTROL_LORA_RESOURCE_SEALS[model] = _ControlNetResourceSeal(
        digest,
        tuple(_resource_tensor_seal(name, tensor) for name, tensor in _resource_tensors(model)),
    )


def _validate_sdxl_control_lora_resource(model: SDXLControlLoRA, digest: str) -> None:
    seal = _CONTROL_LORA_RESOURCE_SEALS.get(model)
    if seal is None or seal.digest != digest:
        raise ControlResourceBindingError(
            "control-resource-binding-mismatch: Control-LoRA assembly provenance is absent or"
            " does not match the declared model digest"
        )
    current = _resource_tensors(model)
    if len(current) != len(seal.tensors):
        raise ControlResourceBindingError(
            "control-resource-binding-mismatch: Control-LoRA resource structure changed"
        )
    if any(
        not _matches_sealed_resource_tensor(model, name, tensor, expected)
        for (name, tensor), expected in zip(current, seal.tensors, strict=False)
    ):
        raise ControlResourceBindingError(
            "control-resource-binding-mismatch: Control-LoRA resource tensors changed"
        )


def _bind_sdxl_controlnet_resource(  # pyright: ignore[reportUnusedFunction]
    model: SDXLControlNet, digest: str
) -> None:
    if type(model) is not SDXLControlNet:
        raise TypeError("SDXL ControlNet resource binding requires an exact SDXLControlNet")
    digest = _require_blake3("SDXL ControlNet resource digest", digest)
    if model in _SDXL_CONTROLNET_RESOURCE_SEALS:
        raise ValueError("SDXL ControlNet resource is already bound")
    _SDXL_CONTROLNET_RESOURCE_SEALS[model] = _ControlNetResourceSeal(
        digest,
        tuple(_resource_tensor_seal(name, tensor) for name, tensor in _resource_tensors(model)),
    )


def _validate_sdxl_controlnet_resource(model: SDXLControlNet, digest: str) -> None:
    seal = _SDXL_CONTROLNET_RESOURCE_SEALS.get(model)
    current = _resource_tensors(model)
    if (
        seal is None
        or seal.digest != digest
        or len(current) != len(seal.tensors)
        or any(
            not _matches_sealed_resource_tensor(model, name, tensor, expected)
            for (name, tensor), expected in zip(current, seal.tensors, strict=False)
        )
    ):
        raise ControlResourceBindingError(
            "control-resource-binding-mismatch: SDXL ControlNet assembly provenance is absent"
            " or changed"
        )


def _bind_sdxl_controlnet_union_resource(  # pyright: ignore[reportUnusedFunction]
    model: SDXLControlNetUnion, digest: str
) -> None:
    if type(model) is not SDXLControlNetUnion:
        raise TypeError("Union resource binding requires an exact SDXLControlNetUnion")
    digest = _require_blake3("Union resource digest", digest)
    if model in _CONTROL_UNION_RESOURCE_SEALS:
        raise ValueError("Union resource is already bound")
    _CONTROL_UNION_RESOURCE_SEALS[model] = _ControlNetResourceSeal(
        digest,
        tuple(_resource_tensor_seal(name, tensor) for name, tensor in _resource_tensors(model)),
    )


def _validate_sdxl_controlnet_union_resource(model: SDXLControlNetUnion, digest: str) -> None:
    seal = _CONTROL_UNION_RESOURCE_SEALS.get(model)
    current = _resource_tensors(model)
    if (
        seal is None
        or seal.digest != digest
        or len(current) != len(seal.tensors)
        or any(
            not _matches_sealed_resource_tensor(model, name, tensor, expected)
            for (name, tensor), expected in zip(current, seal.tensors, strict=False)
        )
    ):
        raise ControlResourceBindingError(
            "control-resource-binding-mismatch: Union assembly provenance is absent or changed"
        )


def _snapshot_sd_control_conditioning(  # pyright: ignore[reportUnusedFunction]
    control: SDControlConditioning,
) -> SDControlConditioning:
    """Validate admission bindings and isolate every hint used by execution."""
    newest_to_oldest: list[SDControlConditioning] = []
    seen: set[int] = set()
    current: SDControlConditioning | None = control
    while current is not None:
        identity = id(current)
        if identity in seen:
            raise ControlResourceBindingError(
                "control-resource-binding-mismatch: resolved control chain must be acyclic"
            )
        seen.add(identity)
        newest_to_oldest.append(current)
        current = current.previous

    snapshot: SDControlConditioning | None = None
    for current in reversed(newest_to_oldest):
        snapshot = SDControlConditioning(
            current.application,
            current.model,
            current.hint.detach().clone(),
            current.model_digest,
            current.hint_digest,
            current.gain,
            snapshot,
            tuple(
                SDEffectMaskSource(mask.declaration, mask.mask.detach().clone())
                for mask in current.effect_masks
            ),
        )
    assert snapshot is not None
    return snapshot
