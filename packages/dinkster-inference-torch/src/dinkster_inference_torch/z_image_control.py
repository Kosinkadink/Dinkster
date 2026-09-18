"""Native Alibaba PAI Z-Image Union control module."""

from __future__ import annotations

import ctypes
import sys
import weakref
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from blake3 import blake3
from dinkster_inference import ContributionGain, ControlApplication, EffectMaskInput
from dinkster_inference.z_image import Z_IMAGE_CONFIG

from .attention import AttentionKernel
from .flux import apply_rope
from .operations import INITLESS, Operations
from .z_image import ZImageAttention, ZImageBlock


class ZImageControlBindingError(ValueError):
    """A Z-Image control resource does not match its declared identity."""


@dataclass(frozen=True, slots=True)
class _ResourceTensorSeal:
    name: str
    tensor: torch.Tensor
    storage: torch.UntypedStorage
    version: int
    dtype: torch.dtype
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    conjugated: bool
    negated: bool


@dataclass(frozen=True, slots=True)
class _ResourceSeal:
    digest: str
    tensors: tuple[_ResourceTensorSeal, ...]


def _tensor_version(tensor: torch.Tensor) -> int:
    try:
        return int(tensor._version)
    except RuntimeError as error:
        raise ValueError("Z-Image control tensors must track mutation versions") from error


def _resource_tensors(model: torch.nn.Module) -> tuple[tuple[str, torch.Tensor], ...]:
    return (
        *((f"parameter:{name}", tensor) for name, tensor in model.named_parameters()),
        *((f"buffer:{name}", tensor) for name, tensor in model.named_buffers()),
    )


def _require_blake3(name: str, digest: object) -> str:
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{name} must be a lowercase BLAKE3 digest")
    return digest


def z_image_control_resource_digest(asset_digest: str, compute_dtype: torch.dtype) -> str:
    """Identity for one assembled Z-Image Fun control artifact."""
    if (
        type(asset_digest) is not str
        or not asset_digest.startswith("blake3:")
        or len(asset_digest) != 71
        or any(character not in "0123456789abcdef" for character in asset_digest[7:])
    ):
        raise ValueError("Z-Image control source digest must be a canonical blake3 asset digest")
    if not compute_dtype.is_floating_point:
        raise TypeError("Z-Image control compute dtype must be floating")
    hasher = blake3()
    hasher.update(b"dinkster.z-image-control-resource.v1\n")
    hasher.update(f"asset={asset_digest}\n".encode("ascii"))
    hasher.update(f"compute_dtype={str(compute_dtype).removeprefix('torch.')}\n".encode("ascii"))
    return hasher.hexdigest()


def z_image_control_hint_digest(hint: torch.Tensor) -> str:
    """Content digest for one canonical Z-Image control latent."""
    if type(hint) is not torch.Tensor:
        raise TypeError("Z-Image control hint must be an exact torch.Tensor")
    if hint.layout is not torch.strided:
        raise TypeError("Z-Image control hint must use strided tensor storage")
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
    hasher.update(b"dinkster.z-image-control-hint.v1\n")
    hasher.update(f"shape={','.join(str(dim) for dim in value.shape)}\n".encode("ascii"))
    hasher.update(f"dtype={str(value.dtype).removeprefix('torch.')}\n".encode("ascii"))
    hasher.update(b"byte_order=little\n\n")
    hasher.update(raw)
    return hasher.hexdigest()


class ZImageControlAttention(ZImageAttention):
    def __init__(self, *, operations: Operations, attention_kernel: AttentionKernel) -> None:
        torch.nn.Module.__init__(self)
        config = Z_IMAGE_CONFIG
        self.heads = config.attention_heads
        self.kv_heads = config.kv_heads
        self.head_dim = config.attention_head_dim
        self.to_q = operations.linear(config.hidden_width, config.hidden_width, bias=False)
        self.to_k = operations.linear(config.hidden_width, config.hidden_width, bias=False)
        self.to_v = operations.linear(config.hidden_width, config.hidden_width, bias=False)
        self.to_out = torch.nn.Sequential(
            operations.linear(config.hidden_width, config.hidden_width, bias=False)
        )
        self.norm_q = operations.rms_norm(self.head_dim, eps=config.qk_norm_eps)
        self.norm_k = operations.rms_norm(self.head_dim, eps=config.qk_norm_eps)
        object.__setattr__(self, "_attention_kernel", attention_kernel)

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.norm_q(self.to_q(x).view(batch, length, self.heads, self.head_dim)).transpose(1, 2)
        k = self.norm_k(self.to_k(x).view(batch, length, self.kv_heads, self.head_dim)).transpose(
            1, 2
        )
        v = self.to_v(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, rope)
        output = self._attention_kernel(q, k, v)
        return self.to_out(output.transpose(1, 2).reshape(batch, length, -1))


class ZImageControlBlock(ZImageBlock):
    def __init__(
        self,
        index: int,
        *,
        projected: bool,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__(
            Z_IMAGE_CONFIG,
            modulated=True,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.attention = ZImageControlAttention(
            operations=operations, attention_kernel=attention_kernel
        )
        self.before_proj = operations.linear(3840, 3840) if projected and index == 0 else None
        self.after_proj = operations.linear(3840, 3840) if projected else None

    def forward_control(
        self,
        control: torch.Tensor,
        base: torch.Tensor,
        rope: torch.Tensor,
        modulation: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if self.before_proj is not None:
            control = self.before_proj(control) + base
        control = super().forward(control, rope, modulation)
        return (None if self.after_proj is None else self.after_proj(control), control)


class ZImageControl(torch.nn.Module):
    injection_blocks = (0, 5, 10, 15, 20, 25)
    accepts_z_image_control_binding = True

    def __init__(
        self,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.control_all_x_embedder = torch.nn.ModuleDict({"2-1": operations.linear(64, 3840)})
        self.control_noise_refiner = torch.nn.ModuleList(
            ZImageControlBlock(
                index, projected=False, operations=operations, attention_kernel=attention_kernel
            )
            for index in range(2)
        )
        self.control_layers = torch.nn.ModuleList(
            ZImageControlBlock(
                index, projected=True, operations=operations, attention_kernel=attention_kernel
            )
            for index in range(6)
        )

    @property
    def resource_digest(self) -> str | None:
        seal = _RESOURCE_SEALS.get(self)
        return None if seal is None else seal.digest

    def embed(self, latent: torch.Tensor) -> torch.Tensor:
        pad_h, pad_w = (-latent.shape[-2]) % 2, (-latent.shape[-1]) % 2
        latent = F.pad(latent, (0, pad_w, 0, pad_h), mode="circular")
        batch, channels, height, width = latent.shape
        patches = latent.view(batch, channels, height // 2, 2, width // 2, 2)
        patches = patches.permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)
        return self.control_all_x_embedder["2-1"](patches)

    def refine(
        self,
        control: torch.Tensor,
        rope: torch.Tensor,
        modulation: torch.Tensor,
    ) -> torch.Tensor:
        for module in self.control_noise_refiner:
            block = cast(ZImageControlBlock, module)
            control = block(control, rope, modulation)
        return control

    def inject(
        self,
        index: int,
        control: torch.Tensor,
        base: torch.Tensor,
        rope: torch.Tensor,
        modulation: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        block = cast(ZImageControlBlock, self.control_layers[index])
        return block.forward_control(control, base, rope, modulation)


_RESOURCE_SEALS: weakref.WeakKeyDictionary[ZImageControl, _ResourceSeal] = (
    weakref.WeakKeyDictionary()
)


def _bind_z_image_control_resource(  # pyright: ignore[reportUnusedFunction]
    model: ZImageControl, digest: str
) -> None:
    """Seal assembly-proven model identity and loaded tensor structure."""
    if type(model) is not ZImageControl:
        raise TypeError("Z-Image control resource must be an exact ZImageControl")
    digest = _require_blake3("Z-Image control model digest", digest)
    if model in _RESOURCE_SEALS:
        raise ValueError("Z-Image control resource is already bound")
    _RESOURCE_SEALS[model] = _ResourceSeal(
        digest,
        tuple(
            _ResourceTensorSeal(
                name,
                tensor,
                tensor.untyped_storage(),
                _tensor_version(tensor),
                tensor.dtype,
                tuple(tensor.shape),
                tuple(tensor.stride()),
                int(tensor.storage_offset()),
                tensor.is_conj(),
                tensor.is_neg(),
            )
            for name, tensor in _resource_tensors(model)
        ),
    )


def validate_z_image_control_resource(model: ZImageControl, digest: str) -> None:
    """Require the model to retain its assembly-proven identity and state."""
    from .module_residency import (
        _residency_assignment_generation,  # pyright: ignore[reportPrivateUsage]
    )

    seal = _RESOURCE_SEALS.get(model)
    current = _resource_tensors(model)
    if (
        seal is None
        or seal.digest != digest
        or len(current) != len(seal.tensors)
        or any(
            name != expected.name
            or tensor.dtype != expected.dtype
            or tuple(tensor.shape) != expected.shape
            or tuple(tensor.stride()) != expected.stride
            or tensor.storage_offset() != expected.storage_offset
            or tensor.is_conj() != expected.conjugated
            or tensor.is_neg() != expected.negated
            or not (
                (
                    tensor is expected.tensor
                    and tensor.untyped_storage() is expected.storage
                    and _tensor_version(tensor) == expected.version
                )
                or (
                    _residency_assignment_generation(model, name.partition(":")[2], tensor)
                    is not None
                    and _tensor_version(tensor) == expected.version + 1
                )
            )
            for (name, tensor), expected in zip(current, seal.tensors, strict=False)
        )
    ):
        raise ZImageControlBindingError(
            "control-resource-binding-mismatch: Z-Image control assembly provenance is absent,"
            " changed, or does not match the declared model digest; load it through"
            " assemble_z_image_control"
        )


@dataclass(frozen=True, slots=True)
class ZImageControlConditioning:
    """One resolved Z-Image Fun control application for a sampling run."""

    application: ControlApplication
    model: ZImageControl
    hint: torch.Tensor
    model_digest: str
    hint_digest: str
    gain: ContributionGain | None = None
    previous: ZImageControlConditioning | None = None
    effect_masks: tuple[EffectMaskInput, ...] = ()

    def __post_init__(self) -> None:
        if type(self.application) is not ControlApplication:
            raise TypeError("control application must be an exact ControlApplication")
        if type(self.model) is not ZImageControl:
            raise TypeError("control model must be an exact ZImageControl")
        if type(self.hint) is not torch.Tensor or self.hint.layout is not torch.strided:
            raise TypeError("Z-Image control hint must be an exact strided tensor")
        if self.hint.ndim != 4 or self.hint.shape[1] != 16:
            raise ValueError(
                f"Z-Image control hint must be [batch x 16 x H x W], got {tuple(self.hint.shape)}"
            )
        if not self.hint.is_floating_point():
            raise TypeError("Z-Image control hint must have a floating dtype")
        if self.hint.shape[0] < 1 or self.hint.shape[2] < 1 or self.hint.shape[3] < 1:
            raise ValueError("Z-Image control hint batch and spatial geometry must be positive")
        model_digest = _require_blake3("Z-Image control model digest", self.model_digest)
        hint_digest = _require_blake3("Z-Image control hint digest", self.hint_digest)
        validate_z_image_control_resource(self.model, model_digest)
        if self.application.hint.id != hint_digest:
            raise ZImageControlBindingError(
                "control-resource-binding-mismatch: control hint reference does not resolve to"
                " the declared hint digest"
            )
        if z_image_control_hint_digest(self.hint) != hint_digest:
            raise ZImageControlBindingError(
                "control-resource-binding-mismatch: declared hint digest does not match the"
                " materialized hint tensor"
            )
        if self.application.mode is not None:
            raise ValueError("Z-Image Fun control does not accept an SD control mode")
        if self.gain is not None and type(self.gain) is not ContributionGain:
            raise TypeError("control gain must be an exact ContributionGain or None")
        if self.gain is not None and (
            self.application.strength != 1.0
            or self.application.window.start_percent != 0.0
            or self.application.window.end_percent != 1.0
        ):
            raise ValueError(
                "explicit ContributionGain requires identity ControlApplication strength/window"
            )
        if type(self.effect_masks) is not tuple or any(
            type(mask) is not EffectMaskInput for mask in self.effect_masks
        ):
            raise TypeError("control effect masks must be exact EffectMaskInput values")
        if self.effect_masks:
            raise ValueError("Z-Image control effect masks are not implemented")
        if self.previous is not None or self.application.previous is not None:
            raise ValueError("Z-Image control chains are an admission limit of one application")


def snapshot_z_image_control_conditioning(
    conditioning: ZImageControlConditioning,
) -> ZImageControlConditioning:
    """Validate and own the admitted hint before execution begins."""
    if type(conditioning) is not ZImageControlConditioning:
        raise TypeError("control must be an exact ZImageControlConditioning")
    return ZImageControlConditioning(
        conditioning.application,
        conditioning.model,
        conditioning.hint.detach().clone(),
        conditioning.model_digest,
        conditioning.hint_digest,
        conditioning.gain,
        None,
        conditioning.effect_masks,
    )


__all__ = [
    "ZImageControl",
    "ZImageControlBindingError",
    "ZImageControlConditioning",
    "snapshot_z_image_control_conditioning",
    "validate_z_image_control_resource",
    "z_image_control_hint_digest",
    "z_image_control_resource_digest",
]
