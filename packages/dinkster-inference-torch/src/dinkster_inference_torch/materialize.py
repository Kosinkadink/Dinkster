"""Materialize stage-4a decode specs into live adapters and patches.

The torch-free decode layer (dinkster_inference.lora) classifies a LoRA
file into specs whose fields are SOURCE file keys; nothing has read
tensor bytes yet. This module performs those reads: spec + tensor
mapping -> slice-1 adapter instances / typed patch values / a whole
PatchSet ready for apply.py.

Scalar-read semantics mirror the reference loads
(comfy/weight_adapter/*.load @ b78cec87): ``.alpha`` companions are
read with ``.item()``, ``.reshape_weight`` with ``.tolist()``.

Deliberate deviation, LOUD: a spec that names a key missing from the
tensor mapping raises MaterializeError. Upstream cannot hit this state
(its loads see the full state dict they classified), so nothing is
lost - it guards Dinkster's split between header-based decode and
deferred tensor reads.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from dinkster_inference import ProviderPatchRef
from dinkster_inference.lora import (
    BOFTSpec,
    DiffPatchRef,
    GLoRASpec,
    LoHaSpec,
    LoKrSpec,
    LoRASpec,
    OFTSpec,
    PatchTarget,
    SetPatchRef,
)
from dinkster_inference.patches import (
    AdapterPatch,
    DiffPatch,
    PatchEntry,
    PatchSet,
    PatchValue,
    SetPatch,
)
from dinkster_inference.registry import Registry

from .adapters import (
    BOFTAdapter,
    GLoRAAdapter,
    LoHaAdapter,
    LoKrAdapter,
    LoRAAdapter,
    OFTAdapter,
)

AnyAdapter = LoRAAdapter | LoHaAdapter | LoKrAdapter | GLoRAAdapter | OFTAdapter | BOFTAdapter


class MaterializeError(Exception):
    """A decode spec references a tensor the source cannot provide."""


PatchProviderMaterializer = Callable[
    [ProviderPatchRef, Mapping[str, torch.Tensor], torch.dtype],
    PatchValue[torch.Tensor],
]


@dataclass(frozen=True)
class PatchProviderDescriptor:
    """Worker-local materializer for one out-of-tree decoded patch type."""

    id: str
    materialize: PatchProviderMaterializer
    aliases: tuple[str, ...] = ()
    behavior_metadata: tuple[tuple[str, str | int | bool | None], ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("patch provider id must be non-empty")
        if (
            any(not alias for alias in self.aliases)
            or tuple(sorted(self.aliases)) != self.aliases
            or len(self.aliases) != len(set(self.aliases))
        ):
            raise ValueError("patch provider aliases must be sorted and unique")
        if not callable(self.materialize):
            raise TypeError("patch provider materialize must be callable")
        keys = [key for key, _value in self.behavior_metadata]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("patch provider behavior metadata must be sorted and unique")
        for key, value in self.behavior_metadata:
            if not key or (value is not None and type(value) not in (str, int, bool)):
                raise TypeError("patch provider behavior metadata must be RPC-clean")


def _tensor(tensors: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    try:
        return tensors[key]
    except KeyError:
        raise MaterializeError(f"spec references missing tensor {key!r}") from None


def _tensor_opt(tensors: Mapping[str, torch.Tensor], key: str | None) -> torch.Tensor | None:
    return None if key is None else _tensor(tensors, key)


def _scalar_opt(tensors: Mapping[str, torch.Tensor], key: str | None) -> float | None:
    """``.item()`` read, how every reference load consumes ``.alpha``."""
    return None if key is None else float(_tensor(tensors, key).item())


def materialize_adapter(
    spec: LoRASpec | LoHaSpec | LoKrSpec | GLoRASpec | OFTSpec | BOFTSpec,
    tensors: Mapping[str, torch.Tensor],
    *,
    intermediate_dtype: torch.dtype = torch.float32,
) -> AnyAdapter:
    """Read the tensors a stage-4a adapter spec names and build the
    matching slice-1 adapter."""
    if isinstance(spec, LoRASpec):
        reshape: tuple[int, ...] | None = spec.reshape_dims
        if reshape is None and spec.reshape is not None:
            # reference: lora[reshape_name].tolist() (lora.py load)
            reshape = tuple(int(d) for d in _tensor(tensors, spec.reshape).tolist())
        return LoRAAdapter(
            _tensor(tensors, spec.up),
            _tensor(tensors, spec.down),
            alpha=_scalar_opt(tensors, spec.alpha),
            mid=_tensor_opt(tensors, spec.mid),
            dora_scale=_tensor_opt(tensors, spec.dora_scale),
            reshape=reshape,
            intermediate_dtype=intermediate_dtype,
        )
    if isinstance(spec, LoHaSpec):
        return LoHaAdapter(
            _tensor(tensors, spec.w1_a),
            _tensor(tensors, spec.w1_b),
            _tensor(tensors, spec.w2_a),
            _tensor(tensors, spec.w2_b),
            alpha=_scalar_opt(tensors, spec.alpha),
            t1=_tensor_opt(tensors, spec.t1),
            t2=_tensor_opt(tensors, spec.t2),
            dora_scale=_tensor_opt(tensors, spec.dora_scale),
            intermediate_dtype=intermediate_dtype,
        )
    if isinstance(spec, LoKrSpec):
        return LoKrAdapter(
            w1=_tensor_opt(tensors, spec.w1),
            w2=_tensor_opt(tensors, spec.w2),
            w1_a=_tensor_opt(tensors, spec.w1_a),
            w1_b=_tensor_opt(tensors, spec.w1_b),
            w2_a=_tensor_opt(tensors, spec.w2_a),
            w2_b=_tensor_opt(tensors, spec.w2_b),
            t2=_tensor_opt(tensors, spec.t2),
            alpha=_scalar_opt(tensors, spec.alpha),
            dora_scale=_tensor_opt(tensors, spec.dora_scale),
            intermediate_dtype=intermediate_dtype,
        )
    if isinstance(spec, GLoRASpec):
        return GLoRAAdapter(
            _tensor(tensors, spec.a1),
            _tensor(tensors, spec.a2),
            _tensor(tensors, spec.b1),
            _tensor(tensors, spec.b2),
            alpha=_scalar_opt(tensors, spec.alpha),
            dora_scale=_tensor_opt(tensors, spec.dora_scale),
            intermediate_dtype=intermediate_dtype,
        )
    if isinstance(spec, OFTSpec):
        return OFTAdapter(
            _tensor(tensors, spec.blocks),
            rescale=_tensor_opt(tensors, spec.rescale),
            alpha=_scalar_opt(tensors, spec.alpha),
            dora_scale=_tensor_opt(tensors, spec.dora_scale),
            intermediate_dtype=intermediate_dtype,
        )
    # BOFTSpec, by elimination of the closed union
    return BOFTAdapter(
        _tensor(tensors, spec.blocks),
        rescale=_tensor_opt(tensors, spec.rescale),
        alpha=_scalar_opt(tensors, spec.alpha),
        dora_scale=_tensor_opt(tensors, spec.dora_scale),
        intermediate_dtype=intermediate_dtype,
    )


def materialize_value(
    decoded: object,
    tensors: Mapping[str, torch.Tensor],
    *,
    intermediate_dtype: torch.dtype = torch.float32,
    provider_registry: Registry[PatchProviderDescriptor] | None = None,
) -> PatchValue[torch.Tensor]:
    """One decoded patch -> one typed patch value."""
    if isinstance(decoded, DiffPatchRef):
        return DiffPatch(_tensor(tensors, decoded.key))
    if isinstance(decoded, SetPatchRef):
        return SetPatch(_tensor(tensors, decoded.key))
    if isinstance(decoded, LoRASpec | LoHaSpec | LoKrSpec | GLoRASpec | OFTSpec | BOFTSpec):
        return AdapterPatch(
            materialize_adapter(decoded, tensors, intermediate_dtype=intermediate_dtype)
        )
    if not isinstance(decoded, ProviderPatchRef):
        raise MaterializeError(f"unsupported decoded patch type {type(decoded).__name__}")
    provider_id = decoded.provider_id
    provider = None if provider_registry is None else provider_registry.get(provider_id)
    if provider is None:
        raise MaterializeError(f"patch provider {provider_id!r} is not registered")
    return provider.materialize(decoded, tensors, intermediate_dtype)


def build_patch_set(
    patches: Mapping[PatchTarget, object],
    tensors: Mapping[str, torch.Tensor],
    *,
    strength: float = 1.0,
    strength_model: float = 1.0,
    intermediate_dtype: torch.dtype = torch.float32,
    structural_digest: str | None = None,
    provider_registry: Registry[PatchProviderDescriptor] | None = None,
) -> PatchSet[torch.Tensor]:
    """A whole LoraDecodeResult.patches mapping -> a PatchSet, the
    ModelPatcher.add_patches step (@ b78cec87) for decoded LoRAs:
    one entry per target, PatchTarget offsets carried through."""
    grouped: dict[str, tuple[PatchEntry[torch.Tensor], ...]] = {}
    for target, decoded in patches.items():
        entry = PatchEntry(
            materialize_value(
                decoded,
                tensors,
                intermediate_dtype=intermediate_dtype,
                provider_registry=provider_registry,
            ),
            strength=strength,
            strength_model=strength_model,
            offset=target.offset,
        )
        grouped[target.key] = grouped.get(target.key, ()) + (entry,)
    return PatchSet(grouped, structural_digest=structural_digest)


__all__ = [
    "AnyAdapter",
    "MaterializeError",
    "PatchProviderDescriptor",
    "PatchProviderMaterializer",
    "build_patch_set",
    "materialize_adapter",
    "materialize_value",
]
