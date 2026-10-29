"""The weight-patch algebra, typed.

ComfyUI represents patches as positional tuples dispatched by length and
magic strings, applied by one 60-line function (comfy/lora.py
calculate_weight, comfy/model_patcher.py patches dict @ b78cec87). The
semantics are sound; the representation is not. This module keeps the
semantics as a closed set of frozen value types:

- Diff: add ``strength * value`` to the weight (optionally padding the
  weight up to the diff's shape first).
- Set: replace the weight outright.
- ModelAsLora: add ``strength * (target - original)``.
- Adapter: delegate the math to a WeightAdapter (LoRA/LoHa/LoKr/...),
  which also predicts its target shape.
- Nested: apply entries against another base, then diff the result in
  (merging an already-patched model).

An entry scales the existing weight by ``strength_model``, applies to an
optional narrow ``offset`` window, and a PatchSet is an immutable map of
parameter key -> entry tuple with a revision identity (the role
patches_uuid plays in ComfyUI).

Application (tensor math, backup/restore, stochastic rounding) is stage
4; shape prediction lives here because it is pure geometry - ported from
comfy/lora.py calculate_shape @ b78cec87.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Generic, Protocol, TypeVar, cast


class SizedTensor(Protocol):
    """The one structural demand contracts place on tensors: a shape.

    torch.Tensor satisfies this without importing torch here. (Static
    contract only - deliberately not runtime_checkable; plugin admission
    validates explicitly when it needs to.)
    """

    @property
    def shape(self) -> Sequence[int]: ...


T = TypeVar("T", bound=SizedTensor)


class WeightAdapter(Protocol[T]):
    """Adapter-family patch math (LoRA and kin), behind one interface.

    ComfyUI's comfy/weight_adapter/ classes (@ b78cec87) already have
    this shape; here it is the contract instead of a convention.
    """

    def target_shape(self, base: tuple[int, ...]) -> tuple[int, ...]:
        """The weight shape after applying this adapter to ``base``."""
        ...

    def calculate(
        self,
        weight: T,
        *,
        strength: float,
        function: Callable[[T], T] | None = None,
    ) -> T:
        """Return the patched weight.

        May mutate ``weight`` in place and return it (the reference
        adapters do); callers pass an owned buffer - comfy's
        patch_weight_to_device hands each adapter a fresh
        intermediate-dtype copy, and Dinkster's apply layer does the
        same. The other inputs (the adapter's own tensors) are never
        mutated.

        ``function`` is the per-entry delta hook (PatchEntry.function),
        applied to the computed diff before it is added to the weight -
        exactly where the reference adapters call ``function(...)``."""
        ...


class PreparedWeightAdapter(WeightAdapter[T], Protocol[T]):
    """Optional adapter extension for storage-dtype payload staging."""

    def payload_tensors(self) -> tuple[T, ...]:
        """Return payload tensors in the adapter's reconstruction order."""
        ...

    def rebuild_payloads(self, replacements: Sequence[T]) -> PreparedWeightAdapter[T]:
        """Return an equivalent adapter using ``replacements`` as payloads."""
        ...


class WeightConverter(Protocol[T]):
    """Converts a stored weight into plain patchable form - upstream's
    ``convert_func`` (comfy/model_patcher.py get_key_weight
    @ b78cec87), e.g. dequantizing a quantized tensor. ``inplace``
    permits mutating ``weight`` (callers pass an owned copy)."""

    def __call__(self, weight: T, /, *, inplace: bool = False) -> T: ...


@dataclass(frozen=True)
class DiffPatch(Generic[T]):
    value: T
    pad_weight: bool = False


@dataclass(frozen=True)
class SetPatch(Generic[T]):
    value: T


@dataclass(frozen=True)
class ModelAsLoraPatch(Generic[T]):
    target: T


@dataclass(frozen=True)
class AdapterPatch(Generic[T]):
    adapter: WeightAdapter[T]


@dataclass(frozen=True)
class NestedPatch(Generic[T]):
    """Apply another model's patch entries against ``base`` first, then
    diff the result in - how ComfyUI merges an already-patched model
    into another (comfy/lora.py "model_as_lora" nested tuples
    @ b78cec87). The outer weight shape is unchanged.

    ``convert`` is the donor's weight converter (upstream: the
    ``convert_func`` bundled with the base weight by get_key_patches),
    applied to an owned intermediate-dtype copy of ``base`` before the
    nested entries run - e.g. dequantizing a quantized donor weight.
    None means the identity."""

    base: T
    entries: tuple[PatchEntry[T], ...]
    convert: WeightConverter[T] | None = None


PatchValue = DiffPatch[T] | SetPatch[T] | ModelAsLoraPatch[T] | AdapterPatch[T] | NestedPatch[T]


@dataclass(frozen=True)
class PatchOffset:
    """Narrow application to ``length`` elements from ``start`` along
    ``dim`` - offset patches never change the whole-weight shape."""

    dim: int
    start: int
    length: int

    def __post_init__(self) -> None:
        if self.dim < 0 or self.start < 0 or self.length <= 0:
            raise ValueError(f"invalid patch offset {self}")


@dataclass(frozen=True)
class PatchEntry(Generic[T]):
    """One patch application: what, how strongly, and where.

    ``function`` is the optional delta hook (upstream: the fifth patch
    tuple element, comfy/model_patcher.py add_patches @ b78cec87):
    applied to the computed, strength-scaled delta right before it is
    added to the weight. None means the identity. Not applicable to
    Set patches (the reference never routes ``set`` through it)."""

    value: PatchValue[T]
    strength: float = 1.0
    strength_model: float = 1.0
    offset: PatchOffset | None = None
    function: Callable[[T], T] | None = None


class PatchPayloadError(ValueError):
    """Patch payload replacements do not match the structural walk."""


def _adapter_surface(adapter: object) -> bool:
    return callable(getattr(adapter, "payload_tensors", None)) and callable(
        getattr(adapter, "rebuild_payloads", None)
    )


def patch_payloads(
    entries: Sequence[PatchEntry[T]],
    *,
    unsupported: Callable[[object], None] | None = None,
) -> tuple[T, ...]:
    """Walk tensor payloads in deterministic patch-application order.

    Third-party adapters predating the payload surface remain valid patch
    adapters. Their leaf is omitted from staging and reported once through
    ``unsupported`` when supplied; surrounding supported values still walk.
    """
    payloads: list[T] = []

    def walk_value(value: PatchValue[T]) -> None:
        if isinstance(value, DiffPatch | SetPatch):
            payloads.append(value.value)
        elif isinstance(value, ModelAsLoraPatch):
            payloads.append(value.target)
        elif isinstance(value, AdapterPatch):
            adapter = value.adapter
            if not _adapter_surface(adapter):
                if unsupported is not None:
                    unsupported(adapter)
                return
            payloads.extend(cast(PreparedWeightAdapter[T], adapter).payload_tensors())
        else:
            payloads.append(value.base)
            for nested_entry in value.entries:
                walk_value(nested_entry.value)

    for entry in entries:
        walk_value(entry.value)
    return tuple(payloads)


def rebuild_patch_entries(
    entries: Sequence[PatchEntry[T]],
    replacements: Sequence[T],
    *,
    unsupported: Callable[[object], None] | None = None,
) -> tuple[PatchEntry[T], ...]:
    """Rebuild ``entries`` from a :func:`patch_payloads` replacement walk.

    Entry strength, model strength, offset, function, nested converters, and
    all non-payload adapter state are retained exactly.
    """
    index = 0

    def take() -> T:
        nonlocal index
        if index >= len(replacements):
            raise PatchPayloadError("too few patch payload replacements")
        value = replacements[index]
        index += 1
        return value

    def rebuild_value(value: PatchValue[T]) -> PatchValue[T]:
        if isinstance(value, DiffPatch):
            return replace(value, value=take())
        if isinstance(value, SetPatch):
            return replace(value, value=take())
        if isinstance(value, ModelAsLoraPatch):
            return replace(value, target=take())
        if isinstance(value, AdapterPatch):
            adapter = value.adapter
            if not _adapter_surface(adapter):
                if unsupported is not None:
                    unsupported(adapter)
                return value
            prepared_adapter = cast(PreparedWeightAdapter[T], adapter)
            count = len(prepared_adapter.payload_tensors())
            adapter_replacements = replacements[index : index + count]
            if len(adapter_replacements) != count:
                raise PatchPayloadError("too few patch payload replacements")
            for _ in range(count):
                take()
            return replace(
                value,
                adapter=prepared_adapter.rebuild_payloads(adapter_replacements),
            )
        return replace(
            value,
            base=take(),
            entries=tuple(
                replace(entry, value=rebuild_value(entry.value)) for entry in value.entries
            ),
        )

    rebuilt = tuple(replace(entry, value=rebuild_value(entry.value)) for entry in entries)
    if index != len(replacements):
        raise PatchPayloadError("too many patch payload replacements")
    return rebuilt


@dataclass(frozen=True, eq=False)
class PatchSet(Generic[T]):
    """An immutable set of patches keyed by parameter name.

    The mapping is snapshotted at construction (a caller mutating the
    dict it passed in cannot change an existing set), and ``revision``
    is the identity for cache/residency decisions ("are the loaded
    weights patched with exactly this?") - fresh per construction and
    never injectable, like ComfyUI's patches_uuid. Two identical-looking
    sets built separately are two revisions; equality follows revision,
    never structure.
    """

    patches: Mapping[str, tuple[PatchEntry[T], ...]]
    structural_digest: str | None = None
    revision: str = field(init=False, default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        snapshot = MappingProxyType({k: tuple(v) for k, v in self.patches.items()})
        object.__setattr__(self, "patches", snapshot)
        if (
            self.structural_digest is not None
            and re.fullmatch(r"[0-9a-f]{64}", self.structural_digest) is None
        ):
            raise ValueError("structural_digest must be a lowercase sha256 hex digest")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PatchSet):
            return NotImplemented
        return self.revision == other.revision

    def __hash__(self) -> int:
        return hash(self.revision)

    def keys(self) -> Iterator[str]:
        return iter(self.patches)

    def entries(self, key: str) -> tuple[PatchEntry[T], ...]:
        return self.patches.get(key, ())

    def merge(self, other: PatchSet[T]) -> PatchSet[T]:
        """Concatenate entry lists per key (order: self, then other) into
        a new PatchSet with a new revision."""
        merged = dict(self.patches)
        for key, entries in other.patches.items():
            merged[key] = merged.get(key, ()) + entries
        return PatchSet(merged)


def calculate_shape(base: tuple[int, ...], entries: Sequence[PatchEntry[T]]) -> tuple[int, ...]:
    """The weight shape after applying ``entries`` to a ``base``-shaped
    weight, without touching tensor data.

    Ported from comfy/lora.py calculate_shape @ b78cec87: offset patches
    never change shape; Set adopts the value's shape; Diff with
    pad_weight adopts the diff's shape; adapters report their own.
    """
    shape = base
    for entry in entries:
        if entry.offset is not None:
            continue
        value = entry.value
        if isinstance(value, SetPatch):
            shape = tuple(value.value.shape)
        elif isinstance(value, DiffPatch) and value.pad_weight:
            shape = tuple(value.value.shape)
        elif isinstance(value, AdapterPatch):
            shape = value.adapter.target_shape(shape)
    return shape


__all__ = [
    "AdapterPatch",
    "DiffPatch",
    "ModelAsLoraPatch",
    "NestedPatch",
    "PatchEntry",
    "PatchOffset",
    "PatchPayloadError",
    "PatchSet",
    "PatchValue",
    "PreparedWeightAdapter",
    "SetPatch",
    "SizedTensor",
    "WeightAdapter",
    "WeightConverter",
    "calculate_shape",
    "patch_payloads",
    "rebuild_patch_entries",
]
