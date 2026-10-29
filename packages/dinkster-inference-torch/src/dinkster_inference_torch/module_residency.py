"""Post-assembly per-module weight residency.

Parameters and buffers remain registered on their modules as the one
source of truth. ``ModuleStateStore`` presents that live state through
the keyed ``StoredWeight`` shape consumed by ``ResidentWeights``;
weight-holding layers switch to mechanism-routed cast-at-use only while
their unit is offloaded, mirroring ComfyUI's ``comfy_cast_weights``
toggle and ModelPatcher low-VRAM placement @ b78cec87.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, fields
from math import gcd
from types import MappingProxyType, TracebackType
from typing import TYPE_CHECKING, Generic, Literal, Protocol, TypeVar, cast, overload
from weakref import ReferenceType, ref

import torch
from dinkster_inference.patches import PatchSet, patch_payloads, rebuild_patch_entries

from .apply import PatchApplyError, StoredWeight, apply_patches
from .assemble import (
    AssembledFlux,
    AssembledLumina2,
    AssembledQwenImage,
    AssembledSD,
    AssembledWan21,
    AssembledZImage,
)
from .gguf_linear import GgufEncodedLinear
from .memory import MemoryPolicy, MpsMemorySnapshot, mps_memory_snapshot
from .operations import INITLESS, bind_fp8_matmul_layer, bind_residency_layer, bound_compute_dtype
from .quant import FP8_DTYPES, Fp8ScaledWeight, Int8PackedWeight, Nvfp4PackedWeight
from .quant_linear import Fp8Linear, Int8Embedding, Int8Linear, Nvfp4Linear
from .residency import (
    ResidencyMechanism,
    ResidencyUnit,
    ResidencyUnitState,
    ResidentWeights,
    UnitResidency,
    WeightLease,
)
from .residency_timing import PartialResidencyTiming
from .rounding import stochastic_rounding, string_to_seed
from .t5_text import T5LayerNorm

if TYPE_CHECKING:
    from .minimax_h3_assembly import AssembledMiniMaxH3Model

__all__ = [
    "ComponentResidencyPlacement",
    "declare_residency_unit",
    "declare_residency_materialization_ceilings",
    "EnrolledAssembly",
    "LayerLease",
    "ModuleStateStore",
    "EnrolledResidency",
    "ResidencyMechanismFactory",
    "ResidencyBinding",
    "STORAGE_DTYPE_POLICY_TOKENS",
    "StorageDtypeOutcome",
    "StorageDtypePolicyError",
    "StorageDtypeReport",
    "detach_residency_enrollment",
    "enroll_assembled",
    "enroll_component",
    "enroll_component_placement",
]


logger = logging.getLogger(__name__)

_MATERIALIZATION_CEILINGS = "_dinkster_residency_materialization_ceilings"
_RESIDENCY_STATE_STORE = "_dinkster_residency_state_store"
_RESIDENCY_UNIT_ROOT = "_dinkster_residency_unit_root"
_RESIDENCY_UNIT_EXPERT = "_dinkster_residency_unit_expert"


StorageDtypeOutcome = Literal[
    "converted",
    "already_at_target",
    "no_floating_state",
    "quantized",
    "mixed_floating",
    "storage_widening",
    "unsupported_dtype",
]
StorageDtypeErrorReason = Literal["unrouted_state", "unmanaged_buffer"]

STORAGE_DTYPE_POLICY_TOKENS = (
    "converted",
    "already_at_target",
    "no_floating_state",
    "quantized",
    "mixed_floating",
    "storage_widening",
    "unsupported_dtype",
    "unrouted_state",
    "unmanaged_buffer",
)
"""The closed storage-dtype policy vocabulary.

Outcome tokens are reported per component; the final two tokens are hard-error
reasons. This set grows only by an explicit API addition. When the policy is
disabled, ``StorageDtypeReport.enabled`` is false and ``outcomes`` is empty.
"""


class StorageDtypePolicyError(ValueError):
    """An otherwise convertible component has unmanaged residency state."""

    def __init__(self, component: str, reason: StorageDtypeErrorReason, detail: str) -> None:
        self.component = component
        self.reason = reason
        super().__init__(
            f"storage_dtype_follows_compute refused component {component!r}: {reason}: {detail}"
        )


@dataclass(frozen=True)
class StorageDtypeReport:
    """Structured outcomes for one assembled-model enrollment.

    Enabled reports contain exactly one closed outcome token per enrolled
    component. Disabled reports have ``enabled=False`` and an empty mapping.
    """

    enabled: bool
    outcomes: Mapping[str, StorageDtypeOutcome]

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcomes", MappingProxyType(dict(self.outcomes)))


R = TypeVar("R", bound="EnrolledResidency")


class EnrolledAssembly(Mapping[str, R], Generic[R]):
    """Residency mechanisms plus their structured storage-dtype report."""

    def __init__(
        self,
        mechanisms: Mapping[str, R],
        *,
        storage_dtype_report: StorageDtypeReport,
    ) -> None:
        self._mechanisms = MappingProxyType(dict(mechanisms))
        self.storage_dtype_report = storage_dtype_report

    def __getitem__(self, key: str) -> R:
        return self._mechanisms[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._mechanisms)

    def __len__(self) -> int:
        return len(self._mechanisms)


class EnrolledResidency(ResidencyMechanism, UnitResidency, Protocol):
    """The manager and per-forward surfaces of an enrolled mechanism."""


class ComponentResidencyPlacement(Mapping[torch.device, EnrolledResidency]):
    """Per-device mechanisms that jointly own one module tree."""

    def __init__(self, mechanisms: Mapping[torch.device, EnrolledResidency]) -> None:
        self._mechanisms = MappingProxyType(dict(mechanisms))

    def __getitem__(self, key: torch.device) -> EnrolledResidency:
        return self._mechanisms[key]

    def __iter__(self) -> Iterator[torch.device]:
        return iter(self._mechanisms)

    def __len__(self) -> int:
        return len(self._mechanisms)

    @property
    def mechanisms(self) -> tuple[EnrolledResidency, ...]:
        return tuple(self._mechanisms.values())


class ResidencyMechanismFactory(Protocol):
    """Constructor shape shared by eager and demand-paged mechanisms."""

    def __call__(
        self,
        weights: MutableMapping[str, StoredWeight],
        *,
        load_device: torch.device | str,
        offload_device: torch.device | str,
        patch_set: PatchSet[torch.Tensor] | None = None,
        units: Sequence[ResidencyUnit] | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
    ) -> EnrolledResidency: ...


class _ConfiguredResidencyMechanismFactory(Protocol):
    def __call__(
        self,
        weights: MutableMapping[str, StoredWeight],
        *,
        load_device: torch.device | str,
        offload_device: torch.device | str,
        patch_set: PatchSet[torch.Tensor] | None = None,
        units: Sequence[ResidencyUnit] | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
        patch_weight_dtype: torch.dtype | None = None,
        patch_key_prefix: str = "",
    ) -> EnrolledResidency: ...


@dataclass(frozen=True)
class _StateSlot:
    module: torch.nn.Module
    name: str
    parameter: bool


@dataclass(frozen=True)
class _AuthorizedAssignment:
    tensor: ReferenceType[torch.Tensor]
    storage: ReferenceType[torch.UntypedStorage]
    generation: int
    version: int | None


def _residency_assignment_generation(  # pyright: ignore[reportUnusedFunction]
    module: torch.nn.Module, key: str, tensor: torch.Tensor
) -> int | None:
    """Return the generation of the current residency-owned assignment."""
    store = module.__dict__.get(_RESIDENCY_STATE_STORE)
    if not isinstance(store, ModuleStateStore):
        return None
    return store._authorized_generation(key, tensor)  # pyright: ignore[reportPrivateUsage]


def _residency_assignment_version(  # pyright: ignore[reportUnusedFunction]
    module: torch.nn.Module, key: str, tensor: torch.Tensor
) -> int | None:
    """Return the version captured for the current residency-owned assignment."""
    store = module.__dict__.get(_RESIDENCY_STATE_STORE)
    if not isinstance(store, ModuleStateStore):
        return None
    return store._authorized_version(key, tensor)  # pyright: ignore[reportPrivateUsage]


def declare_residency_materialization_ceilings(
    module: torch.nn.Module, ceilings: Mapping[str, int]
) -> None:
    """Declare maximum materialized item sizes for direct module state."""
    direct: dict[str, torch.Tensor] = {
        name: parameter
        for name, parameter in module.named_parameters(recurse=False, remove_duplicate=False)
    }
    direct.update(
        {
            name: buffer
            for name, buffer in module.named_buffers(recurse=False, remove_duplicate=False)
        }
    )
    unknown = ceilings.keys() - direct.keys()
    if unknown:
        raise KeyError(f"materialization ceilings name unknown state: {sorted(unknown)!r}")
    declared: dict[str, int] = {}
    for name, itemsize in ceilings.items():
        if type(itemsize) is not int or not 1 <= itemsize <= torch.float32.itemsize:
            raise ValueError(
                f"materialization ceiling for {name!r} must be a positive integer"
                f" no greater than {torch.float32.itemsize}, got {itemsize!r}"
            )
        stored = direct[name]
        if (stored.is_floating_point() or stored.is_complex()) and stored.element_size() > itemsize:
            raise ValueError(
                f"module state {name!r} uses {stored.element_size()} bytes per element;"
                f" declared materialization ceiling is {itemsize}"
            )
        declared[name] = itemsize
    module.__dict__[_MATERIALIZATION_CEILINGS] = MappingProxyType(declared)


def declare_residency_unit(module: torch.nn.Module, *, expert: bool = False) -> None:
    """Group all state-owning descendants of ``module`` into one unit."""
    if type(expert) is not bool:
        raise TypeError("residency unit expert marker must be a boolean")
    module.__dict__[_RESIDENCY_UNIT_ROOT] = True
    module.__dict__[_RESIDENCY_UNIT_EXPERT] = expert


class ModuleStateStore(MutableMapping[str, StoredWeight]):
    """A live keyed view over one module tree's state-dict tensors.

    Quantized Linear weights fold their qdata and weight-scale buffers
    into one package-internal stored value. The physical scale
    state-dict entries remain registered on the module but are
    deliberately absent from this mapping.
    """

    def __init__(self, module: torch.nn.Module) -> None:
        self.module = module
        self._slots: dict[str, _StateSlot] = {}
        self._fp8: dict[str, Fp8Linear] = {}
        self._int8: dict[str, Int8Embedding | Int8Linear] = {}
        self._nvfp4: dict[str, Nvfp4Linear] = {}
        self._folded: dict[str, StoredWeight] = {}
        self._gguf_blocks: set[str] = set()
        self._quantization_protected: set[str] = set()
        self._materialization_ceilings: dict[str, int] = {}
        self._authorized_assignments: dict[str, _AuthorizedAssignment] = {}
        self._assignment_generation = 0

        for key in module.state_dict(keep_vars=True):
            prefix, _, name = key.rpartition(".")
            owner = module.get_submodule(prefix)
            if isinstance(owner, Nvfp4Linear) and name in {
                "weight_scale",
                "weight_scale_2",
                "input_scale",
                "pre_quant_scale",
            }:
                self._quantization_protected.add(key)
            if isinstance(owner, Int8Embedding | Int8Linear) and name == "weight_scale":
                self._quantization_protected.add(key)
            if isinstance(owner, GgufEncodedLinear) and name == "weight_blocks":
                self._quantization_protected.add(key)
                self._gguf_blocks.add(key)
            if isinstance(owner, Fp8Linear) and name == "weight_scale":
                continue
            if isinstance(owner, Int8Embedding | Int8Linear) and name == "weight_scale":
                continue
            if isinstance(owner, Nvfp4Linear) and name in {"weight_scale", "weight_scale_2"}:
                continue
            parameter = dict(owner.named_parameters(recurse=False, remove_duplicate=False)).get(
                name
            )
            if parameter is not None:
                self._slots[key] = _StateSlot(owner, name, True)
            else:
                buffer = dict(owner.named_buffers(recurse=False, remove_duplicate=False)).get(name)
                if buffer is None:
                    raise TypeError(f"state-dict key {key!r} is not direct module state")
                self._slots[key] = _StateSlot(owner, name, False)
            if isinstance(owner, Fp8Linear) and name == "weight":
                self._fp8[key] = owner
            if isinstance(owner, Int8Embedding | Int8Linear) and name == "weight":
                self._int8[key] = owner
            if isinstance(owner, Nvfp4Linear) and name == "weight":
                self._nvfp4[key] = owner
            declarations = getattr(owner, _MATERIALIZATION_CEILINGS, {})
            if not isinstance(declarations, Mapping):
                raise TypeError("residency materialization ceilings must be a mapping")
            declared = declarations.get(name)
            if declared is not None and (
                type(declared) is not int or not 1 <= declared <= torch.float32.itemsize
            ):
                raise ValueError(f"invalid materialization ceiling for module state {key!r}")
            self._materialization_ceilings[key] = (
                owner.compute_dtype.itemsize
                if isinstance(owner, Int8Embedding | Int8Linear) and name == "weight"
                else 1
                if isinstance(owner, GgufEncodedLinear) and name == "weight_blocks"
                else torch.float32.itemsize
                if declared is None
                else declared
            )

    def __getitem__(self, key: str) -> StoredWeight:
        fp8 = self._fp8.get(key)
        if fp8 is not None:
            cached = self._folded.get(key)
            if (
                isinstance(cached, Fp8ScaledWeight)
                and cached.qdata is fp8.weight
                and cached.scale is fp8.weight_scale
                and cached.orig_dtype == fp8.compute_dtype
            ):
                return cached
            folded = Fp8ScaledWeight(
                fp8.weight,
                fp8.weight_scale,
                fp8.compute_dtype,
            )
            self._folded[key] = folded
            return folded
        int8 = self._int8.get(key)
        if int8 is not None:
            cached = self._folded.get(key)
            if (
                isinstance(cached, Int8PackedWeight)
                and cached.qdata is int8.weight
                and cached.scale is int8.weight_scale
                and cached.orig_dtype == int8.compute_dtype
                and cached.convrot == int8.convrot
                and cached.convrot_groupsize == int8.convrot_groupsize
            ):
                return cached
            folded = Int8PackedWeight(
                int8.weight,
                int8.weight_scale,
                int8.compute_dtype,
                int8.convrot,
                int8.convrot_groupsize,
            )
            self._folded[key] = folded
            return folded
        nvfp4 = self._nvfp4.get(key)
        if nvfp4 is not None:
            cached = self._folded.get(key)
            logical_shape = (nvfp4.out_features, nvfp4.in_features)
            if (
                isinstance(cached, Nvfp4PackedWeight)
                and cached.qdata is nvfp4.weight
                and cached.block_scale is nvfp4.weight_scale
                and cached.tensor_scale is nvfp4.weight_scale_2
                and cached.logical_shape == logical_shape
                and cached.orig_dtype == nvfp4.compute_dtype
                and cached.recorder is nvfp4._diagnostics  # pyright: ignore[reportPrivateUsage]
            ):
                return cached
            folded = Nvfp4PackedWeight(
                nvfp4.weight,
                nvfp4.weight_scale,
                nvfp4.weight_scale_2,
                logical_shape,
                nvfp4.compute_dtype,
                nvfp4._diagnostics,  # pyright: ignore[reportPrivateUsage]
            )
            self._folded[key] = folded
            return folded
        slot = self._slots.get(key)
        if slot is None:
            raise KeyError(key)
        value = getattr(slot.module, slot.name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"module state {key!r} is not a tensor")
        return value

    def __setitem__(self, key: str, value: StoredWeight) -> None:
        self._set(key, value)

    def _set(self, key: str, value: StoredWeight) -> None:
        ceiling = self._materialization_ceilings.get(key)
        if ceiling is None:
            raise KeyError(key)
        self._folded.pop(key, None)
        if (
            isinstance(value, torch.Tensor)
            and (value.is_floating_point() or value.is_complex())
            and value.element_size() > ceiling
        ):
            raise ValueError(
                f"module state {key!r} uses {value.element_size()} bytes per element;"
                f" declared materialization ceiling is {ceiling}"
            )
        fp8 = self._fp8.get(key)
        if fp8 is not None:
            if not isinstance(value, Fp8ScaledWeight):
                raise TypeError(f"folded fp8 state {key!r} requires Fp8ScaledWeight")
            fp8.weight = (
                value.qdata
                if isinstance(value.qdata, torch.nn.Parameter)
                else torch.nn.Parameter(value.qdata, requires_grad=False)
            )
            fp8.weight_scale = value.scale
            return

        int8 = self._int8.get(key)
        if int8 is not None:
            if not isinstance(value, Int8PackedWeight):
                raise TypeError(f"folded INT8 state {key!r} requires Int8PackedWeight")
            int8.weight = (
                value.qdata
                if isinstance(value.qdata, torch.nn.Parameter)
                else torch.nn.Parameter(value.qdata, requires_grad=False)
            )
            int8.weight_scale = value.scale
            return

        nvfp4 = self._nvfp4.get(key)
        if nvfp4 is not None:
            if not isinstance(value, Nvfp4PackedWeight):
                raise TypeError(f"folded NVFP4 state {key!r} requires Nvfp4PackedWeight")
            nvfp4.weight = (
                value.qdata
                if isinstance(value.qdata, torch.nn.Parameter)
                else torch.nn.Parameter(value.qdata, requires_grad=False)
            )
            nvfp4.weight_scale = value.block_scale
            nvfp4.weight_scale_2 = value.tensor_scale
            return

        slot = self._slots.get(key)
        if slot is None:
            raise KeyError(key)
        if isinstance(value, Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight):
            raise TypeError(f"plain module state {key!r} requires a tensor")
        if slot.parameter:
            parameter = (
                value
                if isinstance(value, torch.nn.Parameter)
                else torch.nn.Parameter(value, requires_grad=False)
            )
            setattr(slot.module, slot.name, parameter)
        else:
            setattr(slot.module, slot.name, value)

    def _set_authorized(self, key: str, value: StoredWeight) -> None:
        self._set(key, value)
        current = self[key]
        if isinstance(current, torch.Tensor):
            try:
                version = int(current._version)
            except RuntimeError:
                if not current.is_inference():
                    raise
                version = None
            self._assignment_generation += 1
            self._authorized_assignments[key] = _AuthorizedAssignment(
                ref(current),
                ref(current.untyped_storage()),
                self._assignment_generation,
                version,
            )
        else:
            self._authorized_assignments.pop(key, None)

    def _authorized_generation(self, key: str, tensor: torch.Tensor) -> int | None:
        assignment = self._authorized_assignments.get(key)
        if assignment is None:
            return None
        try:
            current = self[key]
        except (KeyError, TypeError):
            return None
        if (
            current is not tensor
            or assignment.tensor() is not tensor
            or assignment.storage() is not tensor.untyped_storage()
        ):
            return None
        return assignment.generation

    def _authorized_version(self, key: str, tensor: torch.Tensor) -> int | None:
        assignment = self._authorized_assignments.get(key)
        if assignment is None or self._authorized_generation(key, tensor) is None:
            return None
        return assignment.version

    def __delitem__(self, key: str) -> None:
        raise TypeError("module state cannot be deleted")

    def __iter__(self) -> Iterator[str]:
        return iter(self._slots)

    def __len__(self) -> int:
        return len(self._slots)

    def max_materialized_itemsize(self, key: str) -> int:
        """Maximum bytes per tensor element materialized through this route."""
        return self._materialization_ceilings[key]

    def uses_raw_residency(self, key: str) -> bool:
        """Whether residency requests for one state key use its stored representation."""
        slot = self._slots.get(key)
        if slot is None:
            raise KeyError(key)
        capability = getattr(slot.module, "_residency_uses_raw_storage", None)
        return callable(capability) and bool(capability(slot.name))

    def storage_pointers(self, key: str) -> frozenset[int]:
        """Underlying storage pointers represented by one store key."""
        stored = self[key]
        if isinstance(stored, Fp8ScaledWeight):
            tensors = (stored.qdata, stored.scale)
        elif isinstance(stored, Int8PackedWeight):
            tensors = (stored.qdata, stored.scale)
        elif isinstance(stored, Nvfp4PackedWeight):
            tensors = (stored.qdata, stored.block_scale, stored.tensor_scale)
        else:
            tensors = (stored,)
        return frozenset(t.untyped_storage().data_ptr() for t in tensors)

    def protected_quantization_keys(self) -> frozenset[str]:
        return frozenset(self._quantization_protected)


class _ResidencyStateWriter(MutableMapping[str, StoredWeight]):
    """Private write capability for residency-owned module assignments."""

    def __init__(self, store: ModuleStateStore) -> None:
        self._store = store

    @property
    def module(self) -> torch.nn.Module:
        return self._store.module

    def __getitem__(self, key: str) -> StoredWeight:
        return self._store[key]

    def __setitem__(self, key: str, value: StoredWeight) -> None:
        self._store._set_authorized(key, value)  # pyright: ignore[reportPrivateUsage]

    def __delitem__(self, key: str) -> None:
        del self._store[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def max_materialized_itemsize(self, key: str) -> int:
        return self._store.max_materialized_itemsize(key)

    def uses_raw_residency(self, key: str) -> bool:
        return self._store.uses_raw_residency(key)


class _ScopedResidencyStateWriter(_ResidencyStateWriter):
    """Write capability restricted to one placement's state keys."""

    def __init__(self, store: ModuleStateStore, keys: Sequence[str]) -> None:
        super().__init__(store)
        self._keys = tuple(keys)
        self._key_set = frozenset(keys)

    def __getitem__(self, key: str) -> StoredWeight:
        if key not in self._key_set:
            raise KeyError(key)
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: StoredWeight) -> None:
        if key not in self._key_set:
            raise KeyError(key)
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        if key not in self._key_set:
            raise KeyError(key)
        super().__delitem__(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)


@dataclass(frozen=True)
class LayerLease:
    """One layer's prefix-scoped view of an open weight lease."""

    weights: WeightLease
    prefix: str
    _keys: dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    def get(self, name: str, *, dtype: torch.dtype) -> torch.Tensor:
        key = self._keys.get(name)
        if key is None:
            key = _join(self.prefix, name)
            self._keys[name] = key
        return self.weights.get(key, dtype=dtype)

    def get_stored(self, name: str) -> StoredWeight:
        key = self._keys.get(name)
        if key is None:
            key = _join(self.prefix, name)
            self._keys[name] = key
        return self.weights.get_stored(key)

    def timing_collector(self) -> PartialResidencyTiming | None:
        return self.weights.timing_collector()


class _LayerLeaseContext(AbstractContextManager[LayerLease]):
    __slots__ = ("_keys", "_manager", "_prefix")

    def __init__(
        self,
        manager: AbstractContextManager[WeightLease],
        prefix: str,
        keys: dict[str, str],
    ) -> None:
        self._manager = manager
        self._prefix = prefix
        self._keys = keys

    def __enter__(self) -> LayerLease:
        return LayerLease(self._manager.__enter__(), self._prefix, self._keys)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return self._manager.__exit__(exc_type, exc_value, traceback)


@dataclass(frozen=True)
class ResidencyBinding:
    """One layer's route into its owning residency unit."""

    mechanism: UnitResidency
    unit: str
    prefix: str
    store: ModuleStateStore
    unit_state: ResidencyUnitState = field(init=False, repr=False)
    _keys: dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit_state", self.mechanism.unit_state(self.unit))

    def key(self, name: str) -> str:
        key = self._keys.get(name)
        if key is None:
            key = _join(self.prefix, name)
            self._keys[name] = key
        return key

    def lease(self) -> AbstractContextManager[LayerLease]:
        """Open the one lease bracket for this layer's forward.

        The operation consuming leased weights must execute inside the
        bracket. Leased tensors must not be stashed or used after it
        closes; only the operation's output tensors may escape.
        """
        return _LayerLeaseContext(self.mechanism.lease(self.unit), self.prefix, self._keys)


def _join(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _direct_store_keys(
    module: torch.nn.Module, store: ModuleStateStore
) -> dict[str, tuple[str, ...]]:
    owners = {prefix for prefix, _owner in module.named_modules()}
    grouped: dict[str, list[str]] = {}
    for key in store:
        prefix, _, _name = key.rpartition(".")
        if prefix in owners:
            grouped.setdefault(prefix, []).append(key)
    return {prefix: tuple(keys) for prefix, keys in grouped.items()}


def _units_and_names(
    module: torch.nn.Module, store: ModuleStateStore
) -> tuple[tuple[ResidencyUnit, ...], dict[str, str]]:
    direct = _direct_store_keys(module, store)
    names = sorted(direct)
    parent = {name: name for name in names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        a = find(left)
        b = find(right)
        if a == b:
            return
        first, second = sorted((a, b))
        parent[second] = first

    owners_by_pointer: dict[int, str] = {}
    for name in names:
        for key in direct[name]:
            for pointer in store.storage_pointers(key):
                other = owners_by_pointer.get(pointer)
                if other is None:
                    owners_by_pointer[pointer] = name
                else:
                    union(name, other)

    declared_units = {
        prefix: owner.__dict__.get(_RESIDENCY_UNIT_EXPERT, False)
        for prefix, owner in module.named_modules()
        if owner.__dict__.get(_RESIDENCY_UNIT_ROOT) is True
    }
    declared_prefixes = tuple(declared_units)
    declaration_order = {prefix: index for index, prefix in enumerate(declared_prefixes)}
    members_by_declaration: dict[str, list[str]] = {prefix: [] for prefix in declared_prefixes}
    for name in names:
        declarations: list[str] = []
        candidate = name
        while True:
            if candidate in members_by_declaration:
                declarations.append(candidate)
            if not candidate:
                break
            candidate, _, _part = candidate.rpartition(".")
        declarations.sort(key=declaration_order.__getitem__)
        if len(declarations) > 1:
            raise ValueError(
                f"declared residency units {declarations[0]!r} and {declarations[1]!r}"
                f" overlap at {name!r}"
            )
        if declarations:
            members_by_declaration[declarations[0]].append(name)

    declared: list[tuple[str, tuple[str, ...]]] = []
    for prefix in declared_prefixes:
        members = tuple(members_by_declaration[prefix])
        if not members:
            raise ValueError(f"declared residency unit {prefix!r} owns no persistent state")
        declared.append((prefix, members))

    tied_groups: dict[str, set[str]] = {}
    for name in names:
        tied_groups.setdefault(find(name), set()).add(name)

    declared_names: dict[str, str] = {}
    expert_roots: set[str] = set()
    for prefix, members in declared:
        member_set = set(members)
        for root in {find(member) for member in members}:
            if not tied_groups[root] <= member_set:
                raise ValueError(
                    f"declared residency unit {prefix!r} has tied state outside its subtree"
                )
        for member in members[1:]:
            union(members[0], member)
        root = find(members[0])
        declared_names[root] = prefix
        if declared_units[prefix]:
            expert_roots.add(root)

    grouped: dict[str, list[str]] = {}
    for name in names:
        grouped.setdefault(find(name), []).append(name)

    units: list[ResidencyUnit] = []
    unit_of_module: dict[str, str] = {}
    for members in sorted(sorted(group) for group in grouped.values()):
        root = find(members[0])
        unit_name = declared_names.get(root, members[0])
        unit_keys = tuple(key for name in members for key in direct[name])
        units.append(ResidencyUnit(unit_name, unit_keys, expert=root in expert_roots))
        for name in members:
            unit_of_module[name] = unit_name
    return tuple(units), unit_of_module


@dataclass(frozen=True)
class _ComponentEnrollment:
    module: torch.nn.Module
    store: ModuleStateStore
    writer: _ResidencyStateWriter
    units: tuple[ResidencyUnit, ...]
    state_owners: tuple[tuple[str, torch.nn.Module, str], ...]


def _require_not_enrolled(module: torch.nn.Module) -> None:
    if hasattr(module, "_dinkster_resident_weights"):
        raise RuntimeError("module is already enrolled for residency")


def _prepare_component(module: torch.nn.Module) -> _ComponentEnrollment:
    _require_not_enrolled(module)

    store = ModuleStateStore(module)
    units, unit_of_module = _units_and_names(module, store)
    state_owners = tuple(
        (prefix, owner, unit_of_module[prefix])
        for prefix, owner in module.named_modules()
        if prefix in unit_of_module
    )
    return _ComponentEnrollment(module, store, _ResidencyStateWriter(store), units, state_owners)


def _storage_dtype_targets(
    prepared: _ComponentEnrollment,
    component_target: torch.dtype,
    *,
    respect_bound_compute_dtype: bool = True,
) -> dict[str, torch.dtype]:
    direct = _direct_store_keys(prepared.module, prepared.store)
    targets: dict[str, torch.dtype] = {}
    for prefix, owner, _unit in prepared.state_owners:
        target = (
            bound_compute_dtype(owner) or component_target
            if respect_bound_compute_dtype
            else component_target
        )
        for key in direct[prefix]:
            targets[key] = target
    if targets.keys() != prepared.store.keys():
        raise RuntimeError("storage dtype targets do not cover every component state key")
    return targets


def _validate_routes(
    prepared: _ComponentEnrollment,
    *,
    policy_component: str | None = None,
) -> None:
    for prefix, owner, _unit in prepared.state_owners:
        if isinstance(
            owner,
            Fp8Linear | GgufEncodedLinear | Int8Embedding | Int8Linear | Nvfp4Linear | T5LayerNorm,
        ) or bind_residency_layer(owner):
            continue
        if policy_component is not None:
            raise StorageDtypePolicyError(
                policy_component,
                "unrouted_state",
                f"module {prefix!r} owns state but has no residency route",
            )
        raise TypeError(f"module {prefix!r} owns state but has no residency route")


def _preflight_quantization_patches(
    prepared: _ComponentEnrollment,
    patch_set: PatchSet[torch.Tensor] | None,
) -> None:
    if patch_set is None:
        return
    for key in prepared.store.protected_quantization_keys():
        if patch_set.entries(key):
            raise PatchApplyError(
                f"packed quantization-state patch/requantization is not supported: {key!r}"
            )


def _preflight_patch_targets(
    prepared: _ComponentEnrollment,
    patch_set: PatchSet[torch.Tensor] | None,
    *,
    component: str,
) -> None:
    if patch_set is None:
        return
    keys = set(prepared.store)
    missing = sorted(key for key in patch_set.keys() if patch_set.entries(key) and key not in keys)
    if missing:
        raise PatchApplyError(
            f"patch targets are not in component {component!r}: "
            + ", ".join(repr(key) for key in missing)
        )


def _enroll_prepared(
    prepared: _ComponentEnrollment,
    *,
    load_device: torch.device | str,
    offload_device: torch.device | str,
    patch_set: PatchSet[torch.Tensor] | None,
    intermediate_dtype: torch.dtype,
    patch_weight_dtype: torch.dtype | None,
    patch_key_prefix: str,
    mechanism_factory: ResidencyMechanismFactory,
) -> EnrolledResidency:
    _preflight_quantization_patches(prepared, patch_set)
    mechanism = _construct_mechanism(
        prepared,
        load_device=load_device,
        offload_device=offload_device,
        patch_set=patch_set,
        intermediate_dtype=intermediate_dtype,
        patch_weight_dtype=patch_weight_dtype,
        patch_key_prefix=patch_key_prefix,
        mechanism_factory=mechanism_factory,
    )

    bound: list[tuple[torch.nn.Module, bool, ResidencyBinding | None]] = []
    try:
        bound = _bind_mechanism(prepared, mechanism)
        prepared.module.__dict__["_dinkster_resident_weights"] = mechanism
        prepared.module.__dict__[_RESIDENCY_STATE_STORE] = prepared.store
    except BaseException as error:
        prepared.module.__dict__.pop("_dinkster_resident_weights", None)
        prepared.module.__dict__.pop(_RESIDENCY_STATE_STORE, None)
        _restore_bindings(bound)
        try:
            mechanism.unload()
        except BaseException as cleanup_error:
            error.add_note(f"residency enrollment cleanup also failed: {cleanup_error!r}")
        raise
    return mechanism


def _storage_dtype_outcome(
    store: ModuleStateStore, targets: Mapping[str, torch.dtype]
) -> StorageDtypeOutcome:
    if store._int8 or store._gguf_blocks:  # pyright: ignore[reportPrivateUsage]
        return "quantized"
    tensors: list[tuple[str, torch.Tensor]] = []
    for key in store:
        stored = store[key]
        if isinstance(stored, Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight):
            return "quantized"
        if stored.is_quantized:
            return "quantized"
        tensors.append((key, stored))

    numeric = [
        (key, tensor)
        for key, tensor in tensors
        if tensor.is_floating_point() or tensor.is_complex()
    ]
    if not numeric:
        return "no_floating_state"
    if any(tensor.dtype in FP8_DTYPES for _key, tensor in numeric):
        return "quantized"
    if all(tensor.dtype == targets[key] for key, tensor in numeric):
        return "already_at_target"
    supported = (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float16,
        torch.bfloat16,
        torch.float32,
    )
    if any(targets[key] not in supported for key, _tensor in numeric):
        return "unsupported_dtype"
    if any(tensor.is_complex() or tensor.layout is not torch.strided for _key, tensor in numeric):
        return "unsupported_dtype"
    if any(type(tensor) not in (torch.Tensor, torch.nn.Parameter) for _key, tensor in numeric):
        return "unsupported_dtype"
    dtypes = {tensor.dtype for _key, tensor in numeric}
    if torch.float64 in dtypes:
        return "unsupported_dtype"
    if any(tensor.dtype not in supported for _key, tensor in numeric):
        return "unsupported_dtype"
    aliases: dict[int, torch.dtype] = {}
    for key, tensor in numeric:
        target = targets[key]
        previous = aliases.setdefault(id(tensor), target)
        if previous != target:
            return "mixed_floating"
    if any(targets[key].itemsize > tensor.dtype.itemsize for key, tensor in numeric):
        return "storage_widening"
    return "converted"


def _unmanaged_buffers(module: torch.nn.Module) -> tuple[str, ...]:
    persistent = set(module.state_dict(keep_vars=True))
    unmanaged = {
        name
        for name, _buffer in module.named_buffers(remove_duplicate=False)
        if name not in persistent
    }
    constants = getattr(module, "_dinkster_residency_constant_buffers", frozenset())
    if not isinstance(constants, frozenset) or not all(isinstance(name, str) for name in constants):
        raise TypeError("residency constant buffers must be a frozenset of strings")
    if unknown := constants - unmanaged:
        raise TypeError(
            "residency constants must name non-persistent buffers: "
            + ", ".join(repr(name) for name in sorted(unknown))
        )
    return tuple(sorted(unmanaged - constants))


def _snapshot_store(store: ModuleStateStore) -> dict[str, torch.Tensor]:
    snapshots: dict[str, torch.Tensor] = {}
    for key in store:
        stored = store[key]
        if isinstance(stored, Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight):
            raise AssertionError("eligible storage cannot contain packed quantization")
        snapshots[key] = stored
    return snapshots


def _restore_store(store: _ResidencyStateWriter, snapshots: Mapping[str, torch.Tensor]) -> None:
    for key, original in snapshots.items():
        store[key] = original


@dataclass(frozen=True)
class _PreparedConversion:
    snapshots: Mapping[str, torch.Tensor]
    replacements: Mapping[str, torch.Tensor]


def _refuse_overlapping_views(
    groups: Mapping[int, Sequence[torch.Tensor]], *, component: str
) -> None:
    by_device: dict[torch.device, list[tuple[int, _OccupiedBytes]]] = {}
    for identity, tensors in groups.items():
        original = tensors[0]
        by_device.setdefault(original.device, []).append((identity, _occupied_bytes(original)))
    for regions in by_device.values():
        active: list[tuple[int, _OccupiedBytes]] = []
        for identity, region in sorted(regions, key=lambda item: item[1].start):
            active = [item for item in active if item[1].end > region.start]
            if any(
                other_identity != identity and _occupied_bytes_overlap(other, region)
                for other_identity, other in active
            ):
                raise PatchApplyError(
                    f"storage conversion refused overlapping storage in component {component!r}"
                )
            active.append((identity, region))


@dataclass(frozen=True)
class _OccupiedBytes:
    start: int
    step: int
    count: int
    width: int

    @property
    def end(self) -> int:
        return self.start + (self.count - 1) * self.step + self.width


def _occupied_bytes(tensor: torch.Tensor) -> _OccupiedBytes:
    if tensor.numel() == 0:
        return _OccupiedBytes(tensor.untyped_storage().data_ptr(), 0, 0, 0)
    if any(stride < 0 for stride in tensor.stride()):
        raise PatchApplyError("storage conversion refused negative-stride storage")
    itemsize = tensor.element_size()
    base = tensor.untyped_storage().data_ptr() + int(tensor.storage_offset()) * itemsize
    dimensions = tuple(
        (int(size), int(stride))
        for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
        if size > 1 and stride != 0
    )
    if not dimensions:
        return _OccupiedBytes(base, 0, 1, itemsize)
    ordered = sorted(dimensions, key=lambda item: item[1])
    step = ordered[0][1]
    expected_stride = step
    count = 1
    for size, stride in ordered:
        if stride != expected_stride:
            raise PatchApplyError("storage conversion refused unsupported strided storage")
        count *= size
        expected_stride = step * count
    return _OccupiedBytes(base, step * itemsize, count, itemsize)


def _occupied_bytes_overlap(left: _OccupiedBytes, right: _OccupiedBytes) -> bool:
    if left.count == 0 or right.count == 0:
        return False
    for left_offset in range(left.width):
        for right_offset in range(right.width):
            if _bounded_progressions_intersect(
                left.start + left_offset,
                left.step,
                left.count,
                right.start + right_offset,
                right.step,
                right.count,
            ):
                return True
    return False


def _bounded_progressions_intersect(
    left: int,
    left_step: int,
    left_count: int,
    right: int,
    right_step: int,
    right_count: int,
) -> bool:
    if left_step == 0:
        return (right_step == 0 and left == right) or (
            right_step != 0
            and right <= left <= right + (right_count - 1) * right_step
            and (left - right) % right_step == 0
        )
    if right_step == 0:
        return (
            left <= right <= left + (left_count - 1) * left_step and (right - left) % left_step == 0
        )

    divisor = gcd(left_step, right_step)
    difference = right - left
    if difference % divisor:
        return False
    reduced_right_step = right_step // divisor
    left_index = (
        ((difference // divisor) * pow(left_step // divisor, -1, reduced_right_step))
        % reduced_right_step
        if reduced_right_step != 1
        else 0
    )
    period = left_step * reduced_right_step
    residue = (left + left_index * left_step) % period
    low = max(left, right)
    high = min(
        left + (left_count - 1) * left_step,
        right + (right_count - 1) * right_step,
    )
    first = residue + max(0, (low - residue + period - 1) // period) * period
    return first <= high


def _prepare_storage_conversion(
    prepared: _ComponentEnrollment,
    *,
    component: str,
    targets: Mapping[str, torch.dtype],
    offload_device: torch.device | str,
    patch_set: PatchSet[torch.Tensor] | None,
    patch_weight_dtype: torch.dtype,
    patch_key_prefix: str,
) -> _PreparedConversion:
    """Build one component's authoritative storage without mutating it."""
    store = prepared.store
    groups: dict[int, list[str]] = {}
    originals: dict[int, torch.Tensor] = {}
    for key in store:
        stored = store[key]
        if isinstance(stored, Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight):
            raise AssertionError("eligible storage cannot contain packed quantization")
        identity = id(stored)
        groups.setdefault(identity, []).append(key)
        originals[identity] = stored
    _refuse_overlapping_views(
        {identity: (original,) for identity, original in originals.items()},
        component=component,
    )

    snapshots = _snapshot_store(store)
    replacements: dict[str, torch.Tensor] = {}
    device = torch.device(offload_device)
    for identity, aliases in groups.items():
        active = [key for key in aliases if patch_set is not None and patch_set.entries(key)]
        if len(active) > 1:
            raise PatchApplyError(
                f"storage conversion refused ambiguous tied patch targets in"
                f" component {component!r}: " + ", ".join(repr(key) for key in active)
            )

        original = originals[identity]
        target = targets[aliases[0]]
        if any(targets[key] != target for key in aliases[1:]):
            raise AssertionError("eligible tied storage must have one conversion target")
        entries = () if not active or patch_set is None else patch_set.entries(active[0])
        seed = string_to_seed(f"{patch_key_prefix}{active[0] if active else aliases[0]}")
        if original.is_floating_point():
            if not entries and original.dtype == target and original.device == device:
                for key in aliases:
                    replacements[key] = original
                continue
            if entries:
                value = original.to(device=device, dtype=patch_weight_dtype, copy=True)
                value = apply_patches(
                    value,
                    entries,
                    key=f"{patch_key_prefix}{active[0]}",
                    intermediate_dtype=torch.float32,
                    original_weight=original,
                )
                value = stochastic_rounding(value.detach(), target, seed=seed)
            else:
                value = stochastic_rounding(
                    original.detach().to(device=device, copy=True), target, seed=seed
                )
        else:
            if not entries and original.device == device:
                replacement = original
                for key in aliases:
                    replacements[key] = replacement
                continue
            value = original.to(device=device, copy=True)
            if entries:
                value = apply_patches(
                    value.to(dtype=patch_weight_dtype),
                    entries,
                    key=f"{patch_key_prefix}{active[0]}",
                    intermediate_dtype=torch.float32,
                    original_weight=original,
                ).to(dtype=original.dtype)
        replacement: torch.Tensor
        if isinstance(original, torch.nn.Parameter):
            replacement = torch.nn.Parameter(value, requires_grad=original.requires_grad)
        else:
            replacement = value
        for key in aliases:
            replacements[key] = replacement
    return _PreparedConversion(snapshots, replacements)


def _snapshot_patch_set(
    patch_set: PatchSet[torch.Tensor] | None,
) -> PatchSet[torch.Tensor] | None:
    if patch_set is None:
        return None
    patches = {}
    for key in patch_set.keys():
        entries = patch_set.entries(key)
        payloads = tuple(
            payload.detach().to(device="cpu", copy=True) for payload in patch_payloads(entries)
        )
        patches[key] = rebuild_patch_entries(entries, payloads)
    return PatchSet(patches, structural_digest=patch_set.structural_digest)


def _construct_mechanism(
    prepared: _ComponentEnrollment,
    *,
    load_device: torch.device | str,
    offload_device: torch.device | str,
    patch_set: PatchSet[torch.Tensor] | None,
    intermediate_dtype: torch.dtype,
    patch_weight_dtype: torch.dtype | None,
    patch_key_prefix: str,
    mechanism_factory: ResidencyMechanismFactory,
) -> EnrolledResidency:
    if patch_weight_dtype is None and not patch_key_prefix:
        return mechanism_factory(
            prepared.writer,
            load_device=load_device,
            offload_device=offload_device,
            patch_set=patch_set,
            units=prepared.units,
            intermediate_dtype=intermediate_dtype,
        )
    configured_factory = cast(_ConfiguredResidencyMechanismFactory, mechanism_factory)
    return configured_factory(
        prepared.writer,
        load_device=load_device,
        offload_device=offload_device,
        patch_set=patch_set,
        units=prepared.units,
        intermediate_dtype=intermediate_dtype,
        patch_weight_dtype=patch_weight_dtype,
        patch_key_prefix=patch_key_prefix,
    )


def _bind_mechanism(
    prepared: _ComponentEnrollment, mechanism: EnrolledResidency
) -> list[tuple[torch.nn.Module, bool, ResidencyBinding | None]]:
    bound: list[tuple[torch.nn.Module, bool, ResidencyBinding | None]] = []
    try:
        for prefix, owner, unit in prepared.state_owners:
            existed = "_residency" in owner.__dict__
            previous = owner.__dict__.get("_residency")
            bound.append((owner, existed, previous))
            binding = ResidencyBinding(mechanism, unit, prefix, prepared.store)
            if isinstance(
                owner,
                Fp8Linear
                | GgufEncodedLinear
                | Int8Embedding
                | Int8Linear
                | Nvfp4Linear
                | T5LayerNorm,
            ):
                owner.bind_residency(binding)
            else:
                if not bind_residency_layer(owner, binding):
                    raise TypeError(f"module {prefix!r} owns state but has no residency route")
    except BaseException:
        _restore_bindings(bound)
        raise
    return bound


def _restore_bindings(
    bound: Sequence[tuple[torch.nn.Module, bool, ResidencyBinding | None]],
) -> None:
    for owner, existed, previous in reversed(bound):
        if existed:
            owner.__dict__["_residency"] = previous
        else:
            owner.__dict__.pop("_residency", None)


def detach_residency_enrollment(module: torch.nn.Module, mechanism: EnrolledResidency) -> None:
    """Detach one component's terminal residency ownership."""
    store = module.__dict__.get(_RESIDENCY_STATE_STORE)
    if not isinstance(store, ModuleStateStore):
        raise RuntimeError("module has no residency enrollment to detach")
    if module.__dict__.get("_dinkster_resident_weights") is not mechanism:
        raise RuntimeError("module residency enrollment is owned by another mechanism")

    for owner in module.modules():
        binding = owner.__dict__.get("_residency")
        if (
            isinstance(binding, ResidencyBinding)
            and binding.mechanism is mechanism
            and binding.store is store
        ):
            owner.__dict__.pop("_residency", None)
            if owner.__dict__.get("_residency_prefetch_binding") is binding:
                owner.__dict__.pop("_residency_prefetch_binding", None)
                owner.__dict__.pop("_residency_prefetch_requests", None)

    module.__dict__.pop("_dinkster_resident_weights", None)
    module.__dict__.pop(_RESIDENCY_STATE_STORE, None)


def _fp8_mps_scope(component: str | None, family: str | None) -> str:
    scope = ""
    if component is not None:
        scope += f" in component {component!r}"
    if family is not None:
        scope += f" of family {family}"
    return scope


def _upcast_fp8_linear(layer: Fp8Linear) -> torch.nn.Linear:
    """One-time dequantization of a scaled-fp8 Linear into a plain
    routed Linear at the layer's compute dtype. The effective weight
    is ``weight.to(compute_dtype) * weight_scale.to(compute_dtype)`` -
    the exact product the fp8 dequant route computes per forward - so
    forwards through the replacement are bitwise identical to
    cast-at-use dequantization."""
    with torch.device("meta"):
        replacement = INITLESS.linear(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
        )
    stored = layer.stored()
    weight = stored.qdata.to(dtype=layer.compute_dtype)
    weight.mul_(stored.scale.to(dtype=layer.compute_dtype))
    replacement.weight = torch.nn.Parameter(
        weight,
        requires_grad=False,
    )
    if layer.bias is not None:
        replacement.bias = torch.nn.Parameter(
            layer.bias.data.to(dtype=layer.compute_dtype), requires_grad=False
        )
    return replacement


@dataclass(frozen=True, slots=True)
class _Fp8UpcastPlan:
    """Validated fp8-to-compute-dtype rewrites, not yet applied."""

    swaps: tuple[tuple[torch.nn.Module, str, Fp8Linear], ...]
    casts: tuple[tuple[torch.nn.Module, str, torch.Tensor, bool, torch.dtype], ...]


def _fp8_upcast_nbytes(plans: Sequence[_Fp8UpcastPlan]) -> int:
    """Peak bytes of compute-dtype storage materialized by the plans."""
    total = 0
    peak = 0
    seen_swaps: set[int] = set()
    seen_casts: set[tuple[int, torch.dtype]] = set()
    for plan in plans:
        for _owner, _name, child in plan.swaps:
            if id(child) in seen_swaps:
                continue
            seen_swaps.add(id(child))
            total += child.weight.numel() * child.compute_dtype.itemsize
            scale_bytes = (
                0
                if child.weight_scale.dtype == child.compute_dtype
                else child.weight_scale.numel() * child.compute_dtype.itemsize
            )
            peak = max(peak, total + scale_bytes)
            if child.bias is not None and child.bias.dtype != child.compute_dtype:
                total += child.bias.numel() * child.compute_dtype.itemsize
            peak = max(peak, total)
        for _owner, _name, tensor, _is_parameter, target in plan.casts:
            key = (id(tensor), target)
            if key in seen_casts:
                continue
            seen_casts.add(key)
            total += tensor.numel() * target.itemsize
            peak = max(peak, total)
    return peak


def _admit_fp8_upcast_for_mps(
    plans: Sequence[_Fp8UpcastPlan],
    device: torch.device,
    *,
    memory_policy: MemoryPolicy,
    mps_snapshot: Callable[[torch.device], MpsMemorySnapshot],
) -> None:
    """Diagnose upcast pressure without replacing actual allocation checks."""
    planned_bytes = _fp8_upcast_nbytes(plans)
    if planned_bytes == 0:
        return
    measured = mps_snapshot(device)
    reserve = memory_policy.minimum_inference_memory()
    available = measured.system_available_bytes
    if planned_bytes > max(0, available - reserve):
        logger.warning(
            f"MPS FP8 upcast needs {planned_bytes} bytes of anonymous"
            f" compute-dtype storage plus the {reserve}-byte inference reserve, but the"
            f" system has {available} bytes available ({measured.describe()});"
            " attempting CPU dequantization and memory-budgeted loading"
        )


def _plan_fp8_upcast_for_mps(
    module: torch.nn.Module,
    *,
    component: str | None = None,
    family: str | None = None,
) -> _Fp8UpcastPlan:
    """Plan MPS-compatible storage without mutating module identity or state."""
    scope = _fp8_mps_scope(component, family)
    if isinstance(module, Fp8Linear):
        raise RuntimeError(
            f"cannot enroll a bare scaled-fp8 Linear{scope} on an MPS load"
            " device: upcasting needs"
            " an owning parent module to hold the replacement layer"
        )
    fp8_matmul_message = (
        "layer {qualified!r}{scope} is bound to the fp8 hardware matmul,"
        " which is unavailable on MPS; using CPU dequantization to the"
        " bound compute dtype and portable linear execution"
    )
    swaps: list[tuple[torch.nn.Module, str, Fp8Linear]] = []
    casts: list[tuple[torch.nn.Module, str, torch.Tensor, bool, torch.dtype]] = []
    for prefix, owner in tuple(module.named_modules(remove_duplicate=False)):
        if isinstance(owner, Fp8Linear):
            continue  # replaced through its parent below
        # Registration slots, not named_children(): the latter deduplicates
        # a child registered under two names, which would leave the second
        # alias unreplaced.
        slots = owner._modules  # pyright: ignore[reportPrivateUsage]
        for name, child in tuple(slots.items()):
            if not isinstance(child, Fp8Linear):
                continue
            qualified = f"{prefix}.{name}" if prefix else name
            if child.fp8_matmul:
                logger.warning(fp8_matmul_message.format(qualified=qualified, scope=scope))
            swaps.append((owner, name, child))
        if isinstance(owner, Int8Embedding | Int8Linear | Nvfp4Linear):
            continue  # packed-quantization scale state, not cast-at-use fp8
        state: tuple[tuple[str, torch.Tensor, bool], ...] = (
            *(
                (name, parameter, True)
                for name, parameter in owner.named_parameters(recurse=False, remove_duplicate=False)
            ),
            *(
                (name, buffer, False)
                for name, buffer in owner.named_buffers(recurse=False, remove_duplicate=False)
            ),
        )
        for name, tensor, is_parameter in state:
            if tensor.dtype not in FP8_DTYPES:
                continue
            qualified = f"{prefix}.{name}" if prefix else name
            if getattr(owner, "fp8_matmul", False):
                logger.warning(fp8_matmul_message.format(qualified=qualified, scope=scope))
            target = bound_compute_dtype(owner)
            if target is None:
                raise RuntimeError(
                    f"module state {qualified!r}{scope} is stored as"
                    f" {tensor.dtype} with no bound compute dtype to upcast"
                    " to"
                )
            casts.append((owner, name, tensor, is_parameter, target))
    return _Fp8UpcastPlan(tuple(swaps), tuple(casts))


def _commit_fp8_upcasts(plans: Sequence[_Fp8UpcastPlan]) -> None:
    replacements: dict[int, torch.nn.Linear] = {}
    upcast_state: dict[tuple[int, torch.dtype], torch.Tensor] = {}
    for plan in plans:
        for owner, name, child in plan.swaps:
            replacement = replacements.get(id(child))
            if replacement is None:
                replacement = _upcast_fp8_linear(child)
                replacements[id(child)] = replacement
            setattr(owner, name, replacement)
        for owner, name, tensor, is_parameter, target in plan.casts:
            upcast = upcast_state.get((id(tensor), target))
            if upcast is None:
                upcast = tensor.data.to(dtype=target)
                upcast_state[(id(tensor), target)] = upcast
            # The binder recognizes FP8 storage before its dtype changes.
            bind_fp8_matmul_layer(owner, False)
            if is_parameter:
                setattr(owner, name, torch.nn.Parameter(upcast, requires_grad=False))
            else:
                setattr(owner, name, upcast)


@overload
def enroll_component(
    module: torch.nn.Module,
    *,
    load_device: torch.device | str,
    offload_device: torch.device | str,
    patch_set: PatchSet[torch.Tensor] | None = None,
    intermediate_dtype: torch.dtype = torch.float32,
    patch_weight_dtype: torch.dtype | None = None,
    patch_key_prefix: str = "",
    memory_policy: MemoryPolicy | None = None,
    mps_snapshot: Callable[[torch.device], MpsMemorySnapshot] | None = None,
) -> ResidentWeights: ...


@overload
def enroll_component(
    module: torch.nn.Module,
    *,
    load_device: torch.device | str,
    offload_device: torch.device | str,
    patch_set: PatchSet[torch.Tensor] | None = None,
    intermediate_dtype: torch.dtype = torch.float32,
    patch_weight_dtype: torch.dtype | None = None,
    patch_key_prefix: str = "",
    mechanism_factory: ResidencyMechanismFactory,
    memory_policy: MemoryPolicy | None = None,
    mps_snapshot: Callable[[torch.device], MpsMemorySnapshot] | None = None,
) -> EnrolledResidency: ...


def enroll_component(
    module: torch.nn.Module,
    *,
    load_device: torch.device | str,
    offload_device: torch.device | str,
    patch_set: PatchSet[torch.Tensor] | None = None,
    intermediate_dtype: torch.dtype = torch.float32,
    patch_weight_dtype: torch.dtype | None = None,
    patch_key_prefix: str = "",
    mechanism_factory: ResidencyMechanismFactory = ResidentWeights,
    memory_policy: MemoryPolicy | None = None,
    mps_snapshot: Callable[[torch.device], MpsMemorySnapshot] | None = None,
) -> EnrolledResidency:
    """Enroll one assembled component in per-unit residency.

    MPS load devices rewrite fp8 storage into compute-dtype storage
    first (see :func:`enroll_assembled`); torch has no fp8 tensors on
    MPS.
    """
    resolved_load_device = torch.device(load_device)
    if resolved_load_device.type == "mps":
        _require_not_enrolled(module)
        upcast_plan = _plan_fp8_upcast_for_mps(module)
        _admit_fp8_upcast_for_mps(
            (upcast_plan,),
            resolved_load_device,
            memory_policy=MemoryPolicy() if memory_policy is None else memory_policy,
            mps_snapshot=mps_memory_snapshot if mps_snapshot is None else mps_snapshot,
        )
        _commit_fp8_upcasts((upcast_plan,))
    prepared = _prepare_component(module)
    _validate_routes(prepared)
    return _enroll_prepared(
        prepared,
        load_device=load_device,
        offload_device=offload_device,
        patch_set=patch_set,
        intermediate_dtype=intermediate_dtype,
        patch_weight_dtype=patch_weight_dtype,
        patch_key_prefix=patch_key_prefix,
        mechanism_factory=mechanism_factory,
    )


def enroll_component_placement(
    module: torch.nn.Module,
    placements: Mapping[str, torch.device | str],
    *,
    offload_device: torch.device | str,
    patch_set: PatchSet[torch.Tensor] | None = None,
    intermediate_dtype: torch.dtype = torch.float32,
    patch_weight_dtype: torch.dtype | None = None,
    patch_key_prefix: str = "",
    mechanism_factory: ResidencyMechanismFactory = ResidentWeights,
) -> ComponentResidencyPlacement:
    """Enroll disjoint module subtrees into one mechanism per device.

    Placement keys are module paths. Each state owner uses its longest
    matching path, and tied storage must remain on one device.
    """
    if not placements:
        raise ValueError("component residency placements must be a non-empty mapping")
    prepared = _prepare_component(module)
    _validate_routes(prepared)
    _preflight_quantization_patches(prepared, patch_set)
    _preflight_patch_targets(prepared, patch_set, component="placed component")
    module_paths = frozenset(name for name, _owner in module.named_modules())
    devices: dict[str, torch.device] = {}
    for prefix, value in placements.items():
        if type(prefix) is not str or prefix not in module_paths:
            raise ValueError(f"component residency placement names unknown module {prefix!r}")
        device = torch.device(value)
        if device.type not in ("cpu", "cuda"):
            logger.warning(
                "Component residency placement on %s uses the portable torch path", device
            )
        if device.type == "cuda" and ":" not in str(device):
            raise ValueError("component residency CUDA placement requires an explicit index")
        devices[prefix] = device

    owner_device: dict[str, torch.device] = {}
    used: set[str] = set()
    for prefix, _owner, _unit in prepared.state_owners:
        matches = tuple(
            candidate
            for candidate in devices
            if not candidate or prefix == candidate or prefix.startswith(f"{candidate}.")
        )
        if not matches:
            raise ValueError(f"component residency placement does not cover module {prefix!r}")
        selected = max(matches, key=len)
        owner_device[prefix] = devices[selected]
        used.add(selected)
    unused = devices.keys() - used
    if unused:
        raise ValueError(f"component residency placements own no state: {sorted(unused)!r}")

    unit_devices: dict[str, torch.device] = {}
    for prefix, _owner, unit in prepared.state_owners:
        device = owner_device[prefix]
        previous = unit_devices.setdefault(unit, device)
        if previous != device:
            raise ValueError(
                f"residency unit {unit!r} contains tied state assigned to multiple devices"
            )

    units_by_device: dict[torch.device, list[ResidencyUnit]] = {}
    for unit in prepared.units:
        units_by_device.setdefault(unit_devices[unit.name], []).append(unit)

    mechanisms: dict[torch.device, EnrolledResidency] = {}
    scoped_prepared: dict[torch.device, _ComponentEnrollment] = {}
    try:
        for device, units in units_by_device.items():
            keys = tuple(key for unit in units for key in unit.keys)
            scoped_patch = (
                None
                if patch_set is None
                else PatchSet(
                    {key: patch_set.entries(key) for key in keys if patch_set.entries(key)},
                    structural_digest=patch_set.structural_digest,
                )
            )
            scoped = _ComponentEnrollment(
                prepared.module,
                prepared.store,
                _ScopedResidencyStateWriter(prepared.store, keys),
                tuple(units),
                tuple(owner for owner in prepared.state_owners if owner_device[owner[0]] == device),
            )
            scoped_prepared[device] = scoped
            mechanisms[device] = _construct_mechanism(
                scoped,
                load_device=device,
                offload_device=offload_device,
                patch_set=scoped_patch,
                intermediate_dtype=intermediate_dtype,
                patch_weight_dtype=patch_weight_dtype,
                patch_key_prefix=patch_key_prefix,
                mechanism_factory=mechanism_factory,
            )
    except BaseException as error:
        for mechanism in reversed(tuple(mechanisms.values())):
            try:
                mechanism.unload()
            except BaseException as cleanup:
                error.add_note(f"residency cleanup also failed: {cleanup!r}")
        raise

    bound: list[tuple[torch.nn.Module, bool, ResidencyBinding | None]] = []
    try:
        for device, mechanism in mechanisms.items():
            bound.extend(_bind_mechanism(scoped_prepared[device], mechanism))
    except BaseException as error:
        _restore_bindings(bound)
        for mechanism in reversed(tuple(mechanisms.values())):
            try:
                mechanism.unload()
            except BaseException as cleanup:
                error.add_note(f"residency cleanup also failed: {cleanup!r}")
        raise

    placement = ComponentResidencyPlacement(mechanisms)
    prepared.module.__dict__["_dinkster_resident_weights"] = placement
    prepared.module.__dict__[_RESIDENCY_STATE_STORE] = prepared.store
    return placement


@overload
def enroll_assembled(
    assembled: AssembledFlux
    | AssembledLumina2
    | AssembledQwenImage
    | AssembledSD
    | AssembledWan21
    | AssembledZImage
    | AssembledMiniMaxH3Model,
    *,
    load_device: torch.device | str,
    offload_device: torch.device | str,
    patch_sets: Mapping[str, PatchSet[torch.Tensor]] | None = None,
    patch_weight_dtype: torch.dtype | None = None,
    patch_key_prefixes: Mapping[str, str] | None = None,
    storage_dtypes: Mapping[str, torch.dtype] | None = None,
    memory_policy: MemoryPolicy | None = None,
    mps_snapshot: Callable[[torch.device], MpsMemorySnapshot] | None = None,
) -> EnrolledAssembly[ResidentWeights]: ...


@overload
def enroll_assembled(
    assembled: AssembledFlux
    | AssembledLumina2
    | AssembledQwenImage
    | AssembledSD
    | AssembledWan21
    | AssembledZImage
    | AssembledMiniMaxH3Model,
    *,
    load_device: torch.device | str,
    offload_device: torch.device | str,
    patch_sets: Mapping[str, PatchSet[torch.Tensor]] | None = None,
    patch_weight_dtype: torch.dtype | None = None,
    patch_key_prefixes: Mapping[str, str] | None = None,
    storage_dtypes: Mapping[str, torch.dtype] | None = None,
    mechanism_factory: ResidencyMechanismFactory,
    memory_policy: MemoryPolicy | None = None,
    mps_snapshot: Callable[[torch.device], MpsMemorySnapshot] | None = None,
) -> EnrolledAssembly[EnrolledResidency]: ...


def enroll_assembled(
    assembled: AssembledFlux
    | AssembledLumina2
    | AssembledQwenImage
    | AssembledSD
    | AssembledWan21
    | AssembledZImage
    | AssembledMiniMaxH3Model,
    *,
    load_device: torch.device | str,
    offload_device: torch.device | str,
    patch_sets: Mapping[str, PatchSet[torch.Tensor]] | None = None,
    patch_weight_dtype: torch.dtype | None = None,
    patch_key_prefixes: Mapping[str, str] | None = None,
    storage_dtypes: Mapping[str, torch.dtype] | None = None,
    mechanism_factory: ResidencyMechanismFactory = ResidentWeights,
    memory_policy: MemoryPolicy | None = None,
    mps_snapshot: Callable[[torch.device], MpsMemorySnapshot] | None = None,
) -> EnrolledAssembly[ResidentWeights] | EnrolledAssembly[EnrolledResidency]:
    """Enroll every module and report storage-dtype policy outcomes.

    Policy conversion, mechanism construction, and binding form one
    transaction. Eligible state is converted only when the target does not
    widen checkpoint storage; wider compute dtypes retain cast-at-use state.
    Patched converted state is applied to an owned weight-dtype value and cast once
    into authoritative target storage before mechanisms are built.

    MPS load devices also rewrite fp8 storage before state is folded, since
    torch has no fp8 tensors on MPS: scaled-fp8 Linear layers are swapped for
    plain routed Linears holding the one-time dequant product, and plain fp8
    cast-at-use state is cast once to its bound compute dtype. Both rewrites
    reproduce the cast-at-use op order exactly, so unpatched forward outputs
    are unchanged (patches apply to the dequantized weight rather than
    requantizing); the affected state stores at compute dtype instead of fp8.
    Hardware-matmul layers use the same rewrite with a diagnostic. Residency
    owners may supply their active memory policy and MPS snapshot provider
    so upcast diagnostics use the same inputs as later placement decisions.
    Historical parity evidence is logged for context and never gates enrollment.
    """
    is_mps = torch.device(load_device).type == "mps"
    patch_sets = {} if patch_sets is None else patch_sets
    patch_key_prefixes = {} if patch_key_prefixes is None else patch_key_prefixes
    storage_dtypes = {} if storage_dtypes is None else storage_dtypes
    resolved_patch_weight_dtype = (
        torch.float32 if patch_weight_dtype is None else patch_weight_dtype
    )
    declared_components = getattr(assembled, "components", None)
    if declared_components is None:
        components = {
            field.name: value
            for field in fields(assembled)
            if isinstance((value := getattr(assembled, field.name)), torch.nn.Module)
        }
    else:
        if not isinstance(declared_components, Mapping):
            raise TypeError("assembled components must be a mapping")
        components: dict[str, torch.nn.Module] = {}
        for name, module in declared_components.items():
            if not isinstance(name, str) or not name:
                raise TypeError("assembled component names must be non-empty strings")
            if not isinstance(module, torch.nn.Module):
                raise TypeError(f"assembled component {name!r} must be a torch module")
            components[name] = module
    if len({id(module) for module in components.values()}) != len(components):
        raise ValueError("assembled components must not share a module instance")
    unknown = set(patch_sets) - components.keys()
    if unknown:
        raise ValueError(
            "patch sets name unknown assembled components: " + ", ".join(sorted(unknown))
        )
    unknown_prefixes = set(patch_key_prefixes) - components.keys()
    if unknown_prefixes:
        raise ValueError(
            "patch key prefixes name unknown assembled components: "
            + ", ".join(sorted(unknown_prefixes))
        )
    if is_mps:
        for component_module in components.values():
            _require_not_enrolled(component_module)
        family_id = getattr(getattr(assembled, "family", None), "id", None)
        upcast_plans = [
            _plan_fp8_upcast_for_mps(
                component_module,
                component=name,
                family=family_id if isinstance(family_id, str) else None,
            )
            for name, component_module in components.items()
        ]
        _admit_fp8_upcast_for_mps(
            upcast_plans,
            torch.device(load_device),
            memory_policy=MemoryPolicy() if memory_policy is None else memory_policy,
            mps_snapshot=mps_memory_snapshot if mps_snapshot is None else mps_snapshot,
        )
        _commit_fp8_upcasts(upcast_plans)
    follows_compute = assembled._storage_dtype_follows_compute  # pyright: ignore[reportPrivateUsage]
    unknown_storage_dtypes = set(storage_dtypes) - components.keys()
    if unknown_storage_dtypes:
        raise ValueError(
            "storage dtypes name unknown assembled components: "
            + ", ".join(sorted(unknown_storage_dtypes))
        )
    enabled = follows_compute or bool(storage_dtypes)
    prepared = {name: _prepare_component(module) for name, module in components.items()}
    for name, component in prepared.items():
        _preflight_quantization_patches(component, patch_sets.get(name))
        if enabled:
            _preflight_patch_targets(component, patch_sets.get(name), component=name)
    compute_dtype = getattr(assembled, "compute_dtype", None)
    compute_dtypes = getattr(assembled, "_component_compute_dtypes", {})
    targets: dict[str, Mapping[str, torch.dtype]] = {}
    outcomes: dict[str, StorageDtypeOutcome] = {}
    for name, component in prepared.items():
        if not enabled:
            _validate_routes(component)
            continue
        target = storage_dtypes.get(name)
        if target is None and follows_compute:
            target = compute_dtype(name) if callable(compute_dtype) else compute_dtypes.get(name)
        if target is None:
            _validate_routes(component)
            continue
        if not isinstance(target, torch.dtype):
            raise TypeError(f"assembled component {name!r} storage dtype must be a torch dtype")
        component_targets = _storage_dtype_targets(
            component,
            target,
            respect_bound_compute_dtype=name not in storage_dtypes,
        )
        targets[name] = component_targets
        outcome = _storage_dtype_outcome(component.store, component_targets)
        outcomes[name] = outcome
        _validate_routes(
            component,
            policy_component=name if outcome == "converted" else None,
        )
        if outcome == "converted" and (unmanaged := _unmanaged_buffers(component.module)):
            raise StorageDtypePolicyError(
                name,
                "unmanaged_buffer",
                ", ".join(repr(buffer) for buffer in unmanaged),
            )

    conversions = {
        name: _prepare_storage_conversion(
            prepared[name],
            component=name,
            targets=targets[name],
            offload_device=offload_device,
            patch_set=patch_sets.get(name),
            patch_weight_dtype=resolved_patch_weight_dtype,
            patch_key_prefix=patch_key_prefixes.get(name, ""),
        )
        for name, outcome in outcomes.items()
        if outcome == "converted"
    }
    replay_patch_sets = {name: _snapshot_patch_set(patch_sets.get(name)) for name in conversions}
    snapshots = {name: conversion.snapshots for name, conversion in conversions.items()}
    enrolled: dict[str, EnrolledResidency] = {}
    bound: list[tuple[torch.nn.Module, bool, ResidencyBinding | None]] = []
    try:
        for name, conversion in conversions.items():
            for key, replacement in conversion.replacements.items():
                prepared[name].writer[key] = replacement

        for name, component in prepared.items():
            enrolled[name] = _construct_mechanism(
                component,
                load_device=load_device,
                offload_device=offload_device,
                patch_set=(None if name in conversions else patch_sets.get(name)),
                intermediate_dtype=torch.float32,
                patch_weight_dtype=patch_weight_dtype,
                patch_key_prefix=patch_key_prefixes.get(name, ""),
                mechanism_factory=mechanism_factory,
            )

        for name, component in prepared.items():
            bound.extend(_bind_mechanism(component, enrolled[name]))
        for name, component in prepared.items():
            component.module.__dict__["_dinkster_resident_weights"] = enrolled[name]
            component.module.__dict__[_RESIDENCY_STATE_STORE] = component.store
    except BaseException as error:
        for component in prepared.values():
            component.module.__dict__.pop("_dinkster_resident_weights", None)
            component.module.__dict__.pop(_RESIDENCY_STATE_STORE, None)
        _restore_bindings(bound)
        for mechanism in reversed(tuple(enrolled.values())):
            try:
                mechanism.unload()
            except BaseException as cleanup_error:
                error.add_note(f"residency enrollment cleanup also failed: {cleanup_error!r}")
        for name, component_snapshots in snapshots.items():
            _restore_store(prepared[name].writer, component_snapshots)
        raise
    for name, module in components.items():
        module.__dict__["_dinkster_storage_converted"] = outcomes.get(name) == "converted"
        module.__dict__["_dinkster_base_patch_set"] = (
            replay_patch_sets[name] if name in conversions else None
        )
    return EnrolledAssembly(
        enrolled,
        storage_dtype_report=StorageDtypeReport(enabled, outcomes),
    )
