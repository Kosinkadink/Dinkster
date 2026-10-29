"""Eager-only per-condition scaled-patch execution.

The context temporarily routes owned linear and two-dimensional convolution
targets through ordinary materialized patch weights selected per conditioning
batch row. It does not decode conditioning carriers, bind lanes, or participate
in ordinary sampling. Compiled execution is deliberately refused.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import math
import threading
import weakref
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from types import MethodType, TracebackType
from typing import Any, Literal, NoReturn, cast

import torch
import torch.nn.functional as F
from dinkster_inference import AdapterPatch, ComponentPlan, DiffPatch, PatchEntry, PatchSet

from .adapters import LoRAAdapter
from .apply import apply_patches, patch_stored_weight
from .assemble import _apply_transform
from .operations import (
    ResidencyRouted,
    _CastConv2d,
    _CastLinear,
    _InitlessConv2d,
    _InitlessLinear,
    bound_compute_device,
)
from .sources import load_tensors


class ScaledPatchError(ValueError):
    """A deterministic scaled-patch refusal with a stable machine code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"scaled-patch:{code}")


@dataclass(frozen=True)
class _TensorIdentity:
    value: torch.Tensor
    version: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    layout: torch.layout
    device: torch.device
    quantized: bool


@dataclass(frozen=True)
class _EntrySnapshot:
    entry: PatchEntry[torch.Tensor]
    value: object
    kind: Literal["diff", "lora"]
    strength: float
    payloads: tuple[_TensorIdentity, ...]
    alpha: float | None = None
    intermediate_dtype: torch.dtype | None = None


@dataclass(frozen=True)
class _ConvGeometry:
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...]
    padding: str | tuple[int, ...]
    dilation: tuple[int, ...]
    groups: int
    padding_mode: str


@dataclass(frozen=True)
class _TargetSnapshot:
    key: str
    owner_name: str
    owner: torch.nn.Linear | torch.nn.Conv2d
    weight: _TensorIdentity
    entries: tuple[_EntrySnapshot, ...]
    conv_geometry: _ConvGeometry | None
    source_plan: ComponentPlan[Any] | None
    storage_converted: bool
    base_entries: tuple[PatchEntry[torch.Tensor], ...]


@dataclass
class _StagedEntry:
    kind: Literal["diff", "lora"]
    strength: float
    first: torch.Tensor
    second: torch.Tensor | None
    alpha_scale: float


@dataclass
class _ExecutionTarget:
    snapshot: _TargetSnapshot
    entries: list[_StagedEntry]
    scale: torch.Tensor
    device: torch.device
    dtype: torch.dtype
    cached_scale: float | None = None
    cached_weight: torch.Tensor | None = None
    source_weight: torch.Tensor | None = None


@dataclass
class _ForwardOverride:
    owner: torch.nn.Linear | torch.nn.Conv2d
    previous: object
    inherited: bool


_ACTIVATION_LOCK = threading.Lock()
_ACTIVE_MODELS: weakref.WeakKeyDictionary[torch.nn.Module, object] = weakref.WeakKeyDictionary()
_TARGET_DTYPES = frozenset((torch.float16, torch.bfloat16, torch.float32, torch.float64))


def _refuse(code: str) -> NoReturn:
    raise ScaledPatchError(code)


def _activation_modules(
    model: torch.nn.Module, targets: tuple[_TargetSnapshot, ...]
) -> tuple[torch.nn.Module, ...]:
    return tuple(dict.fromkeys((model, *(target.owner for target in targets))))


def _reserve_activation(modules: tuple[torch.nn.Module, ...], token: object) -> None:
    if any(module in _ACTIVE_MODELS for module in modules):
        _refuse("active-model")
    for module in modules:
        _ACTIVE_MODELS[module] = token


def _release_activation(modules: tuple[torch.nn.Module, ...], token: object) -> None:
    for module in modules:
        if _ACTIVE_MODELS.get(module) is token:
            del _ACTIVE_MODELS[module]


def _is_plain_float_tensor(value: object, *, rank: int) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and value.ndim == rank
        and value.layout is torch.strided
        and not value.is_quantized
        and torch.is_floating_point(value)
    )


def _is_conv_target(owner: torch.nn.Module) -> bool:
    return type(owner) in (torch.nn.Conv2d, _InitlessConv2d, _CastConv2d)


def _conv_geometry(owner: torch.nn.Conv2d) -> _ConvGeometry:
    return _ConvGeometry(
        owner.kernel_size,
        owner.stride,
        owner.padding,
        owner.dilation,
        owner.groups,
        owner.padding_mode,
    )


def _tensor_identity(value: torch.Tensor) -> _TensorIdentity:
    try:
        version = value._version  # pyright: ignore[reportPrivateUsage]
    except RuntimeError:
        _refuse("tensor-version")
    return _TensorIdentity(
        value=value,
        version=version,
        shape=tuple(value.shape),
        dtype=value.dtype,
        layout=value.layout,
        device=value.device,
        quantized=value.is_quantized,
    )


def _require_finite(value: torch.Tensor, code: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        _refuse(code)


def _unchanged(identity: _TensorIdentity) -> bool:
    value = identity.value
    try:
        version = value._version  # pyright: ignore[reportPrivateUsage]
    except RuntimeError:
        return False
    return (
        version == identity.version
        and tuple(value.shape) == identity.shape
        and value.dtype is identity.dtype
        and value.layout is identity.layout
        and value.device == identity.device
        and value.is_quantized is identity.quantized
    )


def _cancelled(cancel: Callable[[], bool]) -> bool:
    result = cancel()
    if type(result) is not bool:
        _refuse("cancel-result")
    return result


def _check_cancel(cancel: Callable[[], bool]) -> None:
    if _cancelled(cancel):
        _refuse("cancelled")


def _resolve_targets(
    model: torch.nn.Module,
    patch_set: PatchSet[torch.Tensor],
) -> tuple[str, tuple[_TargetSnapshot, ...]]:
    revision = patch_set.revision
    keys = tuple(patch_set.keys())
    if not keys:
        _refuse("empty-patch-set")

    named = tuple(model.named_modules(remove_duplicate=False))
    names_by_module: dict[int, list[str]] = {}
    for name, module in named:
        names_by_module.setdefault(id(module), []).append(name)
    model_state = model.state_dict(keep_vars=True)
    raw_plan = getattr(model, "_dinkster_component_plan", None)
    source_plan = raw_plan if isinstance(raw_plan, ComponentPlan) else None
    storage_converted = getattr(model, "_dinkster_storage_converted", False) is True
    raw_base_patch_set = getattr(model, "_dinkster_base_patch_set", None)
    base_patch_set = raw_base_patch_set if isinstance(raw_base_patch_set, PatchSet) else None
    if storage_converted and source_plan is None:
        _refuse("source-authority")

    targets: list[_TargetSnapshot] = []
    for key in keys:
        if key == "weight":
            owner_name = ""
        elif key.endswith(".weight") and key != ".weight":
            owner_name = key[: -len(".weight")]
        else:
            _refuse("target-name")

        matches = [(name, module) for name, module in named if name == owner_name]
        if len(matches) != 1:
            _refuse("target-resolution")
        owner = matches[0][1]
        if len(names_by_module[id(owner)]) != 1:
            _refuse("shared-module")
        if type(owner) not in (
            torch.nn.Linear,
            _InitlessLinear,
            _CastLinear,
            torch.nn.Conv2d,
            _InitlessConv2d,
            _CastConv2d,
        ):
            _refuse("target-type")
        if not isinstance(owner, (torch.nn.Linear, torch.nn.Conv2d)):
            _refuse("target-type")
        target = owner
        if "forward" in target.__dict__:
            _refuse("target-forward")
        is_conv = _is_conv_target(target)
        if type(target) is _CastLinear and target.fp8_matmul:
            _refuse("fp8-matmul")
        if is_conv:
            conv = cast(torch.nn.Conv2d, target)
            if conv.groups != 1:
                _refuse("conv-groups")
            if conv.padding_mode != "zeros":
                _refuse("conv-padding-mode")
            conv_geometry = _conv_geometry(conv)
        else:
            conv_geometry = None

        weight = target.weight
        if (
            type(weight) is not torch.nn.Parameter
            or not _is_plain_float_tensor(weight, rank=4 if is_conv else 2)
            or weight.dtype not in _TARGET_DTYPES
            or weight.device.type == "meta"
        ):
            _refuse("target-weight")
        if target._parameters.get("weight") is not weight:  # pyright: ignore[reportPrivateUsage]
            _refuse("target-ownership")
        direct_state = target.state_dict(keep_vars=True)
        if direct_state.get("weight") is not weight:
            _refuse("target-ownership")
        target_storage = weight.untyped_storage()._cdata  # pyright: ignore[reportPrivateUsage]
        aliases = [
            name
            for name, value in model_state.items()
            if value is weight
            or (
                value.device == weight.device
                and value.layout is torch.strided
                and value.untyped_storage()._cdata  # pyright: ignore[reportPrivateUsage]
                == target_storage
            )
        ]
        if aliases != [key]:
            _refuse("shared-weight")
        if getattr(target, "_forward_hooks", None) or getattr(target, "_forward_pre_hooks", None):
            _refuse("target-hooked")

        entries = patch_set.entries(key)
        if not entries:
            _refuse("empty-target")
        snapshots: list[_EntrySnapshot] = []
        for entry in entries:
            if type(entry.strength) is not float or not math.isfinite(entry.strength):
                _refuse("strength")
            if type(entry.strength_model) is not float or entry.strength_model != 1.0:
                _refuse("strength-model")
            if entry.offset is not None:
                _refuse("offset")
            if entry.function is not None:
                _refuse("function")

            value = entry.value
            if isinstance(value, DiffPatch) and type(value) is DiffPatch:
                if is_conv:
                    _refuse("diff-target")
                if value.pad_weight is not False:
                    _refuse("diff-pad")
                diff = value.value
                if not _is_plain_float_tensor(diff, rank=2):
                    _refuse("diff-tensor")
                if tuple(diff.shape) != tuple(weight.shape):
                    _refuse("diff-shape")
                _require_finite(diff, "diff-finite")
                snapshots.append(
                    _EntrySnapshot(
                        entry=entry,
                        value=value,
                        kind="diff",
                        strength=entry.strength,
                        payloads=(_tensor_identity(diff),),
                    )
                )
                continue

            if not isinstance(value, AdapterPatch) or type(value) is not AdapterPatch:
                _refuse("patch-kind")
            if type(value.adapter) is not LoRAAdapter:
                _refuse("patch-kind")
            adapter = cast(LoRAAdapter, value.adapter)
            if adapter.mid is not None:
                _refuse("lora-mid")
            if adapter.dora_scale is not None:
                _refuse("lora-dora")
            if adapter.reshape is not None:
                _refuse("lora-reshape")
            up = adapter.up
            down = adapter.down
            allowed_ranks = (2, 4) if is_conv else (2,)
            if not any(_is_plain_float_tensor(up, rank=rank) for rank in allowed_ranks) or not any(
                _is_plain_float_tensor(down, rank=rank) for rank in allowed_ranks
            ):
                _refuse("lora-tensor")
            rank = down.shape[0]
            if rank <= 0:
                _refuse("lora-rank")
            if is_conv:
                conv = cast(torch.nn.Conv2d, target)
                expected_up = (
                    (weight.shape[0], rank) if up.ndim == 2 else (weight.shape[0], rank, 1, 1)
                )
                expected_down = (rank, weight.shape[1], *conv.kernel_size)
                if (
                    tuple(up.shape) != expected_up
                    or down.ndim != 4
                    or tuple(down.shape) != expected_down
                ):
                    _refuse("lora-shape")
            else:
                expected_down = (rank, weight.shape[1])
                expected_up = (weight.shape[0], rank)
                if tuple(down.shape) != expected_down or tuple(up.shape) != expected_up:
                    _refuse("lora-shape")
            _require_finite(up, "lora-finite")
            _require_finite(down, "lora-finite")
            alpha = adapter.alpha
            if alpha is not None and (type(alpha) is not float or not math.isfinite(alpha)):
                _refuse("lora-alpha")
            snapshots.append(
                _EntrySnapshot(
                    entry=entry,
                    value=value,
                    kind="lora",
                    strength=entry.strength,
                    payloads=(_tensor_identity(up), _tensor_identity(down)),
                    alpha=alpha,
                    intermediate_dtype=adapter.intermediate_dtype,
                )
            )
        targets.append(
            _TargetSnapshot(
                key=key,
                owner_name=owner_name,
                owner=target,
                weight=_tensor_identity(weight),
                entries=tuple(snapshots),
                conv_geometry=conv_geometry,
                source_plan=source_plan,
                storage_converted=storage_converted,
                base_entries=() if base_patch_set is None else base_patch_set.entries(key),
            )
        )
    return revision, tuple(targets)


def _validate_snapshot(
    patch_set: PatchSet[torch.Tensor],
    revision: str,
    targets: tuple[_TargetSnapshot, ...],
) -> None:
    if patch_set.revision != revision or tuple(patch_set.keys()) != tuple(
        target.key for target in targets
    ):
        _refuse("patch-set-drift")
    for target in targets:
        if target.owner.weight is not target.weight.value or not _unchanged(target.weight):
            _refuse("target-drift")
        current_entries = patch_set.entries(target.key)
        if len(current_entries) != len(target.entries):
            _refuse("patch-set-drift")
        for current, snapshot in zip(current_entries, target.entries, strict=True):
            if current is not snapshot.entry or current.value is not snapshot.value:
                _refuse("patch-set-drift")
            if (
                current.strength != snapshot.strength
                or current.strength_model != 1.0
                or current.offset is not None
                or current.function is not None
            ):
                _refuse("patch-set-drift")
            if snapshot.kind == "diff":
                value = current.value
                if (
                    not isinstance(value, DiffPatch)
                    or type(value) is not DiffPatch
                    or value.value is not snapshot.payloads[0].value
                ):
                    _refuse("patch-set-drift")
            else:
                value = current.value
                if not isinstance(value, AdapterPatch) or type(value) is not AdapterPatch:
                    _refuse("patch-set-drift")
                if type(value.adapter) is not LoRAAdapter:
                    _refuse("patch-set-drift")
                adapter = cast(LoRAAdapter, value.adapter)
                if (
                    adapter.up is not snapshot.payloads[0].value
                    or adapter.down is not snapshot.payloads[1].value
                    or adapter.alpha != snapshot.alpha
                    or adapter.intermediate_dtype is not snapshot.intermediate_dtype
                    or adapter.mid is not None
                    or adapter.dora_scale is not None
                    or adapter.reshape is not None
                ):
                    _refuse("patch-set-drift")
            if not all(_unchanged(payload) for payload in snapshot.payloads):
                _refuse("payload-drift")


def _validate_activation_targets(
    model: torch.nn.Module,
    targets: tuple[_TargetSnapshot, ...],
) -> None:
    named = tuple(model.named_modules(remove_duplicate=False))
    model_state = model.state_dict(keep_vars=True)
    for target in targets:
        owner = target.owner
        if type(owner) not in (
            torch.nn.Linear,
            _InitlessLinear,
            _CastLinear,
            torch.nn.Conv2d,
            _InitlessConv2d,
            _CastConv2d,
        ):
            _refuse("target-type")
        if type(owner) is _CastLinear and owner.fp8_matmul:
            _refuse("fp8-matmul")
        if isinstance(owner, torch.nn.Conv2d):
            if target.conv_geometry != _conv_geometry(owner):
                _refuse("conv-geometry-drift")
        elif target.conv_geometry is not None:
            _refuse("target-type")
        owner_matches = [(name, module) for name, module in named if name == target.owner_name]
        if owner_matches != [(target.owner_name, owner)]:
            _refuse("target-resolution")
        if sum(module is owner for _name, module in named) != 1:
            _refuse("shared-module")
        if model_state.get(target.key) is not target.weight.value:
            _refuse("target-ownership")
        if owner._parameters.get("weight") is not target.weight.value:  # pyright: ignore[reportPrivateUsage]
            _refuse("target-ownership")
        if owner.state_dict(keep_vars=True).get("weight") is not target.weight.value:
            _refuse("target-ownership")
        weight = target.weight.value
        target_storage = weight.untyped_storage()._cdata  # pyright: ignore[reportPrivateUsage]
        aliases = [
            name
            for name, value in model_state.items()
            if value is weight
            or (
                value.device == weight.device
                and value.layout is torch.strided
                and value.untyped_storage()._cdata  # pyright: ignore[reportPrivateUsage]
                == target_storage
            )
        ]
        if aliases != [target.key]:
            _refuse("shared-weight")
        if getattr(owner, "_forward_hooks", None) or getattr(owner, "_forward_pre_hooks", None):
            _refuse("target-hooked")


def _target_compute_placement(
    target: _TargetSnapshot,
) -> tuple[torch.device, torch.dtype]:
    owner = target.owner
    if type(owner) is _CastLinear:
        device = bound_compute_device(owner)
        return (
            owner.weight.device if device is None else device,
            owner._compute_dtype,
        )
    if type(owner) is _CastConv2d:
        device = bound_compute_device(owner)
        return (
            owner.weight.device if device is None else device,
            owner._compute_dtype,
        )
    device = bound_compute_device(owner)
    return (owner.weight.device if device is None else device, owner.weight.dtype)


def _stage_targets(
    targets: tuple[_TargetSnapshot, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[_ExecutionTarget]:
    staged_targets: list[_ExecutionTarget] = []
    for target in targets:
        staged_entries: list[_StagedEntry] = []
        for entry in target.entries:
            first = entry.payloads[0].value.detach().to(device=device, dtype=dtype).clone()
            _require_finite(first, "staged-payload-finite")
            second = None
            alpha_scale = 1.0
            if entry.kind == "lora":
                second = entry.payloads[1].value.detach().to(device=device, dtype=dtype).clone()
                _require_finite(second, "staged-payload-finite")
                rank = second.shape[0]
                alpha_scale = 1.0 if entry.alpha is None else entry.alpha / rank
            staged_entries.append(
                _StagedEntry(entry.kind, entry.strength, first, second, alpha_scale)
            )
        staged_targets.append(
            _ExecutionTarget(
                target,
                staged_entries,
                torch.empty(0, device=device, dtype=dtype),
                device,
                dtype,
            )
        )
    return staged_targets


def _staged_nbytes(targets: list[_ExecutionTarget]) -> int:
    return sum(
        entry.first.numel() * entry.first.element_size()
        + (0 if entry.second is None else entry.second.numel() * entry.second.element_size())
        for target in targets
        for entry in target.entries
    )


def _source_weight(execution: _ExecutionTarget) -> torch.Tensor:
    if execution.source_weight is not None:
        return execution.source_weight
    plan = execution.snapshot.source_plan
    if plan is None:
        _refuse("source-authority")
    key = execution.snapshot.key
    source_key = plan.keys.get(key)
    if source_key is None:
        _refuse("source-key")
    source = load_tensors(plan.path, (source_key,))[source_key]
    transform = plan.transforms.get(key)
    if transform is not None:
        source = _apply_transform(plan.component, key, source, transform)
    if (
        not _is_plain_float_tensor(source, rank=execution.snapshot.weight.value.ndim)
        or tuple(source.shape) != execution.snapshot.weight.shape
    ):
        _refuse("source-weight")
    execution.source_weight = source.detach()
    return execution.source_weight


def _materialized_weight(execution: _ExecutionTarget, scale: float) -> torch.Tensor:
    if execution.cached_scale == scale and execution.cached_weight is not None:
        return execution.cached_weight
    owner = execution.snapshot.owner
    source = execution.snapshot.weight.value.detach()
    scheduled_entries = tuple(
        replace(snapshot.entry, strength=snapshot.strength * scale)
        for snapshot in execution.snapshot.entries
    )
    entries = (*execution.snapshot.base_entries, *scheduled_entries)
    if execution.snapshot.storage_converted:
        original = _source_weight(execution)
        patched = apply_patches(
            original.to(device="cpu", dtype=torch.float32, copy=True),
            entries,
            key=execution.snapshot.key,
            intermediate_dtype=torch.float32,
            original_weight=original,
        ).to(dtype=execution.snapshot.weight.dtype)
        execution.cached_weight = patched.to(device=execution.device, dtype=execution.dtype)
        execution.cached_scale = scale
        return execution.cached_weight
    binding = owner._offloaded_residency() if isinstance(owner, ResidencyRouted) else None
    if binding is None:
        patched = patch_stored_weight(source, entries, key=execution.snapshot.key)
        execution.cached_weight = patched.to(device=execution.device, dtype=execution.dtype)
    else:
        compute_weight = source.to(device=execution.device, dtype=execution.dtype, copy=True)
        execution.cached_weight = apply_patches(
            compute_weight,
            entries,
            key=binding.key("weight"),
            intermediate_dtype=execution.dtype,
        )
    execution.cached_scale = scale
    return execution.cached_weight


@contextmanager
def _execution_bias(execution: _ExecutionTarget) -> Generator[torch.Tensor | None, None, None]:
    owner = execution.snapshot.owner
    stored = getattr(owner, "bias", None)
    if stored is not None and not isinstance(stored, torch.Tensor):
        _refuse("target-bias")
    binding = owner._offloaded_residency() if isinstance(owner, ResidencyRouted) else None
    if binding is not None:
        with binding.lease() as lease:
            yield None if stored is None else lease.get("bias", dtype=execution.dtype)
        return
    yield None if stored is None else stored.to(device=execution.device, dtype=execution.dtype)


def _execute_weight(
    execution: _ExecutionTarget,
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    geometry = execution.snapshot.conv_geometry
    if geometry is not None:
        return F.conv2d(
            value,
            weight,
            bias,
            geometry.stride,
            geometry.padding,
            geometry.dilation,
            geometry.groups,
        )
    return F.linear(value, weight, bias)


def _validate_io(
    execution: _ExecutionTarget,
    value: torch.Tensor,
    output: object,
) -> torch.Tensor:
    if not _is_plain_float_tensor(value, rank=value.ndim) or value.ndim < 2:
        _refuse("input-contract")
    if not isinstance(output, torch.Tensor) or output.ndim < 2:
        _refuse("output-contract")
    if (
        output.layout is not torch.strided
        or output.is_quantized
        or not torch.is_floating_point(output)
    ):
        _refuse("output-contract")
    if (
        value.device != output.device
        or value.dtype is not output.dtype
        or output.device != execution.device
        or output.dtype is not execution.dtype
    ):
        _refuse("io-device-dtype")
    weight = execution.snapshot.weight.value
    geometry = execution.snapshot.conv_geometry
    if geometry is not None:
        owner = execution.snapshot.owner
        if not isinstance(owner, torch.nn.Conv2d) or geometry != _conv_geometry(owner):
            _refuse("conv-geometry-drift")
        if value.ndim != 4 or value.shape[1] != weight.shape[1] * geometry.groups:
            _refuse("input-shape")
        if output.ndim != 4 or output.shape[:2] != (value.shape[0], weight.shape[0]):
            _refuse("output-shape")
    else:
        if value.shape[-1] != weight.shape[1]:
            _refuse("input-shape")
        if output.shape[-1] != weight.shape[0] or output.shape[:-1] != value.shape[:-1]:
            _refuse("output-shape")
    return output


def _make_forward(
    execution: _ExecutionTarget,
    original_forward: Callable[..., object],
    patch_set: PatchSet[torch.Tensor],
    revision: str,
    targets: tuple[_TargetSnapshot, ...],
    scale_identity: _TensorIdentity,
    cancel: Callable[[], bool],
) -> Callable[..., torch.Tensor]:
    def forward(_module: torch.nn.Module, *args: object, **kwargs: object) -> torch.Tensor:
        if torch.compiler.is_compiling():
            _refuse("compiled")
        _check_cancel(cancel)
        _validate_snapshot(patch_set, revision, targets)
        if not _unchanged(scale_identity):
            _refuse("scale-drift")
        if kwargs or len(args) != 1 or not isinstance(args[0], torch.Tensor):
            _refuse("input-contract")
        value = args[0]
        scales = tuple(float(item) for item in execution.scale.tolist())
        batch = value.shape[0]
        if len(scales) not in (1, batch):
            _refuse("scale-batch")
        if len(scales) == 1 and scales[0] == 0.0:
            return _validate_io(execution, value, original_forward(value))
        with _execution_bias(execution) as bias:
            if len(scales) == 1 or all(scale == scales[0] for scale in scales):
                output = _execute_weight(
                    execution,
                    value,
                    _materialized_weight(execution, scales[0]),
                    bias,
                )
            else:
                output = torch.empty((0,), device=execution.device, dtype=execution.dtype)
                for scale in dict.fromkeys(scales):
                    _check_cancel(cancel)
                    indices = torch.tensor(
                        [index for index, item in enumerate(scales) if item == scale],
                        device=execution.device,
                    )
                    selected = value.index_select(0, indices)
                    selected_output = _execute_weight(
                        execution,
                        selected,
                        _materialized_weight(execution, scale),
                        bias,
                    )
                    if output.numel() == 0:
                        output = torch.empty(
                            (batch, *selected_output.shape[1:]),
                            device=execution.device,
                            dtype=execution.dtype,
                        )
                    output = output.index_copy(0, indices, selected_output)
        _check_cancel(cancel)
        return _validate_io(execution, value, output)

    return forward


def _make_hook(
    execution: _ExecutionTarget,
    patch_set: PatchSet[torch.Tensor],
    revision: str,
    targets: tuple[_TargetSnapshot, ...],
    scale_identity: _TensorIdentity,
    cancel: Callable[[], bool],
) -> Callable[[torch.nn.Module, tuple[object, ...], dict[str, object], object], torch.Tensor]:
    def hook(
        module: torch.nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        output: object,
    ) -> torch.Tensor:
        _check_cancel(cancel)
        _validate_snapshot(patch_set, revision, targets)
        if not _unchanged(scale_identity):
            _refuse("scale-drift")
        if module is not execution.snapshot.owner:
            _refuse("target-drift")
        _check_cancel(cancel)
        if kwargs or len(args) != 1 or not isinstance(args[0], torch.Tensor):
            _refuse("input-contract")
        return _validate_io(execution, args[0], output)

    return hook


def _install_forward(
    execution: _ExecutionTarget,
    patch_set: PatchSet[torch.Tensor],
    revision: str,
    targets: tuple[_TargetSnapshot, ...],
    scale_identity: _TensorIdentity,
    cancel: Callable[[], bool],
) -> _ForwardOverride:
    owner = execution.snapshot.owner
    inherited = "forward" not in owner.__dict__
    previous = owner.forward if inherited else owner.__dict__["forward"]
    original_forward = owner.forward
    owner.forward = MethodType(  # pyright: ignore[reportAttributeAccessIssue]
        _make_forward(
            execution,
            original_forward,
            patch_set,
            revision,
            targets,
            scale_identity,
            cancel,
        ),
        owner,
    )
    return _ForwardOverride(owner, previous, inherited)


def _restore_forward(override: _ForwardOverride) -> None:
    if override.inherited:
        del override.owner.forward
    else:
        override.owner.forward = override.previous  # type: ignore[method-assign]


class PreparedScaledPatches(AbstractContextManager["PreparedScaledPatches"]):
    """Execution-owned patch state for changing conditioning subgroups."""

    def __init__(
        self,
        model: torch.nn.Module,
        patch_set: PatchSet[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        cancel: Callable[[], bool],
        *,
        _resolved: tuple[str, tuple[_TargetSnapshot, ...]] | None = None,
    ) -> None:
        if not callable(cancel):
            _refuse("cancel")
        if device.type == "meta":
            _refuse("execution-device")
        self.model = model
        self.patch_set = patch_set
        self.device = device
        self.dtype = dtype
        self.cancel = cancel
        self._closed = False
        self._poisoned = False
        self._active = False
        with _ACTIVATION_LOCK:
            if model in _ACTIVE_MODELS:
                _refuse("active-model")
        try:
            if _resolved is None:
                self._revision, self._targets = _resolve_targets(model, patch_set)
            else:
                self._revision, self._targets = _resolved
            if dtype not in _TARGET_DTYPES:
                _refuse("execution-dtype")
            if any(
                _target_compute_placement(target) != (device, dtype) for target in self._targets
            ):
                _refuse("execution-placement")
            _check_cancel(cancel)
            self._staged = _stage_targets(self._targets, device=device, dtype=dtype)
            self.staged_bytes = _staged_nbytes(self._staged)
            _validate_snapshot(patch_set, self._revision, self._targets)
            _check_cancel(cancel)
        except BaseException:
            raise

    def __enter__(self) -> PreparedScaledPatches:
        if self._closed:
            _refuse("prepared-closed")
        return self

    def validate_ready(self) -> None:
        """Revalidate target authority without installing hooks or guard state."""

        if self._closed:
            _refuse("prepared-closed")
        if self._poisoned:
            _refuse("prepared-poisoned")
        if self._active:
            _refuse("prepared-active")
        _validate_snapshot(self.patch_set, self._revision, self._targets)
        _validate_activation_targets(self.model, self._targets)
        if any(
            _target_compute_placement(target) != (self.device, self.dtype)
            for target in self._targets
        ):
            _refuse("execution-placement")

    @contextmanager
    def activate(self, scale_vector: torch.Tensor) -> Generator[None, None, None]:
        self.validate_ready()
        activation_modules = _activation_modules(self.model, self._targets)
        activation_token = object()
        if not _is_plain_float_tensor(scale_vector, rank=1) or scale_vector.numel() == 0:
            _refuse("scale-vector")
        _require_finite(scale_vector, "scale-finite")
        _check_cancel(self.cancel)
        identity = _tensor_identity(scale_vector)
        converted_scale = scale_vector.detach().to(device=self.device, dtype=self.dtype).clone()
        _require_finite(converted_scale, "staged-scale-finite")
        scale = scale_vector.detach().to(device="cpu", dtype=torch.float64).clone()
        handles: list[torch.utils.hooks.RemovableHandle] = []
        overrides: list[_ForwardOverride] = []
        with _ACTIVATION_LOCK:
            _reserve_activation(activation_modules, activation_token)
        self._active = True
        try:
            _validate_snapshot(self.patch_set, self._revision, self._targets)
            _validate_activation_targets(self.model, self._targets)
            if any(
                _target_compute_placement(target) != (self.device, self.dtype)
                for target in self._targets
            ):
                _refuse("execution-placement")
            if not _unchanged(identity):
                _refuse("scale-drift")
            _check_cancel(self.cancel)
            for target in self._staged:
                target.scale = scale
            for execution in self._staged:
                overrides.append(
                    _install_forward(
                        execution,
                        self.patch_set,
                        self._revision,
                        self._targets,
                        identity,
                        self.cancel,
                    )
                )
                handles.append(
                    execution.snapshot.owner.register_forward_hook(
                        _make_hook(
                            execution,
                            self.patch_set,
                            self._revision,
                            self._targets,
                            identity,
                            self.cancel,
                        ),
                        with_kwargs=True,
                    )
                )
            yield
        except BaseException:
            self._poisoned = True
            raise
        finally:
            for handle in reversed(handles):
                handle.remove()
            for override in reversed(overrides):
                _restore_forward(override)
            for target in self._staged:
                target.scale = torch.empty(0, dtype=torch.float64)
            self._active = False
            with _ACTIVATION_LOCK:
                _release_activation(activation_modules, activation_token)

    def close(self) -> None:
        if self._active:
            _refuse("prepared-active")
        if self._closed:
            return
        self._closed = True
        self._staged.clear()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def prepare_scaled_patches(
    model: torch.nn.Module,
    patch_set: PatchSet[torch.Tensor],
    device: torch.device | str,
    dtype: torch.dtype,
    cancel: Callable[[], bool],
) -> PreparedScaledPatches:
    """Preflight and stage one immutable patch set for an execution."""

    return PreparedScaledPatches(model, patch_set, torch.device(device), dtype, cancel)


def scaled_patch_context(
    model: torch.nn.Module,
    patch_set: PatchSet[torch.Tensor],
    scale_vector: torch.Tensor,
    cancel: Callable[[], bool],
) -> AbstractContextManager[None]:
    """Activate accepted per-row patches for eager forwards within one scope."""

    return _ScaledPatchContext(model, patch_set, scale_vector, cancel)


class _ScaledPatchContext(AbstractContextManager[None]):
    def __init__(
        self,
        model: torch.nn.Module,
        patch_set: PatchSet[torch.Tensor],
        scale_vector: torch.Tensor,
        cancel: Callable[[], bool],
    ) -> None:
        self._model: torch.nn.Module = model
        self._patch_set: PatchSet[torch.Tensor] = patch_set
        self._scale_vector: torch.Tensor = scale_vector
        self._cancel: Callable[[], bool] = cancel
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._overrides: list[_ForwardOverride] = []
        self._staged: list[_ExecutionTarget] = []
        self._activation_modules: tuple[torch.nn.Module, ...] = ()
        self._activation_token = object()
        self._registered = False
        self._entered = False
        self._finished = False

    def _cleanup(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        for override in reversed(self._overrides):
            _restore_forward(override)
        self._overrides.clear()
        self._staged.clear()
        if self._registered:
            with _ACTIVATION_LOCK:
                _release_activation(self._activation_modules, self._activation_token)
            self._registered = False

    def __enter__(self) -> None:
        if self._entered or self._finished:
            _refuse("context-reentry")
        self._entered = True
        model = self._model
        patch_set = self._patch_set
        scale_vector = self._scale_vector
        cancel = self._cancel
        if not callable(cancel):
            _refuse("cancel")
        with _ACTIVATION_LOCK:
            if model in _ACTIVE_MODELS:
                _refuse("active-model")
        try:
            revision, targets = _resolve_targets(model, patch_set)
            if not _is_plain_float_tensor(scale_vector, rank=1) or scale_vector.numel() == 0:
                _refuse("scale-vector")
            _require_finite(scale_vector, "scale-finite")
            identity = _tensor_identity(scale_vector)
            _check_cancel(cancel)
            for target in targets:
                device, dtype = _target_compute_placement(target)
                if device.type == "meta" or dtype not in _TARGET_DTYPES:
                    _refuse("execution-placement")
                self._staged.extend(_stage_targets((target,), device=device, dtype=dtype))
            _validate_snapshot(patch_set, revision, targets)
            if not _unchanged(identity):
                _refuse("scale-drift")
            _validate_activation_targets(model, targets)
            _check_cancel(cancel)
            for execution in self._staged:
                converted_scale = (
                    scale_vector.detach().to(device=execution.device, dtype=execution.dtype).clone()
                )
                _require_finite(converted_scale, "staged-scale-finite")
                execution.scale = (
                    scale_vector.detach().to(device="cpu", dtype=torch.float64).clone()
                )
            with _ACTIVATION_LOCK:
                _validate_snapshot(patch_set, revision, targets)
                _validate_activation_targets(model, targets)
                if not _unchanged(identity):
                    _refuse("scale-drift")
                self._activation_modules = _activation_modules(model, targets)
                _reserve_activation(self._activation_modules, self._activation_token)
                self._registered = True
            for execution in self._staged:
                self._overrides.append(
                    _install_forward(
                        execution,
                        patch_set,
                        revision,
                        targets,
                        identity,
                        cancel,
                    )
                )
                self._handles.append(
                    execution.snapshot.owner.register_forward_hook(
                        _make_hook(execution, patch_set, revision, targets, identity, cancel),
                        with_kwargs=True,
                    )
                )
            return None
        except BaseException:
            self._cleanup()
            self._finished = True
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if not self._entered or self._finished:
            _refuse("context-reentry")
        self._finished = True
        self._cleanup()
        return None


__all__ = [
    "PreparedScaledPatches",
    "ScaledPatchError",
    "prepare_scaled_patches",
    "scaled_patch_context",
]
