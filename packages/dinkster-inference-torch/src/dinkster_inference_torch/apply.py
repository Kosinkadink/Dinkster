"""PatchSet application: the typed port of comfy/lora.py
calculate_weight @ b78cec87, plus backup/restore over a weight store.

``apply_patches`` walks a key's PatchEntry tuple exactly like the
reference walks its patch-tuple list: narrow to the offset window,
scale by strength_model, dispatch on the value kind (applying the
per-entry ``function`` delta hook and the nested-base ``convert``
exactly where the reference does), restore the full weight after
windowed entries. ``patch_weights`` is the patch_weight_to_device loop
distilled: intermediate-dtype copy (dequantizing packed storage, the
convert_func step) -> apply -> write back to the storage form with the
per-key seed (stochastic rounding for plain tensors,
scale-recalculating requantization for packed storage, both seeded) ->
remember the original for exact restore.

Deliberate deviations, all LOUD (matching the slice-1 adapter
stance - a patch that cannot apply is a failed job, not a silently
unmodified model):

- Diff shape mismatch raises PatchApplyError; upstream logs
  "WARNING SHAPE MISMATCH ... WEIGHT NOT MERGED" and skips the entry.
- A ModelAsLoraPatch with no original weight raises; upstream would
  TypeError on ``original_weights[key]`` with its default None
  (patch_weight_to_device never passes original_weights - only the
  hook path does). For a packed store the original handed to
  model-as-lora entries is a fresh dequantized copy - an extension,
  since upstream's quantized path never passes original_weights either.
- ``patch_weights`` raises on a patch key absent from the store;
  upstream filters silently at add_patches time.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import replace
from typing import TypeVar, cast

import torch
from dinkster_inference.patches import (
    AdapterPatch,
    DiffPatch,
    NestedPatch,
    PatchEntry,
    PatchSet,
    SetPatch,
)

from .quant import (
    Fp8ScaledWeight,
    Int8PackedWeight,
    Nvfp4PackedWeight,
    requantize_fp8_scaled,
    requantize_int8,
    requantize_nvfp4,
)
from .rounding import stochastic_rounding, string_to_seed
from .tensor_ops import cast_to_device, pad_tensor_to_shape, transfer_to_device

StoredWeight = torch.Tensor | Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight
"""What a weight store holds per key: a plain tensor, fp8-scaled
storage, packed INT8 storage, or packed NVFP4 storage (quant.py)."""

W = TypeVar("W", bound=StoredWeight)
"""Store value type. ``patch_weights``/``restore_weights`` are generic
so a plain ``dict[str, Tensor]`` store keeps its value type through
the backup/restore round trip."""


class PatchApplyError(Exception):
    """A patch entry cannot be applied to its target weight."""


def move_stored(
    stored: StoredWeight,
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> StoredWeight:
    """Move a stored weight across devices, preserving its storage
    form. A no-op returns the input unchanged (torch ``.to`` identity
    semantics) - callers must not mutate the result."""

    def move(tensor: torch.Tensor) -> torch.Tensor:
        return transfer_to_device(tensor, device, non_blocking=non_blocking)

    if isinstance(stored, Fp8ScaledWeight):
        qdata = move(stored.qdata)
        scale = move(stored.scale)
        if qdata is stored.qdata and scale is stored.scale:
            return stored
        return replace(stored, qdata=qdata, scale=scale)
    if isinstance(stored, Int8PackedWeight):
        qdata = move(stored.qdata)
        scale = move(stored.scale)
        if qdata is stored.qdata and scale is stored.scale:
            return stored
        return replace(stored, qdata=qdata, scale=scale)
    if isinstance(stored, Nvfp4PackedWeight):
        qdata = move(stored.qdata)
        block = move(stored.block_scale)
        scale = move(stored.tensor_scale)
        if qdata is stored.qdata and block is stored.block_scale and scale is stored.tensor_scale:
            return stored
        return replace(stored, qdata=qdata, block_scale=block, tensor_scale=scale)
    return move(stored)


def _preserve_parameter_registration(original: StoredWeight, moved: StoredWeight) -> StoredWeight:
    if isinstance(original, torch.nn.Parameter) and not isinstance(moved, torch.nn.Parameter):
        assert isinstance(moved, torch.Tensor)
        return torch.nn.Parameter(moved, requires_grad=original.requires_grad)
    return moved


def _stored_device(stored: StoredWeight) -> torch.device:
    if isinstance(stored, Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight):
        return stored.qdata.device
    return stored.device


def _component_tensors(stored: StoredWeight) -> tuple[torch.Tensor, ...]:
    """The constituent tensors of a stored weight. Packed wrappers may
    be constructed fresh per store access (ModuleStateStore folds
    module-owned component tensors into a wrapper in ``__getitem__``),
    so aliasing between store keys is a property of these components,
    not of the wrapper object."""
    if isinstance(stored, Fp8ScaledWeight | Int8PackedWeight):
        return (stored.qdata, stored.scale)
    if isinstance(stored, Nvfp4PackedWeight):
        return (stored.qdata, stored.block_scale, stored.tensor_scale)
    return (stored,)


def apply_patches(
    weight: torch.Tensor,
    entries: Sequence[PatchEntry[torch.Tensor]],
    *,
    key: str = "",
    intermediate_dtype: torch.dtype = torch.float32,
    original_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply ``entries`` in order to ``weight`` (mutating it, like the
    reference - callers pass an owned intermediate-dtype copy) and
    return the result. ``key`` is for error messages only."""
    for entry in entries:
        strength = entry.strength
        value = entry.value
        function = entry.function

        old_weight: torch.Tensor | None = None
        if entry.offset is not None:
            old_weight = weight
            weight = weight.narrow(entry.offset.dim, entry.offset.start, entry.offset.length)

        if entry.strength_model != 1.0:
            weight *= entry.strength_model

        if isinstance(value, NestedPatch):
            # reference: v = (calculate_weight(v[1:], convert_func(
            # cast-copy of v[0][0], inplace=True)),) - the patched
            # donor base becomes a plain diff. The recursion
            # deliberately does not forward original_weight, matching
            # the reference's default-None recursion.
            nested_base = cast_to_device(value.base, weight.device, intermediate_dtype, copy=True)
            if value.convert is not None:
                nested_base = value.convert(nested_base, inplace=True)
            value = DiffPatch(
                apply_patches(
                    nested_base,
                    value.entries,
                    key=key,
                    intermediate_dtype=intermediate_dtype,
                )
            )

        if isinstance(value, AdapterPatch):
            # adapter failures raise AdapterMathError (slice 1)
            weight = value.adapter.calculate(weight, strength=strength, function=function)
            if old_weight is not None:
                weight = old_weight
            continue

        if isinstance(value, DiffPatch):
            diff = value.value
            if value.pad_weight and tuple(diff.shape) != tuple(weight.shape):
                weight = pad_tensor_to_shape(weight, tuple(diff.shape))
            if strength != 0.0:
                if tuple(diff.shape) != tuple(weight.shape):
                    raise PatchApplyError(
                        f"diff shape {tuple(diff.shape)} does not match"
                        f" weight shape {tuple(weight.shape)} for {key!r}"
                    )
                delta = strength * cast_to_device(diff, weight.device, weight.dtype)
                weight += delta if function is None else function(delta)
        elif isinstance(value, SetPatch):
            weight.copy_(value.value)
        else:  # ModelAsLoraPatch, by elimination of the closed union
            if original_weight is None:
                raise PatchApplyError(f"model-as-lora patch for {key!r} needs the original weight")
            diff_weight = cast_to_device(
                value.target, weight.device, intermediate_dtype
            ) - cast_to_device(original_weight, weight.device, intermediate_dtype)
            delta = strength * cast_to_device(diff_weight, weight.device, weight.dtype)
            weight += delta if function is None else function(delta)

        if old_weight is not None:
            weight = old_weight

    return weight


def patch_stored_weight(
    original: W,
    entries: Sequence[PatchEntry[torch.Tensor]],
    *,
    key: str,
    intermediate_dtype: torch.dtype = torch.float32,
    weight_dtype: torch.dtype | None = None,
) -> W:
    """One key of the patch_weight_to_device pipeline @ b78cec87 as a
    pure function: weight-dtype copy (dequantizing packed storage),
    apply_patches in ``intermediate_dtype``, write back to the storage
    form seeded by ``string_to_seed(key)``. ``original`` is never
    mutated; the result preserves its storage kind (see
    ``patch_weights``)."""
    weight_dtype = intermediate_dtype if weight_dtype is None else weight_dtype
    seed = string_to_seed(key)
    out_store: StoredWeight
    if isinstance(original, Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight):
        out = apply_patches(
            original.dequantize(weight_dtype),
            entries,
            key=key,
            intermediate_dtype=intermediate_dtype,
            original_weight=original.dequantize(weight_dtype),
        )
        if isinstance(original, Fp8ScaledWeight):
            out_store = requantize_fp8_scaled(original, out, seed=seed)
        elif isinstance(original, Int8PackedWeight):
            out_store = requantize_int8(original, out, seed=seed)
        else:
            out_store = requantize_nvfp4(original, out, seed=seed)
    else:
        out = apply_patches(
            original.to(dtype=weight_dtype, copy=True),
            entries,
            key=key,
            intermediate_dtype=intermediate_dtype,
            original_weight=original,
        )
        out_store = stochastic_rounding(out, original.dtype, seed=seed)
    # Sound: the writeback preserves the storage kind of ``original``
    # (plain tensor -> stochastic-rounded plain tensor, packed storage
    # -> the same requantized storage kind), so out_store is the same W.
    return cast(W, out_store)


def patch_weights(
    weights: MutableMapping[str, W],
    patch_set: PatchSet[torch.Tensor],
    *,
    intermediate_dtype: torch.dtype = torch.float32,
    weight_dtype: torch.dtype | None = None,
    backup_device: torch.device | None = None,
    key_prefix: str = "",
) -> dict[str, W]:
    """Apply every key of ``patch_set`` to ``weights`` in place and
    return the backup: key -> the original (never-mutated) stored
    weight.

    ``backup_device`` moves each backup there as it is taken - the
    reference's unconditional backup offload (patch_weight_to_device
    @ b78cec87 stores ``weight.to(device=self.offload_device)``), so
    the patching device holds only the patched copy instead of
    original plus patched for every key. A key that shares any
    constituent tensor with another store key keeps the original itself
    as backup: the alias keeps it alive anyway (offloading frees
    nothing) and restoring the identical object is what re-ties the
    aliases.
    ``None`` keeps the original stored object untouched wherever it
    lives.

    Per key this is patch_weight_to_device @ b78cec87 minus working-copy
    device movement (residency owns where weights live): weight-dtype
    copy, apply_patches in ``intermediate_dtype``, write back to the
    storage form seeded by the canonical prefixed key.

    - Plain tensor: ``.to(weight_dtype, copy=True)`` in,
      stochastic-round back to the original storage dtype (the
      set_func-is-None branch).
    - Packed storage: dequantize into the weight dtype (the
      wrapper-cast + convert_func steps collapse to exactly this),
      apply, then requantize with a recalculated scale (the reference
      set_weight branch). The reference independently chooses the
      wrapper-cast dtype (lora_compute_dtype) and calculate_weight's
      intermediate dtype (the fp32 default), so Dinkster exposes both.

    On any failure the store is left untouched for already-unpatched
    keys and restored for patched ones - the exception propagates
    after rollback."""
    backup: dict[str, W] = {}
    aliased_keys: frozenset[str] = frozenset()
    if backup_device is not None:
        # One scan grouping each key's constituent tensors by identity
        # (component tensors, because packed wrappers can be built fresh
        # per access - see _component_tensors). The scanned tensors are
        # held alive together so ids stay unique during the scan; only
        # key names survive it (a retained id could be reused by a later
        # object, and retaining the tensors would keep every original
        # alive).
        first_key_by_id: dict[int, str] = {}
        alive: list[torch.Tensor] = []
        marked: set[str] = set()
        for scan_key in weights:
            for component in _component_tensors(weights[scan_key]):
                alive.append(component)
                first = first_key_by_id.get(id(component))
                if first is None:
                    first_key_by_id[id(component)] = scan_key
                elif first != scan_key:
                    marked.add(first)
                    marked.add(scan_key)
        aliased_keys = frozenset(marked)
        del alive, first_key_by_id
    try:
        for key in patch_set.keys():
            entries = patch_set.entries(key)
            if not entries:
                continue
            if key not in weights:
                raise PatchApplyError(f"patch target {key!r} is not in the weight store")
            original = weights[key]
            patched = patch_stored_weight(
                original,
                entries,
                key=f"{key_prefix}{key}",
                intermediate_dtype=intermediate_dtype,
                weight_dtype=weight_dtype,
            )
            if backup_device is None or key in aliased_keys:
                backup[key] = original
            else:
                moved = move_stored(original, backup_device)
                backup[key] = cast(W, _preserve_parameter_registration(original, moved))
            weights[key] = patched
    except Exception:
        restore_weights(weights, backup)
        raise
    return backup


def restore_weights(
    weights: MutableMapping[str, W],
    backup: Mapping[str, W],
) -> None:
    """Exact restoration: put every backed-up original weight back,
    moved to the device of the stored weight it replaces (identity when
    the backup never left that device)."""
    for key, original in backup.items():
        moved = move_stored(original, _stored_device(weights[key]))
        weights[key] = cast(W, _preserve_parameter_registration(original, moved))


__all__ = [
    "PatchApplyError",
    "StoredWeight",
    "apply_patches",
    "move_stored",
    "patch_stored_weight",
    "patch_weights",
    "restore_weights",
]
