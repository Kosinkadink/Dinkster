"""Cast-at-use: turning a stored weight into a compute-ready tensor.

The reference does this inside module classes (comfy/ops.py
cast_bias_weight @ b78cec87, standard branch), reading state off the
module (``weight_function`` lists, quantized wrappers). Dinkster keeps
the exact pipeline but as one pure function over explicit inputs:

    move to device (storage dtype) -> cast to compute dtype ->
    dequantize fp8-scaled storage -> apply deferred functions

``DeferredPatch`` is the reference's LowVramPatch: a key's patch
entries applied at cast time, at the weight's own compute dtype (NOT
the fp32 intermediate the patched-at-load path uses) - the low-VRAM
mode where patches never touch the stored weight.

Ordering pins, straight from the reference:

- The device move happens at STORAGE dtype (small fp8 transfer), the
  dtype cast after arrival; a quantized weight dequantizes only after
  the cast, directly into the compute dtype.
- Deferred functions run last, on the compute-dtype tensor, and
  receive an owned buffer (the reference forces ``copy=True`` on the
  move when functions exist; they may mutate in place).
- With no functions and no dtype change, the input tensor may be
  returned as-is (the reference does the same) - callers must not
  mutate the result of a bare cast.

Residency decisions (WHERE a weight should live, when to evict) are a
later slice; this module is the mechanism those policies will call.
Keep calls to this outside torch.compile regions: patch application
and storage mutation stay in eager land (docs/native-inference-plan.md).

The reference also skips dequantization entirely when a quantized
weight needs no functions and no dtype change, so quantized matmul
kernels can consume qdata directly. That fast path belongs to the
native ops/Linear surface (ROADMAP), not to this weight-preparation
seam, which always returns a plain tensor.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import torch
from dinkster_inference.patches import PatchEntry, patch_payloads, rebuild_patch_entries

from .apply import apply_patches
from .quant import Fp8ScaledWeight, Int8PackedWeight, Nvfp4PackedWeight

WeightFunction = Callable[[torch.Tensor], torch.Tensor]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeferredPatch:
    """comfy/model_patcher.py LowVramPatch @ b78cec87: apply ``key``'s
    patch entries when the weight is cast for use, with the
    intermediate dtype pinned to the incoming weight's dtype."""

    key: str
    entries: tuple[PatchEntry[torch.Tensor], ...]

    def __call__(self, weight: torch.Tensor) -> torch.Tensor:
        return apply_patches(
            weight,
            self.entries,
            key=self.key,
            intermediate_dtype=weight.dtype,
        )


class PreparedPatchSource:
    """Stream-scoped storage-dtype staging for one frozen patch revision.

    Prepared entries live here, never on :class:`DeferredPatch`. A commit is
    single-use: ``__call__`` consumes it and clears even when patch math fails.
    The mechanism also clears around the whole cast to cover failures before
    deferred functions are invoked.
    """

    def __init__(self, source: DeferredPatch, *, alignment: int = 1024) -> None:
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("patch payload alignment must be a positive power of two")
        self.source = source
        self.alignment = alignment
        self._payloads: tuple[torch.Tensor, ...] | None = None
        self._prepared: tuple[PatchEntry[torch.Tensor], ...] | None = None
        self._unsupported: set[str] = set()
        self._diagnosed = False

    def _note_unsupported(self, adapter: object) -> None:
        self._unsupported.add(type(adapter).__qualname__)

    def payload_tensors(self) -> tuple[torch.Tensor, ...]:
        if self._payloads is None:
            self._payloads = patch_payloads(self.source.entries, unsupported=self._note_unsupported)
            if self._unsupported and not self._diagnosed:
                self._diagnosed = True
                logger.warning(
                    "dinkster.patch_payload_unstaged key=%r adapters=%s",
                    self.source.key,
                    ",".join(sorted(self._unsupported)),
                )
        return self._payloads

    def memory_required(self) -> int:
        return sum(
            (payload.nbytes + self.alignment - 1) & ~(self.alignment - 1)
            for payload in self.payload_tensors()
        )

    def prepare(
        self,
        destination_region: torch.Tensor,
        *,
        payload_sources: Sequence[torch.Tensor] | None = None,
        non_blocking: bool,
    ) -> tuple[PatchEntry[torch.Tensor], ...]:
        """Copy payloads byte-identically and return entries over device views."""
        payloads = self.payload_tensors()
        sources = payloads if payload_sources is None else tuple(payload_sources)
        if len(sources) != len(payloads):
            raise ValueError("prepared patch source count does not match payload walk")
        required = self.memory_required()
        if (
            destination_region.dtype != torch.uint8
            or not destination_region.is_contiguous()
            or destination_region.numel() < required
        ):
            raise ValueError(
                "prepared patch destination must be contiguous uint8 storage"
                f" with at least {required} bytes"
            )
        replacements: list[torch.Tensor] = []
        offset = 0
        for payload, source in zip(payloads, sources, strict=True):
            if source.dtype != payload.dtype or tuple(source.shape) != tuple(payload.shape):
                raise ValueError("prepared patch source changed storage dtype or shape")
            end = offset + payload.nbytes
            view = destination_region[offset:end].view(payload.dtype).reshape(payload.shape)
            view.copy_(source, non_blocking=non_blocking)
            replacements.append(view)
            offset += (payload.nbytes + self.alignment - 1) & ~(self.alignment - 1)
        return rebuild_patch_entries(
            self.source.entries,
            replacements,
            unsupported=self._note_unsupported,
        )

    def commit(self, entries: tuple[PatchEntry[torch.Tensor], ...]) -> None:
        if self._prepared is not None:
            raise RuntimeError(f"prepared patch for {self.source.key!r} is already committed")
        self._prepared = entries

    def clear_prepared(self) -> None:
        self._prepared = None

    def take_prepared(self) -> tuple[PatchEntry[torch.Tensor], ...]:
        entries = self._prepared
        if entries is None:
            return self.source.entries
        self._prepared = None
        return entries

    def __call__(self, weight: torch.Tensor) -> torch.Tensor:
        entries = self.take_prepared()
        return apply_patches(
            weight,
            entries,
            key=self.source.key,
            intermediate_dtype=weight.dtype,
        )


def cast_weight(
    stored: torch.Tensor | Fp8ScaledWeight | Int8PackedWeight | Nvfp4PackedWeight,
    *,
    dtype: torch.dtype,
    device: torch.device | None = None,
    functions: Sequence[WeightFunction] = (),
    non_blocking: bool = False,
) -> torch.Tensor:
    """The cast_bias_weight weight path @ b78cec87 over explicit
    state: move ``stored`` to ``device`` at storage dtype, cast to the
    compute ``dtype`` (dequantizing packed storage), then apply
    ``functions`` in order (DeferredPatch instances and other weight
    hooks)."""
    has_function = len(functions) > 0

    if isinstance(stored, Int8PackedWeight | Nvfp4PackedWeight):
        if device is not None:
            if isinstance(stored, Int8PackedWeight):
                stored = replace(
                    stored,
                    qdata=stored.qdata.to(device=device, non_blocking=non_blocking),
                    scale=stored.scale.to(device=device, non_blocking=non_blocking),
                )
            else:
                stored = replace(
                    stored,
                    qdata=stored.qdata.to(device=device, non_blocking=non_blocking),
                    block_scale=stored.block_scale.to(device=device, non_blocking=non_blocking),
                    tensor_scale=stored.tensor_scale.to(device=device, non_blocking=non_blocking),
                )
        weight = stored.dequantize(dtype)
    elif isinstance(stored, Fp8ScaledWeight):
        qdata = stored.qdata
        scale = stored.scale
        if device is not None:
            qdata = qdata.to(device=device, non_blocking=non_blocking)
            scale = scale.to(device=device, non_blocking=non_blocking)
        weight = qdata.to(dtype=dtype) * scale.to(dtype=dtype)
    else:
        weight = stored
        if device is not None or has_function:
            weight = weight.to(
                device=device if device is not None else weight.device,
                copy=has_function,
                non_blocking=non_blocking,
            )
        if has_function or weight.dtype != dtype:
            weight = weight.to(dtype=dtype)

    for function in functions:
        weight = function(weight)
    return weight


__all__ = [
    "DeferredPatch",
    "PreparedPatchSource",
    "WeightFunction",
    "cast_weight",
]
