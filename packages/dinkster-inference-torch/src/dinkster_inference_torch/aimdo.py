"""The aimdo seam: probing comfy-aimdo behind the residency protocol.

comfy-aimdo ("AI Model Dynamic Offloader") is upstream's VMM-based
dynamic residency backend: per model it reserves a large virtual GPU
address range (ModelVBAR via cuMemAddressReserve), assigns weights
ranges inside it, explicitly faults pages in immediately before an
operation (mapping physical VRAM only if budget allows, falling back
to a temporary tensor on OOM), and unpins after - with an implicit
newest-model-first, watermark-limited eviction policy. It exposes NO
model wrapper or policy callback; the host application owns weight
ordering, fault timing, copies, fallbacks, and unpin sequencing
(comfy-aimdo 0.4.13; ComfyUI @ b78cec87 integrates it by swapping
ModelPatcher for ModelPatcherDynamic and threading VBAR faults through
comfy/ops.py cast_bias_weight).

Dinkster's seam is ``ResidencyMechanism`` (residency.py): an aimdo
VBAR-backed mechanism is a drop-in sibling of ``ResidentWeights``, and
the manager never knows which it drives. That mechanism is NOT built
in this slice - per-op faulting only makes sense once native module
execution exists (stage 5's Ops surface, where fault/unpin brackets
each op's cast) - so the seam ships as the protocol plus this probe;
the ROADMAP carries the implementation with that trigger.

Two integration constraints pinned now so nothing has to be
retrofitted:

- ``comfy_aimdo.control.init()`` must run BEFORE torch is imported
  into the process (upstream calls it at the top of main.py). This
  package imports torch, so init belongs to the worker process
  bootstrap, never to code here; this module only observes whether it
  happened (``initialized``).
- comfy-aimdo is a namespace package with no root ``__init__``; probe
  by importing ``comfy_aimdo.control``, never ``comfy_aimdo``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AimdoStatus:
    """What the probe observed: whether comfy_aimdo is importable,
    whether its native library was already initialized by the process
    bootstrap (control.init before torch), and the GPU vendor its
    detection reports ("cuda"/"rocm"/None)."""

    importable: bool
    initialized: bool
    vendor: str | None


def probe_aimdo() -> AimdoStatus:
    """Capability probe for comfy-aimdo (never a version check).
    Importing comfy_aimdo.control is side-effect free - the native
    library loads only through control.init()."""
    try:
        from comfy_aimdo import (  # pyright: ignore[reportMissingTypeStubs]
            control,
        )
    except ImportError:
        return AimdoStatus(importable=False, initialized=False, vendor=None)
    # capability, not version: older builds predate detect_vendor
    detect = getattr(control, "detect_vendor", None)
    raw_vendor = detect() if callable(detect) else None
    vendor = raw_vendor if isinstance(raw_vendor, str) else None
    initialized = getattr(control, "lib", None) is not None
    return AimdoStatus(importable=True, initialized=initialized, vendor=vendor)


__all__ = [
    "AimdoStatus",
    "probe_aimdo",
]
