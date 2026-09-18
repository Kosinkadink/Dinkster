"""Sampling-preview policy shared by server, engine, and workers.

A preview mode says how much a run may spend on live sampling previews:

- ``off``: no preview work at all; samplers never install state callbacks.
- ``cheap``: only constant-cost decoders (for example latent2rgb matrices).
- ``quality``: prefer model-based decoders (TAE-class), falling back to
  cheap ones when no model decoder is available.
- ``auto``: let the worker pick; today it resolves like ``quality``.

A :class:`PreviewPolicy` is the run-scoped resolution of the user-facing
global/workflow/node settings: one base mode plus optional per-node
overrides, resolved to a single mode per invocation before it crosses the
worker boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

PREVIEW_MODES = ("off", "cheap", "quality", "auto")
PreviewMode = Literal["off", "cheap", "quality", "auto"]

PREVIEW_ANIMATIONS = ("ring", "encoded")
PreviewAnimation = Literal["ring", "encoded"]
"""How animated previews travel: ``ring`` ships per-frame stills the
frontend cycles in a fixed frame ring; ``encoded`` ships one
self-contained animation container (an animated WebP) per emit."""


def validate_preview_mode(mode: object) -> PreviewMode:
    if not isinstance(mode, str) or mode not in PREVIEW_MODES:
        raise ValueError(f"unsupported preview mode: {mode!r} (expected one of {PREVIEW_MODES})")
    return mode


def validate_preview_animation(animation: object) -> PreviewAnimation:
    if not isinstance(animation, str) or animation not in PREVIEW_ANIMATIONS:
        raise ValueError(
            f"unsupported preview animation: {animation!r} (expected one of {PREVIEW_ANIMATIONS})"
        )
    return animation


@dataclass(frozen=True)
class PreviewPolicy:
    """One run's effective preview policy: a base mode plus per-node
    overrides keyed by graph node id. ``resolve`` collapses it to the
    single mode an invocation carries. ``animation`` picks the transport
    for animated previews run-wide; it never varies per node."""

    mode: PreviewMode = "off"
    node_modes: Mapping[str, PreviewMode] = field(default_factory=dict[str, PreviewMode])
    animation: PreviewAnimation = "ring"

    def __post_init__(self) -> None:
        validate_preview_mode(self.mode)
        validate_preview_animation(self.animation)
        node_modes = cast("object", self.node_modes)
        if not isinstance(node_modes, Mapping):
            raise ValueError("PreviewPolicy.node_modes must be a mapping")
        checked = cast("Mapping[object, object]", node_modes)
        for node_id, mode in checked.items():
            if not isinstance(node_id, str) or not node_id:
                raise ValueError("PreviewPolicy.node_modes keys must be non-empty strings")
            validate_preview_mode(mode)
        object.__setattr__(
            self, "node_modes", MappingProxyType(dict(cast("Mapping[str, PreviewMode]", checked)))
        )

    def resolve(self, node_id: str) -> PreviewMode:
        return self.node_modes.get(node_id, self.mode)


__all__ = [
    "PREVIEW_ANIMATIONS",
    "PREVIEW_MODES",
    "PreviewAnimation",
    "PreviewMode",
    "PreviewPolicy",
    "validate_preview_animation",
    "validate_preview_mode",
]
