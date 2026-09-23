"""Document-kind normalization and host-owned snapshot validation."""

from __future__ import annotations

import re
from collections.abc import Callable

SnapshotValidator = Callable[[str, object], str | None]

DOCUMENT_KINDS = ("workflow", "image", "video")
_NAMESPACED_KIND = re.compile(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+")
_LEGACY_KINDS = {
    "workflow": "dinkster.workflow",
    "image": "dinkster.image",
    "video": "dinkster.video",
}


def normalize_document_kind(kind: object = "workflow") -> str:
    if not isinstance(kind, str):
        raise ValueError("document kind must be a namespaced string")
    if kind in _LEGACY_KINDS:
        return _LEGACY_KINDS[kind]
    if _NAMESPACED_KIND.fullmatch(kind) is None:
        raise ValueError("document kind must be namespaced (for example 'extension.type')")
    return kind


def document_kind_wire(kind: str) -> str:
    canonical = normalize_document_kind(kind)
    for legacy, builtin in _LEGACY_KINDS.items():
        if canonical == builtin:
            return legacy
    return canonical


class SnapshotValidatorRegistry:
    """Dispatch snapshots by canonical kind; unregistered kinds remain opaque."""

    def __init__(self) -> None:
        self._validators: dict[str, SnapshotValidator] = {}

    def register(self, kind: str, validator: SnapshotValidator) -> None:
        canonical = normalize_document_kind(kind)
        if canonical in self._validators:
            raise ValueError(f"snapshot validator already registered for {canonical}")
        self._validators[canonical] = validator

    def __call__(self, kind: str, document_id: str, snapshot: object) -> str | None:
        validator = self._validators.get(normalize_document_kind(kind))
        return validator(document_id, snapshot) if validator is not None else None
