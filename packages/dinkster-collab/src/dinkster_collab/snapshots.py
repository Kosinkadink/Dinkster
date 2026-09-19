"""Document-kind normalization and host-owned whole-snapshot validation."""

from __future__ import annotations

import re
from collections.abc import Callable

_NAMESPACED_KIND = re.compile(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+")
_LEGACY_KINDS = {"workflow": "dinkster.workflow", "image": "dinkster.image"}


def normalize_document_kind(kind: object = "workflow") -> str:
    if not isinstance(kind, str):
        raise ValueError("document kind must be a namespaced string")
    if kind in _LEGACY_KINDS:
        return _LEGACY_KINDS[kind]
    if not _NAMESPACED_KIND.fullmatch(kind):
        raise ValueError("document kind must be namespaced (for example 'extension.type')")
    return kind


class SnapshotValidatorRegistry:
    """Dispatch whole snapshots by canonical kind; unregistered kinds stay opaque."""

    def __init__(self) -> None:
        self._validators: dict[str, Callable[[str, object], str | None]] = {}

    def register(self, kind: str, validator: Callable[[str, object], str | None]) -> None:
        kind = normalize_document_kind(kind)
        if kind in self._validators:
            raise ValueError(f"snapshot validator already registered for {kind}")
        self._validators[kind] = validator

    def __call__(self, kind: str, document_id: str, snapshot: object) -> str | None:
        validator = self._validators.get(normalize_document_kind(kind))
        return validator(document_id, snapshot) if validator is not None else None
