"""The ``dinkster.save_target`` value type: structured save destinations.

ComfyUI's save nodes take a raw ``filename_prefix`` string joined onto the
one blessed output directory - a location-dependent value that cannot name
any other granted folder and that historically invited traversal bugs. A
SaveTarget replaces it with structure: an optional MOUNT id (the unit of
filesystem authority, see mounts.py) plus a relative PREFIX (subfolders and
filename stem) under it. Without a mount the save lands in the project's
default output mount on whichever machine runs it. Real host paths never
appear in the value, so a document carrying one is portable across machines
and survives a mount being re-pointed at a different directory.

The wire form is ``{"mount": <id>, "prefix": <"sub/dirs/stem">}``, or
``{"prefix": ...}`` for no mount - exactly what a frontend SAVE_TARGET
widget produces. Validation is strict and
identical everywhere: mount ids follow the mount grammar, prefixes follow
virtual-path rules (no absolute paths, no ``.``/``..``/empty segments, no
backslashes), so an unsafe destination is unrepresentable, not merely
checked at write time.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from dinkster_values import MOUNT_TYPE, SAVE_TARGET_TYPE, TypeRegistry

from .identity import AssetError
from .mounts import MOUNT_ID_PATTERN

__all__ = [
    "MOUNT_TYPE",
    "SAVE_TARGET_TYPE",
    "SaveTarget",
    "register_mount_type",
    "register_save_target_type",
]


def _coerce_mount(obj: object) -> str | None:
    if obj is None:
        return None
    if not isinstance(obj, str) or not MOUNT_ID_PATTERN.match(obj):
        raise AssetError(
            f"save target mount ids are lowercase alphanumerics and "
            f"hyphens, starting alphanumeric: {obj!r}"
        )
    return obj


def register_mount_type(registry: TypeRegistry) -> None:
    """Register a mount identity, or None for the project default, without write authority."""

    def encode(obj: object) -> bytes:
        return json.dumps(_coerce_mount(obj), separators=(",", ":")).encode("utf-8")

    def decode(data: bytes) -> object:
        return _coerce_mount(json.loads(data))

    def validate_encoded(data: bytes, _metadata: Mapping[str, object]) -> None:
        decode(data)

    registry.register(
        MOUNT_TYPE,
        encode=encode,
        decode=decode,
        coerce=_coerce_mount,
        input_convert=_coerce_mount,
        validate_encoded=validate_encoded,
        asset_references_complete=True,
    )


@dataclass(frozen=True)
class SaveTarget:
    """Where to save: an optional mount id plus a relative prefix under it.

    ``prefix`` is ``/``-separated: every segment but the last is a
    subfolder (created on demand), the last is the filename STEM the
    writer appends a counter and extension to. The node owns the actual
    encoding; the target only names the destination. ``mount`` None means
    the project's default output mount, chosen when the file is written."""

    mount: str | None
    prefix: str

    def __post_init__(self) -> None:
        _coerce_mount(self.mount)
        if not self.prefix:
            raise AssetError(
                "save target prefix must name at least a filename stem "
                "(e.g. 'ComfyUI' or 'renders/scene')"
            )
        if "\\" in self.prefix:
            raise AssetError(f"save target prefixes use '/' separators only: {self.prefix!r}")
        if any(seg in ("", ".", "..") for seg in self.prefix.split("/")):
            raise AssetError(
                f"invalid save target prefix (absolute path or empty/./.. segment): {self.prefix!r}"
            )

    @property
    def subfolder(self) -> str:
        """The folder part of the prefix ('' when the stem sits at the
        mount root)."""
        head, sep, _ = self.prefix.rpartition("/")
        return head if sep else ""

    @property
    def stem(self) -> str:
        """The filename stem the writer numbers and suffixes."""
        return self.prefix.rsplit("/", 1)[-1]

    def to_wire(self) -> dict[str, object]:
        if self.mount is None:
            return {"prefix": self.prefix}
        return {"mount": self.mount, "prefix": self.prefix}

    @classmethod
    def from_wire(cls, wire: Mapping[str, object]) -> SaveTarget:
        mount = wire.get("mount")
        prefix = wire.get("prefix")
        if ("mount" in wire and not isinstance(mount, str)) or not isinstance(prefix, str):
            raise AssetError(
                "save target wire form requires a 'prefix' string and, when present, "
                f"a 'mount' string, got: {dict(wire)!r}"
            )
        unknown = set(wire) - {"mount", "prefix"}
        if unknown:
            raise AssetError(f"save target wire form has unknown keys {sorted(unknown)}")
        return cls(mount=cast("str | None", mount), prefix=prefix)


def coerce_save_target(obj: object) -> SaveTarget:
    """Accept a SaveTarget or its wire mapping - the shared coercion the
    type registration and node code both use, so a graph literal and a
    programmatic value follow one rule."""
    if isinstance(obj, SaveTarget):
        return obj
    if isinstance(obj, Mapping):
        return SaveTarget.from_wire(cast("Mapping[str, object]", obj))
    raise AssetError(
        "save target inputs must be a SaveTarget or a mapping with 'prefix' "
        f"and an optional 'mount' - got {type(obj).__name__} (save destinations are "
        "structured, never raw path strings)"
    )


def register_save_target_type(registry: TypeRegistry) -> None:
    """Register ``dinkster.save_target``. Pure data - no resolver, no IO; the
    fingerprint is the canonical wire JSON (two targets naming the same
    destination are the same value everywhere)."""

    def encode(obj: object) -> bytes:
        target = coerce_save_target(obj)
        return json.dumps(target.to_wire(), sort_keys=True, separators=(",", ":")).encode("utf-8")

    def decode(data: bytes) -> object:
        wire: object = json.loads(data)
        if not isinstance(wire, Mapping):
            raise AssetError("save target wire form must be a JSON object")
        return SaveTarget.from_wire(cast("Mapping[str, object]", wire))

    def fingerprint(obj: object) -> str:
        assert isinstance(obj, SaveTarget)  # coerce ran first
        # "" is never a mount id, so no mount stays distinct from every named one.
        return f"{SAVE_TARGET_TYPE}:{obj.mount or ''}/{obj.prefix}"

    def meta(obj: object) -> Mapping[str, object]:
        assert isinstance(obj, SaveTarget)  # coerce ran first
        return obj.to_wire()

    def validate_encoded(data: bytes, _metadata: Mapping[str, object]) -> None:
        decode(data)

    registry.register(
        SAVE_TARGET_TYPE,
        encode=encode,
        decode=decode,
        fingerprint=fingerprint,
        meta=meta,
        coerce=coerce_save_target,
        validate_encoded=validate_encoded,
        asset_references_complete=True,
    )
