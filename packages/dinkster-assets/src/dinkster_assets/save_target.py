"""The ``dinkster.save_target`` value type: structured save destinations.

ComfyUI's save nodes take a raw ``filename_prefix`` string joined onto the
one blessed output directory - a location-dependent value that cannot name
any other granted folder and that historically invited traversal bugs. A
SaveTarget replaces it with structure: a MOUNT id (the unit of filesystem
authority, see mounts.py) plus a relative PREFIX (subfolders and filename
stem) under it. Real host paths never appear in the value, so a document
carrying one is portable across machines and survives a mount being
re-pointed at a different directory.

The wire form is ``{"mount": <id>, "prefix": <"sub/dirs/stem">}`` - exactly
what a frontend SAVE_TARGET widget produces. Validation is strict and
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

from dinkster_values import TypeRegistry

from .identity import AssetError
from .mounts import MOUNT_ID_PATTERN

__all__ = ["SAVE_TARGET_TYPE", "SaveTarget", "register_save_target_type"]

SAVE_TARGET_TYPE = "dinkster.save_target"


@dataclass(frozen=True)
class SaveTarget:
    """Where to save: a mount id plus a relative prefix under it.

    ``prefix`` is ``/``-separated: every segment but the last is a
    subfolder (created on demand), the last is the filename STEM the
    writer appends a counter and extension to. The node owns the actual
    encoding; the target only names the destination."""

    mount: str
    prefix: str

    def __post_init__(self) -> None:
        if not MOUNT_ID_PATTERN.match(self.mount):
            raise AssetError(
                f"save target mount ids are lowercase alphanumerics and "
                f"hyphens, starting alphanumeric: {self.mount!r}"
            )
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
        return {"mount": self.mount, "prefix": self.prefix}

    @classmethod
    def from_wire(cls, wire: Mapping[str, object]) -> SaveTarget:
        mount = wire.get("mount")
        prefix = wire.get("prefix")
        if not isinstance(mount, str) or not isinstance(prefix, str):
            raise AssetError(
                f"save target wire form requires 'mount' and 'prefix' strings, got: {dict(wire)!r}"
            )
        unknown = set(wire) - {"mount", "prefix"}
        if unknown:
            raise AssetError(f"save target wire form has unknown keys {sorted(unknown)}")
        return cls(mount=mount, prefix=prefix)


def coerce_save_target(obj: object) -> SaveTarget:
    """Accept a SaveTarget or its wire mapping - the shared coercion the
    type registration and node code both use, so a graph literal and a
    programmatic value follow one rule."""
    if isinstance(obj, SaveTarget):
        return obj
    if isinstance(obj, Mapping):
        return SaveTarget.from_wire(cast("Mapping[str, object]", obj))
    raise AssetError(
        "save target inputs must be a SaveTarget or a mapping with 'mount' "
        f"and 'prefix' - got {type(obj).__name__} (save destinations are "
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
        return f"{SAVE_TARGET_TYPE}:{obj.mount}/{obj.prefix}"

    def meta(obj: object) -> Mapping[str, object]:
        assert isinstance(obj, SaveTarget)  # coerce ran first
        return obj.to_wire()

    registry.register(
        SAVE_TARGET_TYPE,
        encode=encode,
        decode=decode,
        fingerprint=fingerprint,
        meta=meta,
        coerce=coerce_save_target,
    )
