"""Per-job folder preference, never filesystem authority or a host path."""

from __future__ import annotations

import re
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import cast


@dataclass(frozen=True)
class AssetFolderContext:
    mounts: tuple[str, ...]
    output_mount: str | None = None

    def __post_init__(self) -> None:
        mounts = cast(object, self.mounts)
        if not isinstance(mounts, tuple) or any(
            not isinstance(item, str) or re.fullmatch(r"[a-z0-9][a-z0-9-]*", item) is None
            for item in cast(tuple[object, ...], mounts)
        ):
            raise ValueError("asset folder mounts must be a tuple of mount ids")
        if len(set(self.mounts)) != len(self.mounts):
            raise ValueError("asset folder mount ids must be unique")
        if self.output_mount is not None and self.output_mount not in self.mounts:
            raise ValueError("asset folder output must belong to its mounts")

    def to_wire(self) -> dict[str, object]:
        return {"mounts": list(self.mounts), "outputMount": self.output_mount}

    @classmethod
    def from_wire(cls, raw: object) -> AssetFolderContext:
        if not isinstance(raw, Mapping):
            raise ValueError("asset folders must contain mounts and outputMount")
        wire = cast(Mapping[object, object], raw)
        if set(wire) != {"mounts", "outputMount"}:
            raise ValueError("asset folders must contain mounts and outputMount")
        mounts = wire["mounts"]
        output = wire["outputMount"]
        if not isinstance(mounts, list) or any(
            not isinstance(item, str) for item in cast(list[object], mounts)
        ):
            raise ValueError("asset folder mounts must be a list of mount ids")
        if output is not None and not isinstance(output, str):
            raise ValueError("asset folder output must be a mount id or null")
        return cls(tuple(cast(list[str], mounts)), output)


_asset_folders: ContextVar[AssetFolderContext | None] = ContextVar("asset_folders", default=None)


def current_asset_folders() -> AssetFolderContext | None:
    return _asset_folders.get()


@contextmanager
def use_asset_folders(context: AssetFolderContext | None) -> Generator[None]:
    token = _asset_folders.set(context)
    try:
        yield
    finally:
        _asset_folders.reset(token)
