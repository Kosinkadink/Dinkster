"""Immutable folder preference, never filesystem authority or a host path."""

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
    hidden_models: tuple[str, ...] = ()

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

        hidden = cast(object, self.hidden_models)
        if (
            not isinstance(hidden, tuple)
            or any(
                not isinstance(item, str)
                or re.fullmatch(r"(?:blake3:|local-revision:)[0-9a-f]{64}", item) is None
                for item in cast(tuple[object, ...], hidden)
            )
            or len(set(self.hidden_models)) != len(self.hidden_models)
        ):
            raise ValueError("hidden models must be unique content or local revision identities")

    def to_wire(self) -> dict[str, object]:
        wire: dict[str, object] = {"mounts": list(self.mounts), "outputMount": self.output_mount}
        if self.hidden_models:
            wire["hiddenModels"] = list(self.hidden_models)
        return wire

    @classmethod
    def from_wire(cls, raw: object) -> AssetFolderContext:
        if not isinstance(raw, Mapping):
            raise ValueError("asset folders must contain mounts and outputMount")
        wire = cast(Mapping[object, object], raw)
        if set(wire) not in ({"mounts", "outputMount"}, {"mounts", "outputMount", "hiddenModels"}):
            raise ValueError("asset folders must contain mounts and outputMount")
        mounts = wire["mounts"]
        output = wire["outputMount"]
        if not isinstance(mounts, list) or any(
            not isinstance(item, str) for item in cast(list[object], mounts)
        ):
            raise ValueError("asset folder mounts must be a list of mount ids")
        if output is not None and not isinstance(output, str):
            raise ValueError("asset folder output must be a mount id or null")
        hidden = wire.get("hiddenModels", [])
        if not isinstance(hidden, list) or any(
            not isinstance(item, str) for item in cast(list[object], hidden)
        ):
            raise ValueError("hidden models must be a list of identities")
        return cls(tuple(cast(list[str], mounts)), output, tuple(cast(list[str], hidden)))


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
