"""Immutable validated JSON metadata values."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import cast

from .identity import AssetError


def freeze_json(value: object, path: str = "metadata") -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AssetError(f"{path} must contain finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in cast("Mapping[object, object]", value).items():
            if not isinstance(key, str):
                raise AssetError(f"{path} keys must be strings")
            frozen[key] = freeze_json(item, f"{path}.{key}")
        return MappingProxyType(dict(sorted(frozen.items())))
    if isinstance(value, (list, tuple)):
        sequence = cast("Sequence[object]", value)
        return tuple(freeze_json(item, f"{path}[]") for item in sequence)
    raise AssetError(f"{path} must contain only JSON values")


def thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): thaw_json(item)
            for key, item in cast("Mapping[object, object]", value).items()
        }
    if isinstance(value, tuple):
        return [thaw_json(item) for item in cast("tuple[object, ...]", value)]
    return value
