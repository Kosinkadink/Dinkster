"""Model asset inputs and loading options shared by node packs."""

from __future__ import annotations

import importlib
from typing import Any

from dinkster_api.v1 import (
    ASSET_TYPE,
    CORE_COMBO,
    AssetRef,
    AssetWidget,
    ComboOption,
    ComboWidget,
    InputSpec,
    LocalAssetRef,
    TypeExpr,
    mount_roots_for_kind,
)

_ASSET = TypeExpr.concrete(ASSET_TYPE)
_COMBO = TypeExpr.concrete(CORE_COMBO)


def asset_input(name: str, kind: str, *, doc: str = "") -> InputSpec:
    return InputSpec(
        name,
        _ASSET,
        doc=doc,
        widget=AssetWidget(kind=kind, accept=("application/octet-stream",)),
    )


def asset_path(value: object, name: str) -> str:
    if not isinstance(value, (AssetRef, LocalAssetRef)):
        raise TypeError(f"{name} must be an AssetRef or LocalAssetRef")
    return str(value.local_path())


def embedding_roots() -> list[str]:
    return [str(root) for root in mount_roots_for_kind("model/embedding")]


def clip_options(device: str) -> dict[str, Any]:
    if device == "default":
        return {}
    torch = importlib.import_module("torch")
    return {"load_device": torch.device("cpu"), "offload_device": torch.device("cpu")}


def combo_input(
    name: str, options: tuple[str, ...], default: str, *, advanced: bool = False
) -> InputSpec:
    return InputSpec(
        name,
        _COMBO,
        default=default,
        advanced=advanced,
        widget=ComboWidget(tuple(ComboOption(option) for option in options)),
    )
