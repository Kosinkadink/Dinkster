"""Disposable, portable pack declarations; never process or hardware authority."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from dinkster_protocol.pack_surfaces import pack_surfaces_to_wire
from dinkster_schema import TypeExpr, schema_to_wire

from .manifest import PackManifest
from .session import PackDeclarations

_SKIP = {"__pycache__", ".git", ".hg", ".venv", "node_modules", ".tox"}


@dataclass(frozen=True)
class CatalogTypes:
    type_ids: frozenset[str]
    asset_decoders: frozenset[str]
    batch_merges: frozenset[str]
    equivalences: Mapping[str, str]

    @classmethod
    def from_wire(cls, value: Any) -> CatalogTypes:
        if not isinstance(value, dict):
            raise ValueError("catalog types must be an object")
        data = cast(dict[str, Any], value)
        groups: list[frozenset[str]] = []
        for key in ("typeIds", "assetDecoders", "batchMerges"):
            raw = data[key]
            if not isinstance(raw, list):
                raise ValueError(f"catalog {key} must be a list")
            items = cast(list[Any], raw)
            if any(
                not isinstance(item, str)
                or TypeExpr.runtime_type_atom(item) is None
                or (key != "assetDecoders" and TypeExpr.runtime_type_atom(item) != item)
                or (key == "assetDecoders" and "asset<" in item)
                for item in items
            ):
                raise ValueError(f"catalog {key} must contain type IDs")
            groups.append(frozenset(items))
        raw_pairs = data["equivalences"]
        if not isinstance(raw_pairs, dict):
            raise ValueError("catalog equivalences must be an object")
        pairs = cast(dict[Any, Any], raw_pairs)
        if any(
            not isinstance(left, str)
            or not isinstance(right, str)
            or TypeExpr.runtime_type_atom(left) != left
            or TypeExpr.runtime_type_atom(right) != right
            or left == right
            or pairs.get(right) != left
            for left, right in pairs.items()
        ):
            raise ValueError("catalog equivalences must contain symmetric atom pairs")
        return cls(groups[0], groups[1], groups[2], MappingProxyType(dict(pairs)))

    @property
    def atoms(self) -> frozenset[str]:
        return frozenset(
            atom
            for tid in self.type_ids
            | self.asset_decoders
            | self.batch_merges
            | self.equivalences.keys()
            if (atom := TypeExpr.runtime_type_atom(tid)) is not None
        )


class PackCatalog(PackDeclarations):
    def __init__(self, declarations: Mapping[str, Any], source: str) -> None:
        super().__init__(declarations)
        self.types = CatalogTypes.from_wire(declarations["types"])
        self.source = source


def catalog_path(manifest_path: Path) -> Path:
    # Artifacts exclude __pycache__; a catalog must not change its own source identity.
    return (
        manifest_path.parent
        / "__pycache__"
        / "dinkster-schema-catalog"
        / f"{manifest_path.name}.json"
    )


def source_digest(manifest: PackManifest) -> str:
    digest = hashlib.sha256(manifest.path.name.encode("utf-8") + b"\0")
    for root, dirs, files in os.walk(manifest.root):
        dirs[:] = sorted(name for name in dirs if name not in _SKIP)
        for name in sorted(files):
            path = Path(root) / name
            relative = path.relative_to(manifest.root).as_posix().encode()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            data = path.read_bytes()
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
    return digest.hexdigest()


def worker_declarations(worker: Any) -> dict[str, Any]:
    arms = cast("Mapping[str, tuple[str, ...]]", worker.body_arms or {})
    return {
        "schemas": {name: schema_to_wire(schema) for name, schema in worker.schemas.items()},
        "comboChoices": {name: list(values) for name, values in worker.combo_choices.items()},
        "lazyChoiceIds": sorted(worker.lazy_choice_ids),
        "compatSkips": {name: value.to_wire() for name, value in worker.compat_skips.items()},
        "bodyArms": {name: sorted(values) for name, values in arms.items()},
        "extensionContributions": [
            {
                "scope": scope.value,
                "surfaceId": item.surface_id,
                "mode": item.mode.value,
                **pack_surfaces_to_wire(item.routes, item.events),
            }
            for scope, item in worker.extension_contributions
        ],
    }


def write_catalog(manifest: PackManifest, declarations: Mapping[str, Any], *, source: str) -> None:
    PackCatalog(declarations, source)
    destination = catalog_path(manifest.path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": 1, "source": source, "declarations": declarations}
    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix="catalog-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, sort_keys=True)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_catalog(manifest: PackManifest) -> PackCatalog | None:
    try:
        data = json.loads(catalog_path(manifest.path).read_text(encoding="utf-8"))
        if data["version"] != 1 or data["source"] != source_digest(manifest):
            return None
        return PackCatalog(data["declarations"], data["source"])
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return None
