"""Worker-local out-of-tree patch-provider materialization.

Only an opaque catalog generation key crosses worker control frames. Import
recipes and callables remain in the torch worker, while RPC-clean keyed
declarations can participate in extension behavior identity.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from dinkster_inference import KeyedContribution, Registry

from .materialize import PatchProviderDescriptor

INFERENCE_PATCH_PROVIDERS_SURFACE = "inference.patch-providers"
PATCH_PROVIDER_CATALOG_ENV = "DINKSTER_PATCH_PROVIDER_CATALOG"
_CATALOG_FORMAT = "dinkster.patch-provider-catalog-v1"
_CACHE_LIMIT = 8


@dataclass(frozen=True)
class PatchProviderContribution:
    """Explicit result of one pack's worker-local patch entry point."""

    providers: tuple[PatchProviderDescriptor, ...]

    def __post_init__(self) -> None:
        providers = cast("object", self.providers)
        if not isinstance(providers, tuple) or not providers:
            raise TypeError("providers must be a non-empty tuple")
        if not all(
            isinstance(provider, PatchProviderDescriptor)
            for provider in cast("tuple[object, ...]", providers)
        ):
            raise TypeError("providers must contain PatchProviderDescriptor values")


@dataclass(frozen=True)
class PatchProviderExtensionEntry:
    """Process-local import recipe for one inference extension."""

    extension_id: str
    entry_point: str

    def __post_init__(self) -> None:
        if not self.extension_id:
            raise ValueError("extension_id must be non-empty")
        if (
            self.entry_point.count(":") != 1
            or not all(self.entry_point.split(":"))
            or any(char.isspace() for char in self.entry_point)
        ):
            raise ValueError("entry_point must be a 'module:attr' string")


@dataclass(frozen=True)
class PatchProviderSnapshot:
    """RPC-clean effective declarations in registry insertion order."""

    providers: tuple[KeyedContribution, ...] = ()

    def __post_init__(self) -> None:
        values = cast("object", self.providers)
        if not isinstance(values, tuple) or not all(
            isinstance(value, KeyedContribution) for value in cast("tuple[object, ...]", values)
        ):
            raise TypeError("providers must be a tuple of KeyedContribution values")
        ids = [provider.id for provider in self.providers]
        if len(ids) != len(set(ids)):
            raise ValueError("patch provider ids must be unique")
        if any(
            provider.surface_id != INFERENCE_PATCH_PROVIDERS_SURFACE for provider in self.providers
        ):
            raise ValueError("patch provider declaration uses the wrong surface")


@dataclass(frozen=True)
class MaterializedPatchProviders:
    """Callable registry plus exact declaration evidence."""

    registry: Registry[PatchProviderDescriptor]
    snapshot: PatchProviderSnapshot
    extensions: tuple[tuple[str, tuple[KeyedContribution, ...]], ...]

    @property
    def extension_ids(self) -> tuple[str, ...]:
        return tuple(extension_id for extension_id, _ in self.extensions)


def patch_provider_declaration(
    provider: PatchProviderDescriptor,
) -> KeyedContribution:
    return KeyedContribution(
        surface_id=INFERENCE_PATCH_PROVIDERS_SURFACE,
        id=provider.id,
        aliases=provider.aliases,
        behavior_metadata=provider.behavior_metadata,
    )


def _declaration_to_wire(declaration: KeyedContribution) -> dict[str, object]:
    return {
        "surfaceId": declaration.surface_id,
        "id": declaration.id,
        "aliases": list(declaration.aliases),
        "behaviorMetadata": [list(item) for item in declaration.behavior_metadata],
    }


def _declaration_from_wire(raw: object) -> KeyedContribution:
    if not isinstance(raw, Mapping):
        raise RuntimeError("patch provider declaration must be an object")
    body = cast("Mapping[str, object]", raw)
    aliases_raw = body.get("aliases", ())
    metadata_raw = body.get("behaviorMetadata", ())
    if not isinstance(aliases_raw, list) or not isinstance(metadata_raw, list):
        raise RuntimeError("patch provider declaration has malformed tuples")
    aliases = cast("list[object]", aliases_raw)
    metadata: list[tuple[str, str | int | bool | None]] = []
    for raw_item in cast("list[object]", metadata_raw):
        if not isinstance(raw_item, list):
            raise RuntimeError("patch provider behavior metadata is malformed")
        item = cast("list[object]", raw_item)
        if len(item) != 2 or not isinstance(item[0], str):
            raise RuntimeError("patch provider behavior metadata is malformed")
        value = item[1]
        if value is not None and type(value) not in (str, int, bool):
            raise RuntimeError("patch provider behavior metadata is not RPC-clean")
        metadata.append((item[0], cast("str | int | bool | None", value)))
    return KeyedContribution(
        surface_id=str(body.get("surfaceId", "")),
        id=str(body.get("id", "")),
        aliases=tuple(str(alias) for alias in aliases),
        behavior_metadata=tuple(metadata),
    )


def write_patch_provider_catalog(
    path: Path,
    key: str,
    entries: Sequence[PatchProviderExtensionEntry],
    expected: PatchProviderSnapshot | None = None,
) -> None:
    """Atomically append one immutable worker-local catalog generation."""
    if not key:
        raise ValueError("patch provider catalog key must be non-empty")
    normalized = tuple(sorted(entries, key=lambda entry: entry.extension_id))
    if len({entry.extension_id for entry in normalized}) != len(normalized):
        raise ValueError("patch provider extension ids must be unique")
    document: dict[str, object] = {"format": _CATALOG_FORMAT, "records": {}}
    if path.exists():
        loaded_raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
        if not isinstance(loaded_raw, dict):
            raise RuntimeError(f"invalid patch provider catalog at {path}")
        loaded = cast("dict[str, object]", loaded_raw)
        if loaded.get("format") != _CATALOG_FORMAT:
            raise RuntimeError(f"invalid patch provider catalog at {path}")
        document = loaded
    records_raw = document.get("records")
    if not isinstance(records_raw, dict):
        raise RuntimeError(f"invalid patch provider catalog records at {path}")
    records = cast("dict[str, object]", records_raw)
    record: dict[str, object] = {
        "extensions": [
            {"id": entry.extension_id, "entryPoint": entry.entry_point} for entry in normalized
        ]
    }
    if expected is not None:
        record["expected"] = [
            _declaration_to_wire(declaration) for declaration in expected.providers
        ]
    existing = records.get(key)
    if existing is not None and existing != record:
        raise RuntimeError(f"patch provider catalog key {key!r} already names different data")
    records[key] = record
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            document,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def remove_patch_provider_catalog_record(path: Path, key: str) -> None:
    """Atomically remove one transient staging record."""
    if not path.exists():
        return
    loaded_raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(loaded_raw, dict):
        raise RuntimeError(f"invalid patch provider catalog at {path}")
    loaded = cast("dict[str, object]", loaded_raw)
    if loaded.get("format") != _CATALOG_FORMAT:
        raise RuntimeError(f"invalid patch provider catalog at {path}")
    records_raw = loaded.get("records")
    if not isinstance(records_raw, dict):
        raise RuntimeError(f"invalid patch provider catalog records at {path}")
    records = cast("dict[str, object]", records_raw)
    if records.pop(key, None) is None:
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            loaded,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_record(key: str, path: Path) -> tuple[tuple[PatchProviderExtensionEntry, ...], object]:
    loaded_raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(loaded_raw, dict):
        raise RuntimeError(f"invalid patch provider catalog at {path}")
    loaded = cast("dict[str, object]", loaded_raw)
    if loaded.get("format") != _CATALOG_FORMAT:
        raise RuntimeError(f"invalid patch provider catalog at {path}")
    records_raw = loaded.get("records")
    if not isinstance(records_raw, dict):
        raise RuntimeError(f"patch provider catalog has no generation {key!r}")
    record_raw = cast("dict[str, object]", records_raw).get(key)
    if not isinstance(record_raw, dict):
        raise RuntimeError(f"patch provider catalog has no generation {key!r}")
    record = cast("dict[str, object]", record_raw)
    entries_raw = record.get("extensions")
    if not isinstance(entries_raw, list):
        raise RuntimeError(f"patch provider catalog generation {key!r} has no extensions")
    entries: list[PatchProviderExtensionEntry] = []
    for raw in cast("list[object]", entries_raw):
        if not isinstance(raw, dict):
            raise RuntimeError("patch provider extension entry must be an object")
        body = cast("dict[str, object]", raw)
        entries.append(
            PatchProviderExtensionEntry(
                extension_id=str(body.get("id", "")),
                entry_point=str(body.get("entryPoint", "")),
            )
        )
    return tuple(entries), record.get("expected")


def _resolve_contribution(
    entry: PatchProviderExtensionEntry,
) -> PatchProviderContribution:
    module_name, attr_name = entry.entry_point.split(":", 1)
    for loaded_name in tuple(sys.modules):
        if loaded_name == module_name or loaded_name.startswith(module_name + "."):
            del sys.modules[loaded_name]
    importlib.invalidate_caches()
    module = importlib.import_module(module_name)
    register = getattr(module, attr_name, None)
    if not callable(register):
        raise RuntimeError(
            f"patch entry {entry.entry_point!r} for {entry.extension_id!r} is not callable"
        )
    contribution = register()
    if not isinstance(contribution, PatchProviderContribution):
        raise RuntimeError(
            f"patch entry {entry.entry_point!r} for {entry.extension_id!r} "
            "must return PatchProviderContribution"
        )
    return contribution


_cache: OrderedDict[str, MaterializedPatchProviders] = OrderedDict()
_cache_lock = threading.RLock()


def materialize_patch_provider_registry(
    key: str, *, catalog_path: Path | None = None
) -> MaterializedPatchProviders:
    """Build and bidirectionally validate worker-local patch providers."""
    with _cache_lock:
        cacheable = not key.startswith("candidate:")
        if cacheable:
            cached = _cache.get(key)
            if cached is not None:
                _cache.move_to_end(key)
                return cached
        if catalog_path is None:
            raw_path = os.environ.get(PATCH_PROVIDER_CATALOG_ENV)
            if not raw_path:
                raise RuntimeError(f"${PATCH_PROVIDER_CATALOG_ENV} is not configured")
            catalog_path = Path(raw_path)
        entries, expected_raw = _read_record(key, catalog_path)
        registry: Registry[PatchProviderDescriptor] = Registry()
        extension_declarations: list[tuple[str, tuple[KeyedContribution, ...]]] = []
        for entry in entries:
            contribution = _resolve_contribution(entry)
            declarations = tuple(
                patch_provider_declaration(provider) for provider in contribution.providers
            )
            for provider in contribution.providers:
                registry.register(provider)
            extension_declarations.append((entry.extension_id, declarations))
        snapshot = PatchProviderSnapshot(
            tuple(patch_provider_declaration(provider) for provider in registry)
        )
        if expected_raw is not None:
            if not isinstance(expected_raw, list):
                raise RuntimeError("patch provider catalog expected declarations are malformed")
            expected = PatchProviderSnapshot(
                tuple(_declaration_from_wire(item) for item in cast("list[object]", expected_raw))
            )
            if snapshot != expected:
                produced = tuple(item.id for item in snapshot.providers)
                declared = tuple(item.id for item in expected.providers)
                raise RuntimeError(
                    "patch provider declaration mismatch: "
                    f"worker produced {produced}, snapshot declared {declared}"
                )
        materialized = MaterializedPatchProviders(
            registry=registry,
            snapshot=snapshot,
            extensions=tuple(extension_declarations),
        )
        if cacheable:
            _cache[key] = materialized
            while len(_cache) > _CACHE_LIMIT:
                _cache.popitem(last=False)
        return materialized


__all__ = [
    "INFERENCE_PATCH_PROVIDERS_SURFACE",
    "PATCH_PROVIDER_CATALOG_ENV",
    "MaterializedPatchProviders",
    "PatchProviderContribution",
    "PatchProviderExtensionEntry",
    "PatchProviderSnapshot",
    "materialize_patch_provider_registry",
    "patch_provider_declaration",
    "remove_patch_provider_catalog_record",
    "write_patch_provider_catalog",
]
