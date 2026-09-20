from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import httpx
from dinkster_api.v1 import (
    SCHEMA_WIRE_VERSION,
    NodeSchema,
    pack_logger,
    schema_from_wire,
    schema_signature,
)

_CACHE_FORMAT = "dinkster.remote.catalog-cache/1"

log = pack_logger("dinkster-nodes-remote")


class CatalogError(ValueError):
    pass


@dataclass(frozen=True)
class RemoteSchema:
    schema: NodeSchema
    signature: str


@dataclass(frozen=True)
class CatalogSnapshot:
    epoch: int
    schema_wire: int
    schemas: tuple[RemoteSchema, ...]
    payload: Mapping[str, object]

    @classmethod
    def empty(cls) -> CatalogSnapshot:
        return cls(0, SCHEMA_WIRE_VERSION, (), {})


def _mapping(value: object, subject: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CatalogError(f"{subject} must be an object")
    untyped = cast("Mapping[object, object]", value)
    if not all(isinstance(key, str) for key in untyped):
        raise CatalogError(f"{subject} must be an object")
    return cast("Mapping[str, object]", untyped)


def _remote_schema(node_type: str, value: object, schema_wire: int) -> RemoteSchema:
    entry = _mapping(value, f"catalog node {node_type!r}")
    if entry.get("nodeType") != node_type:
        raise CatalogError("nodeType does not match its catalog key")
    latest = entry.get("latestVersion")
    if type(latest) is not int or latest < 1:
        raise CatalogError("latestVersion must be a positive integer")
    versions = _mapping(entry.get("schemaVersions"), "schemaVersions")
    selected = _mapping(versions.get(str(latest)), f"schemaVersions[{latest}]")
    wire = _mapping(selected.get("schema"), "schema")
    if wire.get("schemaVersion") != schema_wire:
        raise CatalogError("schema entry does not use the catalog's negotiated wire version")
    signature = selected.get("signature")
    if not isinstance(signature, str) or not signature:
        raise CatalogError("schema version requires a signature")
    try:
        schema = schema_from_wire(dict(wire))
    except (KeyError, OverflowError, TypeError, ValueError) as exc:
        raise CatalogError(f"schema wire is invalid: {exc}") from exc
    if schema.node_type != node_type:
        raise CatalogError("decoded schema nodeType does not match its catalog key")
    if not node_type.startswith("dinkster.remote."):
        raise CatalogError("nodeType is outside the reserved dinkster.remote namespace")
    if schema.version != latest:
        raise CatalogError("decoded schema version does not match latestVersion")
    if schema_signature(schema) != signature:
        raise CatalogError("schema signature does not match decoded schema")
    if not schema.io_bound:
        raise CatalogError("remote schema must set ioBound=true")
    if schema.idempotent:
        raise CatalogError("remote schema must set idempotent=false")
    if schema.occupies:
        raise CatalogError("remote schema must not occupy local resources")
    return RemoteSchema(schema, signature)


def parse_catalog(payload: object) -> CatalogSnapshot:
    document = _mapping(payload, "catalog")
    epoch = document.get("catalogEpoch")
    if type(epoch) is not int or epoch < 0:
        raise CatalogError("catalogEpoch must be a non-negative integer")
    schema_wire = document.get("schemaWire")
    if type(schema_wire) is not int or schema_wire != SCHEMA_WIRE_VERSION:
        raise CatalogError(f"catalog selected unsupported schema wire {schema_wire!r}")
    nodes = _mapping(document.get("nodes"), "catalog nodes")
    accepted: list[RemoteSchema] = []
    for node_type, entry in nodes.items():
        try:
            accepted.append(_remote_schema(node_type, entry, schema_wire))
        except CatalogError as exc:
            log.warning("skipping malformed remote catalog entry %r: %s", node_type, exc)
    return CatalogSnapshot(
        epoch=epoch,
        schema_wire=schema_wire,
        schemas=tuple(accepted),
        payload=dict(document),
    )


@dataclass(frozen=True)
class CacheRecord:
    snapshot: CatalogSnapshot
    catalog_etag: str | None


class CatalogCache:
    def __init__(self, path: Path | None) -> None:
        self.path = path

    def load(self) -> CacheRecord | None:
        if self.path is None:
            return None
        try:
            raw: object = json.loads(self.path.read_text(encoding="utf-8"))
            document = _mapping(raw, "remote catalog cache")
            if document.get("format") != _CACHE_FORMAT:
                raise CatalogError("unsupported cache format")
            snapshot = parse_catalog(document.get("payload"))
            etag = document.get("catalogEtag")
            if etag is not None and not isinstance(etag, str):
                raise CatalogError("catalogEtag must be a string")
            return CacheRecord(snapshot, etag)
        except FileNotFoundError:
            return None
        except (CatalogError, OSError, ValueError) as exc:
            log.warning("ignoring invalid remote catalog cache %s: %s", self.path, exc)
            return None

    def store(self, snapshot: CatalogSnapshot, catalog_etag: str | None) -> None:
        if self.path is None:
            return
        document = {
            "format": _CACHE_FORMAT,
            "catalogEtag": catalog_etag,
            "payload": snapshot.payload,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(document, handle, separators=(",", ":"), sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)


class CatalogClient:
    def __init__(
        self,
        base_url: str,
        cache_path: Path | None,
        *,
        timeout: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.cache = CatalogCache(cache_path)
        cached = self.cache.load()
        self.snapshot = cached.snapshot if cached is not None else CatalogSnapshot.empty()
        self.catalog_etag = cached.catalog_etag if cached is not None else None
        self.epoch_etag: str | None = None
        self.observed_epoch = self.snapshot.epoch

    @property
    def nodes_url(self) -> str:
        return f"{self.base_url}/catalog/nodes"

    @property
    def epoch_url(self) -> str:
        return f"{self.base_url}/catalog/epoch"

    def _headers(self, *, conditional: bool) -> dict[str, str]:
        if conditional and self.catalog_etag:
            return {"If-None-Match": self.catalog_etag}
        return {}

    @staticmethod
    def _response_payload(response: httpx.Response) -> object:
        try:
            return response.json()
        except ValueError as exc:
            raise CatalogError("catalog response is not JSON") from exc

    def _accept(self, response: httpx.Response) -> CatalogSnapshot:
        snapshot = parse_catalog(self._response_payload(response))
        if self.snapshot.payload:
            if snapshot.epoch < self.snapshot.epoch:
                raise CatalogError(
                    f"catalog epoch moved backward from {self.snapshot.epoch} to {snapshot.epoch}"
                )
            if snapshot.epoch == self.snapshot.epoch and snapshot.payload != self.snapshot.payload:
                raise CatalogError(
                    f"catalog payload changed without advancing epoch {snapshot.epoch}"
                )
        self.catalog_etag = response.headers.get("ETag")
        self.snapshot = snapshot
        self.observed_epoch = snapshot.epoch
        try:
            self.cache.store(snapshot, self.catalog_etag)
        except OSError as exc:
            log.warning("could not persist the remote catalog cache: %s", exc)
        return snapshot

    def startup(self) -> CatalogSnapshot:
        if not self.base_url:
            return self.snapshot
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(
                    self.nodes_url,
                    headers=self._headers(conditional=True),
                )
            if response.status_code == 304:
                return self.snapshot
            if response.status_code == 406:
                raise CatalogError("remote catalog has no compatible NodeSchema wire version")
            response.raise_for_status()
            return self._accept(response)
        except (CatalogError, httpx.HTTPError, OSError) as exc:
            fallback = "last-good cache" if self.snapshot.schemas else "an empty remote pack"
            log.warning("remote catalog startup fetch failed; using %s: %s", fallback, exc)
            return self.snapshot

    async def refresh(self, *, conditional: bool) -> CatalogSnapshot | None:
        if not self.base_url:
            return None
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    self.nodes_url,
                    headers=self._headers(conditional=conditional),
                )
            if response.status_code == 304:
                return self.snapshot
            if response.status_code == 406:
                raise CatalogError("remote catalog has no compatible NodeSchema wire version")
            response.raise_for_status()
            return self._accept(response)
        except (CatalogError, httpx.HTTPError, OSError) as exc:
            log.warning("remote catalog refresh failed; keeping the current schemas: %s", exc)
            return None

    async def changed(self, announced_epoch: int) -> bool:
        if not self.base_url:
            return False
        headers = {"If-None-Match": self.epoch_etag} if self.epoch_etag else {}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.epoch_url, headers=headers)
            if response.status_code == 304:
                return self.observed_epoch != announced_epoch
            response.raise_for_status()
            document = _mapping(self._response_payload(response), "catalog epoch")
            epoch = document.get("catalogEpoch")
            if type(epoch) is not int or epoch < 0:
                raise CatalogError("catalogEpoch must be a non-negative integer")
            self.epoch_etag = response.headers.get("ETag")
            self.observed_epoch = epoch
            return epoch != announced_epoch
        except (CatalogError, httpx.HTTPError, OSError) as exc:
            log.warning("remote catalog epoch probe failed: %s", exc)
            return False


def cache_path_from_env(environment: Mapping[str, str]) -> Path | None:
    scratch = environment.get("DINKSTER_PACK_SCRATCH", "")
    return Path(scratch) / "catalog.json" if scratch else None


__all__ = [
    "CatalogClient",
    "CatalogError",
    "CatalogSnapshot",
    "RemoteSchema",
    "cache_path_from_env",
    "parse_catalog",
]
