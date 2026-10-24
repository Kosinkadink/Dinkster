"""Native resident pool with Comfy metadata and VAE provenance."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from dinkster_assets import AssetError, require_digest, validate_vae_hint_field
from dinkster_memory.resident import ResidentPool as GenericResidentPool

from .devices import comfy_resident_meta


def _source_text(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    return validate_vae_hint_field(field, value)


@dataclass(frozen=True, slots=True)
class VaeSource:
    digest: str
    name: str | None = None
    logical_id: str | None = None
    latent_space: str | None = None

    def hint(self) -> str:
        value: dict[str, object] = {"sourceDigest": self.digest, "version": 1}
        if self.name is not None:
            value["sourceName"] = self.name
        if self.logical_id is not None:
            value["sourceLogicalId"] = self.logical_id
        if self.latent_space is not None:
            value["latentSpace"] = self.latent_space
        return json.dumps(value, sort_keys=True, separators=(",", ":"))


class ResidentPool(GenericResidentPool):
    """Generic residency mechanics plus native source metadata."""

    def __init__(
        self,
        *,
        cost_of: Callable[[object], Mapping[str, object]] = comfy_resident_meta,
        unload: Callable[[object], None] | None = None,
        terminal_release: Callable[[object], bool] | None = None,
    ) -> None:
        super().__init__(
            cost_of=cost_of,
            unload=unload,
            terminal_release=terminal_release,
        )
        self._sources: dict[str, VaeSource] = {}

    def label_source(
        self,
        obj: object,
        *,
        digest: str,
        name: str | None = None,
        logical_id: str | None = None,
        latent_space: str | None = None,
    ) -> None:
        """Attach path-free VAE lineage without admitting a resident."""
        source = VaeSource(
            require_digest(digest),
            _source_text(name, "sourceName"),
            _source_text(logical_id, "sourceLogicalId"),
            _source_text(latent_space, "latentSpace"),
        )
        with self._coordination_lock:
            rid = self._ids_by_identity.get(id(obj))
            if rid is None or self._objects.get(rid) is not obj or rid not in self._entries:
                raise AssetError("VAE source can only label an admitted resident")
            current = self._sources.get(rid)
            if current is not None and current.digest != source.digest:
                raise AssetError("conflicting VAE source digest for one resident")
            if current is not None:
                source = VaeSource(
                    source.digest,
                    current.name or source.name,
                    current.logical_id or source.logical_id,
                    current.latent_space or source.latent_space,
                )
            self._sources[rid] = source

    def source_for(self, obj: object) -> VaeSource | None:
        """Observe VAE lineage without admission or LRU use tracking."""
        with self._coordination_lock:
            rid = self._ids_by_identity.get(id(obj))
            if rid is None or self._objects.get(rid) is not obj:
                return None
            return self._sources.get(rid)

    def _removed(self, rid: str, obj: object | None) -> None:
        self._sources.pop(rid, None)


_pool: ResidentPool | None = None
_compat_unload: Callable[[object], None] | None = None


def configure_compat_unload(unload: Callable[[object], None]) -> None:
    """Install the optional compatibility arm's foreign-model unload hook."""
    global _compat_unload  # noqa: PLW0603 - process-wide provider configuration
    _compat_unload = unload


def default_pool() -> ResidentPool:
    """The child process's single native pool."""
    global _pool  # noqa: PLW0603 - module-level singleton, child-process scoped
    if _pool is None:
        _pool = ResidentPool(unload=resident_advisory_unload)
    return _pool


def memory_consumers() -> dict[str, ResidentPool]:
    """Return the native pack's governed resident pool."""
    return {"comfy-models": default_pool()}


def resident_advisory_unload(obj: object) -> None:
    """Dispatch advisory VRAM pressure to the configured Comfy backend."""
    patcher = getattr(obj, "patcher", obj)
    if not hasattr(patcher, "load_device"):
        return
    if _compat_unload is None:
        raise TypeError("foreign resident requires a configured compatibility unload hook")
    _compat_unload(obj)
