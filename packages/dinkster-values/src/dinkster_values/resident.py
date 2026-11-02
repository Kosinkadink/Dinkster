"""Process-resident values: the objects never cross the boundary, stubs do.

A loaded model, text encoder, or codec is gigabytes of device state -
encoding it with the default pickle codec would be a catastrophe (huge
transfer, and unpicklable on a torch-less parent). Instead, resident types
register a codec that encodes a *resident id*: the object stays in the
holding process's residency table, other processes carry an interrogable
stub, and when the stub comes back as a later invocation's input the codec
resolves it from the table.

This is the engine-level ResourceHandle's identity-crosses/content-stays
shape, implemented entirely with the public TypeRegistry codec hooks - any
pack or package can register a resident type without touching engine or
worker code. One limit is deliberate:

- fingerprints are per-process resident ids, so cross-run caches will not
  hit on resident values (the loader's own cache key still hits because it
  is built from the loader's inputs).

A bare ResidencyTable holds strong references until the process exits. A
production residency pool may coordinate reference invalidation and remove
an owner together with all resident values that depend on it.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping
from typing import cast

from .model import (
    RESOURCE_ID_META_KEY,
    RESOURCE_OWNER_META_KEY,
    RESOURCE_REFS_META_KEY,
    RESOURCES_META_KEY,
    process_instance_token,
)
from .registry import TypeRegistry, TypeSpec
from .resources import COST_META_KEY


def comfy_resident_meta(obj: object) -> Mapping[str, object]:
    """Read execution residency and cost from a patcher-shaped resident value."""
    resident_cost = getattr(obj, "_dinkster_resident_cost", None)
    if resident_cost is not None:
        if not isinstance(resident_cost, Mapping):
            raise TypeError("_dinkster_resident_cost must map resource names to nonnegative ints")
        raw_cost = cast("Mapping[object, object]", resident_cost)
        if any(
            type(resource) is not str or not resource or type(cost) is not int or cost < 0
            for resource, cost in raw_cost.items()
        ):
            raise TypeError("_dinkster_resident_cost must map resource names to nonnegative ints")
        return {COST_META_KEY: cast("dict[str, int]", dict(raw_cost))}
    patcher = _patcher_of(obj)
    if patcher is None:
        return {}
    devices = _device_strings(patcher)
    nbytes = _model_nbytes(patcher)
    if not devices:
        return {}
    meta: dict[str, object] = {
        RESOURCES_META_KEY: {"gpu": devices[0] if len(devices) == 1 else devices}
    }
    if nbytes > 0:
        # The host offload copy persists; split execution cost across the devices
        # until patchers expose per-device layouts.
        share = nbytes // len(devices)
        cost: dict[str, int] = {f"vram:{device}": share for device in devices}
        cost["ram"] = nbytes
        meta[COST_META_KEY] = cost
    return meta


def _patcher_of(obj: object) -> object | None:
    candidate = getattr(obj, "patcher", obj)
    if hasattr(candidate, "load_device"):
        return candidate
    return None


def _device_strings(patcher: object) -> tuple[str, ...]:
    raw = getattr(patcher, "load_devices", None) or [getattr(patcher, "load_device", None)]
    devices: list[str] = []
    for device in cast("list[object]", list(raw)):
        if device is None:
            continue
        text = str(device)
        if text.startswith(("cuda", "xpu", "mps", "npu")):
            devices.append(text)
    return tuple(devices)


def _model_nbytes(patcher: object) -> int:
    size_fn = getattr(patcher, "model_size", None)
    if not callable(size_fn):
        return 0
    try:
        size = size_fn()
    except Exception:  # noqa: BLE001 - foreign code; no meta beats a crash
        return 0
    return int(size) if isinstance(size, (int, float)) else 0


class ResidentLookupError(Exception):
    """A resident stub arrived in a process that does not hold the value."""


def resident_resource_id(rid: str) -> str:
    """The resident's identity as holders see it: the codec fingerprint,
    the envelope's RESOURCE_ID_META_KEY, and the ID caches invalidate by
    are all this one string."""
    return "resident:" + rid


class ResidencyTable:
    """One per holding process: resident id -> live object.

    Idempotent per object (same object, same id), so fingerprinting and
    payload encoding agree, and re-sending the same value re-uses the id.
    """

    def __init__(self) -> None:
        self._objects: dict[str, object] = {}
        self._ids_by_identity: dict[int, str] = {}

    def rid_for(self, obj: object) -> str:
        rid = self._ids_by_identity.get(id(obj))
        if rid is None or self._objects.get(rid) is not obj:
            rid = uuid.uuid4().hex
            self._objects[rid] = obj
            self._ids_by_identity[id(obj)] = rid
        return rid

    def get(self, rid: str) -> object:
        try:
            return self._objects[rid]
        except KeyError:
            raise ResidentLookupError(
                f"resident value {rid!r} is not held by this process; it "
                "belongs to another worker or an earlier worker lifetime"
            ) from None

    def remove(self, rid: str) -> object | None:
        """Drop the table's strong reference; None when already gone.

        Only for coordinated release (invalidate references first, then
        remove): a stub for a removed resident raises ResidentLookupError
        from then on.
        """
        obj = self._objects.pop(rid, None)
        if obj is not None:
            self._ids_by_identity.pop(id(obj), None)
        return obj

    def __len__(self) -> int:
        return len(self._objects)


_TABLE = ResidencyTable()
"""Module-level: the holding process's single default residency table."""


def resident_owners(obj: object) -> tuple[object, ...]:
    """The primary resident owner followed by additional referenced owners."""
    primary = cast("object", getattr(obj, "_dinkster_resident_owner", obj))
    refs_value = cast("object", getattr(obj, "_dinkster_resident_refs", ()))
    if not isinstance(refs_value, tuple):
        raise TypeError("_dinkster_resident_refs must be a tuple")
    refs = cast("tuple[object, ...]", refs_value)
    owners: list[object] = []
    identities: set[int] = set()
    for owner in (primary, *refs):
        if owner is None:
            raise TypeError("resident owners must not be None")
        if id(owner) in identities:
            continue
        identities.add(id(owner))
        owners.append(owner)
    return tuple(owners)


class ResidentCodec:
    """Reusable codec functions for a type that admits resident values."""

    def __init__(
        self,
        table: ResidencyTable | None = None,
        meta: Callable[[object], Mapping[str, object]] | None = None,
    ) -> None:
        self._table = table if table is not None else _TABLE
        self._meta = meta

    def metadata(self, obj: object) -> Mapping[str, object]:
        owners = resident_owners(obj)
        owner = owners[0]
        enriched: dict[str, object] = dict(self._meta(owner)) if self._meta is not None else {}
        enriched[RESOURCE_ID_META_KEY] = resident_resource_id(self._table.rid_for(owner))
        additional = tuple(
            resident_resource_id(self._table.rid_for(reference)) for reference in owners[1:]
        )
        if additional:
            enriched[RESOURCE_REFS_META_KEY] = additional
        enriched[RESOURCE_OWNER_META_KEY] = process_instance_token()
        return enriched

    def encode(self, obj: object) -> bytes:
        return json.dumps({"residentId": self._table.rid_for(obj)}).encode("utf-8")

    def decode(self, data: bytes) -> object:
        wire = cast("dict[str, object]", json.loads(data))
        rid = wire.get("residentId")
        if not isinstance(rid, str):
            raise ResidentLookupError(f"malformed resident stub: {data!r}")
        return self._table.get(rid)

    def validate_encoded(self, data: bytes) -> None:
        try:
            parsed: object = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ResidentLookupError(f"malformed resident stub: {data!r}") from None
        if not isinstance(parsed, dict):
            raise ResidentLookupError(f"malformed resident stub: {data!r}")
        wire = cast("dict[object, object]", parsed)
        if set(wire) != {"residentId"}:
            raise ResidentLookupError(f"malformed resident stub: {data!r}")
        if not isinstance(wire["residentId"], str) or not wire["residentId"]:
            raise ResidentLookupError(f"malformed resident stub: {data!r}")

    def fingerprint(self, obj: object) -> str:
        declared = getattr(obj, "_dinkster_resident_fingerprint", None)
        if declared is not None:
            if not isinstance(declared, str) or not declared:
                raise TypeError("_dinkster_resident_fingerprint must be a non-empty string")
            return declared
        return resident_resource_id(self._table.rid_for(obj))


def register_resident_type(
    registry: TypeRegistry,
    type_id: str,
    table: ResidencyTable | None = None,
    meta: Callable[[object], Mapping[str, object]] | None = None,
    *,
    coerce: Callable[[object], object] | None = None,
) -> TypeSpec:
    """``meta`` publishes residency/cost for the live object (device lanes,
    VRAM cost) - it rides the envelope across the boundary even though the
    object itself never does. Every resident envelope additionally carries
    RESOURCE_ID_META_KEY: the value is a *reference* to owner-resolved
    state, so holders exclude its cost from their accounting and can be
    asked to drop entries referencing it (invalidate-then-release)."""
    codec = ResidentCodec(table, meta)

    spec = registry.register(
        type_id,
        encode=codec.encode,
        decode=codec.decode,
        fingerprint=codec.fingerprint,
        meta=codec.metadata,
        coerce=coerce,
    )
    registry.register_resident_table(table if table is not None else _TABLE)
    return spec


__all__ = [
    "ResidencyTable",
    "ResidentCodec",
    "ResidentLookupError",
    "register_resident_type",
    "resident_owners",
    "resident_resource_id",
]
