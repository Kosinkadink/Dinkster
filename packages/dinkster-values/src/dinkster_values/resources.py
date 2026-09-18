"""ResourceHandle: envelope values for things that cannot cross a wire raw.

Models/VAE/CLIP are not data - they are loaded hardware state. A
ResourceHandle is the envelope-shaped truth about one: a stable identity
(the basis of cache keys), residency (which concrete devices hold it),
cost (bytes by residency class, what the MemoryGovernor accounts), and -
only where the resource actually lives - the runtime object itself.

Crossing a boundary carries identity + residency + cost and drops the
object: the receiving side gets an interrogable, schedulable handle it
cannot resolve. Placement routes consumers to the owner (or consciously
transfers); a worker that hits ``require_obj()`` on a non-local handle
has been mis-scheduled, and the error says so.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

# RESOURCE_ID_META_KEY's canonical home is model.py (the traversal helpers
# in lists.py need it below the registry layer); re-exported here where the
# resource docs live.
from .model import (
    RESOURCE_ID_META_KEY,
    RESOURCE_OWNER_META_KEY,
    RESOURCES_META_KEY,
    process_instance_token,
)
from .registry import TypeRegistry

COST_META_KEY = "cost"
"""Well-known ValueMeta entry: mapping of residency class -> bytes.

Residency classes are ``"ram"``, ``"disk"``, and ``"vram:<device>"``
(device-qualified, because budgets are per device: ``"vram:cuda:0"``).
Cost is scheduling/accounting metadata only - never part of the
fingerprint, for the same reason residency is not: the same computation
on different hardware is the same computation.
"""

RESOURCE_HANDLE_TYPE = "dinkster.resource"


def unqualified_cost(cost: object) -> dict[str, int]:
    """Compare codec byte counts independently of remote residency qualifiers."""
    if not isinstance(cost, Mapping):
        raise ValueError("cost must be a residency mapping")
    result: dict[str, int] = {}
    for residency, nbytes in cast("Mapping[object, object]", cost).items():
        if not isinstance(residency, str) or type(nbytes) is not int or nbytes < 0:
            raise ValueError("cost requires residency names and nonnegative integer bytes")
        name = residency.partition("@")[0]
        if name in result:
            raise ValueError("cost contains duplicate unqualified residencies")
        result[name] = nbytes
    return result


class ResourceError(Exception):
    """A resource handle was used somewhere its resource does not live."""


@dataclass(frozen=True)
class ResourceHandle:
    """Identity + residency + cost for a non-serializable resource.

    resource_id is the stable identity: derive it from content identity
    plus load parameters (e.g. ``blake3:<hex>?dtype=fp16``), never from
    id(), paths, or load order - it is the fingerprint, so it decides
    cache hits.

    residency uses the RESOURCES_META_KEY shape: resource kind -> concrete
    instance id(s), one (``{"gpu": "cuda:1"}``) or a tuple for a resource
    spanning devices (``{"gpu": ("cuda:0", "cuda:1")}``, the multigpu
    case).

    cost uses the COST_META_KEY shape: residency class -> bytes.
    """

    resource_id: str
    kind: str
    residency: Mapping[str, str | tuple[str, ...]] = field(
        default_factory=dict[str, str | tuple[str, ...]]
    )
    cost: Mapping[str, int] = field(default_factory=dict[str, int])
    obj: object | None = None
    owner: str | None = None
    """Producer-stamped owner provenance for a NON-LOCAL handle: the
    process-instance token of the worker holding the object, carried so a
    relay re-wrapping this handle preserves provenance it did not create
    (RESOURCE_OWNER_META_KEY). Ignored when the handle is local - the
    holding process always stamps its own live token, never a remembered
    one. None means the producer predates owner stamping."""

    def __post_init__(self) -> None:
        residency = {
            key: tuple(value) if not isinstance(value, str) else value
            for key, value in self.residency.items()
        }
        # Private copies preserve boundary serialization (mapping proxies
        # are neither JSON serializable nor pickleable).
        object.__setattr__(self, "residency", residency)
        object.__setattr__(self, "cost", dict(self.cost))

    @property
    def is_local(self) -> bool:
        return self.obj is not None

    def require_obj(self) -> object:
        if self.obj is None:
            raise ResourceError(
                f"resource {self.resource_id!r} ({self.kind}) is not local to "
                "this worker; the scheduler must route its consumers to the "
                "owner or transfer it explicitly"
            )
        return self.obj


def _encode_handle(obj: object) -> bytes:
    handle = _require_handle(obj)
    wire = {
        "resourceId": handle.resource_id,
        "kind": handle.kind,
        "residency": {
            k: list(v) if isinstance(v, tuple) else v for k, v in handle.residency.items()
        },
        "cost": dict(handle.cost),
    }
    # Owner provenance crosses the wire so the receiving side can keep
    # relaying it: a local handle is owned HERE (stamp this lifetime's
    # token), a non-local one keeps whatever its producer stamped.
    owner = process_instance_token() if handle.is_local else handle.owner
    if owner is not None:
        wire["owner"] = owner
    return json.dumps(wire, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _decode_handle(data: bytes) -> object:
    wire = cast(Mapping[str, object], json.loads(data))
    residency_raw = cast(Mapping[str, object], wire.get("residency") or {})
    residency: dict[str, str | tuple[str, ...]] = {}
    for k, v in residency_raw.items():
        if isinstance(v, list):
            residency[k] = tuple(cast(list[str], v))
        else:
            residency[k] = cast(str, v)
    cost = {
        k: int(cast(int, v)) for k, v in cast(Mapping[str, object], wire.get("cost") or {}).items()
    }
    owner_raw = wire.get("owner")
    return ResourceHandle(
        resource_id=cast(str, wire["resourceId"]),
        kind=cast(str, wire["kind"]),
        residency=residency,
        cost=cost,
        obj=None,  # crossing the wire drops locality by construction
        # Carried, not minted: whatever the producing side stamped rides
        # along so a further relay preserves it (malformed means absent).
        owner=owner_raw if isinstance(owner_raw, str) and owner_raw else None,
    )


def _require_handle(obj: object) -> ResourceHandle:
    if not isinstance(obj, ResourceHandle):
        raise TypeError(
            f"{RESOURCE_HANDLE_TYPE} values must be ResourceHandle, got {type(obj).__name__}"
        )
    return obj


def _handle_meta(obj: object) -> Mapping[str, object]:
    handle = _require_handle(obj)
    meta: dict[str, object] = {
        RESOURCE_ID_META_KEY: handle.resource_id,
        "kind": handle.kind,
    }
    if handle.is_local:
        # Owner provenance is stamped where the object actually lives
        # (RESOURCE_OWNER_META_KEY): the holding process stamps its own
        # live token, never a remembered one.
        meta[RESOURCE_OWNER_META_KEY] = process_instance_token()
    elif handle.owner is not None:
        # A decoded non-local handle relaying through this process keeps
        # whatever owner its producer stamped - rewriting provenance we
        # did not create would misroute dispatch.
        meta[RESOURCE_OWNER_META_KEY] = handle.owner
    if handle.residency:
        meta[RESOURCES_META_KEY] = dict(handle.residency)
    if handle.cost:
        meta[COST_META_KEY] = dict(handle.cost)
    return meta


def _handle_fingerprint(obj: object) -> str:
    # Identity is the declared resource_id alone: residency and cost are
    # scheduling facts, so the same model on another device stays a hit.
    return _require_handle(obj).resource_id


def register_resource_handle_type(registry: TypeRegistry) -> None:
    registry.register(
        RESOURCE_HANDLE_TYPE,
        encode=_encode_handle,
        decode=_decode_handle,
        fingerprint=_handle_fingerprint,
        meta=_handle_meta,
    )
