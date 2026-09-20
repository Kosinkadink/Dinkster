"""Core value envelope model (hazard H2: nothing crosses an edge raw).

Every value produced or consumed by a node - including primitives - is a
``Value``: a registered type id, cheap interrogable metadata, a content
fingerprint (the basis of location-independent cache keys), and a payload
accessed through a transport.
"""

from __future__ import annotations

import copy
import hashlib
import math
import threading
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import ClassVar, Protocol, cast, runtime_checkable

TypeId = str
Fingerprint = str
type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]


def _validate_json_value(value: object, *, subject: str) -> None:
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{subject} must be finite JSON data")
        return
    if isinstance(value, list):
        for index, item in enumerate(cast("list[object]", value)):
            _validate_json_value(item, subject=f"{subject}[{index}]")
        return
    if isinstance(value, dict):
        items = cast("dict[object, object]", value)
        if not all(isinstance(key, str) for key in items):
            raise ValueError(f"{subject} must be JSON data")
        for key, item in items.items():
            _validate_json_value(item, subject=f"{subject}.{key}")
        return
    raise ValueError(f"{subject} must be JSON data")


@dataclass(frozen=True)
class CustomWidgetDescriptor:
    """Pack-defined input presentation transported as inert JSON data."""

    widget_type: str
    params: Mapping[str, JsonValue] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        if type(self.widget_type) is not str or not self.widget_type:
            raise ValueError("custom widget type must be a non-empty string")
        params = cast("object", self.params)
        if not isinstance(params, Mapping):
            raise ValueError("custom widget params must be a mapping")
        copied = copy.deepcopy(dict(cast("Mapping[str, JsonValue]", params)))
        if "type" in copied:
            raise ValueError("custom widget params must not contain 'type'")
        _validate_json_value(copied, subject="custom widget params")
        object.__setattr__(self, "params", copied)


RESOURCES_META_KEY = "resources"
"""Well-known ValueMeta entry: a mapping of resource kind -> concrete
instance id(s) - one (``{"gpu": "cuda:1"}``) or several for a value that
spans devices (``{"gpu": ["cuda:0", "cuda:1"]}``, the multigpu case). A
value that owns hardware (a loaded model knows which device(s) it lives on)
declares it here via its type's meta fn; the engine reads it to bind a
consuming node's abstract ``occupies`` kinds to concrete admission lanes,
all of them for a spanning value. Scheduling metadata only - never part of
the fingerprint, so residency never touches cache identity."""

RESOURCE_ID_META_KEY = "resourceId"
"""Well-known ValueMeta entry: stable identity of owner-resolved state
this value *references* (a ResourceHandle, a compat resident stub).

A value carrying this key is a reference, not the bytes: whoever owns the
resource accounts its cost, so holders (caches) must exclude such values
from their own footprint - counting them twice would make the governor
see the same gigabytes in two consumers. The key also makes references
addressable: a cache can drop every entry referencing an identity, which
is the invalidate-then-release step of reclaiming the owner's memory
(DESIGN 3.10). May appear anywhere in a value tree - always read it
through lists.py's traversal helpers, never only off the top envelope."""

RESOURCE_REFS_META_KEY = "resourceRefs"
"""Additional owner-resolved resource identities referenced by one value.

Resident composites use this alongside RESOURCE_ID_META_KEY when their
lifetime depends on more than one independently releasable resource. Read
both keys through lists.py's traversal helpers."""

RESOURCE_OWNER_META_KEY = "resourceOwner"
"""Well-known ValueMeta entry: the process-instance token of the worker
that HOLDS the referenced resource (stage 6, execution dispatch).

Stamped by the producing side only - the process where the live object
actually resides (a resident codec's stub_meta, a local ResourceHandle's
meta fn) - and preserved unchanged by every relay: an intermediate
process must never rewrite provenance it did not create. The token is
:func:`process_instance_token`, so it names one worker LIFETIME: a
restarted worker has a new token, and a stub carrying a dead token is
known-unresolvable without a round trip. Dispatch reads it to pin
consumers of resident state to the owning worker; cache admission reads
it to reject entries whose owner lifetime ended. Ephemeral routing
identity only - NEVER part of a fingerprint or cache key (the same
model reloaded by a new worker lifetime is the same computation)."""

RESOURCE_PRODUCER_ARM_META_KEY = "resourceProducerArm"
"""Well-known ValueMeta entry: the full composed dispatch arm that
produced a resident reference. Unlike the owner token, this distinguishes
multiple bodies sharing one worker session and residency domain."""

_process_token: str | None = None


def process_instance_token() -> str:
    """This process's lifetime identity: one uuid4 hex, minted lazily on
    first use and stable until the process exits. Worker children announce
    it in the hello handshake and stamp it on resident envelopes they
    produce (RESOURCE_OWNER_META_KEY); the host compares the two to decide
    whether a resident reference can still resolve. Deliberately lazy so a
    spawn-imported module in a fresh child mints its own token."""
    global _process_token
    if _process_token is None:
        import uuid

        _process_token = uuid.uuid4().hex
    return _process_token


def stable_hash(parts: Iterable[bytes | bytearray | memoryview]) -> str:
    """Order-sensitive, length-prefixed hash of byte parts (collision-safe framing)."""
    h = hashlib.blake2b(digest_size=20)
    for part in parts:
        h.update(len(part).to_bytes(8, "big"))
        h.update(part)
    return h.hexdigest()


@dataclass(frozen=True)
class ValueMeta:
    """Cheap facts about a value (shape, dtype, length, ...) - interrogable
    without touching the payload."""

    entries: Mapping[str, object] = field(default_factory=dict[str, object])

    def __post_init__(self) -> None:
        copied = dict(self.entries)
        resources = copied.get(RESOURCES_META_KEY)
        if isinstance(resources, Mapping):
            resource_entries = cast("Mapping[object, object]", resources)
            normalized: dict[str, object] = {}
            for key, value in resource_entries.items():
                normalized[str(key)] = (
                    tuple(cast("list[object]", value)) if isinstance(value, list) else value
                )
            copied[RESOURCES_META_KEY] = normalized
        # Keep a private copy rather than MappingProxyType: metadata must
        # remain JSON/pickle serializable at worker and cache boundaries.
        object.__setattr__(self, "entries", copied)

    def get(self, key: str, default: object = None) -> object:
        return self.entries.get(key, default)


@runtime_checkable
class Payload(Protocol):
    """How to actually get at the bytes/object. Which transport backs a value
    is a placement decision, never a node concern (hazard H9)."""

    @property
    def transport(self) -> str: ...  # read-only: frozen implementations satisfy it

    def load(self) -> object: ...


@dataclass(frozen=True)
class PyObjPayload:
    """Same-process fast path: zero overhead when nothing crosses a real boundary."""

    obj: object
    transport: str = "pyobj"

    def load(self) -> object:
        return self.obj


@dataclass(frozen=True)
class InlinePayload:
    """Encoded bytes carried with the envelope; decodes on demand."""

    data: bytes
    decoder: Callable[[bytes], object]
    transport: str = "inline"

    def load(self) -> object:
        return self.decoder(self.data)


class UnresolvablePayload(Exception):
    """A payload's object form cannot be produced in this process (its type
    is not registered here). The envelope stays inspectable and the encoded
    bytes stay relayable - only ``load()`` is off the table."""


class _EncodedBufferStorage:
    """Owned encoded buffer whose backing resource may outlive its sender."""

    def __init__(self, view: memoryview, size: int, release: Callable[[], None]) -> None:
        self._view: memoryview | None = view
        self._size = size
        self._release = release
        self._materialized: bytes | None = None
        self._borrows = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            if self._materialized is not None:
                return len(self._materialized)
            return self._size

    def materialize(self) -> bytes:
        with self._lock:
            if self._materialized is None:
                assert self._view is not None
                self._materialized = bytes(self._view[: self._size])
                self._close_mapping_if_unused()
            return self._materialized

    @contextmanager
    def borrow(self) -> Generator[memoryview]:
        with self._lock:
            if self._materialized is not None:
                view = memoryview(self._materialized)
                mapped = False
            else:
                assert self._view is not None
                view = self._view.toreadonly()
                if view.nbytes != self._size:
                    sized_view = view[: self._size]
                    view.release()
                    view = sized_view
                self._borrows += 1
                mapped = True
        try:
            yield view
        finally:
            view.release()
            if mapped:
                with self._lock:
                    self._borrows -= 1
                    self._close_mapping_if_unused()

    def _close_mapping_if_unused(self) -> None:
        if self._materialized is None or self._borrows or self._view is None:
            return
        self._view.release()
        self._view = None
        self._release()

    def close(self) -> None:
        with self._lock:
            if self._view is None:
                return
            self._view.release()
            self._view = None
            self._release()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


@dataclass(frozen=True)
class EncodedPayload:
    """A payload carried as its type codec's bytes - arrived over a process
    boundary or rehydrated from a persistent cache.

    Keeps the encoded form, either as owned bytes or as an owned read-only
    buffer mapping. Relaying borrows that form without a copy; ``data``
    materializes stable owned bytes only when a consumer explicitly needs
    them. A process that cannot decode the type can still carry, cache,
    interrogate, and forward the value - only ``load()`` needs the type
    registered where it runs (hazard H2).
    """

    type_id: str
    data: bytes
    decoder: Callable[[bytes], object] | None
    transport: str = "inline"
    _encoded: ClassVar[bytes | _EncodedBufferStorage]

    def __init__(
        self,
        type_id: str,
        data: bytes,
        decoder: Callable[[bytes], object] | None,
        transport: str = "inline",
    ) -> None:
        object.__setattr__(self, "type_id", type_id)
        object.__setattr__(self, "_encoded", data)
        object.__setattr__(self, "decoder", decoder)
        object.__setattr__(self, "transport", transport)

    @classmethod
    def from_buffer(
        cls,
        type_id: str,
        view: memoryview,
        decoder: Callable[[bytes], object] | None,
        transport: str,
        release: Callable[[], None],
        size: int | None = None,
    ) -> EncodedPayload:
        """Take ownership of ``view`` and release its backing store when unused."""
        logical_size = view.nbytes if size is None else size
        if logical_size < 0 or logical_size > view.nbytes:
            raise ValueError("encoded buffer size must fit its backing view")
        payload = cls(type_id, b"", decoder, transport)
        object.__setattr__(
            payload,
            "_encoded",
            _EncodedBufferStorage(view, logical_size, release),
        )
        return payload

    def __getattribute__(self, name: str) -> object:
        if name == "data":
            encoded = object.__getattribute__(self, "_encoded")
            return encoded if isinstance(encoded, bytes) else encoded.materialize()
        return object.__getattribute__(self, name)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            EncodedPayload,
            (self.type_id, self.data, self.decoder, self.transport),
        )

    @property
    def size(self) -> int:
        return len(self._encoded)

    @contextmanager
    def borrow_data(self) -> Generator[memoryview]:
        """Borrow a read-only encoded view, pinning its backing store."""
        encoded = self._encoded
        if isinstance(encoded, bytes):
            view = memoryview(encoded)
            try:
                yield view
            finally:
                view.release()
        else:
            with encoded.borrow() as view:
                yield view

    def restamped(self, type_id: str, decoder: Callable[[bytes], object] | None) -> EncodedPayload:
        """Share these codec bytes under an equivalent type stamp."""
        payload = EncodedPayload(type_id, b"", decoder, self.transport)
        object.__setattr__(payload, "_encoded", self._encoded)
        return payload

    def load(self) -> object:
        if self.decoder is None:
            raise UnresolvablePayload(
                f"value type '{self.type_id}' is not registered in this process; "
                "the payload can only be resolved where the type is registered"
            )
        return self.decoder(self.data)


@dataclass(frozen=True)
class Value:
    """The envelope. Node authors never construct these; worker shims do."""

    type_id: TypeId
    fingerprint: Fingerprint
    meta: ValueMeta
    payload: Payload

    def resolve(self) -> object:
        return self.payload.load()
