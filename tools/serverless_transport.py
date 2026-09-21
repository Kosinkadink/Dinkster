"""EXPERIMENTAL on-demand Worker, not a negotiated resident-worker transport.

OnDemandWorker.prepare and schemas use a local, deployment-pinned catalog;
only invoke calls the provider-neutral async dispatch callback. Construct an
InvocationHandler around an InProcessWorker in the provider. Each endpoint
has its own Transport/cache; only the ObjectStore is shared. Every boundary
blob (including metadata and scalars) crosses storage, never provider RPC.

The subset is static-schema, ordinary InvocationResult nodes with core scalar,
image/mask and latent values (and recursive lists). Fingerprints, value metadata
and NodeErrors survive. Resident/model/CLIP/VAE/resource handles, arbitrary
types, lazy/dynamic interfaces, executor/arm selection, session authorities,
continuations, artifacts and events are refused. Whole generation may execute
inside one node, keeping model state internal and returning only data.

Source identity must name the immutable deployment (source and dependencies),
verified by the caller when deploying; equality here is not remote attestation.
Store, codecs and provider are in the same authenticated trust domain. Digest
checks ensure byte integrity, not authorization. Providers must enforce the
same envelope cap before parsing RPC and bounded reads in ObjectStore.get.
Limits bound encoded transfers, not codec allocations or decoded tensor RAM.

There is no retry, scheduling, durable deduplication, cancellation acknowledgement
or result replay. A dispatch failure or cancellation may leave work executing;
do not resubmit automatically, even with the same job/attempt/invocation IDs.
Disk stores do not evict; their lifetime and disk quota belong to the caller.
Cloud SDKs and account configuration belong to dinkster-execution, not this module.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, cast

from dinkster_assets import digest_bytes, require_digest
from dinkster_caches import DiskCAS
from dinkster_protocol import Invocation, InvocationEvent, InvocationResult, OnInvocationEvent
from dinkster_schema import SCHEMA_WIRE_VERSION, NodeSchema, schema_to_wire
from dinkster_values import TypeRegistry, Value, default_decode, list_children, parse_list_type_id
from dinkster_workers import InProcessWorker
from dinkster_workers.boundary import (
    PROTOCOL_VERSION,
    BoundaryError,
    ValueCodec,
    decode_invocation,
    decode_result_outputs,
    encode_invocation,
    encode_result,
    error_from_wire,
)

Envelope = dict[str, Any]
Dispatch = Callable[[Envelope], Awaitable[Envelope]]
_VERSION = 1
_DATA_TYPES = frozenset(
    {
        "core.int",
        "core.float",
        "core.boolean",
        "core.string",
        "core.combo",
        "dinkster.image",
        "dinkster.mask",
        "dinkster.latent",
        "comfy.IMAGE",
        "comfy.MASK",
        "comfy.LATENT",
    }
)
_RESIDENT_META = frozenset(
    {"resourceId", "resourceRefs", "resourceOwner", "resourceProducerArm", "resources"}
)
_INVOKE_FIELDS = frozenset(
    {
        "type",
        "invocationId",
        "jobRef",
        "attemptId",
        "nodeId",
        "nodeType",
        "effectiveSchema",
        "outputMembers",
        "inputs",
        "fp8Matmul",
        "attentionPolicy",
    }
)
_RESULT_FIELDS = frozenset({"type", "invocationId", "executeMs", "error", "outputs", "outputStats"})


class UnsupportedCapability(BoundaryError):
    """This invocation requires more than the experimental stateless subset."""


class UnknownOutcome(BoundaryError):
    """Dispatch failed; execution may have happened. Never retry implicitly."""


@dataclass(frozen=True)
class Limits:
    envelope_bytes: int = 1024 * 1024
    blob_bytes: int = 64 * 1024 * 1024
    payload_bytes: int = 256 * 1024 * 1024
    blob_count: int = 1024

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 1 for value in vars(self).values()):
            raise ValueError("transport limits must be positive integers")


class ObjectStore(Protocol):
    async def put(self, digest: str, data: bytes) -> None:
        """Idempotently publish immutable bytes at their canonical digest."""
        ...

    async def get(self, digest: str, *, max_bytes: int) -> bytes:
        """Read at most max_bytes; raise on missing or oversized content."""
        ...


class DiskObjectStore:
    """Offline object-store adapter using DiskCAS publication and layout."""

    def __init__(self, cas: DiskCAS) -> None:
        self.cas = cas

    async def put(self, digest: str, data: bytes) -> None:
        if digest_bytes(data) != require_digest(digest):
            raise BoundaryError("object-store put digest mismatch")
        await asyncio.to_thread(self.cas.put, data)

    async def get(self, digest: str, *, max_bytes: int) -> bytes:
        hexpart = require_digest(digest).split(":", 1)[1]
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("max_bytes must be nonnegative")

        def read() -> bytes:
            # DiskCAS.get has no bounded-read API; do not materialize a corrupt large file.
            path = self.cas.root / hexpart[:2] / hexpart
            with path.open("rb") as stream:
                data = stream.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise BoundaryError("object-store blob exceeds size limit")
            if digest_bytes(data) != digest:
                raise BoundaryError("object-store blob digest mismatch")
            return data

        return await asyncio.to_thread(read)


def _json_copy(value: object, limit: int) -> Envelope:
    def check(item: object, depth: int = 0) -> None:
        if depth > 32:
            raise BoundaryError("envelope nesting exceeds limit")
        if type(item) is dict:
            for key, child in cast("dict[object, object]", item).items():
                if type(key) is not str:
                    raise BoundaryError("envelope keys must be strings")
                check(child, depth + 1)
        elif type(item) is list:
            for child in cast("list[object]", item):
                check(child, depth + 1)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise BoundaryError("envelope must be JSON-safe")

    check(value)
    if type(value) is not dict:
        raise BoundaryError("envelope must be an object")
    size = 0
    chunks: list[str] = []
    try:
        for chunk in json.JSONEncoder(allow_nan=False, separators=(",", ":")).iterencode(value):
            size += len(chunk.encode("utf-8"))
            if size > limit:
                raise BoundaryError("envelope exceeds size limit")
            chunks.append(chunk)
        return json.loads("".join(chunks))
    except (ValueError, TypeError, UnicodeError) as exc:
        raise BoundaryError("envelope must be strict JSON") from exc


def _data_type(type_id: str, depth: int = 0) -> None:
    if depth > 32:
        raise BoundaryError("value type nesting exceeds limit")
    element = parse_list_type_id(type_id)
    if element is not None:
        _data_type(element, depth + 1)
    elif type_id not in _DATA_TYPES:
        raise UnsupportedCapability(f"unsupported or resident value type: {type_id}")


def _metadata(meta: Mapping[str, object]) -> None:
    if _RESIDENT_META.intersection(meta):
        raise UnsupportedCapability("resident/resource metadata cannot cross on-demand workers")


def _values(values: Mapping[str, Value]) -> None:
    def visit(value: Value, depth: int = 0) -> None:
        if depth > 32:
            raise BoundaryError("value nesting exceeds limit")
        _data_type(value.type_id)
        _metadata(value.meta.entries)
        for child in list_children(value) or ():
            visit(child, depth + 1)

    for value in values.values():
        visit(value)


def _schema(schema: NodeSchema) -> None:
    if not schema.is_static:
        raise UnsupportedCapability("dynamic interfaces are unsupported")
    if any(spec.lazy for spec in schema.inputs):
        raise UnsupportedCapability("lazy inputs are unsupported")


def _invocation(invocation: Invocation, schemas: Mapping[str, NodeSchema]) -> None:
    if invocation.node_type not in schemas:
        raise UnsupportedCapability(f"unknown node type: {invocation.node_type}")
    _schema(invocation.effective_schema)
    if invocation.effective_schema != schemas[invocation.node_type]:
        raise BoundaryError("effective schema does not match pinned catalog")
    if (
        invocation.executor is not None
        or invocation.arm is not None
        or invocation.expected_execution_identity is not None
        or invocation.fp8_matmul
        or invocation.diffusion_dtype is not None
        or invocation.text_dtype is not None
        or invocation.vae_dtype is not None
        or invocation.attention_policy != "auto"
        or invocation.attention_route_token is not None
        or invocation.extension_snapshot_digest is not None
        or invocation.export_snapshot is not None
        or invocation.media_sources
        or invocation.preview_mode != "off"
        or invocation.preview_animation != "ring"
        or invocation.connected_undemanded_inputs
        or invocation.output_members
    ):
        raise UnsupportedCapability("executor, lazy, preview or session-dependent invocation")
    _values(invocation.inputs)


def _correlation(header: Envelope) -> Envelope:
    for field in ("invocationId", "jobRef", "nodeId", "nodeType"):
        if type(header.get(field)) is not str or not header[field]:
            raise BoundaryError(f"invalid {field}")
    if type(header.get("attemptId")) is not int or header["attemptId"] < 1:
        raise BoundaryError("invalid attemptId")
    return {
        key: header[key] for key in ("invocationId", "jobRef", "attemptId", "nodeId", "nodeType")
    }


class Transport:
    """One endpoint's codecs, transfer limits and private read-through disk cache."""

    def __init__(
        self,
        registry: TypeRegistry,
        store: ObjectStore,
        cache: DiskCAS,
        *,
        source_identity: str,
        limits: Limits | None = None,
    ) -> None:
        if not source_identity or len(source_identity) > 256:
            raise ValueError("source_identity must name the pinned deployment")
        self.registry = registry
        self.store = store
        self.cache = DiskObjectStore(cache)
        self.limits = limits or Limits()
        self.identity = {
            "version": _VERSION,
            "protocolVersion": PROTOCOL_VERSION,
            "schemaVersion": SCHEMA_WIRE_VERSION,
            "sourceIdentity": source_identity,
        }

    def codec(self) -> ValueCodec:
        return ValueCodec(self.registry, use_shm=False, accept_shm=False)

    def _refs(self, refs: object) -> list[Envelope]:
        if type(refs) is not list or len(refs) > self.limits.blob_count:
            raise BoundaryError("invalid blob count")
        total = 0
        for ref in refs:
            if type(ref) is not dict or set(ref) != {"digest", "size"}:
                raise BoundaryError("invalid blob reference")
            require_digest(ref["digest"])
            size = ref["size"]
            if type(size) is not int or not 0 <= size <= self.limits.blob_bytes:
                raise BoundaryError("blob exceeds size limit")
            total += size
        if total > self.limits.payload_bytes:
            raise BoundaryError("payload exceeds size limit")
        return refs

    async def pack(self, header: Envelope, blobs: list[bytes], correlation: Envelope) -> Envelope:
        refs = self._refs([{"digest": digest_bytes(blob), "size": len(blob)} for blob in blobs])
        envelope = _json_copy(
            {**self.identity, "correlation": correlation, "frame": header, "blobs": refs},
            self.limits.envelope_bytes,
        )
        self._descriptors(header, blobs)
        for ref, blob in zip(refs, blobs, strict=True):
            await self.store.put(ref["digest"], blob)
            await self.cache.put(ref["digest"], blob)
        return envelope

    async def unpack(
        self,
        envelope: Envelope,
        kind: str,
        expected: Envelope | None = None,
    ) -> tuple[Envelope, list[bytes], Envelope]:
        envelope = _json_copy(envelope, self.limits.envelope_bytes)
        if set(envelope) != {*self.identity, "correlation", "frame", "blobs"}:
            raise UnsupportedCapability("unsupported envelope fields")
        if any(
            type(envelope[key]) is not type(value) or envelope[key] != value
            for key, value in self.identity.items()
        ):
            raise BoundaryError("protocol/schema/source identity mismatch")
        correlation = envelope["correlation"]
        if type(correlation) is not dict or _correlation(correlation) != correlation:
            raise BoundaryError("invalid correlation")
        if expected is not None and correlation != expected:
            raise BoundaryError("result correlation mismatch")
        header = envelope["frame"]
        fields = _INVOKE_FIELDS if kind == "invoke" else _RESULT_FIELDS
        if type(header) is not dict or set(header) - fields:
            raise UnsupportedCapability("unsupported frame fields/capabilities")
        if kind == "invoke" and set(header) != fields:
            raise BoundaryError("incomplete invocation frame")
        if header.get("type") != kind or header.get("invocationId") != correlation["invocationId"]:
            raise BoundaryError("frame identity mismatch")
        if kind == "invoke" and _correlation(header) != correlation:
            raise BoundaryError("invocation correlation mismatch")
        refs = self._refs(envelope["blobs"])
        blobs: list[bytes] = []
        for ref in refs:
            try:
                blob = await self.cache.get(ref["digest"], max_bytes=ref["size"])
            except FileNotFoundError:
                blob = await self.store.get(ref["digest"], max_bytes=ref["size"])
            if (
                type(blob) is not bytes
                or len(blob) != ref["size"]
                or digest_bytes(blob) != ref["digest"]
            ):
                raise BoundaryError("fetched blob digest/size mismatch")
            await self.cache.put(ref["digest"], blob)
            blobs.append(blob)
        self._descriptors(header, blobs)
        return header, blobs, correlation

    def _descriptors(self, header: Envelope, blobs: list[bytes]) -> None:
        used: set[int] = set()

        def blob_at(index: object) -> bytes:
            if type(index) is not int or not 0 <= index < len(blobs):
                raise BoundaryError("invalid blob index")
            used.add(index)
            return blobs[index]

        def visit(wire: Envelope) -> None:
            if type(wire) is not dict or set(wire) not in (
                {"typeId", "fingerprint", "metaBlob", "payload"},
                {"typeId", "fingerprint", "metaBlob", "elements"},
            ):
                raise UnsupportedCapability("unsupported value descriptor")
            if type(wire["typeId"]) is not str or type(wire["fingerprint"]) is not str:
                raise BoundaryError("invalid value identity")
            _data_type(wire["typeId"])
            meta = blob_at(wire["metaBlob"])
            if not meta.startswith(b"json:"):
                raise UnsupportedCapability("non-JSON metadata is unsupported")
            decoded = default_decode(meta)
            if not isinstance(decoded, dict):
                raise BoundaryError("metadata must be an object")
            _metadata(decoded)
            if "elements" in wire:
                if parse_list_type_id(wire["typeId"]) is None or type(wire["elements"]) is not list:
                    raise BoundaryError("invalid list descriptor")
                for child in wire["elements"]:
                    visit(child)
            else:
                payload = wire["payload"]
                if type(payload) is not dict or set(payload) != {"transport", "blob"}:
                    raise UnsupportedCapability("unsupported payload descriptor")
                if payload["transport"] != "inline":
                    raise UnsupportedCapability(
                        "only object-store backed inline frames are supported"
                    )
                data = blob_at(payload["blob"])
                if wire["typeId"].startswith("core.") and not data.startswith(b"json:"):
                    raise UnsupportedCapability("non-JSON scalar is unsupported")

        values = header.get("inputs" if header["type"] == "invoke" else "outputs", {})
        if type(values) is not dict:
            raise BoundaryError("values must be an object")
        for wire in values.values():
            visit(wire)
        if used != set(range(len(blobs))):
            raise BoundaryError("unreferenced blobs")


class OnDemandWorker:
    def __init__(self, transport: Transport, schemas: Mapping[str, NodeSchema], dispatch: Dispatch):
        self.transport = transport
        self.schemas = MappingProxyType(dict(schemas))
        self.dispatch = dispatch

    async def prepare(self, node_types: Sequence[str]) -> None:
        for node_type in node_types:
            if node_type not in self.schemas:
                raise UnsupportedCapability(f"unknown node type: {node_type}")
            _schema(self.schemas[node_type])

    async def invoke(
        self,
        invocation: Invocation,
        on_event: OnInvocationEvent | None = None,
    ) -> InvocationResult:
        del on_event  # The handler refuses actual emissions, even without an observer.
        _invocation(invocation, self.schemas)
        codec = self.transport.codec()
        header, blobs, segments, _ = encode_invocation(codec, invocation)
        assert not segments
        correlation = _correlation(header)
        request = await self.transport.pack(header, blobs, correlation)
        try:
            try:
                response = await self.dispatch(request)
            except Exception as exc:
                raise UnknownOutcome(
                    f"on-demand outcome unknown for {correlation}; do not retry"
                ) from exc
            header, blobs, _ = await self.transport.unpack(response, "result", correlation)
        except asyncio.CancelledError as exc:
            # A cancelled owner lets Engine waiters redispatch the shared computation.
            raise UnknownOutcome(
                f"on-demand outcome unknown for {correlation}; "
                "remote work may continue; do not retry"
            ) from exc
        if ("error" in header) == ("outputs" in header):
            raise BoundaryError("result must contain exactly one of outputs or error")
        if "error" in header:
            error = error_from_wire(header["error"])
            if error.node_id != invocation.node_id or error.node_type != invocation.node_type:
                raise BoundaryError("error node identity mismatch")
            return InvocationResult(error=error)
        outputs, _ = decode_result_outputs(codec, header, blobs, [])
        _values(outputs)
        return InvocationResult(outputs=outputs)


class InvocationHandler:
    def __init__(self, transport: Transport, worker: InProcessWorker) -> None:
        self.transport = transport
        self.worker = worker

    async def __call__(self, envelope: Envelope) -> Envelope:
        header, blobs, correlation = await self.transport.unpack(envelope, "invoke")
        codec = self.transport.codec()
        invocation = decode_invocation(codec, header, blobs, [])
        if header["effectiveSchema"] != schema_to_wire(invocation.effective_schema):
            raise UnsupportedCapability("noncanonical or unsupported schema fields/version")
        _invocation(invocation, self.worker.schemas)
        emitted = False

        def reject_event(event: InvocationEvent) -> None:
            nonlocal emitted
            emitted = True

        await self.worker.prepare([invocation.node_type])
        started = time.perf_counter()
        result = await self.worker.invoke(invocation, reject_event)
        if emitted:
            raise UnsupportedCapability("invocation events are unsupported")
        if type(result) is not InvocationResult:
            raise UnsupportedCapability("continuation/result algebra is unsupported")
        if result.artifacts or result.artifact_candidates:
            raise UnsupportedCapability("saved artifacts are unsupported")
        if (result.error is None) == (result.outputs is None):
            raise BoundaryError("result must contain exactly one of outputs or error")
        _values(result.outputs or {})
        header, blobs, segments = encode_result(
            codec,
            result,
            invocation.invocation_id,
            (time.perf_counter() - started) * 1000,
        )
        assert not segments
        return await self.transport.pack(header, blobs, correlation)
