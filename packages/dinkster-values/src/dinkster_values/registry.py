"""Type registry: value types are registered data, not isinstance guesses.

Registration defaults are correct-everywhere-possibly-slow (DESIGN 3.2):
a type registered with nothing but a name works in-process, crosses
boundaries via the default codec (canonical JSON for JSON-shaped values,
pickle fallback otherwise - workers are our own trust domain), and gets a
content-hash fingerprint of the encoded bytes. Declaring a codec or
fingerprint is an optimization, never a prerequisite.
"""

from __future__ import annotations

import json
import math
import pickle
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from urllib.parse import urlencode

from .assets import ASSET_BASE_TYPE, parse_asset_type_id, runtime_type_atom
from .builtin_types import (
    ARRAY_TYPE,
    BUILTIN_TYPE_IDS,
    CORE_ABSENT,
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    DICT_TYPE,
    HISTOGRAM_TYPE,
)
from .lists import make_list_value, parse_list_type_id, value_children
from .model import (
    EncodedPayload,
    PyObjPayload,
    TypeId,
    Value,
    ValueMeta,
    stable_hash,
)
from .names import claim_covers, claims_conflict
from .streams import MediaStream, make_stream_value, parse_stream_type_id
from .tuples import make_tuple_value, parse_tuple_type_id

if TYPE_CHECKING:
    from .resident import ResidencyTable

Encoder = Callable[[object], bytes]
Decoder = Callable[[bytes], object]
FingerprintFn = Callable[[object], str]
MetaFn = Callable[[object], Mapping[str, object]]
CoerceFn = Callable[[object], object]
InlineFn = Callable[[object], object]
RenderFn = Callable[[object], bytes] | Callable[[object, Mapping[str, str]], bytes]
RenditionMimeFn = Callable[[Mapping[str, object]], str]
RenditionNormalizeFn = Callable[[Mapping[str, str], Mapping[str, object]], Mapping[str, str]]
RenditionResolveAsyncFn = Callable[
    [Mapping[str, object], Mapping[str, str]], Awaitable[tuple[str, Mapping[str, str]]]
]
RenditionMimeAsyncFn = Callable[[Mapping[str, object]], Awaitable[str]]
RenditionRenderAsyncFn = Callable[[Value, Mapping[str, str]], Awaitable["Rendition"]]
ValidateEncodedFn = Callable[[bytes, Mapping[str, object]], None]
ValidateEncodedBufferFn = Callable[[memoryview, Mapping[str, object]], None]
BufferWriter = Callable[[memoryview], int]

INLINE_STRING_CAP = 256
"""Longest string the inline-value channel carries (DESIGN 3.5): inline
values exist so peeking an int or short string never needs a payload round
trip, not so descriptors smuggle documents. Enforced centrally in
``inline_of`` - a pack's inline fn cannot opt out of the cap."""


class InvalidRenditionRequest(ValueError):
    """A rendition selector is malformed, unsupported, or out of range."""


class RenditionUnavailable(Exception):
    """A valid rendition request cannot be served for the current value."""


def _equivalent_fingerprint(
    provider_id: str,
    source_type_id: TypeId,
    target_type_id: TypeId,
    source_fingerprint: str,
) -> str:
    return stable_hash(
        [
            b"type-equivalence",
            provider_id.encode("utf-8"),
            source_type_id.encode("utf-8"),
            target_type_id.encode("utf-8"),
            source_fingerprint.encode("utf-8"),
        ]
    )


def default_encode(obj: object) -> bytes:
    try:
        return b"json:" + json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError):
        return b"pickle:" + pickle.dumps(obj, protocol=5)


def default_decode(data: bytes) -> object:
    kind, _, body = data.partition(b":")
    if kind == b"json":
        return json.loads(body)
    if kind == b"pickle":
        return pickle.loads(body)
    raise ValueError(f"unknown default-codec framing: {kind!r}")


@dataclass(frozen=True)
class BufferEncoding:
    """A prepared encoding that writes into caller-owned storage.

    ``write`` must synchronously fill exactly ``size`` bytes, return that
    count, and not retain the supplied buffer.
    """

    size: int
    write: BufferWriter

    def __post_init__(self) -> None:
        if type(self.size) is not int or self.size < 0:
            raise ValueError("buffer encoding size must be a non-negative integer")


PrepareBufferEncodingFn = Callable[[object], BufferEncoding]


@dataclass(frozen=True)
class TypeSpec:
    type_id: TypeId
    encode: Encoder
    decode: Decoder
    fingerprint: FingerprintFn | None  # None -> hash of encode(obj)
    meta: MetaFn | None
    declared_codec: bool  # False -> default fallback; dev-mode diagnostic hook (DESIGN 3.8)
    coerce: CoerceFn | None = None
    """Normalize a raw object into the type's runtime form at wrap time.

    Graph literals arrive as JSON-shaped data; a type whose runtime form is
    richer (an AssetRef, an array) coerces here, once, before fingerprint,
    meta, and payload ever see the object. Must be idempotent: worker shims
    wrap node outputs through the same path, so an already-runtime-formed
    object passes through unchanged."""

    inline: InlineFn | None = None
    """Declared safe inline serialization (DESIGN 3.5): map the runtime
    object to a small JSON-native scalar for embedding directly in output
    descriptors and event summaries, or return None to decline. Only
    int/float/bool/str survive the central policy in ``inline_of`` -
    declaring inline can never accidentally JSONify an image. None (the
    default) means the type never inlines."""

    validate_encoded: ValidateEncodedFn | None = None
    """Validate codec bytes and envelope metadata before trusting an encoded
    value received from a boundary or cache. The hook must be deterministic,
    pure, and read-only. None (the default) preserves existing behavior."""

    validate_encoded_buffer: ValidateEncodedBufferFn | None = None
    """Validate encoded data through a borrowed read-only buffer. Boundary
    callers may release the buffer as soon as this hook returns."""

    prepare_buffer_encoding: PrepareBufferEncodingFn | None = None
    """Optionally prepare the declared codec's exact bytes for direct writes
    into caller-owned storage. Other transports continue to use ``encode``."""

    input_convert: CoerceFn | None = None
    """Convert storage to the ordinary input form at invocation, without
    mutating the stored object. None preserves it; accepts_storage bypasses
    this hook. Unlike coerce, this never changes output storage or identity."""


@dataclass(frozen=True)
class RenditionSpec:
    """One registered way to turn a value into browser-consumable bytes.

    Renditions are the presentation half of hazard H2: the envelope-first
    value model never puts raw tensors on the client wire, so rich types
    declare how they render (a PNG for an image, a waveform preview for
    audio) and clients negotiate by ``kind``. Open registry, same
    philosophy as type atoms: packs register their own."""

    type_id: TypeId
    kind: str  # stable client-facing name, e.g. "png"
    mime: str | RenditionMimeFn  # static, or resolved from envelope metadata
    render: RenderFn
    default: bool
    version: str | None
    """Renderer identity appended to immutable cache keys.

    Bump this value whenever ``render`` changes bytes for an unchanged value.
    None preserves the kind-only identity for byte-stable renderers."""
    parameters: tuple[str, ...] = ()
    defaults: Mapping[str, str] | None = None
    limits: Mapping[str, object] | None = None
    normalize: RenditionNormalizeFn | None = None
    resolve_async: RenditionResolveAsyncFn | None = None
    mime_async: RenditionMimeAsyncFn | None = None
    render_async: RenditionRenderAsyncFn | None = None
    relay_owner: str | None = None

    def mime_for(self, metadata: Mapping[str, object]) -> str:
        """Resolve the truthful MIME without loading the value payload."""
        return self.mime(metadata) if callable(self.mime) else self.mime

    @property
    def selector(self) -> str:
        """Return the client-visible selector for this renderer."""
        return self.kind if self.version is None else f"{self.kind}/{self.version}"

    def normalize_parameters(
        self, parameters: Mapping[str, str], metadata: Mapping[str, object]
    ) -> Mapping[str, str]:
        if self.normalize is None:
            if parameters:
                raise InvalidRenditionRequest(f"{self.kind} rendition accepts no parameters")
            return {}
        normalized = cast(dict[str, object], dict(self.normalize(parameters, metadata)))
        if set(normalized) != set(self.parameters) or any(
            not isinstance(value, str) for value in normalized.values()
        ):
            raise InvalidRenditionRequest(
                f"{self.kind} rendition normalizer did not return every declared parameter"
            )
        return cast(dict[str, str], normalized)

    def cache_key(self, fingerprint: str, parameters: Mapping[str, str] | None = None) -> str:
        """Return the immutable client cache identity for this renderer."""
        key = f"{fingerprint}/{self.selector}"
        return key if not parameters else f"{key}?{urlencode(sorted(parameters.items()))}"


@dataclass(frozen=True)
class Rendition:
    """Rendered bytes plus what they are."""

    kind: str
    mime: str
    data: bytes


@dataclass(frozen=True)
class AssetDecoderSpec:
    """One registered ``asset<target> -> target`` decode provider.

    The execution half of typed assets (joint contract 2026-07-26):
    renditions turn values into client bytes, decode providers turn asset
    bytes into runtime values. Keyed by the TARGET runtime type id (``T``
    or ``list<T>``), one provider per target, conflict = registration
    error. ``provider_id`` is the provider's stable identity: it joins
    coerced-input cache fingerprints (identity = asset digest + provider
    identity), so it must only change when the decode semantics do.

    ``decode`` receives the base asset type's runtime object (an AssetRef
    in any process that composed dinkster-assets) and returns the raw target
    object - the same shape node code would return, wrapped by the worker
    shim through the ordinary path. Providers yield ONE value; multi-output
    decodes stay explicit Load nodes."""

    target_type_id: TypeId
    provider_id: str
    decode: Callable[[object], object]


@dataclass(frozen=True)
class BatchMergeSpec:
    """One registered ``list<T> -> T`` merge for an internally-batched type.

    The EXPLICIT mechanism scoping multi-asset selection to genuinely
    batchable types (comfy.IMAGE-style ``[B, H, W, C]`` runtime forms):
    a list of decoded assets merges into ONE batched T only when T
    registered a merge, and only as the terminal step of an asset
    coercion - plain list<T> links into scalar-T inputs stay a structural
    error. ``provider_id`` joins coerced cache fingerprints exactly like
    a decoder's."""

    type_id: TypeId
    provider_id: str
    merge: Callable[[Sequence[object]], object]


class TypeRegistry:
    def __init__(self) -> None:
        self._types: dict[TypeId, TypeSpec] = {}
        self._declared_types: dict[str, frozenset[TypeId]] = {}
        self._renditions: dict[TypeId, list[RenditionSpec]] = {}
        self._asset_decoders: dict[TypeId, AssetDecoderSpec] = {}
        self._batch_merges: dict[TypeId, BatchMergeSpec] = {}
        self._type_equivalences: dict[TypeId, tuple[TypeId, str]] = {}
        self._claims: dict[str, tuple[str, ...]] = {}
        self._owners: dict[TypeId, str] = dict.fromkeys(
            BUILTIN_TYPE_IDS, "dinkster-nodes-foundation"
        )
        self._registering_pack: str | None = None
        self._allow_bare = False
        self._resident_tables: list[ResidencyTable] = []

    @property
    def resident_tables(self) -> tuple[ResidencyTable, ...]:
        """Holding tables used by this registry's resident codecs."""
        return tuple(self._resident_tables)

    def register_resident_table(self, table: ResidencyTable) -> None:
        if not any(existing is table for existing in self._resident_tables):
            self._resident_tables.append(table)

    def copy(self) -> TypeRegistry:
        """Return an independent registry with the same registrations."""
        copied = TypeRegistry()
        copied._types = dict(self._types)
        copied._declared_types = dict(self._declared_types)
        copied._renditions = {
            type_id: list(renditions) for type_id, renditions in self._renditions.items()
        }
        copied._asset_decoders = dict(self._asset_decoders)
        copied._batch_merges = dict(self._batch_merges)
        copied._type_equivalences = dict(self._type_equivalences)
        copied._claims = dict(self._claims)
        copied._owners = dict(self._owners)
        copied._resident_tables = list(self._resident_tables)
        return copied

    def replace_from(self, other: TypeRegistry) -> None:
        """Publish all registrations from a staged registry before exposure."""
        self._types = dict(other._types)
        self._declared_types = dict(other._declared_types)
        self._renditions = {
            type_id: list(renditions) for type_id, renditions in other._renditions.items()
        }
        self._asset_decoders = dict(other._asset_decoders)
        self._batch_merges = dict(other._batch_merges)
        self._type_equivalences = dict(other._type_equivalences)
        self._claims = dict(other._claims)
        self._owners = dict(other._owners)
        self._resident_tables = list(other._resident_tables)

    def claim_types(self, owner: str, claims: tuple[str, ...]) -> None:
        """Reserve type ids or namespaces before publishing a pack's codecs."""
        for claim in claims:
            for holder, held_claims in self._claims.items():
                if holder == owner:
                    continue
                for held in held_claims:
                    if claims_conflict(held, claim):
                        raise ValueError(
                            f"type claim {claim!r} owned by {owner!r} overlaps "
                            f"{held!r} owned by {holder!r}"
                        )
            for type_id, holder in self._owners.items():
                if holder != owner and claims_conflict(type_id, claim):
                    raise ValueError(
                        f"type claim {claim!r} overlaps {type_id!r} owned by {holder!r}"
                    )
        self._claims[owner] = claims

    @contextmanager
    def registering_pack(
        self, owner: str, claims: tuple[str, ...], *, allow_bare: bool = False
    ) -> Generator[None]:
        """Attribute registrations and enforce the manifest's type claims."""
        self.claim_types(owner, claims)
        previous = self._registering_pack
        previous_bare = self._allow_bare
        self._registering_pack = owner
        self._allow_bare = allow_bare
        try:
            yield
        finally:
            self._registering_pack = previous
            self._allow_bare = previous_bare

    def owner_of(self, type_id: TypeId) -> str | None:
        """Return the owner even when a declared codec has not loaded yet."""
        if owner := self._owners.get(type_id):
            return owner
        return next(
            (
                owner
                for owner, claims in self._claims.items()
                if any(type_id == claim or claim_covers(claim, type_id) for claim in claims)
            ),
            None,
        )

    def type_owners(self) -> dict[TypeId, str]:
        """Return registered atom owners for registry discovery."""
        return dict(self._owners)

    def declare_types(
        self,
        owner: str,
        claims: tuple[str, ...],
        type_ids: Sequence[TypeId],
        *,
        allow_bare: bool = False,
    ) -> None:
        """Publish worker-owned type identities without importing their codecs."""
        for type_id in type_ids:
            if allow_bare and type_id in BUILTIN_TYPE_IDS:
                continue
            if not any(type_id == claim or claim_covers(claim, type_id) for claim in claims):
                raise ValueError(f"{owner}: type {type_id!r} is outside its declared type claims")
            if (held := self.owner_of(type_id)) is not None and held != owner:
                raise ValueError(f"type {type_id!r} is owned by {held!r}, not {owner!r}")
        self.claim_types(owner, claims)
        self._owners.update(
            dict.fromkeys((tid for tid in type_ids if tid not in BUILTIN_TYPE_IDS), owner)
        )
        self._declared_types[owner] = self._declared_types.get(owner, frozenset()) | frozenset(
            type_ids
        )

    def remove_pack_types(self, owner: str) -> None:
        """Release a pack's claims, codecs, and type-specific registrations."""
        self._declared_types.pop(owner, None)
        removed = {
            type_id
            for type_id, holder in self._owners.items()
            if holder == owner and type_id not in BUILTIN_TYPE_IDS
        }
        self._claims.pop(owner, None)
        for type_id in removed:
            self._owners.pop(type_id)
            self._types.pop(type_id, None)
            self._renditions.pop(type_id, None)
            self._batch_merges.pop(type_id, None)
        self._asset_decoders = {
            target: spec
            for target, spec in self._asset_decoders.items()
            if runtime_type_atom(target) not in removed
        }
        self._type_equivalences = {
            type_id: pair
            for type_id, pair in self._type_equivalences.items()
            if type_id not in removed and pair[0] not in removed
        }

    def register(
        self,
        type_id: TypeId,
        *,
        encode: Encoder | None = None,
        decode: Decoder | None = None,
        prepare_buffer_encoding: PrepareBufferEncodingFn | None = None,
        fingerprint: FingerprintFn | None = None,
        meta: MetaFn | None = None,
        coerce: CoerceFn | None = None,
        input_convert: CoerceFn | None = None,
        inline: InlineFn | None = None,
        validate_encoded: ValidateEncodedFn | None = None,
        validate_encoded_buffer: ValidateEncodedBufferFn | None = None,
    ) -> TypeSpec:
        if (encode is None) != (decode is None):
            raise ValueError(f"{type_id}: encode and decode must be declared together")
        if prepare_buffer_encoding is not None and encode is None:
            raise ValueError(f"{type_id}: buffer encoding requires a declared codec")
        if "<" in type_id or ">" in type_id:
            # The canonical type-id grammar is closed: name | "list<" id ">".
            # Angle brackets are constructor syntax, never part of an atom
            # name - list values are built structurally (wrap/make_list_value),
            # never registered as types.
            raise ValueError(f"type ids must not contain '<' or '>': {type_id}")
        if type_id in self._types:
            raise ValueError(f"type already registered: {type_id}")
        owner = self._registering_pack or self.owner_of(type_id)
        builtin = type_id in BUILTIN_TYPE_IDS
        if self._registering_pack is not None and not self._allow_bare:
            if not claim_covers(self._registering_pack, type_id):
                raise ValueError(f"{owner}: type {type_id!r} must be prefixed by its pack name")
        if (
            self._registering_pack is not None
            and not (builtin and self._allow_bare)
            and not any(
                type_id == claim or claim_covers(claim, type_id)
                for claim in self._claims[self._registering_pack]
            )
        ):
            raise ValueError(f"{owner}: type {type_id!r} is outside its declared type claims")
        if builtin:
            owner = "dinkster-nodes-foundation"
        held_owner = self.owner_of(type_id)
        if owner is not None and held_owner is not None and held_owner != owner:
            raise ValueError(f"type {type_id!r} is owned by {held_owner!r}, not {owner!r}")
        spec = TypeSpec(
            type_id=type_id,
            encode=encode if encode is not None else default_encode,
            decode=decode if decode is not None else default_decode,
            fingerprint=fingerprint,
            meta=meta,
            declared_codec=encode is not None,
            prepare_buffer_encoding=prepare_buffer_encoding,
            coerce=coerce,
            input_convert=input_convert,
            inline=inline,
            validate_encoded=validate_encoded,
            validate_encoded_buffer=validate_encoded_buffer,
        )
        self._types[type_id] = spec
        if owner is not None:
            self._owners[type_id] = owner
        return spec

    def register_rendition(
        self,
        type_id: TypeId,
        kind: str,
        *,
        mime: str | RenditionMimeFn,
        render: RenderFn,
        default: bool = False,
        version: str | None = None,
        parameters: tuple[str, ...] = (),
        defaults: Mapping[str, str] | None = None,
        limits: Mapping[str, object] | None = None,
        normalize: RenditionNormalizeFn | None = None,
    ) -> RenditionSpec:
        """Declare a browser-renderable form for a registered type.

        The first registration for a type becomes its default; at most one
        rendition per type may be the default, so a later ``default=True``
        raises rather than silently re-pointing clients."""
        if type_id not in self._types:
            raise KeyError(f"unregistered value type: {type_id}")
        if not kind:
            raise ValueError(f"{type_id}: rendition kind must be non-empty")
        if version == "":
            raise ValueError(f"{type_id}: rendition version must be non-empty")
        if len(set(parameters)) != len(parameters) or any(not name for name in parameters):
            raise ValueError(f"{type_id}: rendition parameters must be unique and non-empty")
        if set(defaults or ()) - set(parameters):
            raise ValueError(f"{type_id}: rendition defaults must name declared parameters")
        if bool(parameters) != (normalize is not None):
            raise ValueError(f"{type_id}: parameterized renditions require a normalizer")
        if parameters and version is None:
            raise ValueError(f"{type_id}: parameterized renditions require a version")
        existing = self._renditions.setdefault(type_id, [])
        replaced_default = any(
            spec.kind == kind and spec.relay_owner is not None and spec.default for spec in existing
        )
        retained = [spec for spec in existing if spec.kind != kind or spec.relay_owner is None]
        if any(spec.kind == kind for spec in retained):
            raise ValueError(f"{type_id}: rendition kind already registered: {kind}")
        if default and any(spec.default for spec in retained):
            raise ValueError(f"{type_id}: a default rendition already exists")
        spec = RenditionSpec(
            type_id=type_id,
            kind=kind,
            mime=mime,
            render=render,
            default=default or replaced_default or not retained,
            version=version,
            parameters=parameters,
            defaults=None if defaults is None else dict(defaults),
            limits=None if limits is None else dict(limits),
            normalize=normalize,
        )
        for registered in retained:
            collisions = {spec.kind, spec.selector} & {registered.kind, registered.selector}
            if collisions:
                collision = min(collisions)
                raise ValueError(f"{type_id}: rendition selector already registered: {collision}")
        self._renditions[type_id] = [*retained, spec]
        return spec

    def register_relayed_rendition(
        self,
        type_id: TypeId,
        kind: str,
        *,
        mime: str,
        default: bool,
        version: str | None,
        parameters: tuple[str, ...],
        defaults: Mapping[str, str] | None,
        limits: Mapping[str, object] | None,
        owner: str,
        resolve_mime: RenditionMimeAsyncFn,
        resolve: RenditionResolveAsyncFn,
        render: RenditionRenderAsyncFn,
    ) -> RenditionSpec:
        """Register metadata and RPC callbacks for a worker-owned renderer."""
        if not owner:
            raise ValueError("relayed rendition owner must be non-empty")
        if not kind:
            raise ValueError(f"{type_id}: rendition kind must be non-empty")
        if version == "":
            raise ValueError(f"{type_id}: rendition version must be non-empty")
        if len(set(parameters)) != len(parameters) or any(not name for name in parameters):
            raise ValueError(f"{type_id}: rendition parameters must be unique and non-empty")
        if set(defaults or ()) - set(parameters):
            raise ValueError(f"{type_id}: rendition defaults must name declared parameters")
        if parameters and version is None:
            raise ValueError(f"{type_id}: parameterized renditions require a version")
        existing = self._renditions.setdefault(type_id, [])
        if any(spec.kind == kind for spec in existing):
            raise ValueError(f"{type_id}: rendition kind already registered: {kind}")
        if default and any(spec.default for spec in existing):
            raise ValueError(f"{type_id}: a default rendition already exists")
        spec = RenditionSpec(
            type_id=type_id,
            kind=kind,
            mime=mime,
            render=lambda _value: b"",
            default=default or not existing,
            version=version,
            parameters=parameters,
            defaults=None if defaults is None else dict(defaults),
            limits=None if limits is None else dict(limits),
            resolve_async=resolve,
            mime_async=resolve_mime,
            render_async=render,
            relay_owner=owner,
        )
        for registered in existing:
            collisions = {spec.kind, spec.selector} & {registered.kind, registered.selector}
            if collisions:
                collision = min(collisions)
                raise ValueError(f"{type_id}: rendition selector already registered: {collision}")
        existing.append(spec)
        return spec

    def remove_relayed_renditions(self, owner: str) -> None:
        """Remove renderer proxies belonging to one worker generation."""
        for type_id in tuple(self._renditions):
            remaining = [spec for spec in self._renditions[type_id] if spec.relay_owner != owner]
            if remaining:
                self._renditions[type_id] = remaining
            else:
                del self._renditions[type_id]

    def renditions_of(self, type_id: TypeId) -> tuple[RenditionSpec, ...]:
        return tuple(self._renditions.get(type_id, ()))

    def registered_renditions(self) -> tuple[RenditionSpec, ...]:
        """Return every renderer in registration order by value type."""
        return tuple(spec for specs in self._renditions.values() for spec in specs)

    async def resolve_rendition(
        self,
        spec: RenditionSpec,
        metadata: Mapping[str, object],
        parameters: Mapping[str, str],
    ) -> tuple[str, Mapping[str, str]]:
        if spec.resolve_async is not None:
            return await spec.resolve_async(metadata, parameters)
        return spec.mime_for(metadata), spec.normalize_parameters(parameters, metadata)

    async def rendition_mime(self, spec: RenditionSpec, metadata: Mapping[str, object]) -> str:
        if spec.mime_async is not None:
            return await spec.mime_async(metadata)
        return spec.mime_for(metadata)

    def register_asset_decoder(
        self,
        target_type_id: TypeId,
        *,
        provider_id: str,
        decode: Callable[[object], object],
    ) -> AssetDecoderSpec:
        """Declare how ``asset<target_type_id>`` decodes into its target.

        The target is ``T`` or ``list<T>`` whose innermost atom must be
        registered (the provider produces values of it); asset targets are
        rejected - EXACTLY ONE coercion step, never provider chains. One
        provider per target: a conflict is a composition bug, never
        resolved silently."""
        if not provider_id:
            raise ValueError(f"{target_type_id}: decoder provider_id must be non-empty")
        atom = target_type_id
        while (inner := parse_list_type_id(atom)) is not None:
            atom = inner
        if parse_asset_type_id(atom) is not None:
            raise ValueError(
                f"asset decode targets must not themselves be asset types "
                f"(no provider chains): {target_type_id}"
            )
        if atom not in self._types:
            raise KeyError(f"unregistered decode target type: {atom}")
        if target_type_id in self._asset_decoders:
            raise ValueError(f"asset decoder already registered for target: {target_type_id}")
        spec = AssetDecoderSpec(
            target_type_id=target_type_id, provider_id=provider_id, decode=decode
        )
        self._asset_decoders[target_type_id] = spec
        return spec

    def asset_decoder_for(self, target_type_id: TypeId) -> AssetDecoderSpec | None:
        return self._asset_decoders.get(target_type_id)

    def asset_decoder_targets(self) -> tuple[TypeId, ...]:
        """Declared decode targets for portable validation metadata."""
        return tuple(self._asset_decoders)

    def register_batch_merge(
        self,
        type_id: TypeId,
        *,
        provider_id: str,
        merge: Callable[[Sequence[object]], object],
    ) -> BatchMergeSpec:
        """Declare that ``type_id`` is internally batched and a list of its
        values merges into one. Atoms only (a list-of-lists never batches),
        one merge per type."""
        if not provider_id:
            raise ValueError(f"{type_id}: merge provider_id must be non-empty")
        if "<" in type_id or ">" in type_id:
            raise ValueError(f"batch merges register on atom type ids, got: {type_id}")
        if type_id not in self._types:
            raise KeyError(f"unregistered value type: {type_id}")
        if type_id in self._batch_merges:
            raise ValueError(f"batch merge already registered for type: {type_id}")
        spec = BatchMergeSpec(type_id=type_id, provider_id=provider_id, merge=merge)
        self._batch_merges[type_id] = spec
        return spec

    def batch_merge_for(self, type_id: TypeId) -> BatchMergeSpec | None:
        return self._batch_merges.get(type_id)

    def register_type_equivalence(
        self,
        left_type_id: TypeId,
        right_type_id: TypeId,
        *,
        provider_id: str,
    ) -> None:
        """Declare two registered atoms as symmetric wire-compatible types.

        The provider guarantees that either type's encoded bytes are valid
        under the other type's codec. Its stable identity joins fingerprints
        when a value crosses the equivalence; bump it when that contract
        changes. Each atom may belong to at most one pair, keeping admission
        deterministic.
        """
        if not provider_id:
            raise ValueError("type-equivalence provider_id must be non-empty")
        if left_type_id == right_type_id:
            raise ValueError("type equivalence requires two different type ids")
        for type_id in (left_type_id, right_type_id):
            if "<" in type_id or ">" in type_id:
                raise ValueError(f"type equivalence requires atom type ids, got: {type_id}")
            if type_id not in self._types:
                raise KeyError(f"unregistered value type: {type_id}")

        left = self._type_equivalences.get(left_type_id)
        right = self._type_equivalences.get(right_type_id)
        requested_left = (right_type_id, provider_id)
        requested_right = (left_type_id, provider_id)
        if left == requested_left and right == requested_right:
            return
        if left is not None or right is not None:
            occupied = left_type_id if left is not None else right_type_id
            raise ValueError(f"type equivalence already registered for: {occupied}")
        self._type_equivalences[left_type_id] = requested_left
        self._type_equivalences[right_type_id] = requested_right

    def equivalent_type(self, type_id: TypeId) -> TypeId | None:
        """The explicitly registered counterpart of an atom, if any."""
        equivalence = self._type_equivalences.get(type_id)
        return None if equivalence is None else equivalence[0]

    def bridge_equivalent(self, value: Value, target_type_id: TypeId) -> Value:
        """Encode and restamp a value for its registered equivalent type.

        An already encoded payload keeps its bytes (and shared-memory owner)
        without decoding. An in-process payload is encoded exactly once with
        the source codec. The target decoder makes the returned value usable
        by an in-process target too; an isolated target replaces it with its
        own decoder when the bytes cross the boundary.
        """
        equivalence = self._type_equivalences.get(value.type_id)
        if equivalence is None or equivalence[0] != target_type_id:
            raise ValueError(f"no type equivalence from '{value.type_id}' to '{target_type_id}'")
        provider_id = equivalence[1]
        source = self.spec(value.type_id)
        target = self.spec(target_type_id)
        if isinstance(value.payload, EncodedPayload) and value.payload.type_id == value.type_id:
            payload = value.payload.restamped(target_type_id, target.decode)
        else:
            payload = EncodedPayload(
                target_type_id,
                source.encode(value.payload.load()),
                target.decode,
            )
        fingerprint = _equivalent_fingerprint(
            provider_id, value.type_id, target_type_id, value.fingerprint
        )
        return Value(
            type_id=target_type_id,
            fingerprint=fingerprint,
            meta=value.meta,
            payload=payload,
        )

    def render(
        self,
        value: Value,
        kind: str | None = None,
        parameters: Mapping[str, str] | None = None,
    ) -> Rendition:
        """Render a value's payload into client bytes.

        ``kind=None`` means the type's default rendition. Raises KeyError
        when the type has no (matching) rendition and UnresolvablePayload
        when the payload's type is not loadable in this process."""
        specs = self._renditions.get(value.type_id, [])
        if kind is None:
            spec = next((s for s in specs if s.default), None)
        else:
            spec = next((s for s in specs if s.kind == kind), None)
        if spec is None:
            wanted = kind if kind is not None else "<default>"
            raise KeyError(f"{value.type_id}: no rendition {wanted!r}")
        normalized = parameters or {}
        if bool(normalized) != bool(spec.parameters):
            if spec.parameters:
                raise InvalidRenditionRequest(f"{spec.kind} rendition requires parameters")
            raise InvalidRenditionRequest(f"{spec.kind} rendition accepts no parameters")
        renderer = cast(Callable[[object, Mapping[str, str]], bytes], spec.render)
        data = (
            renderer(value.payload.load(), normalized)
            if spec.parameters
            else cast(Callable[[object], bytes], spec.render)(value.payload.load())
        )
        return Rendition(
            kind=spec.kind,
            mime=spec.mime_for(value.meta.entries),
            data=data,
        )

    async def render_async(
        self,
        value: Value,
        kind: str | None = None,
        parameters: Mapping[str, str] | None = None,
    ) -> Rendition:
        specs = self._renditions.get(value.type_id, [])
        if kind is None:
            spec = next((item for item in specs if item.default), None)
        else:
            spec = next((item for item in specs if item.kind == kind), None)
        if spec is not None and spec.render_async is not None:
            return await spec.render_async(value, parameters or {})
        return self.render(value, kind, parameters)

    def inline_of(self, value: Value) -> int | float | bool | str | None:
        """The value's inline scalar under the central policy, or None.

        Policy (DESIGN 3.5): only types with a declared inline fn, never
        lists (children inline individually through descriptor recursion),
        only JSON-native scalars out, strings capped at INLINE_STRING_CAP.
        None always means "not inline" - absence has its own channel, and
        core types never produce None."""
        spec = self._types.get(value.type_id)
        if spec is None or spec.inline is None:
            return None
        try:
            scalar = spec.inline(value.payload.load())
        except Exception:
            # Inline is a peek nicety riding on event/descriptor
            # serialization paths; a buggy pack inline fn must degrade to
            # "not inline", never fail the run or the wire encoding.
            return None
        if isinstance(scalar, bool | int | float):
            return scalar
        if isinstance(scalar, str) and len(scalar) <= INLINE_STRING_CAP:
            return scalar
        return None

    def __contains__(self, type_id: TypeId) -> bool:
        if parse_asset_type_id(type_id) is not None:
            # Parametric asset ids resolve to the base asset spec: an
            # asset<T> value IS a base-asset envelope with a more specific
            # stamp, so it is codable exactly where dinkster.asset is - when
            # the stamp is well-formed and its innermost atom is real.
            atom = runtime_type_atom(type_id)
            return (
                atom is not None and ASSET_BASE_TYPE in self._types and self.has_type_identity(atom)
            )
        return type_id in self._types

    def has_type_identity(self, type_id: TypeId) -> bool:
        """Whether a codec or a worker declaration establishes this atom."""
        return type_id in self._types or any(
            type_id in declared for declared in self._declared_types.values()
        )

    def type_ids(self) -> tuple[TypeId, ...]:
        """All registered atom type ids, in registration order. Tooling
        (``dinkster doctor``) diffs this around a pack's types entry to see
        what the pack registered and how."""
        return tuple(self._types)

    def spec(self, type_id: TypeId) -> TypeSpec:
        base = type_id
        if parse_asset_type_id(type_id) is not None:
            # Parametric asset ids share the base asset spec (see
            # __contains__): same codec, same digest fingerprint - the
            # stamp refines the TYPE, never the value's identity. The same
            # registration bar as __contains__ applies: a well-formed stamp
            # over an unregistered atom must not wrap or code.
            atom = runtime_type_atom(type_id)
            if atom is None:
                raise KeyError(f"malformed asset type id: {type_id!r}")
            # References need the target's identity, not its local codec.
            if not self.has_type_identity(atom):
                raise KeyError(
                    f"cannot resolve '{type_id}': its decode-target atom "
                    f"'{atom}' is not a registered value type"
                )
            base = ASSET_BASE_TYPE
            if base not in self._types:
                raise KeyError(
                    f"cannot resolve '{type_id}': the base asset type "
                    f"'{ASSET_BASE_TYPE}' is not registered in this process"
                )
        try:
            return self._types[base]
        except KeyError:
            raise KeyError(f"unregistered value type: {type_id}") from None

    def input_object(self, type_id: TypeId, obj: object) -> object:
        """Convert a resolved typed input, recursively through list storage.

        Asset references stay references until worker-side decoding supplies
        the target type. Unknown types keep the default pass-through behavior.
        """
        if isinstance(obj, Value):
            if obj.type_id != type_id:
                raise TypeError(f"{type_id} rejects envelope of type {obj.type_id}")
            children = value_children(obj)
            if children is not None:
                converted = [self.input_object(child.type_id, child) for child in children]
                return tuple(converted) if parse_tuple_type_id(type_id) is not None else converted
            obj = obj.resolve()
        positions = parse_tuple_type_id(type_id)
        if positions is not None:
            return tuple(
                self.input_object(position, item)
                for position, item in zip(positions, cast(Sequence[object], obj), strict=True)
            )
        element = parse_list_type_id(type_id)
        if element is not None:
            return [self.input_object(element, item) for item in cast(Sequence[object], obj)]
        spec = self._types.get(type_id)
        return (
            spec.input_convert(obj) if spec is not None and spec.input_convert is not None else obj
        )

    def wrap(self, type_id: TypeId, obj: object) -> Value:
        """Build an envelope for a raw object. Worker shims call this; node
        authors never do (hazard H9).

        Stream type ids require a matching ``MediaStream``. List type ids
        (``list<element>``) wrap structurally: each element is
        wrapped as the element type - recursively for nested lists - and the
        children stay full envelopes inside the list value (DESIGN 3.13).
        Asset ids (``asset<target>``) wrap through the base asset spec but
        keep the parametric stamp."""
        if isinstance(obj, Value):
            if obj.type_id != type_id:
                raise TypeError(f"{type_id} rejects envelope of type {obj.type_id}")
            return obj
        positions = parse_tuple_type_id(type_id)
        if positions is not None:
            if not isinstance(obj, tuple) or len(cast(tuple[object, ...], obj)) != len(positions):
                raise TypeError(f"{type_id} expects a tuple of {len(positions)} positions")
            return make_tuple_value(
                tuple(
                    self.wrap(position, item)
                    for position, item in zip(positions, cast(tuple[object, ...], obj), strict=True)
                )
            )
        stream_element = parse_stream_type_id(type_id)
        if stream_element is not None:
            if not isinstance(obj, MediaStream):
                raise TypeError(f"{type_id} expects a matching MediaStream")
            if obj.closed:
                raise ValueError("cannot wrap a closed MediaStream")
            self.spec(stream_element)
            if obj.type_id == stream_element:
                return make_stream_value(obj)
            equivalence = self._type_equivalences.get(obj.type_id)
            if equivalence is None or equivalence[0] != stream_element:
                raise TypeError(f"{type_id} expects a matching MediaStream")
            source = make_stream_value(obj)
            target = make_stream_value(obj, stream_element)
            return Value(
                type_id=target.type_id,
                fingerprint=_equivalent_fingerprint(
                    equivalence[1], obj.type_id, stream_element, source.fingerprint
                ),
                meta=target.meta,
                payload=target.payload,
            )
        if isinstance(obj, MediaStream):
            raise TypeError(f"{type_id} does not accept a MediaStream")
        element_type = parse_list_type_id(type_id)
        if element_type is not None:
            if isinstance(obj, str | bytes) or not isinstance(obj, Sequence):
                raise TypeError(
                    f"{type_id} expects a sequence of elements, got {type(obj).__name__}"
                )
            if element_type == "*":
                if any(not isinstance(item, Value) for item in cast("Sequence[object]", obj)):
                    raise TypeError("list<*> requires typed child envelopes")
                return make_list_value(element_type, cast("Sequence[Value]", obj))
            children = tuple(
                self.wrap(element_type, item) for item in cast("Sequence[object]", obj)
            )
            return make_list_value(element_type, children)
        spec = self.spec(type_id)
        if spec.coerce is not None:
            obj = spec.coerce(obj)
        if spec.fingerprint is not None:
            fp = spec.fingerprint(obj)
        else:
            fp = stable_hash([type_id.encode("utf-8"), spec.encode(obj)])
        meta = ValueMeta(dict(spec.meta(obj))) if spec.meta is not None else ValueMeta()
        return Value(type_id=type_id, fingerprint=fp, meta=meta, payload=PyObjPayload(obj))


def _coerce_combo(obj: object) -> str | bool | int | float:
    if type(obj) not in (str, bool, int, float):
        raise TypeError(f"COMBO expects a JSON scalar, got {type(obj).__name__}")
    if type(obj) is int and abs(obj) > 2**53 - 1:
        raise ValueError("COMBO expects a JSON-safe integer")
    if type(obj) is float and not math.isfinite(obj):
        raise ValueError("COMBO expects a finite number")
    return cast("str | bool | int | float", obj)


def _validate_combo_encoded(data: bytes, _metadata: Mapping[str, object]) -> None:
    if not data.startswith(b"json:"):
        raise ValueError("COMBO expects canonical JSON encoding")
    obj = default_decode(data)
    _coerce_combo(obj)
    if default_encode(obj) != data:
        raise ValueError("COMBO expects canonical JSON encoding")


def _coerce_histogram(obj: object) -> list[int]:
    if type(obj) is not list:
        raise TypeError("HISTOGRAM requires a list of integers")
    values = cast("list[object]", obj)
    if any(type(value) is not int for value in values):
        raise TypeError("HISTOGRAM requires a list of integers")
    return cast("list[int]", values)


def _validate_histogram_encoded(data: bytes, _metadata: Mapping[str, object]) -> None:
    if not data.startswith(b"json:"):
        raise ValueError("HISTOGRAM expects canonical JSON encoding")
    value = _coerce_histogram(default_decode(data))
    if default_encode(value) != data:
        raise ValueError("HISTOGRAM expects canonical JSON encoding")


def register_histogram_type(registry: TypeRegistry) -> TypeSpec:
    """Register the built-in structural histogram value without replacing an authority."""
    if HISTOGRAM_TYPE in registry:
        return registry.spec(HISTOGRAM_TYPE)
    return registry.register(
        HISTOGRAM_TYPE,
        coerce=_coerce_histogram,
        validate_encoded=_validate_histogram_encoded,
    )


def register_core_types(registry: TypeRegistry) -> None:
    """Primitives are envelope types like everything else (hazard H2)."""
    # Imported here because curve depends on this module's TypeRegistry.
    from .curve import register_curve_type

    # Core scalars inline as themselves (DESIGN 3.5): peeking an int or a
    # short string never needs a payload round trip. core.absent does NOT
    # inline - absence has its own channel, never a smuggled null.
    for type_id in (CORE_INT, CORE_FLOAT, CORE_STRING):
        registry.register(type_id, inline=lambda obj: obj)
    registry.register_rendition(
        CORE_STRING,
        "text",
        mime="text/plain",
        render=lambda obj: cast(str, obj).encode("utf-8"),
    )
    registry.register(
        CORE_COMBO,
        coerce=_coerce_combo,
        inline=lambda obj: obj,
        validate_encoded=_validate_combo_encoded,
    )
    registry.register(CORE_BOOLEAN, inline=lambda obj: obj)
    registry.register(DICT_TYPE)
    registry.register(ARRAY_TYPE)
    register_histogram_type(registry)
    register_curve_type(registry)
    # core.absent registers with the default codec (payload is None, which
    # JSON-encodes) so absences cross boundaries and caches like any value.
    registry.register(CORE_ABSENT)
