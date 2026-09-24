"""Workflow persistence surface: client upload + scoped library (DESIGN
roadmap: "Workflow persistence: workflows as assets" + the identity/scoping
constraint).

Two layers with a hard boundary between them:

- BYTES are immutable, content-addressed, globally deduplicated. Generic
  uploads remain opaque; ImageDocument uses a dedicated validation and
  dependency-adoption route before its bytes become a trusted document.
- RECORDS are mutable and scoped: display name, labels, folder, pointing
  at bytes by digest. Every record operation takes an explicit scope
  (single-user mode is the reserved scope "local"); membership is scoped
  even though bytes deduplicate globally, so knowing a digest or record
  id grants nothing across scopes.

Surface (present only when create_app got a ServerLibrary):

- POST   /api/assets                     upload raw bytes; Content-Type is
                                         recorded nowhere (records carry
                                         mediaType); optional X-Dinkster-Digest
                                         header verified against the bytes;
                                         413 over the limit; -> {"digest"}
                                         (201 new, 200 already-held)
- POST   /api/assets/media               bounded classified media ingest with
                                         exact scoped immutable authority
- GET    /api/assets/{digest}            bytes back, digest-immutable:
                                         quoted digest ETag, If-None-Match
                                         304, forever cache lifetime (the
                                         icon/blueprint contract)
- GET    /api/assets/{digest}/metadata   digest-keyed probe facts (probe.py:
                                         safetensors headers today); 404
                                         when not held; immutable-cache
- GET    /api/assets/{digest}/sources    provenance record (leads, never
                                         authorities); empty record when
                                         unknown  [with a ProvenanceStore]
- POST   /api/assets/{digest}/sources    register leads {urls, license?,
                                         note?}; additive merge
                                         [with a ProvenanceStore]
- POST   /api/library                    create record {scope, name, digest,
                                         mediaType, labels?, folder?}; 409
                                         when the digest is not held here
                                         (upload first)
- GET    /api/library?scope=&q=&label=&limit=&cursor=
                                         query-first, cursor-paged; the
                                         cursor binds the query that made
                                         it (400 on mismatch)
- GET    /api/library/{id}?scope=        one record; wrong scope is 404
- PATCH  /api/library/{id}               {scope, revision, name?, digest?,
                                         mediaType?, labels?, folder?};
                                         409 stale-revision, 404 unknown
- DELETE /api/library/{id}?scope=        removes the record, never bytes
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, TypeVar, cast

from aiohttp import web
from dinkster_assets import (
    LATENT_ASSET_KIND,
    LATENT_MEDIA_TYPE,
    MAX_LATENT_DATA_BYTES,
    AssetError,
    AssetIntegrityError,
    AssetResolver,
    AssetVault,
    LibraryStore,
    MediaGrant,
    PackAssetCatalog,
    ProvenanceRecord,
    ProvenanceStore,
    PublicAcquisitionReceiptStore,
    PublicAcquisitionSourceV1,
    RemoteSource,
    StaleRevision,
    classify_media_file,
    digest_bytes,
    is_digest,
    new_hasher,
    open_verified,
    parse_latent_asset,
    probe_handle,
    valid_vae_hint,
)
from dinkster_values import (
    LATENT_UPLOAD_HEADROOM_BYTES,
    MEBIBYTE,
    MEDIA_AUDIO_UPLOAD_LIMIT_BYTES,
    MEDIA_IMAGE_UPLOAD_LIMIT_BYTES,
    MEDIA_MODEL3D_UPLOAD_LIMIT_BYTES,
    MEDIA_VIDEO_UPLOAD_LIMIT_BYTES,
)

from .asset_stream import open_verified_sized, stream_verified
from .auth import LOCAL_PRINCIPAL, principal_for
from .paging import decode_cursor, encode_cursor

__all__ = ["LIBRARY_KEY", "ServerLibrary", "add_library_routes"]

DIGEST_HEADER = "X-Dinkster-Digest"
IMAGE_DOCUMENT_MEDIA_TYPE = "application/vnd.dinkster.image-document+json"
#: Per-kind byte bounds for POST /api/assets/media, mirroring the limits the
#: frontend asset picker surfaces so the backend enforces them instead of
#: trusting the client. The mapping is also the authority on which media
#: kinds the ingest endpoint accepts.
_MEDIA_UPLOAD_LIMITS: Mapping[str, int] = MappingProxyType(
    {
        "media/image": MEDIA_IMAGE_UPLOAD_LIMIT_BYTES,
        "media/audio": MEDIA_AUDIO_UPLOAD_LIMIT_BYTES,
        "media/video": MEDIA_VIDEO_UPLOAD_LIMIT_BYTES,
        "media/model3d": MEDIA_MODEL3D_UPLOAD_LIMIT_BYTES,
    }
)
_MEDIA_KINDS = frozenset(_MEDIA_UPLOAD_LIMITS)
_MEDIA_TYPES = frozenset(
    (
        "image/png",
        "image/jpeg",
        "image/webp",
        "audio/wav",
        "audio/flac",
        "audio/mpeg",
        "audio/ogg",
        "audio/webm",
        "audio/mp4",
        "video/mp4",
        "video/webm",
        "model/gltf-binary",
        "model/ply",
    )
)
_MEDIA_NAME_LIMIT = 255
_MEDIA_UPLOAD_CHUNK_SIZE = 8 * MEBIBYTE
_LATENT_UPLOAD_IDLE_SECONDS = 30.0
_LATENT_UPLOAD_HEADROOM = LATENT_UPLOAD_HEADROOM_BYTES
_T = TypeVar("_T")


@dataclass(frozen=True)
class ServerLibrary:
    """What create_app needs to serve persistence: a writable vault for
    the bytes and the scoped record store. Bundled because records that
    cannot check byte presence, or uploads with nowhere to browse, are
    each half a feature."""

    vault: AssetVault
    store: LibraryStore
    upload_limit: int = 16 * MEBIBYTE
    #: Per-kind bounds for classified media ingest; must cover _MEDIA_KINDS.
    media_upload_limits: Mapping[str, int] = _MEDIA_UPLOAD_LIMITS
    #: Optional read-only fallback for digest GETs (e.g. filesystem
    #: mounts, composed by the umbrella). Uploads never write here, and
    #: hits are STREAMED, not buffered - a mount can hold model-sized
    #: files, unlike the capped upload vault.
    resolver: AssetResolver | None = None
    #: Optional digest-keyed source leads (where bytes can be obtained).
    #: Enables the sources routes and job-preflight acquisition; records
    #: are leads, never authorities - fetched bytes verify against the
    #: digest or land nowhere.
    provenance: ProvenanceStore | None = None
    #: Digest-keyed probe results (probe_file). Content is immutable by
    #: construction, so entries never invalidate; unbounded in principle,
    #: bounded in practice by the number of distinct assets ever probed.
    probe_cache: dict[str, dict[str, object]] = field(default_factory=dict[str, dict[str, object]])
    #: Optional pack-declared asset catalog ([[pack.assets]]), maintained
    #: by the composed surface: pack artifact roots for verified packaged
    #: acquisition plus declared needs by digest and by requiring node
    #: type. Job preflight consults it; acquisition still verifies every
    #: byte and still waits for digest-exact consent.
    pack_assets: PackAssetCatalog | None = None
    #: Receipt evidence is written only by verified public HTTPS acquisition.
    receipts: PublicAcquisitionReceiptStore | None = None
    #: Current trusted declarations, kept separate from advisory provenance.
    public_sources_for: Callable[[str], Sequence[PublicAcquisitionSourceV1]] | None = None
    #: Optional LAN transport bridge. It is called only from preflight's worker thread.
    lan_resolve: Callable[[str], Path | None] | None = None
    #: Notify the transport owner after verified local materialization.
    p2p_acquired: Callable[[], None] | None = None
    #: Host-owned metadata classifier; this package does not depend on inference.
    model_output_profile: Callable[[Path, BinaryIO, str, int], Mapping[str, object]] | None = None
    #: Serialize the two-resource media publication so one request owns the
    #: byte/grant created result used for exact 200/201 semantics.
    media_ingest_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, init=False, repr=False, compare=False
    )
    latent_ingest_slots: asyncio.Semaphore = field(
        default_factory=lambda: asyncio.Semaphore(2), init=False, repr=False, compare=False
    )
    latent_reservation_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, init=False, repr=False, compare=False
    )
    latent_reserved_bytes: list[int] = field(
        default_factory=lambda: [0], init=False, repr=False, compare=False
    )
    image_document_decode_slots: asyncio.Semaphore = field(
        default_factory=lambda: asyncio.Semaphore(2), init=False, repr=False, compare=False
    )
    image_document_decode_reserved_bytes: list[int] = field(
        default_factory=lambda: [0], init=False, repr=False, compare=False
    )
    image_document_render_slots: asyncio.Semaphore = field(
        default_factory=lambda: asyncio.Semaphore(1), init=False, repr=False, compare=False
    )

    def locate(self, digest: str) -> Path | None:
        """One local-presence answer for every surface (GETs, metadata,
        job preflight): the vault first, then the read-only fallback."""
        path = self.vault.resolve(digest)
        if path is None and self.resolver is not None:
            path = self.resolver.resolve(digest)
        return path


LIBRARY_KEY: web.AppKey[ServerLibrary] = web.AppKey("server_library")


def _json_error(status: int, message: str) -> web.Response:
    return web.json_response({"error": message}, status=status)


def _media_error(status: int, code: str, message: str) -> web.Response:
    return web.json_response(
        {"error": {"code": f"asset.media.{code}", "message": message}},
        status=status,
    )


def _safe_asset_name(name: str) -> bool:
    return bool(
        name
        and name == name.strip()
        and len(name) <= _MEDIA_NAME_LIMIT
        and name not in {".", ".."}
        and "/" not in name
        and "\\" not in name
        and not any(ord(character) < 32 or ord(character) == 127 for character in name)
    )


def _base_media_type(value: str) -> str:
    return value.split(";", 1)[0].strip().lower()


def _spool_media_upload_chunk(temporary: Any, hasher: Any, chunk: bytes) -> None:
    temporary.write(chunk)
    hasher.update(chunk)


async def run_owned_thread(function: Callable[[], _T]) -> _T:
    """Return borrowed resources only after an offloaded operation settles."""
    task = asyncio.create_task(asyncio.to_thread(function))
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        if not task.cancelled():
            task.exception()
        raise asyncio.CancelledError
    return task.result()


async def _write_media_upload_chunk(temporary: Any, hasher: Any, chunk: bytes) -> None:
    await run_owned_thread(lambda: _spool_media_upload_chunk(temporary, hasher, chunk))


def _finish_upload_spool(temporary: BinaryIO) -> None:
    temporary.flush()
    os.fsync(temporary.fileno())
    temporary.close()


def _parse_latent_path(path: Path) -> None:
    with path.open("rb") as source:
        parse_latent_asset(source)


async def _json_body(request: web.Request) -> dict[str, Any]:
    # Mirrors app.py's helper (importing it back would be a cycle:
    # app.py imports this module).
    try:
        raw = cast(object, await request.json())
    except json.JSONDecodeError as exc:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": f"invalid JSON: {exc}"}),
            content_type="application/json",
        ) from exc
    if not isinstance(raw, dict):
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "request body must be an object"}),
            content_type="application/json",
        )
    return cast(dict[str, Any], raw)


def _string_labels(value: object) -> list[str] | None:
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in cast(list[Any], value)
    ):
        return None
    return cast(list[str], value)


async def handle_upload(request: web.Request) -> web.Response:
    """Verified, idempotent ingest. The response digest is canonical
    truth: a client that declared X-Dinkster-Digest gets integrity for free
    (mismatch is 400, nothing lands); one that didn't learns the digest
    the backend computed. Re-uploading held bytes is a cheap 200."""
    library = request.app[LIBRARY_KEY]
    if request.content_length is not None and request.content_length > library.upload_limit:
        return _json_error(413, f"upload exceeds {library.upload_limit} bytes")
    received = bytearray()
    async for chunk in request.content.iter_chunked(64 * 1024):
        received.extend(chunk)
        if len(received) > library.upload_limit:
            return _json_error(413, f"upload exceeds {library.upload_limit} bytes")
    data = bytes(received)
    if not data:
        return _json_error(400, "empty upload")
    digest = await asyncio.to_thread(digest_bytes, data)
    declared = request.headers.get(DIGEST_HEADER)
    if declared is not None and declared != digest:
        return _json_error(400, f"declared digest {declared} does not match bytes ({digest})")

    def ingest() -> bool:
        if library.vault.has(digest):
            return False
        with library.vault.writer(digest) as writer:
            writer.write(data)
            writer.commit()
        return True

    stored = await asyncio.to_thread(ingest)
    return web.json_response({"digest": digest}, status=201 if stored else 200)


def _media_query(request: web.Request) -> tuple[str, str, str] | web.Response:
    expected = {"scope", "kind", "name"}
    if set(request.query) != expected or any(
        len(request.query.getall(field, [])) != 1 for field in expected
    ):
        return _media_error(
            400,
            "invalid_query",
            "exactly one scope, kind, and name query field is required",
        )
    scope = request.query["scope"]
    kind = request.query["kind"]
    name = request.query["name"]
    if not scope or scope != scope.strip() or any(character.isspace() for character in scope):
        return _media_error(400, "invalid_scope", "scope must be non-empty and whitespace-free")
    if kind not in _MEDIA_KINDS:
        return _media_error(400, "invalid_kind", "kind is not a supported media kind")
    if not _safe_asset_name(name):
        return _media_error(
            400,
            "invalid_name",
            "name must be a safe display basename of at most 255 characters",
        )
    return scope, kind, name


def _latent_query(request: web.Request) -> tuple[str, str] | web.Response:
    expected = {"scope", "name"}
    if set(request.query) != expected or any(
        len(request.query.getall(field, [])) != 1 for field in expected
    ):
        return _media_error(400, "invalid_query", "exactly one scope and name is required")
    scope, name = request.query["scope"], request.query["name"]
    if not scope or scope != scope.strip() or any(character.isspace() for character in scope):
        return _media_error(400, "invalid_scope", "scope must be non-empty and whitespace-free")
    if not _safe_asset_name(name):
        return _media_error(400, "invalid_name", "name must be a safe display basename")
    return scope, name


async def handle_latent_upload(request: web.Request) -> web.Response:
    """Stream, validate, and publish one classified latent asset."""
    query = _latent_query(request)
    if isinstance(query, web.Response):
        return query
    scope, name = query
    principal = principal_for(request)
    if principal is not LOCAL_PRINCIPAL and not principal.allows_in(scope, "assets:write"):
        return _media_error(403, "scope_forbidden", f"scope {scope} does not grant assets:write")
    media_type = request.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    if media_type not in {LATENT_MEDIA_TYPE, "application/octet-stream"}:
        return _media_error(415, "unsupported_content_type", "unsupported latent Content-Type")
    declared = request.headers.get(DIGEST_HEADER)
    if declared is not None and not is_digest(declared):
        return _media_error(400, "invalid_digest", "X-Dinkster-Digest is not canonical")
    if request.content_length is not None and request.content_length > MAX_LATENT_DATA_BYTES:
        return _media_error(413, "too_large", "latent upload exceeds 1 GiB")

    library = request.app[LIBRARY_KEY]
    try:
        await asyncio.wait_for(library.latent_ingest_slots.acquire(), timeout=0.001)
    except TimeoutError:
        return _media_error(429, "busy", "at most two latent uploads may run concurrently")
    handle: BinaryIO | None = None
    temp_name: str | None = None
    reserved = False
    try:
        async with library.latent_reservation_lock:
            free = await asyncio.to_thread(lambda: shutil.disk_usage(library.vault.root).free)
            required = (
                library.latent_reserved_bytes[0] + MAX_LATENT_DATA_BYTES + _LATENT_UPLOAD_HEADROOM
            )
            if free < required:
                return _media_error(
                    507, "disk_reserve", "latent upload would consume disk headroom"
                )
            library.latent_reserved_bytes[0] += MAX_LATENT_DATA_BYTES
            reserved = True
        handle = cast(
            BinaryIO,
            tempfile.NamedTemporaryFile(
                mode="w+b", prefix=".latent-upload-", dir=library.vault.root, delete=False
            ),
        )
        temp_name = handle.name
        assert temp_name is not None
        temporary_path = Path(temp_name)
        hasher = new_hasher()
        size = 0
        try:
            while not request.content.at_eof():
                chunk = await asyncio.wait_for(
                    request.content.read(_MEDIA_UPLOAD_CHUNK_SIZE),
                    timeout=_LATENT_UPLOAD_IDLE_SECONDS,
                )
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_LATENT_DATA_BYTES:
                    return _media_error(413, "too_large", "latent upload exceeds 1 GiB")
                await _write_media_upload_chunk(handle, hasher, chunk)
        except TimeoutError:
            return _media_error(408, "idle_timeout", "latent upload body became idle")
        except (ConnectionError, OSError, web.RequestPayloadError):
            return _media_error(400, "interrupted", "latent upload was interrupted")
        if size == 0:
            return _media_error(400, "empty", "latent upload is empty")
        await run_owned_thread(lambda: _finish_upload_spool(handle))
        try:
            await run_owned_thread(lambda: _parse_latent_path(temporary_path))
        except AssetError:
            return _media_error(415, "invalid_latent", "uploaded bytes are not a supported latent")
        digest = "blake3:" + hasher.hexdigest()
        if declared is not None and declared != digest:
            return _media_error(409, "digest_mismatch", "declared digest does not match bytes")
        grant = MediaGrant(
            scope=scope,
            digest=digest,
            kind=LATENT_ASSET_KIND,
            media_type=LATENT_MEDIA_TYPE,
            extension="latent",
            byte_size=size,
        )

        def publish() -> tuple[bool, bool]:
            bytes_created = False
            try:
                with library.vault.writer(digest) as writer, temporary_path.open("rb") as source:
                    while chunk := source.read(_MEDIA_UPLOAD_CHUNK_SIZE):
                        writer.write(chunk)
                    _path, bytes_created = writer.commit_with_result()
                _stored, grant_created = library.store.grant_latent_result(grant)
                return bytes_created, grant_created
            except Exception:
                if bytes_created:
                    library.vault.delete(digest)
                raise

        # Starting this worker is the commit point. Cancellation waits for the
        # publication result so bytes and scoped authority cannot be abandoned
        # mid-mutation. Serialization makes rollback of newly created bytes safe.
        await library.media_ingest_lock.acquire()
        try:
            bytes_created, grant_created = await run_owned_thread(publish)
        finally:
            library.media_ingest_lock.release()
        body = {
            "asset": {
                "digest": digest,
                "name": name,
                "size": size,
                "mediaType": LATENT_MEDIA_TYPE,
                "virtualPath": "",
            },
            "kind": LATENT_ASSET_KIND,
        }
        return web.json_response(body, status=201 if bytes_created or grant_created else 200)
    except (AssetError, OSError, sqlite3.Error):
        return _media_error(503, "storage_unavailable", "latent storage is unavailable")
    finally:
        if handle is not None:
            handle.close()
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
        if reserved:
            library.latent_reserved_bytes[0] -= MAX_LATENT_DATA_BYTES
        library.latent_ingest_slots.release()


async def handle_media_upload(request: web.Request) -> web.Response:
    """Bounded byte-derived media ingest with exact scoped authority."""
    query = _media_query(request)
    if isinstance(query, web.Response):
        return query
    scope, claimed_kind, name = query
    principal = principal_for(request)
    if principal is not LOCAL_PRINCIPAL and not principal.allows_in(scope, "assets:write"):
        return _media_error(
            403,
            "scope_forbidden",
            f"scope {scope} does not grant assets:write",
        )

    raw_media_type = request.headers.get("Content-Type", "")
    media_type = raw_media_type.split(";", 1)[0].strip().lower()
    if media_type not in _MEDIA_TYPES:
        return _media_error(
            415,
            "unsupported_content_type",
            "Content-Type is not a supported media type",
        )

    declared = request.headers.get(DIGEST_HEADER)
    if declared is not None and not is_digest(declared):
        return _media_error(
            400, "invalid_digest", "X-Dinkster-Digest must be a canonical asset digest"
        )

    library = request.app[LIBRARY_KEY]
    # The claimed kind is binding: classification later rejects any body
    # whose actual kind differs (kind_mismatch), so an oversized image
    # cannot ride in under a roomier video claim.
    limit = library.media_upload_limits[claimed_kind]
    if request.content_length is not None and request.content_length > limit:
        return _media_error(413, "too_large", f"media upload exceeds {limit} bytes")
    temporary = tempfile.NamedTemporaryFile(
        mode="w+b", prefix=".media-upload-", dir=library.vault.root, delete=False
    )
    temporary_path = Path(temporary.name)
    hasher = new_hasher()
    received = 0
    publication_handoff = asyncio.Event()
    try:
        try:
            async for chunk in request.content.iter_chunked(_MEDIA_UPLOAD_CHUNK_SIZE):
                received += len(chunk)
                if received > limit:
                    return _media_error(
                        413,
                        "too_large",
                        f"media upload exceeds {limit} bytes",
                    )
                await _write_media_upload_chunk(temporary, hasher, chunk)
        except (ConnectionError, OSError, web.RequestPayloadError):
            return _media_error(400, "interrupted", "media upload was interrupted")
        finally:
            temporary.close()
        if not received:
            return _media_error(400, "empty", "media upload is empty")

        digest = "blake3:" + hasher.hexdigest()
        if declared is not None and declared != digest:
            return _media_error(
                409, "digest_mismatch", "X-Dinkster-Digest does not match the uploaded bytes"
            )
        try:
            classification = await asyncio.to_thread(classify_media_file, temporary_path)
        except AssetError:
            return _media_error(
                415, "unsupported_media", "uploaded bytes are malformed or unsupported"
            )
        if claimed_kind != classification.kind:
            return _media_error(409, "kind_mismatch", "kind does not match the uploaded bytes")
        if media_type != classification.media_type:
            return _media_error(
                409,
                "content_type_mismatch",
                "Content-Type does not match the uploaded bytes",
            )

        grant = MediaGrant(
            scope=scope,
            digest=digest,
            kind=classification.kind,
            media_type=classification.media_type,
            extension=classification.extension,
            byte_size=received,
        )

        def publish() -> tuple[bool, bool]:
            try:
                with library.vault.writer(digest) as writer, temporary_path.open("rb") as source:
                    while chunk := source.read(64 * 1024):
                        writer.write(chunk)
                    _path, bytes_created = writer.commit_with_result()
                _stored, grant_created = library.store.grant_media_result(grant, temporary_path)
                return bytes_created, grant_created
            finally:
                temporary_path.unlink(missing_ok=True)

        try:
            bytes_created, grant_created = await run_media_publication(
                library, publish, handoff=publication_handoff
            )
        except (AssetError, OSError, sqlite3.Error):
            return _media_error(503, "storage_unavailable", "media storage is unavailable")

        body = {
            "asset": {
                "digest": digest,
                "name": name,
                "size": received,
                "mediaType": classification.media_type,
                "virtualPath": "",
            },
            "kind": classification.kind,
        }
        return web.json_response(body, status=201 if bytes_created or grant_created else 200)
    finally:
        temporary.close()
        if not publication_handoff.is_set():
            temporary_path.unlink(missing_ok=True)


async def run_media_publication(
    library: ServerLibrary,
    publish: Callable[[], tuple[bool, bool]],
    *,
    handoff: asyncio.Event | None = None,
) -> tuple[bool, bool]:
    """Keep serialization owned until an uncancellable worker actually settles."""
    await library.media_ingest_lock.acquire()
    task = asyncio.ensure_future(asyncio.to_thread(publish))
    if handoff is not None:
        handoff.set()
    handed_off = False
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        handed_off = True

        def release_lock(settled: asyncio.Task[tuple[bool, bool]]) -> None:
            library.media_ingest_lock.release()
            if not settled.cancelled():
                settled.exception()

        task.add_done_callback(release_lock)
        raise
    finally:
        if not handed_off:
            library.media_ingest_lock.release()


async def handle_asset_get(request: web.Request) -> web.StreamResponse:
    """Bytes for a digest, with the immutable-cache contract every other
    digest-addressed endpoint honors. Vault or mount, the body streams
    from the SAME descriptor the digest was verified on (asset_stream.py):
    an immutable ETag stamped on unverified bytes would let one stale file
    poison a browser cache forever."""
    digest = request.match_info["digest"]
    if not is_digest(digest):
        return _json_error(400, "malformed digest (expected 'blake3:<64 hex>')")
    library = request.app[LIBRARY_KEY]
    etag = f'"{digest}"'
    path = await asyncio.to_thread(library.locate, digest)
    if path is None:
        return _json_error(404, "asset not held here")
    if any(tag.value in (digest, "*") for tag in (request.if_none_match or ())):
        return web.Response(status=304, headers={"ETag": etag})
    try:
        handle, size = await open_verified_sized(path, digest)
    except AssetIntegrityError:
        # Deliberately no filesystem path in the public body.
        return web.json_response(
            {"error": f"asset content on disk no longer matches {digest}"},
            status=409,
            headers={"Cache-Control": "no-store"},
        )
    except AssetError as error:
        return _json_error(500, str(error))
    except OSError:
        return _json_error(404, "asset not held here")  # moved since resolve
    return await stream_verified(
        request,
        handle,
        size,
        headers={
            "ETag": etag,
            "Cache-Control": "private, max-age=31536000, immutable",
        },
    )


async def handle_asset_metadata(request: web.Request) -> web.Response:
    """GET /api/assets/{digest}/metadata - digest-keyed facts about held
    bytes (probe.py: safetensors headers today, more formats additively).

    Immutable by construction - the digest names the bytes, the bytes
    determine the answer - hence the forever-cache contract and the
    process-lifetime probe cache. Not held here is a plain 404: metadata
    describes local content, never remote promises (those are needs)."""
    digest = request.match_info["digest"]
    if not is_digest(digest):
        return _json_error(400, "malformed digest (expected 'blake3:<64 hex>')")
    library = request.app[LIBRARY_KEY]
    metadata = library.probe_cache.get(digest)
    if metadata is None:
        path = await asyncio.to_thread(library.locate, digest)
        if path is None:
            return _json_error(404, "asset not held here")

        def probe_verified() -> dict[str, object]:
            # Facts come from the PROVEN bytes: probing a reopened path
            # could cache the wrong file's metadata under this digest
            # forever (the response carries an immutable cache contract).
            with open_verified(path, digest) as handle:
                result = probe_handle(handle)
                try:
                    latent = parse_latent_asset(handle)
                except AssetError:
                    return result
                result["latent"] = {
                    "profile": latent.profile,
                    "streams": [
                        {
                            "name": tensor.name,
                            "dtype": tensor.dtype,
                            "shape": list(tensor.shape),
                            **({} if tensor.role is None else {"role": tensor.role}),
                        }
                        for tensor in latent.tensors
                    ],
                    "vaeHint": valid_vae_hint(latent),
                }
                return result

        try:
            metadata = await asyncio.to_thread(probe_verified)
        except AssetIntegrityError:
            return web.json_response(
                {"error": f"asset content on disk no longer matches {digest}"},
                status=409,
                headers={"Cache-Control": "no-store"},
            )
        except AssetError as error:
            return _json_error(500, str(error))
        except OSError:
            return _json_error(404, "asset not held here")  # moved since resolve
        library.probe_cache[digest] = metadata
    etag = f'"{digest}"'
    if any(tag.value in (digest, "*") for tag in (request.if_none_match or ())):
        return web.Response(status=304, headers={"ETag": etag})
    return web.json_response(
        {"digest": digest, "metadata": metadata},
        headers={
            "ETag": etag,
            "Cache-Control": "private, max-age=31536000, immutable",
        },
    )


async def handle_model_output_profile(request: web.Request) -> web.Response:
    def respond_error(status: int, message: str) -> web.Response:
        response = _json_error(status, message)
        response.headers["Cache-Control"] = "no-cache"
        return response

    digest = request.query.get("digest")
    revision = request.query.get("revision")
    if digest is None or not is_digest(digest):
        return respond_error(400, "digest must be a canonical blake3 asset digest")
    if revision != "1":
        return respond_error(400, "revision must be '1'")
    library = request.app[LIBRARY_KEY]
    callback = library.model_output_profile
    if callback is None:
        return respond_error(404, "model output profile probing is unavailable")
    path = await asyncio.to_thread(library.locate, digest)
    if path is None:
        return respond_error(404, "asset not held here")

    def probe_verified() -> Mapping[str, object]:
        with open_verified(path, digest) as verified:
            return callback(path, verified, digest, os.fstat(verified.fileno()).st_size)

    try:
        profile = await asyncio.to_thread(probe_verified)
    except AssetIntegrityError:
        return web.json_response(
            {"error": f"asset content on disk no longer matches {digest}"},
            status=409,
            headers={"Cache-Control": "no-cache"},
        )
    except OSError:
        return respond_error(404, "asset not held here")
    except ValueError:
        return web.json_response(
            {"error": "model metadata probe failed: invalid safetensors or profile metadata"},
            status=422,
            headers={"Cache-Control": "no-cache"},
        )
    return web.json_response(profile, headers={"Cache-Control": "no-cache"})


async def handle_asset_sources_get(request: web.Request) -> web.Response:
    """GET /api/assets/{digest}/sources - the provenance record for one
    identity: candidate URLs plus license/note. An identity nobody has
    recorded leads for is an EMPTY record, not a 404 - absence of leads
    is a valid, useful answer for acquisition planning."""
    digest = request.match_info["digest"]
    if not is_digest(digest):
        return _json_error(400, "malformed digest (expected 'blake3:<64 hex>')")
    library = request.app[LIBRARY_KEY]
    store = library.provenance
    assert store is not None  # route registered only with a store
    record = store.get(digest)
    empty: dict[str, object] = {
        "digest": digest,
        "sources": [],
        "license": "",
        "note": "",
        "metadata": {},
    }
    wire: dict[str, object] = record.to_wire() if record is not None else empty
    # Pack-declared leads ([[pack.assets]] urls) merge at READ time, never
    # into the persistent store: the manifest stays the single source of
    # truth, so removing the pack removes its leads instead of leaving
    # stale URLs fossilized in provenance.json.
    if library.pack_assets is not None:
        declared = library.pack_assets.need_for(digest)
        if declared is not None:
            known = cast("list[str]", wire["sources"])
            wire["sources"] = known + [
                source.url
                for source in declared.sources
                if isinstance(source, RemoteSource) and source.url not in known
            ]
    return web.json_response(wire)


async def handle_asset_sources_add(request: web.Request) -> web.Response:
    """POST /api/assets/{digest}/sources - register acquisition leads:
    {"urls": [...], "license"?, "note"?}. Additive merge (provenance
    accretes); URLs must be http(s). Safe to accept from any client the
    server trusts to submit jobs: sources are leads, never authorities -
    a hostile URL can waste bandwidth, never plant wrong bytes."""
    digest = request.match_info["digest"]
    if not is_digest(digest):
        return _json_error(400, "malformed digest (expected 'blake3:<64 hex>')")
    library = request.app[LIBRARY_KEY]
    store = library.provenance
    assert store is not None  # route registered only with a store
    body = await _json_body(request)
    urls_raw = body.get("urls")
    if not isinstance(urls_raw, list) or not all(
        isinstance(url, str) for url in cast(list[Any], urls_raw)
    ):
        return _json_error(400, "'urls' must be a list of strings")
    urls = cast(list[str], urls_raw)
    if not all(url.startswith(("http://", "https://")) for url in urls):
        return _json_error(400, "source URLs must be http(s)")
    license_value = body.get("license", "")
    note = body.get("note", "")
    if not isinstance(license_value, str) or not isinstance(note, str):
        return _json_error(400, "'license' and 'note' must be strings")
    merged = await asyncio.to_thread(
        store.add,
        ProvenanceRecord(digest=digest, sources=tuple(urls), license=license_value, note=note),
    )
    return web.json_response(merged.to_wire())


def _require_scope(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "scope is required (single-user: 'local')"}),
            content_type="application/json",
        )
    return value.strip()


async def handle_library_create(request: web.Request) -> web.Response:
    library = request.app[LIBRARY_KEY]
    body = await _json_body(request)
    scope = _require_scope(body.get("scope"))
    digest = body.get("digest")
    if not isinstance(digest, str) or not is_digest(digest):
        return _json_error(400, "digest must be 'blake3:<64 hex>'")
    if not await asyncio.to_thread(library.vault.has, digest):
        # Membership is scoped but presence is not fabricated: a record
        # must point at bytes this instance actually holds - upload first.
        return _json_error(409, "digest not held here: upload the bytes first")
    name = body.get("name", "")
    media_type = body.get("mediaType", "")
    folder = body.get("folder", "")
    if not all(isinstance(value, str) for value in (name, media_type, folder)):
        return _json_error(400, "name, mediaType and folder must be strings")
    labels = _string_labels(body.get("labels", []))
    if labels is None:
        return _json_error(400, "labels must be a list of strings")
    if (
        _base_media_type(cast(str, media_type)) == IMAGE_DOCUMENT_MEDIA_TYPE
        and await asyncio.to_thread(library.store.get_dependencies, digest) is None
    ):
        return _json_error(409, "image document must be adopted before it enters the library")
    try:
        record = await asyncio.to_thread(
            library.store.create,
            scope,
            cast(str, name),
            digest,
            cast(str, media_type),
            labels=labels,
            folder=cast(str, folder),
        )
    except AssetError as exc:
        return _json_error(400, str(exc))
    return web.json_response(record.to_wire(), status=201)


async def handle_library_list(request: web.Request) -> web.Response:
    library = request.app[LIBRARY_KEY]
    scope = _require_scope(request.query.get("scope"))
    text = request.query.get("q", "")
    label = request.query.get("label", "")
    try:
        limit = min(200, max(1, int(request.query.get("limit", "50"))))
    except ValueError:
        return _json_error(400, "limit must be an integer")
    bound = {"s": scope, "q": text, "l": label}
    after = None
    cursor = request.query.get("cursor")
    if cursor:
        after = decode_cursor(cursor, bound)
    records = await asyncio.to_thread(
        library.store.query,
        scope,
        text=text,
        label=label,
        limit=limit + 1,
        after=after,
    )
    wire: dict[str, object] = {"records": [record.to_wire() for record in records[:limit]]}
    if len(records) > limit:
        last = records[limit - 1]
        wire["cursor"] = encode_cursor(bound, (last.modified, last.id))
    return web.json_response(wire)


async def handle_library_get(request: web.Request) -> web.Response:
    library = request.app[LIBRARY_KEY]
    scope = _require_scope(request.query.get("scope"))
    record = await asyncio.to_thread(library.store.get, scope, request.match_info["record_id"])
    if record is None:
        return _json_error(404, "no such record")
    return web.json_response(record.to_wire())


async def handle_library_update(request: web.Request) -> web.Response:
    library = request.app[LIBRARY_KEY]
    body = await _json_body(request)
    scope = _require_scope(body.get("scope"))
    revision = body.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool):
        return _json_error(400, "revision is required (optimistic concurrency)")
    digest = body.get("digest")
    if digest is not None:
        if not isinstance(digest, str) or not is_digest(digest):
            return _json_error(400, "digest must be 'blake3:<64 hex>'")
        if not await asyncio.to_thread(library.vault.has, digest):
            return _json_error(409, "digest not held here: upload the bytes first")
    name = body.get("name")
    media_type = body.get("mediaType")
    folder = body.get("folder")
    for value in (name, media_type, folder):
        if value is not None and not isinstance(value, str):
            return _json_error(400, "name, mediaType and folder must be strings")
    labels: list[str] | None = None
    if body.get("labels") is not None:
        labels = _string_labels(body.get("labels"))
        if labels is None:
            return _json_error(400, "labels must be a list of strings")
    current = await asyncio.to_thread(library.store.get, scope, request.match_info["record_id"])
    if current is not None:
        target_digest = digest if digest is not None else current.digest
        target_media_type = media_type if media_type is not None else current.media_type
        if (
            _base_media_type(target_media_type) == IMAGE_DOCUMENT_MEDIA_TYPE
            and await asyncio.to_thread(library.store.get_dependencies, target_digest) is None
        ):
            return _json_error(409, "image document must be adopted before it enters the library")
    try:
        record = await asyncio.to_thread(
            lambda: library.store.update(
                scope,
                request.match_info["record_id"],
                revision,
                name=cast("str | None", name),
                digest=digest,
                media_type=cast("str | None", media_type),
                labels=labels,
                folder=cast("str | None", folder),
            )
        )
    except StaleRevision as exc:
        return _json_error(409, str(exc))
    except AssetError as exc:
        return _json_error(400, str(exc))
    if record is None:
        return _json_error(404, "no such record")
    return web.json_response(record.to_wire())


async def handle_library_delete(request: web.Request) -> web.Response:
    library = request.app[LIBRARY_KEY]
    scope = _require_scope(request.query.get("scope"))
    deleted = await asyncio.to_thread(library.store.delete, scope, request.match_info["record_id"])
    if not deleted:
        return _json_error(404, "no such record")
    return web.Response(status=204)


def add_library_routes(app: web.Application, library: ServerLibrary) -> None:
    from .image_document import handle_dependencies, handle_image_document, handle_render

    app[LIBRARY_KEY] = library
    app.router.add_post("/api/assets", handle_upload)
    app.router.add_post("/api/assets/media", handle_media_upload)
    app.router.add_post("/api/assets/latent", handle_latent_upload)
    app.router.add_post("/api/assets/image-document", handle_image_document)
    app.router.add_get("/api/assets/{digest}", handle_asset_get)
    app.router.add_get("/api/assets/{digest}/dependencies", handle_dependencies)
    app.router.add_post("/api/assets/{digest}/render", handle_render)
    app.router.add_get("/api/assets/{digest}/metadata", handle_asset_metadata)
    app.router.add_get("/api/output-profiles/model", handle_model_output_profile)
    if library.provenance is not None:
        # Provenance routes exist only when a store exists (the standard
        # not-registered = 404 posture; no half-open surface).
        app.router.add_get("/api/assets/{digest}/sources", handle_asset_sources_get)
        app.router.add_post("/api/assets/{digest}/sources", handle_asset_sources_add)
    app.router.add_post("/api/library", handle_library_create)
    app.router.add_get("/api/library", handle_library_list)
    app.router.add_get("/api/library/{record_id}", handle_library_get)
    app.router.add_patch("/api/library/{record_id}", handle_library_update)
    app.router.add_delete("/api/library/{record_id}", handle_library_delete)

    async def remove_stale_latent_uploads(app: web.Application) -> None:
        def remove() -> None:
            for path in library.vault.root.glob(".latent-upload-*"):
                if path.is_file() and not path.is_symlink():
                    path.unlink(missing_ok=True)

        await asyncio.to_thread(remove)

    async def close_store(app: web.Application) -> None:
        await library.latent_ingest_slots.acquire()
        await library.latent_ingest_slots.acquire()
        try:
            async with library.media_ingest_lock:
                await asyncio.to_thread(library.store.close)
        finally:
            library.latent_ingest_slots.release()
            library.latent_ingest_slots.release()

    app.on_startup.append(remove_stale_latent_uploads)
    app.on_cleanup.append(close_store)
