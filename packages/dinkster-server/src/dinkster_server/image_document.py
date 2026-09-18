from __future__ import annotations

import asyncio
import json
import mmap
import sqlite3
from typing import Any, cast

from aiohttp import web
from dinkster_assets import (
    AssetError,
    AssetIntegrityError,
    MediaGrant,
    RasterImageFacts,
    digest_bytes,
    is_digest,
    open_verified,
    raster_image_facts,
)
from dinkster_image_document import (
    IMAGE_DOCUMENT_MEDIA_TYPE,
    OUTPUT_ENCODING,
    RENDER_PROFILE,
    RENDERER_CONTRACT,
    InvalidDocument,
    decode_document,
    parse_selector,
    render_document,
    validate_document,
)
from dinkster_image_document.format import MAX_DOCUMENT_BYTES, canonical_json
from PIL import Image

from .auth import LOCAL_PRINCIPAL, principal_for
from .library import (
    DIGEST_HEADER,
    LIBRARY_KEY,
    ServerLibrary,
    run_media_publication,
    run_owned_thread,
)

MEDIA_TYPE = IMAGE_DOCUMENT_MEDIA_TYPE
_MAX_DOCUMENT_BYTES = MAX_DOCUMENT_BYTES
_DECODE_BYTES_PER_PIXEL = {"image/png": 4, "image/jpeg": 10, "image/webp": 16}
_MAX_DECODE_RESERVED_BYTES = 2 * 1024 * 1024 * 1024
_canonical_json = canonical_json

__all__ = [
    "InvalidDocument",
    "handle_dependencies",
    "handle_image_document",
    "handle_render",
    "validate_document",
]


class MissingDependency(InvalidDocument):
    pass


class DependencyMismatch(InvalidDocument):
    pass


class VerificationBusy(Exception):
    pass


def _error(status: int, code: str, message: str) -> web.Response:
    return web.json_response(
        {"error": {"code": f"asset.image_document.{code}", "message": message}}, status=status
    )


def _valid_scope(request: web.Request) -> str | None:
    if (
        set(request.query) != {"scope"}
        or len(request.query.getall("scope", [])) != 1
        or not request.query["scope"]
        or request.query["scope"] != request.query["scope"].strip()
        or any(character.isspace() for character in request.query["scope"])
    ):
        return None
    return request.query["scope"]


def _verify_dependencies(
    library: ServerLibrary, scope: str, dependencies: list[dict[str, object]]
) -> None:
    verified: dict[str, tuple[int, str, int, int, int]] = {}
    for dependency in dependencies:
        digest = cast(str, dependency["digest"])
        grant = library.store.media_grant(scope, digest, "media/image")
        if grant is None:
            raise MissingDependency("a declared resource is not held")
        expected = (
            cast(int, dependency["byteSize"]),
            cast(str, dependency["mediaType"]),
            cast(int, dependency["width"]),
            cast(int, dependency["height"]),
            cast(int, dependency["channelDepth"]),
        )
        if digest in verified:
            if verified[digest] != expected:
                raise DependencyMismatch("resource descriptor does not match held bytes")
            continue
        path = library.locate(digest)
        if path is None:
            raise MissingDependency("a declared resource is not held")
        with open_verified(path, digest) as handle:
            size = handle.seek(0, 2)
            handle.seek(0)
            try:
                with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
                    facts = raster_image_facts(data)
            except AssetError as error:
                raise DependencyMismatch("held resource is not a supported raster image") from error
            actual = (
                size,
                facts.media_type,
                facts.width,
                facts.height,
                facts.channel_depth,
            )
            if actual != expected:
                raise DependencyMismatch("resource descriptor does not match held bytes")
            _verify_raster_decode(handle, facts)
        verified[digest] = actual


def _decode_reservation(dependency: dict[str, object]) -> int:
    media_type = cast(str, dependency["mediaType"])
    pixels = cast(int, dependency["width"]) * cast(int, dependency["height"])
    file_bytes = cast(int, dependency["byteSize"]) if media_type == "image/webp" else 0
    return pixels * _DECODE_BYTES_PER_PIXEL[media_type] + file_bytes


async def _verify_dependencies_bounded(
    library: ServerLibrary, scope: str, dependencies: list[dict[str, object]]
) -> None:
    if not dependencies:
        return
    if library.image_document_decode_slots.locked():
        raise VerificationBusy
    await library.image_document_decode_slots.acquire()
    reserved = 0
    try:
        required = max(_decode_reservation(dependency) for dependency in dependencies)
        if library.image_document_decode_reserved_bytes[0] + required > _MAX_DECODE_RESERVED_BYTES:
            raise VerificationBusy
        library.image_document_decode_reserved_bytes[0] += required
        reserved = required
        task = asyncio.create_task(
            asyncio.to_thread(_verify_dependencies, library, scope, dependencies)
        )
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
        task.result()
    finally:
        library.image_document_decode_reserved_bytes[0] -= reserved
        library.image_document_decode_slots.release()


def _verify_raster_decode(handle: Any, facts: RasterImageFacts) -> None:
    expected_format = {
        "image/png": "PNG",
        "image/jpeg": "JPEG",
        "image/webp": "WEBP",
    }[facts.media_type]
    try:
        handle.seek(0)
        with Image.open(handle) as image:
            if (
                image.format != expected_format
                or image.size != (facts.width, facts.height)
                or getattr(image, "n_frames", 1) != 1
            ):
                raise DependencyMismatch("decoded raster facts do not match its container")
            image.load()
    except (OSError, SyntaxError, ValueError) as error:
        raise DependencyMismatch("held resource pixel payload does not decode") from error


async def handle_image_document(request: web.Request) -> web.Response:
    scope = _valid_scope(request)
    if scope is None:
        return _error(
            400, "invalid_scope", "exactly one non-empty whitespace-free scope is required"
        )
    principal = principal_for(request)
    if principal is not LOCAL_PRINCIPAL and not principal.allows_in(scope, "assets:write"):
        return _error(403, "scope_forbidden", "scope does not grant assets:write")
    if request.headers.get("Content-Type", "").strip().lower() != MEDIA_TYPE:
        return _error(415, "unsupported_content_type", f"Content-Type must be {MEDIA_TYPE}")
    library = request.app[LIBRARY_KEY]
    if request.content_length is not None and request.content_length > _MAX_DOCUMENT_BYTES:
        return _error(413, "too_large", "image document exceeds 16 MiB")
    received = bytearray()
    try:
        async for chunk in request.content.iter_chunked(65536):
            received.extend(chunk)
            if len(received) > _MAX_DOCUMENT_BYTES:
                return _error(413, "too_large", "image document exceeds 16 MiB")
    except (ConnectionError, OSError, web.RequestPayloadError):
        return _error(400, "interrupted", "image document upload was interrupted")
    data = bytes(received)
    digest = await asyncio.to_thread(digest_bytes, data)
    declared = request.headers.get(DIGEST_HEADER)
    if declared is not None and not is_digest(declared):
        return _error(400, "invalid_digest", "X-Dinkster-Digest is not canonical")
    if declared is not None and declared != digest:
        return _error(409, "digest_mismatch", "X-Dinkster-Digest does not match the document bytes")
    try:
        parsed = await asyncio.to_thread(decode_document, data)
        dependencies = list(parsed.dependencies)
        await _verify_dependencies_bounded(library, scope, dependencies)

        def publish() -> tuple[bool, bool]:
            with library.vault.writer(digest) as writer:
                writer.write(data)
                _, created = writer.commit_with_result()
            manifest_created = library.store.put_dependencies(digest, dependencies)
            return created, manifest_created

        created, manifest_created = await run_media_publication(library, publish)
    except MissingDependency:
        return _error(409, "dependency_missing", "an image document resource is not held")
    except DependencyMismatch:
        return _error(
            409,
            "dependency_mismatch",
            "an image document resource descriptor does not match held bytes",
        )
    except VerificationBusy:
        return _error(429, "verification_busy", "raster verification capacity is busy")
    except InvalidDocument:
        return _error(400, "invalid_document", "image document or dependency manifest is invalid")
    except AssetIntegrityError:
        return _error(409, "resource_integrity", "a held resource failed digest verification")
    except (AssetError, OSError, sqlite3.Error):
        return _error(503, "storage_unavailable", "image document storage is unavailable")
    return web.json_response(
        {
            "digest": digest,
            "mediaType": MEDIA_TYPE,
            "byteSize": len(data),
            "dependencies": dependencies,
        },
        status=201 if created or manifest_created else 200,
    )


async def handle_dependencies(request: web.Request) -> web.Response:
    digest = request.match_info["digest"]
    if not is_digest(digest):
        return _error(400, "invalid_digest", "digest is not canonical")
    try:
        dependencies = await asyncio.to_thread(
            request.app[LIBRARY_KEY].store.get_dependencies, digest
        )
    except (AssetError, ValueError, sqlite3.Error):
        return _error(503, "storage_unavailable", "dependency manifest storage is unavailable")
    if dependencies is None:
        return _error(404, "not_adopted", "asset has no adopted dependency manifest")
    etag = f'"{digest}"'
    if any(tag.value in (digest, "*") for tag in (request.if_none_match or ())):
        return web.Response(status=304, headers={"ETag": etag})
    return web.json_response(
        {"digest": digest, "dependencies": dependencies},
        headers={"ETag": etag, "Cache-Control": "private, max-age=31536000, immutable"},
    )


def _render_cache_key(document_digest: str, selector: str) -> tuple[str, dict[str, str]]:
    contract = {
        "documentDigest": document_digest,
        "selector": selector,
        "profile": RENDER_PROFILE,
        "rendererContract": RENDERER_CONTRACT,
        "encoding": OUTPUT_ENCODING,
    }
    encoded = json.dumps(contract, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return digest_bytes(encoded), contract


def _asset_wire(output: dict[str, object]) -> dict[str, object]:
    return {
        "digest": output["digest"],
        "name": "image-document-render.png",
        "size": output["byteSize"],
        "mediaType": "image/png",
        "virtualPath": "",
    }


def _grant_render(library: ServerLibrary, scope: str, output: dict[str, object]) -> None:
    digest = cast(str, output["digest"])
    path = library.locate(digest)
    if path is None:
        raise AssetError("render output is not held")
    library.store.grant_media_result(
        MediaGrant(
            scope=scope,
            digest=digest,
            kind="media/image",
            media_type="image/png",
            extension="png",
            byte_size=cast(int, output["byteSize"]),
        ),
        path,
    )


def _cached_response(
    library: ServerLibrary,
    scope: str,
    cache_key: str,
    manifest: dict[str, object],
) -> web.Response | None:
    output = manifest.get("output")
    if not isinstance(output, dict):
        raise AssetError("stored render provenance is invalid")
    typed_output = cast(dict[str, object], output)
    digest = typed_output.get("digest")
    if not isinstance(digest, str) or library.locate(digest) is None:
        return None
    _grant_render(library, scope, typed_output)
    return web.json_response(
        {
            "cacheKey": cache_key,
            "cached": True,
            "asset": _asset_wire(typed_output),
            "provenance": manifest,
        }
    )


def _render_sync(
    library: ServerLibrary,
    document_digest: str,
    selector_text: str,
    dependencies: list[dict[str, object]],
) -> tuple[bytes, int, int]:
    document_path = library.locate(document_digest)
    if document_path is None:
        raise MissingDependency("image document is not held")
    with open_verified(document_path, document_digest) as handle:
        document_bytes = handle.read(_MAX_DOCUMENT_BYTES + 1)
    parsed = decode_document(document_bytes)
    if list(parsed.dependencies) != dependencies:
        raise DependencyMismatch("adopted dependency manifest does not match document bytes")

    def read_resource(digest: str) -> bytes:
        path = library.locate(digest)
        if path is None:
            raise MissingDependency("an image document resource is not held")
        with open_verified(path, digest) as handle:
            return handle.read()

    rendered = render_document(parsed, read_resource, parse_selector(selector_text))
    return rendered.png, rendered.width, rendered.height


async def handle_render(request: web.Request) -> web.Response:
    scope = _valid_scope(request)
    if scope is None:
        return _error(
            400, "invalid_scope", "exactly one non-empty whitespace-free scope is required"
        )
    principal = principal_for(request)
    if principal is not LOCAL_PRINCIPAL and not principal.allows_in(scope, "assets:write"):
        return _error(403, "scope_forbidden", "scope does not grant assets:write")
    document_digest = request.match_info["digest"]
    if not is_digest(document_digest):
        return _error(400, "invalid_digest", "digest is not canonical")
    try:
        body: object = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _error(400, "invalid_render_request", "render request must be valid JSON")
    if not isinstance(body, dict):
        return _error(400, "invalid_render_request", "render request has unsupported fields")
    body = cast(dict[str, object], body)
    if set(body) - {"selector", "profile"}:
        return _error(400, "invalid_render_request", "render request has unsupported fields")
    selector_text = body.get("selector", "composite")
    profile = body.get("profile", RENDER_PROFILE)
    if not isinstance(selector_text, str) or profile != RENDER_PROFILE:
        return _error(400, "unsupported_profile", f"profile must be {RENDER_PROFILE}")
    try:
        parse_selector(selector_text)
    except InvalidDocument as error:
        return _error(400, "invalid_selector", str(error))
    library = request.app[LIBRARY_KEY]
    cache_key, contract = _render_cache_key(document_digest, selector_text)
    try:
        dependencies = await asyncio.to_thread(library.store.get_dependencies, document_digest)
        if dependencies is None:
            return _error(409, "not_adopted", "image document must be adopted before rendering")
        await _verify_dependencies_bounded(library, scope, dependencies)
        existing = await asyncio.to_thread(library.store.get_derivation, cache_key)
        if existing is not None:
            cached = await asyncio.to_thread(_cached_response, library, scope, cache_key, existing)
            if cached is not None:
                return cached
        if library.image_document_render_slots.locked():
            return _error(429, "render_busy", "reference renderer capacity is busy")
        await library.image_document_render_slots.acquire()
        try:
            png, width, height = await run_owned_thread(
                lambda: _render_sync(library, document_digest, selector_text, dependencies)
            )
        finally:
            library.image_document_render_slots.release()
        output_digest = await asyncio.to_thread(digest_bytes, png)
        output: dict[str, object] = {
            "digest": output_digest,
            "byteSize": len(png),
            "mediaType": "image/png",
            "width": width,
            "height": height,
            "encoding": OUTPUT_ENCODING,
        }
        manifest: dict[str, object] = {
            **contract,
            "source": {
                "digest": document_digest,
                "mediaType": MEDIA_TYPE,
                "dependencies": dependencies,
            },
            "output": output,
        }

        def publish() -> tuple[bool, bool]:
            with library.vault.writer(output_digest) as writer:
                writer.write(png)
                _, created = writer.commit_with_result()
            _grant_render(library, scope, output)
            manifest_created = library.store.put_derivation(cache_key, manifest)
            return created, manifest_created

        await run_media_publication(library, publish)
    except MissingDependency:
        return _error(409, "dependency_missing", "an image document resource is not held")
    except DependencyMismatch:
        return _error(409, "dependency_mismatch", "adopted dependency facts changed")
    except VerificationBusy:
        return _error(429, "verification_busy", "raster verification capacity is busy")
    except InvalidDocument as error:
        return _error(400, "render_invalid", str(error))
    except AssetIntegrityError:
        return _error(409, "resource_integrity", "a held resource failed digest verification")
    except (AssetError, OSError, sqlite3.Error):
        return _error(503, "storage_unavailable", "image document rendering is unavailable")
    return web.json_response(
        {
            "cacheKey": cache_key,
            "cached": False,
            "asset": _asset_wire(output),
            "provenance": manifest,
        },
        status=201,
    )
