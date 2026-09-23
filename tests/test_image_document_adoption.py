"""Trusted ImageDocument validation, CAS adoption, and dependency manifests."""

from __future__ import annotations

import asyncio
import json
import struct
import threading
import zlib
from contextlib import closing
from io import BytesIO
from pathlib import Path

import dinkster_collab as collab
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import (
    AssetError,
    AssetVault,
    LibraryStore,
    digest_bytes,
    raster_image_facts,
)
from dinkster_server import image_document as image_document_module
from dinkster_server.auth import LOCAL_PRINCIPAL, PRINCIPAL_KEY, Principal
from dinkster_server.library import ServerLibrary, add_library_routes
from PIL import Image

from dinkster.serve import _collaboration_snapshot_validators

MEDIA_TYPE = "application/vnd.dinkster.image-document+json"


def _png(mode: str = "RGBA") -> bytes:
    output = BytesIO()
    color = (20, 40, 60, 128) if mode == "RGBA" else (20, 40, 60)
    Image.new(mode, (2, 1), color).save(output, "PNG")
    return output.getvalue()


def _corrupt_png_pixels(data: bytes) -> bytes:
    corrupted = bytearray(data)
    offset = 8
    while offset < len(corrupted):
        length = struct.unpack_from(">I", corrupted, offset)[0]
        kind_at = offset + 4
        payload_at = offset + 8
        if corrupted[kind_at : kind_at + 4] == b"IDAT":
            corrupted[payload_at + length - 1] ^= 1
            crc = zlib.crc32(corrupted[kind_at : payload_at + length])
            struct.pack_into(">I", corrupted, payload_at + length, crc)
            return bytes(corrupted)
        offset += 12 + length
    raise AssertionError("PNG fixture has no IDAT")


def _document(resource: bytes, digest: str) -> dict[str, object]:
    return {
        "format": "dinkster-image",
        "formatVersion": 1,
        "lineage": "lineage-1",
        "canvas": {
            "width": 2,
            "height": 1,
            "colorSpace": "srgb",
            "channelDepth": 8,
            "compositing": "premultiplied-alpha",
        },
        "allocation": {"nextOrdinal": 2},
        "rootLayerIds": ["l1"],
        "layers": {
            "l1": {
                "id": "l1",
                "kind": "raster",
                "name": "Pixels",
                "visible": True,
                "opacity": 65535,
                "transform": {"a": 1000000, "b": 0, "c": 0, "d": 1000000, "tx": 0, "ty": 0},
                "blendMode": "normal",
                "clipping": "none",
                "maskIds": [],
                "resourceId": "r0",
                "sourceRect": {"x": 0, "y": 0, "width": 2, "height": 1},
            }
        },
        "masks": {},
        "resources": {
            "r0": {
                "id": "r0",
                "kind": "raster",
                "digest": digest,
                "byteSize": len(resource),
                "mediaType": "image/png",
                "width": 2,
                "height": 1,
                "colorSpace": "srgb",
                "channelDepth": 8,
                "alphaMode": "straight",
            }
        },
    }


def _blank_document() -> dict[str, object]:
    return {
        "format": "dinkster-image",
        "formatVersion": 1,
        "lineage": "blank",
        "canvas": {
            "width": 1,
            "height": 1,
            "colorSpace": "srgb",
            "channelDepth": 8,
            "compositing": "premultiplied-alpha",
        },
        "allocation": {"nextOrdinal": 0},
        "rootLayerIds": [],
        "layers": {},
        "masks": {},
        "resources": {},
    }


def test_collaboration_snapshot_validation_is_image_specific() -> None:
    validators = _collaboration_snapshot_validators(collab)
    document = _blank_document()
    assert validators("image", "blank", document) is None
    assert (
        validators("image", "different", document) == "ImageDocument lineage must match documentId"
    )
    assert validators("image", "blank", {}) is not None
    assert validators("workflow", "workflow", {}) is None


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


async def _client(
    tmp_path: Path,
    principal: Principal = LOCAL_PRINCIPAL,
    upload_limit: int = 16 * 1024 * 1024,
) -> tuple[TestClient, ServerLibrary]:
    @web.middleware
    async def local_principal(request: web.Request, handler):  # type: ignore[no-untyped-def]
        request[PRINCIPAL_KEY] = principal
        return await handler(request)

    library = ServerLibrary(
        vault=AssetVault(tmp_path / "vault"),
        store=LibraryStore(tmp_path / "library.sqlite"),
        upload_limit=upload_limit,
    )
    app = web.Application(middlewares=[local_principal])
    add_library_routes(app, library)

    async def close_store(_app: web.Application) -> None:
        await asyncio.to_thread(library.store.close)

    app.on_cleanup.append(close_store)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client, library


async def _hold_resource(client: TestClient, resource: bytes, scope: str = "local") -> str:
    response = await client.post(
        f"/api/assets/media?scope={scope}&kind=media/image&name=source.png",
        data=resource,
        headers={"Content-Type": "image/png"},
    )
    assert response.status == 201, await response.text()
    return (await response.json())["asset"]["digest"]


def test_adoption_round_trip_is_idempotent_and_persistent(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, library = await _client(tmp_path)
        resource = _png()
        try:
            child_digest = await _hold_resource(client, resource)
            body = _canonical(_document(resource, child_digest))
            parent_digest = digest_bytes(body)
            response = await client.post(
                "/api/assets/image-document?scope=local",
                data=body,
                headers={"Content-Type": MEDIA_TYPE, "X-Dinkster-Digest": parent_digest},
            )
            assert response.status == 201, await response.text()
            adopted = await response.json()
            assert adopted["digest"] == parent_digest
            assert adopted["mediaType"] == MEDIA_TYPE
            assert adopted["byteSize"] == len(body)
            assert adopted["dependencies"] == [
                {
                    "resourceId": "r0",
                    "digest": child_digest,
                    "byteSize": len(resource),
                    "mediaType": "image/png",
                    "width": 2,
                    "height": 1,
                    "colorSpace": "srgb",
                    "channelDepth": 8,
                    "alphaMode": "straight",
                }
            ]
            repeat = await client.post(
                "/api/assets/image-document?scope=local",
                data=body,
                headers={"Content-Type": MEDIA_TYPE},
            )
            assert repeat.status == 200
            manifest = await client.get(f"/api/assets/{parent_digest}/dependencies")
            assert manifest.status == 200
            assert await manifest.json() == {
                "digest": parent_digest,
                "dependencies": adopted["dependencies"],
            }
            assert manifest.headers["ETag"] == f'"{parent_digest}"'
            cached = await client.get(
                f"/api/assets/{parent_digest}/dependencies",
                headers={"If-None-Match": f'"{parent_digest}"'},
            )
            assert cached.status == 304
            assert library.vault.resolve(parent_digest).read_bytes() == body  # type: ignore[union-attr]
        finally:
            await client.close()
        with closing(LibraryStore(tmp_path / "library.sqlite")) as reopened:
            assert reopened.get_dependencies(parent_digest) == adopted["dependencies"]

    asyncio.run(scenario())


@pytest.mark.parametrize("alpha_mode", ["premultiplied", "opaque"])
def test_adoption_preserves_declared_resource_alpha_interpretation(
    tmp_path: Path, alpha_mode: str
) -> None:
    async def scenario() -> None:
        client, _library = await _client(tmp_path)
        resource = _png()
        try:
            child_digest = await _hold_resource(client, resource)
            document = _document(resource, child_digest)
            document["resources"]["r0"]["alphaMode"] = alpha_mode  # type: ignore[index]
            response = await client.post(
                "/api/assets/image-document?scope=local",
                data=_canonical(document),
                headers={"Content-Type": MEDIA_TYPE},
            )
            assert response.status == 201, await response.text()
            assert (await response.json())["dependencies"][0]["alphaMode"] == alpha_mode
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("mutate", "status"),
    [
        (lambda doc: doc.update(formatVersion=3), 400),
        (lambda doc: doc.update(formatVersion=True), 400),
        (lambda doc: doc.update(extra=True), 400),
        (lambda doc: doc["layers"]["l1"].update(kind="vector"), 400),  # type: ignore[index]
        (lambda doc: doc.update(rootLayerIds=["constructor"]), 400),
        (lambda doc: doc["resources"]["r0"].update(width=3), 409),  # type: ignore[index]
        (lambda doc: doc["resources"]["r0"].update(byteSize=1), 409),  # type: ignore[index]
        (
            lambda doc: doc["layers"]["l1"].update(  # type: ignore[index]
                sourceRect={"x": 1, "y": 0, "width": 2, "height": 1}
            ),
            400,
        ),
    ],
)
def test_invalid_documents_never_publish(
    tmp_path: Path,
    mutate,  # type: ignore[no-untyped-def]
    status: int,
) -> None:
    async def scenario() -> None:
        client, library = await _client(tmp_path)
        try:
            resource = _png()
            child_digest = await _hold_resource(client, resource)
            document = _document(resource, child_digest)
            mutate(document)
            body = _canonical(document)
            parent_digest = digest_bytes(body)
            response = await client.post(
                "/api/assets/image-document?scope=local",
                data=body,
                headers={"Content-Type": MEDIA_TYPE},
            )
            assert response.status == status, await response.text()
            assert library.vault.resolve(parent_digest) is None
            assert library.store.get_dependencies(parent_digest) is None
        finally:
            await client.close()

    asyncio.run(scenario())


def test_hostile_json_and_request_contract_are_rejected(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, library = await _client(tmp_path)
        cases = [
            (
                "?scope=local",
                b'{"format":"dinkster-image","format":"dinkster-image"}',
                MEDIA_TYPE,
                400,
            ),
            ("?scope=local", b'{"format":"dinkster-image","formatVersion":NaN}', MEDIA_TYPE, 400),
            ("?scope=local", b'{"format":"dinkster-image"}', "application/json", 415),
            ("", b'{"format":"dinkster-image"}', MEDIA_TYPE, 400),
        ]
        try:
            for query, body, media_type, expected in cases:
                response = await client.post(
                    "/api/assets/image-document" + query,
                    data=body,
                    headers={"Content-Type": media_type},
                )
                assert response.status == expected
                assert set(await response.json()) == {"error"}
            noncanonical = json.dumps(_document(_png(), "blake3:" + "a" * 64)).encode()
            response = await client.post(
                "/api/assets/image-document?scope=local",
                data=noncanonical,
                headers={"Content-Type": MEDIA_TYPE},
            )
            assert response.status == 400
            assert library.vault.digests() == []
        finally:
            await client.close()

    asyncio.run(scenario())


def test_missing_or_tampered_resources_are_not_adopted(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, library = await _client(tmp_path)
        resource = _png()
        missing_digest = digest_bytes(resource)
        body = _canonical(_document(resource, missing_digest))
        try:
            missing = await client.post(
                "/api/assets/image-document?scope=local",
                data=body,
                headers={"Content-Type": MEDIA_TYPE},
            )
            assert missing.status == 409
            assert (await missing.json())["error"][
                "code"
            ] == "asset.image_document.dependency_missing"
            await _hold_resource(client, resource)
            library.vault.resolve(missing_digest).write_bytes(b"tampered")  # type: ignore[union-attr]
            tampered = await client.post(
                "/api/assets/image-document?scope=local",
                data=body,
                headers={"Content-Type": MEDIA_TYPE},
            )
            assert tampered.status == 409
            assert (await tampered.json())["error"][
                "code"
            ] == "asset.image_document.resource_integrity"
            assert library.vault.resolve(digest_bytes(body)) is None
        finally:
            await client.close()

    asyncio.run(scenario())


def test_corrupt_raster_pixel_payload_is_not_adopted(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, library = await _client(tmp_path)
        resource = _corrupt_png_pixels(_png())
        try:
            child_digest = await _hold_resource(client, resource)
            body = _canonical(_document(resource, child_digest))
            response = await client.post(
                "/api/assets/image-document?scope=local",
                data=body,
                headers={"Content-Type": MEDIA_TYPE},
            )
            assert response.status == 409
            assert (await response.json())["error"][
                "code"
            ] == "asset.image_document.dependency_mismatch"
            assert library.vault.resolve(digest_bytes(body)) is None
        finally:
            await client.close()

    asyncio.run(scenario())


def test_dependency_manifests_are_immutable_and_survive_reopen(tmp_path: Path) -> None:
    path = tmp_path / "library.sqlite"
    digest = "blake3:" + "a" * 64
    first = [{"resourceId": "r0", "digest": "blake3:" + "b" * 64}]
    second = [{"resourceId": "r0", "digest": "blake3:" + "c" * 64}]
    with closing(LibraryStore(path)) as store:
        assert store.put_dependencies(digest, first) is True
        assert store.put_dependencies(digest, first) is False
        with pytest.raises(AssetError, match="conflicts"):
            store.put_dependencies(digest, second)
    with closing(LibraryStore(path)) as reopened:
        assert reopened.get_dependencies(digest) == first


def test_dependency_manifest_exact_concurrent_writes_have_one_creator(tmp_path: Path) -> None:
    path = tmp_path / "library.sqlite"
    digest = "blake3:" + "a" * 64
    dependencies = [{"resourceId": "r0", "digest": "blake3:" + "b" * 64}]
    stores = [LibraryStore(path) for _ in range(8)]
    barrier = threading.Barrier(len(stores))
    results: list[bool] = []
    errors: list[Exception] = []

    def write(store: LibraryStore) -> None:
        try:
            barrier.wait()
            results.append(store.put_dependencies(digest, dependencies))
        except Exception as error:  # noqa: BLE001 - evidence captures competing result
            errors.append(error)
        finally:
            store.close()

    threads = [threading.Thread(target=write, args=(store,)) for store in stores]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert results.count(True) == 1
    assert results.count(False) == len(stores) - 1
    with closing(LibraryStore(path)) as reopened:
        assert reopened.get_dependencies(digest) == dependencies


def test_canonical_json_matches_frontend_number_and_string_encoding() -> None:
    values = [
        1.0,
        1e-7,
        1e-6,
        1e20,
        1e21,
        1.2345678901234567e20,
        float("1000000000000000128"),
        -0.0,
        0.00000123,
        0.000000123,
        3.141592653589793,
    ]
    assert image_document_module._canonical_json(values) == (  # pyright: ignore[reportPrivateUsage]
        "[1,1e-7,0.000001,100000000000000000000,1e+21,123456789012345670000,"
        "1000000000000000100,0,0.00000123,1.23e-7,3.141592653589793]"
    )
    assert (
        image_document_module._canonical_json(  # pyright: ignore[reportPrivateUsage]
            {"\U00010000": "\ud800", "\ue000": "ok"}
        )
        == '{"\U00010000":"\\ud800","\ue000":"ok"}'
    )


def test_raster_facts_cover_supported_containers_and_reject_unsupported_rasters() -> None:
    palette = Image.new("P", (2, 1))
    palette.putpalette([20, 40, 60] + [0] * 765)
    palette_output = BytesIO()
    palette.save(palette_output, "PNG", transparency=0)
    facts = raster_image_facts(palette_output.getvalue())
    assert (facts.media_type, facts.width, facts.height, facts.channel_depth, facts.alpha_mode) == (
        "image/png",
        2,
        1,
        8,
        "straight",
    )

    for image_format, mode, expected_type, expected_alpha in (
        ("JPEG", "RGB", "image/jpeg", "opaque"),
        ("WEBP", "RGB", "image/webp", "opaque"),
        ("WEBP", "RGBA", "image/webp", "straight"),
    ):
        output = BytesIO()
        Image.new(mode, (2, 1)).save(output, image_format)
        facts = raster_image_facts(output.getvalue())
        assert (
            facts.media_type,
            facts.width,
            facts.height,
            facts.channel_depth,
            facts.alpha_mode,
        ) == (
            expected_type,
            2,
            1,
            8,
            expected_alpha,
        )

    sixteen_bit_output = BytesIO()
    Image.new("I;16", (2, 1)).save(sixteen_bit_output, "PNG")
    with pytest.raises(AssetError, match="not 8-bit"):
        raster_image_facts(sixteen_bit_output.getvalue())

    animation_output = BytesIO()
    frames = [Image.new("RGBA", (2, 1), color) for color in ("red", "blue")]
    frames[0].save(
        animation_output,
        "PNG",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    with pytest.raises(AssetError, match="not a single raster"):
        raster_image_facts(animation_output.getvalue())


def test_actor_scoped_ids_and_raster_masks_are_adopted(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, _library = await _client(tmp_path)
        resource = _png()
        try:
            digest = await _hold_resource(client, resource)
            document = _document(resource, digest)
            layers = document["layers"]  # type: ignore[assignment]
            layer = layers.pop("l1")  # type: ignore[union-attr]
            layer["id"] = "l1-alice"  # type: ignore[index]
            layer["maskIds"] = ["m2"]  # type: ignore[index]
            layers["l1-alice"] = layer  # type: ignore[index]
            document["rootLayerIds"] = ["l1-alice"]
            document["allocation"] = {"nextOrdinal": 3, "actorCursors": {"alice": 2}}
            document["masks"] = {
                "m2": {
                    "id": "m2",
                    "kind": "raster",
                    "ownerLayerId": "l1-alice",
                    "enabled": True,
                    "invert": False,
                    "opacity": 65535,
                    "transform": {
                        "a": 1000000,
                        "b": 0,
                        "c": 0,
                        "d": 1000000,
                        "tx": 0,
                        "ty": 0,
                    },
                    "combineMode": "multiply",
                    "channel": "alpha",
                    "resourceId": "r0",
                    "sourceRect": {"x": 0, "y": 0, "width": 2, "height": 1},
                }
            }
            response = await client.post(
                "/api/assets/image-document?scope=local",
                data=_canonical(document),
                headers={"Content-Type": MEDIA_TYPE},
            )
            assert response.status == 201, await response.text()
            assert len((await response.json())["dependencies"]) == 1
        finally:
            await client.close()

    asyncio.run(scenario())


def test_storage_failure_leaves_retryable_bytes_without_an_adoption_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        client, library = await _client(tmp_path)
        resource = _png()
        try:
            digest = await _hold_resource(client, resource)
            body = _canonical(_document(resource, digest))
            parent_digest = digest_bytes(body)

            def fail_manifest(*_args: object) -> bool:
                raise AssetError("manifest unavailable")

            monkeypatch.setattr(library.store, "put_dependencies", fail_manifest)
            response = await client.post(
                "/api/assets/image-document?scope=local",
                data=body,
                headers={"Content-Type": MEDIA_TYPE},
            )
            assert response.status == 503
            assert library.vault.resolve(parent_digest).read_bytes() == body  # type: ignore[union-attr]
            assert library.store.get_dependencies(parent_digest) is None
        finally:
            await client.close()

    asyncio.run(scenario())


def test_fixed_document_limit_ignores_larger_generic_upload_configuration(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, library = await _client(tmp_path, upload_limit=32 * 1024 * 1024)
        try:
            response = await client.post(
                "/api/assets/image-document?scope=local",
                data=bytes(16 * 1024 * 1024 + 1),
                headers={"Content-Type": MEDIA_TYPE},
            )
            assert response.status == 413
            assert library.vault.digests() == []
        finally:
            await client.close()

    asyncio.run(scenario())


def test_fixed_document_limit_ignores_smaller_generic_upload_configuration(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, library = await _client(tmp_path, upload_limit=1)
        body = _canonical(_blank_document())
        try:
            response = await client.post(
                "/api/assets/image-document?scope=local",
                data=body,
                headers={"Content-Type": MEDIA_TYPE},
            )
            assert response.status == 201
            assert library.vault.resolve(digest_bytes(body)) is not None
        finally:
            await client.close()

    asyncio.run(scenario())


def test_raster_verification_refuses_when_decode_slots_are_busy(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, library = await _client(tmp_path)
        resource = _png()
        resource_digest = await _hold_resource(client, resource)
        body = _canonical(_document(resource, resource_digest))
        await library.image_document_decode_slots.acquire()
        await library.image_document_decode_slots.acquire()
        try:
            response = await client.post(
                "/api/assets/image-document?scope=local",
                data=body,
                headers={"Content-Type": MEDIA_TYPE},
            )
            assert response.status == 429
            assert (await response.json())["error"][
                "code"
            ] == "asset.image_document.verification_busy"
            assert library.image_document_decode_reserved_bytes == [0]
            assert library.vault.resolve(digest_bytes(body)) is None
        finally:
            library.image_document_decode_slots.release()
            library.image_document_decode_slots.release()
            await client.close()

    asyncio.run(scenario())


def test_document_without_rasters_bypasses_decode_slots(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, library = await _client(tmp_path)
        await library.image_document_decode_slots.acquire()
        await library.image_document_decode_slots.acquire()
        body = _canonical(_blank_document())
        try:
            response = await client.post(
                "/api/assets/image-document?scope=local",
                data=body,
                headers={"Content-Type": MEDIA_TYPE},
            )
            assert response.status == 201
            assert library.image_document_decode_reserved_bytes == [0]
            assert library.vault.resolve(digest_bytes(body)) is not None
        finally:
            library.image_document_decode_slots.release()
            library.image_document_decode_slots.release()
            await client.close()

    asyncio.run(scenario())


def test_decode_reservation_accounts_for_container_memory() -> None:
    common: dict[str, object] = {"width": 10_000, "height": 10_000, "byteSize": 250_000_000}
    assert (
        image_document_module._decode_reservation(  # pyright: ignore[reportPrivateUsage]
            {**common, "mediaType": "image/png"}
        )
        == 400_000_000
    )
    assert (
        image_document_module._decode_reservation(  # pyright: ignore[reportPrivateUsage]
            {**common, "mediaType": "image/jpeg"}
        )
        == 1_000_000_000
    )
    assert (
        image_document_module._decode_reservation(  # pyright: ignore[reportPrivateUsage]
            {**common, "mediaType": "image/webp"}
        )
        == 1_850_000_000
    )


def test_raster_verification_reserves_decoded_memory_across_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        client, library = await _client(tmp_path)
        resource = _png()
        entered = threading.Event()
        release = threading.Event()

        def blocked_verification(*_args: object) -> None:
            entered.set()
            assert release.wait(timeout=5)

        monkeypatch.setattr(image_document_module, "_verify_dependencies", blocked_verification)
        digest = await _hold_resource(client, resource)
        document = _document(resource, digest)
        resources = document["resources"]
        assert isinstance(resources, dict)
        descriptor = resources["r0"]
        assert isinstance(descriptor, dict)
        descriptor["width"] = 10_000
        descriptor["height"] = 10_000
        descriptor["mediaType"] = "image/webp"
        descriptor["byteSize"] = 250_000_000
        body = _canonical(document)
        first = asyncio.create_task(
            client.post(
                "/api/assets/image-document?scope=local",
                data=body,
                headers={"Content-Type": MEDIA_TYPE},
            )
        )
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            assert library.image_document_decode_reserved_bytes == [1_850_000_000]
            second = await client.post(
                "/api/assets/image-document?scope=local",
                data=body,
                headers={"Content-Type": MEDIA_TYPE},
            )
            assert second.status == 429
            assert library.image_document_decode_reserved_bytes == [1_850_000_000]
            release.set()
            assert (await first).status == 201
            assert library.image_document_decode_reserved_bytes == [0]
        finally:
            release.set()
            if not first.done():
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
            await client.close()

    asyncio.run(scenario())


def test_cancelled_raster_verification_keeps_capacity_until_worker_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        _client_instance, library = await _client(tmp_path)
        entered = threading.Event()
        release = threading.Event()

        def blocked_verification(*_args: object) -> None:
            entered.set()
            assert release.wait(timeout=5)

        monkeypatch.setattr(image_document_module, "_verify_dependencies", blocked_verification)
        dependency: dict[str, object] = {
            "width": 10_000,
            "height": 10_000,
            "byteSize": 1,
            "mediaType": "image/png",
        }
        task = asyncio.create_task(
            image_document_module._verify_dependencies_bounded(library, "local", [dependency])
        )
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert library.image_document_decode_reserved_bytes == [400_000_000]
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert library.image_document_decode_reserved_bytes == [0]
            await asyncio.wait_for(library.image_document_decode_slots.acquire(), timeout=1)
            await asyncio.wait_for(library.image_document_decode_slots.acquire(), timeout=1)
            library.image_document_decode_slots.release()
            library.image_document_decode_slots.release()
        finally:
            release.set()
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            await _client_instance.close()

    asyncio.run(scenario())


def test_concurrent_adoption_instances_commit_one_manifest_without_deleting_bytes(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        first, first_library = await _client(tmp_path)
        second, second_library = await _client(tmp_path)
        resource = _png()
        try:
            digest = await _hold_resource(first, resource)
            body = _canonical(_document(resource, digest))
            parent_digest = digest_bytes(body)
            responses = await asyncio.gather(
                first.post(
                    "/api/assets/image-document?scope=local",
                    data=body,
                    headers={"Content-Type": MEDIA_TYPE},
                ),
                second.post(
                    "/api/assets/image-document?scope=local",
                    data=body,
                    headers={"Content-Type": MEDIA_TYPE},
                ),
            )
            outcomes = [(response.status, await response.text()) for response in responses]
            assert all(status in {200, 201} for status, _body in outcomes), outcomes
            assert first_library.vault.resolve(parent_digest).read_bytes() == body  # type: ignore[union-attr]
            assert first_library.store.get_dependencies(parent_digest) is not None
            assert second_library.store.get_dependencies(parent_digest) is not None
        finally:
            await first.close()
            await second.close()

    asyncio.run(scenario())


def test_image_library_records_require_typed_adoption(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, _library = await _client(tmp_path)
        resource = _png()
        try:
            child_digest = await _hold_resource(client, resource)
            body = _canonical(_document(resource, child_digest))
            parent_digest = digest_bytes(body)
            generic = await client.post("/api/assets", data=body)
            assert generic.status == 201
            record_body = {
                "scope": "local",
                "name": "Image",
                "digest": parent_digest,
                "mediaType": MEDIA_TYPE,
            }
            refused = await client.post("/api/library", json=record_body)
            assert refused.status == 409
            disguised = await client.post(
                "/api/library", json={**record_body, "mediaType": f" {MEDIA_TYPE.upper()} "}
            )
            assert disguised.status == 409
            parameterized = await client.post(
                "/api/library", json={**record_body, "mediaType": f"{MEDIA_TYPE};charset=utf-8"}
            )
            assert parameterized.status == 409
            generic_record = await client.post(
                "/api/library", json={**record_body, "mediaType": "application/json"}
            )
            assert generic_record.status == 201
            generic_wire = await generic_record.json()
            refused_update = await client.patch(
                f"/api/library/{generic_wire['id']}",
                json={
                    "scope": "local",
                    "revision": generic_wire["revision"],
                    "mediaType": MEDIA_TYPE,
                },
            )
            assert refused_update.status == 409

            adopted = await client.post(
                "/api/assets/image-document?scope=local",
                data=body,
                headers={"Content-Type": MEDIA_TYPE},
            )
            assert adopted.status == 201
            created = await client.post("/api/library", json=record_body)
            assert created.status == 201
            accepted_update = await client.patch(
                f"/api/library/{generic_wire['id']}",
                json={
                    "scope": "local",
                    "revision": generic_wire["revision"],
                    "mediaType": MEDIA_TYPE,
                },
            )
            assert accepted_update.status == 200
        finally:
            await client.close()

    asyncio.run(scenario())


def test_adopted_blank_document_can_enter_image_library(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, _library = await _client(tmp_path)
        body = _canonical(_blank_document())
        try:
            adopted = await client.post(
                "/api/assets/image-document?scope=local",
                data=body,
                headers={"Content-Type": MEDIA_TYPE},
            )
            assert adopted.status == 201
            created = await client.post(
                "/api/library",
                json={
                    "scope": "local",
                    "name": "Blank",
                    "digest": digest_bytes(body),
                    "mediaType": MEDIA_TYPE,
                },
            )
            assert created.status == 201
        finally:
            await client.close()

    asyncio.run(scenario())


def test_adoption_requires_write_scope_and_scoped_resource_authority(tmp_path: Path) -> None:
    async def scenario() -> None:
        resource = _png()
        denied_client, _denied_library = await _client(
            tmp_path / "denied", Principal("reader", {"a": frozenset({"assets:read"})})
        )
        try:
            denied = await denied_client.post(
                "/api/assets/image-document?scope=a",
                data=b"{}",
                headers={"Content-Type": MEDIA_TYPE},
            )
            assert denied.status == 403
        finally:
            await denied_client.close()

        client, library = await _client(tmp_path / "scoped")
        try:
            digest = await _hold_resource(client, resource, "a")
            body = _canonical(_document(resource, digest))
            cross_scope = await client.post(
                "/api/assets/image-document?scope=b",
                data=body,
                headers={"Content-Type": MEDIA_TYPE},
            )
            assert cross_scope.status == 409
            assert (await cross_scope.json())["error"][
                "code"
            ] == "asset.image_document.dependency_missing"
            assert library.vault.resolve(digest) is not None
            assert library.vault.resolve(digest_bytes(body)) is None
        finally:
            await client.close()

    asyncio.run(scenario())
