"""Workflow persistence: client upload + scoped library.

Two layers, one boundary: immutable bytes in the vault (global dedup by
digest, backend validates bytes/size/digest only - never document
semantics, which are frontend-owned) and mutable SCOPED records naming
them. Every record operation takes an explicit scope from the first
slice; single-user mode is the reserved scope "local", never "scope
absent". Browse is query-first and cursor-paged with the cursor bound to
its query.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import (
    AssetError,
    AssetVault,
    LibraryStore,
    StaleRevision,
    digest_bytes,
)
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EventListener
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_server import ServerLibrary, create_app
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker
from PIL import Image

WORKFLOW_MEDIA_TYPE = "application/x-dinkster-workflow+json"

STRING = TypeExpr.concrete("core.string")


class Echo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.echo",
            inputs=(InputSpec("text", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )

    @classmethod
    async def execute(cls, *, text: str) -> Mapping[str, object]:
        return cls.outputs(out=text)


NODES = (Echo,)
SCHEMAS = build_schemas(NODES)


def make_engine(on_event: EventListener | None = None) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    return Engine(
        schemas=SCHEMAS,
        registry=registry,
        worker=InProcessWorker(build_node_types(NODES), registry),
        cache=MemoryLRUCache(),
        on_event=on_event,
    )


# -- LibraryStore -------------------------------------------------------------


DIGEST_A = digest_bytes(b'{"graphs": {}}')
DIGEST_B = digest_bytes(b'{"graphs": {"main": {}}}')


def make_store(tmp_path: Path) -> LibraryStore:
    return LibraryStore(tmp_path / "library.sqlite")


def test_store_create_get_roundtrip(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = store.create(
        "local",
        "My Workflow",
        DIGEST_A,
        WORKFLOW_MEDIA_TYPE,
        labels=["video", "wip"],
        folder="projects/demo",
    )
    assert record.revision == 1
    assert record.created == record.modified > 0
    loaded = store.get("local", record.id)
    assert loaded == record
    wire = record.to_wire()
    assert wire["mediaType"] == WORKFLOW_MEDIA_TYPE
    assert wire["labels"] == ["video", "wip"]
    assert wire["folder"] == "projects/demo"
    # Optional folder is omitted-when-absent, matching every other wire.
    bare = store.create("local", "bare", DIGEST_A, WORKFLOW_MEDIA_TYPE)
    assert "folder" not in bare.to_wire()
    store.close()


def test_store_scope_isolation(tmp_path: Path) -> None:
    """Bytes deduplicate globally, records never cross scopes: the same
    digest under two scopes is two records, and a wrong-scope lookup is a
    plain miss (never a hint the id exists elsewhere)."""
    store = make_store(tmp_path)
    mine = store.create("local", "mine", DIGEST_A, WORKFLOW_MEDIA_TYPE)
    store.create("other", "theirs", DIGEST_A, WORKFLOW_MEDIA_TYPE)
    assert store.get("other", mine.id) is None
    assert store.delete("other", mine.id) is False
    assert store.update("other", mine.id, 1, name="stolen") is None
    assert [r.name for r in store.query("local")] == ["mine"]
    assert [r.name for r in store.query("other")] == ["theirs"]
    # Scope is structurally required, never defaulted server-side.
    with pytest.raises(AssetError):
        store.create("", "x", DIGEST_A, WORKFLOW_MEDIA_TYPE)
    with pytest.raises(AssetError):
        store.query(" ")
    store.close()


def test_store_update_optimistic_concurrency(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = store.create("local", "v1", DIGEST_A, WORKFLOW_MEDIA_TYPE)
    updated = store.update("local", record.id, 1, name="v2", digest=DIGEST_B)
    assert updated is not None
    assert updated.name == "v2"
    assert updated.digest == DIGEST_B
    assert updated.revision == 2
    assert updated.created == record.created
    assert updated.modified >= record.modified
    # Unspecified fields keep their values.
    assert updated.media_type == WORKFLOW_MEDIA_TYPE
    # The revision the other tab read is now stale: loud conflict, never
    # last-writer-wins.
    with pytest.raises(StaleRevision):
        store.update("local", record.id, 1, name="v2-conflict")
    assert store.get("local", record.id) == updated
    store.close()


def test_store_delete_keeps_other_records(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    a = store.create("local", "a", DIGEST_A, WORKFLOW_MEDIA_TYPE)
    b = store.create("local", "b", DIGEST_B, WORKFLOW_MEDIA_TYPE)
    assert store.delete("local", a.id) is True
    assert store.delete("local", a.id) is False
    assert store.get("local", b.id) == b
    store.close()


def test_store_query_filters_and_paging(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    for index in range(5):
        store.create(
            "local",
            f"Workflow {index}",
            DIGEST_A,
            WORKFLOW_MEDIA_TYPE,
            labels=["even"] if index % 2 == 0 else ["odd"],
        )
    store.create("local", "Upscale pass", DIGEST_B, WORKFLOW_MEDIA_TYPE)

    # Text search is a case-insensitive name substring.
    assert [r.name for r in store.query("local", text="upscale")] == ["Upscale pass"]
    assert len(store.query("local", text="workflow")) == 5
    # Label filter is exact-match.
    assert {r.name for r in store.query("local", label="even")} == {
        "Workflow 0",
        "Workflow 2",
        "Workflow 4",
    }
    # Newest-modified first; keyset paging never repeats or skips.
    page1 = store.query("local", limit=4)
    assert [r.name for r in page1] == [
        "Upscale pass",
        "Workflow 4",
        "Workflow 3",
        "Workflow 2",
    ]
    last = page1[-1]
    page2 = store.query("local", limit=4, after=(last.modified, last.id))
    assert [r.name for r in page2] == ["Workflow 1", "Workflow 0"]
    store.close()


# -- HTTP surface -------------------------------------------------------------


async def make_client(tmp_path: Path, **library_overrides: object) -> TestClient:
    library = ServerLibrary(
        vault=AssetVault(tmp_path / "vault"),
        store=LibraryStore(tmp_path / "library.sqlite"),
        **library_overrides,  # type: ignore[arg-type]
    )
    app = create_app(make_engine, SCHEMAS, library=library)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


def test_upload_and_fetch_roundtrip(tmp_path: Path) -> None:
    """Upload returns the canonical digest (201 new, 200 already-held);
    fetch serves the bytes back under the immutable-cache contract."""

    async def scenario() -> None:
        client = await make_client(tmp_path)
        try:
            body = b'{"graphs": {"main": {"nodes": {}}}}'
            digest = digest_bytes(body)
            resp = await client.post(
                "/api/assets",
                data=body,
                headers={"Content-Type": WORKFLOW_MEDIA_TYPE},
            )
            assert resp.status == 201
            assert await resp.json() == {"digest": digest}
            # Idempotent: same bytes again is a cheap 200, same digest.
            resp = await client.post("/api/assets", data=body)
            assert resp.status == 200
            assert await resp.json() == {"digest": digest}

            resp = await client.get(f"/api/assets/{digest}")
            assert resp.status == 200
            assert await resp.read() == body
            assert resp.headers["ETag"] == f'"{digest}"'
            assert resp.headers["Cache-Control"] == "private, max-age=31536000, immutable"
            # Digest-immutable: If-None-Match short-circuits to 304.
            resp = await client.get(
                f"/api/assets/{digest}", headers={"If-None-Match": f'"{digest}"'}
            )
            assert resp.status == 304

            assert (await client.get(f"/api/assets/{DIGEST_B}")).status == 404
            assert (await client.get("/api/assets/sha256:abc")).status == 400
        finally:
            await client.close()

    asyncio.run(scenario())


def test_chunked_upload_reads_complete_png_body(tmp_path: Path) -> None:
    """A request read may return one buffered chunk before EOF.

    Exercise the real chunked HTTP path with the first chunk matching the
    truncated live upload size. The response identity, vault readback, and
    image decoder must all see the complete request body.
    """
    pixels = random.Random(1322).randbytes(320 * 320 * 3)
    encoded = BytesIO()
    Image.frombytes("RGB", (320, 320), pixels).save(encoded, "PNG", compress_level=0)
    body = encoded.getvalue()
    assert 300_000 < len(body) < 320_000

    async def chunks():
        boundaries = (78_840, 143_000, 225_000, len(body))
        start = 0
        for end in boundaries:
            yield body[start:end]
            start = end
            await asyncio.sleep(0.01)

    async def scenario() -> None:
        client = await make_client(tmp_path)
        try:
            digest = digest_bytes(body)
            resp = await client.post(
                "/api/assets",
                data=chunks(),
                headers={"Content-Type": "image/png"},
            )
            assert resp.status == 201
            assert await resp.json() == {"digest": digest}

            resp = await client.get(f"/api/assets/{digest}")
            assert resp.status == 200
            received = await resp.read()
            assert len(received) == len(body)
            assert received == body
            with Image.open(BytesIO(received)) as decoded:
                decoded.load()
                assert decoded.size == (320, 320)
                assert decoded.mode == "RGB"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_asset_get_falls_through_to_resolver(tmp_path: Path) -> None:
    """A digest the vault does not hold streams from the fallback
    resolver (the filesystem-mounts seam) under the same digest-immutable
    contract - browsed mount files preview without a vault copy."""

    payload = b"mounted bytes " * 4096  # bigger than one would inline
    mounted = tmp_path / "mounted.bin"
    mounted.write_bytes(payload)
    digest = digest_bytes(payload)

    class OneFileResolver:
        def resolve(self, wanted: str) -> Path | None:
            return mounted if wanted == digest else None

    async def scenario() -> None:
        client = await make_client(tmp_path, resolver=OneFileResolver())
        try:
            resp = await client.get(f"/api/assets/{digest}")
            assert resp.status == 200
            assert await resp.read() == payload
            assert resp.headers["ETag"] == f'"{digest}"'
            assert resp.headers["Content-Length"] == str(len(payload))
            resp = await client.get(
                f"/api/assets/{digest}", headers={"If-None-Match": f'"{digest}"'}
            )
            assert resp.status == 304
            # Unknown everywhere is still a 404; a resolver hit whose file
            # vanished since resolve degrades to 404, never a 500.
            assert (await client.get(f"/api/assets/{DIGEST_B}")).status == 404
            mounted.unlink()
            resp = await client.get(f"/api/assets/{digest}")
            assert resp.status == 404
        finally:
            await client.close()

    asyncio.run(scenario())


def test_upload_declared_digest_and_limits(tmp_path: Path) -> None:
    async def oversized_chunks():
        yield b"x" * 32
        await asyncio.sleep(0)
        yield b"x" * 33

    async def scenario() -> None:
        client = await make_client(tmp_path, upload_limit=64)
        try:
            body = b'{"graphs": {}}'
            # A declared digest gives end-to-end integrity: mismatch is a
            # 400 and nothing lands.
            resp = await client.post(
                "/api/assets", data=body, headers={"X-Dinkster-Digest": DIGEST_B}
            )
            assert resp.status == 400
            assert (await client.get(f"/api/assets/{digest_bytes(body)}")).status == 404
            resp = await client.post(
                "/api/assets", data=body, headers={"X-Dinkster-Digest": digest_bytes(body)}
            )
            assert resp.status == 201

            assert (await client.post("/api/assets", data=b"x" * 65)).status == 413
            assert (await client.post("/api/assets", data=oversized_chunks())).status == 413
            assert (await client.post("/api/assets", data=b"")).status == 400
        finally:
            await client.close()

    asyncio.run(scenario())


def test_library_crud_over_http(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = await make_client(tmp_path)
        try:
            body = b'{"graphs": {}}'
            digest = (await (await client.post("/api/assets", data=body)).json())["digest"]

            # A record must point at bytes this instance holds: upload first.
            resp = await client.post(
                "/api/library",
                json={
                    "scope": "local",
                    "name": "dangling",
                    "digest": DIGEST_B,
                    "mediaType": WORKFLOW_MEDIA_TYPE,
                },
            )
            assert resp.status == 409

            resp = await client.post(
                "/api/library",
                json={
                    "scope": "local",
                    "name": "My Workflow",
                    "digest": digest,
                    "mediaType": WORKFLOW_MEDIA_TYPE,
                    "labels": ["wip"],
                },
            )
            assert resp.status == 201
            record = await resp.json()
            assert record["revision"] == 1
            assert record["labels"] == ["wip"]

            # Scoped read: right scope hits, wrong scope is a 404 miss.
            path = f"/api/library/{record['id']}"
            assert (await client.get(path + "?scope=local")).status == 200
            assert (await client.get(path + "?scope=other")).status == 404
            assert (await client.get(path)).status == 400  # scope required

            # PATCH: optimistic concurrency, unknown digests refused.
            resp = await client.patch(
                path,
                json={"scope": "local", "revision": 1, "name": "Renamed"},
            )
            assert resp.status == 200
            updated = await resp.json()
            assert updated["name"] == "Renamed"
            assert updated["revision"] == 2
            resp = await client.patch(path, json={"scope": "local", "revision": 1, "name": "stale"})
            assert resp.status == 409
            resp = await client.patch(
                path, json={"scope": "local", "revision": 2, "digest": DIGEST_B}
            )
            assert resp.status == 409  # bytes not held

            # DELETE removes the record, never the bytes.
            assert (await client.delete(path + "?scope=other")).status == 404
            assert (await client.delete(path + "?scope=local")).status == 204
            assert (await client.get(path + "?scope=local")).status == 404
            assert (await client.get(f"/api/assets/{digest}")).status == 200
        finally:
            await client.close()

    asyncio.run(scenario())


def test_library_browse_query_first_paging(tmp_path: Path) -> None:
    """The browse surface is query-first: cursors are bound to the query
    that produced them, changing the query invalidates the cursor loudly,
    and pages never repeat or skip."""

    async def scenario() -> None:
        client = await make_client(tmp_path)
        try:
            body = b'{"graphs": {}}'
            digest = (await (await client.post("/api/assets", data=body)).json())["digest"]
            for index in range(5):
                resp = await client.post(
                    "/api/library",
                    json={
                        "scope": "local",
                        "name": f"Workflow {index}",
                        "digest": digest,
                        "mediaType": WORKFLOW_MEDIA_TYPE,
                    },
                )
                assert resp.status == 201

            assert (await client.get("/api/library")).status == 400  # scope required

            resp = await client.get("/api/library?scope=local&limit=2")
            page = await resp.json()
            names = [r["name"] for r in page["records"]]
            assert len(names) == 2 and "cursor" in page

            resp = await client.get(f"/api/library?scope=local&limit=2&cursor={page['cursor']}")
            page2 = await resp.json()
            names += [r["name"] for r in page2["records"]]
            resp = await client.get(f"/api/library?scope=local&limit=2&cursor={page2['cursor']}")
            page3 = await resp.json()
            names += [r["name"] for r in page3["records"]]
            assert names == [f"Workflow {i}" for i in reversed(range(5))]
            assert "cursor" not in page3  # exhausted

            # The cursor binds its query: a different q/label/scope is 400.
            resp = await client.get(
                f"/api/library?scope=local&q=up&limit=2&cursor={page['cursor']}"
            )
            assert resp.status == 400
            resp = await client.get(f"/api/library?scope=other&limit=2&cursor={page['cursor']}")
            assert resp.status == 400
            assert (await client.get("/api/library?scope=local&cursor=garbage")).status == 400

            # Filters ride the same surface.
            resp = await client.get("/api/library?scope=local&q=workflow%203")
            found = await resp.json()
            assert [r["name"] for r in found["records"]] == ["Workflow 3"]
        finally:
            await client.close()

    asyncio.run(scenario())
