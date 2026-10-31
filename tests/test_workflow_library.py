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
import json
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import (
    AssetError,
    AssetRef,
    AssetVault,
    AssetWriter,
    LibraryStore,
    LocalAssetRef,
    MountDef,
    MountTable,
    StaleRevision,
    digest_bytes,
)
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EventListener, ExecutionRuntime
from dinkster_protocol import (
    AssetFolderContext,
    Invocation,
    InvocationResult,
    OnInvocationEvent,
    SavedArtifact,
)
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
from dinkster_server.library import LIBRARY_KEY
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker
from PIL import Image

WORKFLOW_MEDIA_TYPE = "application/x-dinkster-workflow+json"

STRING = TypeExpr.concrete("STRING")


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


def test_catalog_concurrent_initialization_keeps_stores_independent(tmp_path: Path) -> None:
    def create(index: int) -> None:
        store = LibraryStore(tmp_path / f"library-{index}.sqlite")
        try:
            store.create("local", str(index), digest_bytes(str(index).encode()), "text/plain")
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(create, range(8)))
    for index in range(8):
        store = LibraryStore(tmp_path / f"library-{index}.sqlite")
        try:
            assert [record.name for record in store.query("local")] == [str(index)]
        finally:
            store.close()


@pytest.mark.parametrize("catalog", ["fresh", "unversioned", "versioned"])
def test_catalog_alembic_head_preserves_records(tmp_path: Path, catalog: str) -> None:
    path = tmp_path / "library.sqlite"
    before: dict[str, list[tuple[object, ...]]] = {}
    if catalog != "fresh":
        shutil.copyfile(Path(__file__).parent / "fixtures/asset-catalog-previous-main.sqlite", path)
        with sqlite3.connect(path) as connection:
            if catalog == "versioned":
                connection.executescript(
                    "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL PRIMARY KEY);"
                    "INSERT INTO alembic_version VALUES ('0001_asset_records');"
                )
            for table in ("records", "media_grants", "asset_dependencies", "asset_derivations"):
                before[table] = connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
                assert len(before[table]) == 2
    store = LibraryStore(path)
    try:
        assert store.cleanup_limit() == 21474836480
        for row in before.get("records", []):
            record = store.get(str(row[1]), str(row[0]))
            assert record is not None
            assert (record.name, record.digest, record.media_type, record.folder) == (
                row[2],
                row[3],
                row[4],
                row[6],
            )
            assert list(record.labels) == json.loads(str(row[5]))
            assert (record.created, record.modified, record.revision) == row[7:]
        for row in before.get("media_grants", []):
            grant = store.media_grant(str(row[0]), str(row[1]), str(row[2]))
            assert grant is not None
            assert (grant.media_type, grant.extension, grant.byte_size) == row[3:]
        for digest, manifest in before.get("asset_dependencies", []):
            assert store.get_dependencies(str(digest)) == json.loads(str(manifest))
        for key, manifest in before.get("asset_derivations", []):
            assert store.get_derivation(str(key)) == json.loads(str(manifest))
    finally:
        store.close()
    with sqlite3.connect(path) as connection:
        for table, rows in before.items():
            after = connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            if table == "asset_derivations":
                assert [row[:2] for row in after] == rows
                assert all(row[2] is None for row in after)
            else:
                assert after == rows
    args = [
        sys.executable,
        "-m",
        "alembic",
        "-c",
        "packages/dinkster-assets/alembic.ini",
        "-x",
        f"catalog={path}",
    ]
    current = subprocess.run([*args, "current"], text=True, capture_output=True, check=True)
    assert "0006_cleanup_policy (head)" in current.stdout
    checked = subprocess.run([*args, "check"], text=True, capture_output=True, check=True)
    assert "No new upgrade operations detected" in checked.stdout
    LibraryStore(path).close()


@pytest.mark.parametrize(
    "change",
    [
        "CREATE TRIGGER erase_record AFTER INSERT ON records "
        "BEGIN DELETE FROM records WHERE id=NEW.id; END",
        "CREATE VIEW record_names AS SELECT name FROM records",
        "ALTER TABLE media_grants RENAME TO old_media_grants; "
        "CREATE TABLE media_grants (scope TEXT COLLATE NOCASE NOT NULL, "
        "digest TEXT NOT NULL, kind TEXT NOT NULL, media_type TEXT NOT NULL, "
        "extension TEXT NOT NULL, byte_size INTEGER NOT NULL, PRIMARY KEY (scope, digest, kind)); "
        "INSERT INTO media_grants SELECT * FROM old_media_grants; DROP TABLE old_media_grants",
        "DROP TABLE media_grants",
        "ALTER TABLE records ADD COLUMN unexpected TEXT",
        "DROP INDEX records_scope_modified",
        "DROP INDEX records_scope_modified; "
        "CREATE INDEX records_scope_modified ON records(scope, modified, id)",
    ],
)
def test_catalog_refuses_incompatible_unversioned_schema(tmp_path: Path, change: str) -> None:
    path = tmp_path / "library.sqlite"
    shutil.copyfile(Path(__file__).parent / "fixtures/asset-catalog-previous-main.sqlite", path)
    with sqlite3.connect(path) as connection:
        connection.executescript(change)
        tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        rows = {
            table: connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            for (table,) in tables
        }
    with pytest.raises(ValueError, match="unversioned asset catalog"):
        LibraryStore(path)
    with sqlite3.connect(path) as connection:
        for table, expected in rows.items():
            assert connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall() == expected
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name='alembic_version'"
            ).fetchall()
            == []
        )


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


def test_store_read_only_lookup_refuses_writes_and_creation(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    try:
        record = store.create("local", "Saved", digest_bytes(b"saved"), WORKFLOW_MEDIA_TYPE)
        reader = LibraryStore(tmp_path / "library.sqlite", read_only=True)
        try:
            assert reader.get("local", record.id) == record
            assert reader.get("other", record.id) is None
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                reader.update("local", record.id, revision=record.revision, name="Changed")
            assert store.get("local", record.id) == record
        finally:
            reader.close()
    finally:
        store.close()
    missing = tmp_path / "missing" / "library.sqlite"
    with pytest.raises(sqlite3.OperationalError, match="unable to open database file"):
        LibraryStore(missing, read_only=True)
    assert not missing.parent.exists()


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


def test_local_binding_http_reads_live_grants_without_content_identity(tmp_path: Path) -> None:
    from dinkster_server.auth import Principal, StaticBearerAuthenticator

    async def scenario() -> None:
        root = tmp_path / "models"
        root.mkdir()
        source = root / "scene.obj"
        payload = b"v 0 1 2\nv 3 4 5\n"
        source.write_bytes(payload)
        table = MountTable()
        table.add(MountDef("models", root, "read"), kind="media/model3d")
        ref = table.bind_name("scene.obj", "media/model3d")
        assert isinstance(ref, LocalAssetRef)
        library = ServerLibrary(
            AssetVault(tmp_path / "vault"), make_store(tmp_path), local_resolver=table
        )
        auth = StaticBearerAuthenticator(
            {
                "reader": Principal("reader", {"local": frozenset({"assets:read"})}),
                "foreign": Principal("foreign", {"other": frozenset({"assets:read"})}),
                "writer": Principal("writer", {"local": frozenset({"assets:write"})}),
            }
        )
        client = TestClient(
            TestServer(create_app(make_engine, SCHEMAS, library=library, authenticator=auth))
        )
        await client.start_server()
        url = f"/api/assets/local/{ref.binding}"
        reader = {"Authorization": "Bearer reader", "If-None-Match": "*"}
        try:
            assert (await client.get(url)).status == 401
            for token in ("foreign", "writer"):
                assert (
                    await client.get(url, headers={"Authorization": f"Bearer {token}"})
                ).status == 403
            assert (await client.get(url + "?scope=other", headers=reader)).status == 403
            assert (await client.get(url + "?path=scene.obj", headers=reader)).status == 400
            assert (
                await client.get(url + "?scope=local&scope=local", headers=reader)
            ).status == 400
            response = await client.get(url, headers=reader)
            assert response.status == 200
            assert await response.read() == payload
            assert response.headers["Cache-Control"] == "no-store"
            assert "ETag" not in response.headers
            assert table.resolve(digest_bytes(payload)) is None
            for unavailable in ("unknown-binding", "scene.obj"):
                response = await client.get(f"/api/assets/local/{unavailable}", headers=reader)
                assert response.status == 404
                assert str(root) not in await response.text()
                assert response.headers["Cache-Control"] == "no-store"
            replacement = root / "replacement.obj"
            replacement.write_bytes(b"v 9 8 7\n")
            replacement.replace(source)
            response = await client.get(url, headers=reader)
            assert response.status == 404
            assert str(root) not in await response.text()
            fresh = table.bind_name("scene.obj", "media/model3d")
            assert isinstance(fresh, LocalAssetRef)
            fresh_url = f"/api/assets/local/{fresh.binding}"
            assert await (await client.get(fresh_url, headers=reader)).read() == b"v 9 8 7\n"
            table.remove("models")
            response = await client.get(fresh_url, headers=reader)
            assert response.status == 404
            assert response.headers["Cache-Control"] == "no-store"
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["document", "history", "shown-result", "preview"])
def test_retained_assets_http_survive_restart_and_reject_stale_release(
    tmp_path: Path, kind: str
) -> None:
    async def scenario() -> None:
        url = f"/api/library/retentions/{kind}/owner?scope=local"
        client = await make_client(tmp_path)
        try:
            response = await client.put(
                url, json={"revision": 0, "digests": [DIGEST_B, DIGEST_A, DIGEST_A]}
            )
            assert response.status == 200
            assert (await response.json()) == {
                "scope": "local",
                "ownerKind": kind,
                "ownerId": "owner",
                "revision": 1,
                "digests": sorted([DIGEST_A, DIGEST_B]),
                "complete": False,
            }
        finally:
            await client.close()
        client = await make_client(tmp_path)
        try:
            assert (await (await client.get(url)).json())["revision"] == 1
            assert (await client.head(url + "&revision=1")).status == 200
            assert (await (await client.get(url)).json())["digests"] == sorted([DIGEST_A, DIGEST_B])
            assert (await client.get(url.replace("local", "other"))).status == 404
            assert (await client.delete(url + "&revision=0")).status == 400
            response = await client.put(url, json={"revision": 1, "digests": [DIGEST_B]})
            assert response.status == 200
            assert (await response.json())["revision"] == 2
            assert (await client.delete(url + "&revision=1")).status == 409
            response = await client.delete(url + "&revision=2")
            assert response.status == 200
            assert (await response.json())["digests"] == []
            # Reusing the owner must not reset revisions and accept an old tab.
            assert (
                await client.put(url, json={"revision": 0, "digests": [DIGEST_A]})
            ).status == 409
            response = await client.put(url, json={"revision": 3, "digests": [DIGEST_A]})
            assert response.status == 200
            assert (await response.json())["revision"] == 4
            assert (await client.delete(url + "&revision=1")).status == 409
            assert (await (await client.get(url)).json())["digests"] == [DIGEST_A]
            store = client.app[LIBRARY_KEY].store
            assert store.query("local") == []
            assert store.query_assets("local") == []
            assert store.media_grant("local", DIGEST_A, "media/image") is None
            # Retention can name missing bytes without materializing them.
            assert client.app[LIBRARY_KEY].vault.resolve(DIGEST_A) is None
        finally:
            await client.close()

    asyncio.run(scenario())


def test_retained_assets_cross_store_revisions_and_all_scope_write_barrier(tmp_path: Path) -> None:
    first, second = make_store(tmp_path), make_store(tmp_path)
    try:
        first.retain_assets("one", "document", "owner", [DIGEST_A], revision=0, complete=True)
        second.retain_assets("two", "preview", "owner", [DIGEST_B], revision=0, complete=True)
        with first.hold_asset_references() as roots:
            assert roots == {DIGEST_A, DIGEST_B}
            with sqlite3.connect(tmp_path / "library.sqlite", timeout=0) as connection:
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    connection.execute("BEGIN IMMEDIATE")
        barrier = threading.Barrier(2)

        def update(store: LibraryStore) -> str:
            barrier.wait(timeout=10)
            try:
                store.retain_assets(
                    "one", "document", "owner", [DIGEST_B], revision=1, complete=True
                )
            except StaleRevision:
                return "stale"
            return "updated"

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(update, (first, second)))
        assert sorted(results) == ["stale", "updated"]
        with second.hold_asset_references() as roots:
            assert roots == {DIGEST_B}
        with sqlite3.connect(tmp_path / "library.sqlite") as connection:
            connection.execute("UPDATE asset_retentions SET digests = 'broken' WHERE scope = 'two'")
        with pytest.raises(AssetError):
            with first.hold_asset_references():
                pytest.fail("corrupt associations were treated as empty")
    finally:
        first.close()
        second.close()


def test_catalog_asset_roots_include_other_scopes_and_reachable_dependencies(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    observed = digest_bytes(b"kept output")
    orphan = digest_bytes(b"orphaned document")
    unrelated = digest_bytes(b"unreferenced dependency")
    try:
        store.create("private", "Document", DIGEST_A, WORKFLOW_MEDIA_TYPE)
        store.record_asset(
            "another",
            AssetRef(observed, "output.txt", 11),
            original_name="output.txt",
            kind="asset/file",
            origin={"kind": "produced", "job": "run", "node": "save"},
        )
        store.put_dependencies(DIGEST_A, [{"digest": DIGEST_B}])
        store.put_dependencies(DIGEST_B, [{"digest": DIGEST_A}])
        store.put_dependencies(orphan, [{"digest": unrelated}])
        with store.hold_asset_references() as roots:
            assert roots == {DIGEST_A, DIGEST_B, observed}
        with store.hold_asset_references(additional_roots=[orphan]) as roots:
            assert roots == {DIGEST_A, DIGEST_B, observed, orphan, unrelated}
        with sqlite3.connect(tmp_path / "library.sqlite") as connection:
            connection.execute(
                "UPDATE asset_dependencies SET manifest = 'broken' WHERE digest = ?", (DIGEST_A,)
            )
        with pytest.raises(AssetError):
            with store.hold_asset_references():
                pytest.fail("unreadable dependency state was ignored")
    finally:
        store.close()


def test_derivation_references_are_durable_atomic_and_explicit(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    cache_key = digest_bytes(b"derivation key, not asset bytes")
    legacy_key = digest_bytes(b"legacy derivation key")
    decoy = digest_bytes(b"opaque provenance, not an external reference")
    manifest: dict[str, object] = {"fingerprint": decoy}
    try:
        assert store.put_derivation(cache_key, manifest, references=[DIGEST_A, DIGEST_A])
        assert not store.put_derivation(cache_key, manifest, references=[DIGEST_A])
        with pytest.raises(AssetError, match="immutable stored manifest"):
            store.put_derivation(cache_key, {"other": decoy}, references=[DIGEST_B])
        with pytest.raises(AssetError, match="immutable asset references"):
            store.put_derivation(cache_key, manifest, references=[DIGEST_B])
        store.close()
        store = make_store(tmp_path)
        assert store.get_derivation(cache_key) == manifest
        with store.hold_asset_references() as roots:
            assert roots == {DIGEST_A}
        with sqlite3.connect(tmp_path / "library.sqlite") as connection:
            connection.execute(
                "INSERT INTO asset_derivations(cache_key, manifest) VALUES (?, ?)",
                (legacy_key, json.dumps(manifest, separators=(",", ":"), sort_keys=True)),
            )
        with pytest.raises(AssetError, match="unavailable"):
            with store.hold_asset_references():
                pytest.fail("legacy derivation references were assumed empty")
        # Only the producer can certify the roots of the same immutable provenance.
        with pytest.raises(AssetError, match="immutable stored manifest"):
            store.put_derivation(legacy_key, {"different": decoy}, references=[DIGEST_B])
        with pytest.raises(AssetError, match="unavailable"):
            with store.hold_asset_references():
                pytest.fail("a conflicting declaration changed legacy ownership")
        assert not store.put_derivation(legacy_key, manifest, references=[DIGEST_B])
        with store.hold_asset_references() as roots:
            assert roots == {DIGEST_A, DIGEST_B}
        with sqlite3.connect(tmp_path / "library.sqlite") as connection:
            connection.execute("UPDATE asset_derivations SET asset_digests = 'broken'")
        with pytest.raises(AssetError, match="unavailable"):
            with store.hold_asset_references():
                pytest.fail("corrupt derivation references were assumed empty")
    finally:
        store.close()


def test_retained_assets_http_enforce_scope_and_validate_manifest(tmp_path: Path) -> None:
    from dinkster_server.auth import Principal, StaticBearerAuthenticator

    async def scenario() -> None:
        library = ServerLibrary(AssetVault(tmp_path / "vault"), make_store(tmp_path))
        auth = StaticBearerAuthenticator(
            {
                "writer": Principal("writer", {"one": frozenset({"assets:read", "assets:write"})}),
                "reader": Principal("reader", {"one": frozenset({"assets:read"})}),
            }
        )
        client = TestClient(
            TestServer(create_app(make_engine, SCHEMAS, library=library, authenticator=auth))
        )
        await client.start_server()
        url = "/api/library/retentions/document/owner?scope=one"
        writer = {"Authorization": "Bearer writer"}
        reader = {"Authorization": "Bearer reader"}
        try:
            valid = {"revision": 0, "digests": [DIGEST_A]}
            assert (await client.put(url, json=valid)).status == 401
            assert (await client.put(url, json=valid, headers=reader)).status == 403
            assert (
                await client.put(url.replace("one", "two"), json=valid, headers=writer)
            ).status == 403
            for invalid in (
                {"revision": True, "digests": []},
                {"revision": -1, "digests": []},
                {"revision": 0, "digests": ["not-a-digest"]},
                {"revision": 0, "digests": [None]},
                {"revision": 0, "digests": [DIGEST_A] * 4097},
                {"revision": 0, "digests": [], "complete": "true"},
                {"revision": 0, "digests": [], "complete": 1},
                {"digests": []},
            ):
                assert (await client.put(url, json=invalid, headers=writer)).status == 400
            assert (await client.put(url, json=valid, headers=writer)).status == 200
            assert (await client.get(url, headers=reader)).status == 200
            assert (await client.get(url.replace("one", "two"), headers=reader)).status == 403
            assert (await client.delete(url + "&revision=1", headers=reader)).status == 403
        finally:
            await client.close()

    asyncio.run(scenario())


def test_retention_completeness_and_acknowledged_replacement(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = await make_client(tmp_path)
        url = "/api/library/retentions/document/session-generation?scope=local"
        try:
            store = client.app[LIBRARY_KEY].store
            response = await client.put(url, json={"revision": 0, "digests": []})
            assert response.status == 200
            assert not (await response.json())["complete"]
            with pytest.raises(AssetError, match="incomplete"):
                with store.hold_asset_references():
                    pytest.fail("an unknown empty manifest was treated as complete")
            # Retain both versions before changing the displayed document.
            response = await client.put(
                url, json={"revision": 1, "digests": [DIGEST_A, DIGEST_B], "complete": True}
            )
            assert response.status == 200
            acknowledgment = await response.json()
            assert acknowledgment["complete"] and acknowledgment["revision"] == 2
            with store.hold_asset_references() as roots:
                assert roots == {DIGEST_A, DIGEST_B}
        finally:
            await client.close()
        client = await make_client(tmp_path)
        try:
            store = client.app[LIBRARY_KEY].store
            assert (await (await client.get(url)).json()) == acknowledgment
            # Only after switching away from A may its owner release that identity.
            response = await client.put(
                url, json={"revision": 2, "digests": [DIGEST_B], "complete": True}
            )
            assert response.status == 200
            assert (await client.delete(url + "&revision=2")).status == 409
            with store.hold_asset_references() as roots:
                assert roots == {DIGEST_B}
            response = await client.delete(url + "&revision=3")
            assert response.status == 200
            assert (await response.json())["complete"]
            with store.hold_asset_references() as roots:
                assert roots == set()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_legacy_retention_migration_preserves_unknown_ownership(tmp_path: Path) -> None:
    import sqlalchemy as sa
    from alembic import command
    from dinkster_assets.schema import migration_config

    path = tmp_path / "library.sqlite"
    config = migration_config()
    engine = sa.create_engine(sa.URL.create("sqlite", database=str(path)))
    try:
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "0004_derivation_references")
    finally:
        engine.dispose()
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO asset_retentions VALUES (?, ?, ?, ?, ?)",
            ("local", "document", "legacy", 7, json.dumps([DIGEST_A])),
        )
    store = LibraryStore(path)
    try:
        retained = store.retained_assets("local", "document", "legacy")
        assert retained is not None
        assert (retained.revision, retained.digests, retained.complete) == (7, (DIGEST_A,), False)
        with pytest.raises(AssetError, match="incomplete"):
            with store.hold_asset_references():
                pytest.fail("a migrated legacy owner was assumed complete")
        store.retain_assets("local", "document", "legacy", [DIGEST_A], revision=7, complete=True)
        with store.hold_asset_references() as roots:
            assert roots == {DIGEST_A}
    finally:
        store.close()


@pytest.mark.parametrize("kind", ("media/image", "media/audio", "media/video", "asset/file"))
def test_native_media_saves_complete_into_kind_filtered_project_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    import numpy as np
    from dinkster_nodes_media_io import SaveAudio, SaveImage, SaveText
    from dinkster_nodes_media_io.video import SaveVideo
    from dinkster_video import assemble_video
    from dinkster_workers.saved_artifacts import SavedArtifactAuthority

    root = tmp_path / "output"
    root.mkdir()
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot, output_mount="renders")
    table.add(MountDef("renders", root, "readwrite"))
    table.scan("renders")
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))

    class NativeSave(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.native_save",
                outputs=(OutputSpec("out", STRING),),
                output_node=True,
                idempotent=False,
            )

        @classmethod
        def execute(cls) -> Mapping[str, object]:
            target = {"mount": "renders", "prefix": "saved/result"}
            if kind == "media/image":
                SaveImage.execute(
                    images=np.full((1, 3, 5, 3), 0.25, dtype=np.float32), target=target
                )
            elif kind == "media/audio":
                SaveAudio.execute(
                    audio={
                        "waveform": np.zeros((1, 1, 800), dtype=np.float32),
                        "sample_rate": 8000,
                    },
                    target=target,
                )
            elif kind == "media/video":
                video = assemble_video(np.full((2, 16, 16, 3), 0.75, dtype=np.float32), fps=2)
                SaveVideo.execute(video=video, target=target, format="mp4_h264")
            else:
                SaveText.execute(text="ordinary text, not encoded media", target=target)
            return cls.outputs(out="saved")

    schemas = build_schemas([NativeSave])

    async def validate_saved(
        node_id: str, claims: Sequence[Mapping[str, object]], started_at_ns: int
    ) -> tuple[SavedArtifact, ...]:
        return SavedArtifactAuthority(snapshot).capture_mounted(claims, node_id, started_at_ns)

    def engine(on_event: EventListener) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        worker = InProcessWorker(build_node_types([NativeSave]), registry)
        runtime = ExecutionRuntime(worker=worker, validate_saved_artifacts=validate_saved)
        return Engine(
            schemas=schemas,
            registry=registry,
            worker=worker,
            pin_execution=lambda: runtime,
            cache=MemoryLRUCache(),
            on_event=on_event,
        )

    async def scenario() -> None:
        library = ServerLibrary(
            vault=AssetVault(tmp_path / "vault"),
            store=LibraryStore(tmp_path / "catalog.sqlite"),
            local_resolver=table,
            resolver=table,
            upload_writer=AssetWriter(table),
            project_folders_for=lambda _project: AssetFolderContext(("renders",), "renders"),
        )
        client = TestClient(TestServer(create_app(engine, schemas, library=library)))
        await client.start_server()
        try:
            response = await client.post(
                "/api/jobs",
                json={
                    "clientId": "client",
                    "jobId": "save",
                    "project": "film",
                    "graph": {"nodes": {"save": {"nodeType": "test.native_save", "inputs": {}}}},
                    "targets": ["save"],
                },
            )
            assert response.status == 202
            submitted = await response.json()
            async with asyncio.timeout(10):
                while True:
                    status = await (await client.get("/api/jobs/client/save")).json()
                    if status["state"] in {"completed", "failed", "cancelled"}:
                        break
                    await asyncio.sleep(0)
            assert status["state"] == "completed", status
            (artifact,) = status["artifacts"]
            path = root / artifact["virtualPath"].split("/", 2)[2]
            assert artifact["digest"] == digest_bytes(path.read_bytes())
            response = await client.get(
                "/api/assets/catalog",
                params={
                    "scope": "local",
                    "project": "film",
                    "kind": kind,
                },
            )
            assert response.status == 200
            (item,) = (await response.json())["assets"]
            assert item["asset"]["digest"] == artifact["digest"]
            (record,) = library.store.query_assets("local", project="film", kind=kind)[0]
            assert dict(record.origin) == {
                "kind": "produced",
                "job": submitted["jobRef"],
                "node": "save",
            }
            assert record.kind == kind
            if kind != "asset/file":
                assert library.store.media_grant("local", artifact["digest"], kind) is not None
            assert library.store.query_assets("other", project="film", kind=kind) == []
            assert library.vault.digests() == []
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["", "catalog", "revoked"])
def test_completed_artifacts_are_cataloged_in_the_submitted_project(
    tmp_path: Path, failure: str
) -> None:
    async def scenario() -> None:
        root = tmp_path / "output"
        root.mkdir()
        mounts = MountTable()
        mounts.add(MountDef("output", root, "readwrite"))
        mounts.scan("output")
        writer = AssetWriter(mounts)
        data = b"saved by the worker, not an input copy"

        class SavingWorker(InProcessWorker):
            async def invoke(
                self, invocation: Invocation, on_event: OnInvocationEvent | None = None
            ) -> InvocationResult:
                result = await super().invoke(invocation, on_event)
                ref = writer.save_bytes(
                    {"mount": "output", "prefix": "result"},
                    data,
                    suffix=".txt",
                    media_type="video/mp4",
                )
                if failure == "revoked":
                    mounts.remove("output")
                return replace(
                    result,
                    artifacts=(
                        SavedArtifact(
                            invocation.node_id,
                            ref.digest,
                            ref.name,
                            ref.size,
                            ref.media_type,
                            ref.virtual_path,
                        ),
                    ),
                )

        def engine(on_event: EventListener) -> Engine:
            registry = TypeRegistry()
            register_core_types(registry)
            return Engine(
                schemas=SCHEMAS,
                registry=registry,
                worker=SavingWorker(build_node_types(NODES), registry),
                cache=MemoryLRUCache(),
                on_event=on_event,
            )

        database = tmp_path / "library.sqlite"
        library = ServerLibrary(
            vault=AssetVault(tmp_path / "vault"),
            store=LibraryStore(database),
            resolver=mounts,
            upload_writer=writer,
            project_folders_for=lambda project: AssetFolderContext(
                ("output",) if project == "film" else ()
            ),
        )
        if failure == "catalog":
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "CREATE TRIGGER fail_output BEFORE INSERT ON asset_observations "
                    "BEGIN SELECT RAISE(ABORT, 'catalog offline'); END"
                )
        client = TestClient(TestServer(create_app(engine, SCHEMAS, library=library)))
        await client.start_server()
        try:
            response = await client.post(
                "/api/jobs",
                json={
                    "clientId": "client",
                    "jobId": "save",
                    "project": "film",
                    "graph": {
                        "nodes": {"save": {"nodeType": "test.echo", "inputs": {"text": "x"}}}
                    },
                    "targets": ["save"],
                },
            )
            assert response.status == 202, await response.text()
            submitted = await response.json()
            async with asyncio.timeout(10):
                while True:
                    status = await (await client.get("/api/jobs/client/save")).json()
                    if status["state"] in {"completed", "failed", "cancelled"}:
                        break
                    await asyncio.sleep(0.005)
            assert status["state"] == ("failed" if failure else "completed"), status
            records = library.store.query_assets("local", project="film", origin="produced")
            assert len(records) == (0 if failure else 1)
            assert library.store.query_assets("local") == []
            assert library.store.query_assets("other", project="film") == []
            assert library.store.media_grant("local", digest_bytes(data), "media/video") is None
            assert library.vault.digests() == []
            assert (root / "result_00001.txt").read_bytes() == data
            if not failure:
                (record,) = records[0]
                assert record.asset.digest == digest_bytes(data)
                assert record.asset.virtual_path == "mounts/output/result_00001.txt"
                assert dict(record.origin) == {
                    "kind": "produced",
                    "job": submitted["jobRef"],
                    "node": "save",
                }
        finally:
            await client.close()
        if failure == "catalog":
            with sqlite3.connect(database) as connection:
                connection.execute("DROP TRIGGER fail_output")
        restarted = await make_client(
            tmp_path,
            resolver=mounts,
            upload_writer=writer,
            project_folders_for=library.project_folders_for,
        )
        try:
            records = restarted.app[LIBRARY_KEY].store.query_assets("local", project="film")
            assert len(records) == (0 if failure == "revoked" else 1)
            journals = list(library.vault.root.glob(".upload-observations/*.json"))
            assert len(journals) == (1 if failure == "revoked" else 0)
        finally:
            await restarted.close()

    asyncio.run(scenario())


def test_anonymous_vault_inventory_is_operator_only_and_does_not_grant_scope(
    tmp_path: Path,
) -> None:
    from dinkster_assets import MediaGrant
    from dinkster_server.auth import Principal, StaticBearerAuthenticator

    vault = AssetVault(tmp_path / "vault")
    anonymous = (b"first unknown bytes", b"second unknown content", b"third unknown")
    workflow = b'{"graphs": {"main": {}}}'
    observed = b"other project private content"
    corrupt = b"damaged anonymous content"
    for data in (*anonymous, workflow, observed, corrupt):
        with vault.writer(digest_bytes(data)) as writer:
            writer.write(data)
            writer.commit()
    path = vault.resolve(digest_bytes(corrupt))
    assert path is not None
    path.write_bytes(b"wrong bytes")
    store = LibraryStore(tmp_path / "library.sqlite")
    store.create("other", "Private workflow", digest_bytes(workflow), WORKFLOW_MEDIA_TYPE)
    store.record_asset(
        "other",
        AssetRef(digest_bytes(observed), "private.bin", len(observed)),
        original_name="private.bin",
        kind="asset/file",
        project="film",
        origin={"kind": "unknown"},
    )
    store.close()
    capabilities = {
        "local": frozenset({"assets:read", "settings:write"}),
        "other": frozenset({"assets:read"}),
    }
    authenticator = StaticBearerAuthenticator(
        {
            "operator": Principal("operator", capabilities),
            "reader": Principal("reader", {"local": frozenset({"assets:read"})}),
            "agent": Principal("agent", capabilities, kind="agent"),
            "delegated": Principal("delegated", capabilities, delegation_id="delegation"),
        }
    )

    async def scenario() -> None:
        for _ in range(2):
            library = ServerLibrary(vault=vault, store=LibraryStore(tmp_path / "library.sqlite"))
            app = create_app(make_engine, SCHEMAS, library=library, authenticator=authenticator)
            client = TestClient(TestServer(app))
            await client.start_server()
            image = BytesIO()
            Image.new("RGB", (3, 2), "green").save(image, format="PNG")
            image_bytes = image.getvalue()
            with vault.writer(digest_bytes(image_bytes)) as writer:
                writer.write(image_bytes)
                writer.commit()
            library.store.grant_media(
                MediaGrant(
                    "other",
                    digest_bytes(image_bytes),
                    "media/image",
                    "image/png",
                    "png",
                    len(image_bytes),
                ),
                image_bytes,
            )
            headers = {"Authorization": "Bearer operator"}
            params = {"source": "unrecorded", "scope": "local", "limit": "1"}
            try:
                for token in ("reader", "agent", "delegated"):
                    response = await client.get(
                        "/api/assets/catalog",
                        params=params,
                        headers={"Authorization": f"Bearer {token}"},
                    )
                    assert response.status == 403
                found = []
                cursor = None
                for _page in range(len(anonymous) + 1):
                    query = {**params, **({"cursor": cursor} if cursor else {})}
                    response = await client.get(
                        "/api/assets/catalog", params=query, headers=headers
                    )
                    assert response.status == 200
                    assert response.headers["Cache-Control"] == "no-store"
                    body = await response.json()
                    assert body["source"] == "unrecorded"
                    assert len(body["assets"]) == 1
                    item = body["assets"][0]
                    found.append(item["asset"]["digest"])
                    data = next(data for data in anonymous if digest_bytes(data) == found[-1])
                    assert item == {
                        "asset": AssetRef(found[-1], found[-1].split(":")[1], len(data)).to_wire(),
                        "aliases": [found[-1].split(":")[1]],
                        "observations": [],
                    }
                    assert str(tmp_path) not in json.dumps(body)
                    cursor = body.get("cursor")
                    if cursor is None:
                        break
                    for changed in ({"source": "scoped"}, {"q": "different"}):
                        response = await client.get(
                            "/api/assets/catalog",
                            params={**params, "cursor": cursor, **changed},
                            headers=headers,
                        )
                        assert response.status == 400
                assert cursor is None
                assert found == sorted((digest_bytes(data) for data in anonymous), reverse=True)
                response = await client.get(
                    "/api/assets/catalog", params={**params, "q": found[1]}, headers=headers
                )
                assert [item["asset"]["digest"] for item in (await response.json())["assets"]] == [
                    found[1]
                ]
                for changed in (
                    {"scope": "other"},
                    {"project": "film"},
                    {"since": "0"},
                    {"kind": "asset/file"},
                    {"origin": "unknown"},
                    {"source": "invalid"},
                ):
                    response = await client.get(
                        "/api/assets/catalog", params={**params, **changed}, headers=headers
                    )
                    assert response.status == 400
                response = await client.get("/api/assets/catalog?scope=local", headers=headers)
                assert (await response.json())["assets"] == []
                assert library.store.query("local") == []
                assert library.store.query_assets("local") == []
                for digest in found:
                    assert library.store.media_grant("local", digest, "media/image") is None
            finally:
                await client.close()

    asyncio.run(scenario())


def test_catalog_backfills_known_scopes_without_claiming_anonymous_bytes(tmp_path: Path) -> None:
    from dinkster_assets import MediaGrant

    image = BytesIO()
    Image.new("RGB", (3, 5), "red").save(image, format="PNG")
    image_bytes = image.getvalue()
    damaged = BytesIO()
    Image.new("RGB", (2, 4), "blue").save(damaged, format="PNG")
    damaged_bytes = damaged.getvalue()
    workflow = b'{"graphs":{"main":{"nodes":{}}}}'
    unknown = b"anonymous historical content"
    vault = AssetVault(tmp_path / "vault")
    for data in (image_bytes, damaged_bytes, workflow, unknown):
        with vault.writer(digest_bytes(data)) as writer:
            writer.write(data)
            writer.commit()
    store = LibraryStore(tmp_path / "library.sqlite")
    for scope in ("alice", "charlie"):
        store.grant_media(
            MediaGrant(
                scope,
                digest_bytes(image_bytes),
                "media/image",
                "image/png",
                "png",
                len(image_bytes),
            ),
            image_bytes,
        )
    store.grant_media(
        MediaGrant(
            "alice",
            digest_bytes(damaged_bytes),
            "media/image",
            "image/png",
            "png",
            len(damaged_bytes),
        ),
        damaged_bytes,
    )
    store.record_asset(
        "charlie",
        AssetRef(digest_bytes(image_bytes), "original.png", len(image_bytes), "image/png"),
        original_name="original.png",
        kind="media/image",
        origin={"kind": "upload"},
        project="film",
    )
    for name in ("Story", "Alternate story"):
        store.create("bob", name, digest_bytes(workflow), WORKFLOW_MEDIA_TYPE)
    path = vault.resolve(digest_bytes(damaged_bytes))
    assert path is not None
    path.write_bytes(b"corrupted")
    store.close()

    async def scenario() -> None:
        previous: list[dict[str, object]] | None = None
        for _ in range(2):
            client = await make_client(tmp_path)
            try:
                response = await client.get("/api/assets/catalog?scope=alice")
                assert response.status == 200
                body = await response.json()
                assert len(body["assets"]) == 1
                item = body["assets"][0]
                assert item["asset"]["digest"] == digest_bytes(image_bytes)
                assert item["asset"]["size"] == len(image_bytes)
                assert item["aliases"] == [digest_bytes(image_bytes).split(":", 1)[1]]
                assert item["observations"][0]["origin"] == {"kind": "unknown"}
                assert item["observations"][0]["kind"] == "media/image"
                if previous is not None:
                    assert body["assets"] == previous
                previous = body["assets"]
                response = await client.get("/api/assets/catalog?scope=bob")
                body = await response.json()
                assert len(body["assets"]) == 1
                assert set(body["assets"][0]["aliases"]) == {"Story", "Alternate story"}
                assert {o["kind"] for o in body["assets"][0]["observations"]} == {
                    "document/workflow"
                }
                for scope in ("local", "other", "charlie"):
                    response = await client.get("/api/assets/catalog", params={"scope": scope})
                    assert (await response.json())["assets"] == []
                response = await client.get("/api/assets/catalog?scope=charlie&project=film")
                assert (await response.json())["assets"][0]["aliases"] == ["original.png"]
                assert (
                    client.app[LIBRARY_KEY].store.media_grant(
                        "other", digest_bytes(image_bytes), "media/image"
                    )
                    is None
                )
            finally:
                await client.close()

    asyncio.run(scenario())


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


def test_transferred_asset_retains_scoped_history_after_catalog_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_server.auth import Principal, StaticBearerAuthenticator

    data = b"remote model bytes with no invented classification"
    digest = digest_bytes(data)
    vault = AssetVault(tmp_path / "vault")
    authenticator = StaticBearerAuthenticator(
        {"sender": Principal("sender", {"alice": frozenset({"assets:read", "assets:write"})})}
    )

    async def scenario() -> None:
        source = await make_client(tmp_path / "source")
        response = await source.post("/api/assets", data=data)
        assert response.status == 201
        for attempt in range(2):
            library = ServerLibrary(vault=vault, store=LibraryStore(tmp_path / "library.sqlite"))
            client = TestClient(
                TestServer(
                    create_app(make_engine, SCHEMAS, library=library, authenticator=authenticator)
                )
            )
            await client.start_server()
            headers = {
                "Authorization": "Bearer sender",
                "Content-Type": "application/octet-stream",
                "X-Dinkster-Digest": digest,
                "X-Dinkster-Source-Machine": "Studio workstation",
            }
            query = {"scope": "alice", "project": "film", "name": "model.bin"}
            try:
                if attempt == 0:
                    for changed, status in (({"scope": "bob"}, 403), ({"mount": "imports"}, 400)):
                        response = await client.post(
                            "/api/assets/transfers",
                            params={**query, **changed},
                            data=data,
                            headers=headers,
                        )
                        assert response.status == status
                    for machine in (None, "", " ", "x" * 256):
                        test_headers = dict(headers)
                        if machine is None:
                            test_headers.pop("X-Dinkster-Source-Machine")
                        else:
                            test_headers["X-Dinkster-Source-Machine"] = machine
                        response = await client.post(
                            "/api/assets/transfers", params=query, data=data, headers=test_headers
                        )
                        assert response.status == 400
                    response = await client.post(
                        "/api/assets/transfers",
                        params=query,
                        data=data,
                        headers={**headers, "X-Dinkster-Digest": DIGEST_B},
                    )
                    assert response.status == 409
                    assert vault.digests() == []

                    def unavailable(*_args: object, **_kwargs: object) -> None:
                        raise sqlite3.OperationalError("catalog unavailable")

                    with monkeypatch.context() as patch:
                        patch.setattr(library.store, "record_asset", unavailable)
                        remote = await source.get(f"/api/assets/{digest}")
                        response = await client.post(
                            "/api/assets/transfers",
                            params=query,
                            data=remote.content.iter_chunked(7),
                            headers=headers,
                        )
                        assert response.status == 503
                    assert vault.resolve(digest) is not None
                    journals = list((vault.root / ".upload-observations").glob("*.json"))
                    assert len(journals) == 1
                    assert json.loads(journals[0].read_text())["origin"] == {
                        "kind": "transfer",
                        "machine": "Studio workstation",
                    }
                response = await client.get(
                    "/api/assets/catalog",
                    params={"scope": "alice", "project": "film"},
                    headers=headers,
                )
                assert response.status == 200
                items = (await response.json())["assets"]
                assert len(items) == 1
                assert items[0]["asset"] == AssetRef(digest, "model.bin", len(data)).to_wire()
                assert len(items[0]["observations"]) == 1
                observation = items[0]["observations"][0]
                assert observation["origin"] == {
                    "kind": "transfer",
                    "machine": "Studio workstation",
                }
                assert observation["kind"] == "asset/file"
                assert list((vault.root / ".upload-observations").glob("*.json")) == []
                assert library.store.query_assets("alice") == []
                assert library.store.query_assets("bob", project="film") == []
                assert library.store.media_grant("alice", digest, "media/image") is None
                response = await client.get(f"/api/assets/{digest}", headers=headers)
                assert await response.read() == data
                assert not list(vault.root.glob(".media-upload-*"))
                if attempt == 1:
                    response = await client.post(
                        "/api/assets/transfers",
                        params={**query, "name": "renamed.bin"},
                        data=data,
                        headers={**headers, "X-Dinkster-Source-Machine": "Second workstation"},
                    )
                    assert response.status == 200
                    assert await response.json() == {
                        "digest": digest,
                        "asset": AssetRef(digest, "renamed.bin", len(data)).to_wire(),
                    }
                    response = await client.get(
                        "/api/assets/catalog",
                        params={"scope": "alice", "project": "film", "origin": "transfer"},
                        headers=headers,
                    )
                    items = (await response.json())["assets"]
                    assert len(items) == 1
                    assert items[0]["aliases"] == ["model.bin", "renamed.bin"]
                    assert [event["origin"] for event in items[0]["observations"]] == [
                        {"kind": "transfer", "machine": "Studio workstation"},
                        {"kind": "transfer", "machine": "Second workstation"},
                    ]
                    assert vault.digests() == [digest]
            finally:
                await client.close()
                if attempt == 0:
                    await source.close()

    asyncio.run(scenario())


def test_named_model_upload_streams_to_folder_without_redirecting_workflows(tmp_path: Path) -> None:
    header = json.dumps(
        {"weights": {"dtype": "F32", "shape": [300_000], "data_offsets": [0, 1_200_000]}}
    ).encode()
    header += b" " * (-len(header) % 8)
    data = len(header).to_bytes(8, "little") + header + bytes(1_200_000)

    async def scenario() -> None:
        root = tmp_path / "imports"
        root.mkdir()
        mounts = MountTable()
        mounts.add(MountDef("imports", root, "readwrite"))
        mounts.scan("imports")
        client = await make_client(
            tmp_path,
            resolver=mounts,
            upload_writer=AssetWriter(mounts),
            project_folders_for=lambda _project: AssetFolderContext(("imports",)),
        )
        try:
            for folder, expected in (
                ("", "model.safetensors"),
                ("", "model_00001.safetensors"),
                ("checkpoints", "checkpoints/model.safetensors"),
            ):

                async def chunks():
                    for start in range(0, len(data), 70_001):
                        yield data[start : start + 70_001]

                response = await client.post(
                    "/api/assets",
                    params={"scope": "local", "name": "model.safetensors", "folder": folder},
                    data=chunks(),
                    headers={"Content-Type": "application/octet-stream"},
                )
                assert response.status == 201, await response.text()
                result = await response.json()
                assert result["digest"] == digest_bytes(data)
                assert result["asset"] == {
                    "digest": digest_bytes(data),
                    "name": Path(expected).name,
                    "size": len(data),
                    "mediaType": "application/octet-stream",
                    "virtualPath": "mounts/imports/" + expected,
                }
                assert (root / expected).read_bytes() == data
                response = await client.get("/api/assets/" + result["digest"])
                assert response.status == 200
                assert await response.read() == data
            library = client.app[LIBRARY_KEY]
            assert library.vault.digests() == []
            assert library.store.media_grant("local", digest_bytes(data), "media/image") is None
            assert library.store.media_grant("local", digest_bytes(data), "media/video") is None
            assert not list(library.vault.root.glob(".media-upload-*"))
            workflow = b'{"graphs":{"main":{"nodes":{}}}}'
            response = await client.post(
                "/api/assets", data=workflow, headers={"Content-Type": WORKFLOW_MEDIA_TYPE}
            )
            assert response.status == 201
            assert await response.json() == {"digest": digest_bytes(workflow)}
            workflow_path = library.vault.resolve(digest_bytes(workflow))
            assert workflow_path is not None
            assert workflow_path.read_bytes() == workflow
        finally:
            await client.close()

    asyncio.run(scenario())


def test_named_upload_refuses_project_exclusion_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_server import library as library_module

    async def scenario() -> None:
        root = tmp_path / "imports"
        root.mkdir()
        mounts = MountTable()
        mounts.add(MountDef("imports", root, "readwrite"))
        mounts.scan("imports")
        context = [AssetFolderContext(("imports",))]
        client = await make_client(
            tmp_path,
            resolver=mounts,
            upload_writer=AssetWriter(mounts),
            project_folders_for=lambda _project: context[0],
        )
        original = library_module._write_asset_observation_journal

        def exclude(journal: Path, payload: Mapping[str, object]) -> None:
            original(journal, payload)
            context[0] = AssetFolderContext(())

        try:
            with monkeypatch.context() as patch:
                patch.setattr(library_module, "_write_asset_observation_journal", exclude)
                response = await client.post(
                    "/api/assets?scope=local&project=film&name=model.bin", data=b"model bytes"
                )
            assert response.status == 503
            assert not (root / "model.bin").exists()
            assert client.app[LIBRARY_KEY].vault.digests() == []
            assert not list(root.rglob(".dinkster-save-*"))
            for selected in (AssetFolderContext(()), AssetFolderContext(("imports",))):
                context[0] = selected
                response = await client.get("/api/assets/catalog?scope=local&project=film")
                assert response.status == 200
                assert (await response.json())["assets"] == []
                assert not (root / "model.bin").exists()
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ("before-observation", "after-observation"))
@pytest.mark.parametrize("mounted", (False, True))
def test_upload_catalog_recovers_failed_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, mounted: bool
) -> None:
    async def scenario() -> None:
        root = tmp_path / "imports"
        root.mkdir()
        mounts = MountTable()
        mounts.add(MountDef("imports", root, "readwrite"))
        mounts.scan("imports")
        options: dict[str, object] = (
            {
                "resolver": mounts,
                "upload_writer": AssetWriter(mounts),
                "project_folders_for": lambda _project: AssetFolderContext(("imports",)),
            }
            if mounted
            else {}
        )
        image = BytesIO()
        Image.new("RGB", (3, 2), (21, 50, 100)).save(image, format="PNG")
        body = image.getvalue()
        client = await make_client(tmp_path, **options)
        original = LibraryStore.record_asset

        def fail(store: LibraryStore, *args: object, **kwargs: object) -> object:
            if failure == "after-observation":
                original(store, *args, **kwargs)  # type: ignore[arg-type]
            raise sqlite3.OperationalError("injected observation failure")

        try:
            with monkeypatch.context() as patch:
                patch.setattr(LibraryStore, "record_asset", fail)
                response = await client.post(
                    "/api/assets/media?scope=local&name=portrait.png&kind=media/image",
                    data=body,
                    headers={"Content-Type": "image/png"},
                )
            assert response.status == 503, await response.text()
            assert len(list((tmp_path / "vault" / ".upload-observations").glob("*.json"))) == 1
            if mounted:
                assert (root / "portrait.png").read_bytes() == body
                assert client.app[LIBRARY_KEY].vault.digests() == []
            else:
                path = client.app[LIBRARY_KEY].vault.resolve(digest_bytes(body))
                assert path is not None and path.read_bytes() == body
        finally:
            await client.close()

        # Revocation must not be undone by the pending recovery record.
        if mounted:
            mounts.remove("imports")
            client = await make_client(tmp_path, **options)
            try:
                result = await (await client.get("/api/assets/catalog?scope=local")).json()
                assert len(result["assets"]) == (1 if failure == "after-observation" else 0)
                assert len(list((tmp_path / "vault" / ".upload-observations").glob("*.json"))) == 1
            finally:
                await client.close()
            mounts.add(MountDef("imports", root, "readwrite"))
            mounts.scan("imports")

        before = None
        for _restart in range(2):
            client = await make_client(tmp_path, **options)
            try:
                result = await (await client.get("/api/assets/catalog?scope=local")).json()
                assert len(result["assets"]) == 1
                observed = result["assets"][0]
                assert observed["aliases"] == ["portrait.png"]
                assert len(observed["observations"]) == 1
                assert observed["observations"][0]["origin"] == {"kind": "upload"}
                assert observed["asset"]["digest"] == digest_bytes(body)
                assert observed["asset"]["virtualPath"] == (
                    "mounts/imports/portrait.png" if mounted else ""
                )
                assert await (await client.get("/api/assets/" + digest_bytes(body))).read() == body
                assert not list((tmp_path / "vault" / ".upload-observations").glob("*.json"))
                if before is not None:
                    assert result == before
                before = result
            finally:
                await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("published,changed", ((False, False), (True, False), (True, True)))
def test_upload_catalog_survives_process_exit_at_publication(
    tmp_path: Path, published: bool, changed: bool
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio, os, runpy, sys
from pathlib import Path
from io import BytesIO
from PIL import Image
from dinkster_assets import AssetWriter, MountDef, MountTable
from dinkster_protocol import AssetFolderContext

make_client = runpy.run_path('tests/test_workflow_library.py')['make_client']
root = Path(sys.argv[1])
folder = root / 'imports'
folder.mkdir()
mounts = MountTable()
mounts.add(MountDef('imports', folder, 'readwrite'))
mounts.scan('imports')
link = os.link
def interrupted(source, destination, **kwargs):
    if Path(destination).name == 'portrait.png':
        if sys.argv[2] == 'True':
            link(source, destination, **kwargs)
        os._exit(86)
    return link(source, destination, **kwargs)
os.link = interrupted
async def upload():
    client = await make_client(root, resolver=mounts, upload_writer=AssetWriter(mounts),
        project_folders_for=lambda project: AssetFolderContext(('imports',)))
    image = BytesIO()
    Image.new('RGB', (3, 2), (21, 50, 100)).save(image, format='PNG')
    await client.post('/api/assets/media?scope=local&name=portrait.png&kind=media/image',
        data=image.getvalue(), headers={'Content-Type': 'image/png'})
asyncio.run(upload())
""",
            str(tmp_path),
            str(published),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 86, result.stdout + result.stderr
    folder = tmp_path / "imports"
    assert (folder / "portrait.png").exists() == published
    if changed:
        (folder / "portrait.png").write_bytes(b"replaced after publication")

    async def scenario() -> None:
        mounts = MountTable()
        mounts.add(MountDef("imports", folder, "readwrite"))
        # Start recovery before scan readiness, then let the live catalog retry it.
        client = await make_client(
            tmp_path,
            resolver=mounts,
            upload_writer=AssetWriter(mounts),
            project_folders_for=lambda _project: AssetFolderContext(("imports",)),
        )
        try:
            mounts.scan("imports")
            response = await client.get("/api/assets/catalog?scope=local")
            assert response.status == 200
            assets = (await response.json())["assets"]
            image = BytesIO()
            Image.new("RGB", (3, 2), (21, 50, 100)).save(image, format="PNG")
            body = image.getvalue()
            recovered = published and not changed
            assert len(assets) == int(recovered)
            grant = client.app[LIBRARY_KEY].store.media_grant(
                "local", digest_bytes(body), "media/image"
            )
            assert (grant is not None) == recovered
            if recovered:
                assert assets[0]["asset"]["digest"] == digest_bytes(body)
                assert assets[0]["aliases"] == ["portrait.png"]
                assert len(assets[0]["observations"]) == 1
                response = await client.get("/api/assets/" + digest_bytes(body))
                assert response.status == 200 and await response.read() == body
                assert client.app[LIBRARY_KEY].vault.digests() == []
        finally:
            await client.close()

    asyncio.run(scenario())


def test_asset_availability_is_read_only_and_checks_local_identity(tmp_path: Path) -> None:
    class FetchOnResolve:
        def resolve(self, _digest: str) -> Path | None:
            raise AssertionError("availability must not call the acquisition resolver")

    async def scenario() -> None:
        root = tmp_path / "source"
        root.mkdir()
        mounted_body = b"mounted original"
        mounted = root / "same-name.bin"
        mounted.write_bytes(mounted_body)
        mounts = MountTable()
        mounts.add(MountDef("source", root, "read"))
        mounts.scan("source")
        client = await make_client(tmp_path, resolver=FetchOnResolve(), local_resolver=mounts)
        vault_body = b"vault original"
        missing_body = b"same name is not identity"
        digests = [digest_bytes(body) for body in (mounted_body, missing_body, vault_body)]
        try:
            upload = await client.post("/api/assets", data=vault_body)
            assert upload.status == 201
            vault = client.app[LIBRARY_KEY].vault
            before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
            for step, expected in enumerate(
                (
                    ["available", "missing", "available"],
                    ["missing", "missing", "available"],
                    ["available", "missing", "available"],
                    ["missing", "missing", "missing"],
                )
            ):
                response = await client.get(
                    "/api/assets/availability",
                    params=[
                        ("scope", "local"),
                        *[("digest", digest) for digest in digests],
                        ("digest", digests[0]),
                    ],
                )
                assert response.status == 200, await response.text()
                assert response.headers["Cache-Control"] == "no-store"
                result = await response.json()
                assert result == {
                    "assets": [
                        {"digest": digest, "status": status}
                        for digest, status in zip(digests, expected, strict=True)
                    ]
                }
                assert str(tmp_path) not in json.dumps(result)
                if step == 0:
                    assert (
                        sorted(
                            path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")
                        )
                        == before
                    )
                    # Same filename, size and mtime still cannot substitute different bytes.
                    previous = mounted.stat()
                    replacement = root / "replacement.bin"
                    replacement.write_bytes(b"mounted replaced")
                    os.utime(replacement, ns=(previous.st_atime_ns, previous.st_mtime_ns))
                    replacement.replace(mounted)
                elif step == 1:
                    (root / "renamed.bin").write_bytes(mounted_body)
                    mounts.scan("source")
                elif step == 2:
                    mounts.remove("source")
                    assert not vault.has(digests[0]) and not vault.has(digests[1])
                    path = vault.resolve(digests[-1])
                    assert path is not None
                    path.write_bytes(b"vault replaced")

            for query in (
                [("scope", "local")],
                [("digest", "not-a-digest")],
                [("digest", digests[0]), ("acquireAssets", digests[0])],
                [("digest", digests[0]), ("scope", "local"), ("scope", "other")],
            ):
                response = await client.get("/api/assets/availability", params=query)
                assert response.status == 400
            assert await (await client.get("/api/assets/catalog?scope=local")).json() == {
                "assets": []
            }
            assert not list(vault.root.glob(".ingest-*"))
            assert not (vault.root / ".upload-observations").exists()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_asset_catalog_persists_aliases_and_pages_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from dinkster_server import library as library_module

    async def scenario() -> None:
        root = tmp_path / "imports"
        root.mkdir()
        mounts = MountTable()
        mounts.add(MountDef("imports", root, "readwrite"))
        mounts.scan("imports")
        client = await make_client(
            tmp_path,
            resolver=mounts,
            upload_writer=AssetWriter(mounts),
            project_folders_for=lambda _project: AssetFolderContext(("imports",)),
        )
        bodies = (b"alpha", b"beta", b"alpha", b"gamma")
        names = ("first.bin", "second.bin", "alias.bin", "last.bin")
        try:
            for observed_at, name, body in zip(
                (100.0, 110.0, 120.0, 130.0), names, bodies, strict=True
            ):
                with monkeypatch.context() as patch:
                    patch.setattr(
                        library_module, "time", SimpleNamespace(time=lambda now=observed_at: now)
                    )
                    response = await client.post(
                        "/api/assets",
                        params={"scope": "local", "name": name},
                        data=body,
                    )
                assert response.status == 201, await response.text()
            response = await client.get("/api/assets/catalog?scope=local")
            assert response.status == 200, await response.text()
            before = await response.json()
            assert [item["asset"]["digest"] for item in before["assets"]] == [
                digest_bytes(b"gamma"),
                digest_bytes(b"alpha"),
                digest_bytes(b"beta"),
            ]
            assert before["assets"][1]["aliases"] == ["first.bin", "alias.bin"]
            assert [item["origin"] for item in before["assets"][1]["observations"]] == [
                {"kind": "upload"},
                {"kind": "upload"},
            ]
            assert str(root) not in json.dumps(before)
            assert client.app[LIBRARY_KEY].vault.digests() == []
            assert await (await client.get("/api/library?scope=local")).json() == {"records": []}
        finally:
            await client.close()

        client = await make_client(
            tmp_path,
            resolver=mounts,
            upload_writer=AssetWriter(mounts),
            project_folders_for=lambda _project: AssetFolderContext(("imports",)),
        )
        try:
            assert await (await client.get("/api/assets/catalog?scope=local")).json() == before
            seen = []
            cursor = None
            for _page in range(3):
                params = {"scope": "local", "limit": "1"}
                if cursor:
                    params["cursor"] = cursor
                response = await client.get("/api/assets/catalog", params=params)
                assert response.status == 200
                page = await response.json()
                seen.extend(page["assets"])
                cursor = page.get("cursor")
                if _page == 0:
                    mismatch = await client.get(
                        "/api/assets/catalog", params={**params, "cursor": cursor, "q": "alias"}
                    )
                    assert mismatch.status == 400
            assert seen == before["assets"]
            assert cursor is None
            middle = before["assets"][1]["observations"][-1]["created"]
            for query, expected in (
                (
                    {"q": "ALIAS", "kind": "asset/file", "origin": "upload"},
                    [digest_bytes(b"alpha")],
                ),
                ({"since": str(middle), "until": str(middle)}, [digest_bytes(b"alpha")]),
                ({"kind": "media/image"}, []),
                ({"origin": "download"}, []),
                ({"project": "other"}, []),
                ({"scope": "other"}, []),
            ):
                response = await client.get(
                    "/api/assets/catalog", params={"scope": "local", **query}
                )
                assert response.status == 200
                assert [
                    item["asset"]["digest"] for item in (await response.json())["assets"]
                ] == expected
            for query in ({"since": "nan"}, {"until": "inf"}, {"since": "2", "until": "1"}):
                response = await client.get(
                    "/api/assets/catalog", params={"scope": "local", **query}
                )
                assert response.status == 400
            for body in (b"alpha", b"beta", b"gamma"):
                response = await client.get("/api/assets/" + digest_bytes(body))
                assert response.status == 200
                assert await response.read() == body
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("tied_time", (False, True))
def test_asset_catalog_retains_scoped_origin_events_and_redacts_download_leads(
    tmp_path: Path,
    tied_time: bool,
) -> None:
    path = tmp_path / "library.sqlite"
    ref = AssetRef(digest_bytes(b"weights"), "weights.bin", 7)
    store = LibraryStore(path)
    try:
        events = (
            (
                "download.bin",
                {
                    "kind": "download",
                    "url": "https://user:password@assets.example/weights?signature=secret#part",
                },
            ),
            ("remote.bin", {"kind": "transfer", "machine": "workstation"}),
            ("render.bin", {"kind": "produced", "job": "job-7", "node": "Save"}),
        )
        for index in (2, 0, 1):
            name, origin = events[index]
            store.record_asset(
                "a",
                ref,
                original_name=name,
                kind="asset/file",
                origin=origin,
                created=100.0 if tied_time else 100.0 + index,
                observation_id=f"{index + 1 if tied_time else 3 - index:032x}",
            )
        store.record_asset(
            "b", ref, original_name="private.bin", kind="asset/file", origin={"kind": "upload"}
        )
    finally:
        store.close()
    store = LibraryStore(path)
    try:
        observations = store.query_assets("a", text="ASSETS.EXAMPLE", origin="download")
        assert len(observations) == 1
        assert [item.original_name for item in observations[0]] == [
            "download.bin",
            "remote.bin",
            "render.bin",
        ]
        assert [dict(item.origin) for item in observations[0]] == [
            {"kind": "download", "url": "https://assets.example/weights"},
            {"kind": "transfer", "machine": "workstation"},
            {"kind": "produced", "job": "job-7", "node": "Save"},
        ]
        assert store.query_assets("b", origin="download") == []
        assert store.query_assets("a", text="private") == []
        for field in ("/home/user/private.bin", "C:\\Users\\user\\private.bin"):
            with pytest.raises(AssetError, match="basenames"):
                store.record_asset(
                    "a", ref, original_name=field, kind="asset/file", origin={"kind": "upload"}
                )
    finally:
        store.close()


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


@pytest.mark.parametrize("destination", ("anonymous", "named", "transfer"))
def test_upload_declared_digest_and_limits(tmp_path: Path, destination: str) -> None:
    async def oversized_chunks():
        yield b"x" * 32
        await asyncio.sleep(0)
        yield b"x" * 33

    async def scenario() -> None:
        root = tmp_path / "imports"
        root.mkdir()
        mounts = MountTable()
        mounts.add(MountDef("imports", root, "readwrite"))
        mounts.scan("imports")
        client = await make_client(
            tmp_path,
            upload_limit=64,
            resolver=mounts,
            upload_writer=AssetWriter(mounts),
            project_folders_for=lambda _project: AssetFolderContext(("imports",)),
        )
        path = {
            "anonymous": "/api/assets",
            "named": "/api/assets?scope=local&name=source.bin",
            "transfer": "/api/assets/transfers?scope=local&name=source.bin",
        }[destination]
        headers = {"X-Dinkster-Source-Machine": "Sender"} if destination == "transfer" else {}
        try:
            body = b'{"graphs": {}}'
            resp = await client.post(
                path, data=body, headers={**headers, "X-Dinkster-Digest": DIGEST_B}
            )
            assert resp.status == (400 if destination == "anonymous" else 409)
            assert (await client.get(f"/api/assets/{digest_bytes(body)}")).status == 404
            if destination == "transfer":
                assert (await client.post(path, data=body, headers=headers)).status == 400
            headers["X-Dinkster-Digest"] = digest_bytes(body)
            resp = await client.post(path, data=body, headers=headers)
            assert resp.status == 201
            saved_files = sorted(entry.name for entry in root.iterdir())

            assert (await client.post(path, data=b"x" * 65, headers=headers)).status == 413
            assert (await client.post(path, data=oversized_chunks(), headers=headers)).status == 413
            assert (await client.post(path, data=b"", headers=headers)).status == 400
            if destination == "named":
                assert (root / "source.bin").read_bytes() == body
                assert sorted(entry.name for entry in root.iterdir()) == saved_files
                assert client.app[LIBRARY_KEY].vault.digests() == []
            elif destination == "transfer":
                assert not (root / "source.bin").exists()
                assert client.app[LIBRARY_KEY].vault.digests() == [digest_bytes(body)]
            assert not list(client.app[LIBRARY_KEY].vault.root.glob(".media-upload-*"))
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
