"""Persistent execution history: terminal jobs on disk, scope-keyed.

The queue's in-memory history is a polling grace window; HistoryStore is
the durable record. Every terminal job lands with its stable per-run
identity (runId), job identity, principal, scope, timings, error payload,
run summary counts, and the execution-opaque sourceDocument link back to
the uploaded workflow asset.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from collections.abc import Mapping
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from typing import cast

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import AssetError, AssetRef, digest_bytes
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EventListener, ExecutionSelection, PlanExecution, RunResult
from dinkster_graph import Graph, GraphNode, graph_to_wire
from dinkster_library import LibraryClient, LibraryError
from dinkster_protocol import SavedArtifact
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_server import HistoryRecord, HistoryStore, PathRedactor, Principal, create_app
from dinkster_server.history import record_from_job
from dinkster_server.queue import Job, JobKey, JobQueue
from dinkster_values import (
    EncodedPayload,
    TypeRegistry,
    Value,
    ValueMeta,
    make_list_value,
    make_tuple_value,
    register_core_types,
)
from dinkster_workers import InProcessWorker

from tests.async_scenarios import run_scenario

STRING = TypeExpr.concrete("STRING")

SOURCE_DIGEST = "blake3:" + "ab" * 32


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


class Boom(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.boom",
            inputs=(InputSpec("tag", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )

    @classmethod
    async def execute(cls, *, tag: str) -> Mapping[str, object]:
        raise ValueError(f"boom: {tag}")


NODES = (Echo, Boom)
SCHEMAS = build_schemas(NODES)


def make_engine(
    on_event: EventListener | None = None, *, plan_execution: PlanExecution | None = None
) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    return Engine(
        schemas=SCHEMAS,
        registry=registry,
        worker=InProcessWorker(build_node_types(NODES), registry),
        cache=MemoryLRUCache(),
        on_event=on_event,
        plan_execution=plan_execution,
    )


def make_record(
    run_id: str,
    *,
    scope: str = "local",
    finished_at: float | None = None,
    **overrides: object,
) -> HistoryRecord:
    fields: dict[str, object] = {
        "run_id": run_id,
        "scope": scope,
        "client_id": "c1",
        "job_id": f"job-{run_id}",
        "state": "completed",
        "priority": 0,
        "submitted_at": 1.0,
        "started_at": 2.0,
        "finished_at": finished_at if finished_at is not None else time.time(),
        "source_document": "",
        "error": None,
        "executed": 1,
        "cached": 0,
        "skipped": 0,
    }
    fields.update(overrides)
    return HistoryRecord(**fields)  # type: ignore[arg-type]


# -- HistoryStore -------------------------------------------------------------


def test_store_put_get_roundtrip_and_wire(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.sqlite")
    record = make_record(
        "run1",
        principal_id="worker",
        principal_kind="agent",
        source_document=SOURCE_DIGEST,
        error={"kind": "execution", "message": "x"},
        state="failed",
        node_receipts=(
            {"nodeId": "load", "disposition": "executed", "executionArm": "native"},
            {"nodeId": "sample", "disposition": "cached", "executionArm": "comfyui"},
        ),
    )
    store.put(record)
    assert store.get("local", "run1") == record
    wire = record.to_wire()
    assert wire["runId"] == "run1"
    assert wire["jobRef"] == "run1"  # canonical name, same identity as runId
    assert wire["attemptId"] == 1
    assert wire["principalId"] == "worker"
    assert wire["principalKind"] == "agent"
    assert wire["sourceDocument"] == SOURCE_DIGEST
    assert wire["error"] == {"kind": "execution", "message": "x"}
    assert wire["startedAt"] == 2.0
    assert wire["nodeReceipts"] == [
        {"nodeId": "load", "disposition": "executed", "executionArm": "native"},
        {"nodeId": "sample", "disposition": "cached", "executionArm": "comfyui"},
    ]
    # Optional facts are omitted-when-absent, never null.
    bare = make_record("run2", started_at=None)
    store.put(bare)
    bare_wire = bare.to_wire()
    for absent in ("startedAt", "sourceDocument", "error", "thumbnail"):
        assert absent not in bare_wire
    # Idempotent by run_id: re-recording leaves the immutable row unchanged.
    store.put(record)
    assert len(store.query("local")) == 2
    store.put(make_record("run1", state="completed"))
    assert store.get("local", "run1") == record
    store.close()


def test_store_filters_and_keyset_paging(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.sqlite")
    for index in range(5):
        store.put(
            make_record(
                f"run{index}",
                finished_at=100.0 + index,
                client_id="c1" if index % 2 == 0 else "c2",
                state="completed" if index != 3 else "failed",
                source_document=SOURCE_DIGEST if index == 4 else "",
            )
        )
    # Exact-match filters.
    assert [r.run_id for r in store.query("local", client_id="c2")] == [
        "run3",
        "run1",
    ]
    assert [r.run_id for r in store.query("local", state="failed")] == ["run3"]
    assert [r.run_id for r in store.query("local", source_document=SOURCE_DIGEST)] == ["run4"]
    # Newest-finished first; keyset paging never repeats or skips.
    page1 = store.query("local", limit=3)
    assert [r.run_id for r in page1] == ["run4", "run3", "run2"]
    last = page1[-1]
    page2 = store.query("local", limit=3, after=(last.finished_at, last.run_id))
    assert [r.run_id for r in page2] == ["run1", "run0"]
    store.close()


def test_history_asset_roots_cover_accepted_terminal_and_other_scopes(tmp_path: Path) -> None:
    path = tmp_path / "history.sqlite"
    other = "blake3:" + "cd" * 32
    store = HistoryStore(path)
    store.put(make_record("finished", scope="private", source_document=SOURCE_DIGEST))
    store.job_accepted(make_job("queued", scope="another", source_document=other))
    store.put(make_record("no-document"))
    store.close()
    store = HistoryStore(path)
    try:
        with store.hold_asset_references() as roots:
            assert roots == {SOURCE_DIGEST, other}
            with sqlite3.connect(path, timeout=0) as connection:
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    connection.execute("BEGIN IMMEDIATE")
        store.recover_interrupted()
        with pytest.raises(AssetError, match="unavailable"):
            with store.hold_asset_references():
                pytest.fail("an interrupted run was assumed to have no output references")
        assert store.delete("another", "queued")
        with store.hold_asset_references() as roots:
            assert roots == {SOURCE_DIGEST}
        with sqlite3.connect(path) as connection:
            connection.execute(
                "UPDATE history SET source_document = 'broken' WHERE run_id = 'finished'"
            )
        with pytest.raises(AssetError):
            with store.hold_asset_references():
                pytest.fail("unreadable history roots were ignored")
    finally:
        store.close()


def test_terminal_history_retains_nested_target_and_saved_asset_identities(tmp_path: Path) -> None:
    path = tmp_path / "history.sqlite"
    media = AssetRef(digest_bytes(b"source clip"), "clip", 11)
    asset = AssetRef(digest_bytes(b"typed asset"), "data", 11)
    saved_digest = digest_bytes(b"saved output")
    preview_digest = digest_bytes(b"saved output preview")
    decoy = digest_bytes(b"payload fingerprint, not an external identity")
    leaf = Value(
        "VIDEO",
        decoy,
        ValueMeta({"asset_refs": [media.to_wire()]}),
        EncodedPayload("VIDEO", b"opaque payload", None),
    )
    typed = Value(
        "asset<file>",
        asset.digest,
        ValueMeta(asset.to_wire()),
        EncodedPayload("asset<file>", b"opaque identity", None),
    )
    nested = make_tuple_value([make_list_value("VIDEO", [leaf, leaf]), typed])
    store = HistoryStore(path)
    try:
        for run, scope, value in [("one", "private", nested), ("two", "other", leaf)]:
            job = make_job(run, scope=scope, source_document=SOURCE_DIGEST)
            store.job_accepted(job)
            job.result = RunResult(
                run,
                {"target": {"out": value}},
                (),
                ("target",),
                (),
                artifacts=(
                    SavedArtifact(
                        "save",
                        saved_digest,
                        "kept.txt",
                        12,
                        "text/plain",
                        "out/kept.txt",
                        preview=SavedArtifact(
                            "save",
                            preview_digest,
                            "preview.mp4",
                            20,
                            "video/mp4",
                            f"previews/{preview_digest[7:]}/preview.mp4",
                        ),
                    ),
                ),
            )
            job.state = "completed"
            store.job_terminal(job)
        store.close()
        store = HistoryStore(path)
        record = store.get("private", "one")
        assert record is not None
        assert record.asset_digests == tuple(
            sorted((media.digest, asset.digest, saved_digest, preview_digest))
        )
        assert "assetDigests" not in record.to_wire()
        with store.hold_asset_references() as roots:
            assert roots == {
                SOURCE_DIGEST,
                media.digest,
                asset.digest,
                saved_digest,
                preview_digest,
            }
        assert store.delete("private", "one")
        with store.hold_asset_references() as roots:
            assert roots == {SOURCE_DIGEST, media.digest, saved_digest, preview_digest}
        assert store.delete_where("other") == 1
        with store.hold_asset_references() as roots:
            assert roots == set()
        store.put(make_record("corrupt"))
        with sqlite3.connect(path) as connection:
            connection.execute("UPDATE history SET asset_digests = '[\"broken\"]'")
        with pytest.raises(AssetError):
            with store.hold_asset_references():
                pytest.fail("corrupt output identities were assumed empty")
    finally:
        store.close()


@pytest.mark.parametrize("helper_backed", [False, True])
@pytest.mark.parametrize("media", ["image", "video", "preview", "output", "text", "none"])
def test_finished_history_lists_thumbnail_after_reopen(
    tmp_path: Path, helper_backed: bool, media: str
) -> None:
    image = AssetRef(digest_bytes(b"image"), "result.png", 5, "image/png", "out/result.png")
    video = AssetRef(digest_bytes(b"video"), "result.mp4", 5, "video/mp4", "out/result.mp4")
    text = AssetRef(digest_bytes(b"text"), "result.txt", 4, "text/plain", "out/result.txt")
    original = {"image": image, "video": video}.get(media, text)
    preview = image if media == "preview" else None
    expected = (
        image
        if media in ("preview", "output")
        else original
        if media in ("image", "video")
        else None
    )
    job = make_job("thumbnail", source_document=SOURCE_DIGEST)
    if media != "none":
        outputs = {}
        if media == "output":
            outputs = {
                "target": {
                    "out": Value(
                        "asset<image>",
                        image.digest,
                        ValueMeta(image.to_wire()),
                        EncodedPayload("asset<image>", b"opaque", None),
                    )
                }
            }
        job.result = RunResult(
            job.run_id,
            outputs,
            (),
            ("save",),
            (),
            artifacts=(
                SavedArtifact(
                    "save",
                    original.digest,
                    original.name,
                    original.size,
                    original.media_type,
                    original.virtual_path,
                    preview=SavedArtifact(
                        "save",
                        image.digest,
                        image.name,
                        image.size,
                        image.media_type,
                        image.virtual_path,
                    )
                    if preview is not None
                    else None,
                ),
            ),
        )
    with ExitStack() as stack:
        helper = stack.enter_context(LibraryClient(tmp_path / "library")) if helper_backed else None
        store = HistoryStore(
            helper or tmp_path / "history.sqlite", accepted_path=tmp_path / "accepted.sqlite"
        )
        store.job_accepted(job)
        job.state, job.finished_at = "completed", 23.0
        store.job_terminal(job)
        store.close()
    with ExitStack() as stack:
        helper = stack.enter_context(LibraryClient(tmp_path / "library")) if helper_backed else None
        store = HistoryStore(
            helper or tmp_path / "history.sqlite", accepted_path=tmp_path / "accepted.sqlite"
        )
        record = store.get("local", job.run_id)
        assert record is not None
        assert record.thumbnail == expected
        assert store.query("local") == [record]
        assert store.get("private", job.run_id) is None
        assert store.query("private") == []
        if expected is not None:
            assert expected.digest in (record.asset_digests or ())

        async def scenario() -> None:
            client = TestClient(TestServer(create_app(make_engine, SCHEMAS, history=store)))
            await client.start_server()
            try:
                listed = await (await client.get("/api/history?scope=local")).json()
                detail = await (await client.get(f"/api/history/{job.run_id}?scope=local")).json()
                for wire in (listed["records"][0], detail):
                    if expected is None:
                        assert "thumbnail" not in wire
                    else:
                        assert wire["thumbnail"] == expected.to_wire()
            finally:
                await client.close()

        run_scenario(scenario())


# -- HTTP surface + recording -------------------------------------------------


async def make_client(tmp_path: Path) -> tuple[TestClient, HistoryStore]:
    store = HistoryStore(tmp_path / "history.sqlite")
    app = create_app(make_engine, SCHEMAS, history=store)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client, store


def submit_body(graph: Graph, targets: list[str], **overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "clientId": "c1",
        "jobId": "j1",
        "graph": graph_to_wire(graph),
        "targets": targets,
    }
    body.update(overrides)
    return body


async def wait_for_history(
    client: TestClient, *, count: int, timeout: float = 5.0
) -> list[dict[str, object]]:
    """The history write is fire-and-forget off the job event: poll the
    endpoint until the expected rows land."""
    async with asyncio.timeout(timeout):
        while True:
            resp = await client.get("/api/history", params={"scope": "local"})
            assert resp.status == 200
            records = (await resp.json())["records"]
            if len(records) >= count:
                return records
            await asyncio.sleep(0.01)


def test_attention_diagnostic_survives_execution_cache_and_durable_history(tmp_path: Path) -> None:
    async def scenario() -> None:
        diagnostic = f"missing capability evidence at {tmp_path}; using default auto route"
        expected = "missing capability evidence at <library>; using default auto route"

        async def plan(*args: object) -> ExecutionSelection:
            return ExecutionSelection(
                target="utility",
                cache_tag="utility@1",
                attention_diagnostic=diagnostic,
            )

        path = tmp_path / "history.sqlite"
        store = HistoryStore(path)
        app = create_app(
            partial(make_engine, plan_execution=plan),
            SCHEMAS,
            history=store,
            redactor=PathRedactor([("<library>", tmp_path)]),
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        runs: list[str] = []
        try:
            graph = Graph(nodes={"e": GraphNode("test.echo", {"text": "hi"})})
            for index, disposition in enumerate(("executed", "cached"), start=1):
                response = await client.post(
                    "/api/jobs", json=submit_body(graph, ["e"], jobId=f"j{index}")
                )
                assert response.status == 202
                run_id = (await response.json())["jobRef"]
                runs.append(run_id)
                records = await wait_for_history(client, count=index)
                record = next(row for row in records if row["jobRef"] == run_id)
                assert record["state"] == "completed"
                assert record["nodeReceipts"] == [
                    {
                        "nodeId": "e",
                        "disposition": disposition,
                        "executionArm": "native",
                        "attentionDiagnostic": expected,
                    }
                ]
                replay = await (
                    await client.get(f"/api/jobs/by-ref/{run_id}/events?after=0")
                ).json()
                lifecycle = [
                    event
                    for event in replay["events"]
                    if event["type"] in ("node_started", "node_finished", "node_cached")
                ]
                assert lifecycle
                assert all(
                    event["detail"]["attentionDiagnostic"] == expected for event in lifecycle
                )
        finally:
            await client.close()
        reopened = HistoryStore(path)
        try:
            for run_id in runs:
                record = reopened.get("local", run_id)
                assert record is not None
                assert record.node_receipts[0]["attentionDiagnostic"] == expected
        finally:
            reopened.close()

    run_scenario(scenario())


def test_terminal_jobs_are_recorded_with_source_document(tmp_path: Path) -> None:
    """Every terminal job lands durably: completed runs with their summary
    counts and sourceDocument link, failures with their error payload."""

    async def scenario() -> None:
        client, _store = await make_client(tmp_path)
        try:
            graph = Graph(nodes={"e": GraphNode("test.echo", {"text": "hi"})})
            body = submit_body(graph, ["e"], sourceDocument=SOURCE_DIGEST)
            assert (await client.post("/api/jobs", json=body)).status == 202

            boom = Graph(nodes={"b": GraphNode("test.boom", {"tag": "t"})})
            body = submit_body(boom, ["b"], jobId="j2")
            assert (await client.post("/api/jobs", json=body)).status == 202

            records = await wait_for_history(client, count=2)
            by_job = {r["jobId"]: r for r in records}
            done = by_job["j1"]
            assert done["state"] == "completed"
            assert done["scope"] == "local"
            assert done["principalId"] == "local"
            assert done["principalKind"] == "human"
            assert done["clientId"] == "c1"
            assert done["sourceDocument"] == SOURCE_DIGEST
            assert done["executed"] == 1
            assert done["nodeReceipts"] == [
                {"nodeId": "e", "disposition": "executed", "executionArm": "native"}
            ]
            assert done["attemptId"] == 1
            assert "error" not in done
            finished, started, submitted = (
                cast(float, done["finishedAt"]),
                cast(float, done["startedAt"]),
                cast(float, done["submittedAt"]),
            )
            assert finished >= started >= submitted

            failed = by_job["j2"]
            assert failed["state"] == "failed"
            assert "sourceDocument" not in failed
            error = failed["error"]
            assert isinstance(error, dict)
            assert error["kind"] == "execution"

            # The stable per-run identity fetches one durable record.
            run_id = done["runId"]
            resp = await client.get(f"/api/history/{run_id}", params={"scope": "local"})
            assert resp.status == 200
            assert (await resp.json())["jobId"] == "j1"
            # Wrong scope is a 404, not a hint.
            resp = await client.get(f"/api/history/{run_id}", params={"scope": "other"})
            assert resp.status == 404
        finally:
            await client.close()

    run_scenario(scenario())


def test_history_endpoint_query_contract(tmp_path: Path) -> None:
    """Query-first paging: scope required, cursor bound to its query,
    tampering is a loud 400."""

    async def scenario() -> None:
        client, store = await make_client(tmp_path)
        try:
            for index in range(3):
                store.put(make_record(f"run{index}", finished_at=100.0 + index))

            # Scope is structurally required.
            assert (await client.get("/api/history")).status == 400

            resp = await client.get("/api/history", params={"scope": "local", "limit": "2"})
            page = await resp.json()
            assert [r["runId"] for r in page["records"]] == ["run2", "run1"]
            cursor = page["cursor"]

            resp = await client.get("/api/history", params={"scope": "local", "cursor": cursor})
            page2 = await resp.json()
            assert [r["runId"] for r in page2["records"]] == ["run0"]
            assert "cursor" not in page2

            # The cursor binds its query: a different filter is a 400.
            resp = await client.get(
                "/api/history",
                params={"scope": "local", "clientId": "c9", "cursor": cursor},
            )
            assert resp.status == 400
            # Tampered cursors are a 400, never a guess.
            resp = await client.get("/api/history", params={"scope": "local", "cursor": "Zm9v"})
            assert resp.status == 400

            # Filters are exact matches.
            resp = await client.get("/api/history", params={"scope": "local", "state": "cancelled"})
            assert (await resp.json())["records"] == []
        finally:
            await client.close()

    run_scenario(scenario())


def test_history_delete_endpoints(tmp_path: Path) -> None:
    """DELETE mirrors GET's scope discipline: single delete 404s on miss or
    wrong scope, bulk clear takes the list's filters plus before= and
    reports the count, and scope is always required."""

    async def scenario() -> None:
        client, store = await make_client(tmp_path)
        try:
            for index in range(4):
                store.put(
                    make_record(
                        f"run{index}",
                        finished_at=100.0 + index,
                        state="completed" if index % 2 == 0 else "failed",
                        source_document=SOURCE_DIGEST if index == 0 else "",
                    )
                )

            # Scope is structurally required on both DELETE shapes.
            assert (await client.delete("/api/history/run0")).status == 400
            assert (await client.delete("/api/history")).status == 400

            # Single delete: wrong scope is a plain 404, right scope is 204,
            # and the second attempt is a 404 (already gone).
            resp = await client.delete("/api/history/run0", params={"scope": "other"})
            assert resp.status == 404
            resp = await client.delete("/api/history/run0", params={"scope": "local"})
            assert resp.status == 204
            resp = await client.delete("/api/history/run0", params={"scope": "local"})
            assert resp.status == 404

            # Bulk clear with a filter reports how many rows went.
            resp = await client.delete("/api/history", params={"scope": "local", "state": "failed"})
            assert await resp.json() == {"deleted": 2}
            # before= must be a timestamp.
            resp = await client.delete(
                "/api/history", params={"scope": "local", "before": "yesterday"}
            )
            assert resp.status == 400
            resp = await client.delete("/api/history", params={"scope": "local", "before": "150.0"})
            assert await resp.json() == {"deleted": 1}

            resp = await client.get("/api/history", params={"scope": "local"})
            assert (await resp.json())["records"] == []
        finally:
            await client.close()

    run_scenario(scenario())


def test_history_auth_scope_default_ambiguity_and_unknown_scope_posture(
    tmp_path: Path,
) -> None:
    class Authenticator:
        principals = {
            "single": Principal("single", {"a": frozenset({"history:read"})}),
            "multi": Principal(
                "multi",
                {
                    "a": frozenset({"history:read"}),
                    "b": frozenset({"history:read"}),
                },
            ),
        }

        async def authenticate(self, token: str) -> Principal | None:
            return self.principals.get(token)

    async def scenario() -> None:
        store = HistoryStore(tmp_path / "auth-history.sqlite")
        store.put(make_record("run-a", scope="a"))
        store.put(make_record("run-b", scope="b"))
        store.put(make_record("run-c", scope="c"))
        app = create_app(make_engine, SCHEMAS, history=store, authenticator=Authenticator())
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            single = {"Authorization": "Bearer single"}
            records = await (await client.get("/api/history", headers=single)).json()
            assert [record["runId"] for record in records["records"]] == ["run-a"]

            multi = {"Authorization": "Bearer multi"}
            ambiguous = await client.get("/api/history", headers=multi)
            assert ambiguous.status == 400
            assert (await ambiguous.json())["error"] == "scope-required"
            hidden_list = await client.get("/api/history?scope=c", headers=multi)
            assert hidden_list.status == 200
            assert await hidden_list.json() == {"records": []}
            hidden_get = await client.get("/api/history/run-c?scope=c", headers=multi)
            assert hidden_get.status == 404
            assert await hidden_get.json() == {"error": "no such run"}
        finally:
            await client.close()

    run_scenario(scenario())


def test_shutdown_drains_history_writes(tmp_path: Path) -> None:
    """Cleanup order: queue close emits terminal transitions, their rows
    land, THEN the store closes - a job cancelled by shutdown is still in
    the durable record, and graceful shutdown leaves nothing for the
    interrupted sweep."""

    async def scenario() -> None:
        client, _store = await make_client(tmp_path)
        graph = Graph(nodes={"e": GraphNode("test.echo", {"text": "hi"})})
        body = submit_body(graph, ["e"], sourceDocument=SOURCE_DIGEST)
        assert (await client.post("/api/jobs", json=body)).status == 202
        await wait_for_history(client, count=1)
        # A second job held queued by a paused queue: shutdown cancels it,
        # and that cancellation must land durably (not linger as accepted).
        assert (await client.post("/api/queue/pause")).status == 200
        assert (
            await client.post("/api/jobs", json=submit_body(graph, ["e"], jobId="j2"))
        ).status == 202
        await client.close()  # runs on_cleanup: queue close + store close

        reopened = HistoryStore(tmp_path / "history.sqlite")
        records = {r.job_id: r for r in reopened.query("local")}
        assert records["j1"].state == "completed"
        assert records["j1"].source_document == SOURCE_DIGEST
        assert records["j2"].state == "cancelled"
        assert reopened.recover_interrupted() == ()
        assert read_accepted(tmp_path / "history.sqlite") == []
        reopened.close()

    run_scenario(scenario())


# -- durable queue: accepted table + interrupted recovery ---------------------


def make_job(run_id: str, **overrides: object) -> Job:
    fields: dict[str, object] = {
        "key": JobKey("c1", f"job-{run_id}"),
        "graph": Graph(nodes={}),
        "targets": (),
        "priority": 0,
        "run_id": run_id,
        "fingerprint": f"fp-{run_id}",
    }
    fields.update(overrides)
    return Job(**fields)  # type: ignore[arg-type]


def read_accepted(path: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM accepted ORDER BY submitted").fetchall()
    finally:
        conn.close()


def test_recover_interrupted_distinguishes_queued_from_running(tmp_path: Path) -> None:
    """A dead process's leftovers become interrupted history - never-started
    and was-running distinguishable, nothing re-executed, sweep idempotent."""
    db = tmp_path / "history.sqlite"
    store = HistoryStore(db)
    execution = make_engine().pin_execution()
    queued = make_job("r-queued")
    started = make_job("r-started", execution=execution)
    store.job_accepted(queued)
    store.job_accepted(started)
    started.state = "running"
    started.started_at = 9.0
    store.job_started(started)
    store.close()  # simulated crash: no terminal writes ever happen

    reopened = HistoryStore(db)
    recovered = reopened.recover_interrupted()
    assert {r.run_id for r in recovered} == {"r-queued", "r-started"}
    by_run = {r.run_id: r for r in recovered}
    q = by_run["r-queued"]
    assert q.state == "interrupted"
    assert q.started_at is None
    assert q.error is not None
    assert q.error["phase"] == "queued"
    assert "before this job started" in q.error["message"]
    assert "resubmit" in q.error["message"]
    assert q.error["fingerprint"] == "fp-r-queued"
    s = by_run["r-started"]
    assert s.state == "interrupted"
    assert s.started_at == 9.0
    assert s.error is not None
    assert s.error["phase"] == "running"
    assert "while this job was running" in s.error["message"]
    assert s.error["extensionSnapshotDigest"] == execution.extension_snapshot_digest
    assert "extensionSnapshotDigest" not in q.error  # no pinned execution recorded
    # Served through the existing query surface, no new API needed.
    listed = reopened.query("local", state="interrupted")
    assert {r.run_id for r in listed} == {"r-queued", "r-started"}
    # Idempotent: the sweep consumed the table.
    assert reopened.recover_interrupted() == ()
    assert read_accepted(db) == []
    reopened.close()


def test_helper_history_keeps_engine_work_until_publication_and_never_resurrects(
    tmp_path: Path,
) -> None:
    library = tmp_path / "library"
    first_path, second_path = [tmp_path / name / "accepted.sqlite" for name in ("one", "two")]
    with LibraryClient(library) as first, LibraryClient(library) as second:
        with pytest.raises(ValueError, match="engine-owned"):
            HistoryStore(first)
        store = HistoryStore(first, accepted_path=first_path)
        peer = HistoryStore(second, accepted_path=second_path)
        job = make_job("completed", source_document=SOURCE_DIGEST)
        store.job_accepted(job)
        peer.job_accepted(make_job("peer-running"))
        job.state, job.finished_at = "completed", 23.0
        store.job_terminal(job)
        original = peer.get("local", job.run_id)
        assert original == record_from_job(job)
        assert read_accepted(first_path) == []
        job.finished_at = 29.0
        store.job_terminal(job)
        assert peer.get("local", job.run_id) == original
        assert store.delete("private", job.run_id) is False
        assert peer.delete("local", job.run_id)
        store.job_accepted(job)
        store.job_terminal(job)
        assert peer.get("local", job.run_id) is None
        with pytest.raises(AssetError, match="helper-owned"):
            with store.hold_asset_references():
                pass
        for identifier, finished, scope in (
            ("a", 31.0, "local"),
            ("z", 31.0, "local"),
            ("private", 99.0, "private"),
        ):
            store.put(make_record(identifier, scope=scope, finished_at=finished))
        assert [row.run_id for row in peer.query("local", limit=1)] == ["z"]
        assert [row.run_id for row in peer.query("local", after=(31.0, "z"))] == ["a"]
        assert peer.delete_where("local", before=31.0) == 0
        assert peer.delete_where("local", client_id="c1", state="completed", before=32.0) == 2
        assert store.get("private", "private") is not None
        rich = make_record(
            "rich",
            priority=-3,
            attempt=2,
            principal_id="agent",
            principal_kind="agent",
            source_document=SOURCE_DIGEST,
            error={"kind": "execution", "message": "failed"},
            state="failed",
            executed=4,
            cached=2,
            skipped=1,
            node_receipts=({"nodeId": "sample", "disposition": "cached"},),
            asset_digests=(SOURCE_DIGEST,),
        )
        raw = dict(rich.to_wire(), assetDigests=[SOURCE_DIGEST], futureField={"keep": True})
        first.call("history.append", scope="local", record=raw)
        assert peer.get("local", "rich") == rich
        assert peer.query(
            "local", client_id="c1", source_document=SOURCE_DIGEST, state="failed"
        ) == [rich]
        assert peer.query("local", client_id="other", state="failed") == []
        peer.put(make_record("rich", finished_at=999.0))
        assert first.call("history.get", scope="local", id="rich")["record"] == raw
        lost = make_job("lost-reply")
        store.job_accepted(lost)
        lost.state, lost.finished_at = "completed", 37.0
        terminal = record_from_job(lost)
        document = dict(terminal.to_wire(), assetDigests=[])
        assert first.call("history.append", scope="local", record=document)["created"]
        unpublished = make_job("not-published")
        store.job_accepted(unpublished)
        unpublished.state, unpublished.finished_at = "completed", 41.0
        first.close()
        for pending in (lost, unpublished):
            with pytest.raises(LibraryError):
                store.job_terminal(pending)
        assert {row["run_id"] for row in read_accepted(first_path)} == {
            lost.run_id,
            unpublished.run_id,
        }
        assert [row["run_id"] for row in read_accepted(second_path)] == ["peer-running"]
        store.close()
        with LibraryClient(library) as replacement:
            store = HistoryStore(replacement, accepted_path=first_path)
            recovered = store.recover_interrupted()
            assert [(row.run_id, row.state, row.asset_digests) for row in recovered] == [
                (unpublished.run_id, "interrupted", None),
            ]
            assert store.recover_interrupted() == ()
            assert store.get("local", lost.run_id) == terminal
            assert read_accepted(first_path) == []
            assert [row["run_id"] for row in read_accepted(second_path)] == ["peer-running"]
            store.close()
            assert (
                replacement.call("history.get", scope="local", id=lost.run_id)["record"] == document
            )
        peer.close()
        assert second.call("history.get", scope="local", id="completed") == {"record": None}
    assert not (library / "history.sqlite").exists()
    assert not (library / "library.sqlite").exists()


@pytest.mark.parametrize("failure", ["disconnected", "migration"])
def test_helper_history_http_paging_peer_edits_and_refusals(tmp_path: Path, failure: str) -> None:
    async def scenario() -> None:
        root = tmp_path / "library"
        with LibraryClient(root) as helper, LibraryClient(root) as peer:
            store = HistoryStore(helper, accepted_path=tmp_path / "engine" / "accepted.sqlite")
            for number in range(203):
                store.put(make_record(f"run-{number:03}", finished_at=10.0 + number // 2))
            app = create_app(make_engine, SCHEMAS, history=store)
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                page = await client.get("/api/history", params={"scope": "local", "limit": "200"})
                assert page.status == 200
                body = await page.json()
                assert [row["runId"] for row in body["records"]] == [
                    f"run-{number:03}" for number in range(202, 2, -1)
                ]
                page = await client.get(
                    "/api/history",
                    params={"scope": "local", "limit": "200", "cursor": body["cursor"]},
                )
                assert page.status == 200
                assert [row["runId"] for row in (await page.json())["records"]] == [
                    "run-002",
                    "run-001",
                    "run-000",
                ]
                mismatch = await client.get(
                    "/api/history", params={"scope": "private", "cursor": body["cursor"]}
                )
                assert mismatch.status == 400
                assert (
                    await client.get("/api/history/run-202", params={"scope": "private"})
                ).status == 404
                peer.call("history.delete", scope="local", id="run-202")
                assert (
                    await client.get("/api/history/run-202", params={"scope": "local"})
                ).status == 404
                deleted = await client.delete("/api/history/run-201", params={"scope": "local"})
                assert deleted.status == 204
                cleared = await client.delete(
                    "/api/history", params={"scope": "local", "before": "11"}
                )
                assert cleared.status == 200 and await cleared.json() == {"deleted": 2}
                if failure == "disconnected":
                    helper.close()
                else:
                    (root / "history.sqlite").write_bytes(b"legacy history")
                for method, path in (
                    (client.get, "/api/history"),
                    (client.get, "/api/history/run-200"),
                    (client.delete, "/api/history/run-200"),
                    (client.delete, "/api/history"),
                ):
                    refused = await method(path, params={"scope": "local"})
                    assert refused.status == 503
                    assert await refused.json() == {"error": "history store unavailable"}
                if failure == "migration":
                    assert (root / "history.sqlite").read_bytes() == b"legacy history"
            finally:
                await client.close()
            if failure == "migration":
                assert helper.call("projects.list")["items"]
            assert peer.call("projects.list")["items"]

    run_scenario(scenario())


def test_engine_accepted_state_isolated_with_shared_finished_history(tmp_path: Path) -> None:
    history = tmp_path / "history.sqlite"
    first_path = tmp_path / "first" / "accepted.sqlite"
    second_path = tmp_path / "second" / "accepted.sqlite"
    first = HistoryStore(history, accepted_path=first_path)
    second = HistoryStore(history, accepted_path=second_path)
    other_digest = "blake3:" + "cd" * 32
    first.job_accepted(make_job("first-queued", source_document=SOURCE_DIGEST))
    running = make_job("second-running", source_document=other_digest)
    second.job_accepted(running)
    running.started_at = 19.0
    second.job_started(running)
    finished = make_record("already-finished", source_document=SOURCE_DIGEST)
    first.job_accepted(make_job(finished.run_id))
    first.put(finished)
    try:
        for store in (first, second):
            with pytest.raises(AssetError, match="requires helper-owned collection"):
                with store.hold_asset_references():
                    pytest.fail("one engine's pending work cannot certify library-wide roots")
        recovered = first.recover_interrupted()
        assert [record.run_id for record in recovered] == ["first-queued"]
        assert first.get("local", "second-running") is None
        assert first.get("local", finished.run_id) == finished
        assert first.recover_interrupted() == ()
    finally:
        first.close()
        second.close()

    first = HistoryStore(history, accepted_path=first_path)
    second = HistoryStore(history, accepted_path=second_path)
    try:
        assert first.recover_interrupted() == ()
        recovered = second.recover_interrupted()
        assert [record.run_id for record in recovered] == ["second-running"]
        assert recovered[0].started_at == 19.0
        assert recovered[0].error is not None
        assert recovered[0].error["phase"] == "running"
        assert first.get("local", "second-running") == recovered[0]
        assert second.recover_interrupted() == ()
        assert first.get("local", finished.run_id) == finished
        completed = make_job("completed", source_document=other_digest)
        second.job_accepted(completed)
        completed.state = "completed"
        completed.finished_at = 23.0
        second.job_terminal(completed)
        original = first.get("local", completed.run_id)
        assert original is not None and original.state == "completed"
        completed.finished_at = 29.0
        second.job_terminal(completed)
        assert first.get("local", completed.run_id) == original
        assert second.recover_interrupted() == ()
    finally:
        first.close()
        second.close()


def test_recover_interrupted_never_rewrites_existing_history(tmp_path: Path) -> None:
    db = tmp_path / "history.sqlite"
    store = HistoryStore(db)
    store.put(make_record("r1", state="completed"))
    # A leftover accepted row pointing at an already-recorded run must not
    # overwrite the immutable terminal row.
    store.job_accepted(make_job("r1"))
    store.close()

    reopened = HistoryStore(db)
    assert reopened.recover_interrupted() == ()
    record = reopened.get("local", "r1")
    assert record is not None
    assert record.state == "completed"
    assert read_accepted(db) == []
    reopened.close()


@pytest.mark.parametrize("helper_backed", [False, True])
def test_restart_surfaces_interrupted_history_and_empty_queue(
    tmp_path: Path, helper_backed: bool
) -> None:
    from contextlib import ExitStack

    from dinkster_assets import AssetVault, LibraryStore
    from dinkster_assets.produced import ProducedAssetInventory
    from dinkster_library import EngineLifetime
    from dinkster_server import ServerLibrary
    from dinkster_server.cleanup import CleanupPolicy, EngineReferences

    async def scenario() -> None:
        if helper_backed:
            with LibraryClient(tmp_path / "library") as previous:
                previous.call("references.engine.attach", engine="41" * 16)
                LibraryStore(previous).publish_terminal_result(
                    "local",
                    "r-lost",
                    1,
                    [],
                    state="completed",
                    identity={"clientId": "c1", "jobId": "job-r-lost"},
                    artifacts_complete=True,
                )
        with ExitStack() as stack:
            helper = (
                stack.enter_context(LibraryClient(tmp_path / "library")) if helper_backed else None
            )
            lifetime = (
                stack.enter_context(EngineLifetime(tmp_path / "library", "41" * 16))
                if helper is not None
                else None
            )
            db = tmp_path / "state" / "history.sqlite"
            source = helper if helper is not None else db
            accepted = tmp_path / "state" / "accepted.sqlite" if helper is not None else None
            store = HistoryStore(source, accepted_path=accepted)
            lost = make_job("r-lost")
            store.job_accepted(lost)
            lost.state = "running"
            lost.started_at = time.time()
            store.job_started(lost)
            store.close()
            reopened = HistoryStore(source, accepted_path=accepted)

            def bind_library(engine: Engine) -> ServerLibrary:
                assert helper is not None and lifetime is not None
                vault = AssetVault(tmp_path / "vault")
                ProducedAssetInventory.initialize_vault(vault.root, existing_empty=True)
                scratch = tmp_path / "scratch"
                return ServerLibrary(
                    vault,
                    LibraryStore(helper),
                    cleanup_policy=CleanupPolicy(helper),
                    engine_references=EngineReferences(
                        helper,
                        "41" * 16,
                        engine.cache,
                        reopened,
                        engine.registry,
                        scratch,
                        vaults=(vault,),
                        lifetime=lifetime,
                    ),
                    scratch_root=scratch,
                )

            app = create_app(
                make_engine,
                SCHEMAS,
                history=reopened,
                make_library=bind_library if helper is not None else None,
            )
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                if helper is not None:
                    recovered = reopened.get("local", "r-lost")
                    assert recovered is not None and recovered.asset_digests == ()
                    response = await client.post("/api/library/storage/clear", json={})
                    assert response.status == 200, await response.text()
                    assert (await response.json())["collectionEnabled"]
                resp = await client.get("/api/history", params={"scope": "local"})
                assert resp.status == 200
                records = (await resp.json())["records"]
                assert len(records) == 1
                assert records[0]["state"] == "interrupted"
                error = records[0]["error"]
                assert isinstance(error, dict)
                assert error["kind"] == "interrupted"
                assert error["phase"] == "running"
                assert "resubmit" in error["message"]
                status = await (await client.get("/api/queue")).json()
                assert status["queued"] == []
                assert status["running"] == []
                graph = Graph(nodes={"e": GraphNode("test.echo", {"text": "after restart"})})
                response = await client.post("/api/jobs", json=submit_body(graph, ["e"]))
                assert response.status == 202, await response.text()
                await wait_for_history(client, count=2)
                records = (await (await client.get("/api/history?scope=local")).json())["records"]
                assert {record["state"] for record in records} == {"interrupted", "completed"}
            finally:
                await client.close()

    run_scenario(scenario())


def test_failed_accepted_write_refuses_submission(tmp_path: Path) -> None:
    """Durability before acknowledgment: when the accepted-job write fails,
    the submission raises and the queue holds no trace of the job."""

    class ExplodingStore:
        def job_accepted(self, job: Job) -> None:
            raise sqlite3.OperationalError("disk gone")

        def job_started(self, job: Job) -> None:
            pass

        def job_terminal(self, job: Job) -> None:
            pass

    async def scenario() -> None:
        queue = JobQueue(make_engine(), store=ExplodingStore())
        graph = Graph(nodes={"e": GraphNode("test.echo", {"text": "hi"})})
        with pytest.raises(sqlite3.OperationalError):
            queue.submit("c1", "j1", graph, ["e"])
        assert queue.jobs() == []
        assert queue.get("c1", "j1") is None
        await queue.close()

    run_scenario(scenario())


def test_failed_accepted_write_leaves_terminal_predecessor_intact(tmp_path: Path) -> None:
    """A refused resubmission must not retire the terminal job already
    holding the key: the predecessor stays fully queryable."""

    class FlakyStore:
        def __init__(self) -> None:
            self.accepts = 0

        def job_accepted(self, job: Job) -> None:
            self.accepts += 1
            if self.accepts > 1:
                raise sqlite3.OperationalError("disk gone")

        def job_started(self, job: Job) -> None:
            pass

        def job_terminal(self, job: Job) -> None:
            pass

    async def scenario() -> None:
        queue = JobQueue(make_engine(), store=FlakyStore())
        queue.start()
        graph = Graph(nodes={"e": GraphNode("test.echo", {"text": "hi"})})
        first = queue.submit("c1", "j1", graph, ["e"])
        async with asyncio.timeout(5):
            while first.state != "completed":
                await asyncio.sleep(0.005)
        with pytest.raises(sqlite3.OperationalError):
            queue.submit("c1", "j1", graph, ["e"], priority=1)  # new content, write fails
        assert queue.get("c1", "j1") is first
        assert queue.job_for_run(first.run_id) is first
        await queue.close()

    run_scenario(scenario())
