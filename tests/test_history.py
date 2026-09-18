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
from functools import partial
from pathlib import Path
from typing import cast

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EventListener, ExecutionSelection, PlanExecution
from dinkster_graph import Graph, GraphNode, graph_to_wire
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
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

STRING = TypeExpr.concrete("core.string")

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
    for absent in ("startedAt", "sourceDocument", "error"):
        assert absent not in bare_wire
    # Idempotent by run_id: re-recording leaves the immutable row unchanged.
    store.put(record)
    assert len(store.query("local")) == 2
    store.put(make_record("run1", state="completed"))
    assert store.get("local", "run1") == record
    store.close()


def test_store_scope_isolation_and_validation(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.sqlite")
    store.put(make_record("mine"))
    store.put(make_record("theirs", scope="other"))
    # Wrong scope is a plain miss, never a hint the run exists elsewhere.
    assert store.get("other", "mine") is None
    assert [r.run_id for r in store.query("local")] == ["mine"]
    assert [r.run_id for r in store.query("other")] == ["theirs"]
    with pytest.raises(ValueError):
        store.put(make_record("x", scope=" "))
    store.close()


def test_store_migrates_pre_principal_database_with_local_backfill(tmp_path: Path) -> None:
    path = tmp_path / "history.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE history (
        run_id TEXT PRIMARY KEY, scope TEXT NOT NULL, client_id TEXT NOT NULL,
        job_id TEXT NOT NULL, state TEXT NOT NULL, priority INTEGER NOT NULL,
        submitted REAL NOT NULL, started REAL, finished REAL NOT NULL,
        source_document TEXT NOT NULL, error TEXT, executed INTEGER NOT NULL,
        cached INTEGER NOT NULL, skipped INTEGER NOT NULL, attempt INTEGER NOT NULL DEFAULT 1
        )"""
    )
    conn.execute(
        "INSERT INTO history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("old", "local", "c", "j", "completed", 0, 1.0, 2.0, 3.0, "", None, 1, 0, 0, 1),
    )
    conn.commit()
    conn.close()

    store = HistoryStore(path)
    record = store.get("local", "old")
    assert record is not None
    assert record.principal_id == "local"
    assert record.principal_kind == "human"
    assert record.to_wire()["principalId"] == "local"
    assert record.to_wire()["principalKind"] == "human"
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


def test_store_delete_and_delete_where(tmp_path: Path) -> None:
    """Deletion is scoped like everything else, and bulk clear speaks the
    same filter vocabulary as query - plus before= for retention pruning.
    Asset bytes are never in play; only rows go."""
    store = HistoryStore(tmp_path / "history.sqlite")
    for index in range(4):
        store.put(
            make_record(
                f"run{index}",
                finished_at=100.0 + index,
                state="completed" if index % 2 == 0 else "failed",
            )
        )
    store.put(make_record("other-run", scope="other"))

    # Single delete: scoped, idempotent-in-effect (second time is a miss).
    assert store.delete("other", "run0") is False  # wrong scope: plain miss
    assert store.delete("local", "run0") is True
    assert store.delete("local", "run0") is False
    assert store.get("local", "run0") is None

    # Filtered bulk clear.
    assert store.delete_where("local", state="failed") == 2
    assert [r.run_id for r in store.query("local")] == ["run2"]
    # before= prunes strictly-earlier-finished rows.
    store.put(make_record("recent", finished_at=200.0))
    assert store.delete_where("local", before=150.0) == 1
    assert [r.run_id for r in store.query("local")] == ["recent"]
    # No filters clears the scope entirely - and only that scope.
    assert store.delete_where("local") == 1
    assert store.query("local") == []
    assert [r.run_id for r in store.query("other")] == ["other-run"]
    with pytest.raises(ValueError):
        store.delete_where(" ")
    store.close()


def test_store_survives_reopen(tmp_path: Path) -> None:
    """The point of persistence: rows outlive the process (store)."""
    store = HistoryStore(tmp_path / "history.sqlite")
    store.put(make_record("durable", source_document=SOURCE_DIGEST))
    store.close()
    reopened = HistoryStore(tmp_path / "history.sqlite")
    record = reopened.get("local", "durable")
    assert record is not None
    assert record.source_document == SOURCE_DIGEST
    reopened.close()


def test_store_migrates_old_schema_with_attempt_default(tmp_path: Path) -> None:
    path = tmp_path / "history.sqlite"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE history (
            run_id TEXT PRIMARY KEY, scope TEXT NOT NULL, client_id TEXT NOT NULL,
            job_id TEXT NOT NULL, state TEXT NOT NULL, priority INTEGER NOT NULL,
            submitted REAL NOT NULL, started REAL, finished REAL NOT NULL,
            source_document TEXT NOT NULL, error TEXT, executed INTEGER NOT NULL,
            cached INTEGER NOT NULL, skipped INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        "INSERT INTO history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("old", "local", "c", "j", "completed", 0, 1.0, 2.0, 3.0, "", None, 1, 0, 0),
    )
    connection.commit()
    connection.close()

    store = HistoryStore(path)
    old = store.get("local", "old")
    assert old is not None
    assert old.attempt == 1
    assert old.node_receipts == ()
    assert old.to_wire()["attemptId"] == 1
    new = make_record("new", attempt=1)
    store.put(new)
    assert store.get("local", "new") == new
    store.close()


def test_record_from_job_requires_terminal() -> None:
    job = Job(
        key=JobKey("c1", "j1"),
        graph=Graph(nodes={}),
        targets=(),
        priority=0,
        run_id="r1",
        fingerprint="test",
    )
    with pytest.raises(ValueError):
        record_from_job(job)


def test_record_from_job_marks_started_nodes_interrupted() -> None:
    job = Job(
        key=JobKey("c1", "j1"),
        graph=Graph(nodes={}),
        targets=(),
        priority=0,
        run_id="r1",
        fingerprint="test",
        principal_id="worker",
        principal_kind="agent",
        state="cancelled",
        finished_at=3.0,
        node_receipts={
            "sample": {
                "nodeId": "sample",
                "disposition": "running",
                "executionArm": "comfyui",
                "provider": "dinkster-vision-example",
                "pack": "dinkster-vision-example",
                "worker": "gpu-box",
            }
        },
    )
    record = record_from_job(job)
    assert record.principal_kind == "agent"
    assert record.to_wire()["principalKind"] == "agent"
    assert record.node_receipts == (
        {
            "nodeId": "sample",
            "disposition": "interrupted",
            "executionArm": "comfyui",
            "provider": "dinkster-vision-example",
            "pack": "dinkster-vision-example",
            "worker": "gpu-box",
        },
    )


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

    asyncio.run(scenario())


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

    asyncio.run(scenario())


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

    asyncio.run(scenario())


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

    asyncio.run(scenario())


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

    asyncio.run(scenario())


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

    asyncio.run(scenario())


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


def test_store_accepted_row_lifecycle(tmp_path: Path) -> None:
    """job_accepted writes the row, job_started marks it running, and
    job_terminal atomically trades it for the terminal history row."""
    db = tmp_path / "history.sqlite"
    store = HistoryStore(db)
    job = make_job("r1", scope="local", principal_id="worker", principal_kind="agent")
    store.job_accepted(job)
    (row,) = read_accepted(db)
    assert row["run_id"] == "r1"
    assert row["state"] == "queued"
    assert row["started"] is None
    assert row["client_id"] == "c1"
    assert row["job_id"] == "job-r1"
    assert row["principal"] == "worker"
    assert row["principal_kind"] == "agent"
    assert row["fingerprint"] == "fp-r1"
    assert row["execution_identity"] == ""
    job.state = "running"
    job.started_at = 5.0
    store.job_started(job)
    (row,) = read_accepted(db)
    assert row["state"] == "running"
    assert row["started"] == 5.0
    job.state = "completed"
    job.finished_at = 6.0
    store.job_terminal(job)
    assert read_accepted(db) == []
    record = store.get("local", "r1")
    assert record is not None
    assert record.state == "completed"
    store.close()


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


def test_job_terminal_move_is_atomic(tmp_path: Path) -> None:
    """The terminal history insert and the accepted-row delete share one
    transaction: a failure rolls both back, leaving the conservative
    accepted row for the next restart's sweep."""
    db = tmp_path / "history.sqlite"
    store = HistoryStore(db)
    job = make_job("r1")
    store.job_accepted(job)
    # A failing trigger on the delete forces the whole transaction to abort.
    saboteur = sqlite3.connect(str(db))
    saboteur.execute(
        "CREATE TRIGGER boom BEFORE DELETE ON accepted BEGIN SELECT RAISE(ABORT, 'boom'); END"
    )
    saboteur.commit()
    saboteur.close()
    job.state = "completed"
    job.finished_at = 1.0
    with pytest.raises(sqlite3.DatabaseError):
        store.job_terminal(job)
    assert store.get("local", "r1") is None  # history insert rolled back too
    (row,) = read_accepted(db)
    assert row["run_id"] == "r1"
    store.close()


def test_accepted_row_exists_before_submit_acknowledgment(tmp_path: Path) -> None:
    """End to end: a paused queue holds the job, but the accepted row is
    already on disk when POST /api/jobs returns; terminal completion
    retires it."""

    async def scenario() -> None:
        client, _store = await make_client(tmp_path)
        db = tmp_path / "history.sqlite"
        try:
            assert (await client.post("/api/queue/pause")).status == 200
            graph = Graph(nodes={"e": GraphNode("test.echo", {"text": "hi"})})
            assert (await client.post("/api/jobs", json=submit_body(graph, ["e"]))).status == 202
            (row,) = read_accepted(db)
            assert row["state"] == "queued"
            assert row["client_id"] == "c1"
            assert row["job_id"] == "j1"
            assert row["fingerprint"]
            assert row["execution_identity"]
            assert (await client.post("/api/queue/resume")).status == 200
            records = await wait_for_history(client, count=1)
            assert records[0]["state"] == "completed"
            assert read_accepted(db) == []
        finally:
            await client.close()

    asyncio.run(scenario())


def test_restart_surfaces_interrupted_history_and_empty_queue(tmp_path: Path) -> None:
    """A server started over a crashed predecessor's store serves the lost
    jobs as interrupted history and starts with an empty queue - zero
    automatic re-execution."""
    db = tmp_path / "history.sqlite"
    store = HistoryStore(db)
    lost = make_job("r-lost")
    store.job_accepted(lost)
    lost.state = "running"
    lost.started_at = time.time()
    store.job_started(lost)
    store.close()  # simulated crash

    async def scenario() -> None:
        reopened = HistoryStore(db)
        app = create_app(make_engine, SCHEMAS, history=reopened)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
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
        finally:
            await client.close()

    asyncio.run(scenario())


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

    asyncio.run(scenario())


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

    asyncio.run(scenario())
