"""Execution-run journal: durable replay copy of the published job event
stream. Covers wire-event classification, batched flushing, post-run
pruning (last progress per node kept), whole-stream retention, and the
GET /api/runs/{runId}/journal replay page."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EventListener
from dinkster_graph import Graph, GraphNode, graph_to_wire
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
    report_log,
    report_progress,
)
from dinkster_server import ExecutionJournal, JournalStore, create_app
from dinkster_server.execution_journal import execution_run_stream
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

STRING = TypeExpr.concrete("core.string")


# -- wire-event fixtures (shapes mirror app.py's published dicts) ------------


def lifecycle(run_id: str, kind: str, seq: int, node_id: str = "n") -> dict[str, object]:
    return {
        "type": kind,
        "runId": run_id,
        "nodeId": node_id,
        "clientId": "c",
        "jobId": "j",
        "jobRef": run_id,
        "seq": seq,
    }


def log_event(
    run_id: str, seq: int, *, level: str, message: str = "m", node_id: str = "n"
) -> dict[str, object]:
    return {
        "type": "node_event",
        "event": "log",
        "runId": run_id,
        "nodeId": node_id,
        "data": {"level": level, "message": message, "ts": 1.0},
        "clientId": "c",
        "jobId": "j",
        "jobRef": run_id,
        "seq": seq,
    }


def progress_event(run_id: str, seq: int, *, node_id: str, step: int) -> dict[str, object]:
    return {
        "type": "node_event",
        "event": "progress",
        "runId": run_id,
        "nodeId": node_id,
        "data": {"step": step, "total": 10},
        "clientId": "c",
        "jobId": "j",
        "jobRef": run_id,
        "seq": seq,
    }


def job_state(run_id: str, seq: int, state: str) -> dict[str, object]:
    return {
        "type": "job_state",
        "clientId": "c",
        "jobId": "j",
        "state": state,
        "jobRef": run_id,
        "runId": run_id,
        "seq": seq,
    }


@pytest.fixture
def store(tmp_path: Path) -> JournalStore:
    return JournalStore(tmp_path / "execution.sqlite")


def flush(journal: ExecutionJournal) -> None:
    asyncio.run(journal.flush())


# -- classification and batching ---------------------------------------------


def test_flush_persists_classified_wire_events(store: JournalStore) -> None:
    journal = ExecutionJournal(store)
    run = "run-1"
    journal.observe(lifecycle(run, "run_started", 1), scope="local")
    journal.observe(lifecycle(run, "node_started", 2), scope="local")
    journal.observe(log_event(run, 3, level="warning"), scope="local")
    journal.observe(log_event(run, 4, level="info"), scope="local")
    journal.observe(progress_event(run, 5, node_id="n", step=1), scope="local")
    # Transient chatter the journal does not keep:
    journal.observe(
        {"type": "node_event", "event": "preview", "runId": run, "jobRef": run, "seq": 6},
        scope="local",
    )
    journal.observe({"type": "memory_status", "seq": 7}, scope="local")
    blob_event = dict(log_event(run, 8, level="warning"))
    blob_event["_blob"] = b"x"
    journal.observe(blob_event, scope="local")
    flush(journal)

    page = journal.read(run)
    assert [record.name for record in page.records] == [
        "run_started",
        "node_started",
        "log",
        "log",
        "progress",
    ]
    assert [record.durable for record in page.records] == [True, True, True, False, False]
    assert [record.payload["seq"] for record in page.records] == [1, 2, 3, 4, 5]
    assert page.records[2].payload == log_event(run, 3, level="warning")
    assert journal.scope_of(run) == "local"
    assert journal.scope_of("other") is None


def test_events_without_a_run_reference_are_ignored(store: JournalStore) -> None:
    journal = ExecutionJournal(store)
    journal.observe({"type": "node_started", "seq": 1}, scope="local")
    flush(journal)
    assert store.latest_seq(execution_run_stream("")) == 0


def test_finalize_prunes_telemetry_keeping_last_progress_per_node(
    store: JournalStore,
) -> None:
    journal = ExecutionJournal(store)
    run = "run-1"
    journal.observe(lifecycle(run, "node_started", 1, node_id="a"), scope="local")
    journal.observe(progress_event(run, 2, node_id="a", step=1), scope="local")
    journal.observe(progress_event(run, 3, node_id="a", step=2), scope="local")
    journal.observe(log_event(run, 4, level="info"), scope="local")
    journal.observe(log_event(run, 5, level="warning"), scope="local")
    journal.observe(progress_event(run, 6, node_id="b", step=7), scope="local")
    journal.observe(job_state(run, 7, "completed"), scope="local")
    flush(journal)

    page = journal.read(run)
    names = [record.name for record in page.records]
    assert names == ["node_started", "log", "job_state", "progress", "progress"]
    assert all(record.durable for record in page.records)
    keepers = {
        cast("str", record.payload["nodeId"]): record.payload
        for record in page.records
        if record.name == "progress"
    }
    assert keepers["a"] == progress_event(run, 3, node_id="a", step=2)
    assert keepers["b"] == progress_event(run, 6, node_id="b", step=7)
    assert page.coalesced_below > 0


def test_runs_beyond_keep_runs_expire_oldest_first(store: JournalStore) -> None:
    journal = ExecutionJournal(store, keep_runs=1)
    for run in ("run-1", "run-2"):
        journal.observe(lifecycle(run, "node_started", 1), scope="local")
        journal.observe(job_state(run, 2, "completed"), scope="local")
        flush(journal)
    assert journal.scope_of("run-1") is None
    assert journal.read("run-1").latest_seq == 0
    assert journal.scope_of("run-2") == "local"
    assert journal.read("run-2").latest_seq > 0


def test_runs_older_than_keep_days_expire(store: JournalStore) -> None:
    journal = ExecutionJournal(store, keep_days=1.0)
    old_finished = 1000.0  # epoch: far older than any cutoff
    store.transact(
        lambda txn: (
            txn.execute(
                "INSERT INTO execution_runs (run_id, scope, started, finished)"
                " VALUES ('run-old', 'local', ?, ?)",
                (old_finished, old_finished),
            ),
            txn.append(
                execution_run_stream("run-old"),
                "node_started",
                {"seq": 1},
                durable=True,
                timestamp=old_finished,
            ),
        )
    )
    journal.observe(lifecycle("run-new", "node_started", 1), scope="local")
    journal.observe(job_state("run-new", 2, "completed"), scope="local")
    flush(journal)
    assert journal.scope_of("run-old") is None
    assert journal.read("run-old").latest_seq == 0
    assert journal.scope_of("run-new") == "local"


def test_live_runs_never_expire(store: JournalStore) -> None:
    journal = ExecutionJournal(store, keep_runs=0)
    journal.observe(lifecycle("run-live", "node_started", 1), scope="local")
    journal.observe(lifecycle("run-done", "node_started", 1), scope="local")
    journal.observe(job_state("run-done", 2, "completed"), scope="local")
    flush(journal)
    # keep_runs=0: every finished run expires at its own finalize; the
    # still-executing run is untouched.
    assert journal.scope_of("run-done") is None
    assert journal.scope_of("run-live") == "local"
    assert journal.read("run-live").latest_seq == 1


def test_oversized_record_is_dropped_without_losing_the_batch(
    store: JournalStore,
) -> None:
    journal = ExecutionJournal(store)
    run = "run-1"
    journal.observe(lifecycle(run, "node_started", 1), scope="local")
    journal.observe(log_event(run, 2, level="warning", message="x" * 300_000), scope="local")
    journal.observe(log_event(run, 3, level="warning"), scope="local")
    flush(journal)
    page = journal.read(run)
    assert [record.payload["seq"] for record in page.records] == [1, 3]


def test_buffer_cap_sheds_only_coalescible_records(
    store: JournalStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_server import execution_journal as module

    monkeypatch.setattr(module, "MAX_PENDING_RECORDS", 2)
    journal = ExecutionJournal(store)
    run = "run-1"
    journal.observe(log_event(run, 1, level="info"), scope="local")
    journal.observe(log_event(run, 2, level="info"), scope="local")
    journal.observe(log_event(run, 3, level="info"), scope="local")  # shed
    journal.observe(log_event(run, 4, level="warning"), scope="local")  # kept
    flush(journal)
    page = journal.read(run)
    assert [record.payload["seq"] for record in page.records] == [1, 2, 4]


def test_read_scoped_misses_are_indistinguishable(store: JournalStore) -> None:
    journal = ExecutionJournal(store)
    journal.observe(lifecycle("run-1", "node_started", 1), scope="alice")
    flush(journal)
    page = journal.read_scoped("run-1", "alice")
    assert page is not None
    assert [record.name for record in page.records] == ["node_started"]
    assert journal.read_scoped("run-1", "bob") is None
    assert journal.read_scoped("run-2", "alice") is None


def test_close_wakes_the_flush_loop_and_lands_the_buffer(tmp_path: Path) -> None:
    path = tmp_path / "execution.sqlite"

    async def scenario() -> None:
        journal = ExecutionJournal(JournalStore(path), flush_interval=60.0)
        journal.start()
        await asyncio.sleep(0)  # let the loop enter its wait
        journal.observe(lifecycle("run-1", "node_started", 1), scope="local")
        journal.observe(job_state("run-1", 2, "completed"), scope="local")
        async with asyncio.timeout(5):
            await journal.close()

    asyncio.run(scenario())
    reopened = ExecutionJournal(JournalStore(path))
    names = [record.name for record in reopened.read("run-1").records]
    assert names == ["node_started", "job_state"]
    asyncio.run(reopened.close())


def test_close_waits_for_an_in_flight_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "execution.sqlite"

    async def scenario() -> None:
        journal = ExecutionJournal(JournalStore(path), flush_interval=0.01)
        entered = threading.Event()
        release = threading.Event()
        original = journal._write

        def slow_write(batch: list[Any]) -> None:
            entered.set()
            assert release.wait(5)
            original(batch)

        monkeypatch.setattr(journal, "_write", slow_write)
        journal.start()
        journal.observe(lifecycle("run-1", "node_started", 1), scope="local")
        await asyncio.to_thread(entered.wait, 5)
        closer = asyncio.ensure_future(journal.close())
        await asyncio.sleep(0.05)
        # close() must wait for the write, not abandon it to race the
        # store's close.
        assert not closer.done()
        release.set()
        async with asyncio.timeout(5):
            await closer

    asyncio.run(scenario())
    reopened = ExecutionJournal(JournalStore(path))
    assert reopened.scope_of("run-1") == "local"
    asyncio.run(reopened.close())


def test_crash_abandoned_runs_become_retention_eligible(tmp_path: Path) -> None:
    path = tmp_path / "execution.sqlite"
    asyncio.run(ExecutionJournal(JournalStore(path)).close())  # create tables

    def crashed(txn: Any, run_id: str, ts: float) -> None:
        txn.execute(
            "INSERT INTO execution_runs (run_id, scope, started) VALUES (?, 'local', ?)",
            (run_id, ts),
        )
        txn.append(
            execution_run_stream(run_id),
            "node_started",
            {"seq": 1},
            durable=True,
            timestamp=ts + 5.0,
        )

    now = time.time()
    raw = JournalStore(path)
    raw.transact(lambda txn: crashed(txn, "run-old", 1000.0))
    raw.transact(lambda txn: crashed(txn, "run-recent", now))
    raw.close()

    # Reopening stamps both abandoned runs finished at their last
    # persisted event; the ancient one then expires immediately.
    journal = ExecutionJournal(JournalStore(path), keep_days=1.0)
    assert journal.scope_of("run-old") is None
    assert journal.read("run-old").latest_seq == 0
    assert journal.scope_of("run-recent") == "local"
    asyncio.run(journal.close())

    raw = JournalStore(path)
    row = raw.transact(
        lambda txn: txn.execute(
            "SELECT finished FROM execution_runs WHERE run_id = 'run-recent'"
        ).fetchone()
    )
    assert row is not None
    assert row["finished"] == pytest.approx(now + 5.0)
    raw.close()


def test_journal_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "execution.sqlite"
    journal = ExecutionJournal(JournalStore(path))
    journal.observe(lifecycle("run-1", "node_started", 1), scope="local")
    journal.observe(job_state("run-1", 2, "completed"), scope="local")
    asyncio.run(journal.close())

    reopened = ExecutionJournal(JournalStore(path))
    assert reopened.scope_of("run-1") == "local"
    page = reopened.read("run-1")
    assert [record.name for record in page.records] == ["node_started", "job_state"]
    asyncio.run(reopened.close())


def test_observe_after_close_is_a_noop(tmp_path: Path) -> None:
    journal = ExecutionJournal(JournalStore(tmp_path / "execution.sqlite"))
    asyncio.run(journal.close())
    journal.observe(lifecycle("run-1", "node_started", 1), scope="local")  # no crash


def test_retention_bounds_must_be_non_negative(store: JournalStore) -> None:
    with pytest.raises(ValueError, match="keep_runs"):
        ExecutionJournal(store, keep_runs=-1)
    with pytest.raises(ValueError, match="keep_days"):
        ExecutionJournal(store, keep_days=-1.0)


# -- end to end over HTTP -----------------------------------------------------


class Chatty(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="journal.chatty",
            inputs=(InputSpec("text", STRING),),
            outputs=(OutputSpec("out", STRING),),
            idempotent=False,  # never cached: events must fire every run
        )

    @classmethod
    async def execute(cls, *, text: str) -> Mapping[str, object]:
        report_progress(1, 2, text="working")
        report_progress(2, 2, text="working")
        report_log("info", f"info:{text}")
        report_log("warning", f"warning:{text}")
        return cls.outputs(out=text)


CHATTY_SCHEMAS = build_schemas([Chatty])


def make_chatty_engine(on_event: EventListener | None = None) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    return Engine(
        schemas=CHATTY_SCHEMAS,
        registry=registry,
        worker=InProcessWorker(build_node_types([Chatty]), registry),
        cache=MemoryLRUCache(),
        on_event=on_event,
    )


def test_run_journal_endpoint_serves_replay_page(tmp_path: Path) -> None:
    async def scenario() -> None:
        journal = ExecutionJournal(JournalStore(tmp_path / "execution.sqlite"), flush_interval=0.01)
        app = create_app(make_chatty_engine, CHATTY_SCHEMAS, execution_journal=journal)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            graph = Graph(nodes={"n": GraphNode("journal.chatty", {"text": "hi"})})
            resp = await client.post(
                "/api/jobs",
                json={
                    "clientId": "c1",
                    "jobId": "j1",
                    "graph": graph_to_wire(graph),
                    "targets": ["n"],
                },
            )
            assert resp.status == 202, await resp.text()
            run_id = cast("str", (await resp.json())["jobRef"])

            async with asyncio.timeout(5):
                while True:
                    status = await (await client.get("/api/jobs/c1/j1")).json()
                    if status["state"] == "completed":
                        break
                    await asyncio.sleep(0.01)
                # The journal flushes on its own tick; poll until the
                # terminal transition landed.
                while True:
                    resp = await client.get(f"/api/runs/{run_id}/journal?scope=local")
                    body = await resp.json()
                    if resp.status == 200 and any(
                        record["name"] == "job_state" and record["payload"]["state"] == "completed"
                        for record in body["records"]
                    ):
                        break
                    await asyncio.sleep(0.02)

            records = cast("list[dict[str, Any]]", body["records"])
            names = [record["name"] for record in records]
            # Post-run: lifecycle, warning log, and job transitions are
            # durable; info logs are pruned; the last progress per node is
            # re-appended durable.
            assert "run_started" in names
            assert "node_started" in names
            assert "node_finished" in names
            warnings = [
                record
                for record in records
                if record["name"] == "log" and record["payload"]["data"]["level"] == "warning"
            ]
            assert len(warnings) == 1
            assert warnings[0]["payload"]["data"]["message"] == "warning:hi"
            assert not any(
                record["name"] == "log" and record["payload"]["data"]["level"] == "info"
                for record in records
            )
            progress = [record for record in records if record["name"] == "progress"]
            assert len(progress) == 1
            assert progress[0]["payload"]["data"]["step"] == 2
            assert all(record["durable"] for record in records)
            # Replayed payloads carry the live per-job seq, strictly rising
            # in journal order except the re-appended progress keeper.
            seqs = [record["payload"]["seq"] for record in records if record["name"] != "progress"]
            assert seqs == sorted(seqs)
            assert body["latestSeq"] == records[-1]["seq"]
            assert body["coalescedBelow"] > 0

            # Paging: after=<mid> returns only the tail.
            mid = records[1]["seq"]
            resp = await client.get(f"/api/runs/{run_id}/journal?scope=local&after={mid}")
            tail = await resp.json()
            assert [record["seq"] for record in tail["records"]] == [
                record["seq"] for record in records if record["seq"] > mid
            ]

            # Scope handling mirrors history: required, and a wrong scope
            # is indistinguishable from a miss.
            assert (await client.get(f"/api/runs/{run_id}/journal")).status == 400
            assert (await client.get(f"/api/runs/{run_id}/journal?scope=other")).status == 404
            assert (await client.get("/api/runs/no-such-run/journal?scope=local")).status == 404
            assert (
                await client.get(f"/api/runs/{run_id}/journal?scope=local&after=-1")
            ).status == 400
            assert (
                await client.get(f"/api/runs/{run_id}/journal?scope=local&after=x")
            ).status == 400
        finally:
            await client.close()

    asyncio.run(scenario())
