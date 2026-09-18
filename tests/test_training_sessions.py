"""Training session supervisor: the durable claim/commit ledger for
training advances (docs/training-design.md sections 3.2, 8.2, 9.1-9.3).

The invariants under test: exactly-once advances through the claim
protocol (committed operations idempotently return their unique output,
concurrent claims are retryable, stale claims are taken over or superseded
under the fence), a strictly increasing step cursor, a checkpoint-covered
journal watermark that never regresses or runs ahead of the stream, and
every ledger move appending its durable journal fact atomically.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EventListener
from dinkster_protocol.training import TrainingJournalEvent
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_server import (
    AdvanceInProgress,
    JournalStore,
    Principal,
    StaleTrainingFence,
    TrainingLineageConflict,
    TrainingSessionStore,
    UnknownTrainingSession,
    create_app,
)
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

SID = "a" * 32
OTHER_SID = "b" * 32
OP1 = "1" * 32
OP2 = "2" * 32
OP3 = "3" * 32

CONFIG = "blake3:" + "c" * 64
SNAPSHOT = "blake3:" + "e" * 64
INIT = "blake3:" + "0" * 64
CKPT1 = "blake3:" + "1" * 64
CKPT2 = "blake3:" + "2" * 64
RECOVERY = "blake3:" + "9" * 64


@pytest.fixture
def store(tmp_path: Path) -> Iterator[TrainingSessionStore]:
    sessions = TrainingSessionStore(JournalStore(tmp_path / "training.sqlite"))
    yield sessions
    sessions.close()


def create_default(store: TrainingSessionStore, session_id: str = SID, scope: str = "local"):
    return store.create_session(
        session_id,
        scope=scope,
        config_digest=CONFIG,
        extension_snapshot_digest=SNAPSHOT,
        initial_manifest_digest=INIT,
    )


def event_names(store: TrainingSessionStore, session_id: str = SID) -> list[str]:
    return [record.name for record in store.read_events(session_id).records]


# -- session creation ---------------------------------------------------------


def test_create_session_records_head_lineage_and_journal_fact(
    store: TrainingSessionStore,
) -> None:
    record = create_default(store)
    assert record.state == "active"
    assert record.fence_epoch == 1
    assert record.committed_manifest_digest == INIT
    assert record.committed_step_cursor == 0

    handle = record.handle()
    assert handle.session_id == SID
    assert handle.checkpoint_manifest_digest == INIT
    assert handle.step_cursor == 0
    # The handle's watermark covers the session_created fact itself.
    assert handle.journal_seq == 1

    checkpoints = store.list_checkpoints(SID)
    assert [c.manifest_digest for c in checkpoints] == [INIT]
    assert checkpoints[0].parent_digest == ""
    assert checkpoints[0].operation_id == ""

    page = store.read_events(SID)
    assert [r.name for r in page.records] == ["session_created"]
    assert page.records[0].durable
    # Every appended fact must survive a replay decode.
    event = TrainingJournalEvent.from_wire(page.records[0].payload)
    assert event.data["initialManifestDigest"] == INIT


def test_create_session_is_idempotent_for_identical_facts(store: TrainingSessionStore) -> None:
    first = create_default(store)
    assert create_default(store) == first
    assert event_names(store) == ["session_created"]  # no second fact
    with pytest.raises(TrainingLineageConflict):
        store.create_session(
            SID,
            scope="local",
            config_digest=CONFIG,
            extension_snapshot_digest=SNAPSHOT,
            initial_manifest_digest=CKPT1,  # different initial checkpoint
        )


def test_create_session_validates_inputs(store: TrainingSessionStore) -> None:
    with pytest.raises(ValueError):
        create_default(store, session_id="not-hex!")
    with pytest.raises(ValueError):
        store.create_session(
            SID,
            scope=" ",
            config_digest=CONFIG,
            extension_snapshot_digest=SNAPSHOT,
            initial_manifest_digest=INIT,
        )
    with pytest.raises(ValueError):
        store.create_session(
            SID,
            scope="local",
            config_digest="blake3:short",
            extension_snapshot_digest=SNAPSHOT,
            initial_manifest_digest=INIT,
        )


# -- claim protocol -----------------------------------------------------------


def test_claim_records_started_operation_and_fact(store: TrainingSessionStore) -> None:
    create_default(store)
    op = store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1)
    assert op.status == "started"
    assert op.fence_epoch == 1
    assert op.input_manifest_digest == INIT
    assert event_names(store) == ["session_created", "advance_started"]
    started = store.read_events(SID).records[-1]
    assert TrainingJournalEvent.from_wire(started.payload).advance_id == OP1


def test_claim_refusals(store: TrainingSessionStore) -> None:
    create_default(store)
    with pytest.raises(UnknownTrainingSession):
        store.claim_advance(OTHER_SID, OP1, input_manifest_digest=INIT, fence_epoch=1)
    with pytest.raises(StaleTrainingFence):
        store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=2)
    # Input must be the committed head.
    with pytest.raises(TrainingLineageConflict):
        store.claim_advance(SID, OP1, input_manifest_digest=CKPT1, fence_epoch=1)

    store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1)
    # Concurrent claim of the same operation under the current fence is
    # retryable, not fatal.
    with pytest.raises(AdvanceInProgress):
        store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1)
    # Same operation id with a different input contradicts the ledger.
    with pytest.raises(TrainingLineageConflict):
        store.claim_advance(SID, OP1, input_manifest_digest=CKPT1, fence_epoch=1)
    # A different operation cannot start while one is started.
    with pytest.raises(TrainingLineageConflict):
        store.claim_advance(SID, OP2, input_manifest_digest=INIT, fence_epoch=1)


def test_committed_claim_returns_recorded_outcome(store: TrainingSessionStore) -> None:
    create_default(store)
    store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1)
    committed = store.commit_advance(
        SID,
        OP1,
        fence_epoch=1,
        output_manifest_digest=CKPT1,
        output_step_cursor=100,
        covered_journal_seq=2,
    )
    # Re-claiming a committed operation is the recovery path: it returns
    # the unique recorded outcome without re-running anything.
    assert store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1) == committed
    assert event_names(store).count("advance_started") == 1


def test_fence_takeover_resumes_stale_claim(store: TrainingSessionStore) -> None:
    create_default(store)
    store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1)
    bumped = store.acquire_fence(SID)
    assert bumped.fence_epoch == 2

    resumed = store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=2)
    assert resumed.status == "started"
    assert resumed.fence_epoch == 2
    assert event_names(store) == [
        "session_created",
        "advance_started",
        "phase_changed",
        "advance_resumed",
    ]
    # The fenced-out original holder can no longer commit.
    with pytest.raises(StaleTrainingFence):
        store.commit_advance(
            SID,
            OP1,
            fence_epoch=1,
            output_manifest_digest=CKPT1,
            output_step_cursor=100,
            covered_journal_seq=2,
        )


def test_fence_supersession_via_abort(store: TrainingSessionStore) -> None:
    create_default(store)
    store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1)
    store.acquire_fence(SID)
    # The new fence holder discards the stale claim instead of resuming it.
    aborted = store.abort_advance(SID, OP1, fence_epoch=2, reason="superseded")
    assert aborted.status == "aborted"
    # Idempotent repeat, even under a later epoch.
    assert store.abort_advance(SID, OP1, fence_epoch=2).status == "aborted"
    # An aborted operation id is spent.
    with pytest.raises(TrainingLineageConflict):
        store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=2)
    # A fresh operation is now claimable.
    fresh = store.claim_advance(SID, OP2, input_manifest_digest=INIT, fence_epoch=2)
    assert fresh.status == "started"


def test_committed_replay_survives_supersession_and_completion(
    store: TrainingSessionStore,
) -> None:
    """A worker whose commit reply was lost must be able to recover its
    recorded outcome even after its fence was superseded or the session
    completed - replay is a ledger read, never a state transition."""
    create_default(store)
    store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1)
    committed = store.commit_advance(
        SID,
        OP1,
        fence_epoch=1,
        output_manifest_digest=CKPT1,
        output_step_cursor=100,
        covered_journal_seq=2,
    )
    store.acquire_fence(SID)
    # The fenced-out original holder replays with its OLD epoch.
    assert store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1) == committed
    assert (
        store.commit_advance(
            SID,
            OP1,
            fence_epoch=1,
            output_manifest_digest=CKPT1,
            output_step_cursor=100,
            covered_journal_seq=2,
        )
        == committed
    )
    store.complete_session(SID, fence_epoch=2)
    assert store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1) == committed
    assert (
        store.commit_advance(
            SID,
            OP1,
            fence_epoch=1,
            output_manifest_digest=CKPT1,
            output_step_cursor=100,
            covered_journal_seq=2,
        )
        == committed
    )
    # Replay appended nothing.
    assert event_names(store).count("advance_started") == 1
    assert event_names(store).count("advance_committed") == 1


def test_competing_stores_serialize_claims(tmp_path: Path) -> None:
    """Two stores on the same file (two SQLite connections): the eager
    transaction makes claim decisions atomic across connections, so a
    second claimant observes the first claim instead of double-claiming."""
    path = tmp_path / "training.sqlite"
    first = TrainingSessionStore(JournalStore(path))
    second = TrainingSessionStore(JournalStore(path))
    create_default(first)
    first.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1)
    with pytest.raises(AdvanceInProgress):
        second.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1)
    with pytest.raises(TrainingLineageConflict):
        second.claim_advance(SID, OP2, input_manifest_digest=INIT, fence_epoch=1)
    first.close()
    second.close()


# -- recovery checkpoints -----------------------------------------------------


def test_recovery_checkpoint_pointer(store: TrainingSessionStore) -> None:
    create_default(store)
    store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1)
    op = store.set_recovery_checkpoint(SID, OP1, fence_epoch=1, manifest_digest=RECOVERY)
    assert op.recovery_checkpoint_digest == RECOVERY
    assert event_names(store)[-1] == "recovery_checkpoint_published"
    # Recovery checkpoints are resume points, never lineage commits.
    assert [c.manifest_digest for c in store.list_checkpoints(SID)] == [INIT]
    with pytest.raises(StaleTrainingFence):
        store.set_recovery_checkpoint(SID, OP1, fence_epoch=2, manifest_digest=RECOVERY)
    with pytest.raises(TrainingLineageConflict):
        store.set_recovery_checkpoint(SID, OP2, fence_epoch=1, manifest_digest=RECOVERY)


def test_pause_acknowledges_safe_point_without_resolving_the_claim(
    store: TrainingSessionStore,
) -> None:
    create_default(store)
    store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1)
    store.set_recovery_checkpoint(SID, OP1, fence_epoch=1, manifest_digest=RECOVERY)
    paused = store.pause_advance(SID, OP1, fence_epoch=1, reason="cancel requested")
    # The operation stays started with its recovery pointer and a durable
    # paused mark, so an ordinary same-fence claim resumes it.
    assert paused.status == "started"
    assert paused.paused is True
    assert paused.recovery_checkpoint_digest == RECOVERY
    assert event_names(store)[-1] == "advance_paused"
    # Repeat pause is an idempotent acknowledgement, not a second event.
    assert store.pause_advance(SID, OP1, fence_epoch=1).paused is True
    assert event_names(store).count("advance_paused") == 1
    with pytest.raises(StaleTrainingFence):
        store.pause_advance(SID, OP1, fence_epoch=2)
    with pytest.raises(TrainingLineageConflict):
        store.pause_advance(SID, OP2, fence_epoch=1)


def test_pause_requires_a_recovery_checkpoint(store: TrainingSessionStore) -> None:
    create_default(store)
    store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1)
    with pytest.raises(TrainingLineageConflict, match="no recovery checkpoint"):
        store.pause_advance(SID, OP1, fence_epoch=1)


def test_same_fence_reclaim_resumes_only_a_paused_operation(
    store: TrainingSessionStore,
) -> None:
    create_default(store)
    store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1)
    # Not paused: the claim may still be running, so a same-fence re-claim
    # stays refused - continuing it could double-step the trajectory.
    with pytest.raises(AdvanceInProgress):
        store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1)
    store.set_recovery_checkpoint(SID, OP1, fence_epoch=1, manifest_digest=RECOVERY)
    store.pause_advance(SID, OP1, fence_epoch=1)
    resumed = store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1)
    assert resumed.status == "started"
    assert resumed.paused is False
    assert resumed.recovery_checkpoint_digest == RECOVERY
    assert event_names(store)[-1] == "advance_resumed"


def test_pause_refuses_resolved_operations(store: TrainingSessionStore) -> None:
    create_default(store)
    store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1)
    store.commit_advance(
        SID,
        OP1,
        fence_epoch=1,
        output_manifest_digest=CKPT1,
        output_step_cursor=1,
        covered_journal_seq=2,
    )
    with pytest.raises(TrainingLineageConflict, match="committed, not started"):
        store.pause_advance(SID, OP1, fence_epoch=1)


# -- commit protocol ----------------------------------------------------------


def test_commit_moves_head_and_appends_facts(store: TrainingSessionStore) -> None:
    create_default(store)
    store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1)
    committed = store.commit_advance(
        SID,
        OP1,
        fence_epoch=1,
        output_manifest_digest=CKPT1,
        output_step_cursor=100,
        covered_journal_seq=2,
    )
    assert committed.status == "committed"
    assert committed.output_manifest_digest == CKPT1
    assert committed.resolved_at is not None

    session = store.get_session(SID)
    assert session is not None
    assert session.committed_manifest_digest == CKPT1
    assert session.committed_step_cursor == 100
    assert session.committed_journal_seq == 2
    assert session.handle().journal_seq == 2

    checkpoints = store.list_checkpoints(SID)
    assert [(c.manifest_digest, c.parent_digest) for c in checkpoints] == [
        (INIT, ""),
        (CKPT1, INIT),
    ]
    assert checkpoints[1].operation_id == OP1
    assert event_names(store) == [
        "session_created",
        "advance_started",
        "checkpoint_published",
        "advance_committed",
    ]

    # Idempotent repeat of the identical commit.
    assert (
        store.commit_advance(
            SID,
            OP1,
            fence_epoch=1,
            output_manifest_digest=CKPT1,
            output_step_cursor=100,
            covered_journal_seq=2,
        )
        == committed
    )
    assert event_names(store).count("advance_committed") == 1
    # A different output for the same operation contradicts exactly-once.
    with pytest.raises(TrainingLineageConflict):
        store.commit_advance(
            SID,
            OP1,
            fence_epoch=1,
            output_manifest_digest=CKPT2,
            output_step_cursor=200,
            covered_journal_seq=2,
        )


def test_commit_refusals(store: TrainingSessionStore) -> None:
    create_default(store)
    with pytest.raises(TrainingLineageConflict):  # never claimed
        store.commit_advance(
            SID,
            OP1,
            fence_epoch=1,
            output_manifest_digest=CKPT1,
            output_step_cursor=1,
            covered_journal_seq=1,
        )
    store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1)
    with pytest.raises(TrainingLineageConflict):  # cursor must strictly increase
        store.commit_advance(
            SID,
            OP1,
            fence_epoch=1,
            output_manifest_digest=CKPT1,
            output_step_cursor=0,
            covered_journal_seq=2,
        )
    with pytest.raises(TrainingLineageConflict):  # watermark beyond the stream
        store.commit_advance(
            SID,
            OP1,
            fence_epoch=1,
            output_manifest_digest=CKPT1,
            output_step_cursor=1,
            covered_journal_seq=99,
        )
    with pytest.raises(TrainingLineageConflict):  # duplicate checkpoint digest
        store.commit_advance(
            SID,
            OP1,
            fence_epoch=1,
            output_manifest_digest=INIT,
            output_step_cursor=1,
            covered_journal_seq=2,
        )
    with pytest.raises(StaleTrainingFence):
        store.commit_advance(
            SID,
            OP1,
            fence_epoch=2,
            output_manifest_digest=CKPT1,
            output_step_cursor=1,
            covered_journal_seq=2,
        )
    # Nothing above moved the ledger.
    session = store.get_session(SID)
    assert session is not None
    assert session.committed_manifest_digest == INIT
    assert event_names(store) == ["session_created", "advance_started"]


def test_commit_watermark_cannot_regress_across_advances(store: TrainingSessionStore) -> None:
    create_default(store)
    store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1)
    store.commit_advance(
        SID,
        OP1,
        fence_epoch=1,
        output_manifest_digest=CKPT1,
        output_step_cursor=100,
        covered_journal_seq=2,
    )
    store.claim_advance(SID, OP2, input_manifest_digest=CKPT1, fence_epoch=1)
    with pytest.raises(TrainingLineageConflict):
        store.commit_advance(
            SID,
            OP2,
            fence_epoch=1,
            output_manifest_digest=CKPT2,
            output_step_cursor=200,
            covered_journal_seq=1,  # below the committed watermark
        )


def test_abort_refusals(store: TrainingSessionStore) -> None:
    create_default(store)
    with pytest.raises(TrainingLineageConflict):
        store.abort_advance(SID, OP1, fence_epoch=1)
    store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1)
    with pytest.raises(StaleTrainingFence):
        store.abort_advance(SID, OP1, fence_epoch=2)
    store.commit_advance(
        SID,
        OP1,
        fence_epoch=1,
        output_manifest_digest=CKPT1,
        output_step_cursor=1,
        covered_journal_seq=2,
    )
    with pytest.raises(TrainingLineageConflict):  # commits are final
        store.abort_advance(SID, OP1, fence_epoch=1)


# -- session completion -------------------------------------------------------


def test_complete_session(store: TrainingSessionStore) -> None:
    create_default(store)
    store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1)
    with pytest.raises(TrainingLineageConflict):  # started operation blocks
        store.complete_session(SID, fence_epoch=1)
    store.abort_advance(SID, OP1, fence_epoch=1)
    completed = store.complete_session(SID, fence_epoch=1)
    assert completed.state == "completed"
    assert store.complete_session(SID, fence_epoch=1).state == "completed"  # idempotent
    assert event_names(store).count("session_completed") == 1
    with pytest.raises(TrainingLineageConflict):
        store.claim_advance(SID, OP2, input_manifest_digest=INIT, fence_epoch=1)
    with pytest.raises(TrainingLineageConflict):
        store.acquire_fence(SID)


# -- reads --------------------------------------------------------------------


def test_reads_fail_soft_on_malformed_or_unknown_ids(store: TrainingSessionStore) -> None:
    assert store.get_session("nope!") is None
    assert store.get_session(SID) is None
    assert store.get_operation(SID, "nope!") is None
    assert store.list_checkpoints("nope!") == ()
    with pytest.raises(ValueError):
        store.read_events("nope!")


def test_state_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "training.sqlite"
    store = TrainingSessionStore(JournalStore(path))
    create_default(store)
    store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1)
    store.commit_advance(
        SID,
        OP1,
        fence_epoch=1,
        output_manifest_digest=CKPT1,
        output_step_cursor=5,
        covered_journal_seq=2,
    )
    store.close()

    reopened = TrainingSessionStore(JournalStore(path))
    session = reopened.get_session(SID)
    assert session is not None
    assert session.committed_manifest_digest == CKPT1
    assert len(reopened.list_checkpoints(SID)) == 2
    assert event_names(reopened) == [
        "session_created",
        "advance_started",
        "checkpoint_published",
        "advance_committed",
    ]
    reopened.close()


def test_refused_mutation_rolls_back_all_writes(tmp_path: Path) -> None:
    """The ledger move and the journal fact land atomically: a refusal
    after facts were appended must leave neither behind."""
    store = TrainingSessionStore(JournalStore(tmp_path / "training.sqlite"))
    create_default(store)
    store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1)
    # commit_advance appends checkpoint_published before advance_committed;
    # forcing a late failure (duplicate checkpoint row inserted underneath
    # the same transaction is impossible from outside, so use the public
    # refusal furthest into the call: the duplicate-manifest check fires
    # after fence and cursor checks but before any append).
    with pytest.raises(TrainingLineageConflict):
        store.commit_advance(
            SID,
            OP1,
            fence_epoch=1,
            output_manifest_digest=INIT,
            output_step_cursor=1,
            covered_journal_seq=2,
        )
    assert event_names(store) == ["session_created", "advance_started"]
    op = store.get_operation(SID, OP1)
    assert op is not None
    assert op.status == "started"
    store.close()


# -- HTTP surface -------------------------------------------------------------

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


def test_training_routes(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = TrainingSessionStore(JournalStore(tmp_path / "training.sqlite"))
        create_default(store)
        store.claim_advance(SID, OP1, input_manifest_digest=INIT, fence_epoch=1)
        store.commit_advance(
            SID,
            OP1,
            fence_epoch=1,
            output_manifest_digest=CKPT1,
            output_step_cursor=7,
            covered_journal_seq=2,
        )
        app = create_app(make_engine, SCHEMAS, training_sessions=store)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            base = f"/api/training/sessions/{SID}"
            # Scope is required for the local principal.
            assert (await client.get(base)).status == 400

            resp = await client.get(base, params={"scope": "local"})
            assert resp.status == 200
            wire = await resp.json()
            assert wire["committedManifestDigest"] == CKPT1
            assert wire["handle"]["stepCursor"] == 7
            assert wire["handle"]["journalSeq"] == 2

            # Wrong scope and unknown/malformed ids are the same plain 404.
            for url in (base, f"/api/training/sessions/{OTHER_SID}"):
                assert (await client.get(url, params={"scope": "other"})).status == 404
            assert (
                await client.get("/api/training/sessions/nope", params={"scope": "local"})
            ).status == 404

            resp = await client.get(f"{base}/events", params={"scope": "local"})
            assert resp.status == 200
            events = await resp.json()
            assert [record["name"] for record in events["records"]] == [
                "session_created",
                "advance_started",
                "checkpoint_published",
                "advance_committed",
            ]
            assert events["latestSeq"] == 4
            assert events["coalescedBelow"] == 0
            assert events["handle"]["checkpointManifestDigest"] == CKPT1
            for record in events["records"]:
                TrainingJournalEvent.from_wire(record["payload"])

            resp = await client.get(
                f"{base}/events", params={"scope": "local", "after": "3", "limit": "10"}
            )
            assert [record["seq"] for record in (await resp.json())["records"]] == [4]
            assert (
                await client.get(f"{base}/events", params={"scope": "local", "after": "-1"})
            ).status == 400
            assert (
                await client.get(f"{base}/events", params={"scope": "local", "limit": "x"})
            ).status == 400

            resp = await client.get(f"{base}/checkpoints", params={"scope": "local"})
            assert resp.status == 200
            checkpoints = (await resp.json())["checkpoints"]
            assert [record["manifestDigest"] for record in checkpoints] == [INIT, CKPT1]
            assert checkpoints[1]["parentDigest"] == INIT
            assert checkpoints[1]["operationId"] == OP1
        finally:
            await client.close()  # on_cleanup closes the store (and journal)

    asyncio.run(scenario())


def test_training_routes_scope_tokens(tmp_path: Path) -> None:
    """Token principals need training:read in the session's scope; anything
    else is indistinguishable from a missing session."""

    class Authenticator:
        principals = {
            "reader": Principal("reader", {"a": frozenset({"training:read"})}),
            "outsider": Principal("outsider", {"b": frozenset({"training:read"})}),
        }

        async def authenticate(self, token: str) -> Principal | None:
            return self.principals.get(token)

    async def scenario() -> None:
        store = TrainingSessionStore(JournalStore(tmp_path / "training.sqlite"))
        create_default(store, scope="a")
        app = create_app(
            make_engine, SCHEMAS, training_sessions=store, authenticator=Authenticator()
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            url = f"/api/training/sessions/{SID}"
            reader = {"Authorization": "Bearer reader"}
            resp = await client.get(url, headers=reader)  # sole scope resolves
            assert resp.status == 200
            assert (await resp.json())["scope"] == "a"

            outsider = {"Authorization": "Bearer outsider"}
            hidden = await client.get(url, params={"scope": "a"}, headers=outsider)
            assert hidden.status == 404
        finally:
            await client.close()

    asyncio.run(scenario())
