"""Durable per-stream journal store: append-assigned monotonic sequence,
replay-after cursor, and telemetry-only retention (docs/training-design.md
section 9.2).

The invariants under test: seq is assigned at append time and never reused
(even after pruning removes the newest telemetry rows), durable records are
never deleted, and the coalesced watermark tells replay consumers how far
telemetry pruning has gone.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from dinkster_server import JournalStore, JournalTransaction

STREAM = "training-session/" + "0123456789abcdef0123456789abcdef"
OTHER_STREAM = "training-session/" + "f" * 32


@pytest.fixture
def store(tmp_path: Path) -> Iterator[JournalStore]:
    journal = JournalStore(tmp_path / "journal.sqlite3")
    yield journal
    journal.close()


def test_append_assigns_monotonic_seq_per_stream(store: JournalStore) -> None:
    first = store.append(STREAM, "session_created", {}, durable=True, timestamp=1.0)
    second = store.append(STREAM, "metric", {"loss": 0.5}, durable=False, timestamp=2.0)
    other = store.append(OTHER_STREAM, "session_created", {}, durable=True, timestamp=3.0)
    assert (first.seq, second.seq) == (1, 2)
    assert other.seq == 1  # streams sequence independently
    assert store.latest_seq(STREAM) == 2
    assert store.latest_seq("training-session/" + "0" * 32) == 0


def test_read_after_cursor(store: JournalStore) -> None:
    for step in range(5):
        store.append(STREAM, "metric", {"step": step}, durable=False, timestamp=float(step))
    page = store.read(STREAM, after=0)
    assert [record.seq for record in page.records] == [1, 2, 3, 4, 5]
    assert page.latest_seq == 5
    assert page.coalesced_below == 0
    tail = store.read(STREAM, after=3)
    assert [record.seq for record in tail.records] == [4, 5]
    assert [record.payload["step"] for record in tail.records] == [3, 4]
    limited = store.read(STREAM, after=0, limit=2)
    assert [record.seq for record in limited.records] == [1, 2]
    with pytest.raises(ValueError, match="after"):
        store.read(STREAM, after=-1)


def test_records_survive_reopen(tmp_path: Path) -> None:
    path = tmp_path / "journal.sqlite3"
    store = JournalStore(path)
    store.append(STREAM, "session_created", {"a": 1}, durable=True, timestamp=1.0)
    store.append(STREAM, "metric", {}, durable=False, timestamp=2.0)
    store.close()
    reopened = JournalStore(path)
    try:
        page = reopened.read(STREAM, after=0)
        assert [record.name for record in page.records] == ["session_created", "metric"]
        assert page.records[0].payload == {"a": 1}
        assert page.records[0].durable is True
        assert reopened.append(STREAM, "metric", {}, durable=False, timestamp=3.0).seq == 3
    finally:
        reopened.close()


def test_prune_removes_only_coalescible_rows(store: JournalStore) -> None:
    store.append(STREAM, "session_created", {}, durable=True, timestamp=1.0)
    store.append(STREAM, "metric", {}, durable=False, timestamp=2.0)
    store.append(STREAM, "checkpoint_published", {}, durable=True, timestamp=3.0)
    store.append(STREAM, "metric", {}, durable=False, timestamp=4.0)
    removed = store.prune_coalescible(STREAM, 3)
    assert removed == 1  # only the seq-2 metric; both durable rows survive
    page = store.read(STREAM, after=0)
    assert [record.seq for record in page.records] == [1, 3, 4]
    assert all(record.durable for record in page.records if record.seq <= 3)
    assert page.coalesced_below == 3
    assert page.latest_seq == 4


def test_seq_never_reused_after_pruning_newest_rows(store: JournalStore) -> None:
    store.append(STREAM, "metric", {}, durable=False, timestamp=1.0)
    store.append(STREAM, "metric", {}, durable=False, timestamp=2.0)
    assert store.prune_coalescible(STREAM, 2) == 2
    assert store.read(STREAM, after=0).records == ()
    record = store.append(STREAM, "metric", {}, durable=False, timestamp=3.0)
    assert record.seq == 3  # not 1: the counter outlives the rows
    assert store.latest_seq(STREAM) == 3


def test_delete_stream_removes_records_and_counter(store: JournalStore) -> None:
    store.append(STREAM, "session_created", {}, durable=True, timestamp=1.0)
    store.append(STREAM, "metric", {}, durable=False, timestamp=2.0)
    store.append(OTHER_STREAM, "session_created", {}, durable=True, timestamp=3.0)
    assert store.delete_stream(STREAM) == 2
    page = store.read(STREAM, after=0)
    assert page.records == ()
    assert page.latest_seq == 0
    assert page.coalesced_below == 0
    # Other streams are untouched.
    assert store.latest_seq(OTHER_STREAM) == 1
    # A reused id is a NEW stream: its seq restarts.
    record = store.append(STREAM, "session_created", {}, durable=True, timestamp=4.0)
    assert record.seq == 1


def test_transaction_read_sees_its_own_writes(store: JournalStore) -> None:
    def run(txn: JournalTransaction) -> None:
        txn.append(STREAM, "session_created", {}, durable=True, timestamp=1.0)
        page = txn.read(STREAM, after=0)
        assert [record.seq for record in page.records] == [1]
        assert page.latest_seq == 1
        assert page.coalesced_below == 0

    store.transact(run)


def test_transaction_prune_and_delete_roll_back_with_the_transaction(
    store: JournalStore,
) -> None:
    store.append(STREAM, "session_created", {}, durable=True, timestamp=1.0)
    store.append(STREAM, "metric", {}, durable=False, timestamp=2.0)

    def fail(txn: JournalTransaction) -> None:
        txn.prune_coalescible(STREAM, 2)
        txn.delete_stream(OTHER_STREAM)
        raise RuntimeError("abort")

    store.append(OTHER_STREAM, "session_created", {}, durable=True, timestamp=3.0)
    with pytest.raises(RuntimeError, match="abort"):
        store.transact(fail)
    assert [record.seq for record in store.read(STREAM, after=0).records] == [1, 2]
    assert store.read(STREAM, after=0).coalesced_below == 0
    assert store.latest_seq(OTHER_STREAM) == 1


def test_prune_watermark_caps_at_latest_and_never_regresses(store: JournalStore) -> None:
    store.append(STREAM, "metric", {}, durable=False, timestamp=1.0)
    store.prune_coalescible(STREAM, 10_000)
    assert store.read(STREAM, after=0).coalesced_below == 1
    store.append(STREAM, "metric", {}, durable=False, timestamp=2.0)
    store.prune_coalescible(STREAM, 0)
    assert store.read(STREAM, after=0).coalesced_below == 1
    with pytest.raises(ValueError, match="up_to"):
        store.prune_coalescible(STREAM, -1)


def test_append_validates_inputs(store: JournalStore) -> None:
    with pytest.raises(ValueError, match="stream_id"):
        store.append(" ", "metric", {}, durable=False)
    with pytest.raises(ValueError, match="name"):
        store.append(STREAM, "", {}, durable=False)
    with pytest.raises(ValueError, match="string-keyed"):
        store.append(STREAM, "metric", {1: "x"}, durable=False)  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="keys must be strings"):
        store.append(STREAM, "metric", {"nested": {1: "x"}}, durable=False)
    with pytest.raises(ValueError, match="finite"):
        store.append(STREAM, "metric", {"loss": float("nan")}, durable=False)
    with pytest.raises(ValueError, match="canonical-JSON"):
        store.append(STREAM, "metric", {"giant": "x" * (256 * 1024 + 1)}, durable=False)
    with pytest.raises(ValueError, match="timestamp"):
        store.append(STREAM, "metric", {}, durable=False, timestamp=-1.0)
    with pytest.raises(ValueError, match="timestamp"):
        store.append(STREAM, "metric", {}, durable=False, timestamp=float("nan"))
    # The durability class decides what retention may prune; a truthy
    # non-bool must not silently become durable (or prunable).
    with pytest.raises(ValueError, match="durable"):
        store.append(STREAM, "metric", {}, durable="false")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="durable"):
        store.append(STREAM, "metric", {}, durable=None)  # type: ignore[arg-type]
    assert store.latest_seq(STREAM) == 0  # nothing above consumed a seq


def test_returned_record_is_decoupled_from_caller_payload(store: JournalStore) -> None:
    payload: dict[str, object] = {"steps": [1, 2]}
    record = store.append(STREAM, "metric", payload, durable=False, timestamp=1.0)
    steps = payload["steps"]
    assert isinstance(steps, list)
    steps.append(3)
    assert record.payload == {"steps": [1, 2]}
    assert store.read(STREAM, after=0).records[0].payload == {"steps": [1, 2]}


def test_append_is_atomic_across_counter_and_row(store: JournalStore) -> None:
    # Force the row insert to fail after the counter upsert by planting a
    # conflicting primary key; the transaction must roll back both writes.
    store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "INSERT INTO journal (stream, seq, ts, name, durable, payload)"
        " VALUES (?, 1, 0.0, 'planted', 1, '{}')",
        (STREAM,),
    )
    store._conn.commit()  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(sqlite3.IntegrityError):
        store.append(STREAM, "metric", {}, durable=False, timestamp=1.0)
    assert store.latest_seq(STREAM) == 0  # counter upsert was rolled back
    store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "DELETE FROM journal WHERE stream = ?", (STREAM,)
    )
    store._conn.commit()  # pyright: ignore[reportPrivateUsage]
    assert store.append(STREAM, "metric", {}, durable=False, timestamp=2.0).seq == 1


def test_transact_rolls_back_every_write_on_raise(store: JournalStore) -> None:
    """The host-owned transaction API: a raising callable leaves no trace,
    neither appended records nor statements run through execute."""

    class Boom(Exception):
        pass

    def doomed(txn: JournalTransaction) -> None:
        txn.execute("CREATE TABLE IF NOT EXISTS scratch (x INTEGER)")
        txn.execute("INSERT INTO scratch VALUES (1)")
        txn.append(STREAM, "metric", {}, durable=False, timestamp=1.0)
        raise Boom

    with pytest.raises(Boom):
        store.transact(doomed)
    assert store.latest_seq(STREAM) == 0
    assert store.read(STREAM).records == ()
    # The transaction opened BEFORE the callable's statements, so even the
    # DDL rolled back.
    remaining = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT name FROM sqlite_master WHERE name = 'scratch'"
    ).fetchone()
    assert remaining is None

    def committed(txn: JournalTransaction) -> int:
        record = txn.append(STREAM, "metric", {}, durable=False, timestamp=2.0)
        assert txn.latest_seq(STREAM) == record.seq
        return record.seq

    assert store.transact(committed) == 1
    assert store.latest_seq(STREAM) == 1


def test_append_defaults_timestamp_to_now(store: JournalStore) -> None:
    record = store.append(STREAM, "metric", {}, durable=False)
    assert record.timestamp > 0.0


def test_record_to_wire(store: JournalStore) -> None:
    record = store.append(STREAM, "metric", {"loss": 0.25}, durable=False, timestamp=7.0)
    assert record.to_wire() == {
        "streamId": STREAM,
        "seq": 1,
        "timestamp": 7.0,
        "name": "metric",
        "durable": False,
        "payload": {"loss": 0.25},
    }
