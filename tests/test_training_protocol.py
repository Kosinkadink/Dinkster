"""Training session contracts: the graph-boundary handle and the durable
journal event envelope (docs/training-design.md sections 3.2, 9.2).

The handle decode is fail-closed: exact field set, canonical encodings,
unknown schema versions refused. The event vocabulary partitions into
never-drop durable facts and coalescible telemetry.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import cast

import pytest
from dinkster_protocol import (
    COALESCIBLE_TRAINING_EVENTS,
    DURABLE_TRAINING_EVENTS,
    MAX_TRAINING_EVENT_DATA_BYTES,
    TRAINING_SESSION_HANDLE_SCHEMA_VERSION,
    TrainingEventName,
    TrainingJournalEvent,
    TrainingSessionHandle,
    is_durable_training_event,
    training_session_stream,
)

SESSION_ID = "0123456789abcdef0123456789abcdef"
DIGEST_A = "blake3:" + "ab" * 32
DIGEST_B = "blake3:" + "cd" * 32
DIGEST_C = "blake3:" + "ef" * 32


def make_handle(**overrides: object) -> TrainingSessionHandle:
    handle = TrainingSessionHandle(
        session_id=SESSION_ID,
        checkpoint_manifest_digest=DIGEST_A,
        step_cursor=100,
        config_digest=DIGEST_B,
        session_extension_snapshot_digest=DIGEST_C,
        journal_seq=42,
    )
    return replace(handle, **overrides) if overrides else handle  # type: ignore[arg-type]


def make_event(**overrides: object) -> TrainingJournalEvent:
    event = TrainingJournalEvent(
        session_id=SESSION_ID,
        name=TrainingEventName.SESSION_CREATED,
        timestamp=1000.5,
    )
    return replace(event, **overrides) if overrides else event  # type: ignore[arg-type]


class TestSessionHandle:
    def test_roundtrip(self) -> None:
        handle = make_handle()
        wire = handle.to_wire()
        assert wire == {
            "schemaVersion": TRAINING_SESSION_HANDLE_SCHEMA_VERSION,
            "sessionId": SESSION_ID,
            "checkpointManifestDigest": DIGEST_A,
            "stepCursor": 100,
            "configDigest": DIGEST_B,
            "sessionExtensionSnapshotDigest": DIGEST_C,
            "journalSeq": 42,
        }
        assert TrainingSessionHandle.from_wire(wire) == handle

    def test_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            make_handle().session_id = "x"  # type: ignore[misc]

    def test_rejects_unknown_schema_version(self) -> None:
        with pytest.raises(ValueError, match="schema_version"):
            make_handle(schema_version=2)
        wire = make_handle().to_wire()
        wire["schemaVersion"] = 0
        with pytest.raises(ValueError, match="schema_version"):
            TrainingSessionHandle.from_wire(wire)

    @pytest.mark.parametrize("version", [True, 1.0])
    def test_rejects_noncanonical_schema_version_types(self, version: object) -> None:
        # bool and 1.0 compare equal to 1 but serialize differently, which
        # would fork the fingerprint of otherwise equal handles.
        with pytest.raises(ValueError, match="schema_version"):
            make_handle(schema_version=version)
        wire = make_handle().to_wire()
        wire["schemaVersion"] = version
        with pytest.raises(ValueError, match="schema_version"):
            TrainingSessionHandle.from_wire(wire)

    @pytest.mark.parametrize(
        "session_id",
        [
            "",
            "short",
            "0123456789ABCDEF0123456789ABCDEF",  # uppercase is noncanonical
            "0123456789abcdef0123456789abcde",  # 31 chars
            "g" * 32,  # not hex
            "0" * 65,  # too long
        ],
    )
    def test_rejects_malformed_session_id(self, session_id: str) -> None:
        with pytest.raises(ValueError, match="session_id"):
            make_handle(session_id=session_id)

    @pytest.mark.parametrize(
        "digest",
        [
            "",
            "ab" * 32,  # missing prefix
            "blake3:" + "AB" * 32,  # uppercase
            "blake3:" + "ab" * 31,  # short
            "sha256:" + "ab" * 32,  # wrong algorithm
        ],
    )
    def test_rejects_malformed_digests(self, digest: str) -> None:
        for name in (
            "checkpoint_manifest_digest",
            "config_digest",
            "session_extension_snapshot_digest",
        ):
            with pytest.raises(ValueError, match=name):
                make_handle(**{name: digest})

    def test_rejects_negative_and_non_int_counters(self) -> None:
        with pytest.raises(ValueError, match="step_cursor"):
            make_handle(step_cursor=-1)
        with pytest.raises(ValueError, match="journal_seq"):
            make_handle(journal_seq=-1)
        with pytest.raises(ValueError, match="step_cursor"):
            make_handle(step_cursor=True)  # bool is not an int on this wire
        with pytest.raises(ValueError, match="step_cursor"):
            make_handle(step_cursor=1.0)

    def test_from_wire_requires_exact_field_set(self) -> None:
        wire = make_handle().to_wire()
        missing = dict(wire)
        del missing["journalSeq"]
        with pytest.raises(ValueError, match="exact field set"):
            TrainingSessionHandle.from_wire(missing)
        extra = dict(wire)
        extra["workerToken"] = "nope"
        with pytest.raises(ValueError, match="exact field set"):
            TrainingSessionHandle.from_wire(extra)
        with pytest.raises(ValueError, match="exact field set"):
            TrainingSessionHandle.from_wire("not a mapping")

    def test_from_wire_rejects_wrong_value_types(self) -> None:
        wire = make_handle().to_wire()
        wire["stepCursor"] = "100"
        with pytest.raises(ValueError, match="step_cursor"):
            TrainingSessionHandle.from_wire(wire)

    def test_fingerprint_is_stable(self) -> None:
        # Fixed vector: sha256 of the canonical JSON (sorted keys, compact
        # separators) of to_wire(). A change here breaks every persisted
        # handle fingerprint and must bump the handle schema version.
        assert make_handle().fingerprint() == (
            "sha256:a85be35966455a75931077a0411168e573e2ffc1900b84cce50f8e13fa546144"
        )

    def test_fingerprint_covers_every_field(self) -> None:
        base = make_handle()
        assert base.fingerprint() == make_handle().fingerprint()
        assert base.fingerprint().startswith("sha256:")
        variants = [
            make_handle(session_id="f" * 32),
            make_handle(checkpoint_manifest_digest=DIGEST_C),
            make_handle(step_cursor=101),
            make_handle(config_digest=DIGEST_A),
            make_handle(session_extension_snapshot_digest=DIGEST_B),
            make_handle(journal_seq=43),
        ]
        fingerprints = {base.fingerprint(), *(variant.fingerprint() for variant in variants)}
        assert len(fingerprints) == 1 + len(variants)


class TestEventVocabulary:
    def test_exact_vocabulary(self) -> None:
        assert {name.value for name in TrainingEventName} == {
            "session_created",
            "session_recovered",
            "phase_changed",
            "advance_started",
            "train_progress",
            "metric",
            "resource_report",
            "recovery_checkpoint_published",
            "checkpoint_published",
            "preview_started",
            "preview_published",
            "warning",
            "diagnostic",
            "cancel_requested",
            "advance_paused",
            "advance_resumed",
            "advance_committed",
            "advance_aborted",
            "advance_failed",
            "session_completed",
        }

    def test_exact_partition(self) -> None:
        # Only high-frequency telemetry may coalesce/prune; every other
        # event, including previews/warnings/diagnostics, never drops.
        assert COALESCIBLE_TRAINING_EVENTS == {
            TrainingEventName.TRAIN_PROGRESS,
            TrainingEventName.METRIC,
            TrainingEventName.RESOURCE_REPORT,
        }
        assert DURABLE_TRAINING_EVENTS == (
            frozenset(TrainingEventName) - COALESCIBLE_TRAINING_EVENTS
        )
        assert not DURABLE_TRAINING_EVENTS & COALESCIBLE_TRAINING_EVENTS
        assert is_durable_training_event(TrainingEventName.PREVIEW_PUBLISHED)
        assert not is_durable_training_event(TrainingEventName.METRIC)


class TestJournalEvent:
    def test_roundtrip(self) -> None:
        event = TrainingJournalEvent(
            session_id=SESSION_ID,
            name=TrainingEventName.ADVANCE_COMMITTED,
            timestamp=12.5,
            advance_id="adv-1",
            phase="train",
            session_attempt=2,
            fence_epoch=3,
            job_ref="run-9",
            node_attempt=1,
            rank=0,
            data={"steps": 8},
        )
        assert TrainingJournalEvent.from_wire(event.to_wire()) == event

    def test_from_wire_requires_exact_field_set(self) -> None:
        wire = make_event().to_wire()
        del wire["rank"]
        with pytest.raises(ValueError, match="exact field set"):
            TrainingJournalEvent.from_wire(wire)
        smuggled = make_event().to_wire()
        smuggled["seq"] = 7  # seq is store-assigned, never producer-supplied
        with pytest.raises(ValueError, match="exact field set"):
            TrainingJournalEvent.from_wire(smuggled)

    def test_rejects_unknown_event_name(self) -> None:
        wire = make_event().to_wire()
        wire["name"] = "made_up_event"
        with pytest.raises(ValueError):
            TrainingJournalEvent.from_wire(wire)

    def test_advance_scoped_events_require_advance_id(self) -> None:
        for name in (
            TrainingEventName.ADVANCE_STARTED,
            TrainingEventName.ADVANCE_COMMITTED,
            TrainingEventName.ADVANCE_ABORTED,
            TrainingEventName.ADVANCE_FAILED,
            TrainingEventName.ADVANCE_PAUSED,
            TrainingEventName.ADVANCE_RESUMED,
            TrainingEventName.CANCEL_REQUESTED,
            TrainingEventName.RECOVERY_CHECKPOINT_PUBLISHED,
        ):
            with pytest.raises(ValueError, match="advance_id"):
                make_event(name=name)
            make_event(name=name, advance_id="adv-1")

    def test_phase_changed_requires_phase(self) -> None:
        with pytest.raises(ValueError, match="phase"):
            make_event(name=TrainingEventName.PHASE_CHANGED)
        make_event(name=TrainingEventName.PHASE_CHANGED, phase="validate")

    def test_rejects_invalid_envelope_fields(self) -> None:
        with pytest.raises(ValueError, match="timestamp"):
            make_event(timestamp=-1.0)
        with pytest.raises(ValueError, match="timestamp"):
            make_event(timestamp=5)  # int, not float
        with pytest.raises(ValueError, match="session_attempt"):
            make_event(session_attempt=0)
        with pytest.raises(ValueError, match="fence_epoch"):
            make_event(fence_epoch=0)
        with pytest.raises(ValueError, match="node_attempt"):
            make_event(node_attempt=0)
        with pytest.raises(ValueError, match="rank"):
            make_event(rank=-1)
        with pytest.raises(ValueError, match="advance_id"):
            make_event(advance_id="has space")

    def test_rejects_noncanonical_timestamps(self) -> None:
        wire = make_event().to_wire()
        wire["timestamp"] = 1000  # int is not the canonical float encoding
        with pytest.raises(ValueError, match="timestamp"):
            TrainingJournalEvent.from_wire(wire)
        with pytest.raises(ValueError, match="timestamp"):
            make_event(timestamp=float("nan"))
        with pytest.raises(ValueError, match="timestamp"):
            make_event(timestamp=float("inf"))

    def test_data_must_be_rpc_clean_json_and_bounded(self) -> None:
        with pytest.raises(ValueError, match="RPC-clean"):
            make_event(data={"blob": b"\x00"})
        with pytest.raises(ValueError, match="string-keyed"):
            make_event(data={1: "x"})
        with pytest.raises(ValueError, match="keys must be strings"):
            make_event(data={"nested": {1: "x"}})
        with pytest.raises(ValueError, match="finite"):
            make_event(data={"loss": float("nan")})
        with pytest.raises(ValueError, match="RPC-clean"):
            make_event(data={"pair": (1, 2)})  # tuples alias to arrays
        with pytest.raises(ValueError, match="artifact"):
            make_event(data={"giant": "x" * (MAX_TRAINING_EVENT_DATA_BYTES + 1)})

    def test_data_is_snapshotted_against_caller_mutation(self) -> None:
        payload: dict[str, object] = {"steps": [1, 2]}
        event = make_event(data=payload)
        cast("list[object]", payload["steps"]).append(3)
        payload["extra"] = "later"
        assert event.data == {"steps": [1, 2]}


def test_training_session_stream_is_namespaced() -> None:
    assert training_session_stream(SESSION_ID) == f"training-session/{SESSION_ID}"
    with pytest.raises(ValueError, match="session_id"):
        training_session_stream("not-a-session")
