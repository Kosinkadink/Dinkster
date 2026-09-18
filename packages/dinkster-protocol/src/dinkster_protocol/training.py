"""Training session contracts: the graph-boundary session handle and the
durable journal event envelope (docs/training-design.md sections 3.2, 9.2).

The handle is the only training state that crosses a graph state port. It
is reference data, never bearer authorization: possession does not permit
observing or mutating a session, and no worker token, PID, device address,
tensor, or live object may enter it. Journal events are the durable source
of truth for training facts; live graph events mirror journal records and
may coalesce telemetry, so nothing correctness-critical rides only on the
droppable ``InvocationEvent`` channel.

These shapes live in ``dinkster-protocol`` because they cross the same worker
boundary as invocations: the server-side session supervisor, training
workers, and future remote hosts must share them without importing one
another.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeGuard, cast

from .payload import canonical_json_payload

__all__ = [
    "COALESCIBLE_TRAINING_EVENTS",
    "DURABLE_TRAINING_EVENTS",
    "MAX_TRAINING_EVENT_DATA_BYTES",
    "TRAINING_SESSION_HANDLE_SCHEMA_VERSION",
    "TrainingEventName",
    "TrainingJournalEvent",
    "TrainingSessionHandle",
    "is_durable_training_event",
    "is_training_checkpoint_digest",
    "is_training_operation_id",
    "is_training_session_id",
    "training_session_stream",
]


TRAINING_SESSION_HANDLE_SCHEMA_VERSION = 1
"""The only handle schema version this codebase encodes or decodes."""

MAX_TRAINING_EVENT_DATA_BYTES = 64 * 1024
"""Ceiling on one event's canonical-JSON ``data`` payload. Large values
(previews, tensors, manifests) travel as artifact/CAS references, never
inline journal JSON."""

_HEX_DIGITS = frozenset("0123456789abcdef")


def _runtime_value(value: object) -> object:
    """Erase a static annotation so frozen dataclass inputs can be validated."""
    return value


def _is_session_id(value: object) -> bool:
    """Canonical session id: 32-64 lowercase hex chars, host-minted from at
    least 128 bits of entropy. One canonical encoding keeps the handle
    fingerprint and journal stream naming free of aliasing."""
    return (
        type(value) is str and 32 <= len(value) <= 64 and all(char in _HEX_DIGITS for char in value)
    )


def _is_checkpoint_digest(value: object) -> bool:
    return (
        type(value) is str
        and value.startswith("blake3:")
        and len(value) == 71
        and all(char in _HEX_DIGITS for char in value[7:])
    )


def _is_count(value: object) -> bool:
    return type(value) is int and value >= 0


def is_training_session_id(value: object) -> TypeGuard[str]:
    """Whether value is a canonical host-minted training session id."""
    return _is_session_id(value)


def is_training_operation_id(value: object) -> TypeGuard[str]:
    """Whether value is a canonical advance operation id. Operation ids are
    digests of the advance operation identity (training-design.md 3.2), so
    they share the session id encoding: 32-64 lowercase hex characters."""
    return _is_session_id(value)


def is_training_checkpoint_digest(value: object) -> TypeGuard[str]:
    """Whether value is a canonical checkpoint manifest digest."""
    return _is_checkpoint_digest(value)


def training_session_stream(session_id: str) -> str:
    """The journal stream id for one training session. All appenders and
    readers derive it here so session streams can never collide with other
    stream families sharing the journal substrate."""
    if not _is_session_id(session_id):
        raise ValueError("session_id must be 32-64 lowercase hex characters")
    return f"training-session/{session_id}"


_HANDLE_WIRE_KEYS = frozenset(
    {
        "schemaVersion",
        "sessionId",
        "checkpointManifestDigest",
        "stepCursor",
        "configDigest",
        "sessionExtensionSnapshotDigest",
        "journalSeq",
    }
)


@dataclass(frozen=True)
class TrainingSessionHandle:
    """``core.training_session_handle.v1``: exactly the fields that cross a
    graph state port between ``CreateTrainingSession`` and ``AdvanceTraining``.

    ``step_cursor`` is the committed optimizer-step count of the checkpoint
    the handle names. ``journal_seq`` is that checkpoint manifest's
    checkpoint-covered durable journal watermark - not the sequence of any
    later ``advance_committed`` event, so it may trail later durable or
    coalesced telemetry but never exceeds the durable session watermark.
    """

    session_id: str
    checkpoint_manifest_digest: str
    step_cursor: int
    config_digest: str
    session_extension_snapshot_digest: str
    journal_seq: int
    schema_version: int = TRAINING_SESSION_HANDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != TRAINING_SESSION_HANDLE_SCHEMA_VERSION
        ):
            raise ValueError(
                "TrainingSessionHandle.schema_version must be the int "
                f"{TRAINING_SESSION_HANDLE_SCHEMA_VERSION}"
            )
        if not _is_session_id(self.session_id):
            raise ValueError(
                "TrainingSessionHandle.session_id must be 32-64 lowercase hex characters"
            )
        for name in (
            "checkpoint_manifest_digest",
            "config_digest",
            "session_extension_snapshot_digest",
        ):
            if not _is_checkpoint_digest(getattr(self, name)):
                raise ValueError(
                    f"TrainingSessionHandle.{name} must be 'blake3:' + 64 lowercase hex"
                )
        if not _is_count(self.step_cursor):
            raise ValueError("TrainingSessionHandle.step_cursor must be a non-negative int")
        if not _is_count(self.journal_seq):
            raise ValueError("TrainingSessionHandle.journal_seq must be a non-negative int")

    def to_wire(self) -> dict[str, object]:
        return {
            "schemaVersion": self.schema_version,
            "sessionId": self.session_id,
            "checkpointManifestDigest": self.checkpoint_manifest_digest,
            "stepCursor": self.step_cursor,
            "configDigest": self.config_digest,
            "sessionExtensionSnapshotDigest": self.session_extension_snapshot_digest,
            "journalSeq": self.journal_seq,
        }

    @classmethod
    def from_wire(cls, value: object) -> TrainingSessionHandle:
        """Fail-closed decode: the exact field set, canonical encodings only.
        Field values are revalidated by ``__post_init__``, whose checks use
        exact runtime types, so a bool masquerading as an int is refused."""
        if not isinstance(value, Mapping) or set(cast("Mapping[str, object]", value)) != set(
            _HANDLE_WIRE_KEYS
        ):
            raise ValueError("training session handle must contain the exact field set")
        raw = cast("Mapping[str, object]", value)
        return cls(
            schema_version=cast("int", raw["schemaVersion"]),
            session_id=cast("str", raw["sessionId"]),
            checkpoint_manifest_digest=cast("str", raw["checkpointManifestDigest"]),
            step_cursor=cast("int", raw["stepCursor"]),
            config_digest=cast("str", raw["configDigest"]),
            session_extension_snapshot_digest=cast("str", raw["sessionExtensionSnapshotDigest"]),
            journal_seq=cast("int", raw["journalSeq"]),
        )

    def fingerprint(self) -> str:
        """Digest of the canonical serialization of every field. Two handles
        are the same graph value exactly when their fingerprints match."""
        canonical = json.dumps(self.to_wire(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode("ascii")).hexdigest()


class TrainingEventName(StrEnum):
    """The complete journal event vocabulary (training-design.md 9.2)."""

    SESSION_CREATED = "session_created"
    SESSION_RECOVERED = "session_recovered"
    PHASE_CHANGED = "phase_changed"
    ADVANCE_STARTED = "advance_started"
    TRAIN_PROGRESS = "train_progress"
    METRIC = "metric"
    RESOURCE_REPORT = "resource_report"
    RECOVERY_CHECKPOINT_PUBLISHED = "recovery_checkpoint_published"
    CHECKPOINT_PUBLISHED = "checkpoint_published"
    PREVIEW_STARTED = "preview_started"
    PREVIEW_PUBLISHED = "preview_published"
    WARNING = "warning"
    DIAGNOSTIC = "diagnostic"
    CANCEL_REQUESTED = "cancel_requested"
    ADVANCE_PAUSED = "advance_paused"
    ADVANCE_RESUMED = "advance_resumed"
    ADVANCE_COMMITTED = "advance_committed"
    ADVANCE_ABORTED = "advance_aborted"
    ADVANCE_FAILED = "advance_failed"
    SESSION_COMPLETED = "session_completed"


COALESCIBLE_TRAINING_EVENTS = frozenset(
    {
        TrainingEventName.TRAIN_PROGRESS,
        TrainingEventName.METRIC,
        TrainingEventName.RESOURCE_REPORT,
    }
)
"""The only events that may be coalesced before durable append under a
declared policy and pruned by journal retention: high-frequency step
progress, metrics, and resource samples."""

DURABLE_TRAINING_EVENTS = frozenset(TrainingEventName) - COALESCIBLE_TRAINING_EVENTS
"""Never-drop facts. Everything not explicitly coalescible is durable:
checkpoint publication, operation claims/commit, cancellation
acknowledgement, lineage/fence changes, failures, terminals, previews
(artifact refs), warnings, and diagnostics."""


def is_durable_training_event(name: TrainingEventName) -> bool:
    return name in DURABLE_TRAINING_EVENTS


_ADVANCE_SCOPED_EVENTS = frozenset(
    {
        TrainingEventName.ADVANCE_STARTED,
        TrainingEventName.RECOVERY_CHECKPOINT_PUBLISHED,
        TrainingEventName.CANCEL_REQUESTED,
        TrainingEventName.ADVANCE_PAUSED,
        TrainingEventName.ADVANCE_RESUMED,
        TrainingEventName.ADVANCE_COMMITTED,
        TrainingEventName.ADVANCE_ABORTED,
        TrainingEventName.ADVANCE_FAILED,
    }
)
"""Operation-ledger facts: meaningless without the advance they belong to."""


_EVENT_WIRE_KEYS = frozenset(
    {
        "sessionId",
        "name",
        "timestamp",
        "advanceId",
        "phase",
        "sessionAttempt",
        "fenceEpoch",
        "jobRef",
        "nodeAttempt",
        "rank",
        "data",
    }
)


def _is_opaque_ref(value: object) -> bool:
    """Empty (absent) or a non-empty token with no whitespace/control chars."""
    return type(value) is str and (
        value == "" or all(not char.isspace() and char.isprintable() for char in value)
    )


@dataclass(frozen=True)
class TrainingJournalEvent:
    """One durable training fact as appended by the session supervisor.

    The per-stream monotonic ``seq`` is assigned by the journal store at
    append time, never by the producer, so it is deliberately not a field
    here; replay surfaces carry (seq, event) pairs. ``advance_id`` is empty
    only for session-level events. ``job_ref``/``node_attempt`` are optional
    graph provenance linkage, never session identity. ``rank`` is the
    distributed rank of the reporting writer, when any.
    """

    session_id: str
    name: TrainingEventName
    timestamp: float
    advance_id: str = ""
    phase: str = ""
    session_attempt: int = 1
    fence_epoch: int = 1
    job_ref: str = ""
    node_attempt: int | None = None
    rank: int | None = None
    data: Mapping[str, object] = field(default_factory=dict[str, object])

    def __post_init__(self) -> None:
        if not _is_session_id(self.session_id):
            raise ValueError(
                "TrainingJournalEvent.session_id must be 32-64 lowercase hex characters"
            )
        if not isinstance(_runtime_value(self.name), TrainingEventName):
            raise ValueError("TrainingJournalEvent.name must be a TrainingEventName")
        if (
            type(self.timestamp) is not float
            or not math.isfinite(self.timestamp)
            or self.timestamp < 0.0
        ):
            raise ValueError("TrainingJournalEvent.timestamp must be a finite non-negative float")
        if not _is_opaque_ref(self.advance_id):
            raise ValueError("TrainingJournalEvent.advance_id must be empty or an opaque token")
        if self.name in _ADVANCE_SCOPED_EVENTS and not self.advance_id:
            raise ValueError(f"{self.name.value} events require an advance_id")
        if not _is_opaque_ref(self.phase):
            raise ValueError("TrainingJournalEvent.phase must be empty or an opaque token")
        if self.name is TrainingEventName.PHASE_CHANGED and not self.phase:
            raise ValueError("phase_changed events require a phase")
        if type(self.session_attempt) is not int or self.session_attempt < 1:
            raise ValueError("TrainingJournalEvent.session_attempt must be an int >= 1")
        if type(self.fence_epoch) is not int or self.fence_epoch < 1:
            raise ValueError("TrainingJournalEvent.fence_epoch must be an int >= 1")
        if not _is_opaque_ref(self.job_ref):
            raise ValueError("TrainingJournalEvent.job_ref must be empty or an opaque token")
        if self.node_attempt is not None and (
            type(self.node_attempt) is not int or self.node_attempt < 1
        ):
            raise ValueError("TrainingJournalEvent.node_attempt must be None or an int >= 1")
        if self.rank is not None and (type(self.rank) is not int or self.rank < 0):
            raise ValueError("TrainingJournalEvent.rank must be None or a non-negative int")
        canonical = canonical_json_payload(
            _runtime_value(self.data),
            max_bytes=MAX_TRAINING_EVENT_DATA_BYTES,
            description="TrainingJournalEvent.data",
        )
        # Snapshot the validated canonical form so later mutation of the
        # caller's mapping cannot change what this frozen event carries.
        object.__setattr__(self, "data", cast("Mapping[str, object]", json.loads(canonical)))

    def to_wire(self) -> dict[str, object]:
        return {
            "sessionId": self.session_id,
            "name": self.name.value,
            "timestamp": self.timestamp,
            "advanceId": self.advance_id,
            "phase": self.phase,
            "sessionAttempt": self.session_attempt,
            "fenceEpoch": self.fence_epoch,
            "jobRef": self.job_ref,
            "nodeAttempt": self.node_attempt,
            "rank": self.rank,
            "data": dict(self.data),
        }

    @classmethod
    def from_wire(cls, value: object) -> TrainingJournalEvent:
        """Fail-closed decode mirroring the handle: exact field set, unknown
        event names refused, field values revalidated by ``__post_init__``."""
        if not isinstance(value, Mapping) or set(cast("Mapping[str, object]", value)) != set(
            _EVENT_WIRE_KEYS
        ):
            raise ValueError("training journal event must contain the exact field set")
        raw = cast("Mapping[str, object]", value)
        name = raw["name"]
        if type(name) is not str:
            raise ValueError("training journal event name must be a string")
        return cls(
            session_id=cast("str", raw["sessionId"]),
            name=TrainingEventName(name),
            timestamp=cast("float", raw["timestamp"]),
            advance_id=cast("str", raw["advanceId"]),
            phase=cast("str", raw["phase"]),
            session_attempt=cast("int", raw["sessionAttempt"]),
            fence_epoch=cast("int", raw["fenceEpoch"]),
            job_ref=cast("str", raw["jobRef"]),
            node_attempt=cast("int | None", raw["nodeAttempt"]),
            rank=cast("int | None", raw["rank"]),
            data=cast("Mapping[str, object]", raw["data"]),
        )
