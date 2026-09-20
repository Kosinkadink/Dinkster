"""Collaborative document sessions.

A session is a server-authoritative, append-only log of forward JSON
patches over one document. The server orders operations with contiguous
revisions and optimistic concurrency on baseRevision, makes operations
replayable after a revision cursor, and keeps one snapshot checkpoint so
the log stays bounded. It never applies or interprets a patch. Clients
materialize documents and retain inverse patches for undo. Presence is
an ephemeral relay in the routes layer and never touches this module.

Concurrency contract (v1, protocolVersion 1): an op names the revision
it was built on (baseRevision). If that is no longer the head, the
append is refused with the current revision - the client rebases and
resubmits; the server never transforms. The protocol version is carried
on operation and session envelopes so OT/CRDT ordering can replace this
contract later without a new surface.

Idempotency: opId is globally unique per client op. Replaying a
retained opId returns the already-assigned envelope instead of
appending twice. The idempotency window equals the retention window:
ops pruned by a snapshot checkpoint forget their opIds too.

Bounds: a session retains at most ``max_retained_ops`` un-checkpointed
ops. At the cap, appends are refused with snapshot-required until a
client posts a checkpoint - backpressure toward the documented duty,
never silent unbounded growth.

Execution never consumes live session state: jobs are submitted from
immutable document snapshots (the /api/jobs contract), and nothing in
this module reaches the queue or engine. Storage uses an in-memory
working set with optional SQLite persistence across process restarts.
"""

from __future__ import annotations

import json
import math
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from .snapshots import normalize_document_kind

if TYPE_CHECKING:
    from .store import SessionStore

PROTOCOL_VERSION = 1

SESSION_ROLES = ("banned", "viewer", "editor", "owner")
DEFAULT_SESSION_ROLE = "editor"
_ROLE_RANK = {role: rank for rank, role in enumerate(SESSION_ROLES)}

_PATCH_OPS = frozenset({"add", "remove", "replace"})
_NEEDS_VALUE = frozenset({"add", "replace"})


def _normalize_json(value: object, where: str = "value") -> object:
    """Copy JSON-shaped data and reject Python-only values at the boundary."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where} must contain only finite JSON numbers")
        return value
    if isinstance(value, list):
        items = cast("list[object]", value)
        return [_normalize_json(item, f"{where}[]") for item in items]
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        items = cast("dict[object, object]", value)
        for key, item in items.items():
            if not isinstance(key, str):
                raise ValueError(f"{where} object keys must be strings")
            normalized[key] = _normalize_json(item, f"{where}.{key}")
        return normalized
    raise ValueError(f"{where} must be JSON-shaped data, got {type(value).__name__}")


class UnknownSessionError(KeyError):
    """No session with that id (or it was closed)."""


class StaleBaseError(Exception):
    """The op's baseRevision is not the session head: rebase and resubmit."""

    def __init__(self, revision: int) -> None:
        super().__init__(f"base is stale; session is at revision {revision}")
        self.revision = revision


class SnapshotRequiredError(Exception):
    """Retention cap hit: a checkpoint must land before more ops can."""

    def __init__(self, revision: int, snapshot_revision: int) -> None:
        super().__init__("retained-op cap reached; checkpoint a snapshot before appending")
        self.revision = revision
        self.snapshot_revision = snapshot_revision


class ResyncRequiredError(Exception):
    """The requested cursor predates the retained log: refetch the snapshot."""

    def __init__(self, snapshot_revision: int) -> None:
        super().__init__(
            f"ops before revision {snapshot_revision} were pruned; resync from the snapshot"
        )
        self.snapshot_revision = snapshot_revision


class ActorPrincipalMismatchError(Exception):
    """The actor id belongs to a different authenticated principal."""


class ActorLimitError(Exception):
    """The session's durable actor ownership table is full."""


class SessionRoleError(Exception):
    """The principal's session role does not permit the operation."""


class NoSessionOwnerError(Exception):
    """A legacy session without an owner cannot have its ACL replaced."""


class InvalidSnapshotError(ValueError):
    """The document-kind validator refused a session snapshot."""


def validate_patch(patch: object) -> str | None:
    """Shape-validate a forward patch: a list of {op, path, value?} ops
    where op is add|remove|replace and path is an ARRAY of string|int
    segments (the frontend's native DocPath shape - no JSON Pointer
    strings, no ~0/~1 escaping). Returns an error message or None.
    Shape ONLY - paths are not resolved and values are not interpreted;
    the server orders patches, it never applies them."""
    if not isinstance(patch, list):
        return "'patch' must be a JSON array of operations"
    entries = cast(list[object], patch)
    if not entries:
        return "'patch' must contain at least one operation"
    for index, raw in enumerate(entries):
        if not isinstance(raw, dict):
            return f"patch[{index}] must be an object"
        entry = cast(dict[str, object], raw)
        op = entry.get("op")
        if op not in _PATCH_OPS:
            return f"patch[{index}].op must be one of {sorted(_PATCH_OPS)}"
        path = entry.get("path")
        if not isinstance(path, list):
            return f"patch[{index}].path must be an array of segments"
        for seg_index, segment in enumerate(cast(list[object], path)):
            if isinstance(segment, bool) or not isinstance(segment, (str, int)):
                return f"patch[{index}].path[{seg_index}] must be a string or integer segment"
        if op in _NEEDS_VALUE and "value" not in entry:
            return f"patch[{index}] ('{op}') requires 'value'"
    return None


def new_session_id() -> str:
    """Opaque, globally unique session id (ms-timestamp prefix + 80
    random bits, same discipline as the job queue's jobRef mint)."""
    return f"{int(time.time() * 1000):012x}{secrets.token_hex(10)}"


@dataclass(frozen=True)
class SessionOp:
    """One ordered operation: the pinned envelope's server-side record.
    ``revision`` is server-assigned and contiguous; ``base_revision`` is
    always ``revision - 1`` by construction (kept explicit so the wire
    envelope never has to derive it). ``timestamp`` is server-stamped at
    ordering time - the authoritative record, not the client clock."""

    op_id: str
    actor_id: str
    base_revision: int
    revision: int
    patch: tuple[Mapping[str, Any], ...]
    timestamp: float
    principal_id: str = "local"
    actor_kind: str = "human"


@dataclass
class DocumentSession:
    session_id: str
    scope: str
    document_id: str
    snapshot: object
    """Client-materialized document at ``snapshot_revision``. Opaque JSON;
    the server never applies ops to it - checkpoints replace it wholesale."""
    snapshot_revision: int = 0
    created_at: float = field(default_factory=time.time)
    ops: list[SessionOp] = field(default_factory=lambda: [])
    """Retained ops, revisions (snapshot_revision, revision], contiguous."""
    ops_by_id: dict[str, SessionOp] = field(default_factory=lambda: {})
    actor_bindings: dict[str, str] = field(default_factory=lambda: {})
    acl: dict[str, str] = field(default_factory=lambda: {})
    default_role: str = DEFAULT_SESSION_ROLE
    document_kind: str = "workflow"

    def __post_init__(self) -> None:
        self.document_kind = normalize_document_kind(self.document_kind)

    @property
    def revision(self) -> int:
        return self.ops[-1].revision if self.ops else self.snapshot_revision


class SessionService:
    """Session ordering, idempotent append, catch-up reads, and snapshot
    checkpoints. Single-threaded by construction (event-loop discipline,
    like the job queue) - no locks needed."""

    def __init__(
        self,
        *,
        max_retained_ops: int = 4096,
        store: SessionStore | None = None,
        snapshot_validator: Callable[[str, str, object], str | None] | None = None,
    ) -> None:
        if max_retained_ops < 1:
            raise ValueError("max_retained_ops must be >= 1")
        self._max_retained_ops = max_retained_ops
        self._store = store
        self._snapshot_validator = snapshot_validator
        loaded = store.load_all() if store is not None else []
        self._sessions = {session.session_id: session for session in loaded}

    def _validated_snapshot(self, document_kind: str, document_id: str, snapshot: object) -> object:
        normalized = _normalize_json(snapshot, "snapshot")
        if self._snapshot_validator is not None:
            problem = self._snapshot_validator(document_kind, document_id, normalized)
            if problem is not None:
                raise InvalidSnapshotError(problem)
        return normalized

    def create(
        self,
        *,
        scope: str,
        document_id: str,
        snapshot: object,
        document_kind: str = "workflow",
        principal_id: str = "local",
    ) -> DocumentSession:
        document_kind = normalize_document_kind(document_kind)
        session = DocumentSession(
            session_id=new_session_id(),
            scope=scope,
            document_id=document_id,
            snapshot=self._validated_snapshot(document_kind, document_id, snapshot),
            document_kind=document_kind,
            acl={principal_id: "owner"},
        )
        self._sessions[session.session_id] = session
        if self._store is not None:
            self._store.put_session(session)
        return deepcopy(session)

    def _get_authoritative(self, session_id: str) -> DocumentSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise UnknownSessionError(session_id)
        return session

    def get(self, session_id: str) -> DocumentSession:
        return deepcopy(self._get_authoritative(session_id))

    def sessions(self, scope: str) -> list[DocumentSession]:
        return [deepcopy(s) for s in self._sessions.values() if s.scope == scope]

    def role_for(self, session_id: str, principal_id: str) -> str:
        session = self._get_authoritative(session_id)
        return session.acl.get(principal_id, session.default_role)

    def require_role(self, session_id: str, principal_id: str, minimum: str) -> None:
        role = self.role_for(session_id, principal_id)
        if _ROLE_RANK[role] < _ROLE_RANK[minimum]:
            raise SessionRoleError(principal_id)

    def acl_policy(self, session_id: str) -> tuple[str, dict[str, str]]:
        session = self._get_authoritative(session_id)
        return session.default_role, dict(session.acl)

    def replace_acl(
        self,
        session_id: str,
        *,
        principal_id: str,
        default_role: str,
        entries: Mapping[str, str],
    ) -> DocumentSession:
        session = self._get_authoritative(session_id)
        if "owner" not in session.acl.values():
            raise NoSessionOwnerError(session_id)
        self.require_role(session_id, principal_id, "owner")
        if default_role not in ("viewer", "editor"):
            raise ValueError("defaultRole must be 'viewer' or 'editor'")
        normalized_entries = dict(entries)
        if not normalized_entries or "owner" not in normalized_entries.values():
            raise ValueError("at least one owner is required")
        if any(not key for key in normalized_entries):
            raise ValueError("ACL principal ids must be non-empty strings")
        if any(role not in SESSION_ROLES for role in normalized_entries.values()):
            raise ValueError(f"ACL roles must be one of {list(SESSION_ROLES)}")
        if self._store is not None:
            self._store.replace_acl(session_id, default_role, normalized_entries)
        session.default_role = default_role
        session.acl = normalized_entries
        return deepcopy(session)

    def close(self, session_id: str, *, principal_id: str = "local") -> DocumentSession:
        session = self._get_authoritative(session_id)
        minimum_role = "owner" if "owner" in session.acl.values() else "editor"
        self.require_role(session_id, principal_id, minimum_role)
        del self._sessions[session_id]
        if self._store is not None:
            self._store.delete_session(session_id)
        return deepcopy(session)

    def append(
        self,
        session_id: str,
        *,
        op_id: str,
        actor_id: str,
        base_revision: int,
        patch: Sequence[Mapping[str, Any]],
        principal_id: str = "local",
        actor_kind: str = "human",
    ) -> tuple[SessionOp, bool]:
        """Order one op. Returns (op, replayed): replayed=True means the
        opId was already retained and the recorded envelope is returned
        unchanged (idempotent resubmission, e.g. a retried request)."""
        session = self._get_authoritative(session_id)
        self.require_role(session_id, principal_id, "editor")
        normalized_patch = tuple(
            cast("dict[str, Any]", _normalize_json(dict(entry), "patch entry")) for entry in patch
        )
        bound_principal = session.actor_bindings.get(actor_id)
        if bound_principal is not None and bound_principal != principal_id:
            raise ActorPrincipalMismatchError(actor_id)
        existing = session.ops_by_id.get(op_id)
        if existing is not None:
            if existing.actor_id != actor_id:
                raise ActorPrincipalMismatchError(actor_id)
            if bound_principal is None:
                self._bind_actor(session, actor_id, principal_id)
            return deepcopy(existing), True
        self.check_actor_principal(session_id, actor_id, principal_id)
        if base_revision != session.revision:
            raise StaleBaseError(session.revision)
        if len(session.ops) >= self._max_retained_ops:
            raise SnapshotRequiredError(session.revision, session.snapshot_revision)
        op = SessionOp(
            op_id=op_id,
            actor_id=actor_id,
            base_revision=session.revision,
            revision=session.revision + 1,
            patch=normalized_patch,
            timestamp=time.time(),
            principal_id=principal_id,
            actor_kind=actor_kind,
        )
        if self._store is not None:
            self._store.append_op(session_id, op, principal_id)
        session.ops.append(op)
        session.ops_by_id[op_id] = op
        session.actor_bindings.setdefault(actor_id, principal_id)
        return deepcopy(op), False

    def bind_actor(self, session_id: str, actor_id: str, principal_id: str) -> None:
        """Bind an actor id for an accepted ephemeral presence frame."""
        session = self._get_authoritative(session_id)
        self.require_role(session_id, principal_id, "viewer")
        self.check_actor_principal(session_id, actor_id, principal_id)
        if actor_id not in session.actor_bindings:
            self._bind_actor(session, actor_id, principal_id)

    def check_actor_principal(self, session_id: str, actor_id: str, principal_id: str) -> None:
        """Refuse an actor id already bound to another principal."""
        session = self._get_authoritative(session_id)
        bound_principal = session.actor_bindings.get(actor_id)
        if bound_principal is not None and bound_principal != principal_id:
            raise ActorPrincipalMismatchError(actor_id)
        if bound_principal is None and (
            len(session.actor_bindings) >= 4096
            or sum(p == principal_id for p in session.actor_bindings.values()) >= 256
        ):
            raise ActorLimitError(session_id)

    def _bind_actor(self, session: DocumentSession, actor_id: str, principal_id: str) -> None:
        if self._store is not None:
            self._store.put_actor(session.session_id, actor_id, principal_id)
        session.actor_bindings[actor_id] = principal_id

    def ops_after(
        self, session_id: str, after: int, *, principal_id: str = "local"
    ) -> list[SessionOp]:
        """Catch-up read: every retained op with revision > after, in
        order. A cursor older than the snapshot cannot be served (those
        ops are pruned) - the client must resync from the snapshot."""
        session = self._get_authoritative(session_id)
        self.require_role(session_id, principal_id, "viewer")
        if after < session.snapshot_revision:
            raise ResyncRequiredError(session.snapshot_revision)
        return [deepcopy(op) for op in session.ops if op.revision > after]

    def checkpoint(
        self,
        session_id: str,
        *,
        revision: int,
        document: object,
        principal_id: str = "local",
    ) -> DocumentSession:
        """Install a client-materialized snapshot at ``revision`` and
        prune the ops it covers. The revision must name a state the
        session actually reached and advance the existing checkpoint."""
        session = self._get_authoritative(session_id)
        self.require_role(session_id, principal_id, "editor")
        if revision <= session.snapshot_revision:
            raise ValueError(
                f"snapshot revision {revision} does not advance the"
                f" checkpoint (already at {session.snapshot_revision})"
            )
        if revision > session.revision:
            raise ValueError(
                f"snapshot revision {revision} is ahead of the session (at {session.revision})"
            )
        snapshot = self._validated_snapshot(session.document_kind, session.document_id, document)
        session.snapshot = snapshot
        session.snapshot_revision = revision
        retained = [op for op in session.ops if op.revision > revision]
        pruned = [op for op in session.ops if op.revision <= revision]
        session.ops = retained
        for op in pruned:
            del session.ops_by_id[op.op_id]
        if self._store is not None:
            self._store.checkpoint(session_id, json.dumps(session.snapshot), revision)
        return deepcopy(session)
