"""HTTP/WS surface for document sessions - thin routes over
SessionService, mounted by the host app (dinkster-serve today; the package
has no dependency on dinkster-server, so the surface can move to its own
process later without a wire change).

Surface (all JSON; scope discipline mirrors /api/history - explicit
non-empty scope, single-user mode is the reserved scope "local"):

- POST   /api/sessions                    open a session:
                                          {scope, documentId, documentKind?, snapshot}
                                          -> 201 session descriptor
- GET    /api/sessions?scope=             list one scope's sessions
- GET    /api/sessions/{sessionId}        session descriptor; 404 gone
- DELETE /api/sessions/{sessionId}        close (session_closed to
                                          subscribers, sockets closed;
                                          owner role required)
- GET    /api/sessions/{sessionId}/acl    owner-managed defaultRole and
                                          principal role entries
- PUT    /api/sessions/{sessionId}/acl    replace the full ACL; at least
                                          one owner must remain
- POST   /api/sessions/{sessionId}/actors bind {actorId} to its principal;
                                          409 actor-principal-mismatch
                                          or 429 actor-limit on refusal
- POST   /api/sessions/{sessionId}/ops    append one op (the pinned
                                          envelope: protocolVersion,
                                          opId, actorId ([A-Za-z0-9_-]+
                                          and never __proto__, joint
                                          pin - see _ACTOR_ID_RE),
                                          baseRevision,
                                          patch) -> ordered envelope
                                          with server revision +
                                          timestamp; "replayed": true
                                          on idempotent resubmission;
                                          409 stale-base {revision} on
                                          a lost race (client rebases);
                                          an actorId is bound to the
                                          authenticated principal that
                                          first binds it, and another
                                          principal receives 409
                                          actor-principal-mismatch;
                                          409 snapshot-required at the
                                          retention cap; 406 on an
                                          unsupported protocolVersion;
                                          429 rate-limited with
                                          retryAfterMs above 60 ops/sec,
                                          burst 240, per principal/kind;
                                          agents get half that budget
- GET    /api/sessions/{sessionId}/ops?after=N
                                          catch-up: retained ops with
                                          revision > N, in order; 410
                                          resync-required
                                          {snapshotRevision} when N
                                          predates the retained log
- GET    /api/sessions/{sessionId}/snapshot
                                          {revision, document} checkpoint
- PUT    /api/sessions/{sessionId}/snapshot
                                          install a client-materialized
                                          checkpoint {revision, document};
                                          prunes covered ops; 409 when
                                          the revision does not advance
                                          the checkpoint or is ahead of
                                          the session; 400 when kind
                                          validation refuses the document
- GET    /api/sessions/{sessionId}/events WebSocket: session descriptor
                                          on connect, then every ordered
                                          op ("op" envelopes) live;
                                          inbound {"type": "presence",
                                          "actorId", "payload"?} frames
                                          are relayed to the OTHER
                                          subscribers and never stored;
                                          excess frames above 100/sec,
                                          burst 200, per principal/kind
                                          are dropped; agents get half

Ops mutate through HTTP POST only - the WS is delivery plus ephemeral
presence, never an ingestion path, so ordering has exactly one door.
When mounted on an authenticated host, route-layer checks bind every
session operation to the session's scope. The standalone package keeps
its implicit local-superuser posture. Session ACL refusals are 403
{"error": "session-role"}; operations and snapshots require editor,
reads and event sockets require viewer, and close/ACL management require
owner. Legacy persisted sessions without an owner retain open editor
access, can be closed by editors, expose their ACL, and refuse ACL
replacement with 409 no-owner.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from aiohttp import WSMsgType, web

from .sessions import (
    DOCUMENT_KINDS,
    PROTOCOL_VERSION,
    ActorLimitError,
    ActorPrincipalMismatchError,
    DocumentSession,
    InvalidSnapshotError,
    NoSessionOwnerError,
    ResyncRequiredError,
    SessionOp,
    SessionRoleError,
    SessionService,
    SnapshotRequiredError,
    StaleBaseError,
    UnknownSessionError,
    validate_patch,
)
from .store import SessionStore

SESSIONS_KEY = web.AppKey("dinkster_collab_sessions", SessionService)
_SUBSCRIBER_QUEUE_SIZE = 256


class ScopePrincipal(Protocol):
    @property
    def principal_id(self) -> str: ...

    def allows_in(self, scope: str, capability: str) -> bool: ...

    def scopes_for(self, capability: str) -> frozenset[str]: ...


class _LocalPrincipal:
    @property
    def principal_id(self) -> str:
        return "local"

    def allows_in(self, scope: str, capability: str) -> bool:
        return True

    def scopes_for(self, capability: str) -> frozenset[str]:
        return frozenset({"local"})


_LOCAL_PRINCIPAL = _LocalPrincipal()
_PRINCIPAL_FOR_KEY: web.AppKey[Callable[[web.Request], ScopePrincipal]] = web.AppKey(
    "dinkster_collab_principal_for"
)
_RESOLVE_SCOPE_KEY: web.AppKey[Callable[[Any, str, object | None], str]] = web.AppKey(
    "dinkster_collab_resolve_scope"
)

_ACTOR_ID_RE = re.compile(r"[A-Za-z0-9_-]+")  # used with fullmatch
"""Joint contract pin with the frontend (2026-07-26): actorId is
``[A-Za-z0-9_-]+`` and never ``__proto__``. The frontend embeds actor
ids in document node ids (``n5-a1``) and cursor-map keys, where ``.``
is reserved for graph addressing and ``__proto__`` is a JS
prototype-pollution hazard as an object key - so the server refuses
such ids at the door rather than letting a foreign client mint
documents the frontend's validator rejects. UUIDs fit. Mirrored in
``Dinkster-Frontend/docs/promises.md``; change only together."""

_ACTOR_ID_PROBLEM = (
    "'actorId' must be a non-empty string of [A-Za-z0-9_-] and not "
    "'__proto__' (embedded in document ids and cursor-map keys by "
    "collaborating clients)"
)


@dataclass(eq=False)
class _Subscriber:
    ws: web.WebSocketResponse
    queue: asyncio.Queue[str]
    sender: asyncio.Task[None]
    principal_id: str
    authorized: Callable[[], bool]


_SUBSCRIBERS_KEY = web.AppKey("dinkster_collab_subscribers", dict[str, set[_Subscriber]])


@dataclass
class _Bucket:
    tokens: float
    updated_at: float


class _TokenBucketLimiter:
    def __init__(self, rate: float, burst: int, clock: Callable[[], float]) -> None:
        if rate <= 0 or burst < 1:
            raise ValueError("rate must be > 0 and burst must be >= 1")
        self._rate = rate
        self._burst = float(burst)
        self._clock = clock
        self._buckets: dict[tuple[str, str], _Bucket] = {}

    def consume(self, principal_id: str, actor_kind: str) -> int | None:
        now = self._clock()
        # An idle full bucket carries no state. Never evict a depleted bucket.
        self._buckets = {
            key: bucket
            for key, bucket in self._buckets.items()
            if now - bucket.updated_at < self._burst / self._rate * 2
        }
        key = (principal_id, actor_kind)
        ratio = 0.5 if actor_kind == "agent" else 1.0
        burst, rate = max(1.0, self._burst * ratio), self._rate * ratio
        bucket = self._buckets.get(key)
        if bucket is None:
            if len(self._buckets) >= 4096:
                return 1000
            bucket = _Bucket(burst, now)
            self._buckets[key] = bucket
        else:
            elapsed = max(0.0, now - bucket.updated_at)
            bucket.tokens = min(burst, bucket.tokens + elapsed * rate)
            bucket.updated_at = now
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return None
        return max(1, math.ceil((1.0 - bucket.tokens) / rate * 1000))


_OPS_LIMITER_KEY = web.AppKey("dinkster_collab_ops_limiter", _TokenBucketLimiter)
_PRESENCE_LIMITER_KEY = web.AppKey("dinkster_collab_presence_limiter", _TokenBucketLimiter)


def _error(status: int, payload: dict[str, object]) -> web.Response:
    return web.json_response(payload, status=status)


def _bad_request(message: str) -> web.HTTPBadRequest:
    return web.HTTPBadRequest(text=json.dumps({"error": message}), content_type="application/json")


def _principal(request: web.Request) -> ScopePrincipal:
    """Use host authentication when installed; standalone collab stays local."""
    return request.app[_PRINCIPAL_FOR_KEY](request)


def _local_scope(principal: ScopePrincipal, capability: str, explicit_scope: object | None) -> str:
    if explicit_scope is None:
        return "local"
    if not isinstance(explicit_scope, str) or not _valid_scope(explicit_scope):
        raise _bad_request("scope must be a non-empty whitespace-free string")
    return explicit_scope


def _local_principal_for(_request: web.Request) -> ScopePrincipal:
    return _LOCAL_PRINCIPAL


def _valid_scope(scope: str) -> bool:
    return bool(scope) and scope == scope.strip() and not any(ch.isspace() for ch in scope)


async def _json_body(request: web.Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _bad_request("request body must be JSON") from exc
    if not isinstance(body, dict):
        raise _bad_request("request body must be a JSON object")
    return cast(dict[str, Any], body)


def _session_wire(session: DocumentSession) -> dict[str, object]:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "sessionId": session.session_id,
        "scope": session.scope,
        "documentId": session.document_id,
        "documentKind": session.document_kind,
        "revision": session.revision,
        "snapshotRevision": session.snapshot_revision,
        "createdAt": session.created_at,
    }


def _op_wire(session_id: str, op: SessionOp) -> dict[str, object]:
    """The pinned envelope, exactly: protocolVersion, sessionId, opId,
    actorId, baseRevision, server-assigned revision, forward patch,
    server timestamp. Inverse patches never appear - they are client-local."""
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "sessionId": session_id,
        "opId": op.op_id,
        "actorId": op.actor_id,
        "baseRevision": op.base_revision,
        "revision": op.revision,
        "patch": [dict(entry) for entry in op.patch],
        "timestamp": op.timestamp,
    }


def _get_session(
    service: SessionService,
    request: web.Request,
    capability: str,
    minimum_role: str | None = None,
) -> DocumentSession:
    try:
        session = service.get(request.match_info["session_id"])
    except UnknownSessionError:
        raise web.HTTPNotFound(
            text=json.dumps({"error": "no such session"}),
            content_type="application/json",
        ) from None
    if not _principal(request).allows_in(session.scope, capability):
        raise web.HTTPNotFound(
            text=json.dumps({"error": "no such session"}),
            content_type="application/json",
        )
    if minimum_role is not None:
        try:
            service.require_role(session.session_id, _principal(request).principal_id, minimum_role)
        except SessionRoleError:
            raise web.HTTPForbidden(
                text=json.dumps({"error": "session-role"}),
                content_type="application/json",
            ) from None
    return session


def _can_access(
    service: SessionService, session_id: str, principal_id: str, minimum_role: str
) -> bool:
    try:
        service.require_role(session_id, principal_id, minimum_role)
    except SessionRoleError:
        return False
    return True


async def _broadcast(request: web.Request, session_id: str, payload: dict[str, object]) -> None:
    """Queue best-effort fan-out without doing socket I/O in mutation paths.

    Each socket has a bounded sender queue. A dead or overflowing subscriber
    is dropped and closed rather than delaying an append response.
    """
    subscribers = request.app[_SUBSCRIBERS_KEY].get(session_id)
    if not subscribers:
        return
    encoded = json.dumps(payload)
    for subscriber in list(subscribers):
        try:
            subscriber.queue.put_nowait(encoded)
        except asyncio.QueueFull:
            subscribers.discard(subscriber)
            subscriber.sender.cancel()
            asyncio.create_task(subscriber.ws.close())


async def _send_frames(subscriber: _Subscriber, subscribers: set[_Subscriber]) -> None:
    try:
        while True:
            if not subscriber.authorized():
                await subscriber.ws.close(code=1008, message=b"authorization-expired")
                return
            try:
                encoded = await asyncio.wait_for(subscriber.queue.get(), timeout=1)
            except TimeoutError:
                continue
            try:
                if not subscriber.authorized():
                    await subscriber.ws.close(code=1008, message=b"authorization-expired")
                    return
                await subscriber.ws.send_str(encoded)
            finally:
                subscriber.queue.task_done()
    except (asyncio.CancelledError, ConnectionError, RuntimeError):
        pass
    finally:
        subscribers.discard(subscriber)
        # Mark anything still queued as done: the close paths await
        # queue.join() to sequence session_closed before the socket
        # close, and a sender that died on a broken connection with
        # frames still queued must fail that join FORWARD (undelivered,
        # socket gone anyway), never hang it.
        while True:
            try:
                subscriber.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            subscriber.queue.task_done()


async def handle_create_session(request: web.Request) -> web.Response:
    service = request.app[SESSIONS_KEY]
    body = await _json_body(request)
    scope = request.app[_RESOLVE_SCOPE_KEY](
        _principal(request), "sessions:write", body.get("scope")
    )
    document_id = body.get("documentId")
    if not isinstance(document_id, str) or not document_id:
        raise _bad_request("'documentId' must be a non-empty string")
    document_kind = body.get("documentKind", "workflow")
    if not isinstance(document_kind, str) or document_kind not in DOCUMENT_KINDS:
        raise _bad_request(f"'documentKind' must be one of {list(DOCUMENT_KINDS)}")
    if "snapshot" not in body:
        raise _bad_request("'snapshot' is required (the document at revision 0)")
    try:
        session = service.create(
            scope=scope,
            document_id=document_id,
            document_kind=document_kind,
            snapshot=body["snapshot"],
            principal_id=_principal(request).principal_id,
        )
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc
    return web.json_response(_session_wire(session), status=201)


async def handle_list_sessions(request: web.Request) -> web.Response:
    service = request.app[SESSIONS_KEY]
    principal = _principal(request)
    explicit_scope = request.query.get("scope")
    if explicit_scope is not None and not _valid_scope(explicit_scope):
        raise _bad_request("scope must be a non-empty whitespace-free string")
    scopes = (
        [explicit_scope]
        if explicit_scope is not None and principal.allows_in(explicit_scope, "sessions:read")
        else sorted(principal.scopes_for("sessions:read"))
        if explicit_scope is None
        else []
    )
    return web.json_response(
        {
            "sessions": [
                _session_wire(session)
                for scope in scopes
                for session in service.sessions(scope)
                if _can_access(service, session.session_id, principal.principal_id, "viewer")
            ]
        }
    )


async def handle_get_session(request: web.Request) -> web.Response:
    session = _get_session(request.app[SESSIONS_KEY], request, "sessions:read", "viewer")
    return web.json_response(_session_wire(session))


async def handle_close_session(request: web.Request) -> web.Response:
    service = request.app[SESSIONS_KEY]
    session = _get_session(service, request, "sessions:write")
    try:
        service.close(session.session_id, principal_id=_principal(request).principal_id)
    except SessionRoleError:
        return _error(403, {"error": "session-role"})
    await _broadcast(request, session.session_id, {"type": "session_closed"})
    subscribers = request.app[_SUBSCRIBERS_KEY].pop(session.session_id, set())
    if subscribers:
        await asyncio.gather(*(subscriber.queue.join() for subscriber in subscribers))
    for subscriber in list(subscribers):
        subscriber.sender.cancel()
        try:
            await subscriber.ws.close()
        except (ConnectionError, RuntimeError):
            pass
    return web.json_response({"closed": True})


async def handle_bind_actor(request: web.Request) -> web.Response:
    service = request.app[SESSIONS_KEY]
    session = _get_session(service, request, "sessions:read", "viewer")
    body = await _json_body(request)
    actor_id = body.get("actorId")
    if (
        not isinstance(actor_id, str)
        or not _ACTOR_ID_RE.fullmatch(actor_id)
        or actor_id == "__proto__"
    ):
        raise _bad_request(_ACTOR_ID_PROBLEM)
    try:
        service.bind_actor(session.session_id, actor_id, _principal(request).principal_id)
    except ActorPrincipalMismatchError:
        return _error(409, {"error": "actor-principal-mismatch"})
    except ActorLimitError:
        return _error(429, {"error": "actor-limit"})
    return web.json_response({"actorId": actor_id})


async def handle_append_op(request: web.Request) -> web.Response:
    service = request.app[SESSIONS_KEY]
    session = _get_session(service, request, "sessions:write", "editor")
    body = await _json_body(request)
    protocol = body.get("protocolVersion")
    if protocol != PROTOCOL_VERSION:
        # Same posture as ?wire= on /api/nodes: an unsupported version is
        # a loud, machine-readable refusal, never a silent reinterpretation.
        return _error(
            406,
            {
                "error": "protocol-version-unsupported",
                "requested": protocol,
                "supported": [PROTOCOL_VERSION],
            },
        )
    op_id = body.get("opId")
    if not isinstance(op_id, str) or not op_id:
        raise _bad_request("'opId' must be a non-empty string")
    actor_id = body.get("actorId")
    if (
        not isinstance(actor_id, str)
        or not _ACTOR_ID_RE.fullmatch(actor_id)
        or actor_id == "__proto__"
    ):
        raise _bad_request(_ACTOR_ID_PROBLEM)
    base_revision = body.get("baseRevision")
    if not isinstance(base_revision, int) or isinstance(base_revision, bool):
        raise _bad_request("'baseRevision' must be an integer")
    patch = body.get("patch")
    problem = validate_patch(patch)
    if problem is not None:
        raise _bad_request(problem)
    try:
        service.check_actor_principal(
            session.session_id, actor_id, _principal(request).principal_id
        )
    except ActorPrincipalMismatchError:
        return _error(409, {"error": "actor-principal-mismatch"})
    except ActorLimitError:
        return _error(429, {"error": "actor-limit"})
    principal = _principal(request)
    kind = getattr(principal, "kind", "human")
    retry_after_ms = request.app[_OPS_LIMITER_KEY].consume(principal.principal_id, kind)
    if retry_after_ms is not None:
        return _error(429, {"error": "rate-limited", "retryAfterMs": retry_after_ms})
    try:
        op, replayed = service.append(
            session.session_id,
            op_id=op_id,
            actor_id=actor_id,
            base_revision=base_revision,
            patch=cast(list[dict[str, Any]], patch),
            principal_id=_principal(request).principal_id,
            actor_kind=kind,
        )
    except ActorPrincipalMismatchError:
        return _error(409, {"error": "actor-principal-mismatch"})
    except StaleBaseError as exc:
        return _error(409, {"error": "stale-base", "revision": exc.revision})
    except SnapshotRequiredError as exc:
        return _error(
            409,
            {
                "error": "snapshot-required",
                "revision": exc.revision,
                "snapshotRevision": exc.snapshot_revision,
            },
        )
    wire = _op_wire(session.session_id, op)
    if replayed:
        wire["replayed"] = True
    else:
        await _broadcast(
            request, session.session_id, {"type": "op", **_op_wire(session.session_id, op)}
        )
    return web.json_response(wire)


async def handle_ops_after(request: web.Request) -> web.Response:
    service = request.app[SESSIONS_KEY]
    session = _get_session(service, request, "sessions:read", "viewer")
    raw_after = request.query.get("after")
    if raw_after is None:
        raise _bad_request("'after' is required (revision cursor, integer >= 0)")
    try:
        after = int(raw_after)
    except ValueError:
        raise _bad_request("'after' must be an integer revision cursor") from None
    if after < 0:
        raise _bad_request("'after' must be >= 0")
    try:
        ops = service.ops_after(
            session.session_id, after, principal_id=_principal(request).principal_id
        )
    except ResyncRequiredError as exc:
        return _error(
            410,
            {"error": "resync-required", "snapshotRevision": exc.snapshot_revision},
        )
    return web.json_response(
        {
            "protocolVersion": PROTOCOL_VERSION,
            "sessionId": session.session_id,
            "revision": session.revision,
            "snapshotRevision": session.snapshot_revision,
            "ops": [_op_wire(session.session_id, op) for op in ops],
        }
    )


async def handle_get_snapshot(request: web.Request) -> web.Response:
    session = _get_session(request.app[SESSIONS_KEY], request, "sessions:read", "viewer")
    return web.json_response({"revision": session.snapshot_revision, "document": session.snapshot})


async def handle_put_snapshot(request: web.Request) -> web.Response:
    service = request.app[SESSIONS_KEY]
    session = _get_session(service, request, "sessions:write", "editor")
    body = await _json_body(request)
    revision = body.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool):
        raise _bad_request("'revision' must be an integer")
    if "document" not in body:
        raise _bad_request("'document' is required (the materialized state)")
    try:
        service.checkpoint(
            session.session_id,
            revision=revision,
            document=body["document"],
            principal_id=_principal(request).principal_id,
        )
    except InvalidSnapshotError as exc:
        return _error(400, {"error": "document-invalid", "message": str(exc)})
    except ValueError as exc:
        return _error(409, {"error": "snapshot-invalid", "message": str(exc)})
    session = service.get(session.session_id)
    return web.json_response(
        {"snapshotRevision": session.snapshot_revision, "revision": session.revision}
    )


async def handle_session_events(request: web.Request) -> web.WebSocketResponse:
    service = request.app[SESSIONS_KEY]
    session = _get_session(service, request, "sessions:read", "viewer")
    principal_id = _principal(request).principal_id
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    # Capture before registration. Appends may run while ws.prepare
    # yields; the replay below closes that window before live delivery.
    captured_revision = session.revision
    # Registration, the descriptor frame, the gap-free replay, and the
    # closed-in-the-window check form ONE synchronous block - no awaits -
    # so a concurrent DELETE is strictly ordered against it: running
    # AFTER, its broadcast finds this subscriber registered and
    # handle_close_session drains the queue before closing the socket;
    # running BEFORE, the replay finds the session gone and this handler
    # synthesizes the terminal frame the broadcast could not have
    # reached it with. session_closed can no longer fall between (the
    # lossy-DELETE race the frontend caught live, 2026-07-26: the old
    # order sent the descriptor first, so a DELETE landing during that
    # send's await broadcast to a not-yet-registered subscriber, whose
    # late replay then died on UnknownSessionError - socket closed,
    # frame never delivered).
    subscribers = request.app[_SUBSCRIBERS_KEY].setdefault(session.session_id, set())
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_SIZE)
    placeholder = asyncio.create_task(asyncio.sleep(0))
    subscriber = _Subscriber(
        ws=ws,
        queue=queue,
        sender=placeholder,
        principal_id=principal_id,
        authorized=lambda: _principal(request).allows_in(session.scope, "sessions:read"),
    )
    subscriber.sender = asyncio.create_task(_send_frames(subscriber, subscribers))
    subscribers.add(subscriber)
    queue.put_nowait(json.dumps({"type": "session", **_session_wire(session)}))
    closed_in_window = False
    try:
        try:
            for op in service.ops_after(
                session.session_id, captured_revision, principal_id=principal_id
            ):
                queue.put_nowait(json.dumps({"type": "op", **_op_wire(session.session_id, op)}))
        except UnknownSessionError:
            queue.put_nowait(json.dumps({"type": "session_closed"}))
            closed_in_window = True
        if closed_in_window:
            # Deliver descriptor + terminal frame, then close: the
            # DELETE that won the race popped this session's subscriber
            # set before this handler registered, so nobody else will.
            await queue.join()
            try:
                await ws.close()
            except (ConnectionError, RuntimeError):
                pass
            return ws
        async for message in ws:
            if message.type != WSMsgType.TEXT:
                continue
            try:
                decoded: object = json.loads(message.data)
            except json.JSONDecodeError:
                continue
            if not isinstance(decoded, dict):
                continue
            frame = cast(dict[str, object], decoded)
            if frame.get("type") != "presence":
                continue  # ops enter via HTTP only; unknown frames are noise
            actor_id = frame.get("actorId")
            if (
                not isinstance(actor_id, str)
                or not _ACTOR_ID_RE.fullmatch(actor_id)
                or actor_id == "__proto__"
            ):
                # Same actorId pin as the ops door. A WS frame has no
                # status code to refuse with, and presence is ephemeral
                # noise-tolerant fan-out - so out-of-charset ids drop
                # like any other malformed frame, never relayed.
                continue
            try:
                service.require_role(session.session_id, principal_id, "viewer")
                service.check_actor_principal(session.session_id, actor_id, principal_id)
            except (ActorPrincipalMismatchError, ActorLimitError):
                continue
            except (SessionRoleError, UnknownSessionError):
                continue
            principal = _principal(request)
            if not principal.allows_in(session.scope, "sessions:read"):
                await ws.close(code=1008)
                break
            kind = getattr(principal, "kind", "human")
            if request.app[_PRESENCE_LIMITER_KEY].consume(principal_id, kind) is not None:
                continue
            try:
                service.bind_actor(session.session_id, actor_id, principal_id)
            except (ActorPrincipalMismatchError, SessionRoleError, UnknownSessionError):
                continue
            relay: dict[str, object] = {"type": "presence", "actorId": actor_id}
            if "payload" in frame:
                payload = frame["payload"]
                if isinstance(payload, dict) and (
                    "identity" in payload or cast("dict[str, object]", payload).get("v") == 1
                ):
                    payload = dict(cast(dict[str, object], payload))
                    identity = payload.get("identity")
                    identity = (
                        dict(cast(dict[str, object], identity))
                        if isinstance(identity, dict)
                        else {}
                    )
                    identity.update(kind=kind, owner=principal_id)
                    name = getattr(principal, "display_name", None)
                    if name is not None:
                        identity["displayName"] = name
                    payload["identity"] = identity
                relay["payload"] = payload
            encoded = json.dumps(relay)
            for peer in list(subscribers):
                if peer is subscriber:
                    continue  # ephemeral fan-out to OTHERS, never echoed
                try:
                    peer.queue.put_nowait(encoded)
                except asyncio.QueueFull:
                    subscribers.discard(peer)
                    peer.sender.cancel()
                    asyncio.create_task(peer.ws.close())
    finally:
        subscribers.discard(subscriber)
        subscriber.sender.cancel()
        if not subscribers:
            # A subscriber that lost the DELETE race recreated this
            # session's map entry after the pop; registration is atomic
            # with set membership, so an empty set has no other holder -
            # drop it rather than accumulating one per dead session.
            request.app[_SUBSCRIBERS_KEY].pop(session.session_id, None)
    return ws


async def handle_get_acl(request: web.Request) -> web.Response:
    service = request.app[SESSIONS_KEY]
    session = _get_session(service, request, "sessions:read")
    default_role, entries = service.acl_policy(session.session_id)
    if "owner" in entries.values() and not _can_access(
        service, session.session_id, _principal(request).principal_id, "owner"
    ):
        return _error(403, {"error": "session-role"})
    return web.json_response({"defaultRole": default_role, "entries": entries})


async def handle_put_acl(request: web.Request) -> web.Response:
    service = request.app[SESSIONS_KEY]
    session = _get_session(service, request, "sessions:write")
    _default_role, current_entries = service.acl_policy(session.session_id)
    if "owner" not in current_entries.values():
        return _error(409, {"error": "no-owner"})
    if not _can_access(service, session.session_id, _principal(request).principal_id, "owner"):
        return _error(403, {"error": "session-role"})
    body = await _json_body(request)
    default_role = body.get("defaultRole")
    entries = body.get("entries")
    if not isinstance(default_role, str):
        raise _bad_request("'defaultRole' must be a string")
    if not isinstance(entries, dict):
        raise _bad_request("'entries' must be an object of principal ids to roles")
    raw_entries = cast(dict[object, object], entries)
    if any(
        not isinstance(key, str) or not isinstance(role, str) for key, role in raw_entries.items()
    ):
        raise _bad_request("'entries' must be an object of principal ids to roles")
    try:
        updated = service.replace_acl(
            session.session_id,
            principal_id=_principal(request).principal_id,
            default_role=default_role,
            entries=cast(dict[str, str], raw_entries),
        )
    except NoSessionOwnerError:
        return _error(409, {"error": "no-owner"})
    except SessionRoleError:
        return _error(403, {"error": "session-role"})
    except ValueError as exc:
        return _error(409, {"error": "acl-invalid", "message": str(exc)})
    subscribers = request.app[_SUBSCRIBERS_KEY].get(session.session_id, set())
    for subscriber in list(subscribers):
        if not _can_access(service, session.session_id, subscriber.principal_id, "viewer"):
            subscribers.discard(subscriber)
            subscriber.sender.cancel()
            asyncio.create_task(subscriber.ws.close())
    return web.json_response({"defaultRole": updated.default_role, "entries": updated.acl})


def add_session_routes(
    app: web.Application,
    service: SessionService,
    *,
    principal_for: Callable[[web.Request], ScopePrincipal] | None = None,
    resolve_scope: Callable[[Any, str, object | None], str] | None = None,
    op_rate: float = 60.0,
    op_burst: int = 240,
    presence_rate: float = 100.0,
    presence_burst: int = 200,
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> None:
    """Mount the document-session surface onto a host aiohttp app."""
    if (principal_for is None) != (resolve_scope is None):
        raise ValueError("principal_for and resolve_scope must be provided together")
    app[SESSIONS_KEY] = service
    app[_PRINCIPAL_FOR_KEY] = principal_for or _local_principal_for
    app[_RESOLVE_SCOPE_KEY] = resolve_scope or _local_scope
    app[_SUBSCRIBERS_KEY] = {}
    app[_OPS_LIMITER_KEY] = _TokenBucketLimiter(op_rate, op_burst, monotonic_clock)
    app[_PRESENCE_LIMITER_KEY] = _TokenBucketLimiter(presence_rate, presence_burst, monotonic_clock)
    app.router.add_post("/api/sessions", handle_create_session)
    app.router.add_get("/api/sessions", handle_list_sessions)
    app.router.add_get("/api/sessions/{session_id}", handle_get_session)
    app.router.add_delete("/api/sessions/{session_id}", handle_close_session)
    app.router.add_get("/api/sessions/{session_id}/acl", handle_get_acl)
    app.router.add_put("/api/sessions/{session_id}/acl", handle_put_acl)
    app.router.add_post("/api/sessions/{session_id}/actors", handle_bind_actor)
    app.router.add_post("/api/sessions/{session_id}/ops", handle_append_op)
    app.router.add_get("/api/sessions/{session_id}/ops", handle_ops_after)
    app.router.add_get("/api/sessions/{session_id}/snapshot", handle_get_snapshot)
    app.router.add_put("/api/sessions/{session_id}/snapshot", handle_put_snapshot)
    app.router.add_get("/api/sessions/{session_id}/events", handle_session_events)


def install_session_extension(
    app: web.Application,
    *,
    database: Path | None,
    snapshot_validator: Callable[[str, str, object], str | None],
    principal_for: Callable[[web.Request], ScopePrincipal],
    resolve_scope: Callable[[Any, str, object | None], str],
) -> None:
    """Install collaboration state, routes, and persistence cleanup on a host app."""
    store = SessionStore(database) if database is not None else None
    try:
        service = SessionService(store=store, snapshot_validator=snapshot_validator)
        add_session_routes(
            app,
            service,
            principal_for=principal_for,
            resolve_scope=resolve_scope,
        )
    except BaseException:
        if store is not None:
            store.close()
        raise

    if store is not None:

        async def close_collaborative_sessions(_: web.Application) -> None:
            await asyncio.to_thread(store.close)

        app.on_cleanup.append(close_collaborative_sessions)
