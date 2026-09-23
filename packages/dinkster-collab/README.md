# dinkster-collab

Server-ordered collaborative document sessions: the platform plan's
collaboration surface, v1. A separate additive surface beside the job
queue - it shares an aiohttp app with the rest of the API today and can
move to its own process later without a wire change, because nothing in
it touches the engine, queue, or any other Dinkster package.

## What the server does (and refuses to do)

A session is an append-only log of forward JSON patches over one
document, plus one snapshot checkpoint. The server:

- **orders** ops: server-assigned contiguous revisions, optimistic
  concurrency - an op names the `baseRevision` it was built on, and a
  lost race answers 409 `stale-base` with the current revision so the
  client rebases and resubmits;
- **replays** ops: `GET .../ops?after=N` returns everything after a
  revision cursor, so reconnects catch up instead of guessing;
- **bounds** the log: one client-materialized snapshot checkpoint per
  session prunes the ops it covers; at the retention cap appends answer
  409 `snapshot-required` until a checkpoint lands.

It never applies or interprets a patch - materializing documents is the
client's job, so the server cannot corrupt what it does not evaluate.
Inverse patches (undo) stay client-local, and undo itself has no
server-side existence: a client performing undo submits the reverting
change as an ordinary forward op (new opId, current baseRevision),
which the server orders, retains, and replays like any other - an op
whose patch semantically reverts an earlier op is just another op.
Presence is ephemeral WS relay, never stored. Execution never
consumes live session state: jobs are submitted from immutable
document snapshots via `/api/jobs`.

## Setup

Install the package directly or with the `dinkster[collab]` extra. A default
Dinkster install includes it. `dinkster-serve` asks the package to register its
server extension when installed and otherwise serves without collaboration.
To mount the lower-level routes on any aiohttp app:

```python
from dinkster_collab import SessionService, add_session_routes

add_session_routes(app, SessionService())
```

## Surface

All JSON. Scope discipline mirrors `/api/history`: explicit non-empty
scope, single-user mode is the reserved scope `"local"`.

| Route | What |
|-------|------|
| `POST /api/sessions` | open: `{scope, documentId, documentKind?, snapshot}` -> 201 descriptor |
| `GET /api/sessions?scope=` | list one scope's sessions |
| `GET /api/sessions/{id}` | descriptor; 404 once closed |
| `DELETE /api/sessions/{id}` | close; every registered subscriber gets `session_closed` before its socket closes (best-effort against transport failure only) |
| `POST /api/sessions/{id}/ops` | append one op envelope |
| `GET /api/sessions/{id}/ops?after=N` | catch-up; 410 `resync-required` past the retained log |
| `GET /api/sessions/{id}/snapshot` | `{revision, document}` |
| `PUT /api/sessions/{id}/snapshot` | install checkpoint `{revision, document}`; prunes covered ops |
| `GET /api/sessions/{id}/events` | WS: descriptor, then live `op` envelopes; inbound `presence` frames relayed to others |

The op envelope (pinned jointly with the frontend):

```json
{
  "protocolVersion": 1,
  "sessionId": "...",
  "opId": "client-unique",
  "actorId": "who",
  "baseRevision": 4,
  "revision": 5,
  "patch": [{"op": "replace", "path": ["meta", "title"], "value": "x"}],
  "timestamp": 1789.0
}
```

Patch ops are `add | remove | replace` only, matching the frontend
command layer's output exactly - the RFC 6902 `move`/`copy`/`test`
verbs are rejected because no client produces them. `path` is an
ARRAY of string|int segments (the frontend's native DocPath shape),
not a JSON Pointer string: zero conversion at the client boundary and
no `~0`/`~1` escaping footgun. The server shape-checks the envelope
but never resolves paths or applies values.

`actorId` is `[A-Za-z0-9_-]+` and never `__proto__` (joint pin with the
frontend, 2026-07-26): collaborating clients embed actor ids in document
node ids (`n5-<actorId>`) and cursor-map keys, where `.` is reserved for
graph addressing and `__proto__` is a JS prototype-pollution hazard as
an object key, so the server refuses such ids - 400 on ops, frame
dropped on presence. UUIDs fit.

`revision` and `timestamp` are server-assigned. Resubmitting a retained
`opId` returns the recorded envelope with `"replayed": true` instead of
appending twice (idempotency window = retention window). An unsupported
`protocolVersion` answers 406 with the supported list - the version
rides operation and session envelopes so OT/CRDT ordering can replace v1's
server-ordered-optimistic contract later without a new surface.

Ops mutate through HTTP POST only; the WS is delivery plus presence,
never an ingestion path - ordering has exactly one door.

`documentKind` is `workflow` or `image` and defaults to `workflow` for
older clients and persisted sessions. Descriptors always include the kind.
The host may inject a whole-snapshot validator into `SessionService`;
`dinkster-serve` uses that seam to strictly validate ImageDocument snapshots
and lineage at session creation and checkpoint installation. Patch ordering
remains document-semantic-free.

## Durability

`SessionService` accepts an optional `SessionStore` (SQLite, WAL). With a
store, open sessions survive a server restart: snapshots, retained ops,
document kind, actor-to-principal bindings, session ACLs, and session policy are
reconstructed on startup, closed sessions are deleted, and checkpoint
installation prunes the covered rows. Without a store the service runs
in-memory, which remains the mode for ephemeral hosts and tests.

## Identity and access

When the server runs with authentication, every session actor is bound to
an authenticated principal: an actor id may not be reused by a different
principal, and ops or presence from a mismatched principal are refused.
Sessions carry an ACL (roles: banned, viewer, editor, owner); appending
ops requires editor, closing requires owner (editor suffices only for
legacy sessions persisted without an owner entry). Principal-keyed token-bucket
rate limits bound op and presence throughput. Delegated agents are
additionally gated by the user's permission categories in `dinkster_server.auth`
(`edit`, `execute`, `read`, `assets`, `settings`, `queue`), each
toggleable per principal; humans are never gated by categories.

### Delegated agents

An authenticated human mints a credential with `POST /api/auth/delegations`:
`{scope, displayName, sessionId?, expiresInSeconds?}`. Its principal is the user,
its kind is always `agent`, and its grants cannot exceed the user's grants.
The optional session restriction permits only that session's routes and
WebSocket ticket minting; omit it for scope-wide job submission. Session roles
remain authoritative on every session request. Toggle changes apply immediately,
including to connected subscribers. `GET /api/principals` includes the current
JWT or static principal; each human can edit their own agent toggles.

Tokens are opaque random secrets, stored only as SHA-256 hashes in SQLite
alongside principal permissions (`<library-root>/principals.sqlite`). Records
retain the user and agent principal ids, scope, session restriction, creation
time, optional expiry and revocation time. Permission toggles are normalized
by owning principal, not copied at mint. Records survive restart. There is no
default expiry or parent-JWT expiry cap; `expiresInSeconds` selects an optional
expiry. Transactional caps of 32 per user and 4096 overall count the persisted
unrevoked set, including expired records; revoke a record to free a slot.
`GET /api/auth/delegations` lists the caller's unrevoked records without secrets;
`DELETE /api/auth/delegations/{id}` revokes one. Agents cannot delegate or change
permission toggles. The UI and CLI must never pass the user's full JWT to an agent.
The identity verifier is verification-only; no identity signing key is required.

Durability does not grant unattended access. A delegation is usable only while
the server has recently verified a human JWT for its owner. Each successful
JWT authentication updates that user's in-memory monotonic timestamp and role
ceiling. The default freshness window is 600 seconds, configurable with
`--user-session-freshness-seconds`. Agent requests and static credentials cannot
refresh it. Freshness starts empty after restart: the same delegation works
again when the user authenticates, without re-minting. Role grants come from
the most recently verified JWT, never a mint-time snapshot. Identity changes
become visible through subsequent JWTs or window expiry, not identity polling.
Explicitly disabling persistence with `--library-root ''` also makes these
records ephemeral, like session records. Auth-off local agents need no JWT.

Send the token as HTTP Bearer authorization. Mint a fresh single-use
`POST /api/auth/ws-ticket` for every WebSocket connection; the URL contains only
that ticket. Auth-off agents need no credential: `X-Dinkster-Actor-Kind: agent`
selects the local principal's agent budget and attribution. With auth enabled,
kind comes only from the credential; a human credential cannot impersonate an
agent with this header. Presence identity kind and owner are server-stamped.
Stored ops include principal and kind without changing the op envelope. Jobs
and durable run history retain principal/kind and the submitting `clientId`;
agent job callers use their actor id as that client id.

Both session sockets and `/api/events` recheck credential expiry, delegation
revocation, user-session freshness and permission toggles before sending and at one-second intervals
while idle. Withdrawal closes the socket with code 1008 and reason
`authorization-expired`; queued events are not delivered with withdrawn rights.
Missing user freshness instead closes with code 1008 and reason
`user-session-required`. HTTP returns 403 `{"error":"user-session-required"}`.
This suspension is transient: the next verified user JWT makes the same
unrevoked delegation usable again.

`POST /api/sessions/{id}/actors` binds `{actorId}` before editing. Another
principal receives 409 `actor-principal-mismatch`; the browser generates one
fresh id and retries once. Ownership persists with the session, including after
restart. Bindings have hard caps of 256 per principal and 4096 per session;
they cannot expire safely because stale ids must not become claimable. A cap
returns 429 `actor-limit` instead of growing storage without bound.

Op budgets are keyed by principal across all sessions, not by actor id. The
principal has 60 ops/second with burst 240; agent traffic also passes through a
half-size sub-budget, reserving the other half for the human without increasing
the principal's total allowance. Presence budgets use the same hierarchy. Idle
full buckets are discarded, with a 4096-bucket cap per level. Rotating ids cannot
bypass either budget.

The server keeps its existing HTTP error bodies (`error` plus route-specific
fields), including op/bind 409, op 429 and authorization refusals on session,
snapshot, ops and ticket routes. It does not emit `collab.denial` objects.
The frontend `@dinkster/client` normalizes these responses into diagnostics with
`version: 1`, `type: "collab.denial"`, HTTP status, code, message, sessionId and
actorId when known, opId for submissions or operation for transport refusals, and optional
retryAfterMs. 403, actor ownership 409, and 429 terminate submission instead
of retrying indefinitely; read/ticket 401/403 stop probes and reconnects too.
The Problems panel shows the denial and agent `settle()` rejects with it.
The unacknowledged local document remains available to recover. This client
normalization does not change server error bodies or the operation protocol.

Presence may include `activity: {v: 1, type: "agent_tool_call", tool, status,
pendingAsks: [{id, prompt}]}`. Status is `running`, `success`, or `error`.
The frontend bounds tool text to 128 characters, asks to 8, ids to 128 and
prompts to 280; malformed activity is ignored. This is ephemeral display state,
never authorization or an operation. The op envelope, revisions and
`protocolVersion: 1` are unchanged.

## Deliberate limits (pinned in ROADMAP)

- no server-side patch application, and no OT/CRDT - this surface is
  server-ordered optimistic concurrency by design. Clients rebase;
  the server never transforms.

## Future direction pins

These are constraints on future changes, not features of this package.

- **New editor kinds share the same ordering surface.** The descriptor kind
  separates discovery and permits host-owned validation of complete snapshots
  at creation and checkpoint boundaries. The server still never interprets
  patches: command families remain client-side, with the same op envelope,
  routes, and `protocolVersion`.
- **Heavy media never rides ops or presence.** Documents and commands
  reference assets by content digest (the blake3 identity in
  `dinkster-assets`); media bytes travel the asset routes out of band. The
  op log stays small enough to retain, replay, and persist.
- **Session hosting stays a movable seam.** `SessionService` is
  semantics-blind and self-contained, so the ordering role can move to
  another process or another peer without a wire change. Ops are the
  replication unit. Nothing may hardwire one permanent server as the only
  possible orderer: peer-hosted sessions (any peer running the orderer,
  with host migration) and short offline windows (local command queue,
  rebase on reconnect) must stay reachable. Long-divergence symmetric
  merge (true CRDT) is out of scope unless ruled otherwise.
