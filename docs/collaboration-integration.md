# Collaboration integration

## Hosts and clients

Install `dinkster[collab]` or the collaboration package for the composed server.
For an embedding aiohttp host, use `add_session_routes(app, SessionService())`;
select a `SessionStore` when sessions must survive restart. See
[session service](../packages/dinkster-collab/src/dinkster_collab/sessions.py),
[routes](../packages/dinkster-collab/src/dinkster_collab/routes.py) and
[storage](../packages/dinkster-collab/src/dinkster_collab/store.py) for the API.

Clients materialize documents and retain inverse patches locally. Submit undo
as a new forward operation with a new opId and current baseRevision. Rebase
and resubmit after a stale base; post a client-materialized checkpoint to free
retained-log capacity. The server never transforms or applies patches.
Keep actor ids safe for document node ids and JavaScript cursor-map keys:
`.` is graph addressing syntax and `__proto__` is unsafe as an object key.

## Delegated agents

The UI and CLI must never pass the user's full JWT to an agent. Mint a scoped
delegation instead, with a session restriction when appropriate. Keep sending
verified human requests while the user is signed in; durable credentials do
not grant unattended access. See [delegations and freshness](auth.md#durable-delegations-and-user-sessions).
Mint a fresh single-use [WebSocket ticket](auth.md#browser-websocket-tickets)
for every connection; never put a long-lived credential in the URL.
Agent job callers use their actor id as `clientId` for attribution.

After an actor-binding ownership conflict, the browser generates one fresh id
and retries once. Do not retry authorization, ownership or rate-limit failures
indefinitely. The frontend `@dinkster/client` normalizes HTTP refusals into
`collab.denial` diagnostics with version 1, HTTP status, code, message,
session/actor ids when known, opId for submissions or operation for transport
refusals, and optional retryAfterMs. Submission 403, actor ownership 409 and
429 stop submission; read/ticket 401/403 also stop probes and reconnects.
The Problems panel displays the denial and agent `settle()` rejects with it.
Keep the unacknowledged local document available for recovery. This is client
normalization, not a server error-body or operation-protocol change.

Presence activity is ephemeral display state, never authorization or an op.
For `activity: {v: 1, type: "agent_tool_call", tool, status, pendingAsks}`,
status is `running`, `success` or `error`; asks contain `id` and `prompt`.
The frontend limits tool text and ask ids to 128 characters, asks to 8 and
prompts to 280 characters, ignoring malformed activity. Client integrators
must retain those display bounds independently of the server's operation caps.

## Architecture constraints

New editor kinds share the same ordering surface; the host validates complete
snapshots at creation and checkpoint boundaries while commands stay client-side.
Heavy media must not ride operations or presence: reference assets by content
digest and transfer bytes out of band so the log stays small enough to retain,
replay and persist.

Keep session hosting movable without a wire change. Do not hardwire one
permanent server as the only orderer: peer-hosted ordering with host migration
and short offline command queues with rebase on reconnect must remain
reachable. These are constraints, not implemented peer/offline features.
Long-divergence symmetric CRDT merge is out of scope without a separate ruling.
