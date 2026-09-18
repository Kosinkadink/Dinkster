# Inbound authentication and capabilities

Inbound authentication is off by default. Without static or identity-service
authentication configuration, every request receives an implicit `local`
principal with the `local` scope and all capabilities. Existing local
deployments therefore retain their prior HTTP and WebSocket behavior.

## Static credentials

Pass `--auth PATH` to accept v1 static Bearer credentials. The conventional
operator-managed location is `<library-root>/auth.toml`, beside `memory.toml`
and `settings.json`, but the file is loaded only when explicitly named. It is
not a runtime setting and neither tokens nor the file contents appear in
`/api/settings`, API responses, or logs. Missing, malformed, or semantically
invalid files refuse startup.

## File format

The file is strict TOML with version 1 and one or more token entries:

```toml
version = 1

[[tokens]]
token = "replace-with-an-operator-generated-secret"
principalId = "alice"
[tokens.grants]
personal = ["jobs:submit", "jobs:read", "jobs:cancel", "assets:read"]
team = ["history:read", "sessions:read", "sessions:write"]
```

Top-level keys, token-entry keys, and capability names are closed. Tokens use
the standard ASCII Bearer token character set; principal ids must be non-empty
and trimmed; scope names must be non-empty and whitespace-free. Duplicate
tokens, duplicate capabilities, empty token arrays, and unknown capabilities
are startup errors.

## Identity-service tokens

Configure all three identity values together:

```sh
dinkster-serve \
  --identity-jwks-url https://identity.example/.well-known/jwks.json \
  --identity-issuer https://identity.example \
  --identity-audience dinkster-session
```

The equivalent environment variables are `DINKSTER_IDENTITY_JWKS_URL`,
`DINKSTER_IDENTITY_ISSUER`, and `DINKSTER_IDENTITY_AUDIENCE`. The JWKS URL ends in
`/.well-known/jwks.json`; the issuer is the identity service's
`PUBLIC_BASE_URL` without a trailing slash; and the audience exactly matches
the token mint request. Missing one of the three values refuses startup.

Clients mint a short-lived credential with their identity session cookie, then
present the returned access token to Dinkster:

```http
POST /v1/tokens
Content-Type: application/json
Cookie: <identity session>

{"audience":"dinkster-session","contexts":["project:019..."]}
```

The identity service compiles current roles into the signed `grants` claim.
Dinkster verifies the token through one long-lived `TokenVerifier` and passes the
verified principal id, immutable grants, and principal kind directly into its
normal capability and agent-permission checks. Dinkster never derives grants from
organization, billing-account, project, or local role data.

When `--auth` and identity-service configuration are both present, static
credentials are tried first and identity-service JWTs second. A failed static
lookup falls through to JWT verification. The shared identity verifier keeps
its JWKS cache, refresh lock, unknown-key throttle, and stale-while-error state
across requests.

Send the credential as `Authorization: Bearer <token>`. With auth enabled,
every `/api` route except `/api/health` requires a valid Bearer credential. The
`/memory`, `/cache`, and peer `/assets` coordination routes are protected too.
The `/api/events` and session-events WebSockets are authenticated and
capability-checked before the HTTP upgrade, so a rejected handshake is an HTTP
401 or 403 rather than a socket that opens and then closes.

Authentication failures return:

```json
{
  "error": "authentication-required",
  "message": "a valid Bearer credential is required"
}
```

with status 401 and `WWW-Authenticate: Bearer`. Authorization failures return:

```json
{
  "error": "capability-required",
  "capability": "jobs:read",
  "message": "this route requires jobs:read"
}
```

with status 403. Neither shape contains a credential.

A matched route omitted from the capability matrix fails closed with status
403, `error: "authorization-policy-missing"`, and no handler execution. Normal
router-generated 404 and 405 responses remain available after authentication.

## Capability route matrix

| Capability | Routes |
| --- | --- |
| `jobs:submit` | `POST /api/jobs`; `POST /api/compat/comfy/prompt` |
| `jobs:read` | job list/status/event replay, `/api/values`, and `/api/events` |
| `jobs:cancel` | job `DELETE` routes, keyed or by `jobRef` |
| `history:read` | all `/api/history` routes, including delete; v1 has no separate history mutation verb |
| `assets:read` | asset/library/mount reads, `POST /api/assets/guess`, and peer `/assets` reads |
| `assets:write` | asset upload, library/mount/source mutations |
| `settings:read` | settings reads, `GET /api/p2p/status`, and `GET /api/install/activation` |
| `settings:write` | settings writes, P2P transfer actions, install activation, and dev pack reload/removal |
| `sessions:read` | session GET and session WebSocket routes |
| `sessions:write` | session create, op, snapshot, and close routes |
| `queue:control` | queue status, pause, resume, and clear |
| `memory:read` | all `/memory` routes, including shed and lease mutation; v1 has no separate memory control verb |
| `cache:read` | all `/cache` routes, including trim; v1 has no separate cache control verb |

Catalog GETs require authentication but no capability: `/api/nodes` (including
`?wire=`), `/api/workers`, `/api/extensions/snapshot`, choices, composition,
templates, pack icons/blueprints/template bodies, and diagnostics.
`/api/health` remains unauthenticated for ingress probes.

`settings:write` composes with `--allow-settings-changes`: the capability says
which principals may attempt a write, while the CLI grant says which categories
the operator made writable at all. A principal without `settings:write` gets
the RBAC 403. A principal with it still sees the existing
`settings-changes-disabled` response and `writable: false` metadata for an
ungranted category.

## Resource scope binding

A principal carries an immutable mapping from scope to capability set. The
coarse middleware admits a route when any scope grants its capability; resource
handlers then enforce the selected or stamped scope.

`POST /api/jobs` accepts optional body field `scope` and the ComfyUI-compatible
prompt route accepts optional header `X-Dinkster-Scope`. Scope names are non-empty
and whitespace-free. An explicit scope must grant `jobs:submit`; otherwise the
server returns the normal `capability-required` 403 with additive `scope`. When
scope is omitted, exactly one granting scope is selected automatically. More
than one returns status 400 with `error: "scope-required"`, the capability, and
the candidate `scopes`. Accepted jobs are stamped with scope and principal id;
job responses always include `scope`, and durable history includes both `scope`
and `principalId`.

Job lists are the union of scopes granting `jobs:read`. Status, event replay,
values, and cancellation hide jobs outside the principal's readable scope with
the existing no-such-job 404 shape. Cancellation additionally requires
`jobs:cancel` in that job's scope and uses the same 404 posture. Job-attributable
WebSocket events are filtered by that rule; instance events still reach every
principal admitted to the stream.

History retains its `?scope=` filter. With auth enabled, omission selects the
sole `history:read` scope and is a `scope-required` 400 when several qualify. An
explicit unauthorized scope behaves like an unknown scope: lists are empty and
single-record routes return the existing 404. Auth-off history keeps the
existing explicit `scope=local` contract.

Session creation accepts optional `scope` with the same unique-or-explicit rule
for `sessions:write`. Lists return the union of `sessions:read` scopes. Every
per-session read or write checks the stored session scope and hides unauthorized
sessions with the existing no-such-session 404.

## Browser WebSocket tickets

Browsers cannot attach `Authorization` to a WebSocket upgrade. An authenticated
client can therefore `POST /api/auth/ws-ticket`; this route requires identity
but no capability and returns:

```json
{"ticket": "opaque-single-use-value", "expiresInSeconds": 30}
```

The opaque ticket is held only in process, expires against a monotonic clock,
and is consumed at the first upgrade attempt. Only `/api/events` and
`/api/sessions/{sessionId}/events` accept it as `?ticket=`. Unknown, expired, or
reused tickets receive the standard 401 before upgrade. Plain HTTP routes never
accept tickets. Bearer header authentication remains supported for non-browser
WebSocket clients. Long-lived Bearer credentials must never be placed in URLs;
neither tickets nor Bearer values are echoed in errors or application logs.

The server consumes an injected authenticator protocol and stores the resolved
Principal only on the aiohttp request. There is no process-global current-user
state.
