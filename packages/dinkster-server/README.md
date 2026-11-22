# dinkster-server

`dinkster-server` provides Dinkster's aiohttp HTTP and WebSocket surface, job
queue policy, event fanout, discovery, peer clients, and optional persistent
library and history stores. The umbrella `dinkster-serve` entry point composes
these pieces with an engine, caches, assets, memory governance, and packs.

## Setup

This package is a uv workspace member and is not published separately yet.
From the repository root, install the complete workspace:

```sh
uv sync --all-packages
```

The package has no console script of its own. Run the composed server from
the umbrella package:

```sh
uv run dinkster-serve
```

After installation and before the first serve, run
`uv run dinkster-pack prepare-catalogs --defaults`. Rerun it after changing the
checkout or any default-pack dependency. The server refuses to bind when a
required catalog is missing or stale.

## Use

`dinkster_server.create_app` builds the native protocol application, while
`start_server` manages binding and instance discovery. Public exports also
include `JobQueue`, `EventHub`, `ServerLibrary`, `HistoryStore`,
`PeerCacheStore`, `PeerClient`, `StaticBearerAuthenticator`,
`TokenAuthenticator`, and `CompositeAuthenticator`.

`HistoryStore(path, accepted_path=engine_path)` keeps accepted work in an
explicit engine-owned database while terminal history stays shared. Recovery
sweeps only that accepted database. Terminal history commits before accepted
work is retired, so retry preserves the first terminal record after a crash
between commits. Omitting `accepted_path` retains the single-file transaction.
Separate accepted storage refuses the legacy collection barrier, which cannot
account for another engine's accepted work. Ordinary server startup does not
select separate accepted storage.

`dinkster_server.pack_settings.PackSettingsStore` and
`create_app(pack_settings_root=...)` accept a caller-owned `LibraryClient` instead
of a directory to opt into helper-backed pack settings.
Reads fetch the current pack record and project declared fields, supplying schema
defaults for absent fields. Writes validate the complete declared object, then
patch only changed fields against the last fetched record or successful write
revision. A stored value rejected by this schema still supplies a revision, so
a valid complete PUT can repair it, including after GET reports the bad value.
Scalar type changes are explicit even when Python considers the values equal.
A stale write returns HTTP 409; refresh before trying again. Unknown stored fields
survive, pack and environment namespaces stay separate, and helper failures return
HTTP 503 without fallback. App shutdown does not close the supplied client.
Directory and in-memory stores retain their existing behavior. Ordinary engine
startup does not supply a helper.

`RuntimeSettings(path=client, environment=...)` opts into environment-keyed
helper settings instead of `settings.json`. It loads known preferences at
construction, merges saved memory budgets into the supplied defaults/config,
and keeps caller values for categories marked `cli`. Do not also supply
`persisted_values`; the helper supplies that snapshot. Compose queues, governors
and workers from the resulting effective settings, as with file-backed startup.
A settings GET refreshes the persistence revision but reports this engine's
active policy; it does not apply a peer's edit to a running engine. A PUT patches
only the submitted category at the last observed revision before changing live
policy. Memory-budget patches include only submitted devices, and logging edits
remove cleared Dinkster overrides without removing unknown fields. Stale edits
return 409 without live changes; GET refreshes the view for retry. Construction
rejects invalid stored known values. GET only fetches raw values and the revision,
so incompatible peer preferences cannot block viewing this engine's valid active
policy or repairing a category with PUT. Helper failures return 503 with no file
or memory fallback. Unknown categories, fields and logging names survive.
The caller owns the client. Ordinary startup still uses files.

`PrincipalPermissionStore` and `dinkster_server.auth.DelegationStore` accept a
caller-owned `LibraryClient`. Supplying the permission store to `create_app` or
`install_auth` also selects that helper for delegated credentials. Agent requests
read current permission overrides and revocation state; no cached authority is
used after helper failure. Only minted `dinkster_delegate_` credentials query the
delegation store, so human JWT and static authentication need no helper read.
Permission edits patch the last observed principal revision and return 409 if it
changed; read the principal list before retrying.
Unknown categories survive edits. Delegation creation persists only a credential
hash, preserves the existing limits, and returns the credential once. Listings
retain the existing HTTP shape and follow helper cursors. Helper failures return
503 without fallback; federated asset routes use their versioned
`service-unavailable` error. Active WebSockets withdraw authority and close with
1008.
Closing a store or application does not close the supplied client. Legacy
constructors and ordinary startup remain unchanged.

`HistoryStore(client, accepted_path=...)` opts into helper-owned finished history
while keeping accepted jobs in a required engine-owned SQLite file. Terminal
publication commits before accepted work is retired; helper failure leaves that
work available for recovery. Retries preserve the original run and deleted runs
cannot reappear. Reads observe peer history, page through helper limits and keep
the existing HTTP cursors and scope isolation. Helper failures return HTTP 503
without opening a legacy history database. Closing the store does not close the
caller-owned client. Engine-side collection refuses helper-backed history: only
the reservation-holding helper may collect shared files. Ordinary startup remains
legacy, and the host still owns engine-state isolation and reference publication.

The main protocol routes include:

- `GET /api/health`, `/api/nodes`, `/api/workers`, `/api/composition`, and
  `/api/diagnostics`. The workers catalog reports the implicit `local` worker
  and configured remotes without probing their endpoints.
  `/api/health` remains an HTTP 200 liveness endpoint; `ok` is
  false when composition has no nodes or any failed pack
- `GET /api/choices/{choice_id}` for pack-announced combo choice lists (a
  plain JSON array of strings; unknown ids answer 404) - the remote half of
  the schema wire's COMBO widget descriptor. Lazy ids fetch from the owning
  worker per request with a server-owned timeout and answer with JSON errors
  and `Cache-Control: no-store`: 502 for a provider failure or invalid
  result, 503 when the owner is not connected, 504 on timeout
- `POST /api/jobs`, `GET /api/jobs`, and `GET` or `DELETE /api/jobs/{client_id}/{job_id}`.
  Active same-content duplicate submissions return the existing jobRef with
  `duplicate: true`; different content under the active key returns 409.
  Submissions may set `cacheEnabled: false` to execute every node without
  result-cache reads, writes, or single-flight coalescing while preserving the
  server process and loaded model consumers. Omission or `true` keeps normal
  memory or layered result-cache behavior.
  Submissions may include `placement: {"topLevelNodeId": "workerName"}`;
  region hints cover their complete body. Placement participates in job
  idempotency but not the graph document or node input fingerprints;
  the selected implementation arm remains part of node cache identity.
  The optional `attention` object has exactly `requestedPolicy` and
  `requestedRolePolicies`; omission uses the server default. For example:
  `{"requestedPolicy":"auto","requestedRolePolicies":[["flux","comfy_kitchen_int8"]]}`.
  Policies are `auto`, `sdpa`, `flash`, `xformers`, `sage`, `sage3`, and
  `comfy_kitchen_int8`; role overrides may name `unet`, `flux`, `vae`, `clip`,
  `t5`, or `qwen`. Each role may appear once, `auto` and no-op role overrides
  are invalid, and at least one role must keep the global policy.
- `GET` or `DELETE /api/jobs/by-ref/{job_ref}` for status/cancel without
  client affinity, plus `GET /api/jobs/by-ref/{job_ref}/events?after=N` for
  ordered per-job replay (`after` defaults to 0; 410 requires a status
  re-snapshot when retained events no longer cover the cursor)
- `/api/generation` for native JSON or SSE text generation, model discovery,
  explicit lazy load/unload, and session deletion, plus OpenAI-compatible
  `/v1/models`, `/v1/completions`, `/v1/chat/completions`, and `/v1/responses`.
  The model catalogs require authentication, generation requires
  `jobs:submit`, load/unload requires `settings:write`, and session deletion
  requires `sessions:write`. Disconnecting a stream cancels provider work;
  application cleanup closes active streams, sessions, and loaded providers.
- `GET /api/events` for additive WebSocket events; job-correlated events carry
  their per-job `seq`, while job status carries `latestSeq`
- `GET /api/queue` and `POST /api/queue/pause`, `/resume`, or `/clear`
- `GET /api/queue/idle` (also `HEAD`) reports only `{"idle": bool}` for management
  probes, with the same queue authorization but no asset-client admission.
  Full queue reads still admit clients because job values can reference assets.
- `/api/assets`, `/api/library`, and `/api/history` when their stores are composed.
  `POST /api/assets/media?scope=&kind=&name=` is the bounded authenticated media
  ingest: it accepts only the closed image/audio/video/model3d Content-Types,
  derives canonical facts from bytes, and creates exact scoped immutable authority.
  Hosts with an `upload_writer` publish named files to the configured
  `import_mount` by default. Optional `project` (default `default`), `mount`,
  and relative `folder` select a project-member writable destination.
  Original filenames are preserved, collisions are numbered, and the response
  carries the actual virtual path. No durable vault copy is created.
  Destinations are rechecked at publication; unavailable or revoked grants
  fail rather than redirecting. Vault-only embedding hosts keep digest storage
  and reject explicit folder requests.
  `POST /api/assets?scope=&name=` imports an unclassified named file, including
  model weights, through the same bounded streaming writer. It accepts the
  same optional project destination fields and returns `{digest, asset}`;
  it requires configured upload folders and does not create media authority.
  Without query fields, `/api/assets` retains anonymous idempotent digest
  storage for workflows and transfers and returns `{digest}`.
  `POST /api/assets/transfers?scope=&name=&project=` receives an explicit
  digest transfer into the vault and returns `{digest, asset}`. It requires
  `X-Dinkster-Digest` and `X-Dinkster-Source-Machine` headers, `assets:write`
  in the receiving scope, and the existing generic upload size limit.
  `project` defaults to `default`; mount/folder destinations are rejected.
  The source-machine label is caller-reported advisory history, not a verified
  machine identity or filesystem authority. Verified bytes and their scoped
  transfer observation publish through the same recoverable streaming path
  as named uploads; generic transfers create no classified media authority.
- `GET /api/assets/catalog?scope=&project=` lists persistent upload observations,
  grouped by digest, separately from workflow records. `q` searches original
  and stored names and recorded origin details; `kind` and `origin` filter
  exact values. `since`/`until` are inclusive observation times in Unix seconds.
  `limit` and query-bound `cursor` page identities, retaining their aliases
  and history. The route requires `assets:read` in the requested scope,
  defaults to project `default`, and exposes virtual locations only.
  It remains available when the optional federated catalog is enabled.
  Consumers merge project observations and mount rows with federated candidates
  by digest, retaining each source's names, locations and authority separately.
  Observation kinds describe ingest, not model-loader compatibility: a generic
  transfer or model upload is `asset/file`, even with a model-like filename.
  Logical model kind, variant and compatibility must come from the resolution
  catalog; a local copy does not make unknown runtime compatibility compatible.
  Project observations are not copied into the project-unaware resolution store.
- `GET /api/assets/availability?scope=&digest=&digest=` checks exact local
  identities without acquisition, writes, grants or catalog recovery. It
  returns `{assets: [{digest, status}]}` with `available` or `missing` and
  `Cache-Control: no-store`, deduplicating digests in request order. It requires
  `assets:read` in scope and returns no host paths. The managed server checks
  its vault and current mount grants with the existing descriptor-bound read
  verification. Embedding hosts must supply `ServerLibrary.local_resolver`
  for their local-only mount checks and job preflight; the possibly fetching
  `resolver` is used by neither presence check. An absent local resolver checks
  the vault only. Corrupted vault bytes do not shadow a valid granted copy.
  Job acquisition still requires digest-exact consent, including replacement
  of corrupted vault content. Submission retries recover pending catalog
  observations before preflight, without holding the upload lock during downloads.
- `POST /api/assets/latent?scope=&name=` streams and strictly validates native
  or supported ComfyUI latent safetensors under independent concurrency, size,
  idle-timeout, and disk-headroom limits
- `POST /api/assets/image-document?scope=` validates and adopts a bounded
  `application/vnd.dinkster.image-document+json` document and its held raster
  dependencies under bounded decode admission; `GET
  /api/assets/{digest}/dependencies` reads its immutable manifest. `POST
  /api/assets/{digest}/render?scope=` runs the deterministic CPU reference
  renderer for `composite`, `layer:<id>`, or `mask:<id>` after revalidating
  resource authority in the requested scope, publishes the PNG to the asset
  vault, and returns its immutable cache key and provenance.
- `/memory/*` for memory status and reservations, and `/cache/*` for cache sharing

This is the native Dinkster protocol, not ComfyUI's `/prompt` or `/object_info`
surface.

## Result publications

`GET /api/library/result-offers/{token}?scope=` reads an immutable server-created output
publication. It requires scoped `assets:read`, returns 404 for an unknown token,
and never reconstructs output metadata from a current cache entry. Job events,
status, and completed-value discovery expose the same publication for the
corresponding runtime output. Each job receives distinct tokens even when it
reuses cached values.

`GET /api/library/terminal-results/{jobRef}?scope=` reads the sealed terminal
publication independently of queue/cache retirement, ordinary history deletion,
and backend restart. It requires scoped `assets:read` and `history:read`; unknown
or legacy publications return 404 `publication-unavailable`. Terminal artifacts
retain their original saved-file descriptors and optional preview publications.
These publication routes expose recorded result metadata, not output payloads.

## Storage routes

`GET /api/library/storage` requires an authenticated human operator with
`assets:read`, accepts no query parameters or filesystem destinations, and
returns `Cache-Control: no-store`. `PUT /api/library/storage/policy` requires
an operator with `assets:write` and accepts `{limitBytes: number | null}`.
`POST /api/library/storage/clear` requires the same write authority and an empty
JSON object. These routes retain their existing response contracts; storage
cleanup behavior is documented with the implementation that provides it.

## Learn more

See DESIGN.md sections 3.5 (server protocol), 3.10 (multi-instance
coordination), and 3.12 (assets). Focused coverage is in
`tests/test_server.py`, `tests/test_history.py`,
`tests/test_workflow_library.py`, `tests/test_peers.py`, and
`tests/test_cache_share.py`.
