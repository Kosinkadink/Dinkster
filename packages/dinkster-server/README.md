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

The main protocol routes include:

- `GET /api/health`, `/api/nodes`, `/api/workers`, `/api/composition`, and
  `/api/diagnostics`. The workers catalog reports the implicit `local` worker
  and configured remotes without probing their endpoints. `/api/nodes` pack
  entries may carry dedicated `comfyAliases` and `comfyGroups` translation
  registries; these rules and import-only schemas are not copied into native
  node schemas. `/api/health` remains an HTTP 200 liveness endpoint; `ok` is
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
  `{"requestedPolicy":"auto","requestedRolePolicies":[["flux","dinkster_kitchen_int8"]]}`.
  Policies are `auto`, `sdpa`, `flash`, `xformers`, `sage`, `sage3`, and
  `dinkster_kitchen_int8`; role overrides may name `unet`, `flux`, `vae`, `clip`,
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
- `/api/assets`, `/api/library`, and `/api/history` when their stores are composed.
  `POST /api/assets/media?scope=&kind=&name=` is the bounded authenticated media
  ingest: it accepts only the closed image/audio/video Content-Types, derives
  canonical facts from bytes, and creates exact scoped immutable authority
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

## Learn more

See DESIGN.md sections 3.5 (server protocol), 3.10 (multi-instance
coordination), and 3.12 (assets). Focused coverage is in
`tests/test_server.py`, `tests/test_history.py`,
`tests/test_workflow_library.py`, `tests/test_peers.py`, and
`tests/test_cache_share.py`.
