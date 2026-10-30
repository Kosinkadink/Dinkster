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
- `GET`, `PUT`, and `DELETE /api/library/retentions/{owner_kind}/{owner_id}?scope=`
  persist explicit references for `document`, `history`, `shown-result`, or
  `preview` owners. `owner_id` is an opaque non-empty identifier of at most
  512 characters. Reads require `assets:read` in scope; writes require
  `assets:write`. PUT accepts `{revision, digests, complete?}`, with at most
  4096 canonical asset digests. Revision zero creates an owner; replacements
  require the current revision. DELETE requires `revision` in the query and
  publishes an empty manifest with a newer revision, preserving stale-tab
  protection even when an identifier is reused. All successful operations
  return `{scope, ownerKind, ownerId, revision, digests, complete}` with HTTP 200.
  `complete` is a boolean defaulting to false; incomplete or legacy owner
  manifests block collection inspection even when their digest list is empty.
  Missing owners return 404, stale revisions 409, invalid requests 400 and
  unavailable storage 503. Digests are deduplicated and sorted. Missing bytes
  may be retained without fetching, copying, cataloging, or granting media
  access. Associations persist until explicitly replaced or released; there
  is no age expiry. They do not parse opaque workflow documents, register
  existing history automatically, or authorize cleanup on their own. Clients
  must declare complete reference sets; absent associations are not evidence
  that older documents contain no asset values. Cleanup remains disabled.
  `HistoryStore.hold_asset_references()` separately holds the history write
  barrier and reads source-document roots from accepted and terminal jobs in
  every scope. Terminal writes atomically retain external asset identities
  from nested target values, saved artifacts and their optional published
  previews, independently of execution cache retention. These manifests
  persist until their history rows are removed and grant no byte access.
  Legacy and interrupted rows with unknown
  output references block inspection rather than count as empty. History
  responses do not expose these internal manifests. This does not inspect
  opaque workflow bytes, retain live previews, or enable deletion.
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

## Retained asset registration

The retention endpoint acknowledges durable ownership, not display, byte
availability, read permission or permission to collect. A client declares a
complete typed identity set before exposing an asset through a document,
shown result or live preview. `complete: true` certifies only this owner's
set; it cannot certify the existence or completeness of other owners.
There is no request that enables destructive cleanup by setting this flag.
Unknown global ownership keeps cleanup disabled, independently of individual
complete manifests. Disconnect, restart and elapsed time never release roots.

### Identity and ownership

`digests` contains canonical `blake3:<64 lowercase hex>` external asset
identities, recursively enumerated using the owning value type's schema.
Include nested arrays/records, typed document resources and transitively
owned values. Do not use payload-cache keys, arbitrary fingerprints, image-only
display helpers or truncated descriptors as an identity census. Known missing
assets still belong in the set. A successful registration neither fetches nor
materializes them; check availability separately before displaying bytes.
Unsupported or opaque value formats require `complete: false`, even if all
currently understood identities have been listed. Oversized sets must not be
silently truncated to the endpoint's 4096-identity bound; retain an incomplete
owner and block collection until a complete declaration is representable.

Use distinct opaque `owner_id` values for independently retained generations,
not a shared tab label. Scope, owner kind and owner ID together are the key;
the durable revision prevents stale replacement/release after ID reuse.
Persist owner IDs and acknowledged revisions with the state they protect.

- `document` covers current, undo/redo and frozen documents, restored sessions,
  shared-session replacements and imported documents, including graphs not
  instantiated in the current root. Owners may declare their union or use
  separate IDs. Closing one view does not release another view's values.
- `shown-result` covers complete retained result values, including restored
  results and frozen execution views, not just currently rendered descriptors.
- `preview` covers the latest retained still, ring frames, pending reads and
  decoded media lifetimes. Renderer unmount is not a release if any other
  owner can still display the value.
- `history` is available for client-owned persisted values. Server terminal
  targets separately retain their identities with their real history rows;
  this does not infer dependencies of an opaque workflow `sourceDocument`.

### Acknowledgment and replacement

1. Create a fresh generation with PUT `{revision: 0, digests: [...], complete:
   true}`. Do not publish it to the document/result/preview until HTTP 200
   acknowledges the exact canonical set, completeness and new revision.
2. While replacing a still-retained value, either register a new generation
   before releasing the old one, or PUT the union of old and new identities
   at the current revision. Only after acknowledgment may the consumer expose
   the new value. After all old consumers and pending reads have finished,
   replace the union with the new set at the acknowledged revision.
3. DELETE at that revision releases an owner only after all its consumers,
   undo/redo entries and recoverable persisted state relinquish the values.
   The acknowledged response is a complete empty set at the next revision,
   not deletion of the revision counter or of any asset bytes.

If the response is lost, GET the same key and compare the entire canonical
set, completeness and revision before proceeding. Do not blindly replay a
mutation after a conflict. HTTP 409 requires reconciliation with the real
owner; a stale callback must not adopt a newer owner's revision and release
it. Failure, cancellation or disconnection before acknowledgment leaves the
old values retained and the new value unpublished. An incomplete manifest
must be acknowledged before exposing values whose identities are unknown.
Cleanup cannot reclaim from incomplete or unregistered owner state.

### Backend synchronization

Library and history inspection hold their SQLite write transactions; cache
inspection holds the cache admission/publication barriers. External roots
must remain held while library dependency closure is computed and candidates
are considered. A copied root set after those barriers close is not safe.
Managed vault writers and verified readers acquire vault-wide OS ownership
before per-identity ownership, retaining both through their existing lifetime.
Server jobs retain vault ownership from dispatch through engine/native
settlement, terminal recovery and durable history publication. Media publication
workers retain it through their record/reference commit, including when the
request is cancelled. Scratch writers use job-root ownership instead.
An exclusive vault claim is nonblocking and must be acquired after all reference
barriers, then held through the final operation. If a publisher already owns the
vault while waiting on a reference barrier, inspection refuses the claim and
releases its barriers rather than waiting in the reverse order. This closes the
server-job publication-to-history gap, not every possible external handoff.
Unknown global owner completeness still blocks deletion.

## Storage accounting

`GET /api/library/storage` requires an authenticated human operator and
`assets:read`. It accepts no query parameters or filesystem destinations and
returns `Cache-Control: no-store`.

`limitBytes` is the global persisted produced-asset size policy, initially
21474836480 (20 GiB). Null means unlimited only when `policyAvailable` is true;
unavailable policy reports false and null instead of inventing a default.
`PUT /api/library/storage/policy` requires an operator and `assets:write`, and
accepts exactly `{limitBytes: number | null}` with a positive JSON-safe integer.
It returns the acknowledged limit and `collectionEnabled:false`. Unknown keys,
filesystem targets and query parameters are rejected. Changes survive restart
but do not enable deletion or enforce the limit while ownership is unknown.

`POST /api/library/storage/clear` requires the same operator/write authority and
an empty JSON object, with no query parameters. It currently returns HTTP409,
`collectionEnabled:false`, `blocked:["owner_completeness_unknown"]` and zero
`reclaimedBytes`/`reclaimedFiles`. Repeating Clear is nondestructive. No policy
request or individual complete owner manifest can bypass this global gate.

`vaultBytes` counts canonical vault blobs, excluding sidecars and scratch;
`adoptedBytes` sums recorded verified adoption sizes. Either is null when its
state is unavailable. These byte totals are observations, not an atomic
publication snapshot or evidence that bytes are eligible for deletion.
`knownRetainedAssets` counts the union of held memory/disk cache, all-scope
history and library roots plus transitive dependencies. It is null when root
inspection refuses. A partial known set is not a complete owner census.
`vaultOwnershipAvailable` observes the exclusive vault claim while these owners
are held: true means the claim was acquired, false means a managed reader,
publisher or server job was active (`vault_in_use`), and null means ownership or
reference state was unavailable. A missing inventory or lease gate reports
`vault_ownership_state_unavailable`; inspection never creates either. This field
does not establish complete frontend registration or make bytes reclaimable.

Memory barriers remain held until the blocking disk/history/library inspection
settles, including repeated request cancellation. Disk roots are deduplicated;
blocking barriers are acquired and released on the same worker thread.
`temporaryBytes` observes regular files (including journals, sidecars and
interrupted ingest files) only inside recognized per-pack/per-group job stores
under the host-configured scratch root. Unrelated pack scratch is excluded.
The job/attempt path must match its read-only journal owner. Both host-job and
store OS claims remain held during each tree walk. Missing ownership, live
jobs/writers/readers, unreadable walks and symlinks return null with
`temporary_accounting_unavailable`, not a partial total. An explicitly configured
but absent scratch tree reports zero without creating it; an unconfigured root
remains unknown. Normal serve supplies `<library-root>/scratch`.
`reclaimableBytes` is null, `collectionEnabled` is false, and `blocked` always
includes `owner_completeness_unknown`. Missing, pending or unreadable state adds
explicit reasons. An inactive claim is not adoption or deletion authority;
this endpoint never deletes bytes or treats unknown as zero.

### Interrupted publication recovery

Before serving library requests, startup inspects the host-configured scratch
stores under the same host-job and store claims as accounting. Only journaled
publications explicitly offered for handoff are recovered. Missing engine bytes
stream through the existing digest-verifying vault writer, with pending
inventory recorded before copying and adoption committed while publication
ownership remains held. Existing engine bytes must verify their digest and size
before a pending inventory entry can be completed. Untracked existing engine
assets are not reclassified as produced assets.

Unknown journals, active owners, unsafe paths, size conflicts and corrupt bytes
block that store without stopping recovery of another recognized job. An
unreadable or unsafe directory enumeration stops that enumeration rather than
guessing what was omitted. Recovery never reruns a workflow, copies mounted
inputs, creates library records or removes scratch, including unoffered bytes.
The queue also reconciles the finishing job's offered publications after engine
and native settlement, on success, failure and cancellation. It awaits recovery
before publishing the terminal transition or releasing its dispatch slot; a
cancelled offloaded recovery retains ownership until the worker settles.
Recovery failure does not replace the job's original outcome. Other jobs' stores
are not recovered by this per-job pass. Both startup and terminal reconciliation
preserve all scratch and the first verified adoption timestamp.
`lastPublicationRecovery` on operator storage status reports the last completed
startup or per-job pass's `verifiedPublications` and `blockedInspections`, or
null when no scratch root was configured. These counts are observations, not
cleanup eligibility or a complete orphan census. GET status itself performs no
recovery mutation.

## Learn more

See DESIGN.md sections 3.5 (server protocol), 3.10 (multi-instance
coordination), and 3.12 (assets). Focused coverage is in
`tests/test_server.py`, `tests/test_history.py`,
`tests/test_workflow_library.py`, `tests/test_peers.py`, and
`tests/test_cache_share.py`.
