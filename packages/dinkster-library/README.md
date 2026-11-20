# Dinkster library

`dinkster-library` serves shared project records over a private child-process
pipe. It opens `records.sqlite` in the selected library; engines do not open
that database. Existing asset records remain in `library.sqlite`.

## Starting and stopping

`LibraryClient(library)` starts a helper, negotiates API 1, and closes the pipe
when its context exits. `AsyncLibraryClient.start(library)` provides the same
calls without blocking the event loop; call `close()` when finished. A helper
exits on pipe EOF even while a database write is blocked.

Managed engine launches set `DINKSTER_LIBRARY_COMMAND` to a JSON command array
using the tools interpreter. With no override, the client runs
`sys.executable -m dinkster_library`. The command accepts `--library PATH`.
No listener, credential, operating-system service or login item is created.

A request is one JSON line with `api`, `operation` and `arguments`. A reply
holds either `result` or `error: {code, message}`. Calls are serialized per
client, bounded to 16 MiB per message and limited to 60 seconds. A timeout
kills the helper; callers must reopen and reread before attempting another
write, because losing a reply does not prove a write failed.

The helper's `--describe` command returns its API range and capabilities without
opening a library. Engine manifests list `libraryCapabilities` required in addition
to `libraryApi`; this engine needs folder scanning, progress, selected-file scanning
and record browsing.
Additive operations and fields keep API 1. Missing capabilities update the tools
before activation, even when the API number matches, and a replacement must retain
the installed helper's API range and capabilities. Managed installation, activation
and start use that check before releasing a running engine. If the installed tools
cannot meet the engine's requirements, the supervisor acquires the channel's signed
native tools archive, stages its Python tree and checks the new helper before
selecting it.
Selection lives outside engine snapshots, so an engine restore does not
restore older library code. Compatible starts work without a metadata fetch.

## File hashes

The `index.hash` capability offers one-file hashing: pass an absolute `path`
to `index.hash`. It returns the content `digest` and whether bytes were read
(`hashed`). The cache lives under `DINKSTER_HOME`, shared by this account's
libraries. Canonical paths and descriptor metadata identify unchanged files;
a folder lock makes another helper wait and reuse the committed result.
The fingerprint uses device, file identity, size and nanosecond mtime, plus
ctime on POSIX. Windows ctime is excluded because path and descriptor queries
can give it different meanings.
File I/O holds no record write transaction. A changed file is not cached.
`index.hash` alone does not change engine mount resolution.

`index.scan` starts a folder scan and returns immediately with its `id` and
progress. Pass absolute `root` and `output` paths. The output index itself is
excluded from scanning. Optional `legacy` paths seed existing verified indexes,
and `ignore` replaces the default placeholder/OS-junk patterns. `index.progress`
with the scan `id` returns `queued`, `scanning`, `complete` or `failed`, file
and byte counts, elapsed seconds, and the count of files hashed by this scan.
Record requests remain available while a scan runs or waits for a folder lock.
A new scan of an output replaces that output's finished progress record.
Pass `relative` to index exactly one safe relative file without enumerating the
folder or marking sibling rows missing. The selected file uses the same folder
lock, per-file cache and atomic publication as a whole-folder scan.

Scans serialize by canonical folder across helpers and commit each file to the
account cache before continuing. Pipe EOF stops scans with their helper;
reopening reuses committed files. Outputs use the existing worker index JSON
shape and atomic replacement, including missing identities and older content
versions. Existing library indexes remain untouched. Verified rows and worker
write sidecars seed the account cache only when their descriptor metadata
matches; size and modified time alone do not become verified ingest records.

A prepared model folder can carry `.dinkster-asset-index.json` in that same JSON
shape. A whole-folder scan accepts its valid digests as catalog hints when size
and nanosecond modified time still match, even after a disk copy changes device
or file identity. It strips stale verification metadata and does not put hints
in the verified account cache. A known changed cached file cannot fall back to
a shipped hint. Mismatching metadata, malformed digests and missing rows are
hashed normally. Engine publication leaves the shipped list untouched.

These hints make the folder's catalog and digest references available without
reading every model on first run; they are not content proofs. The existing
loader verifies unproven bytes on demand and refuses a digest mismatch, even
for wrong bytes with unchanged size and modified time. Exact-file scans and
`index.hash` also verify on demand and populate the shared cache. No background
full-folder verification is required. Filename-only bindings retain their
existing file-revision requirements; a shipped hint does not grant a verified
filename binding or bypass a loader's integrity checks.

Engine mount scans use the helper already attached for project records. Its
published indexes live beside the engine's worker snapshot, within the same
sandbox read grant. During folder scans the engine loads their catalogs without
hashing or replacing them. Before the first scan, writer publication atomically
creates an initial index only if none exists; worker publications remain readable
through their existing sidecars.
`AsyncLibraryClient.sync` exposes the same serialized client for scan threads.
Standalone asset-library scans retain their direct indexing path.

## API compatibility

The [API register and removal procedure](../../docs/library-api.md) distinguish
wire versions from storage formats. `compatibility.affected` accepts a proposed
`apiMin`/`apiMax` and pages through recorded snapshots and their projects that
would no longer open. The helper computes this from attachment and project
records; the tools render notices only for affected records. No API is removed.

## Engine-free browsing

`records.list` requires `scope` and accepts optional `project`, `mediaType`,
`q` (name substring), `label`, `limit` and `cursor`. It returns `records` and
`cursor` (null at the end), ordered by modified time then ID descending.
Workflow records include their `project`; legacy workflows belong to `default`.
For the legacy catalog a project filter selects only that live project's
workflows, not generic records. Helper-owned saved records all carry a project,
so the filter includes that project's named assets as well.

`assets.list` requires `scope` and accepts `project` (defaulting to `default`),
`q`, namespaced `kind`, `origin`, inclusive numeric `since`/`until`, `limit`
and `cursor`. It returns `assets` and `cursor`, ordered by latest observation
then digest descending. Each entry contains the existing `asset` wire form,
`aliases` and `observations`. Matching any observation retains that digest's
other aliases and observations within the same scope and project. It does not
read asset bytes or inventory unrecorded files.

Both operations accept page limits from 1 to 200, bind cursors to every filter,
and do not create or migrate the legacy asset catalog. When it is absent,
`records.list` reads helper-owned saved records and `assets.list` reads the
helper-owned ingest catalog.

`assets.record` takes `scope` and an `observation` in the observation wire shape
above. The publisher supplies a stable 32-hex `id` and finite nonnegative
`created` time; `project` defaults to `default`. The helper validates the asset
identity, basename, size, virtual path, kind and origin. Download origins omit
URL user information, query and fragment. The response contains `observation`
and `created`, which is false for an identical retry. A conflicting reuse of
an ID is refused, including a change of scope. Unknown observation and asset
fields survive storage and listing. Ingest history does not grant authority to
read bytes. Its digests do join reference reservations, matching the existing
catalog retention policy, so catalog writes wait behind a reservation.
This operation refuses an unmigrated legacy catalog rather than creating a
second writable one.

## Resolver subscriptions and snapshots

`sources.resolvers` publishes trusted engine resolver state without sending a
large snapshot in one pipe message. `begin` takes a 32-lowercase-hex `id` and
expected nonnegative `revision` (0 for a new ID), returning a `token` and byte
`offset`. Each helper stages at most one upload, in memory. `append` takes that
helper's token, the exact current offset and base64 `data` containing 1 byte to
1 MiB. The complete staged document is limited to 64 MiB. `cancel` discards it;
helper exit also discards it without changing any published record. Failed
commits leave staging available for cancellation.

The UTF-8 JSON document contains `subscription` and its derived provenance
`records`. Subscription fields use the engine's persisted shape: `id`, immutable
`source`, `sourceType` (`local` or `url`), `index`, and optional `etag`, `error`,
`completeSnapshot`, `checkedAt` and `refreshedAt`. The index has
`dinksterResolver: 1` and at most 100,000 `entries`. Provenance digests must match
the entry digests. The engine validates canonical source identity, resolver
entry semantics, fetches, official-provider policy and regional URL selection;
the helper does not access a source. It checks storage shape, finite status
timestamps and their order, digest correspondence, uniqueness and limits.

`commit` takes the token, rechecks the expected revision, and atomically stores
the snapshot and replaces `resolver-index:<id>` provenance. Successful commit
returns the new `revision` and serialized `bytes`, and clears staging. At most
64 subscriptions can be active, and active sources are unique. Unknown envelope
and subscription fields survive refreshes that omit them; unknown index/entry
fields survive in the supplied snapshot. Snapshot replacement is explicit,
not an additive merge of old entries. Other provenance layers and base records
are unchanged. Direct provenance replace/remove refuses the reserved
`resolver-index:` prefix. Resolver and provenance change events commit together.

`get` takes `id` and returns a small `subscription` header or null. `list`
returns active headers ordered by ID. Headers contain `id`, `source`,
`revision`, nullable `bytes` and `deleted`. `read` takes ID and revision, a byte
`offset` (default 0), and byte `limit` from 1 to 1 MiB. It returns base64 `data`,
next `offset` and `complete`. A stale revision refuses instead of mixing chunks
from different snapshots. `remove` takes ID and expected revision, atomically
withdraws the snapshot and its leads, and returns the new revision. An empty
revision tombstone prevents an old upload from recreating deleted state;
explicit publication at the tombstone's current revision is allowed.

Raw and merged provenance size limits also apply during publication and
withdrawal; a refusal rolls back snapshots, leads and events together. Resolver
writes remain available during reference reservations and create no asset
roots. All operations refuse unmigrated `library.sqlite`, `provenance.json`
or `resolver-indexes.json` without touching them.

`sources.resolvers.bootstrap.get` returns `configuration`, null until recorded.
`bootstrap.put` records the immutable official-provider choice using a
`configuration` object with `source`, `providerId` and `subscriptionId`; unknown
fields survive. It returns the stored configuration and `created`. A retry may
omit unknown fields, but cannot change a recorded value. The engine validates
URL and provider policy; the helper bounds the strings and document and checks
that an active subscription with this ID has the same source and index name.
A bootstrap may outlive its subscription. Removing a subscription never clears
it, and reading it never recreates an unsubscribed provider.

The first subscription's `commit` may include `officialBootstrap` with the same
configuration shape. Snapshot, leads, bootstrap and changes then commit or roll
back together. Every later publication of the recorded subscription must still
match its official source and provider ID. A separate `bootstrap.put` supports
an already-subscribed provider or an imported choice that has been unsubscribed.
Bootstrap writes create a `resolver-configuration` change only on first insert,
remain independent of reference reservations, and share resolver legacy refusals.
Snapshot JSON refuses duplicate field names at every nesting depth, rather than
silently overwriting part of a persisted document.

## Provenance leads

The `sources.provenance` capability stores acquisition leads, not authority or
asset references. `sources.provenance.add` takes `record` with canonical
`digest`, string-list `sources`, string `license` and `note`, and object
`metadata`. Missing optional fields default to empty values. Base records merge
additively inside the write: source URLs form an ordered union, later nonempty
license/note values win, metadata keys merge, and unknown top-level fields
survive unless explicitly overwritten. Add returns the merged `record`.

`replace` takes a nonempty `source` name of at most 512 characters and `records`
to atomically replace only that named layer. Duplicate digests in the input
merge in order. Layers apply after the base in original insertion order;
replacing a layer does not move it. `remove` withdraws a named layer and returns
`removed`; replacing with an empty list also withdraws it. Recreating a removed
layer puts it last. Withdrawals never erase base records or another layer.
Each write emits a `provenance` change with its source name (empty for the base)
and revision 1, identifying a changed source rather than an editable revision.

`get` takes `digest` and returns the merged `record` or null. `list` returns
merged `items` ordered by digest. Both accept optional `source`: an empty string
selects the base and a named source selects its raw records, preserving unknown
fields independently of other layers. `layers` lists active source names in
lexical order. Both lists take `limit` from 1 to 100 and a query-bound `cursor`,
returning a nullable next cursor. A page reads from one database snapshot.

Raw and merged records are bounded at 128 KiB and the store at 100,000 rows.
Writes, including withdrawal, refuse atomically if the resulting merged
records would exceed the bound; no row or change event is committed. The
100-item page bound keeps replies below the pipe limit. Provenance writes do
not wait on asset reference reservations and never fetch or validate remote
bytes. All operations refuse unmigrated `library.sqlite`, `provenance.json`
or `resolver-indexes.json` rather than hiding existing leads.

## Public acquisition receipts

The `sources.receipts` capability provides `sources.receipts.put/get/list`.
Put takes `receipt`, a version-1 public acquisition receipt attested by the
trusted engine: `receiptId` (32 lowercase hex characters), `digest`, nonnegative
integer `sizeBytes`, `sourceType`, `sourceId`, `sourceRevision`, `listedUrl`,
`finalUrl` and finite nonnegative `fetchedAt`. Source types are
`official-provider`, `declarative-resolver`, `code-resolver` and `manual`.
Source IDs and revisions are trimmed, control-free strings of at most 512
characters. URLs use HTTPS, are at most 4096 characters and contain no
whitespace, user information, query or fragment. These shape checks do not
verify a fetch: the engine still verifies the public HTTPS chain, actual bytes,
digest, size and matching trusted source before recording a receipt.

A receipt is immutable under its ID, including unknown fields. Identical
concurrent retries insert once and return `created: false` thereafter;
conflicting documents refuse without a change. Creation atomically emits an
`acquisition-receipt` change with the receipt ID and revision 1. Get by
`receiptId` returns `receipt` or null. List accepts optional `digest`, pages in
ascending `fetchedAt,receiptId` order with a filter-bound cursor and `limit`
from 1 to 100, and returns `items` and nullable `cursor`. Unknown fields survive
reads and restarts. Receipts are shared across engines, not scoped to a tab.
The engine's `PublicAcquisitionReceiptV1.from_wire` parser accepts exactly its
10 version-1 fields, not this extensible stored document. Consumers must project
`version`, `receiptId`, `digest`, `sizeBytes`, `sourceType`, `sourceId`,
`sourceRevision`, `listedUrl`, `finalUrl` and `fetchedAt` before calling that
parser, and retain the full document separately. Do not replace a stored receipt
with the parser's `to_wire()` result: that would discard unknown fields.
The strict legacy JSON store is not a reader for helper receipt documents.

Serialized receipts are limited to 128 KiB and the library to 100,000 receipts.
The record size and page limit keep complete replies below the pipe limit.
Receipts document past acquisition; they do not grant media authority or keep
bytes alive, and writes remain available during reference reservations. All
operations refuse unmigrated `library.sqlite`, `provenance.json`,
`resolver-indexes.json` or `public-acquisition-receipts.json` rather than hiding
existing source records. No source is contacted by these storage operations.

## Permissions and delegations

The `permissions` capability provides `permissions.get/patch` by `principalId`.
Responses contain `permissions` with `principalId`, `revision` and `values`.
Missing overrides have revision 0 and empty values; engine permission defaults
are not materialized here. A patch requires the observed integer `revision`
and a `patch` of nonempty category names to booleans. Unknown categories survive
later patches. Stale edits, including boolean revisions, refuse without writes.
The full reply is limited to 1 MiB before commit, including JSON escaping and
newline. Writes atomically advance the revision and record a `permissions`
change under the principal ID. These records are shared across engines.

The `delegations` capability provides `delegations.create/get/lookup/list/revoke`.
Create accepts a `record` with immutable `id`, `principalId`, `agentPrincipalId`,
`credentialHash` (canonical SHA-256), `scope`, `displayName`, `createdAt`, optional
nullable `sessionId` and `expiresAt`, and unknown fields that remain preserved.
Delegation IDs are limited to 512 characters so a page cursor stays readable.
Expiry must follow creation. Tokens and caller-selected `revokedAt` are refused.
Records are limited to 64 KiB serialized JSON. Identical creates return
`created: false`, including after revocation, and cannot restore authority.
Conflicting IDs, credential hashes or agent IDs refuse. Existing limits remain:
at most 32 unrevoked delegations per principal and 4096 in the library.

Get by `id` or lookup by `credentialHash` returns `delegation` or null. Returned
records include nullable `revokedAt` and never include `credentialHash`. The
engine still verifies expiry, owner permissions and human-session freshness;
lookup is a storage operation, not an authorization decision. List requires
`principalId`, excludes revoked records, omits hashes, and pages in ascending
`createdAt,id` order using a filter-bound cursor and `limit` from 1 to 200.
Revoke requires the owning `principalId` and `id`; it sets helper time once,
returns `revoked: true` only on the first matching write, and cannot be undone.
Creation and revocation atomically emit `delegation` changes with revision 1
and 2 respectively. Reads and writes do not reserve asset references. These
private-pipe operations trust the engine to enforce principal administration;
no bearer token is sent to the helper. Both capabilities refuse an existing
legacy `principals.sqlite` or `library.sqlite` until the complete migration,
without touching either file.

## Settings records

The `settings` capability provides `settings.get` and `settings.patch`.
Pass exactly one nonempty identity: `environment` for that environment's engine
preferences, or `pack` for shared preferences of that pack. The namespaces are
separate even when the identifiers are equal. A new helper attachment does not
reset preferences. The response is `settings` with the selected identity,
`revision` and `values`; an absent record has revision 0 and empty values.

Patch with the last observed `revision` and a `patch` object. Nested objects
merge recursively, retaining unknown siblings. Scalars, arrays and null replace
the named value; an empty object does not erase an existing object. To remove
values explicitly, pass `remove`, a list of nonempty string-component paths,
for example `[["logging", "overrides", "engine"]]`. Removal runs after merging,
so a path in both selects removal. Missing final keys are harmless; a missing
or non-object parent refuses the entire write, including its patch. A successful
patch advances the revision, including a no-op. Stale revisions, including
booleans, are refused without changing values or recording a change.
The complete JSON reply, including identity, revision, escaping and its newline,
is limited to 1 MiB. A patch exceeding that limit is refused before committing,
leaving the previous values and revision readable and unchanged.

Writes read the latest record inside one SQLite transaction. Settings changes
use kind `settings`, the JSON identity `["environment", id]` or `["pack", id]`
and the new revision in the ordinary changes feed. They do not alter file
references and remain writable during reference reservations. Callers validate
known settings against their runtime/pack schemas and filter unknown fields
before applying them; the helper preserves those fields for newer engines.
These storage operations neither validate a runtime schema nor authorize a
runtime change; they are not a replacement for runtime permission checks.
Unmigrated `library.sqlite`, `settings.json` or `pack-settings`
cause a migration refusal rather than an apparently empty record. No legacy
file is imported, replaced or removed by these calls.

## Records

`hello` returns the API range, capabilities, storage format and other attached
helpers. `projects.register` creates an ID if absent; `projects.get`,
`projects.list`, `projects.patch` and `projects.delete` read and edit it.
Patches carry the last observed `revision`; a conflict returns `stale` without
changing the record. Immutable identity fields cannot be patched.
The `projects.environment-selection` capability allows an environment binding
before a run: `lastSnapshot` and `lastRunAt` are both null or both nonempty
strings. An identity-only environment patch selects that environment without
claiming execution. Changing its ID clears the old run fields unless a complete
new run is supplied; selecting the same ID retains its run. Partial run patches
are merged and then validated as a pair.

`workflows.record` keeps project ownership and engine-supplied facts beside
legacy workflow records. `workflows.get` and `projects.needs` read those
associations; `workflows.read` reads a legacy record for requirements checks.
With `includeWorkflows: true`, `projects.needs` also returns `workflows`, an
array of owned workflow references with `scope`, `id`, `digest` and `name`.
This is advertised as `projects.needs.workflows`; the omitted or false option
preserves the original response. References include legacy `default` workflows
and records whose cached facts are absent or stale. Engines use the verified
bytes to compute current needs, rather than trusting older cached attribution.
Workflow contents are read from the vault by the
engine, not interpreted by the helper. Facts whose digest no longer matches
are unknown. `engines.seen` and `engines.list` retain environment/snapshot
identities. Lists use cursors; `changes` pages through committed revisions.

## Saved workflows and named assets

The `records` capability adds `records.create`, `records.get`, `records.patch`
and `records.delete`. Every operation requires `scope`; get, patch and delete
also require the record `id`. Create accepts a `record` with `name`, `digest`,
`mediaType` and optional `labels` and virtual `folder`, plus a `project` that
defaults to `default`. The helper mints the ID and timestamps. Patch takes a
`patch` of those mutable fields and the last observed `revision`; delete also
requires the revision. Wrong-scope records are absent, not hints. Deletion
retains a tombstone without the document contents. Project ownership and
identity are immutable: putting a workflow in another project requires a copy.
Unknown stored fields survive ordinary patches. Changes join the common feed.

These operations refuse an unmigrated `library.sqlite` rather than create a
second writable catalog. With no legacy catalog, `records.list`,
`workflows.read`, `workflows.get`, `workflows.record` and `projects.needs` read
the helper-owned records. Existing legacy browsing remains read-only. A changed
document digest makes cached workflow facts unknown until the engine records
matching facts. Live saved records across every scope contribute their digests
and reachable dependencies to reference reservations; record mutations wait
behind the reservation without a SQLite transaction held across file work.

## Scoped media grants

The `grants` capability offers `grants.put` and `grants.get`. Both require
`scope`. Put takes a `grant` with `digest`, `kind`, `mediaType`, `extension`
and positive integer `byteSize`; an optional `scope` in the grant must agree.
Get takes the exact `digest` and `kind`. A different scope or kind is a miss.
The stored grant includes its scope and preserves unknown fields. Put returns
`grant` and `created`; an identical retry is harmless and a conflicting fact
for the same scope/digest/kind is refused.

The owning engine verifies the supplied bytes before calling put, including
the strict latent parser. The helper trusts its owning engine, not a network
caller; it validates canonical media facts but does not classify bytes in this
operation. Upload and download routes must keep their existing byte-derived
verification before persisting authority. Grants are not ingest observations
and do not fabricate catalog entries. Their digests and reachable dependencies
join reference reservations across all scopes, and grant writes wait behind
reservations. Both operations refuse an unmigrated legacy catalog.

## Engine-owned result publications

`results.offers.put` takes `scope` and an immutable `offer` in the result wire
shape: a publisher-stable 32-hex `token`, `jobRef`, positive `attemptId`,
`nodeId`, positive `publication`, explicit `digests`, boolean `complete`, and
either `outputId` or an opaque engine-verified `artifact` descriptor.
`results.offers.get` takes the scope and token. The helper must first be bound
with `references.engine.attach`; it supplies the engine identity, not the
request. Wrong-scope and other-engine tokens are absent. Unknown offer fields
survive, while an identical retry is harmless and a conflicting token reuse
is refused. More than 4096 unique roots makes an offer incomplete.

`results.renditions.put` takes `scope`, source `token`, opaque `cacheKey` and
an immutable `rendition` with `documentDigest` and explicit `digests` for its
resources. `results.renditions.get` takes the same source and cache identity.
A put without a source in this engine and scope returns a null rendition and
`created: false`. Both puts return the document and boolean `created`.
They advertise `results.offers` and `results.renditions`, preserve unknown
fields, refuse an unmigrated legacy catalog and wait behind reference
reservations. The helper trusts its owning engine's byte and descriptor
verification; metadata alone does not authorize a network caller to publish.

Unclaimed offers and their rendition resources retain reachable references
across engines and scopes. An incomplete unclaimed offer blocks reservation.
Publisher exit or reattachment does not discard another engine's records or
these durable publications.

The `results.claim` capability transfers source protection to a fresh retained
owner in one transaction. It takes `scope`, `token`, `ownerKind` (`shown-result`
or `preview`), `ownerId` and the retention `client` binding. The returned
`owner` carries the offer's roots/completeness, revision 1 and `sourceToken`.
A missing source returns a null owner. A used owner identity, including an
empty revision tombstone, cannot be replaced by a repeated or lost claim;
reconcile it through `retention.owners.get` instead. Claiming invalidates the
client's acknowledgment and waits behind reference reservations.

After the first claim, an offer no longer retains independently. Its rendition
resources remain retained while any owner in the same engine/scope is bound to
that source. Ordinary owner patches and releases remove that source binding;
release keeps the revision tombstone. Other owners and engines retain their
own roots. A claimed source can supply a fresh owner or accept another rendition
while an existing source-bound owner holds exactly its roots/completeness.
Otherwise a metadata-only request returns null. Repeating the immutable offer
publication does not restore source protection.

The `results.sources` capability supports trusted-engine byte verification.
`results.sources.acquire` takes `scope` and `token`, holds the helper's OS
reference gate and returns `source`: null for a missing source, otherwise
`available` and `digests`. The latter includes source, rendition and transitive
dependency roots, or null for incomplete references. Always call
`results.sources.release` (no arguments), including after a null source.
An acquisition error releases the gate. Helper death releases it as well.
No SQLite transaction spans the engine's file work. Other helpers' reference
writers and collectors wait; this helper may complete its own reference writes.
A collection reservation cannot be nested with a source reservation.

The engine uses `dinkster_assets.records.verified_result_source` to acquire vault
reader ownership before the helper reservation and keep both through its read
or mutation. Unclaimed or exactly bound sources need no new byte proof.
Reviving a released source verifies every root with `open_verified`, resolving
vault files first and mounted files second. Missing or corrupt data refuses.
Within that context only, pass the yielded `verifiedDigests` to `results.claim`.
The helper requires an active source reservation and recomputes the exact closure
inside the claim transaction. Changed roots refuse instead of accepting stale
verification. Claim before calling `results.renditions.put`, which accepts no
byte proof and requires an unclaimed offer or an existing source-bound owner.
This keeps new rendition dependencies retained and avoids extending a released
source with unowned bytes that could prevent its next claim. These arguments
belong on the trusted engine's private pipe, never in an HTTP request; they are not durable
byte proofs and cannot be replayed after releasing the reservation. The helper
holds the gate independently of the engine's lifetime, so collection cannot
race a queued claim after engine death.

The `results.terminal` capability seals finished-run manifests.
`results.terminal.put` takes `scope`, `jobRef`, positive `attemptId`, `tokens`
(the exact publications in artifact order), terminal `state` (`completed`,
`failed` or `cancelled`), an opaque `identity` object and boolean
`artifactsComplete`. Every token must belong to this engine, scope, job and
attempt. Duplicate node/output identities, unavailable tokens and ambiguous
artifact order are refused rather than selecting the latest publication.
Artifact indices must be consecutive integers starting at zero. Artifact
publications require `artifactsComplete: true`.

Put returns `result` and `created`. The immutable result contains the selected
`resultOffers` by node/output and ordered `artifacts` (null when artifacts are
unknown), preserving all offer and identity fields. It is complete only for a
completed run with known artifacts and complete offers. An identical retry is
harmless; a different manifest for the same engine/scope/job is refused.
`results.terminal.get` takes the scope and job reference and returns `result`,
null when absent. Publication waits behind reference reservations without
holding a SQLite transaction while waiting. Reads remain available. Both
operations require an attached engine and refuse an unmigrated legacy catalog.

## Tab retention and holder census

The `retention` capability stores client generations and retained owners under
the helper's bound engine. These operations require `references.engine.attach`
and refuse an unmigrated legacy catalog. Use `results.claim` to transfer a
result token; direct owner patches do not claim or revive a source. A
disconnected client keeps the conservative census veto as well as its roots;
these operations do not expire inactive clients.

`retention.clients.initialize` takes boolean `pristine`, supplied only after
the engine checks its vault and scratch ownership. It establishes a new
engine's census; unknown or legacy ownership cannot later be declared empty.
Initialization and engine reattachment disconnect only this engine's clients,
invalidate acknowledgment and clear its pending request/stream counts. Neither
operation releases roots or changes another engine's clients or owners.

All other `retention.clients.*` operations require `principalId`. `open` takes
`revision: 0` for a new client, or an existing `generation` and its current
revision to rotate that client's generation. Rotation preserves the client ID
and owner bindings, refuses pending requests/streams and invalidates the old
generation. `get` takes the generation and returns its revision, connection
state, pending counts, `complete` and `holders` keyed by classes 1 through 8.
`acknowledge` takes the generation, revision and the exact complete holder
snapshot, including all eight classes. A stale, disconnected or busy client,
an incomplete owner or a changed census is refused. Boolean and numeric values
are not interchangeable in the acknowledgment.

`admit` and `settle` take the generation and optional boolean `stream` (false
for a request). Admission increments the pending count and invalidates the
acknowledgment. Settling a stream also invalidates it; settling a request does
not restore it. `disconnect` invalidates the acknowledgment without releasing
roots. These three operations return an empty object rather than enumerate
holders on the hot request path. Admission without a generation permanently
marks this engine's census legacy/unknown.

`retention.owners.get`, `put` and `release` take `scope`, `ownerKind` and
`ownerId`; kinds are `document`, `history`, `shown-result` and `preview`.
Put takes a revision (zero creates) and a patch of `digests` and/or `complete`.
Digests contain at most 4096 identities, sorted and deduplicated in storage.
Release takes the current positive revision and retains an empty, complete
revision tombstone. Both mutations accept `client: {principalId, generation,
holderClasses}`, with distinct classes from 1 through 8. An owner cannot move
to a different client or class set. Omitting a binding for a new owner marks
its engine's census unknown; a bound owner cannot be edited without its
binding. Mutations invalidate the bound client's acknowledgment atomically.
Unknown stored fields survive patches, releases and generation rotation.

Get and mutations return `owner`, null if absent. Owners are isolated by engine
and scope. Roots and their dependency closure remain reserved across every
engine, including after a disconnect. Unknown universes, unbound owners,
incomplete manifests or clients without an exact acknowledgment block
reservation. All mutations wait behind reference reservations without holding
a SQLite transaction across file work; reads remain available.

## Finished history

The `history` capability offers `history.append`, `history.get`, `history.list`,
`history.delete` and `history.clear`, all requiring a `scope`. Append takes a
terminal wire `record`, including explicit `assetDigests` (a list, or null for
unknown references), and a `project` defaulting to `default`. A run ID is
immutable: retry returns the original record and cannot reassign its scope or
project. Unknown document fields survive reads and retries. Delete and clear
remove document contents but retain the run identity, so retry cannot resurrect
a deleted record. Changes appear in the ordinary `changes` stream.

Get and delete take `id`. List and clear accept optional `project`, `clientId`,
`sourceDocument` and `state` filters. List returns `items` newest-finished first,
with a query-bound `cursor` and `limit` from 1 to 200. Clear additionally accepts
`before`, deleting only records finished strictly earlier than that timestamp.
Get and append return `record` plus `project`; an absent or deleted record is
null. Delete returns a boolean `deleted`; clear returns its deleted count.

## Content manifests

`dependencies.put` takes a parent `digest` and a `dependencies` array of
objects with canonical child digests. It retains every descriptor field,
including fields unknown to the helper. `dependencies.get` takes the parent
digest and returns `dependencies`, or null if absent.

`derivations.put` takes the content-derived cache key as `digest`, the opaque
`manifest` object and explicit `digests` for every retained input/output.
`derivations.get` returns `manifest`, or null if absent. The helper does not
interpret a document's contents to guess references. Put returns `created`;
identical retries are harmless, and conflicting documents or reference sets
are refused. A matching retry can fill missing references on an older
manifest, but a different document cannot. These operations are advertised
as `dependencies` and `derivations`. They refuse while an unmigrated
`library.sqlite` exists instead of serving an empty replacement store.

## Reference reservations

The `references` capability reserves saved records, catalog observations,
media grants, result publications, retained owners, finished history, derivations
and published engine references across helpers without holding a SQLite
transaction. It includes transitive dependencies reachable from those roots, handles cycles,
and does not retain an otherwise unreferenced dependency manifest's contents.
`references.acquire` returns a helper-bound `lease`, sorted `digests` and a
`cursor`; acquire and `references.page` accept `limit` from 1 to 200. Page
takes the lease and previous cursor. `references.release` takes the lease.
Unknown or incomplete manifests refuse acquisition. While reserved, saved-record,
catalog, grant, result-publication, retention-census, history, content-manifest and
engine-reference mutations wait; project edits and reads remain available.
The holder cannot mutate references until release. Release, pipe EOF and
process exit relinquish the OS lock; another helper cannot release it.
Collection must execute in the holder helper: an engine must not keep deleting
from a snapshot after its helper exits. This covers saved records, catalog
observations, media grants, finished history, content manifests and published engine references,
not the other catalog groups or vault ownership.

`references.engine.attach` binds the helper to one stable `engine` identity
(32 lowercase hex digits), with one live publisher per identity. It preserves
that engine's previous digests, increments its revision and marks it incomplete;
no other engine's manifest changes. `references.engine.get` reads the bound
manifest. `references.engine.patch` takes its current `revision` and a `patch`
of `digests` and/or `complete`. Unknown fields survive; unknown patch fields
and stale revisions are refused. Digests are canonical, sorted and deduplicated.
A helper cannot patch another engine. Its exit releases publisher ownership,
not retained references. Reattachment preserves those references but requires
the engine to publish a complete manifest again before collection. Producers
must mark their manifest incomplete before creating unreported references and
publish the complete set before relinquishing that protection.

History reads, mutations and reference acquisition do not import a legacy
history store. While an unmigrated `history.sqlite` exists they refuse with
`migration`, rather than serve a new, empty history. Engine history routes
retain their legacy storage until the complete record migration selects the
helper.

Helpers serialize writes with SQLite and reread records inside each write.
An OS lock identifies each live attachment. Format upgrades take an admission
lock, refuse while another helper is attached, and back up committed records
before changing them. Each attachment prunes format backups to the newest
three. Files held open by another program may remain until a later attachment;
cleanup failure does not prevent library access. Dead helpers release their
locks automatically. API and storage format numbers are separate; an engine
never opens the storage format.
