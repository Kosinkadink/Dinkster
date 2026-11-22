# Dinkster library

`dinkster-library` serves shared project records over a private child-process
pipe. It opens `records.sqlite` in the selected library; engines do not open
that database. Existing asset records remain in `library.sqlite`.

## Record conversion internals

The private migration staging code is not called by ordinary startup or exposed
on the pipe. It holds the direct-engine, account-settings and helper-admission
gates while taking a verified source snapshot under `backups/before-migration-*`.
SQLite's backup API includes committed WAL pages. A durable journal binds retries
to that first complete snapshot; missing or changed backup files refuse rather
than falling back to partially replaced live stores.

Staging copies existing helper projects and converts legacy asset, finished-history
and permission databases into an unpublished database. It preserves record IDs,
revisions, workflow placement, client generations, activity/departure fields,
holder-class bindings and quarantine. Missing tables required by the legacy asset
revision refuse conversion. File-backed project folders, grants, environment and
pack preferences, provenance layers, resolver subscriptions and acquisition
receipts convert from that snapshot into the temporary records database. Unknown
record fields and resolver regional leads survive. Environment preferences require
an explicit environment ID, separate from the engine identity. Account model
locations and exact-file selections stage in `account/model-settings.sqlite`,
with location order and extension fields preserved. The live account is never
opened for conversion. A pre-existing account database is copied at its supported
format; competing JSON and database sources refuse rather than choose one.
Staging does not publish records, remove legacy stores, create a stop-file or
declare helper-era engine support. Backup reads are immutable and unlisted files
refuse conversion.

The same frozen snapshot includes engine working state: execution journals,
collaboration sessions, worker snapshots/indexes and remote values. The temporary
conversion tree contains an `engine-state/` directory with those files and an
`accepted.sqlite` preserving the original accepted-job schema and rows, without
finished history. Nested SQLite stores use logical backups including committed
WAL pages; stale WAL/SHM files are not copied over those backups. A killed
conversion rebuilds from the verified snapshot, not live working files. Backups
without working-state coverage refuse this conversion rather than dropping work.
This staging path does not publish engine state, reset live sessions, recover
jobs, move vault uploads or change ordinary startup.

## Explicit public acquisition receipts

The `sources.receipts` capability supplies immutable `put`, `get` and `list`
operations. `PublicAcquisitionReceiptStore(helper)` uses the caller-owned helper
instead of a JSON file. Its existing verified-fetch, declaration, size and public
HTTPS checks still run before recording. The helper validates the receipt fields
but does not fetch bytes or establish the engine's attestation itself.

A repeated identical raw receipt is idempotent; changing an existing ID refuses.
Unknown fields remain in the raw record while the consumer reads its known v1
view. Pages contain at most 100 records ordered by timestamp and receipt ID;
cursors bind to the digest filter. A receipt is bounded to 128 KiB and the store
to 100,000 records. Committed new receipts appear in the change feed. Existing
legacy asset, provenance, resolver or receipt entries refuse before any receipt
operation; nothing is imported or hidden. Ordinary startup still constructs the
legacy file-backed store. These records contain provenance, not asset bytes.

## Explicit settings consumers

The `settings` capability supplies `settings.get` and `settings.patch` for exactly
one `environment` or `pack` identifier. Replies contain `settings` with the
identity, `revision` and `values`. An absent record has revision 0 and empty values.
A patch requires the last read revision; a stale write changes nothing. Nested
objects merge, including unknown fields and null values. Explicit `remove` paths
are arrays of field names. Invalid removals roll back the entire patch. The change
feed records committed settings revisions. Writes exceeding a 1 MiB encoded reply
refuse before committing.

`RuntimeSettings(path=helper, environment=...)` and `PackSettingsStore(helper)`
use these operations when explicitly supplied a client. Runtime settings refresh
the edit revision without applying a peer's policy to an already-running engine.
Known fields validate under the consumer's schema; unknown fields stay stored.
Stale HTTP edits return 409 and unavailable helper state returns 503. The caller
owns the supplied helper's lifetime. Ordinary startup still supplies legacy paths.
Existing `library.sqlite`, `settings.json` or `pack-settings` entries refuse these
operations until complete migration; no file-backed settings are imported here.

## Starting and stopping

`LibraryClient(library)` starts a helper, negotiates API 1, and closes the pipe
when its context exits. `AsyncLibraryClient.start(library)` provides the same
calls without blocking the event loop; call `close()` when finished. A helper
exits on pipe EOF even while a database write is blocked.

Managed engine launches set `DINKSTER_LIBRARY_COMMAND` to a JSON command array
using the tools interpreter. With no override, the client runs
`sys.executable -m dinkster_library`. The command accepts `--library PATH`.
No listener, credential, operating-system service or login item is created.

A request is a JSON object with `api`, `operation` and `arguments`. A reply
holds either `result` or `error: {code, message}`. Ordinary calls use one line,
bounded to 16 MiB. Helpers advertising `transport.chunks` also accept a sequence
of JSON frames with base64 `chunk` (1 byte to 1 MiB decoded), exact byte `offset`
and boolean `complete`. The final frame completes one serialized request;
no operation runs on an incomplete transfer. An invalid frame discards the
transfer with one error; remaining frames are ignored through its final frame
or the next ordinary request. A malformed ordinary line ends that discard with
its own error. An oversized line is drained through its newline without blocking
later requests. A malformed helper reply kills that helper, so the
client must reopen rather than reuse uncertain pipe state. EOF also discards an
incomplete transfer with the helper. A request sets `transport: "chunks"` to accept
framed replies, which use the same encoding. Without that opt-in, replies stay
ordinary lines and an oversized reply returns a message error. `LibraryClient`
negotiates the capability and handles both directions transparently. Older
helpers still accept ordinary calls; large calls require the capability.

Calls remain serialized per client and normally limited to 60 seconds for the
entire transfer and operation, not per chunk frame. Helpers advertising
`transport.progress` accept `progress: true` on a request. A collection or
scratch-retirement request that opts in can receive `{progress: true}` lines
before its final reply. These
renew the client's 60-second inactivity deadline, not its reservation or deletion
authority. The helper sends them after completed hash chunks and metadata items,
at most ten times per second; no background heartbeat hides a stalled read.
`LibraryClient` requests them only for `references.collect` and
`references.retire` when advertised.
Older clients receive no extra frames and older helpers retain their fixed
request limit. Ordinary calls keep the total deadline.

A timeout kills the helper; callers must reopen and reread before attempting
another write, because losing a reply does not prove a write failed. Documents
are assembled in helper memory before publication; available memory and SQLite's
native row limit still apply.

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
The fingerprint uses device, file identity, size and nanosecond modification
and change times. Windows obtains change time from an open descriptor, not a
path query's creation time. Older Windows fingerprints and verification records
without that change time require hashing again before verified reuse.
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
`q` (name substring), `label`, `limit`, `cursor` and `after`. It returns
`records` and `cursor` (null at the end), ordered by modified time then ID
descending.
Workflow records include their `project`; legacy workflows belong to `default`.
For the legacy catalog a project filter selects only that live project's
workflows, not generic records. Helper-owned saved records all carry a project,
so the filter includes that project's named assets as well.

`assets.list` requires `scope` and accepts `project` (defaulting to `default`),
`q`, exact canonical `digest`, namespaced `kind`, `origin`, inclusive numeric
`since`/`until`, `limit`, `cursor` and `after`. It returns `assets` and `cursor`,
ordered by latest observation then digest descending. Each entry contains the existing `asset` wire form,
`aliases` and `observations`. Matching any observation retains that digest's
other aliases and observations within the same scope and project. It does not
read asset bytes or inventory unrecorded files.

Both operations accept page limits from 1 to 200, bind cursors to every filter,
and do not create or migrate the legacy asset catalog. As an alternative to a
cursor, `after` is a two-element list containing a finite timestamp and a nonempty
ID (record ID for `records.list`, digest for `assets.list`). It excludes that
position and follows the same descending order, including ties. Supplying both
positions refuses. The capabilities `records.list.after`, `assets.list.after`
and `assets.list.digest` advertise these optional fields. Cursors issued without
a digest filter remain valid when that filter is omitted or empty. When the
legacy catalog is absent, `records.list` reads helper-owned saved records and
`assets.list` reads the helper-owned ingest catalog.

`assets.recorded` accepts only `digests`, a list of at most 200 canonical digests,
and returns its sorted distinct subset found in live saved records, media grants
or ingest observations across every scope and project. Empty lists are allowed.
The `assets.recorded` capability advertises the operation. This is operator
inventory membership, not scoped membership or permission to read bytes; the
trusted engine must authorize any operator-facing route that uses it. Other
reference roots alone do not count as catalog membership. The helper reads the
legacy catalog when present, otherwise its own records, without changing either.
Files remain direct: the host inventories and verifies its vault, then uses
bounded calls to exclude already recorded digests. This operation neither scans
files nor grants collection authority.

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
1 MiB. There is no separate 64 MiB envelope cap: provenance derived from a
bounded index can be larger than its input. `cancel` discards staging; helper
exit also discards it without changing any published record. Failed commits
leave staging available for cancellation.

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
64 subscriptions can be active, and active sources are unique. Unknown envelope,
subscription and index fields survive refreshes that omit them. Unknown fields
on retained entry digests survive too; known fields and the entry set are
replaced, so removed entries do not remain as leads. Snapshot replacement is explicit,
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

The `sources.resolvers.transactions` capability adds `state` and `patch`.
`state` returns active `items` (`id`, `revision`, `snapshot`) in original
subscription insertion order and nullable `officialBootstrap`, all from one
read transaction. Optional `revisions` maps up to 64 cached IDs to their revisions;
matching items omit `snapshot`, so unchanged indexes need no payload read or pipe
transfer. Missing IDs are no longer active. `patch` accepts `updates` (`id`, expected `revision`,
`snapshot`) and `remove` (`id`, expected `revision`), with 1 to 64 distinct IDs
across both lists. Optional `officialBootstrap` must name an updated subscription.
All removals, publications, derived leads, bootstrap and events commit in one
write transaction, or none do; `revisions` maps every changed ID to its new
revision. Existing single-upload commit/remove calls use the same writers.
Large state and patch documents use negotiated pipe chunks; available memory
and native database limits still apply. Fetching, parsing resolver semantics
and regional lead derivation remain with the engine, before publication.

A refused publication or withdrawal rolls back snapshots, leads and events
together. Resolver writes remain available during reference reservations and
create no asset roots. All operations refuse unmigrated `library.sqlite`, `provenance.json`
or `resolver-indexes.json` without touching them.

`sources.resolvers.bootstrap.get` returns `configuration`, null until recorded.
`bootstrap.put` records the immutable official-provider choice using a
`configuration` object with `source`, `providerId` and `subscriptionId`; unknown
fields survive. It returns the stored configuration and `created`. A retry may
omit unknown fields, but cannot change a recorded value. The engine validates
URL and provider policy; the helper bounds the strings and document and checks
that an active subscription with this ID has the same source and index name.
A bootstrap may outlive its subscription. Removing a subscription never clears
it, and reading it never recreates an unsubscribed provider. Consumers using the
engine's strict `_OfficialResolverBootstrap.from_wire` project only `source`,
`providerId` and `subscriptionId` for parsing, retaining the raw stored document
rather than replacing it with the strict parser's output.

The first subscription's `commit` may include `officialBootstrap` with the same
configuration shape. Snapshot, leads, bootstrap and changes then commit or roll
back together. Every later publication of the recorded subscription must still
match its official source and provider ID. A separate `bootstrap.put` supports
an already-subscribed provider or an imported choice that has been unsubscribed.
Bootstrap writes create a `resolver-configuration` change only on first insert,
remain independent of reference reservations, and share resolver legacy refusals.
Raw staged snapshot uploads refuse duplicate field names at every nesting depth.
`patch` receives snapshot objects through the ordinary JSON request parser;
duplicate request fields have already collapsed to their last value. Engine
writers serialize objects with unique keys.

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
merge in order. For a digest retained in that source, known provenance fields
are replaced and omitted unknown top-level fields survive. Digests absent from
the replacement are withdrawn. Layers apply after the base in original insertion
order; replacing a layer does not move it. `remove` withdraws a named layer and returns
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

Raw and merged provenance have no artificial record-byte or shared row-count
budget, matching the legacy store. Each resolver keeps its own entry limit.
Large writes, reads and list replies use the negotiated pipe framing; a large
record never becomes unreadable merely because one JSON line cannot hold it.
Withdrawals may expose larger previously overridden values without refusing.
Provenance writes do not wait on asset reference reservations and never fetch
or validate remote bytes. All operations refuse unmigrated `library.sqlite`, `provenance.json`
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

## Atomic folder records

The `folders.transactions` capability adds `folders.get` and `folders.patch`.
A get takes no arguments and returns `{grants, projects}`, each a
`{revision, values}` record, from one read snapshot. A patch requires both
`grants` and `projects` objects, each with the existing `revision`, optional
`patch` and optional `remove` fields. Both revisions must match. Recursive
patches, explicit removals, both revision increments and both folder change
events commit in one transaction. Any stale revision, malformed change or
legacy-store refusal leaves both records and the change feed untouched.
An empty patch still advances that record, as with individual preference writes.

These operations share the individual folder readers and writers, including
unknown-field preservation and the negotiated large-message transport. They
refuse `mounts.toml`, `project-folders.json` or `library.sqlite` without touching
legacy bytes and do not take the asset reference reservation. Validation and
operator authorization of grants and membership remain engine responsibilities;
this transaction provides storage atomicity, not new filesystem authority.

`MountService(..., library=client)` explicitly opts into these joint records.
Grant creation and project membership commit together before live publication;
project edits also guard the grant revision used for validation. Stale edits
return HTTP 409 without publishing a live grant or worker snapshot; unavailable
helper reads return HTTP 503. Peer grant changes are refreshed on folder access.
Live engines preserve unchanged scans and scan newly observed grants; engines
configured for restart-only changes update settings without changing their live
table or worker snapshot. The client remains caller-owned. Without this explicit
argument the service retains legacy TOML/JSON storage; startup does not opt in.

## Filesystem grant configuration

The `folders.grants` capability provides `folders.grants.get` and
`folders.grants.patch` for library-local filesystem grants. The response is
`folders: {revision, values}`; a missing record is revision 0 with empty values.
Its patch/remove protocol and transaction boundaries match project preferences
below, with change identity `["folders", "grants"]`. It uses the same large-message
transport and is independent of reference reservations. Existing `mounts.toml`
or `library.sqlite` refuses reads and writes without touching legacy bytes.

Values use the existing mount configuration structure: `mounts` maps IDs to
`path`, `mode`, `priority` and `kind`, while `settings` contains nullable
`output-mount`. Unknown top-level, settings and retained mount fields survive
patches; removing a mount removes that mount's extensions too. The helper stores
configuration, not filesystem authority: the engine validates known fields and
the authenticated operator's permission before changing live grants. It neither
creates directories nor scans, opens or writes the granted paths.

`MountConfiguration.load(client)` returns an immutable, revision-bound engine
view, using the existing mount validator while ignoring stored extensions.
Its `save(mounts, output_mount=...)` validates the complete candidate, patches
changed mounts/settings and explicit removals, and returns a new view. Keeping
the old view does not permit reuse: a stale save refuses without changing records
or the changes feed. The caller owns the client. Existing TOML readers retain
strict unknown-field validation. Library grants are separate from account-wide
model settings and from an engine's live table, scan state or worker snapshot.

## Account-wide model preferences

The `folders.models` capability provides `folders.models.get/patch`, with the
same `{folders: {revision, values}}` reply and recursive `patch`/explicit
`remove` protocol. Helpers use `DINKSTER_HOME/model-settings.sqlite`, not the
attached library, so separate libraries on one account observe the same model
locations and exact-file selections. Another account remains independent.
The account database has its own format integer and refuses an unsupported
format. Existing account `model-settings.json` or `model-file-selections.json`
refuses reads and writes until complete migration; neither is imported here.

Values contain `locations` keyed by location ID, an `order` array,
`downloadLocation`, and `files` keyed by exact-file virtual path. Unknown root,
location and retained file fields survive. Removing a location removes its
extensions; remembered files are filtered against the current grants and never
grant directory authority. Reads and patches take `model-settings.lock`, also
held by record migration, and return `busy` while another holder owns it. Each
patch checks the account revision under the SQLite write lock. No account change
is emitted in a library's change feed;
consumers reread the account record. Negotiated chunk transport applies.

`ModelSettings.load(client)` and `shared_model_mounts(client)` opt into this
storage. `MountService(..., library=client, shared_models=True)` uses it for model
settings, operator routes and download selection persistence, while project
preferences remain library-local. The caller owns the client. Model edit
revisions are opaque strings and include account edits; stale saves return 409.
Helper or migration failures return 503 on model-management routes without a
legacy fallback. Restart-only engines keep their active model grants until
restart. Without a client, JSON behavior is unchanged; startup remains legacy.

## Project folder preferences

The `folders.projects` capability provides `folders.projects.get` and
`folders.projects.patch` for one library's project folder preferences.
Get takes no arguments and returns `folders: {revision, values}`; a new
library has revision 0 and empty values. Patch takes `revision`, `patch`
and optional `remove`, with the recursive merge, explicit removal and atomic
stale-write refusal described under settings below. Successful patches advance
the revision and publish a `folders` change with identity `["folders", "projects"]`
in the same transaction. These preferences use negotiated large-message
transport rather than the settings-specific 1 MiB limit. Native database and
memory limits still apply. They remain writable during reference reservations.

`ProjectFolders(client)` uses a caller-owned `LibraryClient`, requiring this
capability. Every public read or edit synchronizes first. The engine validates
mount IDs, project IDs, complete model-location groups, required folders,
output membership and hidden-model identities. Writes patch only changed
project fields and preserve unknown top-level and retained hidden-model fields;
removing a hidden-model identity removes that preference, including extensions.
Concurrent stale edits refuse without overwriting another engine's choices.
Initialization alone retries stale publication up to three attempts, re-reading
and recomputing defaults each time so simultaneous starts preserve saved choices.
Callers serialize operations on a `ProjectFolders` instance. The path-based
constructor keeps its existing JSON behavior. Neither constructor closes a
caller-owned client.

The library preference document carries `version`, `shared`, `projects`,
`excluded`, `outputs` and `hiddenModels`. Null output differs from an absent
output: null disables fallback, while removal restores it. Preferences do not
grant filesystem access and do not contain the account-wide model-location
configuration or engine worker snapshots. Unmigrated `library.sqlite` or
`project-folders.json` refuses reads and writes without touching legacy bytes.

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

## Cleanup policy

`cleanup.policy.get` returns `policy: {revision, values}` for the library-wide
size preference. A new record has revision 0 and `limitBytes: 21474836480`.
`cleanup.policy.patch` uses the settings `revision`, recursive `patch` and
explicit `remove` contract, retaining unknown fields. `limitBytes` must remain
present and be null (unlimited) or an integer from 1 through 2**53 - 1; booleans
are not integers. Invalid or stale changes commit nothing. Changes use kind
`policy` and identity `["cleanup", "policy"]`. Legacy `library.sqlite` refuses
with `migration`; calls never import it or grant collection authority.

`dinkster_server.cleanup.CleanupPolicy(client)` is a caller-owned helper
consumer accepted by `ServerLibrary(cleanup_policy=...)`. Storage GET refreshes
that server's edit revision; PUT patches it, returning 409 on a peer change and
503 on helper failure without SQLite fallback. Background policy reads do not
refresh the edit revision. This view is per server, not per browser. Clear and
size enforcement report `helper_collection_unavailable`, and scratch recovery
keeps temporary files rather than collecting through the legacy store. The
ordinary file-backed startup path is unchanged.

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
null. Append also returns `created`, true only for the transaction that first
stored that run. Delete returns a boolean `deleted`; clear returns its deleted
count. List can start at an explicit `after: [finishedAt, runId]` position instead
of a cursor, allowing engine HTTP cursors to retain their own encoding. A later
page uses the returned query-bound cursor; supplying both is refused.

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

## Produced-asset ownership

`asset_identity`, `produced_assets` and `produced_scratch` provide the common
content identity, adoption journal and OS leases used by engines and helpers.
The asset package re-exports these same objects; there is one lease context
and one journal format regardless of which package a consumer imports. These
primitives do not depend on the engine or its value/cache packages. Holding
a vault claim proves writer liveness only; collection also requires the
library reference reservation.

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
`references.collect` (capability `references.collect`) takes the active `lease`,
explicit absolute `vaults`, optional boolean `collect` (default false) and
nullable nonnegative `limitBytes`. Only the helper holding that lease can run
it. Vaults are trusted host-selected produced-asset stores, including retained
preview locations; this is not a filesystem authority grant to browser clients.
Every vault must have readable adoption and lease metadata and no active
publisher or reader. Linked paths, duplicate directories and pending adoptions
refuse before deletion starts. All vault claims remain held together. The helper
verifies unreferenced content before deleting it and its regular sidecars,
then retires the exact adoption row. A total byte limit spans all supplied
vaults, and omitted/null means collect every eligible adoption. Measurement
never removes bytes. The response gives reclaimed/reclaimable bytes and files,
retained files, failed items, remaining vault bytes and adopted bytes. Candidate
verification failure leaves its row and bytes, reports a failed item and makes
reclaimable bytes unknown. Non-reference project writes remain available while
hashing or deleting; no SQLite write transaction spans that work.

Collection executes in the holder helper: an engine cannot keep deleting
from a snapshot after its helper exits. The reference reservation covers saved
records, catalog observations, media grants, finished history, content manifests
and published engine references. The separate vault claims protect managed
readers and publishers; neither kind of reservation replaces the other.

`references.retire` (capability `references.retire`) takes the active `lease`,
absolute `scratch` root, `scope` (`packs` or `groups`), one `pack` directory name,
`job`, positive `attempt`, and the complete `vaults` list. The helper must also
be bound to an engine. The trusted host supplies only that engine's assigned
scratch location, never browser-selected paths. This operation retires exactly
one canonical job-attempt store; it does not discover or reset other engines'
working state. Missing ownership metadata, a mismatched journal owner, links,
live host jobs, scratch writers/readers and busy vaults refuse without removal.
Every offered publication must occur in the reserved reference snapshot and
have an adopted, size- and digest-verified durable copy in a supplied vault.
All handoffs are verified before removing any temporary file. Unoffered work
needs no durable copy. The helper retains the job, scratch and vault claims
through file removal and exact publication-journal retirement, without a SQLite
write transaction. It reports `reclaimedBytes` and `reclaimedFiles`, preserves
the ownership journal and can retry after partial removal or helper death.
A completed retry reports only files it actually removes. Durable copies and
the reference snapshot stay intact; the engine can later republish its smaller
scratch root set. Ordinary startup and HTTP cleanup do not yet call this
operation.

`dinkster_server.cleanup.EngineReferences` accepts the complete host-assigned
produced vault set as `vaults=(main, *previews)`. Its asynchronous `inspect`
publishes local references, takes the bound engine's retention census, and
reserves the library snapshot before measuring or collecting those vaults.
The returned storage status includes the census's library-wide `quarantine`
counts (`departedClients`, `owners`, `bytes`), or null if reference inspection
fails. `collect=True` first retires inactive scratch through `references.retire` and
then collects through `references.collect`; no engine-side deletion or legacy
record-store fallback is used. `limit_bytes` applies across the vault set,
not temporary scratch. Missing vault configuration, incomplete manifests,
unknown cache references, or failed helper calls refuse collection. A failed
mutation reply reports reclaimed counts as unknown, not zero. Cancellation
waits for the owned worker to finish and release the reservation. During that
inspection of a complete reference view, cache and job admissions refuse
locally with `busy`, rather than wait for a hashing request's pipe lock on the
server event loop. Admission resumes only after the owned worker settles,
including on cancellation or failure. A confirmed incomplete view cannot be
reserved, so an inspection of pending accepted work does not block execution,
cache writes or new admissions; collection still refuses until a fresh complete
view is published. The helper remains caller-owned. Ordinary startup does not
select this consumer; storage HTTP routes use it only when explicitly configured.

`references.engine.attach` binds the helper to one stable `engine` identity
(32 lowercase hex digits), with one live publisher per identity. It preserves
that engine's previous digests, increments its revision and marks it incomplete;
no other engine's manifest changes. `references.engine.get` reads the bound
manifest. `references.engine.patch` takes its current `revision` and a `patch`
of `digests`, `complete` and/or `vaults`. Unknown fields survive; unknown patch
fields and stale revisions are refused. Digests are canonical, sorted and
deduplicated. With capability `references.engine.vaults`, absolute produced-vault
paths accumulate in the engine manifest across reattachments; patches cannot
remove a previous location. At acceptance, non-linked paths naming the same
existing directory coalesce to its previously stored spelling. An existing
location reporting a zero device or inode identity refuses registration rather
than discarding a potentially distinct directory. This does not relax the
collection boundary's duplicate-directory refusal. As with the account digest
index, nonzero device/inode identities must correctly identify files; arbitrary
collisions from a faulty filesystem are not detectable from this metadata.
Filesystem identity checks hold the reference barrier but no SQLite transaction;
the transaction rereads the document and checks its revision before committing. The bound
engine's census, scratch retirement and collection must include every accumulated
vault, even a previous preview location. Missing or unsafe previous locations
refuse inspection rather than exempting clients without their complete
produced-byte snapshot. Retained locations must not be renamed or deleted. If
moved, restore their original directory, inventory and lease metadata before
inspection; a missing inventory must not be recreated as empty. `EngineReferences`
persists the current configuration on attachment and uses the accumulated set
for inspection; server composition still checks its current configuration
exactly. Vault paths are local records, not collection authority by themselves.
A helper cannot patch another engine. Its exit releases publisher ownership,
not retained references. With capability `references.engine.lifetime`, an engine
can hold `EngineLifetime(library, engine)` in its own process for its entire
lifetime and send its random `token` as the optional attach field `lifetime`.
The helper checks that lock without owning it and persists the binding. Once
bound, reattachment requires the previous lifetime to have ended and a new held
lifetime; omitting the field or replaying a released token refuses. This prevents
helper death alone from resetting a live engine's pending reads and streams.
`EngineReferences(..., lifetime=...)` forwards the caller-owned lifetime; callers
must keep it held until engine exit, not merely until helper exit. Engines that
have never used a lifetime binding retain the original attach behavior. This
binding alone does not recover a stopped engine; census requires the independent
publisher and vault claims described above. Reattachment preserves those
references but requires the engine to publish a complete manifest again before
collection. Producers
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
