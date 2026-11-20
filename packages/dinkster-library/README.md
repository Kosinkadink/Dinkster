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

Offers and rendition resources retain their reachable references across all
engines and scopes. An incomplete offer blocks reservation. Publisher exit or
reattachment does not discard another engine's records or these durable
publications. These immutable publication operations do not transfer ownership
to a tab, release roots or revive collected bytes.

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
and refuse an unmigrated legacy catalog. They do not claim result tokens or
revive collected bytes. A disconnected client keeps the conservative census
veto as well as its roots; these operations do not expire inactive clients.

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
