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
A project filter selects only that live project's workflows, not generic records.

`assets.list` requires `scope` and accepts `project` (defaulting to `default`),
`q`, namespaced `kind`, `origin`, inclusive numeric `since`/`until`, `limit`
and `cursor`. It returns `assets` and `cursor`, ordered by latest observation
then digest descending. Each entry contains the existing `asset` wire form,
`aliases` and `observations`. Matching any observation retains that digest's
other aliases and observations within the same scope and project. It does not
read asset bytes or inventory unrecorded files.

Both operations accept page limits from 1 to 200, bind cursors to every filter,
and read the existing asset catalog without creating or migrating it.

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

The `references` capability reserves finished history, derivations and published
engine references across helpers without holding a SQLite transaction. It
includes transitive dependencies reachable from those roots, handles cycles,
and does not retain an otherwise unreferenced dependency manifest's contents.
`references.acquire` returns a helper-bound `lease`, sorted `digests` and a
`cursor`; acquire and `references.page` accept `limit` from 1 to 200. Page
takes the lease and previous cursor. `references.release` takes the lease.
Unknown or incomplete manifests refuse acquisition. While reserved, history,
content-manifest and engine-reference mutations wait; project edits and reads
remain available.
The holder cannot mutate references until release. Release, pipe EOF and
process exit relinquish the OS lock; another helper cannot release it.
Collection must execute in the holder helper: an engine must not keep deleting
from a snapshot after its helper exits. This covers finished history, content
manifests and published engine references, not the other catalog groups or
vault ownership.

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
