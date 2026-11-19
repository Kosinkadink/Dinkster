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
to `libraryApi`; this engine needs folder scanning, progress and selected-file scanning.
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

## Records

`hello` returns the API range, capabilities, storage format and other attached
helpers. `projects.register` creates an ID if absent; `projects.get`,
`projects.list`, `projects.patch` and `projects.delete` read and edit it.
Patches carry the last observed `revision`; a conflict returns `stale` without
changing the record. Immutable identity fields cannot be patched.

`workflows.record` keeps project ownership and engine-supplied facts beside
legacy workflow records. `workflows.get` and `projects.needs` read those
associations; `workflows.read` reads a legacy record for requirements checks.
Workflow contents are read from the vault by the
engine, not interpreted by the helper. Facts whose digest no longer matches
are unknown. `engines.seen` and `engines.list` retain environment/snapshot
identities. Lists use cursors; `changes` pages through committed revisions.

Helpers serialize writes with SQLite and reread records inside each write.
An OS lock identifies each live attachment. Format upgrades take an admission
lock, refuse while another helper is attached, and back up committed records
before changing them. Each attachment prunes format backups to the newest
three. Files held open by another program may remain until a later attachment;
cleanup failure does not prevent library access. Dead helpers release their
locks automatically. API and storage format numbers are separate; an engine
never opens the storage format.
