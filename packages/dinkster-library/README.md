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
This operation does not scan folders or change engine mount resolution.

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
before changing them. Dead helpers release their locks automatically. API and
storage format numbers are separate; an engine never opens the storage format.
