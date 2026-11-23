# Server integration

## Composing helper-backed stores

Embedding hosts may supply a caller-owned `LibraryClient` to pack settings,
runtime settings, principal permissions, delegations and finished history.
Keep that client alive for its consumers and close it yourself, not through
application cleanup. Follow the shared [helper lifetime, trust and publication
requirements](library-operations.md).

With `RuntimeSettings(path=client, environment=...)`, do not also supply
`persisted_values`. Compose queues, governors and workers from the resulting
effective settings, as with file-backed startup. Reading preferences does not
apply a peer's settings to a running engine. On a settings conflict, refresh
before retrying; for principal permissions, reread the principal list.
Preserve unknown stored fields rather than replacing extensible records with
an older schema's projected view.

Assign accepted-work storage to one engine. Helper-backed finished history
requires an engine-owned `accepted_path`; the host owns that isolation and
reference publication. Do not collect shared files through engine-side history:
only the reservation-holding helper may collect them. See
[history storage and recovery](../packages/dinkster-server/src/dinkster_server/history.py)
for the supported constructors and transaction behavior.

## Asset consumers

Embedding hosts must supply `ServerLibrary.local_resolver` for granted local
mount availability checks and job preflight. Do not substitute a resolver that
may fetch bytes. The source contract is in
[ServerLibrary](../packages/dinkster-server/src/dinkster_server/library.py).

When combining project upload observations with federated candidates, merge by
digest while retaining each source's names, locations and authority separately.
An observation describes ingestion, not model-loader compatibility: a generic
transfer or model upload remains `asset/file` even with a model-like name.
Get logical model kind, variant and compatibility from the resolution catalog.
A local copy cannot turn unknown compatibility into compatible. Do not copy
project observations into the project-unaware resolution store. Caller-reported
source-machine labels are advisory history, not verified machine or filesystem
authority.

## Result publications

Use the recorded publication routes rather than rebuilding a result from a
current cache entry. The API implementation for output offers and sealed
terminal results is in
[library routes](../packages/dinkster-server/src/dinkster_server/library.py).
For producing documents, follow the existing
[lease and resource ownership procedure](image-document-operations.md).
For helper reference publication and collection, follow
[produced-byte retention](library-operations.md#produced-byte-retention-and-collection).
