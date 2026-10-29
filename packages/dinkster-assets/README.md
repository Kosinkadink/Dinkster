# dinkster-assets

`dinkster-assets` models files by canonical content digest rather than host
path. It provides asset references, catalogs, mounts, libraries, verified
vaults and acquisition, save targets, and value-type registration. It uses
values and BLAKE3 for identity, and SQLAlchemy and Alembic for catalog schema
migrations. Caches, workers, and the server build on it.

Selected indexed files use ordinary `AssetRef` values from the engine-written
index. The live mounts snapshot remains their resolver, so revoked grants stop
resolving without a worker restart. Unindexed selections retain invocation-local
`LocalAssetRef` bindings and cannot be admitted as portable VIDEO until scanned.
Admission never copies selected input files into the vault or worker scratch.
`produced_output_scratch` is only the writable publication store for produced
outputs; uploads, downloads and remote transfers retain their separate vault paths.

## Setup

This package is a uv workspace member and is not published separately yet.
From the repository root, install the complete workspace:

```sh
uv sync --all-packages
```

It does not install a console script.

## Use

Verified content identity is the canonical `blake3:<64 hex>` digest:

```python
from dinkster_assets import AssetRef, digest_bytes

payload = b"example asset bytes"
ref = AssetRef(
    digest=digest_bytes(payload),
    name="example.bin",
    size=len(payload),
)
```

`MountTable.bind_name(name, kind)` can bind a bare model filename before
hashing. It returns a `LocalAssetRef` marked `verification: "unverified"`
with no digest, or a verified `AssetRef` when the scan already knows the
same file revision. Local bindings cross worker boundaries as opaque,
server-scoped grants, not filesystem paths or content identities. File
replacement and mount revocation invalidate them. Their cache fingerprints
use an `unverified-local:` prefix and cannot be redeemed on another host.

For `model/text-encoder`, the same method also accepts slash-separated
source-relative filenames. These select an exact file in the first ranked
exact-kind root, without basename ambiguity. Absolute paths, dot/parent
components and links escaping the grant are rejected. Ordinary ASSET inputs
and filename preflight use the same existing file carriers and grants.

`asset_selection_names(kind)` enumerates source-relative options from live
exact-kind grants. `resolve_asset_selection(kind, name)` returns a checked
directory subtree, an existing invocation-local file reference, an
encoder/decoder pair of those references, or `None` for the file-free
`pixel_space` option. Pair members must coexist in one `model/vae-approx`
root. An existing `AssetRef` or `LocalAssetRef` is returned unchanged only
when its resolved file is an admitted option in the current exact-kind
grants; its verification policy is unchanged. Resolution never searches a
library, vault or ungranted directory for a selection name.

`model/diffusers` discovers `model_index.json`, including the granted root
itself as `.`. `directory/input` lists recursive non-root directories without
following directory links. `directory/dataset` lists non-root directories
containing `metadata.json` or a `.safetensors` file and stops descending below
matches. Dataset lookup selects the first ranked existing same-named directory,
even when only a later root supplied the discovery marker. Selected subtrees
reject escaping links, cycles and non-file members.
The returned paths and pair references are invocation-scoped, not graph values.
`asset_selection_options` exposes only option names and shapes for pickers.

Library scans reject partial-download extensions even with custom ignore
rules and admit files only after two size/mtime observations agree. Unchanged
verified rows avoid rehashing. Missing or replaced identities remain in the
existing index, available through `MountTable.missing_entries(mount_id)` but
excluded from live resolution. A returning digest recovers its identity;
changed bytes get a distinct digest without rebinding the old reference.

Catalogs give virtual paths file-like listing and glob semantics without
exposing real filesystem paths:

```python
from dinkster_assets import AssetCatalog, AssetEntry

catalog = AssetCatalog([
    AssetEntry("models/example.bin", ref.digest, ref.size),
])
matches = catalog.glob("models/**")
```

Combined model files use the conventional `model/checkpoint` kind. Their
optional `AssetComponentManifest` is an ordered declaration of contained
model kinds and advisory architecture, dtype, and JSON metadata. Use
`asset_kind_matches` or `AssetNeed.matches_kind` when a checkpoint should
also match a context that accepts one of its declared components. The
`AssetNeed` wire field is an ordered `components` array of objects with
required `kind` and optional `architecture`, `dtype`, and `metadata` fields;
it is omitted when no manifest was declared.

Other exports cover `AssetVault`, `LocalAssetLibrary`, `MountTable`,
`AssetNeed`, verified `acquire_need`, `AssetWriter`, and registration of the
`dinkster.asset` and `dinkster.save_target` value types. `AssetWriter.save_stream`
publishes bounded seekable sources atomically without buffering the complete
file. `parse_latent_asset` strictly validates native and ComfyUI latent
safetensors headers before tensor data is materialized.

`AssetWriter.save_named_stream(target, source, limit=...)` treats the target's
prefix as the complete relative filename for an explicit import. It preserves
that name when unused and adds a numbered suffix before the extension on a
collision, never overwriting an existing file. It uses the same live writable
mount, containment, bounded streaming and digest-index publication as output
saves. Portable filename rules also reject reserved writer/index names. Existing
`save_stream` output naming is unchanged.

`AssetWriter.output_target(filename_prefix, mount=None)` combines separate save
inputs into the writer's target. Omitting `mount` uses the snapshot's configured
output mount; an explicit mount selects that grant instead. Both forms retain
the writer's prefix validation, live write permissions and containment checks.
Prefixes expand `%width%` and `%height%` from the optional keyword arguments
`image_width` and `image_height` (both default to zero). SaveText uses one for
both dimensions. `%year%`, `%month%`, `%day%`, `%hour%`, `%minute%` and `%second%`
use local time, with two digits except for the year. Unknown tokens stay literal.
Expansion happens before the save target is parsed, including directory names.

## Catalog schema

`LibraryStore` upgrades its SQLite catalog to the Alembic head before serving
records. Record CRUD remains stdlib SQLite. The initial revision creates
`records`, `media_grants`, `asset_dependencies`, `asset_derivations`, and the
`records_scope_modified` index. An existing unversioned catalog is adopted
only when its schema matches; an incompatible schema fails without changing
its records or recording a version. Back up the catalog before upgrading.

`LibraryStore(path, read_only=True)` opens an existing catalog without creating
directories, migrating schemas or changing journal mode. Scoped lookups retain
the same membership rules, and SQLite refuses writes through this connection.
SQLite may create or update its WAL/shared-memory coordination files; these are
not catalog content. Read-only open failures report the underlying SQLite reason
without retrying with write access or bypassing WAL recovery.

Asset observations share this database but are separate from mutable workflow
records. `LibraryStore.record_asset` records an original basename, observed
AssetRef, kind, project, time and origin event in one transaction. Observed
virtual locations are not filesystem authority. Download origin URLs omit
userinfo, query strings and fragments. `query_assets` pages digest identities
within one scope/project and retains all their aliases and origin events even
when only one event matches the name, kind, origin or date filter. Dates are
observation times, not claims about a file's original creation time.
Publication callers can supply an observation ID and time for idempotent
recovery; a repeated ID must describe the identical event.
Startup and catalog recovery backfill verified vault bytes already named by
scoped records or media grants. Existing project observations are not widened
to the default project. Historical media without a recorded name uses its digest
as a label and Unknown origin, not an invented filename or download URL.
Workflow records stay document-kind observations; neither backfill nor an
anonymous blob on disk creates new media or filesystem authority.
`unrecorded_vault_assets` inventories verified vault digests with no records,
media grants or observations in any scope. It returns digest labels and byte
sizes without fabricating names, kinds or dates, and writes no catalog state.
The server exposes it only to human, nondelegated storage operators through
`GET /api/assets/catalog?source=unrecorded`. The response has
`source: "unrecorded"`, ordinary five-field asset refs and empty observations.
Digest text search and descending digest cursor pagination are supported;
project, kind, origin and date filtering are not applicable. This is a local
operator inventory, not a tenant asset source or a federated catalog.

Named imports can prepare metadata with `before_publish` before the writer's
atomic link. The server journals upload observations in the private vault's
`.upload-observations` directory before publishing bytes, removing the journal
only after grant/catalog commits. Startup and catalog queries retry unfinished
observations. Mounted recovery verifies the exact filename/digest and current
writable/project grant, repairs digest indexing, and never creates missing bytes.
Unavailable, changed or revoked locations retain their journal for a later retry.
The catalog cannot turn that journal into filesystem authority.

Consented job acquisition uses the same journal for the verified winning
source before vault publication. Download observations retain only a sanitized
URL; packaged acquisitions record unknown origin rather than inventing a URL.
The request's scope and project own the observation, not the global vault.
Metadata preparation failures abort publication with `AssetPublicationError`
and do not try another mirror. A catalog failure after publication retains the
journal for verified, idempotent recovery at startup, catalog access or job
resubmission. Observation display names are not filesystem components and do
not inherit the mutable workflow record's name-length limit.

From the repository root:

```sh
uv run alembic -c packages/dinkster-assets/alembic.ini -x catalog=/path/to/library.sqlite current
uv run alembic -c packages/dinkster-assets/alembic.ini -x catalog=/path/to/library.sqlite check
```

For a schema change, update `schema.py`, generate a revision with
`alembic revision --autogenerate -m "Describe the schema change"` using the
same configuration and catalog arguments, and review the generated operations.
Migration resources are included in the package wheel.

## Resolver indexes

A resolver index is a data-only JSON document that supplies ordered download
leads for known model digests:

```json
{
  "dinksterResolver": 1,
  "name": "community-models",
  "entries": [
    {
      "digest": "blake3:<64 lowercase hex>",
      "name": "model.safetensors",
      "urls": ["https://models.example/model.safetensors"],
      "kind": "model/diffusion",
      "regions": {"cn": ["https://cn.example/model.safetensors"]}
    }
  ]
}
```

Each entry requires `digest`, `name`, and a nonempty HTTPS `urls` list. Optional
fields are `kind`, `size`, `license`, `gated`, `notes`, `regions`, `family`,
`variant`, and `components`. Every entry URL list, including each regional list,
accepts at most 64 URLs. Version 1 parsers ignore unknown optional top-level,
entry, and component fields. The parser rejects unknown versions, duplicate
digests, documents over 16 MiB, and documents over 100,000 entries.

`ResolverSubscriptionStore` persists local-file and hosted-URL subscriptions.
Hosted sources require HTTPS; loopback HTTP is accepted for local development.
Each subscription owns a replaceable `ProvenanceStore` layer, so refresh and
unsubscribe cannot remove manually registered leads or another index's leads.
Hosted indexes refresh no more than once every six hours and retain their last
valid document after an error. Cursor pages are assembled atomically within one
deadline and aggregate byte/entry limits. Only complete single-page documents
use ETag revalidation; multi-page indexes refetch every page because a page's
ETag cannot attest the whole catalog. Acquisition accepts bytes only after
they match the requested digest.

Fetch deadlines also bound the caller during DNS and TLS setup. Python cannot
cancel a blocked OS DNS call, so at most eight daemon fetches retain capacity
until they return; further fetches fail promptly while all slots are occupied.
Late results never publish provenance.

`bootstrap_official` accepts an explicit export URL and stable provider ID
(the export's exact `name`), with no default values. Its persisted identity
binding prevents refresh identity changes and resubscription after an explicit
removal. Missing or changed configuration refuses bootstrap without replacing
the saved choice. The serve CLI/environment configuration is described in
[serve-cli.md](../../docs/serve-cli.md#--official-resolver-url--official-resolver-provider-id).

## Public acquisition receipts

An acquisition receipt is written only after a credential-free HTTPS transfer
matches both the declared byte size and BLAKE3 digest. Every redirect target's
DNS answers and the connected peer must remain globally routable. Userinfo,
custom headers, cookies, URL query strings, HTTPS downgrades, mixed private DNS
answers, and rebinding deny the receipt. The ordinary verified download path
can still make the file available locally without a receipt.

## Learn more

See DESIGN 3.12 for asset identity, catalogs, mounts, and distribution, and
DESIGN 3.2 for the asset value envelope. The path and node-authoring
invariants are in `docs/hazards.md`, especially H4 and H9. Focused coverage
is in `tests/test_assets.py`, `tests/test_asset_acquisition.py`,
`tests/test_asset_guess.py`, `tests/test_asset_writer.py`,
`tests/test_latent_asset_format.py`, `tests/test_mounts.py`, and
`tests/test_declared_assets.py`. Resolver-index coverage is in
`tests/test_resolver_indexes.py`; public receipt coverage is in
`tests/test_public_acquisition_receipts.py`.
