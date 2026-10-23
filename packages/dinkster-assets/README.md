# dinkster-assets

`dinkster-assets` models files by canonical content digest rather than host
path. It provides asset references, catalogs, mounts, libraries, verified
vaults and acquisition, save targets, and value-type registration; it
depends only on values and BLAKE3, while caches, workers, and the server
build on it.

## Setup

This package is a uv workspace member and is not published separately yet.
From the repository root, install the complete workspace:

```sh
uv sync --all-packages
```

It does not install a console script.

## Use

Identity is always the canonical `blake3:<64 hex>` digest:

```python
from dinkster_assets import AssetRef, digest_bytes

payload = b"example asset bytes"
ref = AssetRef(
    digest=digest_bytes(payload),
    name="example.bin",
    size=len(payload),
)
```

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
