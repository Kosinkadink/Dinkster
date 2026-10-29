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

## BitTorrent v2 descriptors

`derive_p2p_descriptor` makes a canonical single-file BitTorrent v2 descriptor
without starting a client or performing network I/O. It hashes BLAKE3 identity
and BEP 52 trees in one bounded sequential scan. The torrent name is always the
64-character digest portion of the authoritative `blake3:` identity, and the
piece length is always 8 MiB.

Use `P2PDescriptorBuilder.update(bytes)` when bytes arrive incrementally, then
call `finalize()` for the same `P2PDescriptorResult`. Chunk boundaries may fall
anywhere; the builder retains only partial-block state and the piece roots needed
for the result, rather than the whole asset. Empty input is rejected, and a
finalized builder cannot be reused.

`validate_p2p_descriptor` reconstructs the canonical info dictionary from the
asset digest, byte size, and file root. `verify_p2p_descriptor` additionally
checks local bytes and the piece layer. Both reject alternate profiles rather
than accepting equivalent noncanonical torrent metadata.

Generate and verify a portable fixture with:

```sh
python -m dinkster_assets.p2p_fixture model.safetensors > descriptor.json
python -m dinkster_assets.p2p_fixture model.safetensors --verify descriptor.json
```

The fixture contains no trackers, peers, URLs, DHT settings, or alternate asset
identity. Libtorrent 2.1.1 is a development-only conformance dependency.

## Resumable P2P staging

`AssetVault.open_p2p_partial` validates the canonical descriptor against the
asset digest and size, then creates or reopens a sparse transfer at
`<vault>/.p2p/staging/<infoHash>/<digest-hex>`. `write_piece` accepts bounded
out-of-order writes and durably records completed byte ranges, so a later
process can reopen the same descriptor and continue.

`AssetVault.adopt_staged_asset` requires the same validated descriptor and
publishes only after the staged file has the declared size, passes format
policy 1 (strict safetensors or GGUF v3), and matches its BLAKE3 digest. Staging
and publication use handle-relative, non-symlink operations on one filesystem;
publication flushes the verified file before atomically claiming its canonical
vault path.

Inactive partials are retained for seven days. `p2p_staging_usage` reports
both allocated and logical bytes for sparse-file quota decisions, and
`purge_inactive_p2p_partials` removes expired unlocked partials.
`verify_p2p_local_file` provides a no-copy mapping that becomes stale if the
verified external file is changed, replaced, or deleted.

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

Each entry requires `digest`, `name`, and HTTPS `urls`. The URL list may be
empty only with a valid canonical P2P descriptor and size. Optional
fields are `kind`, `size`, `license`, `gated`, `notes`, `regions`, `family`, `variant`,
`components`, and `p2p`. Every entry URL list, including each regional list,
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
ETag cannot attest the whole catalog. Legacy caches retain HTTP leads but need
a complete unconditional refresh before supplying P2P authority. Acquisition
still accepts bytes only after they match the requested digest.

Fetch deadlines also bound the caller during DNS and TLS setup. Python cannot
cancel a blocked OS DNS call, so at most eight daemon fetches retain capacity
until they return; further fetches fail promptly while all slots are occupied.
Late results never publish authority. Derived authority is validated before
replacing persisted subscriptions, so invalid metadata cannot poison a reload.

User-added resolver subscriptions are HTTP-only by default. Enabling their
`trustedForP2P` flag requires the host's P2P settings permission. Both flags
default false; `licenseAuthoritative` identifies metadata authority and is not
an eligibility condition. `bootstrap_official` accepts an explicit export URL
and stable provider ID (the export's exact `name`), with no default values.
It creates a complete official subscription with both flags true, or preserves
an existing subscription's flags. Its persisted identity binding prevents
refresh identity changes, automatic regranting, and resubscription after an
explicit removal. Missing or changed configuration refuses bootstrap without
replacing the saved choice. The serve CLI/environment configuration is described
in [serve-cli.md](../../docs/serve-cli.md#--official-resolver-url--official-resolver-provider-id).
Valid `entries[].p2p` descriptors are semantically rebound to the
entry digest and size; a malformed descriptor disables P2P for that entry
without removing its HTTP leads. A complete successful refresh is the entire
P2P snapshot, so omitted descriptors become tombstones immediately. Failed or
partial refreshes retain HTTP last-good data without renewing P2P authority.
The resolver v1 adapter emits no trackers and uses trackerless DHT.

## Public acquisition and P2P grants

Resolver entries can publish a canonical `p2p` descriptor when `size` and
the entry digest match. This is only a declaration until subscription
trust and the P2P capability are active. The grant policy performs no
peer discovery, transfer, or seeding itself.

An acquisition receipt is written only after a credential-free HTTPS transfer
matches both the declared byte size and BLAKE3 digest. Every redirect target's
DNS answers and the connected peer must remain globally routable. Userinfo,
custom headers, cookies, URL query strings, HTTPS downgrades, mixed private DNS
answers, and rebinding deny the receipt. The ordinary verified download path
can still make the file available locally but grants no P2P authority.

`P2PGrantReconciler` derives revocable `PublicSwarmGrantV1` and `SeedGrantV1`
records from current trusted declarations. Declarative resolver seeding needs a
matching acquisition receipt. Trusted providers, enabled code resolvers,
and manual declarations need their equivalent complete-file evidence. Local
bytes are reverified against the descriptor before a seed grant is issued, so
a matching pre-existing file and advisory provenance alone grant nothing.
License and gated fields are metadata, not seeding or fetching conditions.
Grants expire no later than six hours after refresh and disappear when
their declaration, source revision, set membership, capability, or trust expires.

Inspect persisted receipts and the currently effective no-network policy with:

```sh
dinkster p2p-diagnostics --library-root /path/to/library --json
dinkster p2p-diagnostics --library-root /path/to/library --enabled --json
```

`--enabled` evaluates enabled grants without changing settings or starting
networking.

## Trusted global P2P policy

`dinkster_assets.p2p_global` strictly decodes provider artifact descriptors,
complete enumeration, license and location metadata, approved
trackers, and tombstones without opening a socket. It intersects those facts
with an explicit provider allowlist and adapts eligible rows into
`PublicSwarmDeclarationV1`; `P2PGrantReconciler` remains the single source of
public and seed grant authority. Unsafe formats, malformed enumeration,
expiration, and observed tombstones produce no declaration. Missing or custom
licenses and gated or absent HTTP locations do not suppress P2P authority.
Current matching public grants map to bounded `global-p2p` transport candidates
only while global downloads are allowed; transport ranking preserves LAN-first
selection and HTTP fallback.

Provider declarations and sidecar global leases expire within six hours.
The default `lan-and-internet` scope permits DHT, PEX, TCP, uTP,
approved trackers, UPnP, NAT-PMP, and PCP. Durable byte, ratio, and active
seed-time counters can close internet participation while LAN remains
available. Metered policy closes only global transport, preserving LAN mappings.

## Learn more

See DESIGN 3.12 for asset identity, catalogs, mounts, and distribution, and
DESIGN 3.2 for the asset value envelope. The path and node-authoring
invariants are in `docs/hazards.md`, especially H4 and H9. Focused coverage
is in `tests/test_assets.py`, `tests/test_asset_acquisition.py`,
`tests/test_asset_guess.py`, `tests/test_asset_writer.py`,
`tests/test_latent_asset_format.py`, `tests/test_mounts.py`, and
`tests/test_declared_assets.py`. Resolver-index coverage is in
`tests/test_resolver_indexes.py`; P2P descriptor and staging coverage is in
`tests/test_p2p_descriptor.py` and `tests/test_p2p_storage.py`; public receipt
and P2P grant coverage is in `tests/test_p2p_receipts_and_grants.py`; provider
global policy coverage is in `tests/test_p2p_global.py`.
