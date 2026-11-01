# dinkster-registry

`dinkster-registry` is Dinkster's thin client-side registry and install model.
It builds and verifies BLAKE3-addressed pack archives, records lockfiles and
installation plans, and supports the umbrella package's client for the
registry service. It contains no server, database, or registry command.

## Setup

This package is a uv workspace member and is not published separately yet.
From the repository root:

```sh
uv sync --all-packages
```

It has no console script. The `dinkster-registry` service distribution and
command live in the registry repository.

## Use

Build a deterministic artifact for submission through the client:

```python
from pathlib import Path

from dinkster_registry import build_artifact

digest = build_artifact(Path("my-pack"), Path("my-pack.zip"))
assert digest.startswith("blake3:")
```

The umbrella `dinkster-pack` command publishes, browses, resolves, and downloads
through the service's `/v1` HTTP contract. Artifact helpers build, verify, and
unpack deterministic archives; install models provide lockfiles, plans, and
generation-based activation.

`dinkster-pack publish --version VERSION --locks /scratch/locks.json PACK_DIR`
submits the same deterministic archive with a publisher-produced per-base map.
Each variant maps to `baseId` (exact SHA-256 base identity), `artifactDigest`
(the archive's BLAKE3 digest), and `lock` (the standard pack-lock document).
The client sends those bytes without resolving dependencies; service admission
validates the bindings before acceptance. Omitting `--locks` retains raw-ZIP
submission. Publication errors and unreadable lock files refuse without a queued
receipt; a queued receipt is not an admission verdict.

The host installer's `Installer.apply` checks each declared pack engine range
against its running engine before staging or returning an identical-lock no-op.
The check reads supplied, cached or staged sources without importing pack code;
grouped packs follow the same rule. An incompatible range reports the pack id,
manifest version, range and engine version without changing the active selection
or staged generation content. Manifests without a declaration retain their
legacy unpublished behavior. Read-only manifest inspection still supports plans
for incompatible packs.

Build selected official source archives from committed manifests:

```sh
python -m scripts.build_official_packs --pack dinkster-nodes-foundation --output /scratch/packs
```

Repeat `--pack` for multiple ids or omit it to build the official set. Each
archive and source-lock entry uses that manifest's own `pack.version`, retaining
its engine compatibility declaration. Unknown ids and missing declarations
refuse. Existing engine-release callers can supply the explicit `--version`
override; it is not needed for independently versioned publication. Native
`--variant`, interpreter and constraints inputs produce dependency locks for
the same selected sources. Building does not publish an archive or alter a
generation.

`python -m scripts.select_official_packs --before FULL_COMMIT` selects official
ids changed between that committed push base and the checked-out commit.
An engine-only range returns an empty list. A zero base (the first push) selects
all official packs. Nested vision source changes select their owning pack;
shared vision files select all vision packs. Selection reads Git objects rather
than mutable working-tree manifests. `--tag pack/PACK_ID/vVERSION` selects one
official id only when VERSION exactly equals its declared public version.

The native `scripts.build_release_pack_locks` command accepts `--sources
/scratch/packs` to lock just that independent source catalog using each pack's
own version. `--release` still supplies verified engine artifacts and effective
constraints, not pack identities or versions in this mode. Regenerated source
archives and catalog must match the supplied bytes. Without `--sources`, the
legacy engine-release caller is unchanged. Both modes use the same native base
extraction, resolver and portable wheel-origin mapping.

`python -m scripts.publish_official_packs --sources /scratch/packs
--base-manifest /scratch/linux-cpu.json --base-manifest ...
--pack-lock /scratch/PACK-VERSION-VARIANT.pack-lock.json --pack-lock ...
--output /scratch/publication` prepares publication maps. Repeat the manifest
and lock options for every base in `scripts/base_bundles.json` and every selected
pack/base pair. The command verifies canonical base identities, archive digests,
manifest publication declarations, lock identities, Python compatibility and
credential-free HTTP(S) wheel/index origins before exposing any map. It does not
produce native locks or resolve dependencies. Invalid input leaves no output maps
and makes no publication request.

Add `--publish --registry URL` to submit each unchanged archive and its complete
map through the existing registry client, using `DINKSTER_REGISTRY_TOKEN` when
set. All local validation completes before the first request; separate release
submissions are not a registry transaction. A queued receipt is not scanner
acceptance or proof that the default set is served.

`dinkster.registries.resolve_compatible_release(registry, pack, minimum,
engine_version)` follows the service's version-list pages, chooses the newest
active PEP 440 version at or above the minimum whose declared engine range
admits the target, and returns the existing `LockedPack` model. Active pack
prereleases remain eligible inside these bounds. Historical releases with an
empty engine declaration are ineligible, not a barrier to selecting other
declared versions. Invalid nonempty ranges and malformed versions still refuse.
The default declaration is not a byte pin; the returned installation identity
is digest-pinned so archive acquisition can verify it. Resolution asks exactly
one selected registry and rechecks the chosen release's state and range against
its versioned detail.

The committed `src/dinkster/default-packs.toml` declares default pack ids and
minimum versions only, never exact release versions or archive bytes.
Run source preparation with the target engine's interpreter:

```sh
python -m dinkster.manager --registry https://registry.dinkster.org prepare-registry-sources --catalog-lock /scratch/dinkster.lock --artifacts /scratch/artifacts
```

The JSON result names the engine version, source lock, artifact cache, selected
packs and unavailable-pack diagnostics. Each selected archive is size/digest
verified and safely parsed without importing pack code. Existing cache entries
are checked again before reuse. Repeat `--minimum PACK@VERSION` to select an
explicit set instead of defaults. By default any unavailable pack refuses
before writing the output lock; `--allow-missing` reports each unavailable pack
and writes the available set, including an empty lock when none is available.
This command prepares source inputs only; dependency locks and generation
activation are separate steps. It never changes project or generation state.

For an engine update, pass `--installed-lock /retained/dinkster.lock` instead
of `--minimum`. Only installed registry packs attributed to publisher `dinkster`
are re-resolved, at or above their installed versions. Each uses its recorded
registry, not a newly chosen default; bare registry sources bind to the
selected/default endpoint. Third-party, local and git entries remain unchanged
for subsequent generation admission. This content attribution is not a new
authenticated authority claim. The candidate output must differ from the input.
An empty installed set stays empty. Missing compatible releases, registry errors or
invalid archives refuse before writing the candidate lock; unavailable installed
packs cannot use `--allow-missing`. No new default pack is added during update.

## Template catalog

`ReleaseIndex.template_catalog()` is the version 1 payload served by
`GET /index/templates`. It lists templates from each pack's latest accepted
immutable version in `(pack, template id)` order. Every row contains `pack`,
`version`, `id`, `name`, and the workflow `digest`; optional `description`,
`tags`, `family`, `models`, `assets`, and `thumbnail` metadata are copied from
the admitted release record. Template bodies and thumbnails are fetched from
the immutable versioned routes:

```text
/index/packs/{pack}/versions/{version}/templates/{id}
/index/packs/{pack}/versions/{version}/templates/{id}/thumbnail
```

Services call `ReleaseTemplate.verify_body()` before returning workflow bytes.
Clients reject unknown `catalogVersion` values, may cache the descriptor list,
and treat versioned bodies and thumbnails as immutable. Listing metadata never
contains internal artifact member paths.

## Learn more

The same service supports PostgreSQL-hosted and SQLite self-hosted deployments.
Focused client coverage is in `tests/test_artifact.py`,
`tests/test_install.py`, and `tests/test_installer.py`.
