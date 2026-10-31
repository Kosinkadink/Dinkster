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

`dinkster.registries.resolve_compatible_release(registry, pack, minimum,
engine_version)` follows the service's version-list pages, chooses the newest
active PEP 440 version at or above the minimum whose declared engine range
admits the target, and returns the existing `LockedPack` model. Active pack
prereleases remain eligible inside these bounds. The default declaration is
not a byte pin; the returned installation identity is digest-pinned so archive
acquisition can verify it. Resolution asks exactly one selected registry and
rechecks the chosen release's state and range against its versioned detail.

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
