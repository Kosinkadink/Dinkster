# dinkster-registry

`dinkster-registry` is Dinkster's thin client-side registry and install model.
It builds and verifies BLAKE3-addressed pack archives, records lockfiles and
installation plans, and supports the umbrella package's client for the
registry service. It contains no server, database, or registry command.

`dinkster_registry.archive_admission` provides deterministic archive ingest,
safe extraction, manifest admission, static doctor, a bounded isolated executable
probe and fast security checks. Stage results distinguish blocking findings,
required review and informational signals; review is not acceptance. Ingest
verifies length, BLAKE3, portable bounded paths and canonical ZIP serialization
through the existing artifact builder. Manifest admission validates canonical ids,
normalized public versions, license, repository, contracts and reserved claims.
An optional `engine_version` gates the shared pack compatibility declaration;
returned metadata retains that declaration. These checks do not consult or mutate
registry ownership, namespace grants, catalog versions, dependency availability
or publication state; those observations remain the service caller's responsibility.

`ProbeSettings` bounds time, output and operator tails. The probe provisions
declared CPU dependencies as wheels with a stripped environment, refuses direct
URLs, and requires the full Linux jail by default. Missing or degraded isolation
blocks rather than falling back. `require_isolation=False` is for controlled tests
only. Public findings do not expose arbitrary probe diagnostics; bounded tails
remain separate operator data.

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

## Official pack publication

Official pack manifests declare their own public `pack.version` and PEP 440
`pack.engine` range. These use the shared schema and archive admission contract;
the engine version is not the pack version.

`python -m scripts.build_official_packs --pack PACK_ID --output /scratch/packs`
builds selected committed sources using their manifest versions. Repeat `--pack`
or omit it for the official set. Unknown ids and missing declarations refuse.
The explicit `--version` override remains available for existing engine-release
callers. Native variant, interpreter and constraints inputs produce dependency
locks for the same selected sources without changing an installed generation.

`python -m scripts.select_official_packs --before FULL_COMMIT` selects official
ids changed between that committed push base and HEAD. Engine-only changes
select nothing; a zero base selects all packs. Shared vision files select all
vision packs, while pack-specific sources select their owner. Working-tree
manifests are ignored. `--tag pack/PACK_ID/vVERSION` selects one pack only when
VERSION exactly matches its declared version.

`scripts.build_release_pack_locks --sources /scratch/packs` uses that source
catalog and each pack's own version. `--release` supplies verified published
engine inputs and constraints, not pack versions. Regenerated sources must
match the catalog and archive bytes. Without `--sources`, existing engine
release behavior is unchanged. Locks use native supported-base interpreters
and portable wheel URLs.

`python -m scripts.publish_official_packs --sources /scratch/packs
--base-manifest PATH --pack-lock PATH --output /scratch/publication` validates
and prepares publication maps. Repeat the base and lock options for all bases
in `scripts/base_bundles.json` and every selected pack/base pair. It checks
canonical base identities, archive digests, manifest declarations, lock
identities, Python compatibility and credential-free HTTP(S) origins before
exposing any maps. It neither resolves dependencies nor installs a pack.

Add `--publish --registry URL` to submit unchanged archives and their complete
maps using `DINKSTER_REGISTRY_TOKEN`. The HTTP client's `publish_release` accepts
an optional `locks` path and sends exact archive and map bytes as multipart;
omitting it preserves raw-ZIP submission. Local validation completes before the
first request, but separate release submissions are not a registry transaction.
A queued receipt is not an admission verdict.

`.github/workflows/official-packs.yml` publishes after the same commit's
successful `Versioned wheel release` run on trusted `Kosinkadink/Dinkster`
main. It fetches `main-COMMIT` and verifies the engine manifest's repository,
tag and commit, never selecting by publication time. Failed engine releases
publish nothing. Pack changes select their owners; engine dependency or
publication-workflow changes select all current packs. Unrelated engine-only
changes select none. All six supported bases resolve locks on native hosts.

PRs without engine dependency changes preflight against the exact base-commit
engine release. A PR changing engine dependency inputs instead validates its
own source archives and explicitly reports that published-engine lock preflight
is not applicable until its candidate engine is released. Exact pack tags also
validate sources only. PR, fork and tag refs never submit or receive the
submission credential. A missing credential refuses submission. Registry
versions and lock bindings are immutable, so changed bytes need new manifest
versions. Publication does not imply that scanner admission has completed.

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
