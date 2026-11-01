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
