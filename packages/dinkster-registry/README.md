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

## Learn more

See the [registry repository](https://github.com/Kosinkadink/dinkster-registry)
for the PostgreSQL hosted and SQLite self-hosted service. Focused client
coverage is in `tests/test_artifact.py`, `tests/test_install.py`, and
`tests/test_installer.py`.
