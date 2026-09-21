# Install Dinkster

Dinkster is pre-release. Release assets are private and are not published to a
package index. A GitHub 404 can mean that your account does not have access.

The Desktop app is the supported end-user installation path. It installs an
exact backend release from its wheel set and constraints file. Desktop builds
are not yet available; see the
[Desktop guide](https://github.com/Kosinkadink/Dinkster-Frontend/blob/main/docs/desktop.md)
for the current platform boundaries.

## Inspect a wheel release

Each `vX.Y.Z` release contains:

- one wheel for every Dinkster workspace package;
- the `dinkster` meta-package wheel with its `default` extra;
- a `dinkster-frontend` wheel containing the built browser application;
- the pinned private `dinkster-identity` dependency wheel;
- `constraints.txt`, exported from `uv.lock` with hashes;
- `release-manifest.json` and `SHA256SUMS`; and
- `dinkster-source-X.Y.Z.zip`, a maintainer source archive.

Download all wheels and `constraints.txt` from the same release. A manual
installation for inspection can be created without Git credentials:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --no-deps --require-hashes --find-links . --requirement constraints.txt
.venv/bin/python -m dinkster.cli --help
```

On Windows, use `.venv\Scripts\python.exe`. Do not combine wheels or constraints
from different releases. Models and accelerator runtimes are not bundled.

The registry server is a separate deployment, not part of the Dinkster wheel
set. Hosted PostgreSQL and self-hosted SQLite deployments use the same
`dinkster-registry-service` package and `dinkster-registry` command from the
[registry repository](https://github.com/Kosinkadink/dinkster-registry).
Dinkster remains a client of that service for browsing, publishing, resolving,
and downloading exact pack releases.

## Develop from source

Clone the backend and frontend as sibling directories, then run:

```sh
uv sync --python 3.12 --all-packages --frozen
cd ../Dinkster-Frontend
pnpm install --frozen-lockfile
pnpm --filter @dinkster/app build
cd ../Dinkster
uv run --no-sync dinkster setup
uv run --no-sync dinkster
```

The browser opens at `http://127.0.0.1:3639`. Do not expose the engine directly
to the Internet; read [authentication](auth.md) before configuring shared
access. For native generation, follow the
[runtime setup instructions](../packages/dinkster-inference-torch/README.md).

## Release maintainers

The source ZIP is for maintainers and excludes tests, tools, and benchmarks. It
is not the Desktop installation input. To publish the first release after this
workflow lands, verify the version in every workspace `pyproject.toml`, then run:

```sh
git tag -a v0.0.1 -m "Dinkster 0.0.1"
git push origin v0.0.1
```

The tag workflow verifies tag and package versions, builds and checks all
wheels, installs the exact artifacts on Linux x64, Windows x64, and macOS
arm64, and only then creates the GitHub Release with generated notes. Tagging,
release creation, and package publication are maintainer-only actions.
