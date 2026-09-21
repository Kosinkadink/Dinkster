# Install Dinkster

Dinkster is pre-release. Release assets are private and are not published to a
package index. A GitHub 404 can mean that your account does not have access.

The Desktop app is the supported end-user installation path. It installs an
exact backend release from its wheel set and constraints file. Desktop builds
are not yet available; see the
[Desktop guide](https://github.com/Kosinkadink/Dinkster-Frontend/blob/main/docs/desktop.md)
for the current platform boundaries.
Until that repository becomes public, the guide requires repository access;
authenticated users can make a local copy with
`gh repo clone Kosinkadink/Dinkster-Frontend`.

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
access.

For native generation, run `scripts/setup_envs.sh` on Linux or macOS, or
`scripts/setup_envs.ps1` on Windows. The scripts create `.venv-torch` for CPU
execution and for MPS execution on macOS Apple Silicon. On Linux and Windows
with a detected NVIDIA GPU, they also create `.venv-gpu` for CUDA execution.
Set `DINKSTER_EXECUTION_PYTHON` to the selected environment's Python when
launching:

| Platform | Execution interpreter |
| --- | --- |
| Linux CUDA | `$PWD/.venv-gpu/bin/python` |
| Windows CUDA | `$PWD\.venv-gpu\Scripts\python.exe` |
| Linux/macOS CPU or macOS Apple Silicon MPS | `$PWD/.venv-torch/bin/python` |
| Windows CPU | `$PWD\.venv-torch\Scripts\python.exe` |

The [torch package README](../packages/dinkster-inference-torch/README.md)
contains contributor test and validation details; it is not required for the
first-image setup.

## Model folders

`dinkster setup` creates `<DINKSTER_HOME>/library/mounts.toml`, which defaults
to `~/.dinkster/library/mounts.toml` on Linux and macOS and
`%USERPROFILE%\.dinkster\library\mounts.toml` on Windows. Download model files
into a folder you control and grant that folder read-only access with an
absolute path:

```toml
[mounts.models]
path = "/home/name/Models"
mode = "read"
priority = 0
```

For Windows, forward slashes avoid TOML backslash escaping:

```toml
[mounts.models]
path = "C:/Users/name/Models"
mode = "read"
priority = 0
```

Mount ids must start with a lowercase letter or digit and may contain lowercase
letters, digits, and hyphens. `mode` is `read` or `readwrite`; model folders
should use `read`. Lower `priority` values are considered first when more than
one mount can supply an asset. Keep the `[settings]` and `[mounts.output]`
sections that setup created, restart Dinkster after editing the file, and wait
for the folder scan. Mounted files then appear under **Browse** in asset
pickers such as `Load Checkpoint`.

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
