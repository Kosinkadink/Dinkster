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
`dinkster-registry-service` package and `dinkster-registry` command.
Dinkster remains a client of that service for browsing, publishing, resolving,
and downloading exact pack releases.

## Engine mirror and projects

An engine feed contains a platform base and a hash-locked code layer. The base
contains a relocatable Python interpreter, torch, torchvision, their runtime
dependencies and the offline installation tool. Kitchen, aimdo, Dinkster and
the frontend bundle belong to the code layer. Updating those wheels does not
require rebuilding a base whose Python and torch pins are unchanged.

With the bootstrap `dinkster` command installed, select a mirror explicitly:

```sh
dinkster project create studio --data-root /path/to/studio-data
dinkster --project studio install --mirror https://mirror.example/dinkster --channel github-live --cell linux-cu128
dinkster --project studio serve --port 3639
```

`mirror.example` is a placeholder, not a production download service. A local
S3-compatible development feed can use `--allow-local-http` with a loopback
HTTP URL. Other mirrors require HTTPS. `DINKSTER_ENGINE_MIRROR` sets the default
mirror. The first native cells are `linux-cu128`, `win-cu128` and `mac-arm64`;
the selected channel must actually contain the requested cell. A cell names
an environment variant, not a hardware admission requirement.

Both `stable` and `github-live` channels use the same layout:

```text
channels/stable.json
channels/github-live.json
engine/<commit>/<cell>.json
base/<cell>/<base-id>.tar.gz
store/<wheel-sha256>
```

The channel pins each manifest's digest and size. Each manifest pins its base
archive and every wheel. Downloads resume from partial files and must pass
size and SHA-256 verification before installation. The installer reads only
the configured mirror; environment creation and wheel installation run
offline with no package index. A missing wheel is an error, not permission
to resolve it from PyPI or GitHub. Failed acquisition or installation leaves
the active generation unchanged.

`dinkster project list` lists named roots. Each project stores its own
`generations/` and `current` under `<DINKSTER_HOME>/projects/<name>` and shares
downloaded objects in `<DINKSTER_HOME>/engine-store`. Each generation selects
its base, code manifest, pack lockfile and hosting policy. Every project runs
the supervisor installed in its own active generation; choose distinct ports
when serving several projects concurrently.

For a launcher that owns stop/start, use `install --stage-only --json`, stop
the project's supervisor, then `activate --generation N --json` and start it
again. Activation refuses a staged generation if another update changed its
predecessor. `generations --json` includes the control and execution Python
paths. `rollback` restores the previously activated environment, not an update
that was staged but never activated. Stop and restart the supervisor around
rollback as well; a running process never changes interpreters in place.

`gc` previews unreferenced engine objects, bases and environments;
`gc --apply` deletes that preview after recomputing it under the store lock.
All recorded generations retain their referenced content, including content
used by another registered project. Pack garbage collection remains available
as `dinkster pack gc`.

Models, outputs, history, settings and the library belong in a separate data
root. Installation, activation, rollback and engine GC never enumerate or
delete that data root. Project creation records its path without changing its
contents. Existing model folders can remain elsewhere and be mounted into the
project library.

### Packaged bootstrap runtime

Desktop installers bundle a Dinkster-built control runtime rather than an
engine ZIP. Build it natively from the pinned Dinkster commit:

```sh
python -m scripts.build_engine_feed --feed /path/to/output --cells linux-cu128 --control-runtime
```

Use `win-cu128` on Windows and `mac-arm64` on macOS. The cell selects the native
OS, architecture and Python build only; the resulting runtime contains no
accelerator, model, frontend or installed-engine payload. The output contract
is:

```text
control/<commit>/<os>-<architecture>.json
control/<os>-<architecture>/<archive-sha256>.tar.gz
```

The descriptor format is `dinkster.control-runtime/1`. Its complete fields are
`format`, `commit`, `platform`, `artifact`, `python` and `invocation`.
`artifact` contains `path`, `sha256` and `size`; the path is relative to the
feed output and the digest and size cover the compressed `.tar.gz` bytes.
`python` is a normalized descriptor-relative path (`bin/python3` on Linux and
macOS, `python.exe` on Windows). `invocation` is exactly
`["<python>", "-I", "-m", "dinkster.cli"]`, where `<python>` means the
extracted runtime root joined with `python`.

The archive is a deterministic GNU tar compressed with gzip. Extract only
after verifying its compressed size and SHA-256. Extraction must reject
absolute paths, `..` traversal, special files, hard links, and symlinks whose
relative target escapes the extraction root. Extract into a fresh staging
directory, verify that the descriptor's interpreter resolves to a file inside
that directory, then move the complete runtime into its immutable packaged
location.

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

For native generation, build the execution environments from the Dinkster
repository root with `./scripts/setup_envs.sh`, or
`.\scripts\setup_envs.ps1` on Windows. The scripts require `uv` and create
`.venv-torch` for CPU execution and for MPS execution on macOS Apple Silicon.
On Linux and Windows with a detected NVIDIA GPU, they also create `.venv-gpu`
for CUDA execution. A maintainer evidence checkout is not required. When
`DINKSTER_EVIDENCE_ROOT` selects one, the scripts also install its optional
`dinkster-acceptance` package; otherwise they print a skip notice and complete
normally. Set `DINKSTER_EXECUTION_PYTHON` to the selected environment's Python
when launching:

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
