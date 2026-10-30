# Install Dinkster

Dinkster is pre-release. Release assets are private and are not published to a
package index. A GitHub 404 can mean that your account does not have access.

To run from source without the Desktop app, follow the
[one-command browser quickstart](quickstart.md). Install Git and Node.js
26.12 or newer within 26.x with npm, clone Dinkster, and run `./run.sh` on Linux/macOS or
`.\run.ps1` on Windows. The scripts install uv user-locally if missing;
Linux/macOS obtain Python 3.12 through uv. Windows uses the verified private
Python 3.14.10 runtime described below. The scripts
prepare execution environments, build the pinned frontend, and launch the UI
and backend on one loopback origin. Rerun after `git pull` to update them.

The Desktop app installs an
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
On Windows, first prepare the runtime below and pass its executable to
`uv venv --python` instead of the floating `3.12` selector.

The registry server is a separate deployment, not part of the Dinkster wheel
set. Hosted PostgreSQL and self-hosted SQLite deployments use the same
`dinkster-registry-service` package and `dinkster-registry` command.
Dinkster remains a client of that service for browsing, publishing, resolving,
and downloading exact pack releases.

## Develop from source

The run scripts above are the default source installation path. For a manual
development setup on Linux/macOS, clone the backend and frontend as sibling directories, then run:

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

### Windows interpreter

Windows uses CPython 3.14.10 / cp314 from python-build-standalone 20261026,
with SQLite 3.53.1 plus the upstream
[mapped-shm recovery fix](https://sqlite.org/src/info/7c103a0b7d).
The unmodified interpreter fails WAL reopening while an old shared-memory
mapping survives ([upstream report](https://sqlite.org/forum/forumpost/f9c68d01e2d7f638)).
A SQLite version string cannot distinguish the patched library, so acquisition
and bundle extraction require a functional WAL read and integrity check with
the mapping held open, followed by a released-mapping control.

`scripts/setup_envs.ps1` prepares this runtime in `.run/windows-python` and
uses it for the root, CPU and optional GPU environments. Existing CPU/GPU
environments must also pass the check; use `-Force` to recreate incompatible
environments. For manual root-only setup from the checkout:

```powershell
uv run --no-project --python 3.14.10 scripts/prepare_windows_python.py --output .run/windows-python
uv sync --python .run/windows-python/python/python.exe --all-packages --frozen
```

Preparation verifies immutable Dinkster-Environments source pins, the original
interpreter archive, and the replacement DLL. Zig 0.14.1 is a new build
dependency, installed in an isolated uv environment and verified against the
exact Windows wheel SHA-256. Only `python/DLLs/sqlite3.dll` changes; uv's
interpreter cache is never modified. The base manifest records the official
amalgamation, patch, compiler, flags and DLL hashes, and Windows bundle CI
must reproduce the pinned DLL hash. Linux and macOS are unchanged.
When a verified upstream interpreter contains the fix, remove the override
and move the pins together; retain the functional check.

Interpreter inventory:

| Consumer | Interpreter selection | Bundled SQLite |
| --- | --- | --- |
| Environments Windows bases | PBS 20261026, CPython 3.14.10 / cp314 | 3.53.1, same-version DLL with upstream fix |
| Environments Linux base | PBS 20261026, CPython 3.14.10 / cp314, unchanged | 3.53.1, upstream unmodified |
| Windows full-validation shards, PR runtime job, release install, dependency receipt job | Verified private Windows base interpreter | Patched 3.53.1 |
| Windows developer root/CPU/GPU environments | `setup_envs.ps1`, verified private base | Patched 3.53.1 |
| Windows launcher and acquisition bootstrap | uv CPython 3.14.10, no database use before preparation | Unpatched bootstrap may carry 3.53.1; not used as the engine runtime |
| Linux CI/full-validation/developer environments and macOS release smoke | Floating uv `3.12`, unchanged | Not a fixed artifact; PBS 20261026 CPython 3.12.14 carries 3.53.1 |
| Linux release runtime | CPython 3.14.10, unchanged | PBS 20261026 carries 3.53.1 |
| OpenTimelineIO-only dependency index/import checks, including Windows ARM64 | Floating uv `3.14`, unchanged, no Dinkster stores | Not a fixed artifact; PBS 20261026 carries 3.53.1 |
| Legacy Desktop source installer | Floating uv `3.12` / cp312 | Not fixed; PBS CPython 3.12.14 carries affected 3.53.1 |

The PBS versions above come from the
[20261026 dependency manifest](https://github.com/astral-sh/python-build-standalone/blob/20261026/pythonbuild/downloads.json);
Windows 3.14.10 was also executed from the size/hash-verified archive.
Floating selectors can choose other catalog releases or existing local Python
installations; they do not promise a SQLite version. The Windows functional
gate is therefore applied to the interpreter actually admitted for use.

The legacy Desktop installer is superseded by the
[Desktop-as-client change](https://github.com/Kosinkadink/Dinkster-Desktop/pull/23),
which uses `dinkster-env` base acquisition instead. Until that change lands,
a fresh Desktop install floating to a PBS 3.12.x with SQLite 3.50 or newer
is exposed to the regression. There is no cp314 migration of that legacy path.

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

The selected execution environment installs Dinkster's pinned `dinkster_inference`
runtime when the native or compatibility worker is provisioned.

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

The frontend pin in `scripts/release_sources.json` names a landed
`Dinkster-Frontend` commit and the SHA-256 of its published `bundle.zip`.
The release job downloads the `dinkster-frontend-<sha256>` CI artifact from
a successful frontend CI run and verifies its commit, hash and size. It
packages the verified files under `dinkster_frontend/static`, with the
original manifest alongside them; `bundle_path()` still returns `static`.
No Node or pnpm toolchain builds the frontend during release packaging.
The wheel version follows the engine release; the manifest retains the
independently pinned frontend package version and commit.

To bump the frontend, update the commit and sha256 together. Public CI
artifacts expire after 90 days. A missing or expired artifact fails the
release with its required name and commit, never a rebuild fallback. Bump
the pin to a landed commit with a live artifact, or ask the repository owner
to re-run frontend CI for the pinned commit before releasing.

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
