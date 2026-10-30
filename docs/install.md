# Install Dinkster

Dinkster is pre-release. Release assets are private and are not published to a
package index. A GitHub 404 can mean that your account does not have access.

To run from source without the Desktop app, follow the
[one-command browser quickstart](quickstart.md). Install Git and Node.js
26.12 or newer within 26.x with npm, clone Dinkster, and run `./run.sh` on Linux or
`.\run.ps1` on Windows. The scripts install uv user-locally if missing;
the selected Dinkster-Environments bundle supplies runtime Python. The scripts
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

The registry server is a separate deployment, not part of the Dinkster wheel
set. Hosted PostgreSQL and self-hosted SQLite deployments use the same
`dinkster-registry-service` package and `dinkster-registry` command.
Dinkster remains a client of that service for browsing, publishing, resolving,
and downloading exact pack releases.

## Develop from source

The run scripts above are the default source installation path. For a manual
development setup, clone the backend and frontend as sibling directories, then run:

```sh
uv run --no-project python scripts/setup_envs.py
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

The shared Python setup installs only one runtime in `.venv`: the published
Linux CPU base, or a Windows NVIDIA/CPU base compatible with the detected
hardware. Exact Python and torch versions come from the verified base manifest,
not a local version override. `scripts/base_bundles.json` selects published
content identities from Dinkster-Environments. Hash-locked workspace dependencies
and editable packages are installed on top without resolving torch again.
Use `--variant linux-cpu` or `--variant win-cpu` to force CPU. CPU installation
does not install CUDA wheels. Rerun setup after checkout or dependency changes;
`--force` rebuilds its `.venv`, retaining the verified bundle store.

macOS setup stops with an unsupported-platform message until the
[Apple Silicon bundle](https://github.com/Kosinkadink/comfy-vibe-station/issues/575)
is available; it never falls back to a separate dependency resolution.

A maintainer evidence checkout is optional. If `DINKSTER_EVIDENCE_ROOT` selects
one, setup also installs its `dinkster-acceptance` package. The run scripts set
`DINKSTER_EXECUTION_PYTHON` automatically; for manual launch use:

| Platform | Execution interpreter |
| --- | --- |
| Linux CPU | `$PWD/.venv/bin/python` |
| Windows NVIDIA or CPU | `$PWD\.venv\Scripts\python.exe` |

The selected execution environment installs Dinkster's pinned `dinkster_inference`
runtime when the native or compatibility worker is provisioned.

## Model folders

An environment-managed generation initializes a selected writable `output`
mount under its library root on first serve, so the first run can save files.
With the default library location this is `<project-root>/dinkster-library/output`;
`DINKSTER_LIBRARY_ROOT` or `--library-root` can select another location.
The grant persists in `mounts.toml` and is reported by `GET /api/mounts` as
`outputMount: "output"` with a `readwrite` mount. Existing output selections,
read-only grants, and deliberately removed defaults are preserved. Initialization
alone does not enable runtime mount editing. Managed local `dinkster-env`
launches also enable operator folder editing; direct server launches remain
opt-in. A deliberately unconfigured library can still report `outputMount: null`.

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
