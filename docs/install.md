# Install Dinkster

Dinkster is pre-release. Use the manifest and checksums from the same engine
release for every installation input.

To run from source without the Desktop app, follow the
[one-command browser quickstart](quickstart.md). Install Git and Node.js
26.12 or newer within 26.x with npm, clone Dinkster, and run `./run.sh` on Linux
or Apple Silicon macOS, or `.\run.ps1` on Windows. The scripts install uv user-locally if missing;
the selected Dinkster-Environments bundle supplies runtime Python. The scripts
prepare execution environments, build the pinned frontend, and launch the UI
and backend on one loopback origin. Rerun after `git pull` to update them.

The Desktop app installs an exact backend release through the environment
client. See the
[Desktop guide](https://github.com/Kosinkadink/Dinkster-Desktop#readme)
for platform artifacts and the unsigned Apple Silicon installation procedure.

## Inspect a wheel release

Each `vX.Y.Z` release contains:

- the `dinkster`, `dinkster-api` and `dinkster-supervisor` wheels, with the
  browser application embedded in the engine;
- `constraints.txt`, exported from `uv.lock` with hashes;
- `release-manifest.json` with exact per-variant dependency identities and `SHA256SUMS`;
- `dinkster-source-X.Y.Z.zip`, a maintainer source archive;
- native tools ZIPs with the installer interpreters; and
- the existing official pack artifacts and per-variant pack locks.

No third-party wheels are release assets. Finalization rejects unexpected
payloads and fails above 100,000,000 bytes excluding native tools ZIPs.
Confirmed `dinkster-env` setup/update fetches non-base dependencies from their
locked PyPI artifact URLs into the shared wheel store, checking the exact
filename, size and SHA256. Base-owned names and exact variant-index wheel pins
come from the selected Environments manifest. Generation construction uses
only the store, offline, without resolution or source builds.

`setup --package-index https://packages.example.org/simple` selects a simple
package mirror instead of those locked artifact origins. The index must offer
the exact locked filename; index hashes cannot change the required size/hash.
There is no fallback to PyPI or another version. Confirmed setup saves the
index for updates; `update --package-index URL` changes it transactionally.
This option is separate from `--mirror`, which selects channel, base and
Dinkster release artifact locations. Warm shared-store wheels need no index
access. Repair and move remain offline.

Download the three Dinkster wheels and `constraints.txt` from the same release.
A manual installation for inspection can be created without Git credentials;
the command fetches third-party wheels from the package index using the exported
hashes, rather than selecting a published base:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --no-deps --require-hashes --find-links . --requirement constraints.txt
.venv/bin/python -m dinkster.cli --help
```

On Windows, use `.venv\Scripts\python.exe`. Do not combine wheels or constraints
from different releases. Models and accelerator runtimes are not bundled.
For a variant-specific managed installation, use the
[native environment tools](environments-and-updates.md#environment-tools-bootstrap)
and `dinkster-env setup` instead of this generic inspection environment.
On Windows, first prepare the runtime below and pass its executable to
`uv venv --python` instead of the floating `3.12` selector.

The registry server is a separate deployment, not part of the Dinkster wheel
set. Hosted PostgreSQL and self-hosted SQLite deployments use the same
`dinkster-registry-service` package and `dinkster-registry` command.
Dinkster remains a client of that service for browsing, publishing, resolving,
and downloading exact pack releases.

### Prepare compatible registry pack sources

Run source preparation with the engine interpreter that will load the packs.
The committed `src/dinkster/default-packs.toml` specifies official ids and
minimum versions. It does not pin release bytes: the registry client chooses
the newest active version whose declared engine range admits that interpreter's
engine version, then verifies the selected archive by size and BLAKE3 digest.

```sh
python -m dinkster.manager --registry https://registry.dinkster.org prepare-registry-sources --catalog-lock /scratch/dinkster.lock --artifacts /scratch/artifacts --allow-missing
```

Use writable scratch paths that are separate from an installed generation.
On Windows, use the target environment's `Scripts\python.exe` and Windows
paths. The JSON result lists selected ids and an `unavailable` map with a
reason for each missing pack. With no registry access, `--allow-missing`
produces an empty source lock; without it, any unavailable pack refuses the
command. An engine serving an explicitly empty generation remains healthy
with zero nodes rather than loading checkout defaults.

This command prepares verified sources only. It does not install the engine,
obtain dependency locks, build a generation or activate it. For an update,
use `--installed-lock /retained/dinkster.lock` instead of fresh selection;
the output lock must have a different path. Installed official registry packs
are re-resolved without downgrading, adding defaults or changing their recorded
registry. Other installed selections are preserved for admission. Update
preparation refuses unavailable compatible releases and does not allow
`--allow-missing`.

For a source lock containing registry packs, `prepare-registry-locks` accepts
`--catalog-lock`, a new `--output` directory, `--variant` and the exact
`--base-id sha256:...`. It downloads the dependency locks already published
for those exact source archives and base, verifies their size, BLAKE3 and
pack/variant identities, and reports their paths as JSON. It never resolves
dependencies. Any missing or mismatched lock refuses the whole set without
creating the output directory or changing an installed generation.

Confirmed published `dinkster-env setup` and `update` run these commands with
the acquired target engine, in isolated temporary storage before opening an
activation journal. Fresh setup reports unavailable defaults in
`unavailablePacks` and on stderr, and can activate an empty generation when
the registry is offline. Updates re-resolve the installed official selection;
an unavailable compatible release refuses without changing project settings,
the current generation or its journal. Non-registry sources are retained and
checked against their recorded digests and the target engine; their dependency
locks remain generation-owned. Explicit catalog/artifact and dependency-lock
inputs override automatic preparation. `--from-scratch` remains a separate
checkout-only path.

The preparation commands can also be run separately. Source preparation's
`--retain-sources` requires `--installed-lock` and caches unchanged sources
from that generation's `pack-sources` tree. Lock preparation's `--registry-only`
leaves non-registry dependency locks to the retained generation.

Pack versions are independent of engine versions. A pack's `[pack] version`
and `engine` fields declare its version and PEP 440 engine range; excluded
engines refuse installation, including an identical-lock no-op. See
[pack compatibility and dependency locks](environments-and-updates.md#versions-and-engine-compatibility)
and the [registry client guide](../packages/dinkster-registry/README.md#use).

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
Linux NVIDIA/CPU base, a Windows NVIDIA/CPU base compatible with the detected
hardware, or `mac-arm64` on Apple Silicon with macOS 14 or newer.
Exact Python and torch versions come from the verified base manifest,
not a local version override. `scripts/base_bundles.json` selects published
content identities from Dinkster-Environments. Hash-locked workspace dependencies
and editable packages are installed on top without resolving torch again.
Use `--variant linux-cpu` or `--variant win-cpu` to force CPU. CPU installation
does not install CUDA wheels. Rerun setup after checkout or dependency changes;
`--force` rebuilds its `.venv`, retaining the verified bundle store.

`linux-nvidia` supplies the cu132 torch family and its complete Linux CUDA
closure. It requires Linux x86_64 with glibc 2.28+, NVIDIA driver 580+ and
compute capability 7.5+. Use `--variant linux-nvidia` to request it explicitly;
an incompatible machine is refused rather than resolving another CUDA runtime.

The `mac-arm64` bundle uses PyPI wheels with the MPS backend, not a separate
torch index or CUDA wheels. Intel macOS is unsupported and setup stops with
an explicit message; it never falls back to a separate dependency resolution.

### Apple Silicon macOS

Apple Silicon on macOS 14 or newer is supported through the `mac-arm64` base.
The native `dinkster-tools-macos-arm64.zip` contains its own Python and uv;
environment setup, doctor, engine start, update and rollback do not require
host Python or uv. Verify the ZIP against the same release's manifest and
`SHA256SUMS`, then extract it with executable permissions preserved. Invoke
`python/bin/python3 -I -m dinkster_supervisor.environment` from that tree,
prepending its absolute `python/bin` directory to that command's PATH so it
uses the bundled uv. Do not change the global PATH.
Setup selects `mac-arm64` automatically; select the `main` channel to use a
pre-release engine. Intel Macs and CUDA on macOS are not supported.

The arm64 Desktop ZIP contains an unsigned `.app`. Extract it and copy the app
to a directory you control. macOS may require right-click **Open** or removal
of the `com.apple.quarantine` attribute from that verified app before launch.
This is an unsigned-build procedure, not a signature or notarization check.
Apple signing and notarization are tracked separately in
[Apple signing](https://github.com/Kosinkadink/comfy-vibe-station/issues/501);
they are not prerequisites for the tools or engine environment.

A maintainer evidence checkout is optional. If `DINKSTER_EVIDENCE_ROOT` selects
one, setup also installs its `dinkster-acceptance` package. The run scripts set
`DINKSTER_EXECUTION_PYTHON` automatically; for manual launch use:

| Platform | Execution interpreter |
| --- | --- |
| Linux NVIDIA or CPU | `$PWD/.venv/bin/python` |
| Apple Silicon macOS (`mac-arm64`) | `$PWD/.venv/bin/python` |
| Windows NVIDIA or CPU | `$PWD\.venv\Scripts\python.exe` |

The selected execution environment installs Dinkster's pinned `dinkster_inference`
runtime when the native or compatibility worker is provisioned.

### Explicit from-scratch development setup

Published bundles remain the supported default. For development or a platform
without a suitable bundle, setup refuses and names the supported published
variants. It does not resolve or choose a local replacement automatically.
`dinkster-env --root /absolute/new-project setup
--from-scratch --variant linux-cpu --source /absolute/Dinkster --yes` explicitly
resolves the checkout lock against the variant's torch index and builds engine
wheels locally. The plan states that resolution occurs before downloading.
It does not fetch a channel or an engine/base release. The pinned interpreter
is downloaded and verified from the producer's interpreter definition, or reused
from the store's download cache. Populate that cache before denying interpreter
release access. `--python /absolute/python` explicitly selects a host interpreter
instead, including when a host interpreter row is missing.
`--variants /absolute/variants.toml` selects alternate producer definitions.

Contributor setup exposes the same opt-in with `python scripts/setup_envs.py
--from-scratch --variant linux-cpu` and the same `--python`/`--variants` overrides.
Neither Desktop nor channel updates choose it automatically. Local generations
are marked `source: local`; doctor reports their local origin and checks their
recorded inventory. Channel update refuses them: install a fresh published
generation in a new root without `--from-scratch` to return to channel updates.

### Windows interpreter

Windows bases use CPython 3.14.10 / cp314 from python-build-standalone 20261026,
with SQLite 3.53.1 plus the upstream
[mapped-shm recovery fix](https://sqlite.org/src/info/7c103a0b7d).
The CPython 3.14.10 pin requires **uv 0.13.6 or newer**, the first uv catalog
that lists this interpreter. Before running the commands below, check
`uv --version`; an older uv cannot launch the outer `uv run --python 3.14.10`
command. Keep uv's managed-interpreter downloads enabled (do not set
`UV_PYTHON_DOWNLOADS=never` or use `--no-python-downloads`) while acquiring
the bootstrap interpreter. Move this minimum uv version together with the
interpreter pin. Setup checks uv before its own downloads or environment
creation and refuses an unsupported version; it never upgrades uv automatically.
The unmodified interpreter fails WAL reopening while an old shared-memory
mapping survives ([upstream report](https://sqlite.org/forum/forumpost/f9c68d01e2d7f638)).
A version string cannot distinguish the patched library. Acquisition and bundle
extraction require a functional WAL read and integrity check with the mapping
held open, followed by a released-mapping control.

`scripts/setup_envs.py` installs the published base into its private store and
uses it for `.venv`. On Windows the selected environment must pass the functional
check and use that exact private base, including when reusing an existing store
or environment. Use `--force` to recreate an environment using a different base.
For manual root-only setup from the checkout:

```powershell
uv run --no-project --python 3.14.10 scripts/prepare_windows_python.py --output .run/windows-python
uv sync --python .run/windows-python/python/python.exe --all-packages --frozen
```

Preparation verifies immutable Dinkster-Environments source pins, the original
interpreter archive, and the replacement DLL. Zig 0.14.1 is installed in an
isolated uv environment and verified against its exact Windows wheel SHA-256.
Only `python/DLLs/sqlite3.dll` changes; uv's interpreter cache is never modified.
The base manifest records the amalgamation, patch, compiler, flags and DLL
hashes; Windows bundle CI must reproduce the pinned DLL hash. Linux and macOS
are unchanged. When a verified upstream interpreter contains the fix, move
the pins together and remove the override, retaining the functional check.

| Consumer | Interpreter selection | Bundled SQLite |
| --- | --- | --- |
| Environments Windows bases and shared contributor setup | PBS 20261026, CPython 3.14.10 / cp314 | 3.53.1 with upstream fix |
| Environments Linux base and shared contributor setup | PBS 20261026, CPython 3.14.10 / cp314 | 3.53.1 unmodified |
| Windows full-validation shards, PR runtime job and release install | Verified private Windows interpreter | Patched 3.53.1 |
| Windows dependency receipt job | Published Windows CPU base | Patched 3.53.1 |
| Windows acquisition bootstrap | uv CPython 3.14.10; no Dinkster database use | Unpatched bootstrap may carry 3.53.1 |
| Linux root CI/full-validation and macOS release smoke | Floating uv `3.12` | PBS 20261026 CPython 3.12.14 carries 3.53.1 |
| Linux release runtime | CPython 3.14.10 | PBS 20261026 carries 3.53.1 |
| OpenTimelineIO-only dependency checks, including Windows ARM64 | Floating uv `3.14`; no Dinkster stores | PBS 20261026 carries 3.53.1 |
| Desktop environment client | Published platform base through `dinkster-env` | Windows base carries patched 3.53.1 |

PBS versions come from the
[20261026 dependency manifest](https://github.com/astral-sh/python-build-standalone/blob/20261026/pythonbuild/downloads.json);
Windows 3.14.10 was also executed from the size/hash-verified archive.
Floating selectors can choose other catalog releases or local installations;
they do not promise a SQLite version. Windows admission checks the interpreter
actually selected. The legacy Desktop source installer selected floating uv
`3.12` / cp312 and was exposed when it resolved to affected PBS builds; the
[environment-client installer](https://github.com/Kosinkadink/Dinkster-Desktop/pull/23)
uses published base acquisition instead.

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
