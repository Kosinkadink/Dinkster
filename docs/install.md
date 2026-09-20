# Install Dinkster

Dinkster is pre-release. You need access to the private
[backend release](https://github.com/Kosinkadink/Dinkster/releases). The
Desktop app is unreleased and has no downloadable installer. A GitHub 404 can
mean missing access, not a missing backend release. Nothing here
publishes Dinkster to a public package index. The pinned identity dependency is
bundled in the backend archive; installation needs no Git or GitHub token
after you download the private release assets.

## Choose an installation

- **Desktop app:** unreleased on every platform. The source and current support
  boundaries are documented in the
  [Desktop guide](https://github.com/Kosinkadink/Dinkster-Frontend/blob/main/docs/desktop.md).
- **Backend plus browser frontend:** use the steps below on Windows x64,
  Linux x64, or macOS Apple Silicon. The backend installer installs the CPU
  graph host and pack manager, not model weights or a GPU execution runtime.
  Model execution needs the separately configured runtime described below.

Use a short writable path outside OneDrive or network shares, for example
`C:\Dinkster` or `~/Dinkster`. Allow at least 3 GB for the host and dependencies;
models and accelerator environments require substantially more. No administrator
access is needed. Keep the extracted application directory: its environment
refers to the source files in it. Store your library outside it.

## Install the backend

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
   Restart your terminal and check `uv --version`.
   uv downloads Python 3.12 if necessary; an existing system Python is not
   required. Internet access is needed for Python and declared dependencies.
2. Download the `dinkster-backend-<commit>.zip`, `SHA256SUMS`, and
   `backend-release.json` assets from the same backend release while signed in.
   Check the ZIP's SHA-256 against `SHA256SUMS` before extracting:
   Windows: `Get-FileHash .\dinkster-backend-<commit>.zip -Algorithm SHA256`;
   Linux: `sha256sum -c SHA256SUMS`; macOS: `shasum -a 256 -c SHA256SUMS`.
   Replace `<commit>` with the actual filename's revision.
3. Extract the ZIP and open a terminal in its `dinkster-backend-<commit>` folder.
   Run this identical command on all three operating systems:

   ```sh
   uv run --no-project --python 3.12 scripts/install.py
   ```

   The installer uses `uv.lock` and creates only this folder's `.venv`. It
   does not change system Python, start services, or install a GPU driver.
   In `backend-release.json`, `requiresPython` is the package's compatible
   Python range, while `bootstrap.python` is the recommended interpreter for
   running this installer. uv obtains that Python and the locked dependencies
   over the Internet. Git and repository credentials are not required after
   acquiring the private release assets.
4. Check the installed host:

   ```sh
   uv run --no-sync dinkster demo
   uv run --no-sync dinkster-pack --help
   uv run --no-sync dinkster-installs --help
   ```

Use `--no-sync` for commands after installation so uv does not add development
dependencies or modify a separately configured runtime. Commands below run
from the extracted application directory.

## Install one pack from a registry

Ask your registry operator for its endpoint, pack name, and exact version.
There is no implicit public Dinkster registry. Install only publishers you trust:
pack installation and execution run Python code. A private registry must be
reachable only by authorized users; download authentication depends on its
operator's deployment, not on the backend release being private.

For a reproducible local trial, use the loopback registry below, whose pack is
the template included in this release. Then install it with this command
(one line, identical on PowerShell and POSIX shells):

```sh
uv run --no-sync dinkster-pack --root ../packs --accelerator cpu --registry http://127.0.0.1:8791 --workspace-package packages/dinkster-workers --workspace-package packages/dinkster-protocol --workspace-package packages/dinkster-schema --workspace-package packages/dinkster-values --workspace-package packages/dinkster-api --workspace-package packages/dinkster-memory --workspace-package packages/dinkster-assets --workspace-package packages/dinkster-caches --workspace-package packages/dinkster-inference --workspace-package packages/dinkster-video install my-pack@0.1.0 --yes
```

For your registry, replace the endpoint and `my-pack@0.1.0`. The
`--workspace-package` arguments use this release's worker host code; these
packages are not assumed to exist on public PyPI. The command prints its plan,
verifies the archive identity, provisions an isolated pack environment, and
prepares its schema catalog before activating the new generation. Omit `--yes`
to review and confirm interactively. Do not use `--no-venv` for a normal install.

Start the backend with the installed pack:

```sh
uv run --no-sync dinkster-serve --host 127.0.0.1 --port 3639 --library-root ../library --install-root ../packs --no-default-packs
```

Open `http://127.0.0.1:3639/api/composition` and
`http://127.0.0.1:3639/api/nodes` in a browser. The installed pack should be
announced, not failed or missing. These URLs return JSON, not the editor.
Stop the foreground backend with Ctrl+C. If a port is in use, choose another
port for this installation; do not stop another application's process.

### Loopback registry for the included template

This optional trial publishes only the release's own `templates/pack` code to
a registry on your computer. It is not a public publication. Install the
SQLite CLI from the separate
[registry repository](https://github.com/Kosinkadink/dinkster-registry#sqlite-registry)
and put its `dinkster-registry-sqlite` command on PATH. The backend does not
include a registry server. Use a new `../registry` directory, then run:

```sh
dinkster-registry-sqlite --data ../registry admin add-user local --operator
dinkster-registry-sqlite --data ../registry admin add-publisher local --owner local
```

Mint a short-lived publisher token into a terminal variable, not a file.
Use an expiry in the future. PowerShell:

```powershell
$expires = (Get-Date).ToUniversalTime().AddHours(1).ToString('yyyy-MM-ddTHH:mm:ssZ')
$env:DINKSTER_REGISTRY_TOKEN = dinkster-registry-sqlite --data ../registry admin mint-token local --user local --expires $expires
```

POSIX shell:

```sh
export DINKSTER_REGISTRY_TOKEN="$(dinkster-registry-sqlite --data ../registry admin mint-token local --user local --expires "$(uv run --no-project --python 3.12 python -c 'from datetime import datetime,timedelta,timezone; print((datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())')")"
```

In a second terminal in the application directory, start the registry:

```sh
dinkster-registry-sqlite --data ../registry serve --host 127.0.0.1 --port 8791 --probe-sandbox off
```

`--probe-sandbox off` is only for this loopback trial of bundled trusted code.
It executes the doctor without a Linux sandbox. Never use it for an exposed
registry or to inspect an untrusted pack. Return to the first terminal to
publish:

```sh
uv run --no-sync dinkster-pack --registry http://127.0.0.1:8791 publish templates/pack --version 0.1.0
```

The first namespace claim needs operator approval. For this bundled template
only, the local operator can approve it through the authenticated review API.
PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8791/reviews/my-pack/versions/0.1.0/resolve -Headers @{Authorization="Bearer $env:DINKSTER_REGISTRY_TOKEN"} -ContentType application/json -Body '{"decision":"accepted","reason":"Reviewed the bundled release template"}'
```

POSIX shell (requires curl):

```sh
curl --fail --config - <<EOF
url = "http://127.0.0.1:8791/reviews/my-pack/versions/0.1.0/resolve"
header = "Authorization: Bearer $DINKSTER_REGISTRY_TOKEN"
header = "Content-Type: application/json"
data = "{\"decision\":\"accepted\",\"reason\":\"Reviewed the bundled release template\"}"
EOF
```

On PowerShell use `Remove-Item Env:\DINKSTER_REGISTRY_TOKEN`; on POSIX use
`unset DINKSTER_REGISTRY_TOKEN`. The review response must say `accepted` before
installing. Keep the registry terminal running while installing the pack;
after installation, stop that registry with Ctrl+C.

## Browser frontend and native generation

For a source checkout, the default launcher serves the built companion frontend
and engine on one loopback origin. Build the sibling frontend once, prepare the
local installation, and launch:

```sh
cd ../Dinkster-Frontend
pnpm install --frozen-lockfile
pnpm --filter @dinkster/app build
cd ../Dinkster
uv run --no-sync dinkster setup
uv run --no-sync dinkster
```

The browser opens at `http://127.0.0.1:3639`; the application and API share
that origin. `dinkster setup` creates the default library and pack roots, and
the launch prepares missing or stale catalogs. See the
[browser editor quickstart](quickstart.md) for platform notes and the first
image steps. Do not expose the engine directly to the Internet; read
[authentication](auth.md) before configuring shared access. The
[frontend installation guide](https://github.com/Kosinkadink/Dinkster-Frontend/blob/main/docs/desktop.md)
documents the unreleased Desktop application separately.

For native generation, install a supported PyTorch execution environment using
the [runtime setup instructions](../packages/dinkster-inference-torch/README.md).
Windows/Linux NVIDIA execution installs the exact `dinkster-aimdo==0.5.5.post2`
wheel from PyPI. macOS does not use Aimdo. Models are
not bundled; consult [supported models](supported/model-families-native-execution.md)
before downloading.
Select the execution interpreter with `--comfy-python` and omit
`--no-default-packs` when serving the default suite. After installation, run
`uv run --no-sync dinkster-pack prepare-catalogs --defaults --library-root ../library`
using the same interpreter via `DINKSTER_COMFYUI_PYTHON`. Rerun it after changing
the backend checkout or any default-pack dependency. The server refuses to bind
when a required catalog is missing or stale. Today, the working native SD 1.5
path is a SamplerCustomAdvanced graph served with `--comfy-root`; the audited
run is recorded in [maintainer issue #114](https://github.com/Kosinkadink/comfy-vibe-station/issues/114).
The native-only claim returns when that issue lands. See
[the complete server reference](serve-cli.md).

Catalog preparation probes trusted installed code and validates its runtime
declarations. It is not a substitute for `doctor` authoring checks or registry
publication review. Use the same `DINKSTER_ACCELERATOR`, library root, and uv on
`PATH` for preparation and serving. Default-pack environments are created
under the library's `venvs` directory; `DINKSTER_SERVING_PYTHON` overrides that
isolation and should remain unset for normal installations.

## Register, upgrade, and remove an installation

`dinkster-installs` records installations for the optional station supervisor;
it does not download Dinkster, create environments, or start a station:

```sh
uv run --no-sync dinkster-installs --config ../installs.toml add local --root ../packs --port 3639 --yes
uv run --no-sync dinkster-installs --config ../installs.toml list
uv run --no-sync dinkster-installs --config ../installs.toml show local
```

Use a unique name, root, and port per registered installation. Runtime
`start`, `stop`, and `restart` commands require a running station management
endpoint; use `dinkster-installs --help` rather than treating this registry as a
service installer. `remove local --yes` removes only the registration.

To upgrade, stop your own backend, back up the external library and pack root,
extract the new release beside the old one, and repeat installation. Keep the
old directory for rollback. Refresh catalogs using the new release before
serving. Never replace files in a running environment. Pack changes use
`dinkster-pack --root ../packs update` (with the same workspace and registry
arguments as installation); inspect `dinkster-pack --help` for generation rollback.
To uninstall, stop your own process and remove only its extracted application
directory. Keep or explicitly back up your library and pack data first.

## Release maintainers

[Private backend release](../.github/workflows/release.yml) runs manually from
`main` and creates the immutable tag `backend-<full commit SHA>`. It refuses a public or
foreign repository and requires successful hosted CI for that exact main
commit. The build bundles the exact identity source pinned by the backend
lockfile, substitutes its local path only in the extracted release, and
regenerates the release lockfile without allowing dependency versions to change.
The manifest records `workerProtocol` and `requiresPython` from the selected
backend source, the installer bootstrap requirements, and `identityCommit`
from its lockfile, alongside the archive hash, source SHA, and explicit
`releaseTag`. Revisions without the installer script are rejected.

Supported Windows native pins are bound to the backend revision in
[`scripts/desktop_windows_runtime.json`](../scripts/desktop_windows_runtime.json).
The builder retains this file unchanged in the ZIP and includes its data as
`desktopWindowsRuntime` in the manifest. Changing these pins requires a new
backend revision; a backend commit plus the CPU/NVIDIA variant identifies a
Desktop environment. Desktop's mirrored pins must match the embedded data
before packaging, so frontend-only pin edits cannot redefine that environment.
The builder checks the selected source's Aimdo helper constants, Torch base
version in `uv.lock`, and exact torchvision requirement in the
[BiRefNet pack declaration](../packages/dinkster-nodes-vision/dinkster_vision_birefnet_pack/dinkster-pack.toml).
That pack declaration owns the isolated worker's torchvision dependency;
torchvision is not added to the root lock or CPU host environment.

Windows, Linux, and macOS jobs install the same archive without a Git checkout
or private dependency credentials before publication. Only the build job needs
the existing read-only `DINKSTER_IDENTITY_DEPLOY_KEY` secret.
It creates a private prerelease in this repository, uploads the ZIP, checksum,
and source manifest, then makes the draft visible to repository readers.
It never uploads packages to PyPI or a public mirror. An interrupted upload
leaves a draft for inspection, not a partial published release.

For a local candidate, run `uv run --no-project --python 3.12 scripts/build_release.py`.
The builder needs Git and read access to the private identity repository.
Pass `--identity-source PATH` to use an existing clone containing its pinned
revision. Only committed backend and identity source files enter the archive,
plus the release-specific dependency metadata; uncommitted changes are excluded.
Verify a downloaded published archive as
well as local candidates. After installing, run
`uv run --no-sync python scripts/verify_release_install.py . --registry-command PATH`
(PATH is the separately installed `dinkster-registry-sqlite` executable) to
exercise a real loopback registry publication, isolated pack installation,
backend catalog and installation registration. The verification uses temporary
data and unused loopback ports, and stops only processes it starts. Pass
`--state PATH` with a new directory to retain its evidence instead of deleting
it. Independent review and passing CI remain required before merging release
changes.
