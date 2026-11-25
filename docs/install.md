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

## Install from pip

Install `dinkster` into a Python 3.12 or newer environment, then select the
published runtime variant for your machine:

```sh
python -m pip install dinkster
dinkster setup --variant linux-cpu --channel nightly
dinkster
```

Use `win-cpu` on Windows or `mac-arm64` on Apple Silicon. Other published
variants can be selected explicitly. Setup prints the engine, Python/base,
default packs, exact source and dependency-lock identities, installation paths,
payload sizes and cache reuse before asking for confirmation. It acquires the
reviewed runtime and locked pack wheels only after confirmation. Declining
changes nothing. `--plan` only inspects metadata; `--yes` prints and applies the
plan without prompting for unattended deployment. Without a terminal, applying
requires `--yes`. With runtime inputs, metadata, locks and selected wheels
already cached, `--offline --yes` applies without network access. Missing or
corrupt inputs are listed before installation state changes.

The default environment root remains `~/.dinkster/project`, or
`$DINKSTER_HOME/project`. These existing directory names do not define a project.
Pass `--root PATH` to each command to select a different environment and `--data PATH`
to setup for its shared software store. `dinkster` starts the always-on Dinkster
process at `http://127.0.0.1:3638` if it is not running, opens the environment's
engine behind it when the environment has a snapshot, and opens the browser on that
address; `--no-browser` skips the browser. The page loads before the engine is ready,
and with no environment at all; restarting the engine with none open opens the default
environment, or the only one when the default is not set up. `dinkster serve` and `dinkster-serve` start the active
snapshot on its own port without opening a browser; started without setup they refuse
and install nothing. The supervisor keeps the engine running after the command
returns. `dinkster stop` stops the always-on process and every engine it opened;
`dinkster stop --engine-only` stops only the environment's engine.

Review a channel update with `dinkster update --plan`, then apply it with
`dinkster update` or `dinkster update --yes`. Use
`dinkster restore-previous --plan` to review the previous software snapshot,
then `dinkster restore-previous` to restore it.
An approximate restore requires its displayed proposal approval; `--yes` alone
does not approve substitutions. These operations use the same snapshot
management as Desktop and leave models, outputs and workflows untouched.

## Inspect a wheel release

The `dinkster` wheel does not require torch or `dinkster-inference` by default.
For trusted torch nodes hosted in the serving process, install `dinkster[inference]`
from the same release. The extra pins the inference runtime for explicit library
or in-process use; it does not change public CLI setup or serving. Pip setup and
Desktop build the same managed snapshot, where locked packs share the
snapshot's inference and torch distributions.

Each `vX.Y.Z` release contains:

- the `dinkster`, `dinkster-api` and `dinkster-supervisor` wheels, with the
  browser application embedded in the engine;
- `constraints.txt`, exported from `uv.lock` with hashes;
- `release-manifest.json` with exact per-variant dependency identities and `SHA256SUMS`;
- `dinkster-source-X.Y.Z.zip`, a maintainer source archive;
- native tools ZIPs with the installer interpreters.

Official packs and their per-base dependency locks come from the registry,
not the engine release or wheel. No third-party wheels are release assets.
Finalization rejects pack payloads and other unexpected artifacts and fails
above 100,000,000 bytes excluding native tools ZIPs.
Confirmed `dinkster-env` setup/update fetches non-base dependencies from their
locked PyPI artifact URLs into the shared wheel store, checking the exact
filename, size and SHA256. Base-owned names and exact variant-index wheel pins
come from the selected Environments manifest. Snapshot construction uses
only the store, offline, without resolution or source builds.

`setup --package-index https://packages.example.org/simple` selects a simple
package mirror instead of those locked artifact origins. The index must offer
the exact locked filename; index hashes cannot change the required size/hash.
There is no fallback to PyPI or another version. Confirmed setup saves the
index for updates; `update --package-index URL` changes it transactionally.
This option is separate from `--mirror`, which selects channel, base and
Dinkster release artifact locations. Warm shared-store wheels need no index
access. Repair and move remain offline.

For cached published setup, use `dinkster setup --offline --yes --variant VARIANT`
with the same `--channel`, `--mirror`, `--registry` and `--data` selection used
to populate the cache. Retained channel, release, registry and dependency-lock
metadata, runtime payloads and locked pack wheels must all be available.
Missing or corrupt inputs refuse before changing the environment or store; the
command does not fetch replacements. `--offline --plan` inspects cached inputs
without creating or changing installation state.

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

Use writable scratch paths that are separate from an installed snapshot.
On Windows, use the target environment's `Scripts\python.exe` and Windows
paths. The JSON result lists selected ids and an `unavailable` map with a
reason for each missing pack. With no registry access, `--allow-missing`
produces an empty source lock; without it, any unavailable pack refuses the
command. An engine serving an explicitly empty snapshot remains healthy
with zero nodes rather than loading checkout defaults.

This command prepares verified sources only. It does not install the engine,
obtain dependency locks, build a snapshot or activate it. For an update,
use `--installed-lock /retained/dinkster.lock` instead of fresh selection;
the output lock must have a different path. Installed official registry packs
are re-resolved without downgrading, adding defaults or changing their recorded
registry. Other installed selections are preserved for admission. Update
preparation refuses unavailable compatible releases and does not allow
`--allow-missing`.

For metadata-only review, `plan-registry-sources` takes `--engine-manifest`,
`--engine-release`, `--variant` and `--base-id`. Add `--installed-lock` and
repeat `--add PACK@VERSION` to inspect exact additions or replacements, or
use `--update [PACK ...]` for compatible installed registry updates. No names
means all installed registry packs. Unselected sources remain unchanged;
updates use their recorded registry and never downgrade. The JSON plan includes
published dependency locks and a pack payload size upper bound, but acquires
no pack archives and does not create a snapshot or write a cache.

For an active managed environment, `dinkster pack --root ROOT install PACK@VERSION`
reviews published pack additions or replacements against its retained engine,
base and variant. `update [PACK ...]` reviews compatible installed registry
updates. The terminal displays old/new source identities, the proposed snapshot,
published dependency-lock identities, verified cached versus downloadable payloads,
wheel-byte upper bounds and worker cost before confirmation. Use `--yes` for unattended
application, or `--plan FILE` to save the review outside the environment and store.
`dinkster pack --root ROOT apply FILE` rechecks that review before using the
shared snapshot builder and activation. Saved reviews expire after 15 minutes;
changed environment or selected release metadata requires a new review. Local/git
sources cannot be mixed with published selections; use `dinkster develop` for
explicit unlocked development.

Both confirmed commands acquire the selected archives and their published
dependency locks, then build and activate a fresh snapshot without package-index
resolution. The retained engine and base do not change. Release and lock metadata
failures are reported together; confirmed source and dependency-wheel acquisition
failures are also collected together. A missing retained base refuses before
source or dependency-wheel acquisition, so that refusal does not list later
payload failures.

For a source lock containing registry packs, `prepare-registry-locks` accepts
`--catalog-lock`, a new `--output` directory, `--variant` and the exact
`--base-id sha256:...`. It downloads the dependency locks already published
for those exact source archives and base, verifies their size, BLAKE3 and
pack/variant identities, and reports their paths as JSON. It never resolves
dependencies. Any missing or mismatched lock refuses the whole set without
creating the output directory or changing an installed snapshot.

Confirmed published `dinkster-env setup` and `update` run these commands with
the acquired target engine, in isolated temporary storage before opening an
activation journal. Fresh setup reports unavailable defaults in
`unavailablePacks` and on stderr, and can activate an empty snapshot when
the registry is offline. Updates re-resolve the installed official selection;
an unavailable compatible release refuses without changing environment settings,
the current snapshot or its journal. Non-registry sources are retained and
checked against their recorded digests and the target engine; their dependency
locks remain snapshot-owned. Explicit catalog/artifact and dependency-lock
inputs override automatic preparation. `--from-scratch` remains a separate
checkout-only path.

The preparation commands can also be run separately. Source preparation's
`--retain-sources` requires `--installed-lock` and caches unchanged sources
from that snapshot's `pack-sources` tree. Lock preparation's `--registry-only`
leaves non-registry dependency locks to the retained snapshot.

Pack versions are independent of engine versions. A pack's `[pack] version`
and `engine` fields declare its version and PEP 440 engine range; excluded
engines refuse installation, including an identical-lock no-op. See
[pack compatibility and dependency locks](environments-and-updates.md#versions-and-engine-compatibility)
and the [registry operating notes](registry-operations.md).

## Develop from source

Run `./run.sh` on Linux/macOS or `.\run.ps1` on Windows. The launcher builds
the pinned frontend and prepares a contributor snapshot with a live engine.
For setup and tools without a frontend build:

```sh
uv run --no-project python scripts/develop.py setup --yes
python scripts/develop.py run -- python -m pytest -q
python scripts/develop.py run -- ruff check .
python scripts/develop.py run -- pyright
```

Setup creates a sibling `.<checkout-name>-development` environment, not a root
`.venv`. `run` executes commands in its active snapshot, preserves the working
directory, forwards input/output and returns the command's exit status. Plain
`uv run` and `uv sync` in the checkout cannot replace that snapshot's torch.
Rerun setup after dependency changes. Source-only engine edits need a stop/start:
`python scripts/develop.py stop`, then `python scripts/develop.py start`.
Use `python scripts/develop.py --root /absolute/environment ...` to select another
environment consistently for setup, tools and runtime commands.

CPU is the default on Linux and Windows. Select `--variant linux-nvidia` or
`--variant win-nvidia` explicitly for an NVIDIA development runtime. Apple Silicon
uses `mac-arm64` with MPS; Intel macOS is unsupported. Development resolves the
checkout's locked requirements against the selected variant's torch index,
records the resulting inventory and uses the snapshot's isolated interpreter.
It does not change published setup or channel selection.

The terminal and served catalog label live development as unreviewed and not
reproducible from dependency ranges. For captured development instead, use
[`dinkster develop`](#development-packs-from-local-folders). Do not expose the
engine directly to the Internet; read [authentication](auth.md) before configuring
shared access.

### Apple Silicon macOS

Apple Silicon on macOS 14 or newer is supported through the `mac-arm64` base.
The native `dinkster-tools-macos-arm64.zip` contains its own Python and uv;
environment setup, doctor, engine start, update and restore do not require
host Python or uv. Verify the ZIP against the same release's manifest and
`SHA256SUMS`, then extract it with executable permissions preserved. Invoke
`python/bin/python3 -I -m dinkster_supervisor.environment` from that tree,
prepending its absolute `python/bin` directory to that command's PATH so it
uses the bundled uv. Do not change the global PATH.
Setup selects `mac-arm64` automatically; select the `nightly` channel to use a
pre-release engine. Intel Macs and CUDA on macOS are not supported.

The arm64 Desktop ZIP contains an unsigned `.app`. Extract it and copy the app
to a directory you control. macOS may require right-click **Open** or removal
of the `com.apple.quarantine` attribute from that verified app before launch.
This is an unsigned-build procedure, not a signature or notarization check.
Apple signing and notarization are tracked separately in
[Apple signing](https://github.com/Kosinkadink/comfy-vibe-station/issues/501);
they are not prerequisites for the tools or engine environment.

A maintainer evidence checkout is optional. Select its acceptance package with
`--library-source /absolute/evidence/packages/dinkster-acceptance` during
contributor setup. `python scripts/develop.py run -- COMMAND` sets
`DINKSTER_EXECUTION_PYTHON` to the active snapshot's interpreter automatically.
The snapshot includes Dinkster's pinned `dinkster_inference` runtime.

### User-defined environment variants

User-defined torch versions and indexes are a supported `dinkster-env` path,
not a development override and not `--from-scratch`. Put `variants.toml` in
the environment root or the shared store, or select it with `--variants PATH`.
An explicit file wins over the environment file, which wins over the store file.
Use a distinct variant name for your custom base. For example:

```toml
[variants.linux-custom-cuda]
os = "linux"
arch = "x86_64"
torch = "2.14.1"
torch-index = "https://download.pytorch.org/whl/cu130"
extras-index = "https://pypi.org/simple"
```

Choose an exact torch version that your index supplies for the native Python
and platform. `torchvision` may also name an exact version; otherwise it is
resolved with torch. `extras-index` defaults to PyPI. Torch comes only from
`torch-index`; other base dependencies try that index, then `extras-index`,
then PyPI. Index URLs must use HTTPS without embedded credentials.

```sh
dinkster-env --root /absolute/environment setup --channel nightly \
  --variant linux-custom-cuda --variants /absolute/variants.toml --yes
```

Setup builds a real content-addressed base in the shared store using a verified
native interpreter. Its manifest records the source file, definition digest,
build time and wheel origins. It then installs the selected channel engine
offline onto that base. Status, doctor and Desktop show the user-defined
variant, torch version and torch index.

Channel updates retain the custom base rather than replacing it with a
published one or resolving it again. An engine dependency requiring a newer
torch refuses before activation, reports required and present versions, and
explains how to rebuild the custom base. Editing `variants.toml` alone does
not mutate existing snapshots. Selecting a rebuilt variant uses the normal
confirmed update and restore journal. Restore rebuilds recorded software as a
fresh snapshot from retained inputs, reusing the recorded published or custom
base without rebuilding that base.

### Frontend selection

Managed `dinkster setup` and `dinkster update` accept `--frontend-version VERSION`
to select an exact published `dinkster-frontend` release instead of the channel
pin. This explicit online operation reads the permanent PyPI wheel, verifies
its size and SHA-256, and checks its declared engine API range before acquiring
or activating a snapshot. Plans disclose the selected version, hash and API
range. JSON plans expose `frontendApproval`; repeat it with
`--frontend-approval DIGEST --yes` to require the reviewed wheel identity.
The snapshot retains the selection for exact restore and export.
Without this option, the engine/channel frontend pin remains authoritative.
Release installation does not contact PyPI implicitly.

`dinkster-serve --frontend-root PATH` serves an already-built frontend bundle.
`dinkster-serve --frontend-dev URL` proxies a running frontend development
server, including websocket live reload. Neither flag builds a frontend or
changes the managed snapshot's package selection.

### Managed engine sources

On an installed release environment, `dinkster source --root ROOT add
https://github.com/Kosinkadink/Dinkster --ref BRANCH --yes` selects an explicit
Git engine without enabling develop mode. Omit `--yes` to inspect the plan.
The snapshot keeps the installed release base and pinned frontend package,
reuses matching locked wheel bytes, and resolves missing checkout dependencies
from PyPI or `--package-index URL`. Workspace packages run editable from the
captured checkout. An incompatible worker protocol or base dependency refuses
before activation.

`dinkster source --root ROOT update --ref OTHER --yes` selects another ref;
omit `--ref` to fetch the previously selected ref again. Status and history
label source snapshots unpublished and non-reproducible and record the
original URL, requested ref and exact commit. Restore uses the recorded commit,
not the current branch tip. Recovery requires retained locally built wheel
bytes; unavailable inputs are reported rather than replaced by a new build.
`dinkster source --root ROOT release --yes` restores the most recent recorded
published snapshot. Ordinary release setup remains unchanged and does not
fetch Git sources implicitly.

For engine, pack, library and frontend add/update commands, `--plan --json`
returns `sourceApproval`. Repeat the same selection with
`--source-approval sha256:DIGEST_FROM_PLAN --yes` to bind confirmation to the
reviewed environment state, exact Git content, mirror/package-index settings and
frontend CI bundle (or local-build fallback). Changed inputs require a new plan;
temporary checkout paths do not invalidate an unchanged selection. The same flag
is accepted by `update --engine-source URL`. Direct `--yes` remains an explicit
opt-in without a prior plan; `source release` retains history restore semantics.

### Managed pack sources

Select a single pack without replacing the installed engine or other packs:

```sh
dinkster source --root ROOT add https://github.com/OWNER/PACK --pack PACK_NAME --ref BRANCH --yes
dinkster source --root ROOT update --pack PACK_NAME --ref OTHER --yes
```

Omit `--ref` on update to refresh that pack's recorded ref. Omit `--yes` to
inspect its source identity without changing the environment. `--git-mirror` and
`--package-index` apply to the explicit source and its dependencies. The pack
must declare a compatible engine range and version. Its dependencies resolve
against the installed snapshot's constraints; incompatible requirements
refuse before activation. Unselected source entries and dependency locks remain
unchanged. The snapshot retains its release base, engine and pinned frontend.
Status and history disclose the pack URL, requested ref, exact commit and
unpublished/non-reproducible status. Restore uses the retained source archive
or verifies an archive reacquired at the recorded commit, never a moved ref.
Channel updates retain that disclosure and mirror identity while the selected
pack source and archive digest remain unchanged. Replacing the pack drops its
old source identity. The local directory `--pack` development loop is unchanged.

### Managed inference and helper sources

Select a shared Python library on an installed release base:

```sh
dinkster source --root ROOT add https://github.com/OWNER/LIBRARY --library DISTRIBUTION --ref BRANCH --yes
dinkster source --root ROOT update --library DISTRIBUTION --ref OTHER --yes
```

Use `--library dinkster-inference` for inference. A repository subdirectory is
selected with `#subdirectory=PATH`. Omit `--yes` to inspect the Git identity;
omit `--ref` on update to refresh the recorded ref. These operations build a
captured library wheel without enabling develop mode or rebuilding the release
base. Engine and frontend wheels, unselected library pins and pack selections
remain intact. Missing dependencies resolve against those exact pins. A
conflicting dependency or attempt to replace a base, engine or frontend
package refuses before activation. `--git-mirror` and `--package-index` have
the same meanings as for managed pack sources.

Status and history record each library's original URL, requested ref, exact
commit, content digest and built wheel identity, and disclose unpublished,
non-reproducible software. Restore selects the recorded wheel bytes rather
than rebuilding from a moving ref. Recovery requires retained local builds;
missing inputs are reported, not silently rebuilt. `source release --yes`
restores the most recent published snapshot.

### Managed Git frontend assets

```bash
dinkster source --root ROOT add https://github.com/Kosinkadink/Dinkster-Frontend --frontend --ref BRANCH --plan
dinkster source --root ROOT add https://github.com/Kosinkadink/Dinkster-Frontend --frontend --ref BRANCH --yes
dinkster source --root ROOT update --frontend --ref OTHER --yes
```

An official frontend commit uses its unexpired bundle from a successful frontend
CI run. Discovery uses the public GitHub API; the public nightly.link transport
needs no GitHub login. The GitHub artifact digest, inner bundle hash/size,
package version and exact commit are verified before packaging browser assets.
A corrupt or mismatched artifact refuses rather than falling back to a build.
If no eligible bundle exists, or the source is a fork, local building requires
both Node and pnpm on PATH. Missing tools are named explicitly. Local builds run
`pnpm install --frozen-lockfile` and `pnpm --filter @dinkster/app build` in the
captured checkout, so source selection executes that repository's build scripts.
The selected checkout's frontend API range must include the installed engine API.

The release base, engine, other libraries and pack selections remain unchanged.
Status/history record frontend URL/ref/commit, captured source digest, CI or local
build origin and exact wheel bytes; the snapshot is unpublished and
non-reproducible. Ordinary updates retain those bytes, not a moving frontend ref.
An explicit `--frontend-version` replaces the Git frontend with that published
selection. Recovery requires retained built bytes and never silently rebuilds;
`source release` returns to the most recent published snapshot. Unselected
release installs do not discover CI artifacts or require Node, pnpm or Git.

### Public git refs and mirrors

`dinkster source refs https://github.com/Kosinkadink/Dinkster --json` lists
branches, tags and other advertised refs through bundled libgit2, without a
system git executable or GitHub API access. It does not install or execute
repository code or create an environment.

Set `DINKSTER_GIT_MIRROR=https://your-mirror.example` to rewrite explicit
`https://github.com/Kosinkadink/REPOSITORY` acquisitions to
`https://your-mirror.example/Kosinkadink/REPOSITORY.git`. The `.git` suffix is
always present. Other owners and hosts are not rewritten. `source refs` and
`develop` also accept `--git-mirror URL`, which overrides the environment
setting. Original source URLs, requested refs and exact commits remain the
recorded identities; a mirror changes transport, not provenance. Mirrors
require credential-free HTTPS; loopback HTTP is allowed for local mirrors.
There is no fallback to GitHub when a selected mirror fails. This setting
does not change published release mirrors or enable git sources implicitly.

### Development packs from local folders

The standalone local/git path is development-only. `dinkster pack --root ROOT
install --yes ./pack` and `dinkster pack --root ROOT install --yes
git+https://host/pack.git@REF` retain plan/consent and resolve dependency ranges
without requiring a published lock. The plan labels these sources as unreviewed
and not reproducible from ranges. Serve that selection explicitly with
`python -m dinkster.serve --no-default-packs --install-root ROOT`; the terminal
and node catalog carry the development label. A published registry spec is
refused on the standalone install route, never silently resolved as development.
Published entries in standalone saved plans, snapshots, workflow stamps and
restore targets also refuse, as does serving a standalone published inventory.
Use the managed environment's pack install/update and software history instead;
there is no automatic conversion of a standalone root. Catalog preparation
cannot range-resolve a published pack missing its snapshot interpreter.

Use the explicit development command when an engine, pack or helper library has
not been published. The source engine must be a Dinkster checkout. Select a new
environment outside those folders and the base variant for the host:

```sh
dinkster develop --root /absolute/environment --source /absolute/Dinkster --pack-source /absolute/pack --library-source /absolute/helper --variant linux-cpu --plan
dinkster develop --root /absolute/environment --source /absolute/Dinkster --pack-source /absolute/pack --library-source /absolute/helper --variant linux-cpu --yes
```

Repeat `--pack-source` and `--library-source` for additional local folders; omit
the library option when none is needed. The command shows its plan before
confirmation. `--yes` confirms unattended builds. Dependency ranges resolve at
build time, without a published lock. Command output and the served catalog label
this as development, not reproducible from ranges. Captured runtime artifacts
can still be restored exactly. Published setup never selects development mode
automatically. See [development snapshots](supported/installed-pack-startup.md#development-snapshots-from-local-folders-or-public-git)
for rebuilding, starting, inspecting and restoring the captured runtime.

To edit without rebuilding, add `--live-engine`, `--live-pack /absolute/pack`
and/or `--live-library /absolute/helper` to the plan and build commands. Each
live pack/library must also be selected with its source option. Mark inference
with `--library-source /absolute/dinkster-inference --live-library /absolute/dinkster-inference`.
Pack code reloads while serving; engine and library changes need stop/start,
not another build. Dependency or manifest edits need a new build. Live packs
require isolated hosting. Live snapshots change with their author folders and
cannot be restored exactly; capture without live marks for exact recovery.
Local packs support both flat and `src` Python layouts.

Development environments may select a built frontend with
`--frontend-root /absolute/bundle` (requires `index.html`), a running frontend
server with `--frontend-dev http://localhost:5173`, or a captured Git frontend
with `--frontend-source https://github.com/Kosinkadink/Dinkster-Frontend.git#ref=COMMIT`.
Git frontend selection reuses verified exact-commit CI assets when available,
otherwise Node and pnpm are required to build the selected checkout. The frontend
must support the selected engine schema API. The source and CI artifact selection
are included in the source approval; built wheels remain available for exact
recovery without another Git fetch or frontend build.
These options are exclusive, need no contributor/live engine, and persist across stop/start. HTTP and websocket
requests are proxied for live reload. Dev servers must use a loopback address
unless `--allow-frontend-dev-remote` is explicitly selected. The origin and remote
permission are part of the reviewed source approval; changing either requires a
new plan. Credentials, paths, queries and fragments are not accepted in origins.

### Explicit from-scratch development setup

Published bundles remain the supported default. For development or a platform
without a suitable bundle, setup refuses and names the supported published
variants. It does not resolve or choose a local replacement automatically.
`dinkster-env --root /absolute/new-environment setup
--from-scratch --variant linux-cpu --source /absolute/Dinkster --yes` explicitly
resolves the checkout lock against the variant's torch index and builds engine
wheels locally. The plan states that resolution occurs before downloading.
It does not fetch a channel or an engine/base release. The pinned interpreter
is downloaded and verified from the producer's interpreter definition, or reused
from the store's download cache. Populate that cache before denying interpreter
release access. `--python /absolute/python` explicitly selects a host interpreter
instead, including when a host interpreter row is missing.
With `--from-scratch`, `--variants /absolute/variants.toml` selects the
interpreter and variant definitions used for checkout-local resolution;
without it, the same option selects the supported custom-base rows above.

Contributor setup uses this local resolution through `python scripts/develop.py
setup --variant linux-cpu --yes`, with the same `--python`/`--variants` overrides.
Neither Desktop nor channel updates choose it automatically. Local snapshots
are marked `source: local`; doctor reports their local origin and checks their
recorded inventory. Channel update refuses them: install a fresh published
snapshot in a new root without `--from-scratch` to return to channel updates.

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

Contributor setup prepares the patched interpreter privately before constructing
the snapshot. The selected environment must pass the functional check,
including when reusing cached interpreter artifacts:

```powershell
uv run --no-project python scripts/develop.py setup --variant win-cpu --yes
python scripts/develop.py run -- python -m pytest -q
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

An environment-managed snapshot initializes a selected writable `output`
mount under its library root on first serve, so the first run can save files.
With the default library location this is `<environment-root>/dinkster-library/output`;
`DINKSTER_LIBRARY_ROOT` or `--library-root` can select another location.
The grant persists in `mounts.toml` and is reported by `GET /api/mounts` as
`outputMount: "output"` with a `readwrite` mount. Existing output selections,
read-only grants, and deliberately removed defaults are preserved. Initialization
alone does not enable runtime mount editing. Managed local `dinkster-env`
launches also enable operator folder editing; direct server launches remain
opt-in. A deliberately unconfigured library can still report `outputMount: null`.

Model downloads use folders shared by all projects. Managed first serve creates
`~/.dinkster/model-settings.json` (`%USERPROFILE%\.dinkster\model-settings.json`
on Windows), or the same filename under `DINKSTER_HOME` when set. Its default
location is `~/Dinkster/Models`, with ordinary ComfyUI subfolders including
`checkpoints`, `loras`, `vae`, `text_encoders`, `diffusion_models`, `controlnet`,
`upscale_models`, `embeddings`, and `clip_vision`. An existing `clip` folder is
also recognized; new text encoders prefer `text_encoders`. The shared folders
are visible in model pickers and do not live inside an environment library.

The model settings file contains no credentials. To choose another download
root or share an existing ComfyUI models tree, stop the engine, edit the file,
and restart. Paths must be absolute. For example:

```json
{
  "version": 1,
  "downloadLocation": "downloads",
  "locations": [
    { "id": "downloads", "path": "/home/name/Models", "mode": "readwrite" },
    { "id": "comfy", "path": "/home/name/ComfyUI/models", "mode": "read" }
  ]
}
```

On Windows use paths such as `C:/Users/name/Models`. `downloadLocation` must
name a writable location; other locations may be read-only or writable. Each
location is a root containing the ComfyUI category subfolders, not an individual
checkpoint folder. The default writable root is preferred for downloads.
Existing files are not moved or renamed. Files with the same name but different
content are preserved; a new download receives a digest suffix on collision.
To find or delete a downloaded model, open its displayed download folder in
your file manager. Stop work that uses it before deleting it; Dinkster can ask
to acquire it again when a workflow references it. A stock ComfyUI installation
can use the same root without copying files or changing provider filenames.

The library keeps per-project output and mount configuration. The
managed engine initializes `<environment-root>/dinkster-library/mounts.toml`
on first serve. With the default environment this is
`~/.dinkster/project/dinkster-library/mounts.toml` on Linux and macOS and
`%USERPROFILE%\.dinkster\project\dinkster-library\mounts.toml` on Windows.
For a single model directory rather than a shared category tree, keep a
project-specific read-only grant with an absolute path:

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
