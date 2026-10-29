# Environments and updates

How Dinkster ships Python environments, updates engine code without rebuilding
them, keeps packs with extra dependencies from breaking the engine, and lets a
user roll back. Dinkster owns all of it; Dinkster Desktop is a client that
adds a Windows installer, code signing, the Electron shell updater, and window
management. Nothing described here is Desktop-only: `dinkster-env` gives a
user without Desktop the same generations, channels, and rollback from the
command line. Developers who change Dinkster itself work from a checkout
([install.md](install.md)) and use none of this.

## Vocabulary

| Term | Meaning | Where it lives |
|---|---|---|
| Engine release | Dinkster at one commit: the pure-Python wheels for every workspace package, the `dinkster-frontend` wheel, any third-party wheel the lock needs that the base does not carry, `constraints.txt`, `SHA256SUMS` | GitHub release of `Kosinkadink/Dinkster`, mirrored to R2 |
| Lock | Dinkster's `uv.lock` at that commit, the single resolution every wheel hash comes from; `constraints.txt` is its export | `Kosinkadink/Dinkster` |
| Variant | One hardware target: `win-nvidia`, `win-nvidia-cu126`, `win-cpu`, `linux-cpu` | a row in `Dinkster-Environments` |
| Base bundle | A python-build-standalone interpreter plus the heavy locked wheels for one variant (torch, torchvision, numpy, the parts of the lock that change rarely), archived once and named by content hash, described by a base manifest | built by `Dinkster-Environments`, downloaded to the wheel store |
| Wheel store | Local directory of downloaded wheels plus the uv cache, shared by every project on the same volume; the only package source for engine installs | `<data>/wheels/`, `<data>/uv-cache/` |
| Generation | One base bundle plus one engine release synced into its own directory from the wheel store; immutable once built | `<root>/generations/<n>/` |
| Pack venv | A small venv over a generation's interpreter holding one pack's extra dependencies | `<root>/generations/<n>/packs/<pack>/` |
| Channel | A small document naming the current engine release and base bundle by hash | `channels/<name>.json` on the mirror |
| Mirror | A host serving channel files, base bundles and engine releases: the GitHub releases, R2, or another host in the client's configured list | client configuration |
| Project | An install root: the generations, packs, library and journal for one installation, bound to one channel; a project may point at a remote engine and have no local generation | `<root>` |
| Engine | A running Dinkster server process started from a generation (or reached over the network), with the GPUs assigned to it; a project binds one or more | supervisor |
| Supervisor | The `dinkster-supervisor` process that owns the public port, starts an engine on an internal port, and proxies to it once healthy; the station form manages several installations | `packages/dinkster-supervisor` |
| workerProtocol | The version of the worker boundary protocol an engine release's `dinkster-workers` speaks; peers that ship separately compare it in the hello exchange and refuse a mismatch | `dinkster_workers.boundary.PROTOCOL_VERSION` |
| Registry | The pack registry service a project trusts for browsing, resolving and downloading packs | `dinkster-registry` |
| `dinkster-env` | The command that installs, updates, rolls back, moves, and checks environments; Desktop calls it | `Kosinkadink/Dinkster` |

The base bundle and the engine release change at very different rates. A bug
fix produces a new engine release of a few tens of megabytes; the base bundle
changes when torch or another heavy pin moves. That split is the whole reason
updates are cheap.

## Layout

```
<data>/                     Dinkster data directory, one per volume that holds projects
  wheels/                   downloaded wheels, content-named
  uv-cache/                 UV_CACHE_DIR; uv unpacks each wheel here exactly once
  bases/<id>/               unpacked base interpreter trees

<root>/                     one project
  store                     path of the <data> directory this project uses
  generations/
    1/                      interpreter + site-packages
      generation.json       base id, engine release, constraints hash, variant, workerProtocol
      packs/<pack>/         pack venvs layered over this generation
    2/
  current                   file naming the active generation number
  journal/                  update operation journal, retained on failure
  library/                  models and mounts, never touched by updates
```

The wheel store is shared: every project on a volume records the same
`<data>` directory in its `store` file, so a second project costs no second
base. A project on another volume uses a store on that volume, because the
links that make generations cheap only work within one volume.

Generations are synced with `uv pip sync --offline --no-index --find-links
<data>/wheels` from the engine release's `constraints.txt`. uv links every
file from `uv-cache` into the generation (hardlinks on Windows, copy-on-write
clones on macOS and Linux; "link" below means whichever applies), so a
generation that differs from the previous one by a few wheels costs a few
megabytes and builds in well under a second. The interpreter is linked the
same way from `bases/<id>/`, so every generation holds a complete
interpreter tree and deleting any other generation cannot break it.

The engine environment is immutable and hash-checked. `dinkster-env doctor`
compares the active generation against its constraints and reports drift.
Repair re-syncs from the wheel store without network access.

### Project records

`current` is the active positive decimal generation number followed by one
newline. `store` is the absolute data-directory path followed by one newline.
Each generation holds its complete interpreter under `python/`, including
`python/python.exe` on Windows or `python/bin/python3` on Linux, and retains
the engine release's `constraints.txt` beside `generation.json`.

`generation.json` uses `format: dinkster.generation/1`. Its fields are:

- `generation`: positive integer matching the directory name.
- `base`: `sha256:` content identifier of the base manifest.
- `engine`: object containing `version` and the `sha256:` release identifier.
- `constraints`: `sha256:` digest of the retained constraints file.
- `variant`: portable variant name; `workerProtocol`: positive integer.
- `inProcessPacks`: optional array of pack ids, empty when omitted. These packs
  have locks under `packs/<pack>/pack-lock.json` but no venv. Their dependencies
  are staged into a fresh generation before its immutable file inventory.
- `files`: object mapping portable generation-relative paths to `sha256:`
  file digests. It covers the interpreter, installed packages and constraints,
  not `generation.json` itself or the independently checked pack venvs.
  Runtime bytecode not present in this inventory is ignored by drift checks.

Writers flush pointer and journal contents before atomic replacement. JSON
objects are UTF-8, ASCII-escaped, sorted by key and newline-terminated. These
records are shared by the command-line client, Desktop and pack provisioning;
none of those clients defines a second generation format.

The per-user `dinkster/stores.json` lives below `LOCALAPPDATA` on Windows or
`XDG_DATA_HOME` (default `~/.local/share`) on Linux. Its format is
`dinkster.stores/1`, with `volumes` mapping decimal platform `st_dev` identities
to absolute data-directory paths. An unregistered volume defaults to
`dinkster-data` beside the project. An explicit conflicting store, another
volume or overlapping project/store paths is refused. A store's `projects.json`
uses `dinkster.store/1` and a sorted `projects` array of absolute project roots.
These references are atomically updated under registry/store/project OS locks
after confirmation and journal publication. Failed installs may retain references
for recovery and safe garbage collection; they do not authorize data deletion.

## Channels

`channels/<name>.json` on the mirror:

```json
{
  "format": "dinkster.channel/1",
  "engine": { "version": "0.3.1", "release": "sha256:..." },
  "base": { "win-nvidia": "sha256:...", "win-cpu": "sha256:..." },
  "minimumDesktopVersion": "0.2.0",
  "signature": null
}
```

The channel names artifacts by content hash, never by "latest GitHub release".
A channel update is one small file write, so a release that was built earlier
and kept unpublished becomes live in seconds.

`stable` is written when a `vX.Y.Z` tag's release workflow finishes. `main`
is written when a push to `main` finishes the same workflow without a
version: the engine release it names is a prerelease of the next version
identified by commit. Both channels are served by the same client code path
and differ only in the channel file. Neither resolves on the user's machine:
every wheel an engine install needs is in the base bundle or the engine
release, and when a commit adds a dependency the base does not carry, that
commit's engine release includes the wheel until the next base picks it up.
The base is shared between the channels because it rebuilds only when its
part of the lock changes.

`main` exists so that people who want the newest code get it through a
supported path, and so that a bug reported from `main` reproduces on
`stable` once the tag lands. There is no nightly channel; `main` makes it
redundant.

`minimumDesktopVersion` lets a channel refuse a Desktop shell too old to
drive the engine it names; the shell updates itself first.

`signature` is reserved. Mirrors Dinkster operates serve documents with a
null signature, checked only by the sha256 digests they carry and by the
TLS origin. A mirror Dinkster does not operate may serve channel and base
manifest documents only when `Dinkster-Environments` signs them with an
ed25519 key whose public half ships in the client; a client configured for
such a mirror refuses an unsigned or wrongly signed document.

## Base bundles and the environments repository

`Dinkster-Environments` holds one definition per variant: the python
version, the python-build-standalone release, the torch index and CUDA tag,
and the list of lock entries that belong to the base rather than the engine.
Its build workflow:

1. Reads the lock from the Dinkster commit it is building for.
2. Downloads the base wheels for the variant from their indexes into a
   staging directory and verifies them against the lock hashes.
3. Writes the base manifest: `format: dinkster.base/1`, variant, python
   version, every wheel with its filename, size, sha256 and source index,
   the `workerProtocol` the engine wheels in this lock declare, the minimum
   NVIDIA driver and compute capability for the CUDA tag, and
   `signature: null`.
4. Archives the interpreter and wheels as split 7z parts under 2 GB each so
   GitHub releases can host them, with the base manifest listing part hashes.
5. Publishes to a GitHub release of `Dinkster-Environments` and, when the R2
   secrets are present, to `releases.dinkster.org/base/<variant>/<id>/`.

The base is rebuilt only when its part of the lock changes. The workflow
compares the base-relevant lock entries with the newest published base
manifest and skips the build when they match, so a tag that only touches
Dinkster code produces no new base.

A per-variant extras index lives next to the base: wheels such as attention
kernels that are compiled against one exact torch, CUDA and Python triple
and are not on PyPI for that triple. A pack that needs a compiled wheel the
extras index does not carry takes the own-environment path described below.

Every variant shares one set of base pins: Python 3.14, torch 2.14.1 with
the matching torchvision, numpy 2.5, `opencv-contrib-python-headless` 5.0.
Python 3.14 receives bug fixes until late 2027 and security fixes until
2030; every dependency of the engine and of the popular ComfyUI packs
ships a cp314 or abi3 Windows wheel; `comfy-kitchen` and `triton-windows`
are abi3 against 3.12 and run unchanged. The pins move together, by one
change to the lock, and a change that moves them produces a new base. The
one exception is `win-nvidia-cu126`: torch 2.14.1 is the last release with
cu126 wheels, so that variant keeps torch 2.14.1 and the matching
torchvision when the others move, and receives only the pins that still
build against it.

Supported variants: `win-nvidia` (cu132), `win-nvidia-cu126`, `win-cpu` and
`linux-cpu`. The CPU variants run everything, an order of magnitude slower
than the slowest GPU; they serve users who run the engine locally with
remote workers, and CI and development machines without a GPU. Adding a
variant is a new row, not new machinery: `linux-nvidia` is the same two
rows on Linux wheels, `win-rocm` is torch from AMD's
`stable.repo.amd.com/rocm/whl-next/` index with a per-architecture extra,
and `win-xpu` is torch from the `xpu` index.

### Choosing the NVIDIA variant

cu132 wheels run on Turing (RTX 20 series, compute capability 7.5) and
newer and require driver 580 or newer on both Windows and Linux, the floor
for any CUDA 13 runtime. cu126 wheels run on Maxwell through Ada and
Hopper (GTX 900 through RTX 40 series) on drivers 525 and newer and do not
support RTX 50 series. torch 2.14.1 is the last release with cu126 wheels,
so `win-nvidia-cu126` stays on it when `win-nvidia` moves on; the R580
driver branch is NVIDIA's last to support GPUs older than Turing and is
supported until mid-2028, so a machine on a current driver can run either
variant.

Detection runs `nvidia-smi --query-gpu=index,name,driver_version,compute_cap,memory.total --format=csv,noheader`
and, on Windows when `nvidia-smi` is absent, reads `Win32_VideoController`
to tell an NVIDIA GPU without a driver from no NVIDIA GPU. Selection looks
at the driver first and then at the GPUs the user assigned to the engine,
applying the rule to the lowest compute capability among them, not to every
GPU in the machine: a system with an old driver kept for a GTX 10 series
display card and an RTX card for compute gets cu126 for the RTX card
without a refusal, and the same system after a driver update is offered
cu132. The choice is shown at plan time with the reason and the user can
override it. When no NVIDIA variant fits the assigned GPUs (driver too old
for cu126, or no NVIDIA GPU at all), the plan says what driver or hardware
would be needed and offers the CPU variant as an explicit choice; the client
never falls back to CPU silently. Each base manifest carries the minimum
driver and the minimum compute capability its torch build supports, so the
rule is data in the base manifest rather than code in the client.

## Updating

```diagram
tag v0.3.1 -> release workflow builds engine wheels -> channels/stable.json
              names engine 0.3.1 + base win-nvidia@<id>

client (dinkster-env or Desktop on behalf of the user):
  fetch channel ............................... 1 KB
  base <id> already in wheels/? yes
  download engine wheels not in wheels/ ....... tens of MB, verified by sha256
  sync generations/N+1 from constraints.txt ... links for everything unchanged
  start supervisor for N+1 on a new port, wait for /api/health
  write current = N+1, stop N's supervisor
  keep N on disk for rollback
```

When the channel names a base the wheel store does not have, the client
downloads it first and shows the size before starting. Downloads are ranged
and resumable, hashed while streaming, and retried across the configured
mirror list.

Rollback is `dinkster-env rollback`: start the previous generation's
supervisor, repoint `current`, stop the newer one. Nothing is rebuilt. The
newer generation stays on disk until garbage collection, which keeps the
active and previous generation of each project, and keeps every wheel and
base in the store that any retained generation references.

Moving a project is `dinkster-env move <new-root>`. Within a volume it moves
the root, rewrites the `home` line of every pack venv's `pyvenv.cfg`, and
runs `doctor`. Across volumes it creates a store on the destination volume
if none exists, re-syncs each retained generation there from the moved
wheels, and rebuilds the pack venvs, so the link sharing holds on the new
volume. A store that no project references any more is deleted by garbage
collection.

An engine swap while a job is running is refused with a message to finish or
cancel the job.

### Journal and failure handling

Every update writes `journal/<timestamp>.json` before it changes anything:
the operation, the previous and target generation, and the stage reached.
If sync fails, the previous generation keeps serving and the target is
deleted. If the target starts but never reports health, the client stops it,
restarts the previous generation, and keeps the journal for diagnosis. A
client that finds a journal with an incomplete operation at startup finishes
the rollback before doing anything else. The journal is the only record of
update outcomes; Dinkster sends no telemetry, and a user reporting a problem
can export the journal directory.

Journal records use `format: dinkster.journal/1`, `operation` (`setup`,
`update`, `rollback`, `move`, `repair` or `gc`), `previous` (positive integer
or null), `target` (positive integer), and `stage`. Stages are `planned`,
`building`, `starting`, `switching`, `complete` and `rolled-back`. Only
`complete` and `rolled-back` are terminal. A failed operation may also carry
an `error` string, retained after rollback. A malformed record is an error,
not permission to ignore a possibly interrupted operation.

Activation records also retain `bindings`, an array of objects with `name`,
`port`, `gpus` (device-id strings), `running` (boolean) and `targetPort`.
These preserve previous assignments and candidate ports before construction
or startup. Stopped bindings keep `targetPort` equal to `port`; only running
bindings use new candidate ports, excluding the retained stopped assignments.
Private process records live at
`runtime/<generation>/<name>.json`; old and new same-name bindings can coexist
without overwriting either process identity. Status follows `current`, but
mutation safety checks inspect all recorded engines. Recovery restores the
previous pointer and running assignments, stops only authenticated candidate
processes and discards newly built failed targets. A retained rollback target
is never discarded. Process identity ambiguity leaves the journal incomplete.

Initial setup records `previous: null` and the requested initial binding before
construction. It uses the same health-before-pointer activation path. Recovery
stops the candidate, removes any published target pointer and incomplete target,
and retains a rolled-back journal without selecting a nonexistent predecessor.
Every managed engine receives its explicit absolute `--generation` path, so
candidate startup and rollback do not follow the current pointer for sources.

Per-instance process receipts are private runtime state under
`runtime/<generation>/<name>.process/<sha256 of instance>/state.json`, containing
the instance, lifecycle state and supervisor PID. A launch guard holds the
directory's OS lock through child cleanup. Recovery may cancel a still-pending
launch under that lock; a delayed supervisor then exits without an engine.
The supervisor writes a stopped receipt only after cleanup, so interruption
of the updater before its own record write cannot erase shutdown evidence.
A dead supervisor without that receipt remains ambiguous. No process receipt
or shutdown credential is added to the exported operation journal.

### Shipping code before it is public

For a release that must exist before an announcement, the engine wheels are
built from the embargoed commit in a private repository, uploaded to the
mirror under their content hashes, and left unreferenced by any channel. At
launch the commits are pushed to the public repository and `stable` is
rewritten to name the already-uploaded artifacts; clients see it within
seconds. The public tag and its GitHub release follow and carry the same
wheels.

## Packs and their dependencies

Externally supplied packs run isolated by default ([pack-authoring.md](pack-authoring.md)). A pack
declares its dependencies in its pack manifest and ships no wheels. At
publish time the publisher, or the registry on the publisher's behalf,
resolves those dependencies against each supported variant's base and
records a pack lock per variant with wheel hashes, so an install on a user's
machine fetches exactly those wheels from their indexes by hash and never
resolves. The no-resolution rule for engine installs applies here too: a
pack install fetches only hashed wheels named by a pack lock. A registry
may mirror those wheels so an install depends on no third-party index.

Some distributions install the same module under different names, and
installing two of them corrupts the module. The pack lock step maps those
names onto the one the base carries and applies the pack's version
constraint to it. The base carries `opencv-contrib-python-headless`, a
strict superset of `opencv-python`, `opencv-python-headless` and
`opencv-contrib-python` for a server process (the GUI functions it drops
cannot run without a display), so a pack naming any of the four, with or
without the `[ffmpeg]` extra, is satisfied by it. The alias table is data in
the engine.

A pack that declares dependencies gets a venv created with `uv venv --python
<generation interpreter> --system-site-packages`, so everything in the
generation is importable and nothing in it is copied.

uv does not consult the parent interpreter's site-packages when installing,
so a naive install into that venv would download a second torch. Provisioning
therefore runs in two steps:

1. Take the published pack lock for the generation's variant, or resolve
   the pack's requirements with the generation's `constraints.txt` as
   `--constraints` when there is none (a local directory or git install). A
   pack lock that disagrees with the generation on a shared package, or a
   pack that needs a version the generation cannot satisfy (torch 2.15
   against a 2.14.1 base, a different Python), fails here with the conflict
   named, and the plan offers the own-environment path instead.
2. Install the locked set minus every package the generation already
   provides, from the wheel store first and the pack's declared indexes
   second, each wheel verified against its hash. Wheels shared between pack
   venvs are linked from the uv cache. The pack lock used is recorded with
   the install.

A pack that cannot layer takes its own environment. That is the case when
its pack lock conflicts with the generation on a shared package, when it
needs a compiled wheel that neither PyPI nor the extras index carries for
the generation's triple. An own environment is a full base bundle of the
variant it needs, shown with its size at plan time, and its own
`dinkster-workers` copy, which then negotiates `workerProtocol` in the hello
exchange like a remote worker instead of assuming it shipped with the
engine.

In-process packs ([in-process-hosting-design.md](in-process-hosting-design.md)) have no venv; their dependencies
must layer cleanly into the generation and installing one produces a new
generation.

Pack venvs belong to the generation they were created over. A new generation
rebuilds them from their locks; a pack whose compiled wheels do not exist for
the new base reports that in the plan before the swap.

"Open a terminal" opens into a pack venv or a scratch venv created over the
generation, never into the generation itself. Drift inside a venv is reported
by `doctor` with a rebuild-from-lock offer. Hand edits to the generation are
detected the same way and repaired from the wheel store.

## What a workflow needs

A workflow document already names everything it depends on: every node type
carries the pack id the host attributed to it at schema collection, and
every model reference is a `blake3:` digest with optional resolver leads.
A workflow's requirements are therefore derived from the document, never
hand-written into it. Opening a workflow computes the set of packs (with
the version the workflow was last saved against, recorded by the frontend
on save) and the set of asset digests, and the missing ones become one
plan: install these packs, materialize these assets, from the registries
and resolvers the project trusts. The plan is printed first and runs only on
confirmation, like every other mutating command.

Requirements are hardware-agnostic because packs are. A pack is one id and
one version on every variant; the per-variant pack lock decides which
wheels that means, so the same workflow resolves on `win-nvidia`,
`linux-cpu` or a remote engine without the document saying anything about
hardware. The only hardware a workflow can require is what a pack declares
in its pack manifest (`[pack.sandbox] gpu`, the node's `devices` list): a pack that
exists only for CUDA fails the plan on a CPU variant with that reason, and
a pack with a compiled wheel for one torch triple and not another fails
the same way. Nothing else in the workflow is variant-specific. A sharing
export can include the derived requirement list next to the document for
tools that cannot compose, but the document stays the source of truth.

A project carries the same thing one level up: the packs it has installed
and the assets it has materialized are the union of what its workflows
needed plus what the user added. Switching a project to another variant
replans the pack set against that variant's base; the project's workflows
do not change.

## Frontend

Dinkster pins one `Dinkster-Frontend` commit in `scripts/release_sources.json`
and the release workflow packages that commit as the `dinkster-frontend`
wheel, so the frontend version is pinned by the generation and nothing
else. `Dinkster-Frontend` publishes its built bundle per commit as a CI
artifact named by hash, and the release workflow downloads that bundle
instead of running node; bumping the frontend is a one-line change to the
pin. Desktop serves whatever bundle the active generation installed; during
development the engine serves a sibling checkout's build (`discover_frontend_bundle`).
Pack frontend extension modules are served by the engine from the pack's
own files and are unaffected.

## Publishing the engine

The tag workflow uploads every workspace wheel and the frontend wheel to
PyPI with trusted publishing, so `pip install dinkster` and the `stable`
channel install the same artifacts: the channel fetches them by hash from
the release, pip resolves the same versions from the index. Every
dependency that comes from another Dinkster repository (`dinkster-inference`
for `dinkster-native`, `dinkster-token-verifier` for `dinkster-server`) is a
versioned PyPI release declared like any other dependency, never an archive
or release URL, because PyPI rejects URL dependencies and the setup scripts
must not be the only way to obtain one. torch is declared without an index:
a pip user chooses the index for their hardware the way ComfyUI users do, or
runs `dinkster-env setup`, which applies the variant rule above and installs
the base for the machine.

## What Desktop adds

- The NSIS installer and Windows code signing; the macOS app bundle, Developer
  ID signing and notarization. The signing workflows run on `vX.Y.Z` tag
  push and their signing jobs target a GitHub environment that holds the
  signing credentials and requires the repository owner's approval, so an
  agent can tag, build and test, and only the owner's approval produces a
  signed installer. A run dispatched by hand targets the same environment
  and waits for the same approval.
- The Electron shell updater: checks on startup only, installs on the next
  startup rather than at quit, records each attempt in a sidecar marker and
  stops retrying the same version after three failures, and keeps the previous
  installer in the feed so a shell downgrade is possible.
- Projects: a list of install roots with their channels, including a project
  that is only a remote server URL and has no local generation. Most users
  have one project and treat it as their install. Switching a project to
  another engine release or variant is a selection, not a reinstall: the
  wheel store is shared, so the new generation syncs from wheels already on
  the volume and only missing wheels are downloaded. A project may bind more
  than one engine, each on its own GPU or remote, so that several workflows
  run at once.
- Windows and the tray, including the single top bar: `titleBarStyle:
  'hidden'` with the window-controls overlay on Windows and Linux and inset
  traffic lights on macOS, while the frontend reserves the overlay area and
  marks its bar as the drag region.
- Hardware checks before a download, applying the variant rule above to the
  GPUs the user assigned to the engine, with the choice and the reason shown
  and an override; when no NVIDIA variant fits, the dialog names what would
  be needed and offers CPU as a choice rather than falling back to it.

Everything else Desktop does for environments is a call into `dinkster-env`
or the supervisor station, and the same commands work without Desktop.

## Hosting

GitHub releases host every artifact first. R2 under `releases.dinkster.org`
mirrors them with no egress cost, and clients try mirrors in a configured
order. Base bundles are the only large objects and are kept for the ten
newest ids per variant; every base manifest is kept forever. A client that
needs a base whose archive is gone rebuilds it locally from the base
manifest: it fetches each listed wheel by hash from the source index the
manifest names and the interpreter from python-build-standalone, which is
the one case where an engine install reads a package index. Engine releases
are small and are kept indefinitely. A region-specific mirror is one more
entry in the client's mirror list and, when Dinkster does not operate it,
serves only signed documents.

## Decisions

Settled:

- Base pins: Python 3.14, torch 2.14.1, numpy 2.5,
  `opencv-contrib-python-headless` 5.0; `win-nvidia` on cu132;
  `win-nvidia-cu126` keeps torch 2.14.1 when the other variants move.
- Engine installs never resolve; they fetch artifacts named by hash on
  either channel. The one index read is rebuilding a base whose archive is
  no longer hosted, from its retained base manifest.
- Separate `Dinkster-Environments` repository; variants `win-nvidia`,
  `win-nvidia-cu126`, `win-cpu` and `linux-cpu`; a per-variant extras index
  for wheels compiled against one torch triple.
- NVIDIA variant chosen by driver, then by the lowest compute capability
  among the engine's assigned GPUs, detected with `nvidia-smi` and a WMI
  fallback, with an override; CPU is offered explicitly when no NVIDIA
  variant fits, never chosen silently.
- Packs declare dependencies and ship no wheels; a pack lock per variant is
  recorded at publish time; wheels are fetched by hash at install.
- The base carries `opencv-contrib-python-headless` and the pack lock step
  maps every opencv distribution name onto it.
- One wheel store and uv cache per volume, shared by every project on it;
  switching a project's engine release or variant is a selection.
- Generations are linked from the shared store, hold their own interpreter
  tree, and are immutable; rollback is reactivation; the previous
  generation is retained; an interrupted update found in the journal at
  startup is rolled back before anything else.
- `dinkster-env move` relocates a project within or across volumes and
  keeps the link sharing.
- Pack venvs layered with `--system-site-packages` and two-step constrained
  provisioning; rebuilt from their pack locks for each new generation; own
  environment when a pack conflicts with the base or needs a compiled wheel
  no index carries; installing an in-process pack makes a new generation.
- A terminal opens into a pack or scratch venv, never into a generation;
  `doctor` reports drift and repairs from the store.
- Frontend pinned by commit in Dinkster and packaged into the engine release
  from the per-commit bundle `Dinkster-Frontend` publishes.
- Channels `stable` and `main`, both prebuilt by the same workflow; no
  nightly; `minimumDesktopVersion` in the channel file.
- Engine wheels published to PyPI by the tag workflow; cross-repository
  dependencies are versioned PyPI releases.
- Signing workflows run on tag push behind an owner-approved environment;
  a hand-dispatched run needs the same approval.
- Electron shell updater checks on startup only, installs on the next
  startup, stops retrying a version after three failures, keeps the previous
  installer in the feed.
- Environment operations live in Dinkster; Desktop is a client.
- Engine swap refused while a job runs.
- No telemetry; local journal only.
- Embargoed releases: artifacts uploaded unreferenced, made live by one
  channel write.
- Hosting: GitHub releases first, R2 mirror; ten newest bases per variant
  retained, base manifests and engine releases retained indefinitely.
- Manifest signing is a reserved field, required for any mirror Dinkster
  does not operate.
- A registry is the trusted source for pack browsing, resolution and
  download; a registry may also mirror pack wheels.
- Workflow and project requirements are derived from the document (pack
  attribution and asset digests), never declared by hand, and are
  hardware-agnostic except where a pack manifest says otherwise.
