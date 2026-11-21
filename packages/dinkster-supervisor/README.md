# dinkster-supervisor

`dinkster-supervisor` is the layer-0 process that owns the public port, starts an
engine host on an internal port, and proxies HTTP and WebSocket traffic only
after `GET /api/health` succeeds. Its station process applies the same
version-independent machinery to multiple configured Dinkster installations.

## Engine endpoint admission

An engine manifest can declare `capabilities = ["bound-endpoint/1"]`. Managed
launches then pass `--engine-bound-endpoint` to the supervisor. Direct launches
can supply the same flag only when their engine supports the capability.
The supervisor passes `--port 0 --bound-endpoint PATH`; the engine holds its
listener and atomically writes `{"format":"dinkster.bound-endpoint/1","port":N}`
to that private path. Health probes use only the announced port and the requested
internal host. The normal startup deadline includes waiting for this announcement.

Engines without the declaration retain fixed `--host/--port` launch support.
Its bind-close-reuse allocation can lose the port to another process; the
supervisor logs that limitation. It does not retry allocation or engine startup.

## Setup

This package is a uv workspace member and is not published separately yet.
From the repository root:

```sh
uv sync --all-packages
uv run dinkster-installs --help
uv run dinkster-supervisor --help
uv run dinkster-station --help
```

All three console scripts are supplied directly by this package. The supervisor
defaults to `127.0.0.1:3639` (same default as a bare engine - the port a
frontend points at means the same thing either way); the station's
management port defaults to `127.0.0.1:3649`, and the station reads an
installs TOML file and gives each install its configured public port.
Dinkster deliberately stays out of the ComfyUI port neighborhood
(8188/8189/8199) so a Dinkster install never collides with a running
ComfyUI or comfy-runner allocation.

Contributor wheel acquisition uses the resolver's SHA-256 to verify each selected
wheel. When resolver metadata omits the size, the downloader measures the verified
file rather than issuing a separate HEAD request. An already verified retained
wheel needs no network request. Explicit sizes still enforce byte counts, and
unknown-size interrupted HTTP downloads restart without guessing a resume range.

## Signed release channels

Stable and Nightly selections require an independently pinned Ed25519 signature
before use, including local copies and offline channel caches. Third-party HTTPS
mirrors do not supply their own trust roots. Artifact hashes and sizes remain
mandatory. See [channel signing](../../docs/channel-signing.md) for key
provisioning, rotation, lost/leaked-key response and offline trust limits.

## Signed security feed refresh

The supervisor-owned `dinkster_supervisor.security.host.SecurityHost` reads independently provisioned
trust from `<root>/security.sqlite3`. Refresh requests the complete signed
`/v1/security/blocks` document from each trusted issuer. A local
`<root>/security.toml` can select a mirror without changing the issuing identity:

```toml
refuse_vulnerable = false
request_timeout_seconds = 5.0
refresh_interval_seconds = 60.0
# maximum_feed_age_seconds = 86400

[feed_urls]
"https://registry.example.org" = "https://mirror.example.org/security/index.json"
```

The API and supervisor wheels also expose the same source as `dinkster_security`.
Overrides require independently provisioned trust for the exact issuer. Feed
responses cannot provision keys. Refresh failures retain the accepted checkpoint
and decisions; downloads are limited to 32 MiB and do not send registry bearer
credentials. Default policy has no feed-age refusal. An explicit maximum age
uses signed `issuedAt`, not the time of the latest successful download. Explicit
vulnerability refusal applies to retained releases too. Local policy reloads
at host creation and on refresh; malformed policy refuses rather than silently
reverting to defaults.
Installer acquisition, install, restore and serving selection refresh before
their policy check. Managed snapshot pack selection also refreshes its environment
root. Serving uses the selected snapshot's environment root, otherwise the install
root, otherwise the library root, when security trust or configuration exists
at launch. The serving host owns a periodic refresh task and cancels it during
cleanup. It checks release identities before catalog preparation, worker import,
lazy activation and dispatch, including previously pinned routes. A new block
revokes execution before closing isolated workers or shared process owners;
affected in-process execution requests whole-engine shutdown. A lift never
resumes a revoked session. `GET /api/security` uses `SecurityHost.status` to
report affected release digests, issuing registries, full exact-notice details,
decisions, last verification times, configured age refusals and refresh failures.
Feed observations distinguish cached, expired, not-received and not-provisioned
issuers. Recovery reports policy allowance, not whether a worker is running, and
marks repairs as not reviewed until a compatible plan is requested. Status does
not refresh, acquire payloads or change software. Flagged releases are logged
once per notice revision and answer during a host lifetime. Runtime work inventory
names registered or revoked pack workers, not whether an invocation is running.
The runtime route also reports scoped queued and retained jobs, submitted workflows
and nested node types; these identify submitted work, not incident impact.

The public environment CLI provides `security --json list` and
`security --json details --pack PACK`, using the current snapshot or an explicit
retained `--snapshot N` (`--generation N` remains accepted). The selected engine's
`dinkster.manager security-status --generation PATH [--pack PACK]` reads sealed
source/dependency provenance without importing packs or refreshing feeds. It
returns full shared-host findings with snapshot/software identity, including
blocked releases. The tools revalidate that identity and the current selection
before returning. Unknown packs, drift and linked sealed inputs refuse. Offline
job/worker inventory is unavailable, not an assertion that no work exists. No
payload or snapshot changes occur.
Omit `--json` for labeled notice, decision, verification, policy and recovery
output. Repair previews print exact release and dependency locks, retained
approval digests and compatible-fix failure reasons. Workflow impact remains
explicitly unavailable offline. Malicious or affected incident advice explains
that blocking is not cleanup and recommends isolation, trusted-machine credential
rotation, exposure assessment and a rebuild from trusted inputs.

The public environment CLI accepts `security --json approve` or `disapprove`
with `--registry`, `--notice` and `--revision`. The exact answer persists and may
be changed; stale revisions and malicious notices cannot be approved. These
commands do not start or resume execution. `security --json import --registry
ISSUER --file PATH` verifies a signed feed through the same trusted store before
replacing its checkpoint; rejected files leave accepted evidence and answers intact.
`security --json policy` reads the current local policy without a network request
or creating a configuration file. Explicit `--refuse-vulnerable` or
`--no-refuse-vulnerable` and `--maximum-feed-age-seconds N` or
`--no-maximum-feed-age` persist those choices while preserving mirrors and refresh
timing. Invalid settings leave the previous file intact. These changes do not
install software or resume work; serving hosts reload policy on refresh.

## Ingress

An optional `[ingress]` table adds a fleet-facing port:

```toml
[ingress]
port = 8300
members = ["main", "experiments"]
primary = "main"
state_path = "/var/lib/dinkster/ingress.sqlite3"
```

The ingress persists non-expiring `(scope, clientId, jobId)` and `jobRef`
ownership in SQLite. Slice 2a sends all submissions to `primary`, routes job
reads through those assignments, serves honest all-member queue/job
aggregates, and merges member event WebSockets without rewriting frames.
Catalog and machine-local state surfaces use the primary until fleet cohort
and shared-library proofs permit execution spreading.

The ingress route matrix is explicit:

- `POST /api/jobs` claims `(default, clientId, jobId)` before forwarding to
  the owner and records `jobRef` before returning the engine's 202.
- Keyed status, cancel, and `/api/values` use the key assignment. By-ref
  status, cancel, event replay, and history detail use the jobRef assignment.
- `GET /api/jobs`, `/api/queue`, and `/api/history` merge every member and
  return 503 rather than a partial answer when any member is unanswerable.
  Member requests run concurrently under a five-second per-member deadline.
- `/api/health` and `/supervisor/*` are ingress-owned. The latter reports
  `role: "ingress"` and never reaches an engine.
- Catalog reads, queue mutations, assets, library, mounts, settings, memory,
  and cache routes go to the primary. Compatibility prompt routes are not
  exposed by the ingress.
- `/api/events` merges one reconnecting upstream WebSocket per member through
  a 64-frame/eight-MiB queue and one heartbeat-enabled downstream writer.
  Upstream frames are bounded at four MiB; oversize closes the downstream
  socket with code 1009. Primary frames pass verbatim; secondary frames pass
  only when their JSON text or binary header carries a `jobRef`. Reconnects
  use jittered exponential backoff and reset only after five stable seconds.
- Primary HTTP routes stream request and response bodies without buffering;
  the ingress accepts the engine's 16 MiB asset-upload limit. `POST /api/jobs`
  alone buffers up to four MiB because its accepted response must be validated
  and persisted before the ingress returns it.

The store defaults to `ingress.sqlite3` beside `installs.toml`. Assignments do
not expire or evict. Ambiguous engine failures retain ownership; only a
definitive owner 404 permits a conditional owner-matched deletion.

## Use

Start one supervised engine by supplying the engine command after the
supervisor options:

```sh
uv run dinkster-supervisor -- dinkster-serve
```

The single-engine management surface is:

- `GET /supervisor/status`
- `POST /supervisor/engine/restart`
- all non-reserved routes proxied when healthy, or answered with 503 while unavailable

The station management port adds:

- `GET /supervisor/installs`
- `POST /supervisor/installs/{name}/start`
- `POST /supervisor/installs/{name}/stop`
- `POST /supervisor/installs/{name}/restart`

`InstallDef`, `load_installs`, `create_supervisor_app`, and
`create_station_app` are the main Python composition exports. The supervisor
and station deliberately import no engine code.

Supervisor, station and ingress listeners validate exactly one inbound HTTP
Host before routing, CORS preflights or WebSocket upgrades. Loopback IP
literals and the configured `--host` are accepted; other hostnames require
repeatable `--allow-host HOST`. Unlisted, malformed, missing or duplicate
Host headers receive `421` with `error: "host-not-allowed"` before proxying.
Python app factories take the same `host` and `allow_hosts` inputs.

This boundary prevents DNS rebinding even when the browser legitimately
sends `Sec-Fetch-Site: same-origin` for its original hostname. The upstream
Host is still rewritten for the internal engine connection; no forwarded
Host header grants authority. Engine Host and Origin checks are unchanged.

## Design contract

The supervisor owns the public port and almost nothing else: it binds in
milliseconds, spawns the engine host behind it, narrates startup at
`GET /supervisor/status`, answers other routes 503 with a pointer body
until `GET /api/health` succeeds, then transparently proxies HTTP and the
events WebSocket. It never imports engine code - subprocess + HTTP only -
so the layer that manages installations shares no venv with any engine
version. The `/supervisor/` prefix is reserved and never falls through to
the engine proxy. CORS is owned by whoever owns the public port: opt-in
per origin via repeatable `--allow-origin`, default none, with upstream
Access-Control-* headers stripped so a child cannot double-stamp. The
station applies the same version-independent machinery to multiple
installs from a TOML file. Focused coverage is in
`tests/test_supervisor.py`, `tests/test_station.py`, and
`tests/test_installs.py`.

## Environment records

The environment state module owns the environment records consumed by environment
management and Desktop. Full layout: [environments and updates](../../docs/environments-and-updates.md).

- `<root>/current`: positive decimal snapshot number and newline.
- `<root>/store`: absolute shared data-directory path and newline.
- `<root>/environment.json`: saved environment settings.
- `<root>/builds/<n>/python/`: complete interpreter tree; executable is
  `python.exe` on Windows, `bin/python3` on Linux.
- `<root>/builds/<n>/constraints.txt`: the release's hashed requirements.
- `build.json`: `format: dinkster.generation/1`, `generation` integer,
  `base` digest, `engine: {version, release}`, `constraints` digest, `variant`,
  `workerProtocol` integer, and `files` (relative path -> `sha256:` digest).
  The inventory covers owned interpreter/package files and constraints, not
  itself, runtime bytecode or pack venvs. Drift reports name missing, modified
  and unexpected files, even when installed distribution versions agree.
  `inProcessPacks` is an optional array of pack ids, empty by default; these
  packs retain their dependency locks but have no pack venv. Inventory is
  finalized only after their dependencies are staged into a fresh snapshot.
- `<root>/journal/<timestamp>.json`: `format: dinkster.journal/1`, `operation`,
  `previous` integer or null, `target` integer, `stage`, and optional `error`.
  Operations are setup/update/rollback/restore/move/repair/gc. Stages are planned,
  building, starting, switching, complete and rolled-back; the last two are
  terminal. Interrupted records must be recovered before another operation.

A stopped legacy environment migrates once: settings move from `project.json`
to `environment.json`, software moves from `generations` to `builds`, and
`build.json` becomes authoritative. Existing history and metadata bytes do not
change. An immutable engine without `build-metadata/1` receives a byte-identical
`generation.json` compatibility copy, including when restored from history.
The settings-free `project.json` refusal marker and
`journal/update-dinkster-supervisor-to-0.0.13.json` sentinel are published after
the new layout. The sentinel is a JSON string, not a journal: released tools
reject it before dispatching management commands and report its update
instruction in the filename. Current tools ignore only its exact bytes and
refuse a changed or linked sentinel. Neither marker contains settings or history.

`dinkster-env --root ROOT library-list projects --json` reads one local project
page without starting an engine or migrating the environment. Use `workflows`
or `assets` with `--project ID` for that project's saved workflows or recorded
files. `--limit` accepts 1..200 (default 50); `--cursor` continues the returned
page. JSON results include `library.kind`, the helper's `items`, `records` or
`assets` array, and a nullable `cursor`. Queries use the tools' helper and local
scope; each command closes and reaps its helper before exiting. No stored file
bytes, unrecorded folder inventory or preview is returned. An absent library
returns an empty page without creating it; an unreadable library fails with one
plain by-the-way sentence. `DINKSTER_LIBRARY_ROOT` selects the same shared
library as engine launches; otherwise it is ROOT/dinkster-library. A fork's
`environment.json` saves an absolute `libraryRoot` binding; library queries,
compatibility checks and engine launches use it and reject a conflicting
`DINKSTER_LIBRARY_ROOT` override.

`dinkster-env --root ROOT project-get ID --json` returns the authoritative
`project` record and a nullable candidate `environment: {id, snapshot}`. It
refuses unknown or deleted projects and unavailable libraries.
`project-check ID` additionally checks the existing managed identity, active
snapshot, installed helper admission and engine library readers before a caller
stops any running owner. A project already bound elsewhere refuses; an unbound
project may use the candidate. Interrupted operations require explicit recovery.
`project-select ID --revision N --plan` checks the same candidate but permits
changing the association; `--yes` applies the revision-checked selection. The
returned project remains unchanged for a plan. Selection changes only that
project's environment ID, clearing previous run metadata when the ID changes;
selecting the same ID preserves it. These commands do not migrate environments,
create identities, download tools or start/stop engines. A caller must recheck
before activation because a successful preflight does not reserve the library.

`dinkster-env --root SOURCE fork ENTRY /absolute/DESTINATION --plan` reuses the
recorded snapshot reconstruction and approval rules without replacing the source.
`--yes` builds a new stopped environment with a fresh identity, setup journal
and registration in the source wheel store. It retains only software-selection
settings and the source library binding, not user storage settings or credentials.
Setup recovery removes an interrupted destination's settings and registration;
the source current pointer, history and running engine are left untouched.

`dinkster-env --root ROOT history` lists successful software activations;
`history show ID` includes the snapshot and structured differences. Both accept
`--json` and return the usual `dinkster.environment-result/1` envelope.
`<root>/history/<id>.json` stores `dinkster.install-history/1` entries with a
`dinkster.install-snapshot/1` software description, timestamp, reason, snapshot
number (the retained `generation` field), and differences from the preceding
entry. These files are independent of snapshot retention. Capture admits only sealed software manifests, constraints,
source locks and pack dependency locks, plus hosting and host facts; workflows,
models, outputs, libraries and mounts are not captured.

`setup --adopt-from OLD_ROOT --engine-release SHA --base-id SHA --variant NAME`
creates a new supervisor root from a standalone pack-only installation. Both
digests must match the explicitly selected channel; `catalog` exposes them.
Each installed pack requires an explicit target-variant `--pack-lock PATH`.
Planning shows the complete source lock, target dependency locks and in-process
placement. It never records the ambient interpreter as an engine/base release.
The plan compares local installed distribution metadata to target pack locks,
showing shared-store accelerator scopes separately without guessing which one
the old root used. It never runs old interpreters, pack code or .pth files.
Ambient/inherited dependencies are explicitly unrecorded, not an exact freeze;
a distribution absent from a target pack lock may still belong to the target base.
Changes to inspected distribution metadata invalidate the approved plan.
Shared-venv groups refuse rather than being silently split. The destination must
not exist and neither it nor the selected store may overlap the old root or its
shared content store. Confirmation copies verified software archives into owned
scratch and uses ordinary setup construction, health checks and history
publication. Source metadata and archive digests are rechecked before activation.
The old installation, its history, settings and user data are not copied or changed.

`restore ID [--plan | --yes]` rebuilds an exact published snapshot as a fresh
snapshot through normal health-checked activation, even if its old directory
was collected. Recorded manifests, locations, constraints and locks drive
acquisition; current channels do not select replacement versions. The new entry
names `restoredFrom` and both recorded and executing tools versions. Tools are
advisory and are never installed, replaced or downgraded by restore. Unavailable
inputs produce a read-only `restorePlan` with missing payloads, candidate snapshots
and every proposed substitution. Candidates minimize changed software identity
fields; equal-cost alternatives remain explicit rather than selecting the newest.
A base-only proposal preserves recorded pack sources, dependency locks and
in-process placement, and must pass their compatibility and availability checks
against the proposed base. Engine replacements also require the sealed pack
manifests' declared versions and engine ranges to agree with the recorded sources
and proposed engine. Older entries without those declarations do not infer them.
Snapshots retain pack manifests, not pack code. Acquisition provenance preserves
the original source archive cache release across engine replacements; target-owned
source admission still verifies the exact recorded pack digests.
`restore ID --approve DIGEST` approves one complete proposal. `--yes` alone never
approves substitutions. A changed environment, current pointer or history invalidates
the approval. Without an obtainable candidate, restore makes no changes.
Failed restores append separate immutable receipts under
`<root>/attempts/`, exposed as `attempts` in the history JSON result.
The activation journal records `restoredFrom` before construction; recovery of an
interrupted restore appends its failure receipt before making the journal terminal.
Repeated recovery never duplicates an attempt. Source archives are retained in
the engine cache after admission. Restore uses the target engine's digest-verified
source acquisition in disposable storage, without standalone provisioning or
activation. It collects missing sources before copying the set to the retained
cache; it never resolves newer pack versions.
Pack provisioning uses the staged snapshot locks as its complete source set,
not the current snapshot's locks, so later-added packs are not inherited.

Activation journals retain a prepared `historyEntry` before starting the target.
Only a completed activation publishes that entry. Recovery republishes an
interrupted publication idempotently; existing history bytes are never replaced.

`dinkster-env --root SOURCE move DESTINATION [--data STORE]` prints its plan
before confirmation. Move keeps source data until destination doctor and engine
health succeed, preserves named ports/devices and refuses an existing destination.
Same-volume moves retain interpreter/package sharing and rewrite pack venv homes;
cross-volume moves use local retained bases/wheels and offline pack rebuilding.
Absolute or escaping environment symlinks refuse planning. The old root retains
diagnostic journals and reports the destination instead of starting stale engines.

Optional `relocation` on a move journal contains absolute `sourceRoot`,
`targetRoot`, `sourceStore`, `targetStore`, and `sourceFiles` (relative paths to
SHA256 digests or `link:` plus relative link text). `setupRegistration` snapshots
destination registration ownership too. A temporary `healthProbe: true` binding
checks health when all original engines are stopped. Destination journal
publication uses `.dinkster-move-<timestamp>` beside the destination; interrupted
preparation rolls back before another confirmed operation. Cleanup refuses
changed source files; the original diagnostic journal remains after success.

All content identifiers are `sha256:` followed by 64 lowercase hexadecimal
digits. Relative file paths use `/`, never absolute paths or parent traversal.
Pointer and journal writes flush to disk before atomic replacement. The
`Generation`, `Journal`, `read_current` and `set_current` APIs read and write
these records; the latter refuses missing or misnumbered target snapshots.

Snapshot integrity checks hash every recorded file and stream the tree to
detect unexpected files. Pack environments and bytecode directories are pruned
from that discovery walk, not collected and sorted before being ignored. Pack
environments have separate dependency checks. Recorded files remain checked even
inside excluded directories; only unexpected files are sorted for diagnostics.
Pack reviews retain their integrity checks before and after registry preparation.

`packs-plan --progress /absolute/host-owned/progress.json` atomically publishes
stage-local `{sequence, stage, completed, total}` observations without changing
its JSON result on stdout. Integrity scans report batches of 128 checked files
and discovered directories; runtime, registry and preflight transitions report
their actual starts. This is not a timer heartbeat or overall percentage. Writes
within a stage are coalesced to at most four per second, except stage completion.
Creating `progress.json.cancel` requests cancellation at the next observation;
it never authorizes apply. The host retains the scratch directory until the
command has actually exited, including while an uninterruptible child finishes.

The channel format is `dinkster.channel/1`, with
`engine: {version, release, locations}`, `base: {variant: {id, locations}}`,
`minimumDesktopVersion`, and `signature`. Locations are a nonempty ordered array:
`{type: "mirror", path}` names the complete manifest path relative to a configured
mirror; `{type: "github", repository, tag, asset}` names an immutable release asset.
New mirror paths are `engine/<YYYY-MM-DD>-<version>-<hash8>/release-manifest.json`
and `base/<variant>/<YYYY-MM-DD>-py<python>-torch<torch>[-cu<nnn>]-<hash8>/manifest.json`.
Legacy full-hash directory paths remain supported. GitHub assets use those same manifest
basenames, never alias tags. Both stable and nightly name exact content digests,
never a latest-release lookup. Location hints do not replace digest verification.
A null signature is acceptable only for a Dinkster-operated mirror; signed
documents are refused until signature verification is available. Engine and base
artifact layouts are owned by their publishing tools, not this package.

### Extraction-only Desktop bootstrap

`scripts/build_environment_tools.py` builds the native tools payload from a
size/SHA256-pinned standalone `python/` tar.gz and the release-built supervisor
wheel. Run it with the same native Python patch version, implementation,
architecture and ABI as the standalone interpreter, including free-threaded
status; a mismatch is refused before wheel acquisition:

```sh
uv run --locked python -m scripts.build_environment_tools \
  --interpreter /absolute/path/cpython-install-only.tar.gz \
  --interpreter-size BYTES --interpreter-sha256 LOWERCASE_SHA256 \
  --supervisor-wheel /absolute/path/dinkster_supervisor-VERSION-py3-none-any.whl \
  --output /absolute/new/tools-output
```

`--interpreter` also accepts an HTTPS download URL; the complete size and hash
are checked before extraction or execution. Obtain those values from the
selected standalone interpreter's verified producer input, not a latest alias.
The independent tools workflow builds the supervisor wheel at this package's
own version, not an engine release version or a separately resolved PyPI version.
The command downloads only compatible
wheels whose filenames, versions, sizes and hashes occur in `uv.lock` and its
supervisor/tools export. uv is pinned by the supervisor's `environment-tools`
dependency group. Installation into the owned standalone tree is offline,
index-free, binary-only and hash-required. An offline `uv pip check` must
confirm that the supplied supervisor wheel and every installed dependency
have a satisfied closure before packaging. No engine or torch is installed.

The output is `dinkster-tools-<os>-<arch>.zip`, where os is `linux`, `windows`
or `macos`, and arch is `x86_64` or `arm64`. stdout is the
artifact row `{name,size,sha256}`; `SHA256SUMS` records the same archive hash.
`scripts/build_tools_release.py` finalizes all three native ZIPs into the
hash-verified `dinkster.tools/1` manifest and complete `SHA256SUMS`.
The `Supervisor tools publication` workflow runs when supervisor sources,
native build/publication scripts or tools build inputs change. It publishes
`tools/<date>-<public-supervisor-version>-<manifest-hash8>/`, `tools/index.json`
and the corresponding `tools-<label>` GitHub prerelease, verifies both sources,
then moves the Nightly channel's `tools` pointer without changing its engine/base
selection. An unpublished local output is not a production download pin.

`dinkster-env --root /absolute/tools-cache tools --channel nightly --mirror
https://releases.dinkster.org --yes --json` acquires the channel's native ZIP
using its tools manifest hash and archive size/hash, without acquiring engine
or base payloads. Omit `--yes` to inspect the selection without creating a cache.
Local ZIP folders use `<date>-<public-supervisor-version>/`, reserving `--2`,
`--3` for different manifest identities with the same label. Existing hash8-suffixed
folders remain reusable. Planning previews collisions without changing the cache.
Channels must supply an independent `tools` pointer for acquisition. A missing
or invalid pointer fails without falling back to engine assets; engine
releases do not carry native tools ZIPs.

Desktop selects that exact basename from its pinned tools manifest or the
channel's tools publication, downloads it from the selected release directory,
verifies its full size/SHA256,
and safely extracts it into a new owned directory. ZIP is only the bootstrap
container: it needs no Python/zstandard decoder before the tools exist and
does not change the producer's tar+zstd base contract. Internal interpreter
aliases are materialized as ordinary files/directories, so Windows extraction
does not require symlink privileges. The extractor must retain Unix executable
mode bits from ZIP attributes. Reject escaping paths and do not overwrite an
existing installation.

Invoke `<tools>/python/bin/python3 -I -m dinkster_supervisor.environment` on
Linux/macOS or `<tools>/python/python.exe -I -m dinkster_supervisor.environment`
on Windows. For each child prepend `<tools>/python/bin` (Windows
`<tools>/python/Scripts`) to PATH so the client's offline snapshot commands
find bundled uv; do not set a global PATH. Console-script shebangs are not the
relocatable interface. Extraction does not run pip, uv or package resolution.
The tools builder and the shared lifecycle builder at
`tests/fixtures/environments/build.py` are separate: the latter supplies tiny
production-format mirror inputs for Desktop/CLI equivalence, not production
tools or release artifacts.

`python -m dinkster_supervisor.environment_channels --release RELEASE_DIRECTORY
--base-manifest BASE_MANIFEST --channel nightly --output-root MIRROR_DIRECTORY
--minimum-desktop-version X.Y.Z` builds a local channel document. Repeat
`--base-manifest` for every selected variant; `--channel stable` uses the same
builder. It hashes the final release manifest bytes, derives the labelled location
for a new manifest or retains the embedded tag for a legacy manifest,
verifies every listed local engine artifact, validates each unchanged base
manifest and checks worker protocol and effective constraint compatibility.
Only after all inputs pass does it atomically replace `channels/<name>.json`.
It never downloads, changes publisher inputs or publishes to a remote service.
The release workflow validates a document after finalization, including on PRs.
Its push-only channel job runs after GitHub asset publication and calls
`scripts.publish_environment_channel` with the same selected base manifests.
With all R2 settings present, it checks published engine/base manifest bytes,
uploads only missing immutable engine objects, verifies their served hashes,
and replaces the selected channel last. Failed payload checks leave the old
channel untouched. An older main run cannot replace a newer main selection.
Absent R2 settings skip publication explicitly. The workflow compatibility
floor is `minimumDesktopVersion: "0.2.9"`, selected by the shared workflow
`DINKSTER_MINIMUM_DESKTOP_VERSION`. Raise it only when engine layout or lifecycle
compatibility needs a newer Desktop, and publish that Desktop before the channel
requires it. The local builder remains nonpublishing.

`environment_downloads.fetch_artifact(mirrors, relative_path, destination,
sha256, size)` downloads a known artifact from ordered local directories or
HTTPS mirrors (HTTP is limited to loopback fixtures). Its digest argument is
bare lowercase hexadecimal, not a `sha256:` identifier. Transfers hash bytes
while streaming, retain interrupted `.part` files and validate exact HTTP
range offsets on resume. Remote size/digest failures share the three-attempt
network retry budget: short bodies resume, full corrupt bodies restart from
zero, and every failed attempt appears in terminal errors. Local mirrors do
not use that retry loop. An ignored Range also restarts from zero. Final
publication requires both the expected size and digest. Signed query redirects
are allowed without logging their URLs; external non-TLS redirects and
credential-bearing mirror URLs are rejected. Calls have a bounded per-socket
network timeout.

`environment_artifacts.discover(mirrors, channel)` reads bounded metadata
without creating an environment, cache or store. It verifies engine manifest bytes
and the base producer's canonical identity, including replacement membership,
then returns the intact manifests for catalog display. Unsigned remote metadata
is accepted only from `releases.dinkster.org` or the official Dinkster GitHub
repositories; local directories and loopback servers support offline fixtures.
Unknown remote mirrors require signing support and are refused. Invalid or
corrupt metadata falls through to the next configured location.

Under the caller's store writer lock, `acquire(catalog, variant, store)` fetches
and verifies base parts, safely extracts a missing base, verifies its selected
wheels and populates `wheels/<wheel filename stem>/<filename>`. After wheel verification,
it atomically retains the producer base document at
`bases/<variant>/<date>-py<python>-torch<version>/manifest.json`
if absent, preserving an existing document's custom provenance. Base identity
remains the producer's canonical fields, independent of JSON formatting.
Each cache category records full content identities in `.objects.json`; a durable
allocation journal and OS lock protect collision reservations. Different bytes
with the same label receive `--2`, `--3`, and so on. A changed label for the same
identity reuses its original location. Existing full-hash paths remain supported
without renaming installed interpreters or environments. Base archive parts use
the same base label under `downloads/`. Unknown build dates use the first local
acquisition date in UTC. Wheel filenames remain unchanged, including backend tags.
Pack archives use `artifacts/<pack>/<date>-<version>/pack.zip`, with the date
of first local acquisition. Standalone pack sources use the same label beneath
`store/`; per-pack virtual environments use `venvs/<pack>/<date>-<version>/<pack>/`,
with an accelerator parent in shared stores. Grouped environments use bounded
sorted member labels under `venvs/groups/`, retaining their full aggregate
identity. Explicit collection resolves those identities rather than treating
readable namespace directories as disposable objects. Existing digest-named
archives and environments remain in place.
Engine-owned wheel and constraints artifacts are verified through the same
downloader. Base-owned
engine wheels are omitted, and tools/source archives are not installed into
the snapshot. Effective constraints are stored under
`engines/<date>-<version>/constraints-<variant>.txt`. The original verified engine
manifest bytes are atomically retained beside those constraints,
without reserialization or another metadata fetch. Missing or inconsistent
catalog bytes refuse acquisition before cache creation. A base without hosted parts
currently reports that manifest reconstruction is required; it never resolves
or substitutes dependencies silently.

`environment_constraints.effective_constraints(engine_text, base_manifest)`
uses the verified base manifest's `replaces` set, not package-name prefixes.
It removes every named engine requirement, including remove-only CUDA
transitives, and appends the selected `wheels` versions and hashes. Unrelated
requirements retain their markers and hashes. Selected versions must match
engine pins and `lockedVersion`; torch/torchvision backend suffixes may refine
the same locked public version. Invalid membership, non-exact requirements,
missing hashes and wheel filename/version mismatches fail without resolution.
The resulting constraints are retained beside the snapshot record.

`environment_generations.build_generation` constructs a complete target from
an already verified base resolved by its full identity and effective hashed
requirements. It runs uv offline with no indexes and binary wheels only,
reading the store's wheels and uv cache. Linux attempts CoW interpreter clones
and falls back to hardlinks when the filesystem cannot clone; Windows uses
hardlinks. macOS uses native CoW clones, with independent copies on volumes that
cannot clone, so executable identities remain local to each snapshot. Package
installation uses the matching uv link mode. The environment and store must share a
volume. A failed sync removes only its incomplete
target; it never changes `current` or the existing snapshot. The snapshot
interpreter, packages and retained constraints are hash-inventoried on success.

When a source snapshot, explicit pack locks or in-process hosting ids are
supplied, construction invokes the target interpreter's
`dinkster_workers.pack_environment` subprocess in preflight, rebuild and doctor
order before final inventory. All source locks remain mandatory unless explicitly
replaced by `--lock`; the adapter owns lock parsing and provisioning. The caller
supplies variant, worker protocol, measured own-environment size and shared store
paths explicitly. A nonzero exit, conflicting pack, doctor drift or malformed
single-object JSON result fails construction and removes the target. Supervisor
never imports the engine's worker packages to perform these operations. The
provisioner receives `--offline`: all extra wheels must already be verified in
the store, with no fallback to a dependency lock's URL.

When supplied together, `catalog_lock` and `artifacts` run the target interpreter
with `-B -m dinkster.generation_packs --generation ABS --catalog-lock ABS
--artifacts ABS --json` after dependency doctor and before inventory. Source
admission owns the existing registry lock and digest-named archives; the caller
does not parse a second catalog. Success requires a single object containing
distinct pack ids, an empty errors array, zero exit status and the written
`dinkster.lock`. Every admitted pack must appear in dependency doctor's result,
including packs with an empty extra dependency closure. An admission error,
missing dependency coverage, malformed result or missing inventory removes
the entire target, including any partial sources or interpreter changes. The
target interpreter then runs `-B -m dinkster.manager --root "" prepare-catalogs
--generation ABS` with repeated `--in-process-pack` selections. This command
verifies admitted sources and uses their prepared pack interpreters, without
legacy default discovery or dependency resolution. A preparation error removes
the target before activation. Catalogs enter the sealed file inventory even
though their `__pycache__/dinkster-schema-catalog` location is excluded from
source artifact identities. Additional `prepare_packs` staging runs before sealing.

For a custom base, the target interpreter can run that same source command with
`--prepare-locks ABS --variant NAME` to prepare standard dependency locks in a
new caller-owned scratch directory. It verifies the selected lock's exact
archives and manifest claims, then resolves their declared requirements against
the unsealed target's `constraints.txt`, using repeated `--index` and
`--find-links` inputs and optional `--accelerator`. JSON adds `locks`, the output
paths. This mode does not admit sources, change the source lock or seal the
snapshot. The caller retains the scratch directory through provisioning and
removes it afterward; no custom lock schema or store path is introduced.
`environment_custom.prepare_custom_pack_locks` invokes this mode in the acquired
target interpreter with cached constraint wheels, the custom extras index and
PyPI fallback. The engine's accelerator selection honors `DINKSTER_ACCELERATOR`
before host detection when including pack extras. It returns scratch lock paths
only; the existing snapshot builder still owns preflight, provisioning, source
admission and sealing.

`environment_archives.extract_base(parts, destination, platform=..., executable=...,
expected_parts=...)` verifies every ordered part's size and SHA-256 before streaming
one zstd decoder into tar extraction with the data filter. The caller supplies the
manifest's platform, interpreter path and part rows, and holds the store writer lock.
Extraction accepts only `python/` and `wheels/`, rejects traversal, absolute paths,
hardlinks, devices, Windows device names and exact duplicate paths. Internal relative
symlinks are preserved where supported by the host filesystem; unsupported or dangling
link graphs refuse extraction before promotion. Later members cannot write through
a symlink parent.
Case-ambiguous paths are refused for Windows/macOS targets or case-insensitive
destination filesystems; Linux case-sensitive trees retain distinct spellings.
Safe permission bits are preserved. A missing interpreter execute bit is restored
to 0755 with a warning. The manifest interpreter must pass an isolated sys/sysconfig
probe before promotion.
It returns `destination/python`. Existing bases are never replaced; failed
verification or extraction removes staging trees.

## Optional launch pack selection

Save a replacement launch filter in the existing environment configuration while
all environment engines are stopped:

```sh
dinkster-env --root /absolute/environment packs --only local-pack --only helper-pack --yes
dinkster-env --root /absolute/environment start --selected-packs --yes
dinkster-env --root /absolute/environment stop --yes
dinkster-env --root /absolute/environment start --yes
```

The final ordinary start uses all installed snapshot packs, not the previous
launch filter. `packs --none --yes` saves an empty selection. `packs
--exclude-partners --yes` excludes packs declaring a remote-gateway catalog;
it may be combined with an explicit `--only` allowlist. It changes node
availability, not browser networking or sandbox grants. `packs --yes` resets the
saved filter to all packs. `--plan` and `--json` use the ordinary preview/result
contract. `start --selected-packs` requires a saved `environment.json.packSelection`.
Changing a live binding's selection requires stopping it first.

All pack ids must be canonical and installed. Include every required pack and
capability provider explicitly: filtered launches refuse incomplete selections
before executing pack code. Selection does not remove retained bytes, change
software history, bypass inventory/security checks, or rebuild environments.
It projects a shared worker onto selected members only, retaining its admitted
interpreter and excluding removed members' manifests and import paths.

## Managed offline runtime

With all environment engines stopped, run `dinkster-env --root /absolute/environment
network offline --yes`. This persists `runtimeOffline: true` in the existing
`environment.json`; `--plan` previews without saving. Restart to apply it.
`network online --yes` restores the default online behavior after stopping.
Reload open product tabs after restarting: browser policy is applied when a
document or worker loads, not retroactively to an already-open online page.
This is separate from setup/update `--offline`, which uses cached installation
inputs and does not select runtime behavior.

Offline runtime enforces a browser Content Security Policy on product responses:
HTTP requests and WebSockets may reach the origin serving the product, including
its configured HTTPS address, but not another origin. Local frontend files,
API requests, event sockets, data/blob media and same-origin workers remain usable.
External scripts, frames, fonts and styles are blocked. Development frontends must
use the public product proxy for live-reload sockets, not a separate dev-server
origin. `X-Dinkster-Runtime-Network` reports `offline` or `online` on responses;
managed engine status includes `runtimeOffline: true` for an offline launch.

Partner packs identified by their remote-gateway catalog keep their admitted
local declarations but report an actionable unavailable reason and refuse
execution before starting their worker. A missing/stale local catalog refuses
startup rather than refreshing declarations in offline runtime. Ordinary members
of a shared worker do not import the disabled partner member. Select online to
refresh partner declarations and execute partner nodes.

This policy is not an operating-system firewall: unrelated pack/host networking,
user navigation outside the product, and other applications are not restricted.
The independent `packs --exclude-partners` selection removes partner catalogs
only on `start --selected-packs` and does not set this browser policy.

## Managed HTTP response compression

With all environment engines stopped, run `dinkster-env --root /absolute/environment
compression on --yes` to persist `responseCompression: true` in the existing
`environment.json`. Use `--plan` to inspect without saving, or `compression
off --yes` to disable it (the default). Restart the managed engine after changing
this policy.

Enabled engines negotiate gzip with `Accept-Encoding` for GET JSON and
text responses of at least 1024 bytes, and frontend text files. Encoding weights,
explicit gzip refusal and identity preference are respected. Responses vary on
Accept-Encoding; compressed representations use weak ETags. Frontend bundles
serve their original files rather than automatically selecting precompressed
siblings. Development-proxy responses retain upstream encoding; eligible plain
responses use the same policy. Supervisor forwarding preserves encoded bytes.

HEAD responses, streaming responses, event streams, WebSockets, byte ranges, no-transform
responses, already-encoded responses and binary media are not recompressed.
Compression is opt-in: do not enable it for a deployment where an attacker can
combine reflected input with secrets in compressed responses and measure sizes.

## Managed saved-preview storage

With all environment engines stopped, select an existing absolute directory:

```sh
dinkster-env --root /absolute/environment previews --directory /absolute/preview-disk --plan
dinkster-env --root /absolute/environment previews --directory /absolute/preview-disk --yes
```

The existing `environment.json` stores `previewStorage.directory`, retained
`previewStorage.previousDirectories`, and a stable `previewStorage.identity`
that survives environment relocation. New optional saved-file preview bytes use
an owned `dinkster-previews-<identity>/vault` child. Originals, uploads,
standalone encoded outputs and image documents keep their existing locations.
The operator-selected parent is not granted to workers.

Changing the directory keeps earlier locations readable without copying or
removing their bytes. Running `previews --yes` without `--directory` restores
default writes while retaining earlier locations for existing digest references.
Keep those locations attached: missing assigned storage, changed ownership,
links, junctions, retained-data overlap or a failed write probe refuse startup.
Do not manually move or remove an assigned vault to migrate storage. Select the
new location through this command instead; existing previews stay where they
were published and new previews use the newly selected location.

## Managed scratch storage

Select an existing absolute storage directory while all environment engines are stopped:

```sh
dinkster-env --root /absolute/environment scratch --directory /absolute/scratch-disk --plan
dinkster-env --root /absolute/environment scratch --directory /absolute/scratch-disk --yes
dinkster-env --root /absolute/environment start --sandbox-packs --yes
```

The policy is saved in `environment.json.scratchStorage.directory`. It applies to
engine upload/pack staging, scoped pack scratch, and ordinary isolated-worker
temporary files without moving retained library inputs or outputs. The selected
parent is never granted to a sandboxed worker. Missing directories, links or
junctions, unowned assigned roots, retained-root overlap, and failed write probes
refuse configuration/startup instead of silently using another medium.

The assigned root is `<directory>/dinkster-scratch-<environment-path-hash>`, with
`staging`, `packs`, and `workers` children. Engine staging and pack scratch use
these paths on Linux, macOS, and Windows. Each isolated worker gets a private
random child under `workers`, removed when it closes. On Linux the child's
backing directory is bound at private `/tmp` inside bubblewrap; on macOS
Seatbelt grants that child's absolute path, and on Windows ProcessContainer
grants only that child's path for TEMP/TMP/TMPDIR. Windows may expose sibling
existence; ungranted sibling reads and writes remain denied. Windows refuses
storage whose assigned worker path exceeds 259 UTF-16 code units, including
the environment-owned prefix and random worker name. Choose a shorter parent if
configuration reports this limit: longer assigned directories can deny contained
writes or hang native GetTempPath APIs.
The selected filesystem provides the medium; selecting a RAM filesystem uses RAM.
Transport endpoints remain in their existing short host-temporary directories,
not under the potentially long selected path. Managed stop requests engine cleanup
through its private endpoint before bounded forced termination, including on
Windows where process termination does not run Python cleanup handlers.

`scratch --yes` clears the policy and restores existing defaults: engine and
pack scratch under the library, Linux private worker `/tmp` on tmpfs, macOS
private scratch beside its endpoint, and Windows executor defaults. Plain
unsandboxed workers are not a security boundary. Neither clearing nor replacing
policy removes old scratch: pending produced-output handoffs stay intact and
are recovered by reselecting their original storage. Ordinary staged files and
closed worker temporary files follow their normal cleanup lifecycle. Retained
inputs/outputs and unrelated files on the selected storage are never collected
by this command. `--plan` does not create directories; `--json` returns the
saved policy and, when selected, the assigned root. Ambient TMPDIR does not
select managed scratch storage.

## Managed node exclusions

Replace the persistent node-type exclusion list while all engines are stopped:

```sh
dinkster-env --root /absolute/environment nodes --exclude example.blocked --yes
dinkster-env --root /absolute/environment start --yes
dinkster-env --root /absolute/environment stop --yes
dinkster-env --root /absolute/environment nodes --yes
```

`--exclude NODE_TYPE` is repeatable. `nodes --yes` clears all exclusions.
`--plan` and `--json` follow the ordinary preview/result contract. Settings live
in `environment.json.excludedNodeTypes`, apply to every managed launch (including
ordinary starts), and remain until explicitly replaced or cleared. Stop all
engines before changing them. Malformed or duplicate IDs refuse configuration;
unknown IDs refuse startup against the composed node surface, including when a
pack allowlist omits their pack.

Excluded nodes remain discoverable with `unavailableReason` and refuse direct
workflow submission with the same reason. Other nodes in the same pack remain
usable. Clearing the list restores execution without modifying pack artifacts,
software selections, admission refusals, or network grants. Standalone launches
without a managed build do not read this setting.

## Environment queries and runtime

`setup --mirror PATH --channel nightly --variant VARIANT --data PATH` plans a first
snapshot; `update` plans the configured channel's next snapshot. Both require
`--yes` or terminal confirmation before creating environment/store state or acquiring
payloads. `--plan` and unconfirmed JSON calls create nothing. Confirmed calls
acquire and preflight target inputs before journaling and registration, sync
offline, check the candidate engine's health, publish `current`, then stop the old engine. Update preserves
existing engine bindings; setup accepts `--engine`, `--port` and repeatable `--gpu`.
The setup confirmation plan includes that binding name, loopback port and ordered
`CUDA_VISIBLE_DEVICES` assignment, including the empty CPU-only assignment.
Stopped bindings keep their assigned ports; only running bindings need new ports
for candidate health checks alongside their previous processes.
Managed local snapshots start with the environment's `dinkster-library` persistence
root (or the inherited `DINKSTER_LIBRARY_ROOT`), stale-catalog preparation and
operator-only mount editing enabled. Snapshot selection remains the only pack
source. Folder grants persist across stop/start; fixed sandbox grants still
require a controlled restart. Remote connections and arbitrary supervisor/server
commands do not acquire these options.
Failed acquisition/build/startup retains the previous serving tree and a rolled-back
journal. A confirmed retry first recovers interrupted activation.

`environment_generations.prepare_engine_interpreter` prepares a new,
caller-owned directory from an acquired base and effective hashed constraints.
It returns the target Python after the same offline sync used by snapshot
construction. The caller holds the store writer lock and owns directory cleanup,
including failures. This operation does not acquire inputs, prepare packs or
write environment, journal or snapshot metadata. Snapshot construction separately
retains base provenance, prepares packs and seals the inventory.

Snapshot construction copies a cached `engines/<release hex>/release-manifest.json`
byte-for-byte into `engine-manifest.json` before pack preparation and inventory.
The retained bytes must hash to the selected engine release. A missing legacy
cache is tolerated; present but unreadable or corrupt content refuses the build
and removes the incomplete target.

Existing per-variant dependency locks can be supplied with repeated `--pack-lock`,
and in-process placement with repeated `--in-process-pack`. Paired absolute
`--catalog-lock` and `--artifacts` inputs feed verified source admission in the
target interpreter. Without source overrides, confirmed published operations
prepare a temporary target interpreter from the acquired constraints under the
store lock, before allocating a snapshot or opening its journal. Isolated
`-I -B -m dinkster.manager` commands resolve fresh registry defaults or re-resolve
the installed official source set, and acquire exact-base published dependency
locks. The supervisor checks the target version, returned paths and diagnostics;
the target engine owns source/lock parsing and verification. Temporary inputs
survive candidate construction and are removed on success or failure.

Fresh setup reports unavailable defaults as `unavailablePacks` and on stderr;
an offline registry can produce a healthy empty snapshot. Update refuses a
missing compatible selection before journal/environment mutation, retains other
source provenance and verifies reconstructed non-registry source bytes.
Explicit dependency locks replace automatic registry lock acquisition. With an
explicit source catalog but no dependency locks, legacy release-carried locks
remain usable. No supervisor dependency parser or resolver is involved.

`prepare_registry_inputs` accepts an optional `prepare_locks` callback after
source selection, before published lock retrieval. It receives the acquired
scratch engine directory with effective constraints, selected catalog lock,
exact archive cache and new output directory. A custom-base caller can resolve
standard locks from those inputs without a published base binding or a second
source selector. Returning an empty tuple still replaces published retrieval;
raising refuses preparation and removes the scratch tree. Returned files must
belong to the output directory. Explicit `pack_locks` take precedence. The optional
`preflight(engine, catalog, artifacts)` runs after source selection and before
either automatic or explicit dependency locks are used.

Update plans with registered workflows or explicit additions use the active
snapshot's requirements reader to snapshot requirements and input revisions.
Unconfirmed plans explicitly report source feasibility as pending; they never
download payloads or install a planning interpreter. SQLite read-only planning can
touch WAL coordination files without changing logical records or environment content.
After confirmation, the acquired target interpreter runs
`dinkster.workflow_requirements project-plan` against the sole selected registry
catalog and archive cache, before a snapshot or journal exists. Explicit
catalog/cache inputs take precedence and use the same target preparation. Changed
requirements/revisions and unavailable saved versions refuse admission. Pack and
asset actions appear in the existing plan strings. Catalog digests and input
revisions are checked again before journaling and application.
The sealed target interpreter then consumes the approved plan, checks its
admitted pack identities and acquires only approved assets. Workflow input
revisions are checked before publication and after candidate health checks;
activation also requires an empty target requirement plan. Failure retains the
previous snapshot, serving state and diagnostic journal; verified cached asset
bytes may remain. Workflow records are never rewritten.
First setup with requirements needs a runnable snapshot planner. Update can
select another `--channel` or `--variant` through the same confirmed construction
and workflow replan. Environment selection changes only after target health and final
revalidation; other environment fields are preserved. These commands are not a
complete released-engine setup claim.

`stores.json` lives under `dinkster/` in `LOCALAPPDATA` on Windows or
`XDG_DATA_HOME` (default `~/.local/share`) on Linux. It has
`{format: "dinkster.stores/1", volumes: {"<st_dev>": "<absolute data path>"}}`.
The first plan defaults to `dinkster-data` beside the environment; later environments
on the same volume reuse the registered store. A conflicting `--data`, cross-volume
store or overlapping environment/data paths is refused. The store's `projects.json`
retains `{format: "dinkster.store/1", projects: ["<absolute environment root>"]}`;
the filename and field name are compatibility identifiers, not the project list.
Registry, store and environment writers hold OS locks while registering references.
Failed initial setup restores prior environment metadata and removes only the store
and volume references that setup added. Cached bytes, pre-existing settings,
other environments' references and the diagnostic journal remain. A retry can select
another `--data` when no other environment references the old store. Failed updates
keep the existing environment's references and active snapshot.
Identity/health/shutdown requests use direct loopback connections, never inherited
HTTP proxies.

`dinkster-env --root <absolute-path> catalog|status|doctor --json` emits one
`dinkster.environment-result/1` object with `operation`, `root`, `current`,
`plan` and `status`. Catalog accepts repeatable `--mirror` and `--channel`
stable/nightly; it adds `channel`, `channelManifest` and intact `variants`.
Status returns configured `channel`/`variant` and observed `engines`, never
starts a process and never creates a prospective root. Doctor reports content
and pack drift without repair and exits nonzero on drift or errors.

Published setup/update reads the engine's per-variant dependency lock and checks
its exact base id and wheel identities against the producer manifest. Non-base
wheels are fetched from their locked origins into the shared wheel store using
exact filenames, sizes and SHA256; snapshot construction stays offline.
Before constructing a snapshot, acquired non-base wheel metadata is checked
against base-owned versions using its dependency specifiers, target Python
markers and requested transitive extras. Incompatible or absent base
distributions fail with the requiring distribution, specifier and base version.
`--package-index URL` selects a simple package mirror offering those same locked
filenames, without resolution, source builds, hash overrides or origin fallback.
Confirmed setup saves it as environment `packageIndex`; updates preserve it or change
it through the existing activation transaction. This is independent of channel
and Dinkster artifact `--mirror`. Intact cached wheels require no index access.

`doctor --repair [--plan | --yes]` rebuilds the same release offline into a new
snapshot using the existing `repair` activation journal. Plans are read-only;
apply requires confirmation. It rejects absolute, escaping or cyclic base links,
verifies relative internal aliases and regular files against sealed file hashes,
and ignores disposable bytecode. It selects constraints with the recorded digest and admits retained source
locks from the release's cached archives. Missing or corrupt local inputs refuse
instead of fetching a channel or reading an index. The shared store lock protects
uv cache cleanup and reconstruction: hand edits to installed hardlinks can also
corrupt extracted cache entries, so repair repopulates that disposable cache from
hash-verified wheels. Pack rebuilding receives explicit `--offline`, refusing
missing or corrupt dependency wheels before any URL access. Retained snapshots
are not edited. Candidate file names
must match the original inventory, and normal health/activation checks run before
switching current. Failure keeps the previous selection and diagnostic journal.

`start --engine <name> [--port <port>] [--gpu <device> ...]` and
`stop --engine <name>` share the same result envelope. They show a plan and
require `--yes` or terminal confirmation; `--plan` and unconfirmed machine
invocations do not launch/stop processes or create runtime/lock files. Main
uses the configured port by default; a new engine name requires an explicit
port. Existing bindings default to their saved port, including after update or
restore. An empty device list hides GPUs, including inherited host assignments.
Identical ready assignments are idempotent; a changed assignment requires stop
first. Applied results include observed engines and the started engine URL.

Runtime writers serialize through the environment's persistent `.lock` file using
an OS-held nonblocking lock. The same primitive serializes shared-store writes;
callers needing both acquire store then environment. Process exit releases the
lock, but its file is never removed. Apply rechecks environment/current under lock and
recovers incomplete activation journals before performing another mutation.

Runtime launches the snapshot interpreter through the independent supervisor.
The engine receives `--generation <absolute-snapshot-root>` explicitly,
including candidate validation before the pointer switch and subsequent starts
of that snapshot. Restore reconstructs a fresh snapshot from recorded inputs;
it does not reactivate the old directory.
Source selection never follows the working directory or the current pointer.
Public engine rows contain `name`, `generation`, `state` and `url`; private
`runtime/<generation>/<name>.json` records retain process identity and shutdown
credentials. This permits old and candidate same-name bindings to coexist;
status displays the snapshot selected by `current`, while idle checks inspect
every persisted snapshot's engines.
Graceful shutdown requires both the live instance identity and its private
token. An unreachable or ambiguous supervisor cannot establish an idle engine;
queue checks refuse mutation unless every observed engine is safely idle.
Shutdown closes proxied WebSockets with a going-away frame and cancels active
HTTP proxy transfers, so stalled readers cannot prevent engine replacement.
Each WebSocket proxy handler cancels and joins both pumps before releasing its upstream.
Supervisor and engine output goes to private runtime logs, not CLI JSON stdout.

`restore-previous [--plan | --yes] --json` selects the preceding successful history
entry and rebuilds it as a fresh snapshot through the same path as `restore ID`.
The old snapshot directory is not required. The new entry retains the serialized
reason `rollback` and names its source in `restoredFrom`. Missing inputs require
the same complete proposal approval with `restore-previous --approve DIGEST`;
`--yes` never substitutes software identities. Environment settings and data stay
unchanged, including edits made after the recorded activation. Apply revalidates
the plan and current state under the writer locks; an unconfirmed restore writes
nothing. The released `rollback` command remains an alias for the same operation.

Channel/variant selection journals also carry optional `projectSettings`, an
object containing `previous` and `target` environment settings text strings. The
field name and historical text remain readable after the disk migration.
Activation writes target settings during `switching`, immediately before
`current`. Interrupted switching restores the previous exact text along with the
pointer; earlier stages never changed settings. Unexpected settings during
recovery refuse rather than overwrite an external edit. Software restore never
reverses this settings pair: only software identities are restored. Journals
without this field leave environment settings alone.

Activation journals retain `bindings`: objects containing `name`, `port`,
`gpus` (device-id strings), `running` (boolean), and `targetPort`. The prior
assignments and new ports are durable before construction or process launch.
Stopped bindings remain stopped across activation. Recovery stops authenticated
target processes, restores the previous pointer and running assignments, removes
only a newly constructed failed target, and marks the journal `rolled-back`.
Failed rebuilt restore-previous targets are removed, just like failed restore targets.
Ambiguous process
identity leaves the journal incomplete rather than killing an unrelated process.

Initial activation uses the same journal with operation `setup` and previous
`null`. `begin_setup` records the requested binding, port and devices before
construction. Activation publishes `current` only after target health succeeds.
Interrupted setup stops its candidate, removes any target pointer and incomplete
tree, and retains the rolled-back journal; it does not invent a prior snapshot.
CLI setup also records `setupRegistration`: absolute `store` and `registry` paths,
prior environment settings text in `project` and store-pointer text in `pointer`
(each nullable), and boolean `addedProject`, `addedVolume`, `hadProjects` and
`hadRegistry` flags. Recovery restores local metadata exactly, subtracts only its
own environment reference, and removes its volume mapping only if it still names that
store and no environments remain. It locks registry, store and environment in that order;
content replacements and removals are durable before the journal becomes terminal.
Cache directories and persistent OS-lock inodes are not deleted.

For engine manifests declaring `library-helper/1`, the tools' stdlib-only
pre-import wrapper selects working state under
`engine-state/<engine name>/working-<id>`. The parent lifetime lock prevents
competing selections. `state.json` binds the library path, `lastSnapshot`,
writer engine release digest and selected `directory` with format
`dinkster.engine-state-selection/1`; unknown fields survive updates. Reuse
requires a nondecreasing snapshot number and a compatible writer. The current
tools derive compatible writers from installation history: lower PEP 440 engine
versions, the same release, or equal-version releases first recorded no later
than the selected release. Rebuilding older software under a higher snapshot
number therefore does not adopt newer state. The wrapper validates the selected
engine and compatible release digests without importing engine dependencies.
An older snapshot or incompatible writer selects a fresh directory without
opening or removing the newer directory or its vault-local uploads. A return to
a newer snapshot then carries that fresh directory forward. Selection publishes through a flushed
`.state.pending` file and atomic replacement before creating/opening working
state. Interrupted publication retries without deleting retained state.
Populated unbound parents, invalid selectors, another library, linked paths
and multiply linked staging files refuse. Legacy manifests do not select state.

Managed supervisors also use a private per-instance lifecycle directory at
`runtime/<generation>/<name>.process/<sha256 of instance>/`. Its persistent
`.lock` is held from launch admission through child/listener cleanup, and
`state.json` contains `instance`, `state` and `pid`. States are `pending`,
`running`, `stopped` and `cancelled`. Recovery can cancel an unclaimed pending
launch under the same lock, so a delayed supervisor cannot start afterward.
Only the supervisor writes `stopped`, after cleanup; that receipt survives a
client exit before its binding record is updated. A dead supervisor with a
`running` receipt does not prove its child exited and still refuses mutation.
These private process receipts contain no shutdown token and do not replace
snapshot metadata, dependency locks or operation journals.

`gc [--plan | --yes] --json` collects obsolete snapshot trees, retaining current
and its most recent successful journal predecessor rather than the largest two
numbers. Live or ambiguous obsolete bindings refuse collection. The journal's
optional `collectedGenerations` array names distinct positive obsolete numbers;
`previous` and `target` both name current. Trees move durably into
`.garbage/<journal timestamp>/<number>` before the journal becomes complete.
Incomplete collection restores the trees; only complete collection trash is
deleted, with cleanup retried by later confirmed operations. Previously successful
snapshot numbers are never reused after collection. Environment libraries remain
unchanged. A registered store also collects unreferenced digest-named bases,
engine caches, archive downloads and wheel directories, plus its disposable uv
cache. All extant snapshots in other registered environments retain their
references. Missing environments, damaged retained constraints or unsafe paths refuse
store deletion. Dependency references come from the target provisioner's
read-only `references` result (`wheelHashes`), never a supervisor pack parser.
Pack dependency-lock files are included in the snapshot inventory; missing
whole pack directories therefore refuse reference collection too.

Optional journal fields `collectionStore` and `collectedStorePaths` name the
absolute store and disposable relative entries (`bases/<sha256>`,
`engines/<sha256>`, `downloads/<sha256>`, `wheels/<sha256>`, `uv-cache`). They are
quarantined under `<store>/.garbage/<journal timestamp>/<relative entry>` before
the same completion write. `<store>/collection` holds the absolute owning journal
path plus newline until recovery/cleanup finishes, so another environment can recover
the store before mutation. Registry, store and registered environment locks protect
plan revalidation and quarantine. Missing references never imply abandonment.
