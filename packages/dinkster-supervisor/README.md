# dinkster-supervisor

`dinkster-supervisor` is the layer-0 process that owns the public port, starts an
engine host on an internal port, and proxies HTTP and WebSocket traffic only
after `GET /api/health` succeeds. Its station process applies the same
version-independent machinery to multiple configured Dinkster installations.

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

The environment state module owns the project records consumed by environment
management and Desktop. Full layout: [environments and updates](../../docs/environments-and-updates.md).

- `<root>/current`: positive decimal generation number and newline.
- `<root>/store`: absolute shared data-directory path and newline.
- `<root>/generations/<n>/python/`: complete interpreter tree; executable is
  `python.exe` on Windows, `bin/python3` on Linux.
- `<root>/generations/<n>/constraints.txt`: the release's hashed requirements.
- `generation.json`: `format: dinkster.generation/1`, `generation` integer,
  `base` digest, `engine: {version, release}`, `constraints` digest, `variant`,
  `workerProtocol` integer, and `files` (relative path -> `sha256:` digest).
  The inventory covers owned interpreter/package files and constraints, not
  itself, runtime bytecode or pack venvs. Drift reports name missing, modified
  and unexpected files, even when installed distribution versions agree.
  `inProcessPacks` is an optional array of pack ids, empty by default; these
  packs retain their dependency locks but have no pack venv. Inventory is
  finalized only after their dependencies are staged into a fresh generation.
- `<root>/journal/<timestamp>.json`: `format: dinkster.journal/1`, `operation`,
  `previous` integer or null, `target` integer, `stage`, and optional `error`.
  Operations are setup/update/rollback/move/repair/gc. Stages are planned,
  building, starting, switching, complete and rolled-back; the last two are
  terminal. Interrupted records must be recovered before another operation.

`dinkster-env --root SOURCE move DESTINATION [--data STORE]` prints its plan
before confirmation. Move keeps source data until destination doctor and engine
health succeed, preserves named ports/devices and refuses an existing destination.
Same-volume moves retain interpreter/package sharing and rewrite pack venv homes;
cross-volume moves use local retained bases/wheels and offline pack rebuilding.
Absolute or escaping project symlinks refuse planning. The old root retains
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
these records; the latter refuses missing or misnumbered target generations.

The channel format is `dinkster.channel/1`, with
`engine: {version, release, locations}`, `base: {variant: {id, locations}}`,
`minimumDesktopVersion`, and `signature`. Locations are a nonempty ordered array:
`{type: "mirror", path}` names the complete manifest path relative to a configured
mirror; `{type: "github", repository, tag, asset}` names an immutable release asset.
Mirror paths are `engine/<release hex>/release-manifest.json` and
`base/<variant>/<base hex>/manifest.json`. GitHub assets use those same manifest
basenames, never alias tags. Both stable and main name exact content digests,
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
The supervisor wheel is the same wheel passed to the engine release builder,
not a separately resolved PyPI version. The command downloads only compatible
wheels whose filenames, versions, sizes and hashes occur in `uv.lock` and its
supervisor/tools export. uv is pinned by the supervisor's `environment-tools`
dependency group. Installation into the owned standalone tree is offline,
index-free, binary-only and hash-required. An offline `uv pip check` must
confirm that the supplied supervisor wheel and every installed dependency
have a satisfied closure before packaging. No engine or torch is installed.

The output is `dinkster-tools-<os>-<arch>.zip`, where os is `linux`, `windows`
or `macos`, and arch is `x86_64` or `arm64`. stdout is its existing engine
artifact row `{name,size,sha256}`; `SHA256SUMS` records the same archive hash.
The release builder accepts the archive using its existing `--artifact` flag,
which includes that row in `dinkster.engine/1`. Release publication owns calling
the native builder and attaching all platform archives. There is no separate
tools manifest or extra PyPI distribution. An unpublished local output is not
a production download pin.

Desktop selects that exact basename from the hash-verified engine manifest,
downloads it from the same release directory, verifies its full size/SHA256,
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
`<tools>/python/Scripts`) to PATH so the client's offline generation commands
find bundled uv; do not set a global PATH. Console-script shebangs are not the
relocatable interface. Extraction does not run pip, uv or package resolution.
The tools builder and the shared lifecycle builder at
`tests/fixtures/environments/build.py` are separate: the latter supplies tiny
production-format mirror inputs for Desktop/CLI equivalence, not production
tools or release artifacts.

`python -m dinkster_supervisor.environment_channels --release RELEASE_DIRECTORY
--base-manifest BASE_MANIFEST --channel main --output-root MIRROR_DIRECTORY
--minimum-desktop-version X.Y.Z` builds a local channel document. Repeat
`--base-manifest` for every selected variant; `--channel stable` uses the same
builder. It hashes the final release manifest bytes, retains its exact tag,
verifies every listed local engine artifact, validates each unchanged base
manifest and checks worker protocol and effective constraint compatibility.
Only after all inputs pass does it atomically replace `channels/<name>.json`.
It never downloads, changes publisher inputs or publishes to a remote service.
The release workflow must invoke it only after successful asset publication;
this local builder does not select published bases or wire that workflow.

`environment_downloads.fetch_artifact(mirrors, relative_path, destination,
sha256, size)` downloads a known artifact from ordered local directories or
HTTPS mirrors (HTTP is limited to loopback fixtures). Its digest argument is
bare lowercase hexadecimal, not a `sha256:` identifier. Transfers hash bytes
while streaming, retain interrupted `.part` files and validate exact HTTP
range offsets on resume. A corrupt saved prefix retries once from zero;
an ignored Range also restarts from zero. Final publication requires both
the expected size and digest. Signed query redirects are allowed without
logging their URLs; external non-TLS redirects and credential-bearing mirror
URLs are rejected. Calls have a bounded per-socket network timeout.

`environment_artifacts.discover(mirrors, channel)` reads bounded metadata
without creating a project, cache or store. It verifies engine manifest bytes
and the base producer's canonical identity, including replacement membership,
then returns the intact manifests for catalog display. Unsigned remote metadata
is accepted only from `releases.dinkster.org` or the official Dinkster GitHub
repositories; local directories and loopback servers support offline fixtures.
Unknown remote mirrors require signing support and are refused. Invalid or
corrupt metadata falls through to the next configured location.

Under the caller's store writer lock, `acquire(catalog, variant, store)` fetches
and verifies base parts, safely extracts a missing base, verifies its selected
wheels and populates `wheels/<sha256 hex>/<filename>`. Engine-owned wheel and
constraints artifacts are verified through the same downloader. Base-owned
engine wheels are omitted, and tools/source archives are not installed into
the generation. Effective constraints are stored under
`engines/<release hex>/constraints-<variant>.txt`. A base without hosted parts
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
The resulting constraints are retained beside the generation record.

`environment_generations.build_generation` constructs a complete target from
an already verified base under `bases/<hex>/python/` and effective hashed
requirements. It runs uv offline with no indexes and binary wheels only,
reading the store's wheels and uv cache. Linux attempts CoW interpreter clones
and falls back to hardlinks when the filesystem cannot clone; Windows uses
hardlinks. Package installation uses the matching uv link mode. The project
and store must share a volume. A failed sync removes only its incomplete
target; it never changes `current` or the existing generation. The generation
interpreter, packages and retained constraints are hash-inventoried on success.

When a source generation, explicit pack locks or in-process hosting ids are
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

## Environment queries and runtime

`setup --mirror PATH --channel main --variant VARIANT --data PATH` plans a first
generation; `update` plans the configured channel's next generation. Both require
`--yes` or terminal confirmation before creating project/store state or acquiring
payloads. `--plan` and unconfirmed JSON calls create nothing. Confirmed calls
write a journal before registration/acquisition, sync offline, check the candidate
engine's health, publish `current`, then stop the old engine. Update preserves
existing engine bindings; setup accepts `--engine`, `--port` and repeatable `--gpu`.
The setup confirmation plan includes that binding name, loopback port and ordered
`CUDA_VISIBLE_DEVICES` assignment, including the empty CPU-only assignment.
Stopped bindings keep their assigned ports; only running bindings need new ports
for candidate health checks alongside their previous processes.
Failed acquisition/build/startup retains the previous serving tree and a rolled-back
journal. A confirmed retry first recovers interrupted activation.

Existing per-variant dependency locks can be supplied with repeated `--pack-lock`,
and in-process placement with repeated `--in-process-pack`. Paired absolute
`--catalog-lock` and `--artifacts` inputs feed verified source admission in the
target interpreter. Without those overrides, a selected release carrying
`official-packs.lock` supplies its source inputs automatically. The client downloads
that lock and the release's ZIP payloads with the manifest's SHA256/size checks,
then caches each ZIP as `engines/<release-id>/artifacts/<blake3-hex>.zip`.
It does not parse the registry lock or unpack sources; target-interpreter admission
still verifies identity and claims before sealing. A missing admission command
refuses the generation rather than omitting sources. Without explicit `--pack-lock`
inputs, the client acquires release artifacts named
`<pack>-<public-version>-<variant>.pack-lock.json` for its selected variant with
manifest SHA256/size verification. It stores original bytes under
`engines/<release-id>/` and passes their paths to the existing provisioner.
Explicit lock inputs replace this bundled selection; no supervisor dependency
parser or resolver is involved.
Update plans with registered workflows or explicit additions call the active generation's
`dinkster.workflow_requirements project-plan` adapter. Explicit catalog/cache
inputs take precedence; otherwise selected-release sources must already exist in
the cache with verified SHA256/size and BLAKE3 archive names. Planning never
downloads or repairs source payloads. Pack and asset actions appear in the existing
plan strings; unavailable saved versions refuse planning. The core adapter retains
ownership of requirement derivation. SQLite read-only planning can touch WAL
coordination files without changing logical records or project content.
On confirmation, the target interpreter consumes the approved plan, checks its
admitted pack identities and acquires only approved assets. Workflow input
revisions are checked before publication and after candidate health checks;
activation also requires an empty target requirement plan. Failure retains the
previous generation, serving state and diagnostic journal; verified cached asset
bytes may remain. Workflow records are never rewritten.
First setup with requirements needs a runnable generation planner. Update can
select another `--channel` or `--variant` through the same confirmed construction
and workflow replan. Project selection changes only after target health and final
revalidation; other project fields are preserved. These commands are not a
complete released-engine setup claim.

`stores.json` lives under `dinkster/` in `LOCALAPPDATA` on Windows or
`XDG_DATA_HOME` (default `~/.local/share`) on Linux. It has
`{format: "dinkster.stores/1", volumes: {"<st_dev>": "<absolute data path>"}}`.
The first plan defaults to `dinkster-data` beside the project; later projects on
the same volume reuse the registered store. A conflicting `--data`, cross-volume
store or overlapping project/data paths is refused. The store's `projects.json`
has `{format: "dinkster.store/1", projects: ["<absolute project root>"]}`.
Registry, store and project writers hold OS locks while registering references.
Failed initial setup restores prior project metadata and removes only the store
and volume references that setup added. Cached bytes, pre-existing settings,
other projects' references and the diagnostic journal remain. A retry can select
another `--data` when no other project references the old store. Failed updates
keep the existing project's references and active generation.
Identity/health/shutdown requests use direct loopback connections, never inherited
HTTP proxies.

`dinkster-env --root <absolute-path> catalog|status|doctor --json` emits one
`dinkster.environment-result/1` object with `operation`, `root`, `current`,
`plan` and `status`. Catalog accepts repeatable `--mirror` and `--channel`
stable/main; it adds `channel`, `channelManifest` and intact `variants`.
Status returns configured `channel`/`variant` and observed `engines`, never
starts a process and never creates a prospective root. Doctor reports content
and pack drift without repair and exits nonzero on drift or errors.

`doctor --repair [--plan | --yes]` rebuilds the same release offline into a new
generation using the existing `repair` activation journal. Plans are read-only;
apply requires confirmation. It rejects absolute, escaping or cyclic base links,
verifies relative internal aliases and regular files against sealed file hashes,
and ignores disposable bytecode. It selects constraints with the recorded digest and admits retained source
locks from the release's cached archives. Missing or corrupt local inputs refuse
instead of fetching a channel or reading an index. The shared store lock protects
uv cache cleanup and reconstruction: hand edits to installed hardlinks can also
corrupt extracted cache entries, so repair repopulates that disposable cache from
hash-verified wheels. Pack rebuilding receives explicit `--offline`, refusing
missing or corrupt dependency wheels before any URL access. Retained generations
are not edited. Candidate file names
must match the original inventory, and normal health/activation checks run before
switching current. Failure keeps the previous selection and diagnostic journal.

`start --engine <name> [--port <port>] [--gpu <device> ...]` and
`stop --engine <name>` share the same result envelope. They show a plan and
require `--yes` or terminal confirmation; `--plan` and unconfirmed machine
invocations do not launch/stop processes or create runtime/lock files. Main
uses the configured port by default; a new engine name requires an explicit
port. Existing bindings default to their saved port, including after update or
rollback. An empty device list hides GPUs, including inherited host assignments.
Identical ready assignments are idempotent; a changed assignment requires stop
first. Applied results include observed engines and the started engine URL.

Runtime writers serialize through the project's persistent `.lock` file using
an OS-held nonblocking lock. The same primitive serializes shared-store writes;
callers needing both acquire project then store. Process exit releases the lock,
but its file is never removed. Apply rechecks project/current under lock and
recovers incomplete activation journals before performing another mutation.

Runtime launches the generation interpreter through the independent supervisor.
The engine receives `--generation <absolute-generation-root>` explicitly,
including before the pointer switch and when reactivating a retained tree.
Source selection never follows the working directory or the current pointer.
Public engine rows contain `name`, `generation`, `state` and `url`; private
`runtime/<generation>/<name>.json` records retain process identity and shutdown
credentials. This permits old and candidate same-name bindings to coexist;
status displays the generation selected by `current`, while idle checks inspect
every persisted generation's engines.
Graceful shutdown requires both the live instance identity and its private
token. An unreachable or ambiguous supervisor cannot establish an idle engine;
queue checks refuse mutation unless every observed engine is safely idle.
Supervisor and engine output goes to private runtime logs, not CLI JSON stdout.

`rollback [--plan | --yes] --json` selects the previous generation from the
latest successful activation journal, starts its retained interpreter on new
ports, waits for health, replaces `current`, then stops the newer processes.
It does not sync or rebuild either tree. Apply revalidates project bytes and
current under the writer lock; unconfirmed rollback does not write anything.

Channel/variant selection journals also carry optional `projectSettings`, an
object containing `previous` and `target` project.json text strings. Activation
writes target settings during `switching`, immediately before `current`.
Interrupted switching restores the previous exact text along with the pointer;
earlier stages never changed settings. Unexpected settings during recovery refuse
rather than overwrite an external edit. Rollback reverses the successful pair
without rebuilding; settings edited since selection refuse rollback rather than
discard user changes. Journals without this field leave project settings alone.

Activation journals retain `bindings`: objects containing `name`, `port`,
`gpus` (device-id strings), `running` (boolean), and `targetPort`. The prior
assignments and new ports are durable before construction or process launch.
Stopped bindings remain stopped across activation. Recovery stops authenticated
target processes, restores the previous pointer and running assignments, removes
only a newly constructed failed target, and marks the journal `rolled-back`.
A rollback target is retained even if its activation fails. Ambiguous process
identity leaves the journal incomplete rather than killing an unrelated process.

Initial activation uses the same journal with operation `setup` and previous
`null`. `begin_setup` records the requested binding, port and devices before
construction. Activation publishes `current` only after target health succeeds.
Interrupted setup stops its candidate, removes any target pointer and incomplete
tree, and retains the rolled-back journal; it does not invent a prior generation.
CLI setup also records `setupRegistration`: absolute `store` and `registry` paths,
prior project settings text in `project` and store-pointer text in `pointer`
(each nullable), and boolean `addedProject`, `addedVolume`, `hadProjects` and
`hadRegistry` flags. Recovery restores local metadata exactly, subtracts only its
own project reference, and removes its volume mapping only if it still names that
store and no projects remain. It locks registry, store and project in that order;
content replacements and removals are durable before the journal becomes terminal.
Cache directories and persistent OS-lock inodes are not deleted.

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
generation metadata, dependency locks or operation journals.

`gc [--plan | --yes] --json` collects obsolete generation trees, retaining current
and its most recent successful journal predecessor rather than the largest two
numbers. Live or ambiguous obsolete bindings refuse collection. The journal's
optional `collectedGenerations` array names distinct positive obsolete numbers;
`previous` and `target` both name current. Trees move durably into
`.garbage/<journal timestamp>/<number>` before the journal becomes complete.
Incomplete collection restores the trees; only complete collection trash is
deleted, with cleanup retried by later confirmed operations. Previously successful
generation numbers are never reused after collection. Project libraries remain
unchanged. A registered store also collects unreferenced digest-named bases,
engine caches, archive downloads and wheel directories, plus its disposable uv
cache. All extant generations in other registered projects retain their
references. Missing projects, damaged retained constraints or unsafe paths refuse
store deletion. Dependency references come from the target provisioner's
read-only `references` result (`wheelHashes`), never a supervisor pack parser.
Pack dependency-lock files are included in the generation inventory; missing
whole pack directories therefore refuse reference collection too.

Optional journal fields `collectionStore` and `collectedStorePaths` name the
absolute store and disposable relative entries (`bases/<sha256>`,
`engines/<sha256>`, `downloads/<sha256>`, `wheels/<sha256>`, `uv-cache`). They are
quarantined under `<store>/.garbage/<journal timestamp>/<relative entry>` before
the same completion write. `<store>/collection` holds the absolute owning journal
path plus newline until recovery/cleanup finishes, so another project can recover
the store before mutation. Registry, store and registered project locks protect
plan revalidation and quarantine. Missing references never imply abandonment.
