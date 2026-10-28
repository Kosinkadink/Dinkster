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
never imports the engine's worker packages to perform these operations.

`environment_archives.extract_base(parts, destination)` reconstructs verified
split 7z parts in order and stages their contents beside the final base path.
The caller verifies part digests and holds the store writer lock. Extraction
accepts only `python/` and `wheels/`, rejects links, traversal, Windows device
names and case-ambiguous paths, and promotes the complete tree only on success.
It returns `destination/python`. Existing bases are never replaced; failed
reconstruction or extraction removes temporary archives and staging trees.
