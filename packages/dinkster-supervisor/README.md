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
uv run dinkster-supervisor --help
uv run dinkster-station --help
```

Both console scripts are supplied directly by this package. The supervisor
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
- Catalog reads, compat prompt, queue mutations, assets, library, mounts,
  settings, memory, and cache routes go to the primary in slice 2a.
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
