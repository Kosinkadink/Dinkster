# dinkster-serve command reference

`dinkster-serve` runs the native protocol server. It binds a healthy zero-node
diagnostic host, then composes the installed default packs with their exact
artifact provenance, each configured pack, and an optional ComfyUI
compatibility surface. First-party packs may run in-process in the shared
Dinkster environment; each `--pack` runs in its own worker process by default.
The native generation provider and pinned ComfyUI import schemas need no
ComfyUI installation. Its execution interpreter must have Dinkster's torch
runtime installed; `--comfy-python` can select it separately from the host.
This is a complete reference for every command-line argument, grounded in
`src/dinkster/serve.py`.

## Server

### --host

Default: `127.0.0.1`. Bind address for the HTTP server.

Non-loopback binds require an existing inbound credential configuration:
`--auth`, or all three identity-token options. This fails closed before the
server binds. Static deployments should use the existing `auth.toml` Bearer
tokens documented in [auth.md](auth.md), not a separate server secret.

### --port

Default: `3639`. TCP port for the HTTP server.

### --allow-host

Syntax: `--allow-host HOST` (repeatable).

Requests are accepted only when the HTTP `Host` names the bind address, an
explicitly allowed host, or a loopback IP literal. Other hosts are rejected
before HTTP routing or WebSocket upgrade to prevent DNS rebinding. A wildcard
bind such as `0.0.0.0` therefore needs one `--allow-host` for each hostname or
address clients use. Ports in request `Host` headers do not affect matching.

### --allow-origin

Syntax: `--allow-origin ORIGIN` (repeatable; `*` allows any).

Default: none (no CORS headers).

Enables CORS for the given browser origin. State-changing requests carrying a
different browser `Origin` are rejected. Requests without `Origin` continue to
use the normal Bearer-authentication policy, and cross-origin reads receive no
CORS response headers unless their origin is allowed. Same-origin development
proxies continue to work when they preserve the browser's standard
`Sec-Fetch-Site: same-origin` header while rewriting the upstream `Host`.

## Packs

### Persisted schemas and lazy execution

Managed installs and installed defaults announce schemas, static choices, lazy
choice IDs, body-arm declarations, and extension contributions from their
persisted catalogs. `/api/composition` reports announcement or an actionable
composition failure; `/api/nodes` serves the schemas. Required catalogs are
validated before the API binds; a missing or stale catalog exits with the
`prepare-catalogs` instruction. Neither request starts a pack worker. Selecting a node's execution provider (or requesting a lazy
choice) activates its worker once, checks its live declarations against the
catalog, and uses the normal execution transport. Concurrent requests share
activation; a failed activation is not retried inside that serving session.
Cataloged type names, equivalences, and coercion declarations allow whole-graph
validation without importing unused packs. In-process custom codecs load when
demanded input/output types need them, including types supplied by another pack.
Disconnected targets, undemanded lazy inputs, and unexecuted region bodies do
not activate their type providers.

`dinkster-pack install` and updates refresh catalogs in the staged artifact store.
Refresh an existing install with `dinkster-pack --root PATH prepare-catalogs`; refresh the
default suite with `dinkster-pack prepare-catalogs --defaults --library-root PATH` using the
same runtime configuration as the server. This includes the native generation
schemas and the `dinkster-native` provider, plus the library's training schemas and
executor. Set `DINKSTER_COMFYUI_PYTHON` to the native execution interpreter
when it differs from the host. Use the same `--remote-catalog-base` and
`--remote-gateway-base` options as the server, or their environment variables,
when preparing remote nodes. Direct pack authors can use
`dinkster-doctor path/to/dinkster-pack.toml`. Import/schema failures invalidate the
catalog. `prepare-catalogs` executes trusted installed code and checks runtime
declarations; it does not replace full `doctor` checks for pack authors or
publication. Static authoring findings remain health diagnostics but do not
discard valid declarations; the existing install health gates still apply.
Restart the server after changing an installed generation or refreshing its
catalog. Explicit development `--pack` entries retain live discovery rather
than reusing a catalog prepared for another runtime configuration. Managed
installs never silently fall back to boot-time imports.

In-process packs defer imports too. An explicitly configured process group is
one activation unit: invoking one member starts that group's shared worker,
not unrelated groups. Replicas retain their existing placement and execution
contracts. Catalogs contain no PIDs, host paths, worker tokens, hardware
capabilities, or executable callables. Their relative-source digest permits
copying the pack to another host; attention evidence and transport/session
authority are negotiated live. Remote daemons retain their authenticated
boundary protocol, independently of the local installed-pack catalog.

The remote gateway's cold surface is the doctor-time snapshot. Rerun doctor
and restart to discover new remote nodes before using any remote node. On first
use, a dynamic pack can publish its live declarations without replacing the
newly started worker. A node whose schema changed refuses the stale invocation;
the same check runs before persistent-cache lookup. Refresh the graph and retry.
Active remote workers retain their catalog polling and live schema reloads.
Metadata probing stays in the doctor subprocess, not an unsandboxed host-side poller.

The boot regression budget is **10 seconds**, unchanged for 0, 10, 50, and
200 installed packs. `tools/benchmark_schema_catalog.py` measures from process
creation through a successful `/api/composition` response with every installed
pack announced, not merely an HTTP listener or a pending progress response.
It also checks every installed schema through `/api/nodes` and checks that no
pack import occurred and the serving interpreter has no child processes.
Before importing the CLI, a fresh-process import guard rejects the built-in
native execution module, compatibility entry point, and Torch runtimes. The same
guard covers startup with and without default packs, including an extra launcher.
The post-bind callback identifies that interpreter by PID, so a Windows
virtualenv redirector is not mistaken for a pack worker. Each test pack
contributes one real constant node; installation and doctor run before timing.
Each sample uses a new server process, local storage, no default suite,
no ComfyUI checkout, and no GPU.
JSON output pairs the timings with the host's 1/5/15-minute load averages
immediately before each sample (null on platforms without load averages).
The OS file cache is not flushed. Source validation scales with installed pack
bytes, and catalog parsing with serialized schema bytes. The fixed budget is a
bounded regression contract, not a claim that arbitrarily large packs or catalogs
take constant time.

```sh
uv run python tools/benchmark_schema_catalog.py --repeats 3
```

Measured on 2026-09-06 with Python 3.12.3, Linux 7.0.0-29, local storage,
and an AMD Ryzen 7 5800XT (8 cores). Three fresh processes per count;
OMP/OpenBLAS threads limited to one, empty GPU compute inventory, and no
concurrent test suite in this checkout.

| Installed packs | Median (s) | Maximum (s) | Pack workers started |
| --- | --- | --- | --- |
| 0 | 0.816 | 0.817 | 0 |
| 10 | 0.857 | 0.863 | 0 |
| 50 | 1.016 | 1.052 | 0 |
| 200 | 2.293 | 2.307 | 0 |

The 1-minute load averages before individual samples ranged from 1.080 to
1.391. Before the 200-pack run, `uptime` reported 1.07 / 1.21 / 1.46 for the
1/5/15-minute averages and NVIDIA's compute inventory was empty. The JSON
output retains each sample's three load averages beside its elapsed time.

### --no-default-packs

Default: off. Serve only the managed install root and explicitly configured packs
instead of adding the installed default suite. This also permits an empty
diagnostic installation and isolates installed-pack-count boot measurements.

### --pack

Syntax: `--pack PATH` (repeatable).

Default: none.

A `dinkster-pack.toml` or its directory. Each pack runs isolated in its
own process. Served nodes appear under `/api/nodes` and are tracked on
`/api/composition`. The foundation and media-I/O packs come from the installed
default set and require no flags.

Installed standard vision packs use their declared software dependencies.
Prepare their catalogs with `dinkster-pack prepare-catalogs --defaults` before serving.
`DINKSTER_SERVING_PYTHON` is an advanced override that runs
those packs on the named interpreter instead; that interpreter must already
satisfy their manifests.

`/api/health` remains an HTTP 200 liveness endpoint. Its `ok` field is false
when zero node types composed or any pack failed, and `compositionState` reports
the composed and failed counts plus the current composition epoch.

### --install-root

Syntax: `--install-root PATH`.

Default: `$DINKSTER_INSTALL_ROOT` (empty if unset).

Managed install root: serve the packs of its current generation from
the immutable store. Explicit `--pack` entries are additions on top;
name collisions fail composition loudly.

Populate the same root with `dinkster-pack`. An install first prints the exact
pack versions, artifact digests, provenance, and environment work it would
apply, then requires explicit confirmation:

```console
dinkster-pack --root /path/to/install-root install pack@version
dinkster-pack --root /path/to/install-root install ./path/to/pack
dinkster-pack --root /path/to/install-root install 'git+https://example.com/pack.git@ref'
```

Use `--yes` for a reviewed noninteractive install, or add `--plan plan.json`
and later run `dinkster-pack --root /path/to/install-root apply plan.json`.
`DINKSTER_INSTALL_ROOT` can replace `--root`; pass that root to `dinkster-serve` or
set the same environment variable when serving.

### --comfy-root

Syntax: `--comfy-root PATH`.

Default: `$DINKSTER_COMFYUI_ROOT` (empty if unset).

ComfyUI install directory. Adds the translated core surface as the
`comfy` pack, served on that install's interpreter. Also derives
conventional `comfy-input` and `comfy-output` mounts unless the operator
already configured those ids. With `--library-root`, it also probes
ComfyUI's initialized `folder_paths` table and derives one read-only mount
per ordered root of every approved model-file category. Those mounts use
collision-safe `comfy-model-<category>-<index>` ids and semantic
`model/...` metadata; their real host paths stay off the API surface.

Without this option, the provider uses native implementations.
`/api/compat/comfy/schemas` lists pinned source schemas as importable metadata,
not executable IDs. Supported API prompts submitted to `/api/compat/comfy/prompt`
are lowered to canonical `dinkster.*` nodes before planning. The single-job
multi-GPU options also work without a ComfyUI install. Legacy quarantine
still requires this option.

### --comfy-python

Syntax: `--comfy-python PATH`.

Default: `$DINKSTER_COMFYUI_PYTHON`, then `<comfy-root>/venv/bin/python`,
then the serving process's interpreter.

Interpreter for native or compat execution. The resolution chain is: CLI value,
then the `DINKSTER_COMFYUI_PYTHON` environment variable, then the install's
own venv when `--comfy-root` is set, then the current Python.

### CPU and Apple Silicon workers

Workers select from their own PyTorch capabilities: CUDA/ROCm, XPU, MPS,
then CPU. A GPU-less worker automatically enables ComfyUI's CPU memory
policy before importing its nodes. The server interpreter remains torch-free.

To force CPU execution, including on a GPU-equipped machine:

```sh
DINKSTER_ACCELERATOR=cpu dinkster-serve --comfy-root /path/to/ComfyUI \
  --comfy-python /path/to/ComfyUI/.venv/bin/python
```

On Apple Silicon, install native arm64 PyTorch in the worker environment.
MPS is selected automatically when available; `DINKSTER_ACCELERATOR=mps`
requires it and refuses an unavailable backend rather than silently using CPU.
`scripts/setup_envs.sh` prepares the repository's macOS test environments and
runs the MPS capability smoke check without a system-wide install.

`DINKSTER_ACCELERATOR` applies to the process and its children, including native
inference and compatibility workers. Set it on each remote worker service or
cloud worker independently; the engine's hardware does not select a remote
worker's device. Device selection does not change shared-memory or network
value transport. Restart workers to change the selection. Raw `--cpu` remains
denied in `worker-comfy-args`; use the supported accelerator selection instead.

### --comfy-arg

Syntax: `--comfy-arg=ARG` (repeatable), or bare `--comfy-arg` to clear.

Default: the persisted `worker-comfy-args` setting, else no arguments.

Adds one ComfyUI startup argument to every compat worker. Use the equals
form when the argument begins with `--`, and repeat the option for values:

```
--comfy-arg=--preview-method --comfy-arg=auto
```

Explicit CLI occurrences replace the persisted array; they do not append
to it. A bare occurrence explicitly replaces a persisted array with the
empty tuple and cannot be combined with argument values. The complete
ordered array travels only through child argv. Dinkster
rejects server-owned and memory/aimdo-owned flags listed in
[settings-api.md](settings-api.md); every other argument is validated by
the installed ComfyUI parser when the worker starts.

### --legacy-pack

Syntax: `--legacy-pack PATH` (repeatable).

Default: none.

Unmodified ComfyUI custom node pack (directory or single `.py`). Loads
in the legacy quarantine worker and attributes as `comfy.<pack>`.
Requires `--comfy-root` (or `$DINKSTER_COMFYUI_ROOT`).

### --strict-packs

Default: off (flag).

A pack that fails to compose aborts the whole process. Without this
flag, failures are recorded on `/api/composition` with a `pack_failed`
event, and every pack that did load serves. Remote workers follow the
same rule: a daemon that is down or misconfigured is recorded (or, with
this flag, fatal) and everything else serves.

### --remote-workers

Syntax: `--remote-workers PATH`.

Default: none on the CLI; when `--library-root` is set,
`<library-root>/remotes.toml` loads if present (a missing file means no
remotes - the `memory.toml` convention).

A `remotes.toml` naming `dinkster_workers.service` daemons to compose at
startup, after the packs. Each daemon's announced node types join the
serving surface attributed to the worker's name, and invocations of
those types execute on the daemon. One table per remote:

```toml
[worker.upscale-box]
endpoint = "192.168.1.53:5151"
token_file = "/home/op/.config/dinkster/upscale-box.token"
# nodes = ["esrgan.upscale"]   # optional; default: everything announced
# trust_reserved = true        # optional; default false

[worker.upscale-box.memory]
ram = "24G"
"vram:cuda:0" = "20G"
```

The table name is the remote's identity: its pack id on the composed
surface and the `@<name>` qualifier on every device fact it reports.
`nodes` restricts composition to the listed announced types; naming a
type the daemon does not serve refuses. `trust_reserved` allows node
types under reserved namespace roots, the same host-vouches escape
hatch as pack composition. `[worker.NAME.memory]` budgets use the
remote's own unqualified device keys (sizes as in `memory.toml`); the
host qualifies them to `ram@NAME` / `vram:cuda:0@NAME` before the
governor sees them, and explicit `--memory-budget` entries win on
collision. See [remote-workers.md](remote-workers.md) for daemon-side
setup and the security caveat.

`GET /api/workers` reports `local` and every configured remote with its
known status, routed node types, and device qualifier. A remote may serve
the same node type as a local pack when their schemas match. In that case,
`POST /api/jobs` can select it with the optional top-level placement map:

```json
{"placement": {"upscale": "upscale-box"}}
```

Placement keys are top-level graph node ids. A region id applies to every
node in its body, including nested regions. The server rejects unknown
workers, unknown or non-top-level node ids, and workers that cannot execute
the selected node type. See [remote-workers.md](remote-workers.md) for the
complete submission contract.

When `--library-root` is set, remote workers share a persistent value
store at `<library-root>/value-store`: bulk boundary values land there
once and later runs send digest references instead of bytes, when the
daemon also runs with a `--value-store` (see
[remote-workers.md](remote-workers.md), "Persistent value transport").

### --advertise-assets

Syntax: `--advertise-assets URL`.

Default: none.

Asset base URL remote worker daemons pull declared assets from when the
engine stages them before dispatch (bytes at `URL/assets/{digest}`;
advertise this server's API base, e.g. `http://HOST:PORT/api`). Offered
to daemons as the first staging source, ahead of any remote URLs the
declaration itself carries. Without it, daemons can stage only from
declared URLs. See [remote-workers.md](remote-workers.md) for the
staging flow.

### --advertise-assets-token-file

Syntax: `--advertise-assets-token-file PATH`.

Default: none. Requires `--advertise-assets`.

File containing the bearer token daemons present at the advertised
asset endpoint: a static `auth.toml` token granting `assets:read`.
Required when `--auth` is set and `--advertise-assets` points at this
server. The credential rides only requests to the advertised endpoint,
never URLs a declaration carries.

## Pack sandbox

### --sandbox-packs

Default: off (flag). Linux only.

Runs isolated pack workers in a fail-closed bubblewrap jail. In-process
first-party packs are not affected. Each pack's `[pack.sandbox]` table
requests GPU, network, and writable-mount access, but the declaration grants
nothing by itself. Writable access is limited to mounts the operator already
configured as `readwrite`; all other configured mounts remain read-only.
Every jailed worker has an unshared network namespace, including workers with
granted egress. Allowed HTTPS connections cross a per-worker Unix-socket proxy;
the worker never receives host or raw network access.
`--allow-mount-changes` cannot be combined with this flag because a running
mount namespace cannot add or revoke bind grants.

### --sandbox-grant-gpu

Syntax: `--sandbox-grant-gpu PACK` (repeatable). Requires `--sandbox-packs`.

Allows the named pack to receive GPU access when its manifest also requests
`gpu = true`. Pack names are canonicalized, so `_` and `.` are equivalent to
`-`. A host grant without the matching manifest request gives no GPU access.

### --sandbox-grant-network

Syntax: `--sandbox-grant-network PACK=HTTPS_ORIGIN` (repeatable). Requires
`--sandbox-packs`.

Allows the named pack to reach one exact HTTPS origin when its manifest also
requests `network = true`. Repeat the option for every origin, including the
origins used by signed upload or download URLs. Origins cannot contain
credentials, paths, queries, or fragments. DNS resolution happens in the host
proxy, which refuses private, loopback, link-local, reserved, and otherwise
non-global addresses before connecting to a validated address. TLS remains
end-to-end between the worker client and the destination through HTTP CONNECT.
Packs in one worker group share the union of origins granted to its members.

For the installed partner pack, `--comfy-api-base` automatically grants that
URL's exact origin. Any separate signed-transfer origins still require explicit
`dinkster-nodes-partner=HTTPS_ORIGIN` grants.

## Memory and aimdo

### --multi-gpu-devices

Syntax: `--multi-gpu-devices INDEX,INDEX[,INDEX...]`.

Default: disabled.

Runs concurrent native model jobs across one isolated compat worker per
listed CUDA device. The ordered indices are relative to the server's visible
CUDA devices, so `CUDA_VISIBLE_DEVICES` can exclude reserved physical GPUs.
At least two unique non-negative indices are required; three or more devices
are supported. Unless `--max-running-jobs` is explicit, the queue concurrency
is raised to at least the replica count. Each job remains on one replica for
its full execution. This improves aggregate throughput for concurrent jobs;
one job still runs at single-GPU speed. It does not split CFG/conditioning,
pool VRAM, or shard one model across GPUs.

### --single-job-multi-gpu-devices / --single-job-multi-gpu-mode

`--single-job-multi-gpu-devices INDEX,INDEX[,INDEX...]` fixes two or more
ordered logical CUDA ranks for each single-job workgroup. Indices are relative
to `CUDA_VISIBLE_DEVICES`; Dinkster does not discover or add devices. It is
mutually exclusive with `--multi-gpu-devices`.

The mode is `auto`, `guidance`, `sequence`, or `window` (default `auto`).
These modes distribute one sampling job; they are not whole-job replicas and
do not guarantee lower latency:

- `guidance` assigns model-evaluated guidance lanes to separate full-model
  ranks and gathers each denoiser evaluation. It requires at least conditional
  and unconditional model lanes.
- `sequence` uses every selected rank for pure Ulysses sequence parallelism
  (`U=rank count`, `R=1`, guidance degree 1). It admits only an exact registered
  sequence receipt and is never selected by `auto`.
- `window` scatters a windowed Flux plan's per-step window evaluations across
  full-model ranks with a bit-exact merge. It refuses when the plan has fewer
  than two joint windows or no registered window receipt, and never falls back
  to another topology.
- `auto` uses receipted guidance for H3 and SD1.5, or window scattering for
  eligible Flux windowed plans. It never selects sequence parallelism. It
  fails before sampling when none is eligible rather than running
  duplicate single-device work on every rank.

Guidance mode currently supports SD1.5 FP16 and registered MiniMax H3 BF16 and
INT8 ConvRot FL2VA plans on two SM 8.9 Ada or two SM 12.0 Blackwell GPUs with
builtin samplers and schedulers. Other model, dtype, rank-count, and architecture combinations
refuse until they have their own receipt. Sequence mode currently supports
MiniMax H3 FL2VA BF16 with built-in SDPA under torch 2.13.0+cu130 on two
SM 12.0 Blackwell GPUs. Separate receipts also admit explicitly selected
comfy-kitchen 0.2.31 and 0.2.32 INT8 attention under that torch runtime.
Other provider versions and runtimes refuse.

Every rank is a separate process with one selected GPU and its own model
residency. Rank 0 owns progress and the final result. A rank failure or
cancellation terminates the workgroup because a timed-out NCCL communicator
cannot be reused. Mode and device selection are fixed for the server lifetime;
they are not currently workflow widgets or per-request settings.

### --memory-budget

Syntax: `--memory-budget DEVICE=SIZE` (repeatable).

Default: none on the CLI; persisted defaults may come from
`<library-root>/memory.toml`.

Declared memory budget for one residency class. `DEVICE` is a
residency-class key (e.g. `ram`, `vram:cuda:0`). `SIZE` is a byte count
with an optional binary suffix `K`/`M`/`G`/`T` (case-insensitive,
powers of 1024). Example:

```
--memory-budget ram=24G --memory-budget vram:cuda:0=20G
```

Budgeted devices gate worker reservations through the memory governor and cap
classic or AIMDO model residency in compat workers. Governor admission changes
live; residency receives the current budget when a worker starts or reloads.
`GET /memory/status` shows both values and whether a worker restart is pending.
Unbudgeted devices admit and place freely. Each CLI entry overrides the
persisted value for that device while other persisted devices remain in effect.

When `--library-root` is set, persisted defaults load from
`<library-root>/memory.toml`:

```toml
[budgets]
ram = "24G"
"vram:cuda:0" = 21474836480
```

String sizes are byte counts with optional binary `K`/`M`/`G`/`T`
suffixes; integers are raw byte counts. Zero means "admit nothing and use no
model residency capacity on this device"; negative values refuse. A budget
above a measured device total is accepted for WDDM/shared-memory setups and is
otherwise ineffective. A missing `memory.toml` is fine.

### --aimdo

Syntax: `--aimdo {auto,on,off}`.

Default: `auto`.

Weight residency mechanism for native execution. `auto` enables comfy-aimdo
partial weight offload on supported NVIDIA CUDA workers on Linux and Windows.
`on` requests comfy-aimdo wherever the capability chain passes. If Aimdo is
unavailable, selection reports the failed capability and explicitly falls back
to eager residency. After Aimdo is selected, component construction failures
fail the load without changing mechanisms or caching an eager replacement.
Temporary per-operation materialization remains Aimdo. `off` disables dynamic
residency.

### --fp8-matmul

Default: the persisted `fp8-matmul` setting, else off (flag).

Opts native checkpoint loading into fp8 matrix multiplication, analogous to
upstream ComfyUI's `--fast fp8_matrix_mult` feature. The flag records a
request, not a server-side capability guess: the torch-bearing worker checks
its device, compute capability, and torch version before loading and refuses
an unsupported request without silently falling back. Explicit CLI use wins
over a persisted false value. The policy is folded into every native
checkpoint identity, including checkpoints without fp8 storage.

### --diffusion-dtype, --text-encoder-dtype, --vae-dtype

Syntax: each accepts `auto`, `float16`, `bfloat16`, or `float32`.

Default: the corresponding persisted `dtype-policy` value, else `auto`.

The three selectors are independent. Explicit CLI values override only their
component in persisted policy. Native runtime identity records the effective
component dtypes, while compatibility workers receive ComfyUI's equivalent
UNet, text-encoder, and VAE flags.

### --reserve-vram

Syntax: `--reserve-vram BYTES` (binary `K`/`M`/`G`/`T` suffix accepted).

Default: `256M` (268435456 bytes).

Physical accelerator headroom shared by classic and AIMDO residency. Dinkster
deliberately unifies on this visible server default instead of ComfyUI's hidden
400 MiB default (600-700 MiB on Windows). WDDM operators who need the larger
cushion can raise this one setting.
Both mechanisms also retain Dinkster's 0.8 GiB inference working reserve. AIMDO
includes that reserve in its process-wide headroom so demand-paged weights do
not consume memory needed by activations and kernel workspaces.
When the `memory-headroom` settings category is writable, a successful
runtime edit applies live to every policy worker and is also used for workers
spawned later. AIMDO additionally protects transient governor reservations;
classic residency does not count that transient reserve.

### --max-running-jobs

Syntax: `--max-running-jobs N` (integer).

Default: `1`.

Concurrent jobs. Engine admission keeps hardware safe regardless of
this value; the limit controls how many jobs run at once.

## Dev and benchmark

### --dev

Default: off (flag).

Dev-mode diagnostics and affordances: `cache_miss` events explaining
why a node recomputed, per-invocation boundary cost logging on
`dinkster.dev.boundary`, and pack hot reload/removal:

- `POST /api/packs/{packId}/reload` -- restart that pack's worker and
  swap its nodes on the live surface.
- `DELETE /api/packs/{packId}` -- retract the pack's nodes and stop its
  worker.

### --watch-packs

Default: off (flag).

Hot-reload node packs on source changes. Polls each composed pack's
source directory; when files change and settle, restarts that pack's
worker and swaps its nodes. Requires `--dev` (same swap and failure
semantics as `POST /api/packs/{packId}/reload`).

### --benchmark

Default: off (flag).

Write one benchmark record per job: per-occurrence timings, cache
hit/miss attribution, boundary costs, and a hardware sample timeline
(psutil/NVML when installed). Implies cache-miss explanations.

### --benchmark-dir

Syntax: `--benchmark-dir PATH`.

Default: `./benchmarks`.

Directory for benchmark records written by `--benchmark`.

### --benchmark-interval

Syntax: `--benchmark-interval SECONDS` (float).

Default: `0.5`.

Hardware sampling interval for `--benchmark`.

## Persistence and security

### --auth

Syntax: `--auth PATH`.

Default: off.

Requires inbound Bearer authentication using a strict static-token TOML file.
`<library-root>/auth.toml` is the conventional location, but the file is loaded
only when explicitly named. Invalid files refuse startup. Tokens never enter
runtime settings, API responses, or logs. See [auth.md](auth.md) for the file
format, capability route matrix, and 401/403 contracts.

### --identity-jwks-url, --identity-issuer, --identity-audience

Syntax: all three options must be present together:

```
--identity-jwks-url https://identity.example/.well-known/jwks.json \
--identity-issuer https://identity.example \
--identity-audience dinkster-session
```

Defaults: `$DINKSTER_IDENTITY_JWKS_URL`, `$DINKSTER_IDENTITY_ISSUER`, and
`$DINKSTER_IDENTITY_AUDIENCE`, respectively. If one value is configured, all
three are required. This enables short-lived identity-service JWTs as inbound
Bearer credentials. When combined with `--auth`, static credentials are tried
first and identity-service JWTs second. See [auth.md](auth.md) for token minting,
grant handling, and JWKS cache behavior.

### --federated-assets-store, --federated-assets-policy, --federated-assets-cursor-key

Syntax: all three options must be present together:

```
--federated-assets-store PATH \
--federated-assets-policy PATH \
--federated-assets-cursor-key PATH
```

Default: all absent. Ordinary serving registers no federated catalog routes.

The store is the dedicated resolution SQLite database. The policy is strict
version-1 TOML with exactly `version` and `scopes` at the top level:

```toml
version = 1

[scopes]
local = []
workspace = ["provider-a", "organization/provider-b"]
```

Scopes are nonempty whitespace-free names. Each value is an array of unique
canonical provider ids using lowercase letters, digits, `.`, `_`, `/`, and
`-`. Empty arrays are valid. Without `--auth`, the policy must include `local`;
with `--auth`, existing Bearer grants select scopes and require `assets:read`.
Only scopes present in this policy are served, including local-only scopes with
an empty provider array.

The cursor-key file is read as exact raw bytes and must contain at least 32
bytes. Its contents are never logged or returned. Invalid, unreadable, partial,
or malformed configuration refuses startup before the server binds.

This configuration registers only read-only `POST /api/catalog` and
`POST /api/catalog/candidates`. Trusted acquisition sources are empty, so the
surface does not acquire or publish bytes. `/api/catalog/resolve` is not
registered. Route paths and acquisition sources are not operator-configurable.
The process owns one resolution store and closes it during application cleanup.

### --library-root

Syntax: `--library-root PATH`. Pass `''` (empty string) to disable.

Default: `$DINKSTER_LIBRARY_ROOT`, else `./dinkster-library`.

Persistence root. When set, the following live under it:

| Path | Contents |
|------|----------|
| `vault/` | Content-addressed uploaded bytes |
| `library.sqlite` | Scoped library records |
| `history.sqlite` | Persistent execution history |
| `provenance.json` | Acquisition leads |
| `resolver-indexes.json` | Resolver-index subscriptions and cached documents |
| `mounts.toml` | Durable filesystem mount grants |
| `execution-cache/` | Layered execution-result manifests and content-addressed payloads |
| `value-store/` | Persistent value store for remote workers (see [remote-workers.md](remote-workers.md)) |
| `venvs/<accelerator>/<pack-digest>/<pack>/` | Standard vision-pack runtime environments provisioned before workers announce |
| `scratch/packs/<pack>/` | Persistent scratch assigned to one solo pack |
| `scratch/groups/<group>/` | Persistent scratch shared by one worker group |
| `memory.toml` | Persisted memory budget defaults |
| `settings.json` | Atomically persisted runtime settings edits |
| `auth.toml` | Conventional operator-managed static Bearer credentials (loaded only via `--auth`) |

When disabled (`''`), library persistence is not mounted, the execution cache
defaults to memory-only, and `--allow-mount-changes` is rejected. An explicit
`--execution-cache-dir` can retain execution results independently. Standard
vision runtime environments remain cached under the platform user-state
directory: `%LOCALAPPDATA%/dinkster/venvs` on Windows,
`~/Library/Application Support/dinkster/venvs` on macOS, and
`$XDG_STATE_HOME/dinkster/venvs` (default `~/.local/state/dinkster/venvs`) on other
systems. Isolated workers receive their exact scratch path in
`DINKSTER_PACK_SCRATCH`. Scratch survives worker restarts and upgrades until the
operator removes it. Moving a pack into or out of a worker group changes its
scratch path, and every member of a group can read and write the group's shared
scratch. With `--sandbox-packs`, only that process's scratch directory is
writable, sibling scratch directories are absent from its jail, and `HOME` and
`TMPDIR` point at its private temporary `/tmp`.

Runtime settings are documented in [settings-api.md](settings-api.md).

### --execution-cache-mode, --execution-cache-memory-entries, --execution-cache-dir, --execution-cache-disk-budget

The standard server defaults to `--execution-cache-mode layered` when a
persistent directory is available. The memory LRU fronts a disk store at
`<library-root>/execution-cache`; a disk hit is promoted so the next lookup in
the process is a memory hit. Pass `--execution-cache-mode memory` to keep the
process-local behavior. With `--library-root ''`, the default is memory unless
`--execution-cache-dir PATH` supplies an independent persistent root.

`--execution-cache-memory-entries N` bounds the memory layer by entry count
(default `1024`). `--execution-cache-disk-budget BYTES` bounds manifests plus
content-addressed payloads (default `10G`; binary `K`/`M`/`G`/`T` suffixes are
accepted). Disk eviction is least-recently-used by last hit. Shared payloads
remain until their last manifest is evicted, and interrupted or orphaned files
are cleaned under a root-wide lock. Outputs tied to live process resources or
without persistable bytes remain memory-only.

`node_cached` event details and benchmark occurrence records include
`cacheLayer: "memory"` or `cacheLayer: "disk"` for layered-cache hits.

### --resolver-region

Syntax: `--resolver-region REGION`.

Default: `$DINKSTER_RESOLVER_REGION`, else no region preference.

Prefer matching regional resolver-index URLs before each entry's default
URLs. Region names contain 1-32 lowercase letters, digits, or hyphens.

With a library root, resolver indexes are managed through
`GET/POST /api/assets/resolver-indexes`,
`PATCH /api/assets/resolver-indexes/{subscriptionId}`,
`DELETE /api/assets/resolver-indexes/{subscriptionId}`, and
`POST /api/assets/resolver-indexes/refresh`. Mutations require
`assets:write`; listing requires `assets:read`. The guess endpoint includes
exact, case-sensitive basename suggestions from subscribed indexes and never
downloads a suggestion.

The PATCH body is
`{"trustedForP2P":boolean,"licenseAuthoritative":boolean}`. Both flags default
false, and setting them requires the host's `p2p` settings grant. A subscription
with `trustedForP2P` enabled can authorize its resolver `p2p` entries;
`licenseAuthoritative` identifies metadata authority, not transfer eligibility.
Provider names and URLs never imply either flag.

Hosted subscription sources require HTTPS. Loopback HTTP is accepted for
local development.

### --official-resolver-url / --official-resolver-provider-id

Syntax: `--official-resolver-url URL --official-resolver-provider-id ID`.
Environment alternatives: `DINKSTER_OFFICIAL_RESOLVER_URL` and
`DINKSTER_OFFICIAL_RESOLVER_PROVIDER_ID`. CLI values take precedence. Neither has
a built-in default; both are required for bootstrap, with a library root.
The operator supplies the official provider's private export URL and stable ID;
the export's `name` must match that ID exactly. HTTPS is required except for
loopback HTTP fixtures. URLs must be ASCII: percent-encode non-ASCII paths and
queries, and use IDNA hostnames. Requests send no authorization or cookies.

Startup creates the subscription with both `trustedForP2P` and
`licenseAuthoritative` true only after fetching a complete, identity-matching
export. Missing configuration, invalid identity, or fetch failure emits an
`official resolver bootstrap refused` diagnostic and leaves ordinary serving
available. Provider deployment still requires live allowlist, canonical
descriptor, full-enumeration, and omission/tombstone validation.

Bootstrap is recorded once in `resolver-indexes.json`. An existing matching
subscription retains its trust flags. Later trust changes and unsubscribe
survive restart; bootstrap never regrants trust or recreates a removed
subscription. A different configured URL/ID pair is refused rather than
silently replacing the recorded choice. Refresh rejects provider identity
changes without renewing cached authority. Other subscriptions still default
both flags false. Bootstrap never changes P2P settings, saved opt-outs,
`--disable-p2p`, or settings-write permissions.

### --disable-p2p

Disable peer-to-peer downloads and background seeding at startup. Fresh settings
otherwise enable both on LAN and Internet; existing saved opt-outs are preserved.
This override does not modify saved settings or grant settings-write access.

### --allow-settings-changes

Syntax: `--allow-settings-changes [CATEGORY]` (repeatable).

Default: none. Runtime settings remain readable, but no category is
writable.

Grant runtime mutation for one category. The v1 categories are
`memory-budgets`, `memory-headroom`, `aimdo-policy`, `dtype-policy`,
`fp8-matmul`, `worker-comfy-args`, `jobs`, `logging`, and `p2p`. Repeat the
flag to grant a union:

```
--allow-settings-changes=memory-budgets --allow-settings-changes=jobs
```

A bare occurrence or the value `all` grants every category:

```
--allow-settings-changes
--allow-settings-changes=all
```

Unknown categories are startup errors.

### --allow-mount-changes

Default: off (flag).

Enables `POST /api/mounts` and `DELETE /api/mounts`: grant and revoke
filesystem mounts while running (the desktop-shell folder-picker
flow). Granted mounts persist to `<library-root>/mounts.toml`. Off by
default: runtime mount mutation is a filesystem capability grant and
stays operator-only. Requires `--library-root`.

## Logging

### --log-level

Syntax: `--log-level {debug,info,warning,error,critical}`.

Default: `info`.

Verbosity for the whole dinkster tree. Worker subprocesses inherit the
resolved level via the `DINKSTER_LOG_LEVEL` environment variable.

### --log

Syntax: `--log NAME=LEVEL` (repeatable).

Default: none.

Per-origin override. Example: `--log dinkster.pack.mypack=debug`.
Overrides are exported to worker subprocesses via the
`DINKSTER_LOG_OVERRIDES` environment variable as `name=level` pairs.
