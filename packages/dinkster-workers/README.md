# dinkster-workers

`dinkster-workers` supplies the local, isolated-process, remote, routing,
placement, governance, and sandbox implementations of the `Worker`
protocol from `dinkster-protocol`. It depends on that protocol package plus
schema, values, memory, and assets - never on the engine itself, so a
worker child interpreter carries no scheduler; the server composes these
workers and node packs run behind them.

## Setup

This package is a uv workspace member and is not published separately yet.
From the repository root, install the complete workspace:

```sh
uv sync --all-packages
```

The package installs `dinkster-doctor`, the pack linter and publish gate:

```sh
uv run dinkster-doctor path/to/pack
uv run dinkster-doctor --json path/to/pack
uv run dinkster-doctor --sandbox path/to/pack
```

Exit code 0 means healthy, 1 means unhealthy, and 2 means a usage error.
`--json` emits the versioned machine report with `reportVersion`,
`apiVersion`, and `doctorVersion`.

`--sandbox` runs the import probe - the one doctor stage that executes pack
code - inside a network-less bubblewrap jail with CPU and file-size limits:
the pack bytes are read-only, the only writable surface is a private tmpfs
`/tmp`, and the network is always unshared. It requires Linux with a usable
bwrap (`detect_bubblewrap` names the distro-specific fix when the kernel
posture blocks it); a requested sandbox that cannot be built is a loud exit
2, never a silent unjailed probe. Findings are identical jailed or unjailed -
the jail changes what a hostile import can do, not what the report means.

## Use

Run doctor against the pack directory containing `dinkster-pack.toml` before
publishing it. The human report gives every finding a concrete fix; use the
JSON form in CI and other tooling.

Registry and admission tooling can call `diagnose_static(pack_root, manifest)`
to run the versioned manifest and extracted-tree checks without importing or
executing pack code. `diagnose()` adds the separate import probe, schema, and
namespace findings used by the full doctor command.

Import/schema-valid doctor probes atomically persist portable declarations in the
pack's `__pycache__/dinkster-schema-catalog/<manifest-filename>.json`. The catalog
is disposable data, excluded from pack artifacts, and keyed by the selected
manifest filename and source content rather than host paths. Manifest variants
in one directory have separate catalogs. Import/schema failures remove the
selected catalog; static authoring findings do not discard valid declarations.
Installed-pack composition uses it without importing the pack; invocation
validates the live announcement before cache lookup or execution. Type names,
equivalences, and decoder/merge targets are validation metadata, not callables;
host codecs load only when demanded types need them. Refresh the catalog after
changing pack code or its runtime dependencies. `dinkster-serve` validates required
installed catalogs before binding its API and exits with the `prepare-catalogs`
instruction when one is missing or stale. Worker tokens, attention capabilities,
shared-memory handles, and other process authority are never persisted there.

The Python exports also include `InProcessWorker`, `IsolatedWorker`,
`RemoteWorker`, `RoutingWorker`, `PlacementWorker`, manifest loading,
provisioning, boundary diagnostics, and sandbox launchers. Those are
composition surfaces for Dinkster hosts; pack authors normally use the doctor
CLI rather than importing worker machinery.

Remote services retain admitted invocations in memory across a physical
connection loss. The same engine process can reconnect to the same daemon
process for `--resume-grace` seconds (default 120). Running work is rebound and
completed results replay until acknowledged. Cancellation intent survives the
gap. Progress and preview events while disconnected are deliberately lossy;
only the latest event without binary data is retained. A missing or expired
record fails rather than replaying the invocation, choosing another worker, or
claiming recovery across a daemon restart.

## Serving sandbox

An isolated process and virtual environment provide dependency and crash
isolation, not a host security boundary. Host security requires the separate
serving sandbox:

| Platform | `dinkster-serve --sandbox-packs` | `dinkster-doctor --sandbox` |
| --- | --- | --- |
| Linux | Full bubblewrap user, PID, mount, and network namespaces for local isolated packs | Bubblewrap import-probe jail |
| Windows | Unavailable; isolated workers keep the user's filesystem and network authority | Unavailable |
| macOS | Unavailable; isolated workers keep the user's filesystem and network authority | Unavailable |

The Linux serving sandbox is opt-in and fail-closed. Missing bubblewrap,
blocked user namespaces, a mount-only degraded capability, or an unsafe bind
policy refuses the pack launch instead of falling back to an ordinary child.
It does not cover in-process packs or remote worker daemons.

The jail exposes only the worker's interpreter, pack, required Python import
roots, a read-only system surface, and explicit policy binds. It provides a
private `/tmp` plus fresh `/proc` and `/dev`; points `HOME` and `TMPDIR` at
`/tmp`; and starts the worker from an empty environment populated only by the
fixed and policy allowlists plus launch-specific host values delivered through
a private one-use file descriptor. It protects library, install, and configured
secret paths and limits per-process file size and process count. Workers using
shared-memory value transport receive the host `/dev/shm`, and a GPU grant adds
the required host device entries. The whole asset vault and configured read
mounts are visible read-only. When a library root is configured, a pack or
worker group receives one persistent writable scratch directory. A configured
writable mount is writable only when both the manifest and host policy allow
it.

GPU devices and external origins also require both a manifest request and a
host grant. The network namespace is always unshared. One per-worker
Unix-socket CONNECT proxy carries all allowed exact HTTPS origins. Each proxy
request resolves the destination on the host and rejects private, loopback,
link-local, reserved, multicast, unspecified, and otherwise non-global
answers. CONNECT then opens one validated numeric answer without another DNS
lookup. The proxy authorizes the destination host and port but does not inspect
tunneled bytes or make the destination trustworthy.

This boundary is not a VM. A GPU grant exposes the host driver, writable
mounts expose their contents, and every jailed pack can read the whole local
asset vault when one is configured. Shared-memory workers see host
`/dev/shm`. Worker-group members share scratch and the union of member grants,
so group membership is a trust boundary. Kernel and driver exploits,
malicious outputs, and memory or VRAM exhaustion remain outside this boundary.
There is currently no Landlock fallback, Windows low-integrity or AppContainer
rung, or macOS Seatbelt rung.

A pack may declare an optional `[pack.entry] choices = "module:attr"`
callable returning combo choice lists (choice-list id -> sequence of value
strings or a zero-argument callable, ids namespaced under the pack's claims
like node types). The worker host enumerates the mapping once at startup,
validates and deduplicates static values, and announces static lists in the
hello's `choices` and callable-backed ids in `lazyChoiceIds`; the composed
server serves both at `/api/choices/{id}` for remote COMBO widgets. A lazy
provider is never invoked at composition or startup: the server sends the
worker a fetch request only when the route is fetched or refreshed, invokes
the provider once per request off the worker event loop, and validates the
result with the same grammar and bounds as static lists. Choice lists are UI
vocabulary, never schema or execution identity.

A compat pack may declare an optional
`[pack.entry] skips = "module:attr"` callable returning classified
translation refusals (source node name -> reason). The worker snapshots the
mapping once at startup beside choices; the composed server attributes the
entries to packs and exposes them through `/api/diagnostics`. Skips are
advisory diagnostics, never schema or execution identity.

A native pack may ship `comfy-aliases.json` beside `dinkster-pack.toml`. The file
uses the `dinkster-comfy-alias/1` envelope and must be included beside the
manifest in the built wheel. Manifest loading reads it as bounded, strict JSON
without importing pack code. Worker startup verifies that every carrier is an
owned native schema and validates the shared replacement references. The data
is import metadata only; it never adds executable nodes or native schema
replacements.

A native pack may also ship `comfy-groups.json` beside the manifest. The
`dinkster-comfy-group/1` envelope declares exact foreign subgraphs, their
import-only source and collapsed schemas, and maintained replacement rules.
It uses the same bounded strict-JSON loading and carrier ownership checks.
Source and collapsed schemas never become executable or searchable nodes.

## Benchmark memory evidence

`backend_env.validate_benchmark_report` checks benchmark identity, correctness
and actual residency routes. Passing it is not proof of driver-spill avoidance.
The shared-usage samples and their warm-to-final difference are observations;
`shared_spill_detected: null` means unassessed, including when usage is stable.
Pinned staging and driver spill both contribute to shared usage. Attribute them
with allocation-owner lifetimes, process RAM, device budgets and timings rather
than a process-wide or machine-wide growth threshold. Missing or invalid Windows
counter samples are null, not zero. Process RAM is physical memory currently
resident for that process; virtual mappings are not equivalent to resident pages.

Canonical reports require the explicit null assessment. Historical boolean flags
remain readable outside canonical validation; they are not spill proof. Preserve
historical reports with their original source and validator, without rewriting
them. The standalone residency probe uses `dinkster-residency-probe/4` and the same
unassessed meaning; its timing-outlier flag is not a spill verdict either.

## Learn more

See DESIGN 3.3 for the worker boundary, DESIGN 3.6 for pack manifests,
DESIGN 3.9 for doctor and diagnostics, DESIGN 3.10 for placement and memory,
and DESIGN 3.11 for sandboxing. Boundary and authoring invariants are in
`docs/hazards.md`; focused coverage is in `tests/test_doctor.py`,
`tests/test_isolated.py`, `tests/test_remote.py`, `tests/test_placement.py`,
`tests/test_egress.py`, `tests/test_sandbox.py`, and `tests/test_transport.py`.
