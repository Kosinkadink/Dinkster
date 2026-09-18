# Hazards - load-bearing invariants

Living document, mirroring Dinkster-Frontend's practice. Each entry: the invariant,
why it exists (usually the exact way ComfyUI got it wrong), and the failure mode
if broken. Add entries whenever a design decision is load-bearing; never delete,
only mark superseded.

## H1. The schema object is the only node interface description

The engine, validator, cache-key builder, wire encoder, and isolation layer consume
`NodeSchema` objects. No `INPUT_TYPES`-style dicts in core, ever. ComfyUI's V3
schema compiles down to v1 dicts and smuggles features through reserved type
strings; the richest model became derived instead of authoritative, and every
consumer re-parses strings. Failure mode: two descriptions of one node drift, and
features get designed to fit the weaker format.

## H2. Nothing crosses an edge raw

Every value between nodes - including Int/Float/String/Bool - is a `Value` envelope
with type_id, meta, fingerprint, and payload transport. ComfyUI passes bare tensors
and ints, so nothing can be interrogated, fingerprinted, or moved across a process
boundary without per-case retrofits. Failure mode: one raw value on an edge and
remote workers, CAS caching, and type interrogation all grow special cases.

## H3. The engine never calls node code

All execution goes through `Worker.invoke(Invocation)`; in-process is just the
trivial transport. ComfyUI calls `getattr(obj, obj.FUNCTION)` directly, so process
isolation had to be faked with host stub classes. Failure mode: any direct call
becomes a feature that only works in-process, and the boundary rots.

## H4. Cache keys derive only from schema signature + input fingerprints

Never from node IDs, never from in-process object identity, never from
wall-clock. Nodes declaring `idempotent=False` get a unique key component per
execution - so they are never reused or coalesced - but even that component is
random, never a node ID. This is what makes cache entries shareable across
processes/machines. Failure mode: caches silently become location-bound and
cross-run reuse dies.

## H5. Packs register through typed registries; loading has no host side effects

A pack that only uses `dinkster_api.v1` works fully isolated. Core packs use the same
API as third-party packs - if core needs a private hook, the contract is wrong.
Manifest loading imports no pack code; pack entry imports happen inside the
worker host, and `dinkster doctor` diagnoses import-time output, thread spawns,
and out-of-door imports (host machinery is an error, internal packages a
warning). ComfyUI custom nodes mutate globals at import time, making every load
order-dependent and every internal change a breaking change. Failure mode: one
privileged core shortcut and the extension API stops being trustworthy.

## H6. One-way package dependencies

values <- schema <- graph <- engine <- server, with the execution boundary
split out below the engine: protocol (the Worker/CacheStore contracts, over
schema/values only) <- (engine, workers, caches) - workers and caches
implement the protocol and never import the engine, so a worker child
interpreter carries no scheduler. memory and assets are additional
bottom-layer branches; node packs see only the extension API
(`dinkster_nodes_foundation` and `dinkster_nodes_media_io` may import nothing but
`dinkster_api`). The
authoritative edge list is the `ALLOWED` map in `tests/test_dependency_rule.py`,
which parses every module of every `packages/dinkster-*` package (a completeness
check keeps new packages from escaping) and runs in CI. Failure mode: one upward
import and the packages stop being independently testable, which is how
execution.py became 1300 tangled lines.

## H7. Pre-1.0 wire formats are unstable, and only the current version decodes

Version numbers may bump pre-1.0 (schema wire is at v3), but no backward
decoders or migration shims accrete: decoders accept exactly the current
version and refuse everything else loudly (tested). Real migration guarantees
begin at first public release; after that, changes are additive-first with
explicit schemaVersion bumps. Failure mode: a best-effort parse of a stale
version, and pre-release formats become a compatibility surface nobody
promised.

## H8. Partner nodes are data, not code, for the standard cases

*(Directional - M4 is deferred by decision; RemoteNodeDefinition is design-only
and nothing here is implemented yet. `RemoteWorker` in dinkster-workers is
unrelated transport, not this.)* RemoteNodeDefinition (schema + typed HTTP
contract + operation kind) interpreted by a generic runtime. Imperative escape
hatch stays inside the pack boundary. Failure mode: one provider implemented as
bespoke in-tree code and the "update providers without releasing Dinkster"
property is lost.

## H9. The boundary is free for node authors

A standard node is `define_schema()` plus an `execute()` that takes and returns
plain Python values. The worker shim does all envelope wrapping/unwrapping from the
schema; authors never construct envelopes, never see transports/fingerprints/
placement, and never write location-aware code. Envelope-level APIs are strictly
opt-in for advanced cases. The test: a node written naively against `dinkster_api.v1`
runs unchanged in-process, in another venv, and across a network service
boundary (proven in CI over loopback TCP; the venv leg is slow-test gated).
Failure mode:
one required envelope-touching step in the standard authoring path, and the
ecosystem's node quality drops to the sophistication of its median author - or
they route around the API entirely, recreating ComfyUI's hack culture.

## H10. Dynamic interfaces elaborate once, before all consumers

Elaboration is a pure function `(schema, stored input ids, stored output members)
-> effective concrete interface` - never solved types, never runtime values, so it
cannot oscillate (the frontend's anti-feedback invariant, adopted verbatim). Input
and output families share one representation and one elaboration pass.

**Output membership is document state.** The graph node persists an ordered
member-suffix list per output family; input/UI changes that alter output shape are
document transactions applied *before* submission, so execution receives a fully
known topology and a node can never change its interface by executing. Genuinely
data-dependent cardinality is a collection-typed value on one static output, not
variable graph ports.

**One immutable run snapshot, one elaboration function, every consumer.**
`Engine.run()` deep-snapshots the document at entry (caller-owned dicts must not
mutate topology across awaits - TOCTOU), elaborates the top-level document once,
and feeds that effective map to validation, cache keys, invocations, worker
validation, and cache-hit validation (planning consumes the snapshot's
topology; elaboration precedes it). Region bodies are re-elaborated per region
execution - legal only because elaboration is pure over the same immutable
snapshot, so the results are identical by construction. Invocations carry the effective
schema plus explicit output-family provenance; the worker cross-checks both, so a
version-skewed worker fails loudly. Cache entries whose output ids/types do not
match the effective schema are misses, never answers. Node authors see one
reserved `output_spec` parameter (`OutputInterface`) and return grouped values;
the shim flattens and validates exact membership.

Downstream core (validation, planner, cache keys, worker shims) consumes only the
elaborated interface - plain concrete ids - and never branches on "is this
dynamic". Failure mode: one consumer that re-derives, re-elaborates, or
special-cases dynamic membership, and interface shape becomes distributed mutable
state - the bookkeeping headache the elaboration step exists to prevent.

## H11. Legacy compatibility is quarantined

The ComfyUI compat loader is one isolated Worker, purpose-bound to migration and
quick testing, best-effort by design. Packs that monkey-patch ComfyUI's server,
executor, or samplers get a diagnostic, not emulation. No parameter, branch, or
schema feature may exist in dinkster-schema/values/graph/engine because the adapter
needs it - if the adapter cannot express something, the adapter stays imperfect.
That core-boundary rule is absolute; narrow bug-for-bug reproductions *inside
the quarantine* may be granted case by case (each explicit in DESIGN 3.8,
naming the v1 behavior, rationale, and blast radius, and revocable). Native
ports (helped by `dinkster-port` codegen) are the destination. Failure mode:
chasing bug-for-bug compatibility until Dinkster is a second ComfyUI carrying the
exact tech debt it was created to shed.

## H12. Parallelism is a scheduler property, never a node property

Any interleaving must produce identical results: nodes communicate only through
immutable value envelopes (H2/H3), runs execute an immutable document snapshot
(H10), and identical computations coalesce through engine-wide single-flight on
cache keys - so concurrency changes wall-clock time and nothing else. Concurrency
*limits* are resource declarations (worker/device slots), never node code, and
node authors never observe or control scheduling (H9). Non-idempotent nodes are
exempt from coalescing by construction (unique keys), not by special-casing.
Failure mode: one pair of nodes relying on execution order or on out-of-band
shared state, and every scheduler change becomes a behavior change - parallelism
turns opt-out-by-bug.

## H13. Memory decisions flow through one governor

Budgets are declared, costs are accounted at the boundary (envelopes carry bytes
by residency class), and every core-managed release of memory - cache eviction,
resource unload - is triggered by the per-instance MemoryGovernor's pressure
signals, never by scattered free-then-hope callers. Core-planned large
allocations reserve before they materialize (arbitrary node-internal
allocations cannot be policed; the rule binds everything core registers or
plans). Cross-instance coordination (shed/reserve/trim endpoints, heartbeat
discovery) is cooperative and advisory: leases have TTLs, measured free memory
stays ground truth, and a crashed or ignoring peer degrades throughput, never
correctness. Failure mode: any component that frees or allocates significant
memory outside the governor recreates ComfyUI's model_management - distributed
guessing where every fix is another guess.

## H14. Boundary payloads are single-hop handoffs

A value crossing a process boundary travels as its registered codec's bytes,
and any out-of-band carrier for those bytes (a shared-memory segment today;
CUDA-IPC handles later) lives exactly one hop: the sender creates it, the
receiver retains a read-only mapping and acknowledges, then the sender unlinks
the segment name while the receiver's mapping remains valid. Anything
unacknowledged when either side shuts down is unlinked by its creator. A
receiver that cannot decode the type keeps the bytes (EncodedPayload) and may
relay them onward verbatim - the envelope is inspectable everywhere; the
payload is usable where its type lives (H2). Failure mode: segments treated
as shared state instead of handoffs leak on crash, get unlinked while still
mapped, or turn into an invisible side channel between nodes that H9 promised
could not exist.

## H15. Presentation is pixels, never identity

Everything declared for display - `[pack.presentation]` (display_name, abbr,
mark, color, icon), node display names, categories, description/doc prose,
deprecation prose, search visibility, replacement rules - stays out of schema
signatures, documents, compiled prompts, type solving, and cache identity
(`schema_signature` strips every presentation field; tested). The pack name and node_type are
the only keys; marks/abbrs/colors may collide across packs, and the frontend
owns all fallback derivation (the backend never synthesizes presentation -
mixing declared and synthesized fields on the wire would hide what the author
actually said). Malformed presentation warn-and-drops at load; it can never
stop a pack from loading or a graph from running. Failure mode: one
presentation field in a signature, and renaming a badge invalidates caches
and migrates workflows - ComfyUI's display-name-as-identity problem reborn.

## H16. Clients get renditions and descriptors, never raw payloads

Rich values reach browsers only through registered renditions (declared
kinds with MIME types, negotiated by the client; digest/fingerprint-keyed,
immutably cacheable) and through descriptor facts (typeId, fingerprint,
length, meta, policy-gated inline scalars). Raw tensor envelopes never
serialize onto the client wire. Frozen-value lookup (`/api/values`) is
execution-scoped - (job, runtime node, output) is the identity, fingerprint
is a cache tag only - and a missing or evicted value is a structured refusal
(`available: false` with a reason; evicted distinct from unknown), never a
silent substitution of a newer value. Failure mode: one endpoint resolving
"latest" instead of refusing, and every frozen-view feature built on "never
silently wrong" becomes silently wrong.

## H17. Execution provenance is monotonic occurrences in one closed grammar

A runtime node id runs and terminates exactly once: scheduling mints each
occurrence id once, and each emits one `node_started` and exactly one terminal
event (tested end-to-end through nested regions). Everything that multiplies
execution - region iterations today, stream chunks if/when `stream<T>` ships
*(planned, not implemented)* - mints new occurrence ids in the single closed
grammar (`node`, `node[i]`, `/`-nested; `[`, `]`, `/` banned in document node
ids - rejected at validation and wire decode - so iteration ids strip
mechanically back to document nodes). Never re-enter a terminal id, never
invent a second path dialect. Streaming, if built, is executor-owned
scheduling over that same machinery: a value must never embed a pull-driven
sub-engine (the rejected ComfyUI IMAGE_STREAM shape) - hidden execution
inside a payload is invisible to caching, events, provenance, and worker
transport all at once. Failure mode: one re-entrant node_finished or one
value that executes, and every consumer that reconciles execution state -
frontend stores, caches, event normalizers - needs bespoke repair logic.

## H18. Asset identity is content, never location

An asset is its canonical `blake3:` digest. `AssetRef` equality/hashing exclude
resolver and path; `dinkster.asset` values fingerprint by digest alone and reject
path literals; every newly acquired byte stream (vault ingest, mirror fetch,
peer fetch) must verify against the declared digest before it becomes
materializable content. Catalogs, libraries, mirrors, and vaults are
interchangeable *locations* - swapping one for another (local disk, shared
vault, remote peer) changes latency, never identity or results. ComfyUI keys
models by filename under hardcoded folder trees, so a rename is a different
model and a moved file is a broken workflow. Failure mode: one code path that
treats a filesystem path as identity, or trusts resolver bytes without
verification, and workflows stop being portable while caches silently serve
wrong content.

## H19. Absence is a typed value, never an in-band sentinel

Deliberate "no value" is a `core.absent` envelope with deterministic
fingerprint and root provenance; consumers declare a policy (`skip`, `omit`,
`accept`, `fail`) on each input, applied by the engine before invocation.
Absent, skipped, failed, and undemanded stay distinct states. ComfyUI's
ExecutionBlocker is a magic object passed *into* unsuspecting node code, so
every node either special-cases it or breaks. Failure mode: one absence
represented as None-in-band or a sentinel payload, and every downstream node
needs defensive checks while caches can no longer distinguish "computed
nothing" from "never ran".

## H20. Collections are values; repetition is graph structure

List cardinality lives in explicit `list<T>` envelopes (children are full
envelopes, recursively); repetition lives in explicit region nodes (map/fold/
while with declared zip/cross bindings, state chains, and gathers). The
executor never implicitly maps a node over a list, never zips, clamps,
broadcasts, or unwraps mismatched cardinalities - a list-into-scalar edge is a
validation diagnostic, not a calling convention. ComfyUI's INPUT_IS_LIST/
OUTPUT_IS_LIST made every execution secretly variadic with silent zip-and-clamp
semantics no one could predict. Failure mode: one implicit coercion and edge
types stop meaning anything - every node must be read to know what an edge
carries.

## H21. Pack identity comes from granted claims, never load order

Pack names and namespace claims live in one closed lowercase grammar with
separator-equivalent canonicalization; every node type must fall under its
pack's exclusively granted claim; reserved roots (`core`, `comfy`, ...) require
explicit host trust; composition rejects duplicate canonical names, overlapping
claims, and uncovered types outright. A manifest *requests* identity - it never
establishes ownership, and no pack wins a collision by loading first. The
registry (M8) inherits this: registry grants, not manifest claims, are the
authority. ComfyUI's registry regexed v1 registrations, missed V3 nodes, had no
pack prefixes, and let git-repo capitalization make one pack look like two.
Failure mode: one load-order-dependent name grab and installing a pack can
silently retarget another pack's nodes.

## H22. Requested sandboxing fails closed

When a launch requests a sandbox, an unavailable capability (no bubblewrap,
degraded probe) or a policy that would dissolve confinement (binding `/`,
`/home`, `/etc`) aborts the launch with `SandboxUnavailable` and remediation -
it never silently degrades to a plain subprocess. A subprocess is an
engineering boundary, not a security boundary; only an explicit no-sandbox
request gets one. Failure mode: one silent fallback and "this pack ran
sandboxed" becomes a lie exactly when the user depended on it.

## H23. Diagnostics carry their origin; packs never own the channel

Human-facing logs flow through the host-configured `dinkster` logger tree with
explicit per-origin names (`dinkster.<subsystem>` vs `dinkster.pack.<id>`), so verbosity is
tunable per origin and a log line always says who spoke; core code cannot squat
the pack namespace, packs get a logger through the door but never configure
handlers, and records never propagate to the root logger. Machine-facing node
telemetry (progress, previews) goes through the separate non-blocking Reporter
channel. Neither channel may carry correctness-critical state. `dinkster doctor`
warns on raw import-time output and cross-origin logging. ComfyUI prints from
everywhere, so nobody can attribute a line or silence a pack. Failure mode:
one subsystem logging without attribution and debugging a 50-pack install
regresses to grepping interleaved prints.
