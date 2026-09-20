# Dinkster - Design

Dinkster is a clean-slate ComfyUI backend. Like Dinkster-Frontend, it is not shackled by
backwards compatibility: legacy behavior is something we *import or adapt at a
boundary*, never something we build the core around. This document holds the
architecture rationale: what is wrong with the current backend, the design that
fixes it, and what we reuse. What is supported today is linked from the
[SUPPORTED.md index](SUPPORTED.md); implementation detail lives in the code and package
READMEs; work history and deferred design questions live in GitHub issues.

**Mission.** Dinkster is the successor to ComfyUI, not an orchestrator of it. The fact
that ComfyUI runs *inside* Dinkster (isolated compat workers, translated schemas,
governed memory) is the proof of a general capability: Dinkster can swallow other
software the same way - host it at a boundary, govern its resources, and translate
its surface - while the native core grows to replace what it hosts. There is no
timeline map for this; the direction is fixed, the pace is slice-by-slice.

Companion project: [Dinkster-Frontend](https://github.com/Kosinkadink/Dinkster-Frontend)
(clean-slate frontend, TypeScript/SolidJS monorepo). Several Dinkster decisions are made
*with* the frontend: most importantly the native node-schema wire format, for which
the frontend's normalized `NodeSchema` model (`@dinkster/core` `schema/model.ts`) is the
reference spec.

---

## 1. Why a new backend

Findings from the current ComfyUI codebase (paths refer to Comfy-Org/ComfyUI):

1. **The ancient "v1" dict schema is the engine's source of truth.** The execution
   engine consumes `INPUT_TYPES()` dicts, `RETURN_TYPES` tuples, and `FUNCTION`
   strings everywhere. The modern V3 schema exists but is *down-converted to v1*
   for execution and for `/object_info`; V3 features that don't fit v1 are smuggled
   through reserved type strings the frontend must parse back out. The richest
   schema is the *derived* one; the authoritative one is the weakest.

2. **Execution, caching, queueing, and serving are braided together.**
   `execution.py` (~1300 lines) contains the executor, cache orchestration, v1 AND
   v3 input resolution, validation, list/batch mapping, and the `PromptQueue`;
   `server.py` owns the queue instance; `main.py` owns the worker loop. None of it
   is separately usable or testable.

3. **Values on edges are raw Python objects.** A LATENT is a dict, an IMAGE is a
   bare torch tensor. Nothing on an edge can be interrogated, fingerprinted for
   caching, or serialized across a process/machine boundary without ad-hoc
   per-case code.

4. **Node execution assumes same-process, same-venv.** The engine calls
   `getattr(obj, obj.FUNCTION)(**inputs)` directly. Process isolation (pyisolate)
   is mature but necessarily *wrapped around* the engine. In Dinkster the boundary is
   the engine's native shape, and in-process is just the trivial transport.

5. **There is no defined extension API.** Custom nodes mutate `NODE_CLASS_MAPPINGS`,
   monkey-patch server routes, samplers, and model management at import time.
   Nothing is predictable, so nothing is safely evolvable.

6. **Partner (API) nodes live in-tree** and ship on the core release cadence, even
   though they are almost pure data: schema + typed request/response contract +
   polling/upload conventions against `/proxy/<provider>/...`.

---

## 2. Principles

Shared with Dinkster-Frontend, adapted for a Python backend:

- **Single source of truth; derive, don't duplicate.** One schema model. One value
  model. Wire formats, caches, and compat shims are derived projections.
- **The boundary is the architecture.** Every node invocation crosses an explicit
  boundary (even in-process). Everything that crosses it is a typed, serializable,
  fingerprintable envelope. If a feature "just needs" to reach around the boundary,
  that is a contract bug to fix, not a shortcut to take.
- **Typed registries, never monkey patches.** Core node packs use the exact same
  extension API as third-party packs. If core needs a hook an extension can't
  have, the contract is wrong.
- **One canonical representation per concept; no unions of shapes.**
- **Advisory vs authoritative split.** Structural graph validity is authoritative;
  type compatibility on edges is checked at plan time but the type *system* is
  interrogable data, not isinstance checks scattered through the engine.
- **Pre-1.0 formats are unstable and revised in place.** Real migration guarantees
  start at first public release; after that, wire changes are additive-first with
  explicit version bumps.
- **Demo-first.** Every milestone ends in something runnable end-to-end.
- **hazards.md is a living document** recording load-bearing invariants, why they
  exist, and the failure mode if broken. Seeded in `docs/hazards.md`.

---

## 3. Architecture

Python >= 3.12. `uv` workspace monorepo, strict typing (`pyright` strict + `ruff`),
asyncio-native core, `pydantic` for wire models only (core models are plain
dataclasses/protocols so the engine has no serialization framework in its hot path).

### Package layout

```diagram
+----------------------------------------------------------------------+
|                            dinkster-server                              |
|        HTTP + WS API, queue, sessions, events, artifact serving      |
+---------------+--------------------------------------+---------------+
                |                                      |
+---------------v--------------+       +---------------v---------------+
|         dinkster-engine         |       |           dinkster-api           |
|  planner, scheduler, cache   |       |  the versioned extension door |
|  orchestration, invocation   |       |  (pack-author surface)        |
+------+---------------+-------+       +---------------+---------------+
       |               |                               |
+------v------+ +------v-------+       +---------------v---------------+
|dinkster-workers| |dinkster-caches  |       |          node packs           |
| in-process, | | memory-lru,  |       | foundation + media I/O,       |
| venv/subproc| | ram-aware,   |       | dinkster-compat-comfy,           |
| remote      | | disk/CAS,    |       | dinkster-partner-nodes,          |
|             | | remote       |       | third-party packs             |
+------+------+ +------+-------+       +---------------+---------------+
       |               |                               |
+------v---------------v-------+                       |
|        dinkster-protocol        |                       |
| Worker/CacheStore contracts  |                       |
| + Invocation data: the       |                       |
| execution boundary           |                       |
+------+-----------------------+                       |
       |                                               |
+------v-----------------------------------------------v---------------+
|                     dinkster-values  +  dinkster-schema                    |
|   value envelopes, type registry, codecs, payload transports  |      |
|   node/type schema model (the single source of truth), wire enc      |
+----------------------------------------------------------------------+
```

Dependency rule (enforced in CI): arrows only point downward. `dinkster-schema` and
`dinkster-values` depend on nothing above them and have **no torch dependency**; the
engine imports no server code; workers/caches implement the `dinkster-protocol`
contracts and never import the engine; node packs see only the extension API.

- **`dinkster-schema`** - the V3-native node schema model: typed inputs/outputs/
  widgets, one ordered interface list, real output IDs (no positional tuples),
  type constraints on inputs AND outputs, dynamic constructs as structured
  objects, node metadata. Plus the wire encoding: the `/object_info` successor,
  isomorphic to the frontend's normalized `NodeSchema`, carrying a
  `schemaVersion` field.
- **`dinkster-values`** - the value envelope system (3.2).
- **`dinkster-graph`** - prompt/graph model, structural validation, plan building.
  Pure: no IO, no torch.
- **`dinkster-protocol`** - the execution boundary itself: the `Worker` and
  `CacheStore` protocols plus the frozen data they exchange. A leaf over
  schema/values only.
- **`dinkster-engine`** - the scheduler behind the boundary; progress/event emission
  and cancellation. Never imports node code.
- **`dinkster-workers`** - `Worker` implementations: in-process, isolated (subprocess
  in its own venv), remote (same protocol over the network). Also pack manifests
  and venv provisioning.
- **`dinkster-caches`** - `CacheStore` implementations: memory LRU, disk CAS, peer.
  Composable as layers; governed consumers that evict only when the
  MemoryGovernor says so.
- **`dinkster-memory`** - the MemoryGovernor (3.10): per-device budgets,
  reservation-before-allocation admission, pressure-driven shedding.
- **`dinkster-assets`** - content-digest asset identity, virtual-folder catalog,
  libraries and resolvers (3.12).
- **`dinkster-api`** - the versioned extension door: `dinkster_api.v1` re-exports the
  pack-author surface, frozen by a golden compat test suite.
- **`dinkster-server`** - aiohttp HTTP/WS server, job queue, multi-client sessions,
  the native protocol (3.5), artifact upload/serving.
- **Node packs** - `dinkster-nodes-foundation` (logic, math, text, lists,
  utilities), `dinkster-nodes-media-io` (asset-backed media I/O),
  `dinkster-nodes-generation` (provider-independent loading, conditioning,
  sampling, and codec schemas), model-family packs such as `dinkster-model-wan`,
  `dinkster-compat-comfy` (the quarantined ComfyUI surface and an execution
  provider for generation schemas), and `dinkster-nodes-partner` (3.7).
  `dinkster-nodes-std` is the metadata-only install suite for the foundation and
  media packs.

### 3.1 Schema as the single source of truth

Node authors declare a V3-style schema (`define_schema()` returning a typed
`NodeSchema` object) and an `execute()` (sync or async). The engine, the validator,
the cache-key builder, the wire encoder, and the isolation layer all consume the
*schema object*. There is no `INPUT_TYPES` dict anywhere in core; a v1 projection
would live in a quarantined module designed for deletion.

Types use the frontend's closed `TypeExpr` model natively (`concrete | union |
wildcard | variable | list`), so the wire encoding is a near-identity mapping to
the frontend's normalized `NodeSchema`.

**Dynamic interfaces (inputs and outputs)** adopt the frontend's proven elaboration
model wholesale (hazard H10):

- Dynamic constructs are structured objects from a closed set (autogrow families,
  dynamic combo/slot), symmetric for inputs and outputs.
- **Interface membership is document state, never runtime discovery.** The graph
  node stores ordered member lists; by the time a run starts, topology is
  completely known. A node can never change its interface by executing. True
  runtime structural fan-out is the explicit region-expansion feature (3.13),
  never interface mutation mid-run.
- **Elaboration is a pure function** `(schema, stored input ids, stored output
  members) -> effective concrete interface` - never solved types, never runtime
  values, so there is no feedback loop. Malformed membership fails elaboration
  deterministically; it never clamps or repairs.
- **One snapshot, one elaboration artifact, every consumer.** `Engine.run()` takes
  a deep immutable snapshot of the document, elaborates once, and that same
  effective map feeds validation, planning, cache keys, invocations, worker-side
  validation, and cache-hit validation. What validation approved is exactly what
  executes.
- Member identity is stable ids/paths (`parts.a`), never ordinals. Invocations
  carry the effective interface plus explicit output-family provenance; the worker
  cross-checks it and wraps `execute()` returns against the *elaborated* schema.
  A cache entry whose output ids or types do not match the effective schema is a
  miss, not a wrong answer.
- **Node authors stay boundary-oblivious** (H9). A node declaring output families
  receives one reserved parameter, `output_spec`, exposing the ordered membership;
  static nodes never see it.

**Lifecycle metadata is schema data, never validity.** Deprecation, search
visibility, and pack-declared replacement rules (a closed, serializable rule
vocabulary mirroring the frontend's ReplacementRule model - guards, source
mappings, transforms; static interface ids only) ride the schema wire additively
and are excluded from the schema signature: migration advice must not invalidate
caches. Cross-schema reference validation is advisory at server startup and an
error in `dinkster doctor`.

### 3.2 Value envelopes: nothing crosses an edge raw

Every value produced or consumed by a node - including Int/Float/String/Bool - is
wrapped in a `Value`:

```python
class Value(Protocol):
    type_id: TypeId              # registered type, e.g. "core.image"
    meta: ValueMeta              # shape/dtype/count/etc - interrogable, cheap
    def fingerprint(self) -> Fingerprint: ...   # content-based, for cache keys
    def payload(self) -> PayloadRef: ...        # how to actually get the bytes
```

- **TypeId + registry.** Types are registered data with declared metadata schema,
  codecs, and conversion edges. The backend can answer "what is on this edge"
  without isinstance guessing; generic tooling hangs off the registry.
- **Sane defaults for type registration.** Declaring codec/fingerprint is an
  optimization, never a prerequisite. A type registered with nothing but a name
  works in-process; crossing a real boundary uses a default codec and a
  content-hash fingerprint. Defaults are correct-everywhere-possibly-slow; their
  absence is at worst a perf diagnostic, never an error.
- **PayloadRef + transports.** Payloads are accessed through a transport: `inline`,
  `pyobj` (same-process fast path), `shm`, `cuda-ipc`, `file`, `cas`. Which
  transport is used is a *placement decision*, not a node concern.
- **Fingerprints replace `IS_CHANGED` guessing.** Cache keys are
  `(node signature, resolved input fingerprints)`; entries are
  location-independent and shareable across processes and machines.
- **Handles for non-serializable resources.** Models/VAE/CLIP are `ResourceHandle`
  values: the envelope carries identity + residency (which worker owns it), and
  the engine schedules consumers accordingly.

### 3.3 Boundary-first execution engine

The engine never calls node functions. It builds a plan from the graph, then for
each node emits an `Invocation` to a `Worker`:

```python
class Worker(Protocol):
    async def prepare(self, node_types: list[NodeTypeRef]) -> None: ...
    async def invoke(self, inv: Invocation) -> InvocationResult: ...
    async def cancel(self, inv_id: InvocationId) -> None: ...
```

- `InProcessWorker` resolves ValueRefs to Python objects and awaits `execute()`.
- `IsolatedWorker` runs a per-pack venv subprocess with RPC and shm/CUDA-IPC
  tensor passing; because inputs are already envelopes with transports, isolation
  needs no per-type serializer bolted on.
- `RemoteWorker` is the same protocol over a network transport. Nothing in the
  engine changes.
- A **placement policy** assigns nodes to workers (isolation requirements,
  resource residency, user config). Scheduling is output-driven with lazy edges,
  as a pure planner over the graph package.
- **Node authors never see any of this.** The worker shim resolves incoming refs
  and wraps returns using the declared output types; `execute()` receives and
  returns ordinary Python values. Writing a node must feel like writing a typed
  function, or the boundary has failed (H9). Outputs are returned as a mapping
  keyed by output id - identity, never position.

### 3.4 Caching as a first-class interface

```python
class CacheStore(Protocol):
    async def get(self, key: CacheKey) -> CacheHit | None: ...
    async def put(self, key: CacheKey, outputs: dict[OutputId, Value], cost: Cost) -> None: ...
    async def evict(self, pressure: PressureSignal) -> None: ...
```

Keys derive from schema signature + input fingerprints (3.2). Stores compose in
layers (memory -> disk CAS -> peer); eviction policies are store concerns, not
engine concerns. Because values are envelopes, a disk or peer cache persists
payloads content-addressed and rehydrates them on any worker - cross-run and
cross-machine caching falls out of the design.

Two content-addressing scopes exist, deliberately distinct and never mixed:
the **persistent CAS** uses canonical `blake3:` digests - the same namespace
dinkster-assets uses, so payload and asset blobs can share a store - while
**conversation-scoped boundary dedup** (the `cas` value transport) uses stdlib
digests that die with the socket. The disk store keeps one manifest per cache key
naming per-output type/fingerprint/digest; blobs are shared across entries and
garbage-collected reference-aware. A peer store is a read-only pull layer over
another instance's cache endpoints: every failure is a miss, never an exception
on the execute path. Trust posture: manifests carry codec bytes, so cache sharing
lives in the same trust domain as workers - peers you would hand a worker token
to.

### 3.5 Server and native protocol

- aiohttp (proven for WS + binary frames; no extra stack on the hot path).
- Native schema endpoint replacing `/object_info`, emitting the
  NodeSchema-isomorphic wire format with `schemaVersion`.
- Jobs identified as `(clientId, jobId)`, so multi-backend/multi-client is
  first-class from day one.
- **Run control is queue policy, not engine surgery**: pause/resume gates
  dispatch, clear cancels queued work, running jobs are cancelled individually.
  Terminal jobs stay pollable for a bounded history, which also bounds per-run
  bookkeeping.
- WS protocol with a feature-flag handshake and a typed event catalog: status, a
  node-state map (never a single execution cursor), progress, previews as
  self-describing binary frames (never base64 JSON), errors with structured
  provenance.
- **Node reporting is ambient and typed**: nodes call `report_progress` /
  `report_preview` / `report_event` as plain module functions, so `execute()`
  signatures stay values-only. The worker installs the reporter; the same code
  reports identically in-process, cross-venv, and cross-machine. Events are
  droppable chatter - anything correctness-critical belongs in outputs - and per
  invocation reach the engine before the result does.
- **Memory telemetry is pushed, not scraped**: governed instances broadcast a
  low-rate `memory_status` event; frontends subscribe like any peer.
- **ComfyUI API prompt submission** is umbrella-owned glue over a pure translator:
  legacy prompts translate to ordinary native graphs; the engine never learns v1
  existed.
- **Pack provenance, badges, icons, and blueprints are presentation data.** The
  nodes endpoint carries a packs table (display metadata, digest-addressed icons
  with immutable caching, starter-workflow blueprints validated as data and
  served byte-identical) and per-node pack attribution assigned by the host.
  None of it joins schema signatures or cache identity.
- **Environment stamping is a record, never identity**: an optional environment
  block (dinkster version, wire version, per-pack pins/sources, per-node schema
  signatures) lets clients stamp saved workflows; drift is advisory diagnostics,
  documents are never rejected.
- **Frozen-run value peeking**: a read-only endpoint serves a terminal job's
  output values by execution-scoped identity (fingerprints validate, never look
  up). Rich types register browser-renderable renditions in the type registry;
  raw tensors never go to a client. Small scalars inline at the source. Refusal
  is structured (`{"available": false, "reason": ...}`) - a peek can be
  unavailable but never silently newer.
- **Video (contract pre-agreed with the frontend, unshipped)**: every video
  output declares a static poster rendition so generic peek renders it with zero
  video-specific code; playable containers ride range requests; live preview
  stays frame-based during a run; streaming, if built, is an executor concern -
  never a value-embedded pull protocol (that shape hides a sub-engine inside a
  value, breaking cache identity, events, and worker transport).
- Cache-sharing and asset endpoints are opt-in at app construction; trust
  postures in 3.4 and 3.12 apply.

### 3.6 Extension API: predictable, no hacks required

- **Typed registries** for every extension point: node types, value types, codecs,
  payload transports, workers, cache stores, asset resolvers, server routes
  (namespaced), event subscribers. Registration is declarative and inspectable;
  conflicts are diagnostics, not last-import-wins.
- **Versioned, frozen API surface**: packs import `dinkster_api.v1` only; internal
  modules are underscore-private. Since this is Python, people *can* still reach
  into internals - the goal is that they never *need* to, and everything
  reachable through the door is covered by compat tests (a golden name list pins
  each export by identity; within v1 the surface only grows, breaking changes
  mean a v2 module). Host machinery - wire decoding, elaboration, the
  Value/Payload model, the governor - stays deliberately outside the door: packs
  consume its effects, never its objects.
- **Isolation by default** makes the contract honest: a pack that only uses the
  API works isolated; a pack that reaches into host internals visibly breaks in
  dev, not in users' installs.

### 3.7 Partner (API) nodes as data

Partner nodes reduce to: schema + typed request/response contract + endpoint path
+ polling/upload conventions. Dinkster makes that literal:

- A **`RemoteNodeDefinition`**: NodeSchema + typed HTTP contract + operation kind
  (sync / poll / upload-then-poll) + media in/out mapping - pure data, no
  imperative node code for the standard cases. An escape hatch allows imperative
  nodes for genuinely weird providers, still inside the pack boundary.
- A generic **remote-execution runtime** interprets definitions: auth, retries,
  rate limits, polling, progress, interruption, media transfer into value
  envelopes.
- **Packaging precedes remoting**: partner nodes ship first as a normal Dinkster
  pack - independently versioned and updatable, no Dinkster release needed to update
  a provider. The end state is definitions fetched from a provider catalog at
  runtime: pure data validated against the closed definition grammar, never code.
  Reload swaps what NEW jobs see; jobs execute against the snapshot they compiled
  with (H10).
- **Local validation of remote constraints**: provider input limits are declarable
  as data on the definition (a constraint grammar, not validation code) so
  obviously-invalid calls fail locally - a fast-fail courtesy; the remote answer
  remains authoritative.

### 3.8 Legacy ComfyUI node packs: a quarantined on-ramp, never a support surface

A compatibility loader is feasible precisely because of the boundary-first engine:
it is **one more Worker**, not a core feature. A dedicated isolated worker (venv
with ComfyUI installed) loads v1 packs using ComfyUI's own loading machinery; an
adapter synthesizes a NodeSchema from `INPUT_TYPES`/`RETURN_TYPES`, wraps
`FUNCTION`, and passes values as ordinary envelopes. Node ids are namespaced per
pack (`comfy.<pack>.<name>`), so v1's flat global namespace cannot cause
collisions. Every pack load yields a structured report (translation skips with
reasons, routes counted, runtime modules referenced). The child stands up
ComfyUI's real PromptServer headless because packs read it at import - environment
fidelity inside the quarantine, not emulation in core.

Bounds, decided up front (hazard H11):

- **Purpose-bound**: migration and quick testing, best-effort by design. Packs
  that monkey-patch ComfyUI's server, executor, or samplers get a clear
  diagnostic, not emulation. We never chase bug-for-bug compatibility - that road
  ends in a second ComfyUI. Any future deliberate exception must be listed here
  explicitly with its blast radius.
- **Compat never shapes core.** No parameter, branch, or schema feature may exist
  in dinkster-schema/values/graph/engine because the legacy adapter needs it.
- **Porting is the endorsed path**, and tooling makes it cheap: `dinkster port`
  translates a pack's declared schemas (v1 and V3) into a doctor-clean native
  pack skeleton - schemas translated faithfully, behavior ported by a human,
  never faked. The compat worker is the on-ramp; ported packs are the
  destination. See docs/pack-authoring.md.

### 3.9 Developer tooling: make inefficiency and bad practice visible

Sane defaults (3.2) and an oblivious authoring path (H9) only stay healthy if
developers can *see* what the defaults cost them. Tooling is first-class, and it
is the carrot that replaces ComfyUI's stick (where the only feedback is users
reporting breakage):

- **Pack watch mode (`--watch-packs`)** turns the boundary's natural observability into
  per-invocation diagnostics: execute vs serialization vs transfer time,
  transport per edge, payload sizes, fallback-codec hits, fingerprinting cost,
  cache-miss explanations - everything the engine already knows, surfaced as
  structured events.
- **`dinkster doctor <pack>`** - a pack linter: manifest completeness, import-time
  side effects, private-module imports, types sent across boundaries without
  codecs, blocking sync `execute()`, oversized inline payloads.
- **Diagnostics carry the fix, not just the finding.** Perf findings stay
  warnings; contract violations are errors even when Python cannot physically
  prevent them at runtime.
- **Benchmarking is assembly, not surgery.** The record is the event stream,
  structured: one record per job assembled host-side from events the engine and
  boundary already emit. If a phase matters and is not visible as an event, the
  fix is a new event, never a patch. A host-owned hardware sampler shares a
  monotonic clock with engine events; in-node phases are declared through the
  Reporter channel; records key occurrences by cache identity so cross-build
  comparisons are principled. Observation never participates: benchmark data
  enters no signature, cache key, or scheduling decision.
- **Pack hot-reload is a worker restart, never interpreter surgery.** A reload
  starts the fresh worker first and validates it fully while the old one keeps
  serving; only then does the swap commit atomically at every layer, with one
  epoch bump and one `schema_changed` event published after the new surface
  serves. Runs in flight pin the schema mapping they entered with, so a mid-run
  reload can fail a job but never corrupt one. A successful reload clears the
  result cache: keys fingerprint signature + inputs, never implementation, so
  changed code behind an unchanged signature would stale-hit forever. Dev-only
  surface (endpoint, file watcher, pack removal); production installs change
  packs through the manager's plan/apply flow.
- **Logging: origin is the logger name.** Plain stdlib logging; core subsystems
  log under `dinkster.<subsystem>`, packs under `dinkster.pack.<name>` - identity is
  declared, never guessed, so per-origin verbosity works. Handler configuration
  is host policy, never exported to packs. Worker children inherit stderr and
  environment, so one CLI flag reaches every pack process with origins intact.
  Logs are for humans; anything a frontend renders goes through `report_*`
  events.

### 3.10 Parallel execution and memory governance

Parallelism is a scheduler property, never a node property (hazard H12). The
invariants already paid for make it cheap: values are immutable envelopes,
invocations share no state, runs snapshot their document at entry, and cache keys
are location-independent. Concurrency therefore cannot change results - only
wall-clock time - and node authors never see it.

**Intra-graph parallelism.** The planner's output is a dependency DAG; the engine
runs a ready-set scheduler dispatching every satisfied node concurrently to its
worker. How much runs concurrently is a resource question answered by placement
and admission, not by nodes.

**Inter-run parallelism.** `Engine.run()` is reentrant by construction; the queue
above it is policy. The engine guarantees **single-flight**: an inflight table
keyed by cache key coalesces identical computations across runs. Non-idempotent
nodes have unique keys by construction, so they are never coalesced.

**Resource admission** is distinct from memory reservation: an execution slot
bounds how many invocations occupy a device; a VRAM reservation bounds whether
allocations fit. A schema declares *what* its execution occupies
(`occupies=("gpu",)`), never how it is scheduled. Which GPU is a property of the
values flowing in: hardware-owning values declare residency in envelope meta, and
admission binds each occupied kind to every instance the resolved inputs declare.
Three admission classes give limited parallelism by default: hardware-occupying
nodes serialize per concrete device (overlapping freely across devices - multigpu
parallelism works by default); `io_bound` nodes skip the compute lane and overlap
freely; plain local nodes share an implicit `compute` lane with default capacity
1 - raising it is a deliberate opt-in. Admission is engine-wide; only real
invocations consume permits. None of this touches identity: `occupies`/`io_bound`
ride the schema wire but are excluded from signatures, and residency lives in
meta, never fingerprints - the same model on a different device is the same
computation.

**Memory governance** inverts ComfyUI's distributed free-and-hope (hazard H13):
budgets are declared, costs are accounted at boundaries, and eviction flows
through one place.

- **Costs ride the envelope** (bytes by residency class), so accounting is
  observation at the boundary, not instrumentation in nodes.
- **One MemoryGovernor per instance** owns budgets and registered consumers, and
  is the sole arbiter of "make room".
- **Reservation before allocation.** A worker about to materialize something
  large requests a reservation; the governor sheds or delays until it fits. Packs
  declare reservation planners as manifest data (pure observation of the
  invocation's envelopes); across the isolated boundary, lease frames are
  serviced off the read loop and released on every exit path.
- **Placement consumes residency.** Any input's device residency pins the
  invocation to the owning worker; inputs owned by different workers are an
  error, not a guess; unpinned invocations go to a pluggable policy.

**Dynamic offload composes under the governor, not beside it.** An aimdo-style
allocator-level offloader owns the **mechanism** (which pages move, when) and
reacts at allocator speed; the governor owns **policy** (budgets, reservations,
who sheds first) and reconciles at admission speed. The bridge is headroom: a
granted reservation raises the offloader's device headroom, so its reactive
eviction honors commitments it never has to know about.

**Cross-instance coordination (same machine)** is cooperative and advisory by
design, spoken over each instance's API: status (budgets, usage, consumer
footprints, live leases), shed requests (replies with what was actually freed -
possibly short, never inflated), and TTL leases whose renewal is the liveness
heartbeat - a crashed instance cannot deadlock its peers. Instances discover each
other through atomically written per-machine heartbeat files; stale means absent.
Measured free memory remains ground truth; declared budgets are caps. An
ungoverned instance answers with a clean "no". Lifecycle is safe by ordering:
bind, announce the actual endpoint, re-announce on a heartbeat, withdraw before
stopping. Peer verdicts are relayed, not rewritten, so callers reason about one
admission model whether memory is local or a socket away. The same surface,
spoken over the network, is what a multi-machine scheduler consumes -
same-machine coordination is the local case of distribution.

**Memory observability is a product surface.** Consumers may implement a detail
contract: named items with stable IDs, real display names (asset identity, never
Python class names), byte decomposition by residency class, and page-residency
maps with explicit geometry, so no client hardcodes what the server knows.
Status reports measured telemetry next to declared budgets - the numbers
disagreeing is signal, not error. Item-level controls ("unload this model")
reuse consumer targeting with stable IDs, safe under concurrent mutation. The
compat pack's `ResidentPool` is the first consumer: vram pressure unloads
advisorily (identity survives, comfy reloads on next use); ram pressure is
invalidate-then-release - only safe once nothing can resolve the stub.

**Governance crosses the process boundary by relay.** Packs declare governed
consumers as manifest data; the worker announces them at hello; the host
registers one relay proxy per consumer. Footprint/detail reads never await a
boundary crossing (the child pushes snapshots at state-changing moments); shed is
the one live round trip, and only advisory vram pressure crosses. A dead worker's
proxies read zero and free zero.

**Cross-process ram release is a two-phase gate**, composed from run pins
(condemn/absolve on a `ResourcePins` table shared by engine and guard), two-phase
consumer release (propose with use-clock tokens, release only unmoved tokens),
in-flight result holds (a result's resources are held until the parent acks), and
a parent gate that runs every cache invalidator before condemning. Failure is
conservative in one direction only: anything unprovable frees nothing - a
spurious recompute beats a dangling stub.

**Remote workers** are the same boundary code over a socket: leases, the relay,
the ram gate, and conservative failure cross the network because they are
literally the same implementation. The genuinely remote decisions: the daemon
outlives its clients (one engine conversation at a time; pack state stays warm);
auth is a pre-shared token presented first and compared constant-time - it
authenticates, transport security is deployment's job; hello negotiates protocol
version and transports, refusing before any value crosses; shared memory is
structurally refused across machines; bulk values dedup by content within one
conversation, negotiated, never assumed; and every remote device fact is
qualified (`cuda:0@box1`) so remote footprints land on remote budgets, never
silently on local ones.

An admitted invocation is keyed by the server job reference, job attempt, and
invocation id. A daemon keeps running and completed records in memory for a
bounded reconnect grace (120 seconds by default). The same engine process can
rebind only when both process identities match; a daemon-authored owner epoch
fences the replaced socket. Final results and memory-release state replay until
acknowledged. Disconnected event chatter is lossy: only the latest event without
a binary preview is retained. A missing record, changed process identity,
expired grace, or malformed rebind fails the invocation without resubmitting or
placing it elsewhere. Daemon restart recovery and graph-wire changes are not
part of this boundary contract.

#### Autoregressive generation sessions

Logical generation is independent of execution chunking. A request fixes the
provider, immutable model identity, prompt or chat messages, ordered sampler
chain, stop conditions, and seed; provider microbatches and transport chunks do
not enter that value. Native and forwarded engines expose the same pull-based
event stream. Pulling grants backpressure one event at a time, cancellation is a
polled callback, and consumers own streams with a context manager so early
exit closes the iterator and releases in-flight work.

Mutable generation state never crosses the provider boundary. The first request
may ask to open a session, and its terminal result returns an opaque handle bound
to the provider and model identity. Continuations present that handle; providers
refuse cross-provider or cross-model use and explicitly close sessions. Handles
are references, not authorization, and contain no tensors, process addresses, or
cache contents. A session admits one request at a time. Successful terminal
results atomically commit generated tokens; cancellation, abandonment, and
errors roll back the request, and provisional sessions are discarded. Active
session close is refused, while repeated close after release is harmless.
Provider capabilities declare chat, session, token-id, ordered sampler, and
sampler-stage support before execution, so adapters refuse semantic downgrades
rather than silently approximating a request.

OpenAI-compatible forwarding remains stateless because the protocol does not
advertise durable continuation ownership. Its model identity binds endpoint,
remote model, and compatibility dialect, while API credentials remain transport
authority and never enter identity, payload, representation, or diagnostics.
Synchronous streams share a provider-owned asynchronous HTTP client and
acknowledge one normalized event at a time; cancellation closes the in-flight
response.
Dialect adapters send neutral values for server-side sampler defaults and
refuse any stage whose order or history scope the endpoint cannot preserve.

Incremental KV memory is a dynamic residency class, separate from immutable
model weights. Session owners account KV pages to the memory governor, may move
or evict them under policy, and preserve handle identity while the backing
placement changes. This lets continuous batching schedule logical requests
without making batch composition, page geometry, or device placement part of
request identity.

Native continuous Qwen execution uses a provider-owned fixed slot pool so
decode never assembles or pads KV tensors per step. Admission requires the
committed prefix, prompt, and maximum output to fit one slot. Prefill advances
in round-robin chunks, with a bound on consecutive decode batches while
prefill is waiting. Decode combines only streams at the same cache position
and with the same prefetch route; unequal cohorts take round-robin turns
rather than taking a masked ragged-attention path or following admission
order. Removing a stream compacts its slot without changing its session
handle, and provider close rolls back active streams before releasing the
pool and worker.

Sparse-MoE routing is model semantics, while expert placement is provider
policy. The router completes its full-expert softmax and top-k selection before
residency decisions. Each expert's gate, up, and down projections form one
residency unit, and only experts selected for the current token rows enter the
dynamic prefetch walk. This keeps router results independent of whether expert
weights are resident, demand-paged, or host-backed. Sharded checkpoints are
strictly admitted as one complete layout and transferred one shard at a time,
so full-device loading does not first materialize the complete model in host
memory.

### 3.11 Portable isolation and sandboxing

Windows is the most-used OS among Comfy users, so process isolation is not
allowed to be a Unix feature with a "later" asterisk. Framing and the value codec
are transport-neutral; sandboxing is a launcher ladder:

- **Plain subprocess** (all platforms, the default): dependency isolation and
  crash containment. This is what most users need - pack A's numpy cannot break
  pack B's.
- **bwrap on Linux**: deny-by-default read-only binds, GPU/network/env grants
  only by policy, private tmpfs. Two Dinkster choices: a requested sandbox never
  silently degrades to a plain subprocess (unavailable raises with
  distro-specific remediation), and a policy that would dissolve the jail is a
  build-time error, not a warning.
- **macOS / Windows**: unsandboxed at the plain-subprocess level initially;
  platform mechanisms slot in as more launchers without changing any interface.

CUDA-IPC remains Linux-only - a CUDA platform fact. On Windows, placement keeps
GPU-heavy chains co-located or crosses via shm as host memory; no node or schema
ever mentions it.

### 3.12 Assets: files as identity, not paths

**Nodes never receive filesystem paths from the graph; they receive `AssetRef`
values**, and the asset system is the only way bytes-on-disk enter a workflow. A
path means nothing on another machine; an identity means the same thing
everywhere.

**Identity is a content digest** - canonically `blake3:<64 hex>`, the same form
ComfyUI's asset DB uses, so identities interoperate. Identity survives renames,
moves, and machine hops. The catalog separates three things legacy Comfy
conflates:

- **Content** (immutable): digest, size, media type - one row per unique blob.
- **References** (the file-like layer): virtual paths in namespaces, tags,
  metadata, pointing at an identity. Virtual folders are namespace prefixes:
  list/glob/query behave like a filesystem without granting filesystem access.
- **Provenance** (per identity): source URLs, mirrors, license, expected digest.
  A workflow references an identity; any machine lacking the bytes consults
  provenance and fetches from a mirror it chooses, verifying the digest it
  already knows.

**Hashing happens once, at ingest.** Loaders and model assembly consume the
asset system's identity; re-reading or re-hashing model bytes at load time is
a defect (a 30-60GB checkpoint on a network drive must load without a full
read-through beyond what loading itself requires - a cost ComfyUI does not
pay). The only runtime content hash is lazy first-time ingest of a
not-yet-cataloged user file when its identity is actually needed, streaming
and cancellable, with the result cached through the asset system. Digests
composed from content identity bind the blake3 asset digest plus load knobs,
never a fresh file hash; sha256 appears only in publisher-issued provenance
records and small metadata/structure digests, never as content identity.

**`AssetStore` is the resolution interface**, layered like CacheStore: local
model directories (scanned with resumable hashing; the scan index lives inside
the model folder so any number of instances share hash work), disk vault,
HTTP/object stores, remote Dinkster peers. An `AssetRef` crossing a worker boundary
carries identity + metadata only; bytes move only when a worker resolves and its
local stores miss. The worker shim materializes a ref before `execute()` sees it.
An asset input's fingerprint *is* its content digest, so cache keys for
model-dependent nodes are location-independent for free.

**Distribution moves bytes by identity**: the vault ingests with streaming
verification (a failed or tampered download rolls back to nothing; reads trust
the verified name), provenance records are additively merged leads, fetching
tries candidates in order and exhaustion is a miss, and peer endpoints stream
verified bytes by digest. **A resolver provides leads, never authority** - the
digest in the graph is the only authority, so a malicious resolver can waste
bandwidth but never plant bytes. Resolvers are an extension point in two
deliberately distinct roles: materialization (`digest -> local path`) and source
discovery ("where might these bytes live") - conflating them would turn a
metadata service into an implicit trusted mirror.

**Internet P2P is a second, narrower authority intersection.** A provider's
per-artifact BitTorrent descriptor is usable globally only when the same digest
appears in its current trusted P2P enumeration, the descriptor and size agree,
the format is safe, and no observed tombstone applies. License and gated fields
are metadata, not transfer conditions. The resulting lease is capped
at six hours even if the provider asks for longer. Scope independently
gates the network: the default `lan-and-internet` may enable DHT, PEX, TCP, uTP,
provider-approved trackers, UPnP, NAT-PMP, or PCP. Closing that scope removes all
internet capabilities while leaving enabled LAN discovery and HTTP fallback
available. Metered policy, seeding ratio bytes, active seed time, and durable
counters are evaluated before each global announcement rather than trusted to
provider data. Resolver subscriptions are not provider authority unless their
P2P-trust flag is explicit; their separate license-authority flag describes
metadata. Their version 1 adapter emits trackerless snapshots and treats
complete-refresh omission as a tombstone. Partial or failed refreshes cannot
renew authority. Shared staging admission reserves missing bytes across LAN and
global downloads without counting no-copy seed files.

**Filesystem mounts: N explicit directory grants, runtime-mutable.** Real
directories the operator granted, each with an id and mode, cataloged under
`mounts/<id>/...` and referenced by digest - real paths never enter documents, so
granting, revoking, or moving a mount never invalidates one. The live table is
the runtime authority over a durable TOML record; mutation endpoints are
policy-gated off by default (a remotely reachable server should not accept
filesystem grants from anyone who can reach the port). Workers see grants live
through an atomic snapshot file - no restart, because an idle torch process
holds real VRAM. A ComfyUI install's own input/output directories derive as
well-known compat mounts.

### 3.13 Collections and repetition: lists, combinators, and regions

v1's list feature bundles three orthogonal things into one executor codepath
(unequal lists silently clamp to the last element; `INPUT_IS_LIST` flips an
invisible per-class calling convention; loops shipped engine complexity but no
core nodes because there was no collection value to accumulate into). Dinkster
separates the three concerns, and each becomes small:

**Collections as data: `list<T>`.** A list is an ordinary envelope containing
element envelopes, so element type and count are interrogable anywhere and the
list fingerprint derives from element fingerprints - caching is element-aware by
construction. The schema says `list<IMAGE>`, so the hidden calling convention
does not exist. **There is no implicit coercion in either direction**: a
list-to-scalar edge is a document-time type error whose diagnostic names the fix,
never executor magic. The type grammar is recursive (`list<list<core.int>>`);
canonical runtime ids share one parser between schema and values.

**Combining as combinator nodes.** Where v1 guessed, the graph says it:
`CrossProduct` produces two aligned lists (the explicit sweep), batch<->list
conversions make the tensor-batch-vs-list distinction visible, and zip is
deliberately NOT a node - zip semantics live in a region's binding over element
ports, hard error on length mismatch. Combinators are dumb nodes producing list
values - zero engine involvement, so packs ship their own. Generic (type-variable)
interfaces solve at the worker boundary from the authoritative runtime type ids
on input envelopes, never at document time, so elaboration stays pure and
validation stays advisory.

**Repetition as graph regions.** Mapping is never implicit; it is a visible
region boundary in the document, and the engine gets exactly one primitive:
*expand this subgraph once per binding set, with an optional sequential state
chain* - the same expansion machinery as dynamic outputs (3.1). Three region
kinds as validation profiles over that one primitive:

- **Map** binds elements of zipped (or cross-product) list inputs; non-list
  inputs are visible broadcast constants; no carried state means iterations may
  run in parallel.
- **Fold** declares state port pairs; iteration N's state feeds N+1; the final
  state exits the region. Accumulation is carried state or a built-in gather -
  no special node types.
- **While** is a fold with a boolean continue output and a mandatory,
  engine-enforced max-iteration cap; reaching the cap with continue still true
  is a loud error, never silent truncation.

Because each iteration is an ordinary invocation, existing invariants pay rent
with no new rules: per-item caching (change one element of a 40-item list, one
iteration's dependents re-run), per-item parallelism under existing lanes and
memory governance, per-item errors with completed items retained. Iteration node
ids are namespaced `region[3]/node`; the grammar is closed - `/`, `[`, `]` are
banned in document node ids at every level, so parsing needs no heuristics.
Zero-length lists mean zero iterations, which is also the conditional story: a
zero-or-one-element list is "execute this subgraph, or don't". Regions emit
expansion/finish events; list output descriptors carry runtime length - length
is data, never schema. Node authors stay oblivious (H9): no node ever knows it
is inside a region, so every existing node works in one on day one. Bounds
(H11): map/fold/while and core combinators ship as core; raw expansion is not a
public extension API - packs extend via combinators and ordinary nodes.

The division of labor is strict: the **value level** (`list<T>`) carries
cardinality known at runtime; the **schema level** (dynamic families, 3.1)
carries interface shape known at authoring time; the **graph level** (regions)
carries repetition of computation. With `list<T>`, the legacy quarantine (3.8)
translates `INPUT_IS_LIST`/`OUTPUT_IS_LIST` honestly as `list<T>` sockets,
calling the function exactly as v1 does - the hard part was only ever the
implicit executor zipping, which Dinkster does not have.

Regional samplers that alternate base and masked denoising over one sigma
schedule are compatibility orchestration, not another sampling engine. A
recognized foreign subgraph may collapse to one orchestrator that partitions
masks, invokes `CustomSamplingRuntime` for every interval, and composites the
results. Solver execution, admission, cancellation, previews, and family
capabilities remain properties of that single custom-sampling seam. General
repetition and multi-region authoring remain graph regions rather than new
family-specific sampling paths.

Sampler nodes select behavior from the runtime's sampling capabilities, not
membership in a model-family list or the presence of text encoders and codecs.
A diffusion-only runtime can inherit the same KSampler composition as an
assembled checkpoint. Family-specific sigma spaces and conditioning adapters
provide model semantics; shared sampling checks and numerical receipts govern
cross-cutting features on every entry point.

The core inference package owns backend-agnostic descriptors, option schemas,
assembly plans, checkpoint inspection, and family detection. It does not own
executing numerical mirrors. `dinkster-inference-torch` binds the shared
sampler descriptors and model contracts to torch kernels; a sibling backend
that does not use torch can bind the same declarations to its own kernels.
Only the documented float64 `schedules.py` pair remains because those functions
are part of schedule declaration rather than backend execution.

### 3.14 Model interposition: patch programs as values, not mutation

Evidence base: a usage-ranked census of heavy ModelPatcher consumers (RES4LYF,
KJNodes, Easy-Use, WanVideoWrapper, UltimateSDUpscale). ComfyUI's patch taxonomy
is right - packs use all of it - but packs bypass the official surface for four
recurring reasons: clone-scoped registration over shared underlying state means
"patch a clone" quietly mutates siblings; extending patch *calculation* requires
subclassing ModelPatcher; addressing is hardcoded dotted paths that break per
model family; and process-global knobs (ops overrides, sampler swaps) have no
scoped alternative and are racy under any parallelism.

Dinkster keeps the taxonomy and fixes the model:

- **A patched model is a derived value**: an immutable base `ResourceHandle` plus
  an ordered, declarative **patch program**. Applying a patch never mutates
  anything - it produces a new envelope whose fingerprint is
  `h(base fp, program fp)`. Patched models cache like any value, compare cheaply,
  cross boundaries by description, and the clone-aliasing bug class is
  unrepresentable.
- **Entry taxonomy from ComfyUI, proven in the field**: weight-delta,
  module-replace, site-patch, sampler-phase wrapper (a closed enum), lifecycle
  callback, structured options (merged data, not blind dicts).
- **Symbolic sites, not dotted paths.** Each model family's loader publishes a
  site map as registry data; entries address sites symbolically; an unknown site
  is a bind-time diagnostic naming the available sites, not a KeyError
  mid-sample.
- **Code refs, not closures.** Interposition functions are registered pack
  capabilities; entries carry the ref, tensors travel as ordinary payloads. If
  the executing worker lacks the pack, placement co-locates or falls back to a
  visible-perf-diagnostic RPC, never silently.
- **Weight application strategies are registered adapters** (LoRA kinds,
  quantized-aware deferred application, fused-QKV splitting) consulted by the
  parameter's storage kind - no subclassing, no global swaps.
- **Scoped runtime requirements replace global knobs.** A program may declare
  `requires` (fp16 accumulation, ops override); the lane scheduler serializes
  conflicting requirement sets per device, turning a process-global race into an
  admission constraint.
- **Transient transforms derive** (per-tile context entries in the fingerprint),
  and **new model families are loaders** publishing a handle + site map, not
  surgery on someone else's layout assumptions.

In dev mode the host fingerprints base weights around a run, turning silent
cross-clone corruption into a named diagnostic.

### 3.15 Absence: typed "no value" instead of ExecutionBlocker

ComfyUI's `ExecutionBlocker` is a *value* that means an *engine state*: it hides
inside lists, replays oddly from cache, and surfaces diagnostics at whatever
downstream consumer trips on it. Four distinct situations collapse into one
sentinel. Dinkster keeps them separate:

- **Not demanded**: the engine only plans ancestors of the run's targets; an
  undemanded node has no state at all.
- **Present**: an ordinary envelope.
- **Deliberately absent**: an ordinary envelope of type `core.absent` whose meta
  carries the origin, reason, and stood-in type. It caches, fingerprints, and
  crosses boundaries like any value.
- **Skipped**: an engine decision, recorded in the run result and a
  `node_skipped` event - never a value smuggled through node returns.
- **Failed**: an error attributed to the failing node.

**Producing**: an output declared `optional=True` may return the `ABSENT` marker
from `execute()`; the worker builds the typed envelope. Node authors never
construct absent envelopes (H9). **Consuming**: each input declares `on_absent`,
applied by the engine *before* invocation - `skip` (default for required inputs:
node does not run, outputs become absent carrying the *root* origin, so a cascade
ten nodes deep still names the producer), `omit` (default for optional inputs:
the input becomes exactly the unconnected shape, so cache keys match a document
where the edge never existed), `accept` (node receives plain `None`, opt-in),
`fail` (loud, names origin and reason; beats skip regardless of input order).

Declared maybe-absence on outputs is a schema property frontends can render;
engine-derived absence after a skip is runtime state propagation - which is why
cache-hit validation only accepts a stored absent for outputs declared optional.
Absence is not control flow: choosing which branch to even compute is
demand-driven laziness, which belongs to region expansion (3.13) - a conditional
is a region expanded zero or one times. Absence is the data substrate those
features compose with.

---

## 4. What we reuse (and what we don't)

Reuse (adapt, with attribution):

- **pyisolate internals, forked into `dinkster-workers`**: venv provisioning, process
  supervision/RPC skeleton, and the shm + CUDA-IPC tensor serializer. Forked, not
  depended on: its transparent proxy-object RPC model does not match Dinkster's
  explicit Invocation + envelope boundary (its serializer becomes our payload
  transports; the magic proxying evaporates). No transparent proxies anywhere:
  cross-worker resource interrogation, if ever needed, is a small versioned
  engine-mediated query set on ResourceHandle. Version-mismatched workers are
  workers that cannot negotiate `shm`/`cuda-ipc`; the fallback tier is a portable
  binary tensor transport, with JSON reserved for small non-tensor values.
- **`comfy` library** (model management, samplers, model detection) via
  `dinkster-compat-comfy` - the *transitional* inference backend and the test oracle
  for the native inference program (docs/native-inference-plan.md). Dinkster grows a
  typed native inference substrate that reuses Comfy's proven algorithms and
  detection knowledge as reference material while rejecting its untyped,
  global-state architecture; comfy demotes from foundation to optional backend as
  native stages land.
- **comfy-aimdo** as an optional residency backend under the governor (3.10): its
  page-faulting VBAR mechanism is exactly the mechanism layer a governed shedder
  wants; its implicit recency-only policy is not adopted - the governor owns
  policy. Its process-locality is the gap 3.10's discovery + lease layer covers.
- **`comfy_api/latest/_io.py`** as the semantic starting point for the schema
  model (its sin is compiling down to v1).
- **`comfy_api_nodes` pydantic contracts** as seed data for RemoteNodeDefinitions;
  its HTTP client core as the transport seed.
- **Frontend `@dinkster/core` `schema/model.ts`** as the wire-format reference spec.
- **`comfy_execution/caching.py`** eviction ideas re-expressed against CacheStore.
- **kijai/ComfyUI-MemoryVisualization** as the requirements spec for the memory
  observability surface (3.10): every private internal it scrapes marks a field
  the native detail contract must carry.
- **ComfyUI `app/assets`** concepts (3.12): the asset/reference split, canonical
  `blake3:` identity, resumable hashing, tag-based category mapping. Discarded:
  reference-UUID as API identity, opt-in hashing, the missing folder-query and
  provenance layers.
- **pyisolate `_internal/sandbox.py`** as the reference for the Linux bwrap
  launcher (3.11).
- **comfy-runner** as optional development tooling only (provisioning ComfyUI
  checkouts for the compat pack, regression runs). Never a core dependency.

Not reused: v1 dict schema anywhere in core, `execution.py`'s recursive-executor
shape, `PromptQueue`/server entanglement, `NODE_CLASS_MAPPINGS` import-time
mutation, in-tree provider node modules.
