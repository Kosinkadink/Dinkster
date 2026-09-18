# Archived founding design

This is the initial `DESIGN.md` from commit `2038ad61`, retained as research
history. Its milestones, timelines, and reuse decisions are not current product
direction. The authoritative architecture rationale is [`DESIGN.md`](../../DESIGN.md),
and current capability status is in the [`SUPPORTED.md` index](../../SUPPORTED.md).

# Dinkster - Design and Plan

Dinkster is a clean-slate ComfyUI backend. Like Dinkster-Frontend, it is not shackled by
backwards compatibility: legacy behavior is something we *import or adapt at a
boundary*, never something we build the core around. This document is the founding
plan: what is wrong with the current backend, the architecture that fixes it, what we
reuse, and the milestone ladder.

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
   strings everywhere (`execution.py`, `comfy_execution/graph.py`). The modern V3
   schema (`comfy_api/latest/_io.py`: `Schema`, typed `Input`/`Output`, `ComfyNode`)
   exists but is *down-converted to v1* for execution and for `/object_info`
   (`create_input_dict_v1`, `GET_NODE_INFO_V1`). V3 features that don't fit v1 are
   smuggled through reserved type strings (`COMFY_MATCHTYPE_V3`,
   `COMFY_AUTOGROW_V3`, ...) that the frontend must parse back out. The richest
   schema is the *derived* one; the authoritative one is the weakest.

2. **Execution, caching, queueing, and serving are braided together.**
   `execution.py` (~1300 lines) contains the executor, cache orchestration, v1 AND v3
   input resolution, validation, list/batch mapping, and the `PromptQueue`;
   `server.py` owns the queue instance; `main.py` owns the worker loop. None of it is
   separately usable or testable.

3. **Values on edges are raw Python objects.** A LATENT is a dict, an IMAGE is a
   bare torch tensor, an INT is an int. Nothing on an edge can be interrogated
   ("what type is this really?"), fingerprinted for caching, or serialized across a
   process/machine boundary without ad-hoc per-case code. This is exactly what the
   pyisolate integration branch had to bolt on afterwards (its `adapter.py`
   serializers and `runtime_helpers.py` proxies exist because values carry no
   contract).

4. **Node execution assumes same-process, same-venv.** The engine calls
   `getattr(obj, obj.FUNCTION)(**inputs)` directly. Process isolation
   (`origin/pyisolate-support`, PR ComfyUI#13646) is mature but necessarily
   *wrapped around* the engine: host stubs pretend to be v1 classes so the engine
   doesn't notice. In Dinkster the boundary is the engine's native shape, and
   in-process is just the trivial transport.

5. **There is no defined extension API.** Custom nodes mutate `NODE_CLASS_MAPPINGS`,
   monkey-patch server routes, samplers, and model management at import time. Hooks
   exist only as "whatever happens to be reachable". Nothing is predictable, so
   nothing is safely evolvable.

6. **Partner (API) nodes live in-tree** (`comfy_api_nodes/`: 40 provider modules,
   36 pydantic schema files) and ship on the core release cadence, even though they
   are almost pure data: schema + typed request/response contract + polling/upload
   conventions against `/proxy/<provider>/...`.

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
  extension API as third-party packs (the frontend's "core eats the same dogfood"
  rule). If core needs a hook an extension can't have, the contract is wrong.
- **One canonical representation per concept; no unions of shapes.**
- **Advisory vs authoritative split.** Structural graph validity is authoritative;
  type compatibility on edges is checked at plan time but the type *system* is
  interrogable data, not isinstance checks scattered through the engine.
- **Pre-1.0 formats are unstable and revised in place** (same policy as the
  frontend's `formatVersion: 1`). Real migration guarantees start at first public
  release; after that, wire changes are additive-first with explicit version bumps.
- **Demo-first.** Every milestone ends in something runnable end-to-end.
- **hazards.md is a living document** recording load-bearing invariants, why they
  exist, and the failure mode if broken. Seeded in `docs/hazards.md`.

---

## 3. Architecture

Python >= 3.12, `uv` workspace monorepo, strict typing (`pyright` strict + `ruff`),
asyncio-native core, `pydantic` for wire models only (core models are plain
dataclasses/protocols so the engine has no serialization framework in its hot path).

### Package layout

```diagram
+----------------------------------------------------------------------+
|                            dinkster-server                              |
|        HTTP + WS API, queue, sessions, events, artifact serving      |
+---------------+--------------------------------------+---------------+
                |                                      |
+---------------v---------------+      +---------------v---------------+
|         dinkster-engine          |      |           dinkster-ext           |
|  planner, scheduler, cache    |      |  manifests, typed registries, |
|  orchestration, invocation    |      |  lifecycle, capability grants |
+------+---------------+--------+      +---------------+---------------+
       |               |                               |
+------v------+ +------v------+        +---------------v---------------+
|dinkster-workers| |dinkster-caches |        |          node packs           |
| in-process, | | memory-lru, |        | dinkster-nodes-std,              |
| venv/subproc| | ram-aware,  |        | dinkster-compat-comfy,           |
| remote      | | disk/CAS,   |        | dinkster-partner-nodes,          |
|             | | remote      |        | third-party packs             |
+------+------+ +------+------+        +---------------+---------------+
       |               |                               |
+------v---------------v-------------------------------v---------------+
|                     dinkster-values  +  dinkster-schema                    |
|   value envelopes, type registry, codecs, payload transports  |      |
|   node/type schema model (the single source of truth), wire enc      |
+----------------------------------------------------------------------+
```

Dependency rule (enforced in CI): arrows only point downward. `dinkster-schema` and
`dinkster-values` depend on nothing above them and have **no torch dependency**; the
engine imports no server code; workers/caches implement engine protocols; node packs
see only the extension API.

- **`dinkster-schema`** - the V3-native node schema model: typed inputs/outputs/widgets,
  one *ordered interface list* (true interleave of inputs/widgets/sections), real
  output IDs (no positional tuples), MatchType/template constraints expressible on
  inputs AND outputs, dynamic constructs (autogrow, dynamic combo/slot) as structured
  objects, node metadata (category, deprecation + replacement rules, isolation
  requirements). Plus the wire encoding: the `/object_info` successor, isomorphic to
  the frontend's normalized `NodeSchema`, carrying a `schemaVersion` field.
- **`dinkster-values`** - the value envelope system (section 3.2).
- **`dinkster-graph`** - prompt/graph model, structural validation, plan building
  (topological order, lazy edges, partial-execution scoping). Pure: no IO, no torch.
- **`dinkster-engine`** - the scheduler and the two protocols everything plugs into:
  `Worker` (invocation) and `CacheStore`. Also progress/event emission and
  cancellation. Never imports node code.
- **`dinkster-workers`** - implementations of `Worker`: `InProcessWorker` (trivial),
  `IsolatedWorker` (subprocess in its own venv; pyisolate-derived RPC + tensor
  transport), `RemoteWorker` (same invocation protocol over the network; later).
- **`dinkster-caches`** - implementations of `CacheStore`: memory LRU, RAM-pressure
  (port of `RAMPressureCache` ideas), disk content-addressed store, remote store
  (later). Composable as layers.
- **`dinkster-ext`** - the extension API: pack manifests, typed registries
  (nodes, value types, codecs, payload transports, workers, cache backends, server
  routes, event subscribers, model providers), explicit lifecycle, capability
  grants.
- **`dinkster-server`** - aiohttp HTTP/WS server, job queue, multi-client sessions,
  the native protocol (section 3.5), artifact upload/serving.
- **Node packs** - `dinkster-nodes-std` (primitives, math, image basics),
  `dinkster-compat-comfy` (wraps the existing `comfy` library for real model
  loading/sampling; the pragmatic path to day-one usefulness),
  `dinkster-partner-nodes` (section 3.6; in-repo initially, hard package boundary,
  designed to split into its own repo).

### 3.1 Schema as the single source of truth

Node authors declare a V3-style schema (`define_schema()` returning a typed
`NodeSchema` object) and an `execute()` (sync or async). The engine, the validator,
the cache-key builder, the wire encoder, and the isolation layer all consume the
*schema object*. There is no `INPUT_TYPES` dict anywhere; a v1 projection exists only
if/when we ship a legacy-frontend facade, and it lives in a quarantined
`dinkster-compat-v1` module designed for deletion (mirroring the frontend's quarantined
V1 adapter).

The frontend already parses today's smuggled V3 markers into a small closed
`TypeExpr` model (`concrete | union | wildcard | variable`). Dinkster's schema uses that
same closed model natively, so the wire encoding becomes a near-identity mapping and
the frontend swaps its adapter for a thin decoder.

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
  codecs, and conversion edges (e.g. mask -> image). The backend can finally answer
  "what is on this edge" without isinstance guessing, and generic tooling (previews,
  converters, validators) hangs off the registry instead of special cases.
- **PayloadRef + transports.** A payload is accessed through a transport:
  `inline` (small values, JSON-safe), `pyobj` (same-process fast path - zero
  overhead when nothing crosses a real boundary), `shm` (CPU tensors via /dev/shm,
  from pyisolate's tensor serializer), `cuda-ipc` (GPU tensors, Linux), `file`,
  `cas` (content-addressed blob, local or remote). Which transport is used is a
  *placement decision*, not a node concern.
- **Fingerprints replace `IS_CHANGED` guessing.** Cache keys are
  `(node signature, resolved input fingerprints)`; because fingerprints are content
  hashes carried by the envelope, cache entries are location-independent and can be
  shared across processes and machines.
- **Handles for non-serializable resources.** Models/VAE/CLIP are `ResourceHandle`
  values: the envelope carries identity + residency (which worker owns it), and the
  engine schedules consumers accordingly (or proxies calls, as the isolation branch's
  `runtime_helpers` do today). This is the honest version of what pyisolate's
  ModelPatcher proxies bolt on.

### 3.3 Boundary-first execution engine

The engine never calls node functions. It builds a plan from the graph, then for each
node emits an `Invocation` to a `Worker`:

```python
class Worker(Protocol):
    async def prepare(self, node_types: list[NodeTypeRef]) -> None: ...
    async def invoke(self, inv: Invocation) -> InvocationResult: ...
    async def cancel(self, inv_id: InvocationId) -> None: ...
    # Invocation = (node_type, node_state_ref, {input_id: ValueRef}, context)
    # InvocationResult = {output_id: Value} | NodeError, plus ui/progress stream
```

- `InProcessWorker` resolves ValueRefs to Python objects and awaits `execute()` -
  same performance profile as today when everything is local.
- `IsolatedWorker` is the pyisolate-support core, made native: per-pack venv
  (manifest-declared dependencies), bidirectional RPC, shm/CUDA-IPC tensor passing.
  Because inputs are already envelopes with transports, the isolation layer needs no
  per-type serializer bolted on.
- `RemoteWorker` (later) is the same protocol over a network transport, with the CAS
  transport carrying payloads. Nothing in the engine changes.
- A **placement policy** assigns nodes to workers (pack isolation requirements,
  resource residency, user config). Scheduling stays output-driven with lazy edges
  and partial-execution scoping, like today's `ExecutionList`, but as a pure planner
  over the graph package.

### 3.4 Caching as a first-class interface

```python
class CacheStore(Protocol):
    async def get(self, key: CacheKey) -> CacheHit | None: ...
    async def put(self, key: CacheKey, outputs: dict[OutputId, Value], cost: Cost) -> None: ...
    async def evict(self, pressure: PressureSignal) -> None: ...
```

Keys derive from schema signature + input fingerprints (section 3.2). Stores compose
in layers (memory -> disk CAS -> remote). Eviction policies (classic per-prompt, LRU,
RAM-pressure) are store concerns, not engine concerns. Because values are envelopes,
a disk/remote cache can persist payloads via the CAS transport and rehydrate them on
any worker - cross-run and cross-machine caching falls out of the design instead of
being impossible.

### 3.5 Server and native protocol

- aiohttp (same as ComfyUI - proven for WS + binary frames; FastAPI adds a stack we
  don't need on the hot path).
- Native schema endpoint replacing `/object_info`, emitting the NodeSchema-isomorphic
  wire format with `schemaVersion`.
- Jobs identified as `(clientId, jobId)`; the frontend already treats
  `(connectionId, promptId)` as the execution identity, so multi-backend/multi-client
  is first-class from day one.
- WS protocol with the feature-flag handshake the frontend already sends, typed event
  catalog (status, node state map - never a single execution cursor - progress,
  previews with binary frames, errors with structured provenance).
- Optional `dinkster-compat-v1` facade (v1 `/object_info` + `/prompt`) only if we want
  to demo against the legacy frontend; quarantined and deletable. Primary integration
  target is Dinkster-Frontend's native decoder.

### 3.6 Extension API: predictable, no hacks required

- **Manifest-first packs** (evolving the isolation branch's `manifest_loader`):
  declared dependencies (own venv by default), declared capabilities (needs GPU,
  needs network, provides routes, provides types), declared entry points. No
  import-time side effects on the host process - loading a pack cannot touch the
  server, the engine, or other packs except through registries.
- **Typed registries** for every extension point: node types, value types, codecs,
  payload transports, workers, cache stores, server routes (namespaced), event
  subscribers, model providers/samplers (via the compat pack's surface initially).
  Registration is declarative and inspectable; conflicts are diagnostics, not
  last-import-wins.
- **Versioned, frozen API surface**: packs import `dinkster.api.v1` only; internal
  modules are underscore-private. The current `comfy_api` versioning idea, but as the
  *only* door instead of a side door next to an open wall. Since this is Python,
  people *can* still reach into internals - the goal is that they never *need* to,
  and that everything reachable through the door is covered by compat tests.
- **Isolation by default** makes the contract honest: a pack that only uses the API
  works isolated; a pack that reaches into host internals visibly breaks in dev, not
  in users' installs.

### 3.7 Partner (API) nodes as data

Today's `comfy_api_nodes` already reduce to: schema + pydantic request/response
contract + endpoint path under `/proxy/<provider>/` + polling/upload conventions.
Dinkster makes that literal:

- A **`RemoteNodeDefinition`**: NodeSchema + typed HTTP contract + operation kind
  (sync / poll / upload-then-poll) + media in/out mapping - pure data (JSON/pydantic),
  no imperative node code for the standard cases.
- A generic **remote-execution runtime** in `dinkster-partner-nodes` interprets
  definitions: auth, retries, rate limits, polling, progress, interruption, media
  upload/download into value envelopes (adapting `comfy_api_nodes/util/client.py`,
  which is already well-factored).
- Definitions load from a directory/registry at startup and can be **updated
  out-of-band** - new provider endpoints ship without a Dinkster release. An escape
  hatch allows imperative nodes for genuinely weird providers, still inside the pack
  boundary. The package sits behind the same extension API as any third-party pack,
  so splitting it into its own repo later is `git mv`, not surgery.

---

## 4. What we reuse (and what we don't)

Reuse (adapt, with attribution):
- **pyisolate** as a dependency or vendored core: venv management, bidirectional RPC,
  shm + CUDA-IPC tensor transport.
- **ComfyUI `origin/pyisolate-support` (PR #13646)** concepts: manifest loading,
  serializer adapters, resource proxies - reborn as native envelope transports and
  ResourceHandles instead of retrofit wrappers.
- **`comfy` library** (model management, samplers, model detection) via
  `dinkster-compat-comfy` - we do not rewrite inference; we orchestrate it. Rewriting
  the inference stack is explicitly out of scope for Dinkster's first year.
- **`comfy_api/latest/_io.py`** as the semantic starting point for the schema model
  (it is decent; its sin is compiling down to v1).
- **`comfy_api_nodes/apis/*`** pydantic contracts as seed data for
  RemoteNodeDefinitions; `util/client.py` as the transport core.
- **Frontend `@dinkster/core` `schema/model.ts`** as the wire-format reference spec.
- **`comfy_execution/caching.py`** eviction ideas (LRU generations, RAM-pressure
  scoring) re-expressed against CacheStore.

Not reused: v1 dict schema anywhere in core, `execution.py`'s recursive-executor
shape, `PromptQueue`/server entanglement, `NODE_CLASS_MAPPINGS` import-time mutation,
in-tree provider node modules.

---

## 5. Milestones (demo-first, thin slice through all layers first)

- **M0 - skeleton end-to-end.** Monorepo scaffold (uv workspace, pyright strict,
  ruff, pytest, CI). `dinkster-schema` + `dinkster-values` + `dinkster-graph` +
  `dinkster-engine` + `InProcessWorker` + memory `CacheStore`. `dinkster-nodes-std` toy
  pack. Demo: CLI runs a small graph (ints, strings, image ops via PIL/numpy),
  caching observable, re-run hits cache, envelope fingerprints proven by tests.
- **M1 - server + native protocol.** Queue, WS events, native schema endpoint, job
  identity `(clientId, jobId)`, previews. Demo: Dinkster-Frontend loads Dinkster's schema
  through a native decoder and submits/monitors a job. (Coordination point with the
  frontend thread: the decoder should be near-identity by design.)
- **M2 - the boundary proves itself.** `IsolatedWorker`: one pack in its own venv,
  shm tensor transport, per-pack manifests, placement policy. Demo: the same M0
  graph runs with the image pack out-of-process; zero engine changes.
- **M3 - real inference.** `dinkster-compat-comfy` pack wrapping comfy model
  loading/sampling; ResourceHandle values for models; RAM-pressure + disk CAS cache
  layers. Demo: full SD/SDXL text-to-image through Dinkster, driven from
  Dinkster-Frontend.
- **M4 - partner nodes as data.** RemoteNodeDefinition format + generic runtime;
  port 2-3 providers (e.g. OpenAI images, one poll-based video provider) from
  `comfy_api_nodes` as pure definitions. Demo: update a provider definition without
  restarting/releasing.
- **M5 - distribution.** `RemoteWorker` over network transport, CAS payload store,
  shared remote cache. Demo: two machines, one graph, cache hits across both.
- **M6 - extension API freeze candidate.** `dinkster.api.v1` surface review, compat
  test suite, docs, third-party pack template. hazards.md audit.

---

## 6. Open questions (deliberately deferred, tracked here)

- Exact fingerprinting strategy for large tensors (full hash vs sampled hash vs
  producer-lineage keys) - M0 starts with full hashes on CPU, revisit with numbers.
- Whether `dinkster-compat-comfy` pins a comfy version per release or tracks master.
- Windows support level for shm/CUDA-IPC transports (pyisolate is Linux-first).
- Whether a `dinkster-compat-v1` frontend facade is ever worth building, or whether
  Dinkster-Frontend's native decoder makes it dead on arrival.
- Auth model for partner nodes (keep comfy.org proxy vs direct provider keys).
