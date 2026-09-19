# Dinkster Unified Training Architecture

Adoption record (2026-07-29): committed to Dinkster as the training program of
record by inference coordinator T-019f9e63-822e-762f-ba79-ab160235078c after
the graph/worker-hybrid reconciliation (decision 12, user-fixed) and a final
Oracle READY review. Decisions 1-11 and 13-20 are recommendations the user
may still veto; decision 12 and the extension-design.md 3.2
reconstruction-recipe invariant are fixed. The durable-journal shape
(decision 18) requires a joint decision with the backend/extension-S5
program before either ships. Research evidence lives in the private
Kosinkadink/dinkster-research repository (dives/report-t1..t3); see
[the workspace research index](https://github.com/Kosinkadink/comfy-vibe-station/blob/main/notes/research/README.md).

Status: DESIGN ONLY. This document proposes contracts and implementation slices. It
does not claim that Dinkster currently implements training.

Dinkster source inspected: `/home/kosin/comfy-vibe-station/pr-tracker/stations/station13/Dinkster`
(at the commit recorded in the adoption above).
Research evidence: the three reports under `/home/kosin/node-analysis/dives`,
archived in Kosinkadink/dinkster-research under `dives/`.
All Dinkster paths below are relative to the inspected Dinkster root. Report paths are
absolute so the evidence can be checked independently.

## 1. Decision

Dinkster should have one semantic model system and two execution-policy families,
not one inference implementation stretched across backpropagation and not a
second training model stack.

- **SHARED** means one definition or host service is authoritative for both
  inference and training: family identity, architecture and parameter names,
  checkpoint/storage metadata, latent and conditioning contracts, prediction
  math, artifact identity, extension composition, graph/job transport, memory
  governance, and the durable journal substrate.
- **DUAL-PATH** means one shared definition is bound to an explicit execution
  policy before execution: inference versus adapter training versus full
  training, with different kernel, materialization, graph-lifetime, residency,
  and train/eval rules.
- **TRAINER-ONLY** means the training session owns policy and durable state:
  data iteration, bucketing, objectives, timestep/noise draws, losses,
  optimization, EMA, validation cadence, checkpoint cadence, and distributed
  semantics.

This boundary follows the strongest common result from all three studies.
ComfyUI's in-graph trainer collides with global inference policy and has partial
resume and memory accounting
(`/home/kosin/node-analysis/dives/report-t1-comfyui-training.md:49-61`). Kohya
shares substantial adapter orchestration but duplicates procedural full-model
trainers (`/home/kosin/node-analysis/dives/report-t2-kohya.md:16-24,39-40`).
ai-toolkit shares model infrastructure but separates training policy, and its
normal low-memory path trains float adapters over a frozen quantized base
(`/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:19-22`).

The first implementation should be an extension-provided training session for
one image family and LoRA, executed by a dedicated isolated training worker and
orchestrated by ordinary graph composition. A graph node may create or advance
the session, but it never owns a microbatch loop, tensor state, or a live
autograd graph. One graph loop iteration is one checkpoint interval; the
supervised session, complete resume state, event journal, and governor-accounted
hot residency remain below that graph boundary.

## 2. Ownership matrix

The matrix is normative. A row marked DUAL-PATH still shares the definition
named in that row; only execution policy and state differ.

| Subsystem | Boundary | Dinkster integration point | Evidence and reason |
|---|---|---|---|
| Family detection and registration | **SHARED** | Extend `ModelFamily`/`FamilyRegistry`; detection is evidence-ranked and ambiguity is loud (`packages/dinkster-inference/src/dinkster_inference/families.py:1-13,69-89,108-151`). | Training and previews need one family identity. All three analyses converge on shared family definitions (`/home/kosin/node-analysis/dives/report-t2-kohya.md:145-155`; `/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:270-275`). |
| Architecture, component plans, and canonical parameter identities | **SHARED definition, DUAL-PATH binding** | `ComponentPlan` already normalizes model keys, source keys, dtypes, transforms, and quant metadata (`packages/dinkster-inference/src/dinkster_inference/assembly.py:122-175`); native constructors receive an `Operations` factory (`packages/dinkster-inference-torch/src/dinkster_inference_torch/operations.py:1-18,64-115`). | Kohya needs stable trainable parameter names and local model control (`/home/kosin/node-analysis/dives/report-t2-kohya.md:43-52,149-150`); ai-toolkit likewise separates common model metadata from runtime mode (`/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:84-116`). |
| Weight inspection, storage layout, quant metadata, and source identity | **SHARED** | Keep header-first `WeightSource` and assembly mappings authoritative (`packages/dinkster-inference/src/dinkster_inference/weights.py:1-7,110-138`; `packages/dinkster-inference/src/dinkster_inference/assembly.py:122-149`). | Storage dtype is not execution dtype, and training should not create a second checkpoint naming universe (`/home/kosin/node-analysis/dives/report-t2-kohya.md:149-150`; `/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:270-272`). |
| Prediction parameterization, sigma/timestep conversion, latent and conditioning meaning | **SHARED primitives, DUAL-PATH orchestration** | Reuse parameterization math and family latent/sampling descriptors (`packages/dinkster-inference/src/dinkster_inference/parameterizations.py:1-17,27-94`; `packages/dinkster-inference/src/dinkster_inference/families.py:69-88`). Training bindings add target construction, not competing math. | Kohya separates shared scheduler/model conventions from training density/target and inference solver orchestration (`/home/kosin/node-analysis/dives/report-t2-kohya.md:164-166`). ai-toolkit family support also requires train scheduler, target, latent, cache, and preview conventions (`/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:100-116`). |
| Grad/no-grad, train/eval state, and nested preview mode | **DUAL-PATH** | Bind an explicit execution mode through the typed operation/runtime seam; do not hide mode reads in `forward` (`packages/dinkster-inference-torch/src/dinkster_inference_torch/operations.py:12-18`). Current native VAE code deliberately contains no blanket no-grad guard (`packages/dinkster-inference-torch/src/dinkster_inference_torch/autoencoder_kl.py:34-43`). | ComfyUI has to puncture executor-wide `inference_mode` (`/home/kosin/node-analysis/dives/report-t1-comfyui-training.md:322-348`); kohya preview must preserve mode, RNG, and placement (`/home/kosin/node-analysis/dives/report-t2-kohya.md:176-193`). |
| Effective-weight computation and kernel dispatch | **DUAL-PATH** | Add training-capability implementations behind `Operations`; current fp8 fused matmul explicitly refuses autograd (`packages/dinkster-inference-torch/src/dinkster_inference_torch/quant_linear.py:190-210`). | Quantized inference capability does not prove backward capability. ComfyUI has separate effective-weight and bypass paths (`/home/kosin/node-analysis/dives/report-t1-comfyui-training.md:348-419`); ai-toolkit distinguishes frozen-base input gradients from QAT masters (`/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:169-177`). |
| Static inference patches versus trainable adapters | **SHARED codecs/math, DUAL-PATH attachment** | Reuse typed patch algebra and LoRA dialect normalization, but do not make immutable `PatchSet` values optimizer state (`packages/dinkster-inference/src/dinkster_inference/patches.py:1-25,52-82`; `packages/dinkster-inference/src/dinkster_inference/lora.py:1-18,58-111`). | A trainable adapter has live parameters, dropout, optimizer groups, and mutable lifecycle, unlike an inference overlay (`/home/kosin/node-analysis/dives/report-t1-comfyui-training.md:701-724`; `/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:306-318`). |
| Weight residency and graph lifetime | **DUAL-PATH under SHARED governor** | Current `WeightLease` is valid for one module forward and leased values may not escape (`packages/dinkster-inference-torch/src/dinkster_inference_torch/residency.py:152-186`). Training needs backward rematerialization or an explicitly budgeted graph lease. | ai-toolkit proves forward staging plus backward restaging for frozen weights (`/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:179-191,279-304`). ComfyUI's inference-shaped offload is unmanaged at training lifetimes (`/home/kosin/node-analysis/dives/report-t1-comfyui-training.md:528-577`). |
| Global memory admission, device inventory, and shedding | **SHARED service, DUAL-PATH plans** | Extend `MemoryGovernor`, reservation requests, and worker-boundary admission, preserving reservation-before-allocation and one shedding authority (`packages/dinkster-memory/src/dinkster_memory/governor.py:1-23,93-108,120-170`; `packages/dinkster-memory/src/dinkster_memory/reservations.py:1-10,25-40,69-107`; `packages/dinkster-workers/src/dinkster_workers/governed.py:1-13,56-83`). | Training has activations, gradients, optimizer, EMA, and backward temporaries absent from weight-only planning (`/home/kosin/node-analysis/dives/report-t1-comfyui-training.md:543-577`; `/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:320-338`). |
| Asset identity, cache bytes, and provenance | **SHARED substrate, TRAINER-ONLY cache policy** | Use `AssetRef` digest identity/materialization and `DiskCAS` atomic content-addressed bytes; add training-specific manifest schemas above them (`packages/dinkster-assets/src/dinkster_assets/model.py:1-8,30-50,70-103`; `packages/dinkster-caches/src/dinkster_caches/cas.py:1-18,37-98`). | Kohya validates latent/text cache compatibility with augmentations and trainable encoders (`/home/kosin/node-analysis/dives/report-t2-kohya.md:109-114`); ai-toolkit also caches several modalities but disables incompatible caches (`/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:195-200,213-215`). |
| Dataset ingestion, captions, transforms, bucketing, prior preservation, video clip sampling, and cursor | **TRAINER-ONLY** | Training extensions consume shared assets/codecs; none of this belongs in the inference model handle. | Kohya's dataset schemas and aspect buckets are substantial trainer policy (`/home/kosin/node-analysis/dives/report-t2-kohya.md:105-114`). ComfyUI has basic resident buckets but no streaming cursor, epoch, or cache fingerprint (`/home/kosin/node-analysis/dives/report-t1-comfyui-training.md:197-251`). |
| Timestep/noise sampling, objective, target, loss shaping, accumulation, optimizer, scaler, EMA | **TRAINER-ONLY plugin surfaces** | Register these through the extension composition vocabulary, not model-family `forward`. Dinkster declares keyed-registry, ordered-list, exclusive, wrapper-chain, and observer vocabulary, but has not migrated existing seams onto it (`packages/dinkster-protocol/src/dinkster_protocol/extensions.py:106-134`). Training must implement and activate the surfaces listed in section 7. | Kohya's broad loss/optimizer/distributed policies are trainer concerns and lacks model EMA (`/home/kosin/node-analysis/dives/report-t2-kohya.md:117-129`). ai-toolkit's optimizer and EMA features confirm separate state and memory costs (`/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:221-223`). |
| Validation and preview inference | **SHARED inference capability, DUAL-PATH nesting, TRAINER-ONLY cadence** | Invoke the same family/model definitions with a nested validation policy and a temporary governor reservation. Do not duplicate a trainer-local sampler. | Kohya borrows live modules but duplicates sampling orchestration (`/home/kosin/node-analysis/dives/report-t2-kohya.md:130-140,176-193`); ai-toolkit also previews from its loaded wrapper without checkpoint reload (`/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:224,232-240`). |
| Session checkpoint and inference export | **TRAINER-ONLY state over SHARED artifacts/codecs** | Store a versioned candidate manifest and immutable shards through the asset/CAS layer, then select it through the durable session transaction; export adapters through shared external codecs. These are two products. | ComfyUI resume omits optimizer, scaler, scheduler, RNG, cursor, and metadata (`/home/kosin/node-analysis/dives/report-t1-comfyui-training.md:267-308`). Kohya saves fuller Accelerate state (`/home/kosin/node-analysis/dives/report-t2-kohya.md:128`), while ai-toolkit still has selection/provenance weaknesses (`/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:223`). |
| Graph repetition and training orchestration | **SHARED graph primitive, TRAINER-ONLY session policy** | Use `RegionNode` map/fold/while. Fold and while carry state sequentially; while requires a boolean continuation and `max_iterations` (`packages/dinkster-graph/src/dinkster_graph/model.py:90-130`; `packages/dinkster-graph/src/dinkster_graph/validate.py:763-792`). The state value is an RPC-clean session handle and each iteration advances one checkpoint interval. | Graph composition should own validation, preview, early-stop, LR-policy, and curriculum topology without moving microbatch/autograd state into the graph. |
| Queue, identity, cancellation, history, event delivery, and replay | **SHARED graph-job transport plus training-session substrate** | The ordinary graph remains the queued job (`packages/dinkster-server/src/dinkster_server/queue.py:43-100`). Reuse live fanout and bounded per-job replay, but add a session-stable typed durable journal below graph runs; current `node_event` is explicitly droppable chatter (`packages/dinkster-server/src/dinkster_server/events.py:25-37,107-120`; `packages/dinkster-server/src/dinkster_server/app.py:582-621`). Session cancellation must translate graph cancellation into a worker safe-point request rather than raw task interruption. | ai-toolkit's detached jobs use SQLite flags and UI polling, which is the management plane to avoid (`/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:226-228,242-252`). |
| Training worker process and hot session residency | **DUAL-PATH process, SHARED worker/governor substrate** | Extend the isolated-worker boundary: it already runs pack code in a child process, streams events, stamps a process-lifetime owner token, and relays worker memory consumers to the parent governor (`packages/dinkster-workers/src/dinkster_workers/isolated.py:1-28,74-140`; `packages/dinkster-workers/src/dinkster_workers/host.py:657-705`; `packages/dinkster-workers/src/dinkster_workers/session.py:229-250,430-450`). Training requires a dedicated process class and a session-scoped residency hold that outlives invocations. | Process isolation prevents inference workers from inheriting training autograd/global state, while one governor still sees hot model/optimizer/EMA/RNG bytes. |
| Distributed rank ownership, sampler partition, reductions, checkpoint barriers | **TRAINER-ONLY semantics, host-supervised resources** | Extend job/resource planning with rank topology without putting DDP policy into inference modules. | Kohya has mature Accelerate/DeepSpeed support (`/home/kosin/node-analysis/dives/report-t2-kohya.md:128-129,174`); ai-toolkit's unprepared loaders demonstrate why process count alone is not distributed correctness (`/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:254-260`). |
| Extension activation, composition, privilege, and behavior identity | **SHARED, with two pin lifetimes** | Training is an extension contribution using RPC-clean declarations and explicit composition modes (`packages/dinkster-protocol/src/dinkster_protocol/extensions.py:1-6,18-42,76-113`). S0-B pins one `ExtensionSnapshot` to each graph job at admission (`packages/dinkster-server/src/dinkster_server/queue.py:67-93,216-227`; `packages/dinkster-engine/src/dinkster_engine/engine.py:1475-1506`). A training session separately persists its own snapshot/config digests because it outlives many graph executions; an advance must use that session pin or an explicit migration, never silently adopt the current graph generation. | ai-toolkit's eager convention scanner and UID reduction are fragile (`/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:49-59`). Dinkster's extension design requires declarative activation, immutable execution snapshots, and explicit composition (`docs/extension-design.md:33-46,94-125`). |

## 3. Execution-policy architecture

### 3.1 Graph-orchestrated, worker-executed hybrid

The user-facing orchestration unit is a normal Dinkster graph job. Session setup
produces a typed handle; a fold or while region feeds that handle through an
`AdvanceTraining` node and receives the next handle. Fold and while are already
sequential state chains in the engine, and while checks its boolean continuation
after each iteration with a mandatory finite cap
(`packages/dinkster-graph/src/dinkster_graph/model.py:90-130`;
`packages/dinkster-engine/src/dinkster_engine/engine.py:1348-1402`). The normative
granularity is:

- one region iteration = one checkpoint interval / `advance N optimizer steps`;
- the complete microbatch, accumulation, backward, and optimizer loop executes
  inside the training worker;
- the only loop-carried value is the session handle in section 3.2, never a
  tensor, module, optimizer, RNG object, or autograd graph;
- validation, preview sampling, metrics, early stopping, LR-policy decisions,
  and curricula are graph nodes around `AdvanceTraining`, using checkpoint
  artifacts or scoped requests against the session snapshot; and
- the training worker is a dedicated isolated process with its own torch import,
  CUDA context, autograd configuration, and process globals. It is never an
  inference worker temporarily switched into training mode. Dinkster's existing
  isolated worker already gives pack code a separate interpreter/process
  (`packages/dinkster-workers/src/dinkster_workers/isolated.py:1-28`), but the
  long-lived training-session lifecycle is new.

The graph job can end while the training session remains resumable. Conversely,
a session can span many graph jobs and extension generations. A graph run's
worker/schema/extension snapshot is pinned at that run's admission; the session
snapshot and config are pinned when the session is created. Neither pin
substitutes for the other.

```text
Graph RegionNode (fold / while, finite)
        |
        | RPC-clean TrainingSessionHandle
        v
AdvanceTraining node -- cancel/event bridge -- durable session journal
        |
        v
Dedicated isolated training worker
  hot model + optimizer + EMA + RNG + cursor
  internal microbatch/accumulation loop
        |
        v
fence-conditional ledger commit selects CAS checkpoint -> next handle
```

### 3.2 Advance node, session handle, effects, and recovery

#### Graph-boundary handle

Recommendation (vetoable): define `core.training_session_handle.v1` as the
canonical RPC-clean object below. These are exactly the fields that cross a
graph state port:

```text
schemaVersion: 1
sessionId: opaque host-minted 128-bit-or-stronger identifier
checkpointManifestDigest: blake3:<64 lowercase hex>
stepCursor: non-negative committed optimizer-step integer
configDigest: blake3:<64 lowercase hex>
sessionExtensionSnapshotDigest: blake3:<64 lowercase hex>
journalSeq: non-negative checkpoint-covered durable journal watermark
```

The handle fingerprint is the canonical serialization of every field. Decode
rejects unknown schema versions, malformed ids/digests, negative cursors, and
noncanonical encodings. Before materialization, the supervisor authorizes the
current graph job's principal/scope against the session, resolves the
authoritative session record, verifies that the manifest is committed on that
session's lineage, and verifies its digest. The worker then requires exact
equality for `sessionId`, committed step, config digest, session extension
snapshot digest, and the manifest's checkpoint-covered journal watermark.
`journalSeq` is exactly that manifest watermark, not the later
`advance_committed` event sequence; it may trail later durable or coalesced
telemetry but cannot exceed the durable session watermark.

The config digest covers session-scoped trainer, dataset, objective, precision,
kernel, seed, and contribution invariants. A declared schema of graph-supplied
per-advance policy inputs, such as the next LR or curriculum phase, is excluded
from `configDigest` but included in the operation identity and durable journal.
The manifest transitively commits to the complete model-handle reconstruction
recipe required by the host invariant (`docs/extension-design.md:149-162`) and
to all session-state shards. No worker token, PID, device address, tensor, or
live object enters the handle. Handle possession is reference data, not bearer
authorization to observe or mutate a session.

#### Hot and cold paths

Recommendation (vetoable): route by best-effort `sessionId` affinity, but keep
affinity out of computation/cache identity. A worker whose hot state exactly
matches the input checkpoint digest and session/config pins advances directly.
Each session has one durable single-writer fence, so concurrent advances cannot
fork one optimizer trajectory.

If that worker died or its safe-point state was evicted, any compatible training
worker may verify the handle, acquire the session reservation, reconstruct the
base model handle from its manifest-pinned declarative recipe, and layer the
optimizer, live trainable/EMA values, RNG, cursor, and plugin state from CAS
checkpoint shards. This is the host's normal recipe materialization mechanism,
not an ad-hoc training loader (`docs/extension-design.md:149-162`). Cold recovery
is transparent to graph values and emits a durable `session_recovered` event
with the old/new attempt and recovery latency. It is surfaced as a node failure,
not hidden, when artifacts are missing/corrupt, the recipe cannot reconstruct
semantically identical base state, the pinned session extension generation
cannot be supplied, the config or semantic execution policy differs, required
hardware capability is absent, or the durable operation record is inconsistent.
A hot worker that is ahead never rolls back to an arbitrary stale handle: it
returns the recorded result for a known operation or rejects an unknown stale
lineage.

Worker process identity is liveness, not computation identity. Existing Dinkster
already treats worker instance tokens as process-lifetime owner facts and cache
misses resident outputs from a dead owner
(`packages/dinkster-workers/src/dinkster_workers/session.py:274-280`;
`packages/dinkster-engine/src/dinkster_engine/engine.py:968-980`). Training extends
that rule to cold-reloadable session state rather than graph-resident tensors.

#### Effect and cache contract

`AdvanceTraining` is effectful: it mutates an optimizer trajectory, appends a
journal, and commits a new checkpoint manifest through the session transaction.
It is also retry-safe under the idempotency protocol below. Those are different
properties. Current
`NodeSchema` has one `idempotent` bit: false means side-effecting and never
cached, while true receives ordinary pure-node cache/single-flight behavior
(`packages/dinkster-schema/src/dinkster_schema/model.py:975-1000`;
`packages/dinkster-engine/src/dinkster_engine/engine.py:290-294,570-607`). Therefore
the advance node MUST NOT simply declare `idempotent=True` and treat cache
storage as its transaction log.

Recommendation (vetoable): Slice 0 adds an explicit engine effect policy for a
transactional, idempotent command (illustrative name
`effectful_idempotent`), distinct from both pure-cacheable and never-cacheable.
The durable operation ledger is the correctness boundary. The ordinary engine
cache may replay the already committed output handle and single-flight
identical attempts, but only under the same operation identity and only after
the worker's durable commit. Until that engine contract exists, the safe
fallback is `idempotent=False` plus worker-ledger deduplication; it sacrifices
cache hits but never double-steps.

`AdvanceTraining` also declares the accelerator `occupies` lane selected by its
inputs/session. That lane limits concurrent execution; it is not memory
accounting and does not replace the session-scoped governor hold
(`packages/dinkster-schema/src/dinkster_schema/model.py:995-1015`;
`packages/dinkster-engine/src/dinkster_engine/engine.py:430-459`).

The operation identity is:

```text
blake3("dinkster.training.advance.v1",
       sessionId, input checkpointManifestDigest, configDigest,
       sessionExtensionSnapshotDigest, requested target/delta steps,
       seed-policy digest, behavior-affecting advance options,
       selected training implementation runtime identity)
```

Under the future transactional effect policy, every listed item must also be
represented in the engine cache components: typed input fingerprints, effective
schema signature, extension behavior, and selected executor/cache tag. The safe
`idempotent=False` fallback instead uses the ledger as identity and intentionally
gets no reusable engine key. Dinkster currently chooses executor identity before
lookup and includes both selected arm and cache tag in the key
(`packages/dinkster-engine/src/dinkster_engine/engine.py:106-145,570-607,906-935`).
The backend review must explicitly answer: does the training arm's cache tag
training behavior changes bump that epoch, and why should those changes not
a separate training epoch folded through the existing execution-selection
identity; changing training math or resumable state transitions rotates it.

#### Idempotency and retry

Before work starts, the supervisor durably claims `(sessionId,
advanceOperationId, input checkpoint digest)` for one worker attempt under the
session fence. The operation record moves through `started`, optional safe-point
recovery checkpoints, and either `committed` with exactly one output handle or
`aborted`. Claim and commit are both fence-epoch-conditional durable writes. A
commit under a superseded epoch is rejected even if that worker already wrote
CAS shards and a candidate manifest. Updating the operation's private recovery
checkpoint pointer is fence-conditional too: a fenced-out attempt may leave
unreachable CAS bytes but cannot replace resumable operation state.

CAS shards and the immutable candidate manifest are written first. One durable
operation/journal transaction is the commit point: it revalidates the fence,
selects that candidate as committed session state, maps the operation to its
unique output handle, and appends the correctness-critical journal facts before
the worker returns. A CAS manifest without this ledger commit is provisional,
never becomes session state, and is eligible for GC. The engine records or
caches only the resulting committed handle.

- Crash before any durable progress: retry restores the input manifest and
  executes the interval.
- Cancellation or crash after an internal safe point: retry restores the
  operation's complete private recovery checkpoint and performs only the
  remaining steps. That digest is substrate state, not a graph value.
- Commit succeeds but the worker reply or engine cache write is lost: the same
  operation id returns the already committed handle without another optimizer
  step.
- A concurrent claim for an already active operation id returns a typed,
  retryable `advance_in_progress`; it never runs a second worker. A paused or
  lost attempt may be taken over under a new fence and resumes that operation's
  private checkpoint. A committed duplicate returns the committed handle.
- Two different operation ids from the same input conflict while either has an
  active claim, and a competing id conflicts permanently once one commits. An
  uncommitted operation may instead be explicitly superseded under the session
  fence: the supervisor appends `advance_aborted`, discards its private recovery
  checkpoint after the retention window, and only then permits a new operation
  from the still-committed input. Explicit branching requires a separate
  clone-session operation with a new `sessionId`.
- An unknown stale input conflicts loudly. Operation-private recovery shards
  become GC-eligible only after durable commit or abort and the declared
  retry/debug retention window.

This closes the exact crash window between worker completion and engine result
recording. Engine cache contents improve latency; they never decide whether an
optimizer step happened.

### 3.3 One model definition, explicit modes

Every executable model view is created from a shared `ModelDefinition` plus an
immutable `ExecutionPolicy`. Construction or binding selects operations before
`forward`, preserving Dinkster's compile discipline
(`packages/dinkster-inference-torch/src/dinkster_inference_torch/operations.py:12-18`).
The minimum modes are:

| Mode | Gradients | Base storage | Allowed operations |
|---|---|---|---|
| `INFERENCE` | None | Dense or packed/quantized | Fused inference kernels, disposable casts, inference overlays, aggressive forward residency. |
| `TRAIN_ADAPTER` | Inputs and live adapters only | Frozen dense or packed/quantized | Only kernels with declared grad-input behavior; adapter branch remains in autograd; no base-weight gradient. |
| `TRAIN_FULL` | Selected floating masters and inputs | Floating master is authoritative | Differentiable kernels, grad-weight support, activation checkpoint replay, optimizer-aware residency. Packed deployment weights are outputs, not trainable truth. |
| `VALIDATION` | None, nested in a training session | Current raw or EMA trainable snapshot plus the same base identity | Inference semantics scoped to validation, with temporary eval/no-grad/autocast/residency and exact restoration. |

Mode is carried in a session/execution context, not a process-global switch.
The policy also names device and precision, deterministic requirements,
checkpoint/recompute eligibility, quantized-backward algorithm, and selected
kernel capabilities. Unsupported combinations fail before allocation with a
per-layer diagnostic. This directly avoids ComfyUI's process-global mode and
coarse quantized-backward choice
(`/home/kosin/node-analysis/dives/report-t1-comfyui-training.md:610-647,813-815`).

Cross-process execution follows Dinkster's existing identity pattern rather than
inventing ambient worker state. `Invocation` currently carries an immutable
host-selected execution identity and fp8 policy
(`packages/dinkster-protocol/src/dinkster_protocol/__init__.py:43-97`); the worker
boundary serializes and validates both
(`packages/dinkster-workers/src/dinkster_workers/boundary.py:473-555`), then exposes
them through a task-local `ExecutionContext`
(`packages/dinkster-workers/src/dinkster_workers/execution.py:1-40`). A future
training-session protocol must carry execution mode, precision/kernel policy,
base identity, session snapshot/config identity, and advance operation id
atomically by the same discipline. It must not infer them from installed
weights, the current graph generation, or a worker-global flag. Public
pack-facing contracts belong in the frozen `dinkster_api.v1` door, which already
re-exports extension and memory declarations while leaving runtime activation
to the host (`packages/dinkster-api/src/dinkster_api/v1.py:1-23,73-96`).

```text
shared family + component plan + canonical parameter IDs
                           |
                    ModelDefinition
                           |
          +----------------+----------------+
          |                |                |
   INFERENCE binding  TRAIN_ADAPTER    TRAIN_FULL
   no grad, fused     frozen base,     floating masters,
   kernels allowed    grad-input +     grad-input/weight
                      live adapters
          |                |                |
          +-------- shared governor --------+
                           |
             graph orchestration + session journal
```

### 3.4 Effective-weight and materialization lifecycle

The operation contract must advertise capabilities independently of storage
format and distinguish exact gradients from approximations:

- `forward_no_grad`: fused or dequantized inference.
- `forward_grad_input_exact`: frozen base whose forward and input derivative are
  one mathematically consistent operation.
- `forward_grad_input_ste`: frozen base with an explicitly named straight-through
  estimator and documented forward/backward mismatch.
- `forward_grad_weight`: floating trainable weight/master and weight gradient.
- `checkpoint_replay_safe`, `deterministic`, supported devices/dtypes/shapes,
  estimated forward/backward saved bytes, and workspace bytes.

For `TRAIN_ADAPTER`, the preferred quantized linear implementation wraps only
the frozen-base operation in a custom autograd function. The live adapter branch
must remain outside that function because `torch.autograd.Function.forward`
runs without ordinary grad recording:

1. `FrozenBaseFn.forward` acquires a normal forward `WeightLease`, stages packed
   storage, and computes only the frozen base result.
2. Ordinary autograd computes `y = FrozenBaseFn.apply(x, identities) +
   adapter_branch(x)`. The live adapter residual is therefore recorded outside
   the custom function and gradients reach its `Parameter`s and input.
3. `FrozenBaseFn` saves only graph-valid inputs plus immutable materialization identity and a
   mechanism/version handle. Never save a tensor borrowed from the closed
   forward lease.
4. `FrozenBaseFn.backward` reacquires/rematerializes the frozen base weight,
   computes grad-input, releases it, and returns no base-weight gradient. The
   separate adapter branch receives its normal autograd gradients.

This extends, rather than violates, the current forward-only lease contract
(`packages/dinkster-inference-torch/src/dinkster_inference_torch/residency.py:152-186`).
It matches the proven ai-toolkit forward/backward restaging pattern
(`/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:185-191,292-304`) and
ComfyUI's `QuantLinearFunc` result that a frozen packed base can still carry
input gradients (`/home/kosin/node-analysis/dives/report-t1-comfyui-training.md:445-485`).

A graph-long lease is an allowed future mechanism only when it is explicit,
pin-counted, version-stable, and fully reserved through backward. It must not
be the first implementation because it prevents useful shedding and makes a
forward-scoped API deceptively unsafe.

Dinkster's current fp8 fused route correctly refuses grad-enabled inputs
(`packages/dinkster-inference-torch/src/dinkster_inference_torch/quant_linear.py:190-210`).
Its input quantization means `grad_output @ dequant(weight)` would be an STE,
not an exact derivative of the fused forward. Dinkster already specifies that
gradient work stays on the dequant route
(`packages/dinkster-inference-torch/src/dinkster_inference_torch/quant_linear.py:296-303`),
so Slice 2 defaults to a dequantized frozen-base matmul for exact,
forward-consistent grad-input. A fused-fp8-forward/STE-backward option may be
added only as a separately named, quality-characterized policy. Because a
custom autograd `forward` disables ordinary grad recording, the fused route's
runtime `torch.is_grad_enabled()` guard is not protection inside that function;
preflight capability negotiation is mandatory. Training policy selects a
declared grad-input fallback or refuses, never catches an error and silently
runs an unreported mixed policy.

Backward cannot await memory admission: `MemoryGovernor.reserve` is async
(`packages/dinkster-memory/src/dinkster_memory/governor.py:317-335`). The training
microbatch plan therefore pre-admits every backward rematerialization, transfer
slot, and workspace byte before creating the forward graph. Backward only uses
that already-held allowance and performs no new admission. GGUF or other packed
storage can support adapter training only if its operation advertises exact or
explicitly approximate grad-input and fits this lifecycle. It cannot support
full tuning without a floating master or an explicit QAT contract.

### 3.5 Nested inference inside training

Validation and previews are a scoped transition, not a second loaded model by
default:

1. Stop only at a declared training safe point (normally after an optimizer
   step and before the next forward).
2. Snapshot adapter version and choose raw or EMA values.
3. Snapshot every named RNG stream and module train/eval state.
4. Ask the governor for a validation peak reservation; it may temporarily move
   the codec/text encoder/model according to a planned policy.
5. Enter `VALIDATION` with local no-grad/autocast and run the shared inference
   sampler/service.
6. Restore parameters, adapter multiplier, module state, residency policy, and
   RNG streams exactly, even on cancellation or error.

The surrounding session must never enter global `torch.inference_mode`.
Validation cadence, prompts, metrics, seeds, and raw-versus-EMA choice remain
trainer policy. A separate instance is permitted when the governor chooses it
and accounts both copies; it is not a family-specific trainer workaround.

## 4. Trainable adapter contract

### 4.1 `TrainableAttachment`

The training extension API should expose a live `TrainableAttachment`, distinct
from `WeightAdapter` and `PatchSet`. The contract contains:

- algorithm id/version: initially `lora`, then `loha`, `lokr`, `oft`, and
  `full_delta`;
- a resolved target manifest containing stable target IDs, logical shape,
  operation kind, base parameter key, family binding version, and match reason;
- owned trainable parameters and deterministic optimizer group descriptors;
- rank/alpha/scaling, convolution dimensions, dropout variants, dtype policy,
  per-target overrides, and train/eval/multiplier state;
- operation strategy: effective-weight delta, parallel input residual, or
  exact output transform, with declared quantized-base compatibility;
- snapshot, restore, EMA enumeration, import/export, and merge capabilities;
- a complete consumed/unmatched-key diagnostic.

LoRA, LoHa, LoKr, and full-delta naturally produce a weight delta when the
target operation supports it. OFT must preserve exact weight-transform and bias
semantics; an output rotation is accepted only when mathematically equivalent
for that operation. The contract must refuse unsupported biased or grouped
operations rather than approximate them. ComfyUI's bypass OFT demonstrates why
`R(Wx+b)` and `R(Wx)+b` cannot be treated as universally interchangeable
(`/home/kosin/node-analysis/dives/report-t1-comfyui-training.md:425-445`).

The contract must not structurally preclude a second attachment class of
trainable AUXILIARY MODULES that own new parameters (zero-initialized control
branches such as ControlNet/LLLite-style adapters) and bind to declared family
injection seams rather than existing parameters; Slice 8 builds on this
without reopening the contract.

The host model handle remains reconstructible from its complete declarative
recipe (`docs/extension-design.md:149-162`). Live adapter/auxiliary-module
masters are session-owned mutable state layered over that recipe-reconstructible
base, not additions to model identity that exist only in worker memory. The
session manifest stores their declared attachment semantics and state shards so
fresh materialization, device replication, safe eviction, and cold recovery all
use the same host mechanism.

No attachment may replace arbitrary `module.forward`, install mutable global
hooks, or live solely in a side manager. Native architecture construction
already has the correct typed operation seam
(`packages/dinkster-inference-torch/src/dinkster_inference_torch/operations.py:1-18`).

### 4.2 Stable targets and ecosystem codecs

Internal target identity is not a kohya key. It is a structured Dinkster identity:

`family-id / component-id / canonical-module-path / parameter-or-operation-role`.

Family training bindings publish target points and tags (for example attention
q/k/v/projection, MLP, convolution, norm, embedding), exclusions, default rank
rules, and supported adapter algorithms. Resolution produces a target manifest
before optimizer allocation; unmatched selectors or duplicate target IDs fail
loudly. This follows Dinkster's semantic-target extension principle
(`docs/extension-design.md:54-61`) and avoids ComfyUI's overbroad implicit target
selection (`/home/kosin/node-analysis/dives/report-t1-comfyui-training.md:651-664`).

External layouts are bidirectional codecs over that identity:

- kohya/sd-scripts safetensors key layouts and metadata are required for first
  ecosystem interoperability;
- ComfyUI and family-native layouts can reuse Dinkster's current LoRA dialect
  normalization (`packages/dinkster-inference/src/dinkster_inference/lora.py:1-18,86-111`);
- import must report every consumed, ignored, missing, and unknown key;
- export records base asset digest, family/binding version, target manifest,
  algorithm metadata, precision, and effective static base overlays;
- a portable inference export is separate from the resumable session bundle.

Kohya's adapter system demonstrates why rank/alpha, prefixes, target mapping,
optimizer grouping, and merge/export form one ecosystem contract
(`/home/kosin/node-analysis/dives/report-t2-kohya.md:54-103`).

### 4.3 Precision and RNG

Default trainable masters are float32. Execution may cast adapter computation
to bf16 under policy, while optimizer masters and state stay explicitly typed.
Base packed tensors always have `requires_grad=False`; Dinkster already freezes
stored parameters when installing module state
(`packages/dinkster-inference-torch/src/dinkster_inference_torch/module_residency.py:122-148`).

Each session owns named, checkpointed RNG streams at minimum for:

- adapter initialization;
- dataset order and bucket choice;
- caption mutation and augmentation;
- timestep and noise;
- adapter/module dropout;
- validation/sample generation;
- stochastic rounding or quantization, when enabled.

Plugins receive only their assigned generator, not ambient process RNG. Device
and CPU generator states, NumPy state if a declared plugin uses it, and any
sampler-specific state are durable. Adapter initialization is stable under
target-order changes by deriving each target seed from `(session seed, stream
id, target id, parameter role)`. This fixes ComfyUI's ambient initializer and
cross-node RNG coupling (`/home/kosin/node-analysis/dives/report-t1-comfyui-training.md:621-647`).

## 5. Training-aware memory governance

The `MemoryGovernor` remains the sole admission and shedding authority. The
current service deliberately reserves before allocation and orders shedders
(`packages/dinkster-memory/src/dinkster_memory/governor.py:1-23,93-108,155-170`).
Training extends its resource model; it does not add a trainer-local free-VRAM
oracle.

### 5.1 Accounted resources

Every training plan reports bytes by device, owner, phase, and lifetime:

1. immutable base storage and resident packed/dense units;
2. trainable parameters and optional floating masters;
3. gradients, including accumulation lifetime;
4. optimizer state and optimizer-step workspace;
5. EMA shadow state;
6. grad scaler and scheduler state;
7. saved activations/checkpoint inputs and backward temporaries;
8. dequantized/effective weights and transfer/prefetch ring slots;
9. input batches, cache staging, pinned host memory, and output staging;
10. validation model/codec/text-encoder peaks and preview bytes.

The resource declaration includes estimated baseline, per-microbatch peak,
optimizer-step peak, validation peak, and sheddability. The host reconciles
estimates with measured allocator/device telemetry after warmup and records the
high-water correction in session diagnostics. Adam state created lazily at the
first step must be pre-reserved from optimizer metadata, not discovered by OOM.
The reports identify this exact blind spot in ComfyUI
(`/home/kosin/node-analysis/dives/report-t1-comfyui-training.md:726-740`) and EMA
as a distinct ai-toolkit memory category
(`/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:221-222`).

### 5.2 Reservation and safe shedding

Add a long-lived `TrainingReservation` for session baseline plus scoped phase
sub-reservations. Current `ReservationRequest` is invocation-duration and byte
only (`packages/dinkster-memory/src/dinkster_memory/reservations.py:25-40,69-107`), so
training needs resource category, owner/session, lifetime, and safe-point
metadata without exposing the governor to trainer code.

The training device must have a declared budget; current unbudgeted devices are
tracked but do not block admission
(`packages/dinkster-memory/src/dinkster_memory/governor.py:20-23,199-204`). Session
admission also defines concurrency explicitly. The initial policy is exclusive
training admission per accelerator: conflicting inference/training jobs queue
behind the owning session or fail fast, according to the request's deadline,
with a diagnostic naming the session. They must not wait indefinitely after
shedding. This matters because the current governor waits for outstanding
reservations to release when a request still does not fit
(`packages/dinkster-memory/src/dinkster_memory/governor.py:317-327,337-389`). Later
time-slicing requires measured phase releases and remains a separate policy.

Each byte has one accounting owner. Session-baseline allocations either remain
covered by the long-lived reservation and are excluded from consumer
`footprint`, or atomically transfer from reservation to a pinned,
non-sheddable consumer after materialization. They are never counted in both;
the current fit equation includes both reserved and footprint bytes
(`packages/dinkster-memory/src/dinkster_memory/governor.py:383-386`). The transfer
must not create an unreserved interval.

Live autograd state is pinned and never sheddable. Placement changes happen at
declared safe points: before a forward graph is created, after backward has
released saved state, after optimizer commit, at checkpoint barriers, or around
the nested validation transition. The existing two-phase release pattern -
propose, gate against current use, then release - is the right model for remote
or referenced resources (`packages/dinkster-memory/src/dinkster_memory/release.py:1-21,35-78`).

Block swap, layer offload, optimizer offload, gradient offload, activation
checkpointing, and backward rematerialization are governor-selected policy
implementations over model-declared residency units and recomputation regions.
They are not trainer options that monkeypatch `.to()` or `forward`. Policy is
deterministic and recorded in the session manifest. ai-toolkit demonstrates
useful transfer rings but also the wrong random-per-module offload policy
(`/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:179-191,407-410`).

Admission can propose, in order: compatible kernel fallback, stronger
checkpointing, planned block swap, optimizer/EMA offload, smaller cache
prefetch, or a lower user-approved microbatch. A semantic change such as batch
size, accumulation, precision, or objective is never made silently. If no plan
fits, fail before training with a category-level memory report.

### 5.3 Isolated process and session-scoped residency

The training worker is a dedicated process, not an inference worker arm. Its
torch import, CUDA context, allocator, autograd settings, modules, optimizer,
EMA, generators, and transient graphs are process-local. No training API may
toggle an inference worker's global mode or pass a training tensor/object back
through a graph port. The existing isolated boundary proves separate-process
execution and process-lifetime owner identity, but its ordinary reservation is
invocation-scoped and cancellation currently tears down the invocation task
(`packages/dinkster-workers/src/dinkster_workers/isolated.py:1-28`;
`packages/dinkster-workers/src/dinkster_workers/host.py:763-817,870-890,1066-1079`).
Training therefore needs an explicit adapter rather than pretending current
invocation lifecycle already has safe training semantics.

Recommendation (vetoable): a session acquires one governor-accounted residency
hold when it becomes hot. The hold covers process/context overhead and every
baseline allocation category in section 5.1, survives across advance
invocations, and is keyed by `sessionId` plus the live worker instance. Phase
sub-reservations cover activation/backward/optimizer/validation peaks. The
worker exposes hot state as named consumer details to the parent governor; the
host records either a reservation or a footprint for each byte, never both.
Dinkster's relay already projects child consumer snapshots into parent-side
shedders and collapses them when the worker dies
(`packages/dinkster-workers/src/dinkster_workers/relay.py:1-37,137-143`).

Only checkpoint-safe state is sheddable. At a safe point, policy may retain the
session hot, demote selected state to RAM/disk, or fully cold-evict it after the
CAS checkpoint is verified. Live autograd state, an optimizer mutation, and an
uncommitted candidate manifest are non-sheddable. Process death releases the
accounting hold and invalidates affinity; it does not invalidate the last
committed checkpoint. Initial exclusive accelerator admission applies to the
session hold, not merely one node invocation, so graph gaps between advances cannot let
another workload consume bytes still occupied by hot optimizer/model state.

## 6. Dataset and cache contracts

### 6.1 Trainer-owned data plane

A `DatasetPlan` is immutable configuration plus input artifact identities. A
`DatasetCursor` is mutable durable session state. The trainer owns:

- source enumeration and caption/metadata policy;
- include/exclude, repeats, class/instance split, and prior-preservation ratio;
- deterministic train/validation split;
- resolution/aspect buckets, no-upscale rules, crop and resize policy;
- bucket-local permutation, epoch, sampler partition, and dynamic batch/token
  budgets;
- image/video transforms, masks, controls, frame/clip windows, stride, temporal
  buckets, and clip cursor;
- cache eligibility, build timing, prefetch, traversal, and invalidation;
- sample weights and loss normalization across uneven buckets.

Family bindings contribute constraints such as spatial/temporal divisibility,
latent packing, conditioning components, and cacheable boundaries. They do not
choose data order. Kohya's bucket and cache contracts establish the parity bar
(`/home/kosin/node-analysis/dives/report-t2-kohya.md:109-114`); ComfyUI's fully
resident lists and absent cursor establish what not to copy
(`/home/kosin/node-analysis/dives/report-t1-comfyui-training.md:197-251`).

Prior preservation is represented as a second named sample stream with its own
manifest, cursor, cache namespace, weighting, and deterministic mixing rule. It
is not a special directory convention hidden in one family trainer.

### 6.2 Shared artifacts, immutable cache shards

The artifact system stores bytes and identity; the trainer assigns meaning.
Add versioned artifact schemas for:

- dataset manifests and normalized sample records;
- latent cache shard plus shard index;
- text/vision/control embedding cache shard plus shard index;
- validation set/prompt manifest;
- training session manifest/checkpoint shards;
- portable adapter/model export and preview media.

`AssetRef` already provides digest identity and resolver-bound materialization
(`packages/dinkster-assets/src/dinkster_assets/model.py:30-50,70-103`), while `DiskCAS`
provides digest-verified atomic, idempotent bytes
(`packages/dinkster-caches/src/dinkster_caches/cas.py:1-18,56-98`). New schemas should
reference immutable shards by digest. Section 8.2's durable session transaction,
not CAS storage alone, makes a training checkpoint manifest authoritative.

A cache key includes every behavior input: source asset digest; selected
frame/clip and crop; transform parameters; family and codec/text-encoder asset
identity; tokenizer/config; latent/conditioning schema version; execution
precision where numerically relevant; static base overlays; extension snapshot;
cache-plugin versions; and augmentation compatibility. Random crop, caption
dropout/shuffle, or a trainable encoder either move randomness before the cached
boundary with its seed in the key, cache a lower-level representation, or make
the cache ineligible. Never reuse a cache merely because a filename exists.

Large image/video datasets are streamed through bounded prefetch and shard
leases. No API requires all cache shards or all videos in memory. Cursor state
identifies manifest digest, split, epoch, global permutation seed/counter,
bucket, bucket offset, sample/clip IDs, and distributed partition.

## 7. Trainer extensions and plugin composition

Training should dogfood Dinkster's extension architecture. Current declarations
are scope-separated, RPC-clean, and capability-validated
(`packages/dinkster-protocol/src/dinkster_protocol/extensions.py:1-6,18-42,76-103`),
and composition-mode vocabulary is explicit. It is currently declarative only:
the source says existing seams have not been migrated onto it
(`packages/dinkster-protocol/src/dinkster_protocol/extensions.py:106-134`). The design
must implement and activate the training contribution surfaces below; it does
not assume active host registries and does not add import scanners or
registration side effects.

| Contribution surface | Composition | Ownership/rule |
|---|---|---|
| Trainer/session type and advance-node body | Keyed registry | Unique stable id/version. One selected per session; graph orchestration remains host-owned. |
| Family training binding | Keyed registry by family id | Exactly one selected binding version per family; may compose declared capability fragments only if the owner defines that mode. |
| Adapter algorithm and external codec | Keyed registries | Algorithm identity is separate from file-layout codec. Duplicate key is activation failure. |
| Dataset source/cache codec/optimizer/timestep sampler/base objective/metric | Keyed registries | Explicit user/session selection. Capability report before allocation. |
| Objective and loss modifiers | Ordered lists | Stable resolved order enters behavior identity. Each declares required tensors and reduction semantics. |
| Dataset transforms and batch decorators | Ordered lists | Stable order, cache effect, RNG stream, and provenance required. |
| Validation/checkpoint/telemetry callbacks | Observer lists | Cannot mutate optimizer/model unless using a separately declared command surface. Failures have explicit fatal/nonfatal policy. |
| Execution/residency strategy | Exclusive owner-selected strategy or delegating wrapper chain | Never implicit last-writer-wins; host governor remains final authority. |

The session pins one immutable `ExtensionSnapshot`: extension ids/versions,
package digests, contribution order, provider selections, capabilities, and
behavior configuration are represented and canonically hashed
(`packages/dinkster-protocol/src/dinkster_protocol/extensions.py:141-164,205-275`).
Resume requires the same snapshot digest or an explicit migration accepted by
each changed stateful plugin.

This session pin is not S0-B's graph-execution pin. The queue captures an
`ExecutionRuntime` at graph-job admission and the engine holds it only for that
run (`packages/dinkster-server/src/dinkster_server/queue.py:67-93,216-227`;
`packages/dinkster-engine/src/dinkster_engine/engine.py:1475-1506`). Session creation
copies the selected training snapshot digest and behavior config digest into the
session manifest/handle. Every later graph execution has its own S0-B snapshot,
which governs that graph's node schemas and orchestration, while
`AdvanceTraining` must resolve the older session generation or fail loudly. A
new graph generation never silently migrates optimizer/plugin state.

Authority is the dedicated `TRAINING` execution scope with declared
capabilities: `accelerator` and `artifacts` (checkpoint/adapter read/write)
plus the existing `filesystem`, `background-jobs`, `downloads`, and
`model-family-registration` names. `ExtensionScope.TRAINING`, the manifest
`training` entry point, and both capability names are in the closed vocabulary
(`packages/dinkster-protocol/src/dinkster_protocol/extensions.py`); the generic pack
worker never resolves training entries - the dedicated training worker owns
them. Server routes and UI contributions stay in their existing scopes.

Activation resolves every keyed collision, missing capability, family-binding
incompatibility, and unavailable kernel before publishing the generation. A
session receives a frozen view. Pack removal/reload affects only new sessions;
running and resumable sessions retain a generation lease or fail loudly if the
generation can no longer be supplied. Dinkster's extension design already calls
for declarative transactional activation and immutable execution snapshots
(`docs/extension-design.md:94-125`).

S0-B now stages and publishes a complete graph execution generation atomically,
and rolls back a failed activation
(`src/dinkster/activation.py:136-218`). Its retirement policy closes old workers
immediately and permits an in-flight old-generation run to fail loudly
(`src/dinkster/activation.py:223-239`). A resumable training session therefore
still needs a longer generation lease or reproducible re-provisioning by pinned
package digest. This is new training lifecycle work, not a reason to weaken
S0-B's per-graph pin.

## 8. Durable training session state

### 8.1 Two products

1. **Portable inference export:** selected raw or EMA adapter/model tensors,
   external codec metadata, target/base provenance, and no optimizer internals.
2. **Resumable session checkpoint:** complete continuation state. Loading only
   adapter tensors is initialization, never resume.

This distinction directly fixes ComfyUI's adapter-only resume
(`/home/kosin/node-analysis/dives/report-t1-comfyui-training.md:267-308,923-935`).

### 8.2 Session manifest

The versioned manifest must reference immutable shards and contain:

- session id, checkpoint id/digest, parent checkpoint, creation time, committed
  global step, microstep, epoch, and worker attempt; optional graph run/node
  provenance is linkage, never the session identity;
- the complete declarative base-model reconstruction recipe required by
  `docs/extension-design.md:149-162`, including exact base asset/component
  digests, storage-provider config, family/config and assembly-plan identity,
  normalized static overlays, declared attachment semantics, and execution
  policy;
- session extension snapshot digest, canonical config digest, every selected
  contribution id/version/config, and migration schema versions;
- target manifest and trainable algorithm metadata;
- raw live attachment/trainable parameters layered over the reconstructible
  base, optional EMA parameters, and selected export view;
- optimizer class/config and complete state, LR scheduler, scaler, gradient
  accumulation counters, pending gradients only if mid-accumulation checkpoints
  are supported, clipping/precision policy, and optimizer step counters;
- all named RNG states, including CPU/device/plugin streams;
- dataset/cache manifest digests and complete cursor: split, epoch, permutation,
  bucket, offset, sample/clip position, repeats, and distributed partition;
- objective/timestep/noise/loss/metric state and plugin-owned opaque versioned
  state;
- distributed world/topology, rank-owned shards, sampler partition, reduction
  state, and checkpoint barrier record;
- governor/residency policy for diagnostics (placement can be replanned on
  compatible hardware; semantic precision/kernel changes require consent);
- metric history/watermarks and the checkpoint-covered durable journal sequence;
- a diagnostic snapshot of the fence epoch and operation-ledger watermark or
  digest as of candidate creation. The authoritative mutable fence and operation
  ledger live in the durable session store/shared journal substrate, not in this
  immutable manifest;
- checksum/digest for every referenced shard.

Checkpoints are committed at a safe point. Each writer first stores immutable
CAS shards, ranks synchronize if distributed, and the coordinator stores the
digest-addressed candidate manifest last. The fence-conditional durable
operation/journal transaction is the commit point: it selects that manifest as
session state, records it as the operation's unique result, and appends the
commit facts before the worker replies. A crash before ledger commit leaves
unreachable or provisional CAS objects eligible for GC, not a partially valid
session checkpoint. A crash after ledger commit but before reply is resolved by
operation-id lookup, not by stepping again. Resume validates the reconstruction
recipe, all digests, authoritative fence/ledger state, and compatibility before
allocating model or optimizer state.

Default checkpoints occur only after optimizer commit, so no pending gradient
shard is needed. A future mid-accumulation checkpoint must save gradients and
microbatch cursor exactly; otherwise it is refused. This avoids pretending a
step number is continuity. ComfyUI loses optimizer/scaler/RNG/cursor and resets
some adapter state (`/home/kosin/node-analysis/dives/report-t1-comfyui-training.md:267-308,666-680`).

## 9. Graph jobs over a supervised session substrate

Training is not a second user-facing queued job type. The user submits an
ordinary graph job containing `CreateTrainingSession`, `AdvanceTraining`, and
ordinary composition around them. The session is a durable supervised resource
below those finite graph runs. It owns lineage, hot/cold worker state,
idempotency records, safe-point cancellation, and events across many graph job
references. Existing graph jobs already have opaque identity, scoped principal,
lifecycle, a graph, and individual cancellation
(`packages/dinkster-server/src/dinkster_server/queue.py:43-100`).

### 9.1 Graph and observation surfaces

- `CreateTrainingSession` validates config, targets, capabilities, memory, and
  session extension snapshot, binds an owner/authorization policy, commits an
  initial checkpoint, and returns the section 3.2 handle. A dry-run variant
  returns the report without creating a session.
- `AdvanceTraining(handle, interval, policy inputs...)` performs one effectful,
  retry-safe checkpoint interval and returns the next handle plus ordinary
  metric/artifact outputs. The interval must request at least one optimizer step;
  zero or negative advances are refused.
- RegionNode fold/while carries the handle. Validation, preview, LR adjustment,
  early stop, export, merge, and curricula remain ordinary graph nodes.
- The existing graph submission/cancel API remains the user-facing execution
  surface. Read-only `GET /api/training/sessions/{id}`,
  `/events?after=<seq>`, and `/checkpoints` views are recommended for
  observability/recovery; they do not create a parallel queue or control plane.

Every create/read/advance operation authorizes the current graph principal and
scope against the session independently of handle possession. Per-advance policy
inputs have a declared canonical schema, enter operation identity and the
journal, and do not mutate session-scoped `configDigest` invariants. A replayed
fold/while must derive byte-identical advance inputs from its immutable graph
inputs and prior handles; nondeterministic policy nodes use named durable RNG
state. A different operation id for an input that already committed is a lineage
conflict, not permission to run a divergent interval.

There is no separate `save-now` or `sample-now` session command in the first
contract: checkpoint cadence is the advance interval, and sampling/validation
are graph composition. Pause means cancel the graph invocation safely; resume
means re-run from the same handle/operation identity. A future out-of-band
operator command must use the same idempotency and journal rules rather than a
database flag.

### 9.2 One durable typed event stream

Current `InvocationEvent`/`node_event` is explicitly droppable chatter, and
per-job replay is in-memory and soft-bounded
(`packages/dinkster-protocol/src/dinkster_protocol/__init__.py:100-115`;
`packages/dinkster-server/src/dinkster_server/events.py:25-37,107-120`;
`packages/dinkster-server/src/dinkster_server/app.py:582-621`). Correctness-critical
training facts therefore live first in a durable per-session journal. Live graph
events mirror journal records and may coalesce telemetry; they are not the
source of truth.

Each durable event has `sessionId`, monotonic `seq`, timestamp, `advanceId`,
phase, session attempt/fence epoch, optional graph `jobRef` and node attempt,
and optional distributed rank. The initial union is:

- `session_created`, `session_recovered`, `phase_changed`;
- `advance_started`, `train_progress`, `metric`, `resource_report`;
- `recovery_checkpoint_published`, `checkpoint_published`;
- `preview_started`, `preview_published` with artifact refs, never giant JSON;
- `warning`, `diagnostic`;
- `cancel_requested`, `advance_paused`, `advance_resumed`;
- `advance_committed`, `advance_aborted`, `advance_failed`, `session_completed`.

Checkpoint publication, operation claims/commit, cancellation acknowledgement,
lineage/fence changes, failures, and terminal facts never drop. High-frequency
step progress, metrics, and resource samples may be coalesced before durable
append under a declared policy. Reconnect uses `after=seq`; a replay gap returns
the current authoritative handle/checkpoint, floor, and latest sequence.

Journal ownership (decided): `dinkster_server.journal.JournalStore` is the one
host-owned journal substrate - SQLite (WAL) behind a process-wide write lock,
one sequence/gap/retention schema for every stream family. Its
`transact`/`JournalTransaction` API is the host-owned transaction boundary: a
stream-family owner (the training session supervisor's claim/commit ledger,
and server-extension managed jobs when they adopt the substrate) keeps its
own tables in the same SQLite file and moves them in the same transaction
that appends its journal facts, so ledger state and durable facts can never
diverge. Workers still append through an authenticated boundary, never by
opening the file. No component may ship a private journal, incompatible
event envelope, or independent sequence semantics.

### 9.3 Safe-point cancellation through the graph path

Graph cancellation propagates to the running advance's `advanceId`. The host
durably appends `cancel_requested` and sends a cooperative cancellation request
to the training adapter. It stops admission of the next microbatch, but the
worker finishes the current indivisible operation, unwinds backward safely,
and reaches an optimizer-boundary safe point. It then stores a complete private
recovery checkpoint, appends `advance_paused`, releases transient autograd/phase
reservations, and acknowledges cancellation. The durable `advance_paused`,
`advance_committed`, or `advance_failed` journal fact is the supervisor's
acknowledgement channel; an invocation reply future that cancellation may have
abandoned is not authoritative. Re-running the same graph resumes that operation
for only its remaining steps; the graph-visible handle advances only when the
full interval commits.

Raw Python task cancellation is not this protocol. Today the generic boundary
turns engine cancellation into a `cancel` frame and the worker host calls
`task.cancel()` (`packages/dinkster-workers/src/dinkster_workers/session.py:478-490`;
`packages/dinkster-workers/src/dinkster_workers/host.py:1066-1079`). The training
worker adapter must intercept its cancellation as a safe-point token and delay
invocation teardown until acknowledgement or a bounded force-termination
deadline. Force termination is allowed only after the host records loss of the
hot attempt; recovery then uses the latest complete operation checkpoint. It
must never claim that an interrupted optimizer mutation or candidate-manifest
write was committed. ai-toolkit proves the command need but its SQLite flags and
polling are not the protocol to copy
(`/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:242-252`).

## 10. Distributed execution posture

### Designed now

The first implementation is one process and one accelerator, but the contracts
include:

- `worldId`, rank, world size, local rank, coordinator rank, device assignment,
  and membership epoch;
- deterministic global sample order and explicit rank partition with no
  duplicate data unless configured;
- rank-derived RNG streams and checkpointed rank states;
- metric/loss reduction semantics and rank ownership for previews, logs, and
  artifact publication;
- checkpoint barriers, immutable rank shards, coordinator-last manifest
  publication, and complete-world resume validation;
- cancellation/error fanout and all-rank safe-point acknowledgement;
- optimizer/model/gradient shard descriptors without selecting a framework.

This prevents ai-toolkit's failure mode where Accelerator wraps models but data
loaders remain unsharded
(`/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:254-260`).

### Deferred implementation

Do not make Accelerate, DeepSpeed, DDP, FSDP, or a custom model-parallel engine
a core dependency in the proof slices. Add a distributed orchestrator plugin
when one of these triggers is met:

- a supported family/model or dataset does not fit a single target accelerator;
- measured throughput demand justifies multi-GPU data parallel;
- a production deployment needs elastic recovery or multi-host execution;
- parity acceptance requires a concrete kohya distributed workflow.

The first implementation should be DDP-style replicated data parallel because
its semantics are simplest. FSDP/ZeRO and model parallel follow only with
measured memory need and explicit optimizer/checkpoint state contracts. Kohya's
Accelerate and DeepSpeed support defines the eventual parity bar
(`/home/kosin/node-analysis/dives/report-t2-kohya.md:128-129`).

## 11. Demoable roadmap

Each slice ends in a user-visible demo and a resumable artifact, not only an
internal abstraction.

### Slice 0: contracts and dry-run capability report

- Training extension scope/capability decision, registries, execution mode,
  target manifest, exact session handle, graph/session snapshot distinction,
  model-handle recipe validation, training-worker process boundary, resource
  categories, complete session manifest, authorization binding, and typed
  journal events.
- Coordinate with extension S2's recipe materializer. The fake Slice 0 proof may
  use a contract-faithful stub if S2 has not landed, but it must not create a
  production training-only loader; real cold recovery depends on the shared S2
  mechanism.
- Jointly settle with backend S5 the shared durable-journal owner/schema, and
  settle with the engine owner the advance effect policy, cache-key/runtime
  identity, training runtime identity, fence-conditional operation transaction,
  duplicate/supersession rules, single-writer fence, safe-point cancellation
  adapter, and session-scoped governor hold.
- One fake trainer extension proves a finite RegionNode handle loop, hot advance,
  recipe-based cold reload after worker death, concurrent duplicate refusal,
  retry after commit-before-cache failure, provisional-manifest crash recovery,
  transactional activation/session pinning, safe cancellation, and checkpoint
  commit without executing a microbatch in the graph process.
- Demo: submit dry-run and receive exact targets, precision/kernel support,
  estimated memory by category, and explicit refusals; then run a fake
  checkpoint-interval loop whose only state port is the canonical handle.

### Slice 1: one-family bf16 LoRA baseline

- Choose one already native image family; frozen floating base, float32 LoRA
  masters, image/caption dataset, one objective, AdamW, accumulation, previews.
- Complete optimizer/scaler/RNG/cursor resume from an interrupted run.
- Demo: cancel during an advance, restart the training worker, re-run the same
  graph operation, and match uninterrupted parameters/metrics at a fixed later
  step without double-stepping; export kohya-compatible LoRA and use it in
  Dinkster inference.

### Slice 2: quantized frozen-base LoRA

- Grad-input capability negotiation, custom autograd rematerialization, and
  governor accounting of forward/backward staging.
- Refuse unsupported packed layers loudly; no quantized full tuning.
- Demo: train the same adapter on a supported packed base under a smaller VRAM
  budget and show deterministic resume plus measured resource ledger.

### Slice 3: data parity foundation

- Aspect buckets, streaming shards, deterministic captions/transforms, latent
  and text-embedding cache manifests, prior preservation, validation set.
- Cache invalidation proofs and bounded image/video-ready shard interfaces.
- Demo: multi-resolution DreamBooth-style run with prior stream, cache reuse,
  preview events, and cursor-exact resume.

### Slice 4: adapter and serialization breadth

- LoHa, LoKr, OFT, full-delta, per-layer ranks/alphas, target include/exclude,
  adapter dropout, import/continue/export through kohya codecs.
- Demo: import a supported external adapter as initialization, train it, export,
  and compare Dinkster inference application against the source ecosystem.

### Slice 5: family, objective, and video breadth

- Additional image and video families through family training bindings;
  flow-matching densities, Min-SNR/debias/masked loss, control/mask datasets,
  temporal buckets and cache shards.
- Demo: extension adds a family binding/objective without modifying the common
  trainer state machine.

### Slice 6: full tuning, EMA, advanced optimizers, and offload

- Floating-master full tuning, optional fully resumable EMA, optimizer zoo,
  activation checkpoint regions, governor-owned block/optimizer offload.
- Demo: raw-versus-EMA validation and resume, plus a planned low-memory run with
  no trainer-local placement hooks.

### Slice 7: distributed and formal parity

- DDP-style backend first, then triggered FSDP/ZeRO support; deterministic rank
  partition and sharded atomic checkpoint.
- Publish a feature/format/quality parity matrix against the pinned kohya and
  ai-toolkit commits.
- Beyond parity: deterministic crash recovery, event replay, capability-first
  quant diagnostics, measured memory plans, and immutable provenance.

### Slice 8: control-adapter training (user directive 2026-07-29)

Controllability adapters - ControlNet, control-LoRA, LLLite-class lightweight
control branches, and T2I-style adapters - are a training target once the core
program is locked in. Evidence base: kohya ships per-family procedural
ControlNet trainers (train_control_net.py, sdxl_train_control_net.py,
flux_train_control_net.py) plus its own ControlNet-LLLite architecture and a
ControlNet dataset schema; ai-toolkit trains control_net/control_lora/t2i/ip
adapter types with control_path dataset streams, multi-control, control
dropout, and automatic control-image synthesis (toolkit/control_generator.py:
depth/pose/line/inpaint/mask). ComfyUI has no control-adapter training.

What Slice 8 adds on top of the core contracts:

- AUX-MODULE ATTACHMENT: extend TrainableAttachment (4.1) with a second
  attachment class - trainable auxiliary modules that OWN new parameters
  (zero-initialized control branches) and bind to the family definition's
  declared control injection seams, the same seams inference control uses.
  A control adapter trained here is inference-loadable with no port step.
- PAIRED CONDITIONING STREAMS: the dataset contract (6.1) gains ordered
  per-sample conditioning streams (control images/video), with per-stream
  dropout policy and cache shards keyed like latent shards.
- CONTROL SYNTHESIS PREPROCESSORS: ai-toolkit's auto-generation convenience
  becomes a dataset-prep artifact stage (preprocessor -> CAS-cached control
  shards), trainer-triggered but artifact-system owned.
- CUSTOMIZABLE ADAPTER ARCHITECTURES: the control branch structure itself
  (LLLite-style light branches vs full ControlNet copies vs new designs) is
  a trainer-domain plugin surface over the shared injection seams, so new
  control architectures do not require new trainers.

Prerequisites: Slices 1-4 (policies, quantized frozen base, data parity,
adapter/serialization breadth) and the inference-side control seam program.
Demo: train a depth-control adapter on one family, then drive it in an
ordinary Dinkster inference graph without conversion.

## 12. Vetoable decisions and open questions

These are decisions the user should be able to veto before Slice 0 closes,
except decision 12: its graph/worker hybrid is the accepted decision of record.
Decisions 13-20 are vetoable implementation contracts under that fixed hybrid.
The model-handle reconstruction-recipe invariant is also an accepted upstream
host decision (`docs/extension-design.md:149-162`), not a training-specific
alternative.

1. **Dedicated `TRAINING` extension scope?** Decided: yes. Training has a
   long lifecycle and accelerator/filesystem/artifact privileges distinct from
   inference; an inference-scope capability would conflate authority and
   process lifecycle. `ExtensionScope.TRAINING` and the `accelerator`/
   `artifacts` capabilities are in the closed vocabulary
   (`packages/dinkster-protocol/src/dinkster_protocol/extensions.py`).
2. **Backward rematerialization or graph-long weight leases?** Recommendation:
   custom-autograd rematerialization first for frozen packed bases. Permit
   graph-long leases later only when explicitly pinned and measured. Current
   leases forbid post-forward use
   (`packages/dinkster-inference-torch/src/dinkster_inference_torch/residency.py:152-186`).
3. **Canonical adapter target identity?** Recommendation: stable Dinkster
   family/component/module/role identity internally; kohya and other keys only
   at codecs. Never make an external spelling the runtime ABI.
4. **Default adapter precision?** Recommendation: float32 trainable masters and
   optimizer state, optional bf16 execution cast. This costs more than bf16
   parameters but gives a clear numerical baseline; expose a measured lower
   precision policy later.
5. **EMA default?** Recommendation: supported and fully resumable, opt-in at
   first because it is a parameter-sized memory cost. Validation/export select
   raw or EMA explicitly. Kohya's missing model EMA is a gap to exceed
   (`/home/kosin/node-analysis/dives/report-t2-kohya.md:127,220`).
6. **Checkpoint format?** Recommendation: versioned manifest plus immutable CAS
   shards, with the candidate manifest stored last and selected atomically by the
   durable session transaction. Do not make one monolithic mutable file the only
   resume product.
7. **Event replay durability?** Recommendation: monotonic per-session journal
   with bounded online replay and explicit gap recovery. Existing in-memory
   replay is useful but insufficient across a server restart
   (`packages/dinkster-server/src/dinkster_server/app.py:579-618`).
8. **Distributed framework?** Recommendation: framework-neutral rank/state
   contracts now; one DDP-style plugin only after a trigger. Do not choose a
   framework before the single-device contracts are proven.
9. **Full tuning of quantized/GGUF bases?** Recommendation: explicitly
   unsupported until a floating-master/QAT and deployment-requantization
   contract exists. Adapter training support must not imply base training
   (`/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:169-177,424`).
10. **Share a live instance for validation?** Recommendation: share model
    definitions always and share the instance when the governor can reserve and
    restore it safely. Allow a separately accounted validation instance, but
    never a duplicated family sampler stack.
11. **Checkpoint only at optimizer boundaries?** Recommendation: yes initially.
    Mid-accumulation resume adds gradient shards and harder plugin consistency;
    implement only after a demonstrated latency need.
12. **Training as graph nodes? ACCEPTED HYBRID, not open for relitigation.**
    Training executes in a dedicated isolated training worker with its own
    torch/autograd/process-global state. RegionNode fold/while orchestrates it
    through sequential state ports and a mandatory finite cap for while
    (`packages/dinkster-graph/src/dinkster_graph/model.py:90-130`). One graph
    iteration is one checkpoint interval, never one microbatch. The graph
    carries only an RPC-clean session handle and composes validation, previews,
    early stopping, LR policy, and curricula. The worker keeps hot state across
    advances and reloads complete CAS state when cold. Durable supervision,
    event journal, safe cancellation, idempotency, and governor accounting live
    below the graph, not in a second user-facing job type.
13. **Exact session handle?** Recommendation: the six-field v1 contract in
    section 3.2: `sessionId`, `checkpointManifestDigest`, `stepCursor`,
    `configDigest`, `sessionExtensionSnapshotDigest`, and `journalSeq`, plus
    `schemaVersion=1`. Validate all canonical encodings and require every
    duplicated manifest field to match before allocation. `journalSeq` is
    exactly the checkpoint-covered manifest watermark, not the later commit
    event. Authorize the graph principal separately; do not add worker affinity
    or live-object fields.
14. **Hot/cold behavior?** Recommendation: best-effort affinity by `sessionId`
    with one durable session fence. A dead/evicted hot worker reconstructs the
    base through the standard model-handle recipe, layers complete optimizer,
    trainable/EMA, RNG, cursor, and plugin checkpoint state, and emits
    `session_recovered`; corrupt/missing state, failed semantic reconstruction,
    incompatible hardware/policy, unavailable pinned generation, or unknown
    stale lineage is surfaced. Affinity never enters computation identity.
15. **Advance effect/cache classification?** Recommendation: add an explicit
    transactional `effectful_idempotent` engine policy. Ordinary cache replay
    is allowed only after durable commit; cache is not the transaction log.
    Until that exists, use never-cacheable execution plus worker deduplication.
    Backend review must pin every key component and decide the distinct
16. **Advance retry semantics?** Recommendation: a durable operation id over
    input checkpoint, session/config/snapshot pins, requested interval/seed
    policy, per-advance options, and training runtime identity. Claim, private
    recovery-pointer updates, and commit are fence-conditional; the
    operation/journal transaction, not CAS manifest storage, is the commit point
    and maps it to exactly one output handle. Active duplicate claims return
    retryable `advance_in_progress`;
    reply/cache-write loss returns the committed handle; partial work resumes
    from an operation-private checkpoint. An uncommitted operation may be
    durably aborted/superseded before a changed operation claims the same input;
    committed competing lineage conflicts. Explicit forks mint a new session id.
17. **Cancellation path?** Recommendation: graph cancellation durably requests
    cancellation by advance id; the training adapter stops at an optimizer-safe
    boundary, writes complete private recovery state, releases transient
    resources, and acknowledges through the durable journal, not a possibly
    abandoned RPC reply. Re-running the same operation performs only remaining
    steps. Generic `task.cancel()` is prohibited for the training mutation path
    except bounded force termination after the hot attempt is marked lost.
18. **Durable journal owner?** Decided: `dinkster_server.journal.JournalStore`
    owns durability, ordering, replay, and retention for every stream family,
    and its `transact`/`JournalTransaction` API owns the transaction boundary -
    stream-family ledger tables live in the same SQLite file and commit
    atomically with their journal appends. Managed jobs adopt this same
    substrate when they need one; workers append through an authenticated
    boundary. No private journals.
19. **Worker isolation and governor accounting?** Recommendation: a dedicated
    training process plus a session-scoped governor hold covering process,
    model, optimizer, EMA, RNG, and other hot baseline bytes across invocations,
    with phase sub-reservations. Every byte is reservation-owned or
    footprint-owned, never both; dead-worker accounting collapses and recovery
    starts cold. The node's `occupies` lane controls concurrency but never
    substitutes for byte accounting.
20. **Graph versus session extension pins?** Recommendation: preserve S0-B's
    per-graph `ExecutionRuntime` pin unchanged, and separately persist the
    training session's extension snapshot/config digests across graph runs.
    Advance resolves the session generation or explicitly migrates; it never
    adopts the current graph generation implicitly. Add a training generation
    lease/re-provision contract because current S0-B retirement may close old
    workers (`src/dinkster/activation.py:223-239`).

## 13. What must not be built

| Prohibited anti-pattern | Evidence | Preventing design rule |
|---|---|---|
| In-graph trainer node retaining resolution graphs | ComfyUI trains inside graph execution and accumulates activation graphs (`/home/kosin/node-analysis/dives/report-t1-comfyui-training.md:18-25,197-218`). | The graph node is a checkpoint-interval RPC adapter; microbatches and every autograd graph begin and end inside the training worker. |
| Global `inference_mode` or process-global training boolean | ComfyUI punches through executor inference mode and uses global mode (`/home/kosin/node-analysis/dives/report-t1-comfyui-training.md:49-55,322-348`). | Training has a dedicated isolated process plus immutable per-session execution policy; inference workers are never switched. |
| Live worker/tensor identity in a graph state port | Worker instance tokens are process-lifetime liveness facts (`packages/dinkster-workers/src/dinkster_workers/session.py:274-280`). | The canonical handle contains only session/checkpoint/config/snapshot/journal values; affinity is advisory and validated cold rebuild is the canonical recovery path. |
| Pure-cache declaration for mutating advance | Current `idempotent=True` is ordinary caching and false is never cached (`packages/dinkster-schema/src/dinkster_schema/model.py:975-1000`; `packages/dinkster-engine/src/dinkster_engine/engine.py:570-607`). | Advance is an explicitly effectful idempotent command with a durable operation ledger; cache is an acceleration after commit, never the transaction log. |
| Saving a forward-leased tensor for backward | Dinkster's lease explicitly expires after module forward (`packages/dinkster-inference-torch/src/dinkster_inference_torch/residency.py:152-186`). | Custom autograd saves mechanism identity and rematerializes, or uses an explicit graph lease. |
| Weight-only memory accounting or `free-and-hope` | ComfyUI misses activations, gradients, optimizer, scaler, and EMA (`/home/kosin/node-analysis/dives/report-t1-comfyui-training.md:543-577`). | Governor ledger accounts every training resource and admits before allocation. |
| Adapter-tensor-only resume called resume | ComfyUI creates a new optimizer and omits continuity state (`/home/kosin/node-analysis/dives/report-t1-comfyui-training.md:267-308`). | Versioned complete session manifest; tensor-only load is initialization. |
| One procedural full trainer per family | Kohya duplicates full-model orchestration by family (`/home/kosin/node-analysis/dives/report-t2-kohya.md:30-40,212-215`). | Common trainer state machine plus keyed family training bindings. |
| EMA omitted from state architecture | Kohya has no model EMA (`/home/kosin/node-analysis/dives/report-t2-kohya.md:127,220`). | Optional first-class EMA with memory reservation, validation swap, and resume. |
| Convention scanner and import-side-effect registration | ai-toolkit eagerly imports child packages and reduces UID mappings (`/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:49-59`). | Declarative manifests, transactional activation, and explicit composition. |
| Duplicate plugin UID silently overwrites | ai-toolkit's scanner reduction makes UID identity convention-driven (`/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:49-59`). | Keyed-registry collision fails activation with owner diagnostics. |
| Quantization failure swallowed per module | ai-toolkit prints and swallows quantization exceptions (`/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:135-146,407-409`). | Preflight capability report and loud layer-specific refusal; selected policy enters identity. |
| SQLite flags and UI polling as training protocol | ai-toolkit uses detached processes, SQLite controls, and polling (`/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:226-228,242-252`). | Ordinary graph jobs over one typed durable session journal, with safe-point cancellation and read-only replay views. |
| Separate training and S5 journals | Both need durable operation/event identity across process restart. | Journal owner/schema is a mandatory joint backend decision; neither slice ships a private sequence or store. |
| Ad-hoc training reload path or live module identity | The host requires model handles to remain reconstructible from declarative recipes (`docs/extension-design.md:149-162`). | Cold session recovery uses ordinary recipe materialization plus complete session shards; worker modules/tensors are only a cache. |
| Per-trainer random block swap or `.to()` monkeypatch | ai-toolkit randomly chooses modules for percentage offload (`/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:179-191,410`). | Model declares units/regions; governor owns deterministic placement policy. |
| Updating packed quantized bases without masters | Ordinary ai-toolkit quantized training freezes the base; QAT master is separate (`/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:156-177`). | `TRAIN_FULL` requires floating masters or explicit QAT; otherwise refuse. |
| Validation mutates training RNG, mode, or placement | Kohya must explicitly preserve these around live preview (`/home/kosin/node-analysis/dives/report-t2-kohya.md:176-193`). | Nested validation scope snapshots and restores all state in exception-safe cleanup. |
| Training-only checkpoint, cache, model-key, or sampler naming universe | The reports converge on sharing definitions and infrastructure while separating execution policy (`/home/kosin/node-analysis/dives/report-t2-kohya.md:145-174`; `/home/kosin/node-analysis/dives/report-t3-ai-toolkit.md:264-275,357-378`). | One family/storage/artifact/codec registry; training adds policy and durable state, not parallel semantics. |

## 14. Oracle review disposition

The design received an initial foundations review and a fresh review after the
graph/worker hybrid reconciliation. Both reviews checked load-bearing claims
against the cited Dinkster source.

### Foundations review incorporated

- Clarified that the custom autograd function wraps only the frozen base;
  `adapter_branch(x)` remains in ordinary autograd so adapter parameters receive
  gradients.
- Split exact grad-input from STE capability. The default quantized adapter path
  uses Dinkster's dequant route; a fused-fp8 forward with approximate backward is a
  separately declared future policy.
- Made pre-admission of backward rematerialization mandatory because synchronous
  autograd cannot await the governor's async reservation API.
- Defined initial exclusive accelerator admission for long training holds,
  required a declared device budget, and prohibited reservation/footprint
  double-counting.
- Corrected composition wording: Dinkster ships the declarative mode vocabulary,
  not active composition registries for these training surfaces.
- Recorded that the fused fp8 route's grad-enabled runtime guard does not protect
  code inside a custom autograd `forward`; capability preflight is authoritative.

### Post-hybrid review incorporated

- Made the fence-conditional operation/journal transaction the sole commit
  point. A CAS candidate without ledger commit is provisional rather than
  session state, closing the manifest-publication/ledger gap.
- Required fence checks at both claim and commit, so a stalled superseded worker
  cannot land after cold takeover.
- Moved authoritative fence and operation-ledger state out of the immutable
  checkpoint manifest; the manifest keeps only an as-of diagnostic digest or
  watermark while the shared durable session substrate owns mutable truth.
- Defined `journalSeq` as the checkpoint-covered manifest watermark, not the
  later `advance_committed` event sequence.
- Added active-duplicate refusal, supervised same-operation takeover, and
  explicit abort/supersession of uncommitted operations so a canceled claim does
  not brick its input checkpoint. Private recovery state now has commit/abort
  retention and GC rules.
- Made journal facts, rather than an abandoned invocation reply, the cancellation
  acknowledgement channel and added `advance_aborted` to the event union.
- Separated session-scoped config invariants from canonical per-advance policy
  inputs, required deterministic replay inputs, and bound session access to the
  current graph principal rather than handle possession.
- Clarified that `occupies` is a concurrency lane rather than governor byte
  accounting, refused zero-step advances, and limited reusable engine key
  components to the proposed transactional effect policy.
- A final re-review returned READY. Its two non-blocking observations were folded
  in by fencing private recovery-pointer updates and making the Slice 0 fake
  recipe proof coordinate explicitly with extension S2 rather than introduce a
  training-only materializer.
- The review found the hybrid, six-field handle, hot/cold recovery, process
  isolation, dual extension-snapshot lifetimes, S5 journal joint-decision scope,
  and no-private-job-model rule otherwise internally consistent.

### Foundations suggestions rejected or deferred

- The optional suggestion to name a concrete patch/version handle was deferred
  to Slice 0 contract design. The requirement remains normative: backward saves
  immutable mechanism/materialization identity, never a closed lease tensor.
- Graph-long leases remain permitted only as a later measured mechanism; the
  Oracle agreed rematerialization should be the first implementation.

## 15. Verification checklist

- [x] Oracle review incorporated and disposition recorded.
- [x] ASCII-only scan passes.
- [x] Every cited path exists and every cited line is in range.
- [x] Required sections cover ownership matrix, execution, adapters, memory,
  data/cache, extensions, durable resume, jobs/events, distributed posture,
  roadmap, vetoable questions, and prohibited anti-patterns.
- [x] No repository was modified; this design file is the only output.
