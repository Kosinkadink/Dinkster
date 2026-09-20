# Dinkster Extension Architecture - Design Proposal (rev 2, Oracle-reviewed)

Synthesized from: AGGREGATE.md (747 packs, 90% of local usage, usage-weighted)
and 8 deep-dive contract reports (dives/report-01..08). Evidence citations live
in those reports; this document is the decision layer. Rev 2 folds in Oracle
review findings (activation contract, vertical slicing, composition modes,
ControlNet contract, compatibility resolution, missing seams).

Status: PROPOSAL - no implementation until accepted.

---

## 1. What the ecosystem actually demands (usage-weighted)

| Demand | Usage % |
|---|---|
| Frontend JS extension | ~55% |
| Conditioning manipulation | ~55% (upper bound) |
| ModelPatcher-style patching | 39% |
| Server routes / ws events | 34% / 32% |
| Sampler registration | 32% |
| Model-folder/artifact registration | 30% |
| transformer_options-style per-call context | 24% |
| ops/weight-layer interposition | 21% |
| Out-of-tree model architectures | 16% |
| Core monkeypatching (missing-seam indicator) | 9% |

Most important negative result: ComfyUI's official hooks API - designed for
exactly these use cases - has 1.2% pack adoption. Packs route around sanctioned
APIs whenever the sanctioned path is harder than patching. **A seam that is not
the easiest way to do the job does not exist.**

## 2. Design principles (each traceable to observed failures)

P1. **Every seam declares its composition mode.** No implicit last-writer-wins.
    The modes are:
    - ordered transform list (e.g. pre/post-CFG transforms);
    - delegating wrapper chain with `next(...)` (e.g. attention kernel);
    - exclusive strategy, selected explicitly by the caller/owner, with
      collision diagnostics (e.g. the CFG reducer, owned by the guider; the
      full-loop SamplingDriver);
    - fan-out observer list (progress, previews, events);
    - keyed registry entry (samplers, schedulers, model families).
    ComfyUI's singular slots (`sampler_cfg_function`, optimized-attention
    override, replacement dicts) are the top source of pack conflicts
    (report-06); its list-valued seams are its most successful.

P2. **Typed, scoped state; no process globals.** Every hook receives an
    immutable per-evaluation context plus namespaced session/invocation scratch
    storage with exception-safe teardown. Easy-Use patches
    `comfy.samplers.sample` globally *just to see the sigma schedule*
    (report-03); ADE stores state in process globals (report-01).

P3. **Semantic targets, not object paths.** Extensions address blocks via
    host-resolved selectors; per-architecture adapters map semantic points onto
    real modules. Semantic point IDs are **capabilities published by each model
    adapter**, not a promise that every architecture has equivalent semantics:
    portable selectors match common tags; model-specific point IDs remain valid
    and diagnosable. Packs today hard-code SD block indices, check class names,
    and replace `__class__` at runtime (reports 01, 02, 05, 07). The attention
    seam must cover DiT paths, not just attn1/attn2.

P4. **Extensions declare requirements and preferences; the host owns
    arbitration.** Memory, device placement, cancellation, progress are host
    services; extensions submit declarative plans (residency groups, prefetch
    distance, workspace bytes). Even a full-loop takeover runs under host
    supervision (see 3.6). WanVideoWrapper bypasses host memory management and
    unloads other models globally - the wrapper-pack pattern is a vote of no
    confidence in these seams (report-05).

P5. **Everything is clone-/transaction-scoped.** Patches attach to model clones
    or executions, install/remove transactionally (including on exceptions),
    and are diagnosable (collisions, unmatched selectors, missing capabilities).

P6. **Registries over mutation.** Samplers, schedulers, guiders, noise
    providers, model families, latent schemas, weight-storage backends, codecs,
    artifact kinds: runtime-registerable with owner, version, collision policy,
    and unload. No static core lists; no path-table mutation.

P7. **Typed inter-extension discovery.** A versioned capability/service
    registry replaces `__DINKLINK`, imports of other packs' private modules,
    and NODE_CLASS_MAPPINGS lookups (reports 01, 03). Services declare their
    scope: server-local, inference-local, frontend-local, or explicit RPC -
    the registry never silently crosses process boundaries.

P8. **Four privilege levels, separately authorized:** (a) schema/widget,
    (b) graph-editor/canvas, (c) app/workflow, (d) server/event producers
    (report-08). Capability gates, route namespaces, and path confinement are
    **authorization and audit controls, not sandboxing**: in-process Python
    extensions are not contained. Actual containment requires a process/OS
    boundary (pyisolate-style), which is a deployment choice layered on the
    same declared capabilities.

## 3. The extension kernel (backend)

### 3.0 Activation, snapshots, and behavior identity (Phase 0 - prerequisite)
- Torch-free manifest discovery with separate entry points per scope: schema,
  server, inference, frontend, training. An inference or training dependency
  must never pull torch into the server process.
- S0-A uses one additive ``[pack.extension]`` TOML table, mirroring the
  established ``[pack.entry]`` key-to-``module:attr`` shape: the scope
  names are optional entry keys, while ``privileges`` and ``capabilities`` are
  explicit closed-vocabulary string lists. A scope entry requires its matching
  privilege. Discovery parses strings only; resolution/import belongs to later
  activation in that scope. This single table was chosen over per-scope
  tables so the requested authority and all code entry points remain one
  auditable declaration without changing any existing ``[pack.entry]`` field.
- Declarative registration, no import-time side effects. Transactional
  activation: every contribution of a pack activates or none do.
- Each execution pins an **immutable extension snapshot**; load/unload creates
  a new snapshot rather than mutating active runs.
- Behavior-identity inputs include: extension id/version/package
  digest, resolved contribution ids and order, selector resolutions, service
  provider choices, and behavior-affecting configuration. Cache invalidation
  and replay rules are defined against snapshot changes.
- The shared S0-A data/serialization contract lives in the RPC boundary leaf,
  ``dinkster-protocol`` (re-exported to pack authors by ``dinkster-api.v1``), so
  workers and future engine/server consumers share frozen types without an
  import cycle. Its v1 snapshot hash includes only extension/package identity,
  resolved contribution ids and behavior order, selector resolutions, service
  provider choices, capabilities, and behavior-affecting configuration.
  Presentation fields such as display names, descriptions, icons, and search
  terms are absent and therefore excluded.
- Without this, runtime registration would invalidate Dinkster's golden compat,
  caching, and reproducibility guarantees.
- S1 fixes inference-scope execution locality: inference entry points import
  and execute in the sampling worker, never in the parent or the pack's
  ordinary worker. The parent pins only RPC-clean keyed declarations and an
  import recipe keyed by the extension snapshot digest. Activation asks the
  sampling worker to materialize the callable registry and validates produced
  declarations in both directions before publishing the generation. V1
  therefore requires inference packs to be importable in the host inference
  environment; conflicting-dependency isolation is deferred in ROADMAP.

### 3.1 Execution context
Typed, immutable-per-call context threaded through sampling: full sigma
schedule; outer step ordinal AND model-evaluation ordinal (they differ in
multistage samplers); sigma/timestep; seed + named RNG streams; cond/uncond
identity and batch layout; latent descriptor + masks; frame/context-window
indices; device/dtype policy; cancellation token; hierarchical progress scope;
namespaced extension state (session/invocation lifetimes). This one contract
removes the single largest class of observed monkeypatches.

S1 ships the subset the current sampler loop can populate honestly: sigma
schedule, outer-step and model-evaluation ordinals, current sigma, seed with
deterministic named-stream derivation, cooperative cancellation, flat progress,
and invocation-lifetime extension namespaces. Builtin ``SolverFn`` remains
unchanged; a descriptor-marked context-aware solver is adapted at build time.
Session-lifetime state and nested progress remain deferred to their owning
slices in ROADMAP.

### 3.2 Model handle (ModelPatcher successor)
Cloneable model handle with: weight-patch overlays (normalized descriptors, see
3.3); object/module patches applied only while loaded; ordered keyed wrappers
at named phases (prepare-sampling, outer-sample, sampler-step, predict,
cond-batch, model-apply, diffusion-forward, control-apply); lifecycle callbacks
(clone/load/unload/pre-run/inject/eject/cleanup), exception-safe; declared
clone/share/device semantics per attachment; host-owned parent/lifetime.
**Region-transformable attachments**: an extension can clone a model view and
apply a scoped spatial transform to attached patch payloads with deterministic
restoration (USDU currently restores global state from `__del__`, report-04).
Typed block points beyond attention: FFN, normalization, residual injection,
positional/RoPE, model input/output, and reversible structural insertion
(reports 01, 03).
**Reconstruction-recipe invariant (user directive 2026-07-29)**: a model
handle carries, at all times, the complete declarative recipe to build itself
from scratch - weight-source digests plus storage-provider config (3.3), the
normalized patch overlay stack, attachments with their declared
clone/share/device semantics, and family/config identity. Live modules and
tensors are a materialization cache, never identity: dropping every
materialized byte and rebuilding from the recipe must yield a semantically
identical handle. Fresh-clone semantics, multi-GPU replication (a clone on
another device is a fresh materialization of the same recipe - the behavior
retrofitted into ComfyUI's ModelPatcher for multigpu), memory eviction/reload,
process-isolated workers, and training cold-session reload are then one
mechanism, not five. S2 ships this invariant; an extension attaching state
that cannot be rebuilt from serializable data must declare a rebuild callback
or the attachment refuses cloning loudly.

### 3.3 Weight storage + patch overlay protocol
comfy.ops' dependency-injection insight, formalized (report-05): lazy logical
parameters; pluggable storage providers (dense, GGUF, fp8, nvfp4, mmap);
materialize-for-(device,dtype,stream) lifecycle with explicit release; LoRA
parsed once into backend-neutral descriptors, and the storage backend chooses:
eager merge, deferred compute on dequant temporary, fused low-rank, requantize,
or explicit refusal. Resource accounting includes overlay bytes. **Compile
compatibility is part of this contract**: compilable regions, graph-break
declarations for dynamic materialization, scoped numerical settings, and
host-managed restoration of global backend settings (report-03; Dinkster's
aimdo/prepared-patch layer is the substrate this generalizes).

### 3.4 Guidance/CFG pipeline
Typed phases: cond evaluation (wrapper chain) -> pre-CFG transforms (ordered
list) -> CFG reducer (**one explicit reducer, owned by the guider**, collision-
diagnosed) -> post-CFG transforms (ordered list). Narrow guider protocol: a
custom guider overrides prediction policy only, inheriting lifecycle/device/
cleanup (NAG copied the whole lifecycle because the seam wasn't factored,
report-06). First-class re-entrant `evaluate_conditions` service for PAG/SEG-
style auxiliary predictions, with explicit phase participation and recursion
isolation.

S3 ships the non-re-entrant vertical slice: worker-local exclusive strategy
and reducer contributions, delegating cond-evaluation wrappers, ordered pre-
and post-CFG transforms, typed lane plans and prediction provenance, explicit
compose/bypass participation, deterministic ownership/order, strict callback
validation and cancellation boundaries, transactional unload, and native
SD/Flux wiring. Executed ComfyUI goldens cover ordinary CFG and CFG-rescale;
the empty registry preserves the exact prior native identity and active
PAG/SEG/auxiliary re-entry and terminal wrappers stay in S6; regional/masked
conditioning and per-condition scale vectors stay in conditioning scheduling;
ControlNet stays in S8. The
other retained boundaries and their exact revival triggers are recorded in
ROADMAP "S3 guidance retained deferrals".

Published inference generations are retained for the lifetime of their
sampling worker. Retention is therefore bounded by activation cycle count,
not by an LRU that could evict a generation still pinned by admitted work.
Replace this worker-lifetime bound with lease/refcount release when the
server/queue can signal that no admitted or queued execution references a
retired generation.

The accepted forced-uncond plan refusal occurs at the first denoiser call,
before model evaluation but after solver entry.

### 3.5 Attention/block extension
Three composable layers: q/k/v input transforms (ordered), kernel wrappers with
`next(q,k,v,ctx)` (termination must be explicit), output transforms (ordered).
Semantic block selectors resolved by adapters per P3; per-call context carries
shape/heads/polarity metadata; invocation-local scratch for paired pre/post
patches (attention-couple). Per-model attention-backend registry instead of a
host-global function. Reference/style read/write hooks (AdaIN etc.) as declared
points, not block-forward replacement (reports 01, 02, 06, 07).

### 3.6 Sampling as a library
Re-entrant sampling service callable from any extension: model-or-guider,
conds, latent+metadata, seed or exact noise tensor, sampler/scheduler or
explicit sigmas, denoise or sigma sub-range, masks, add-noise/leftover-noise
semantics, callbacks, cancellation - deterministic and nestable (report-04).
Noise as injectable data objects (deterministic, maskable, mixable,
batch-aware) with named RNG streams and retained Brownian paths (reports 02,
04). **Sampler checkpoints v1 are a host envelope**: sampler id/version,
schedule fingerprint, position, RNG snapshot, plus an opaque plugin-owned
serializable payload; a universal schema is deferred until two stateful
samplers prove common fields (RES4LYF evidence, Oracle change 8). Trajectory
middleware (split/resume schedules) is a later capability separate from the
first nested-sampling milestone. A genuinely different algorithm may register
an exclusive **SamplingDriver** (full-loop takeover) that runs under an
**ExecutionSupervisor**: host model/resource/progress/cancellation services
remain in force (Wan evidence, report-05).

S1 also ships ``inference.samplers`` as the first real ``keyed_registry``
surface. Each published generation is builtins plus worker-materialized pack
contributions through the existing collision-loud ``Registry``. The pinned
snapshot contains only sampler ids, aliases, and behavior metadata. Native
sampler lookup and both native sampler choice lists derive from that composed
registry, so registration, execution, and dropdown vocabulary have one source
of truth. SamplingDriver, nested sampling, and checkpoint envelopes remain
later slices.

### 3.7 Conditioning system
Typed conditioning records: channels beyond text (hints, concat, camera,
reference latents), masks/areas with model-aware materialization, normalized
start/end percent ranges converted to sigma centrally, clone/combine/
region-transform (`for_region`) covering every metadata kind (report-04), and
extension metadata that survives preparation. Scheduled patch/hook groups
shared by text encoder and diffusion model, with declared caching policy
(repatch vs cached variants) (report-07). Token-layout descriptors per model
(image-token ordering, text streams, temporal packing) so regional
conditioning does not require class replacement (report-02).
**Text-encoder seams** (report-07): tokenizer access with stable token/span
metadata, per-encoder prompt routing, a supported wrapper around token-weight
encoding, post-encode tensor/pooled transforms, schedule-aware re-encoding
with effective-range intersection.

### 3.8 Plan compilation (graph expansion)
Deterministic, cache-aware expansion of compact nodes into ordinary operations
with stable generated identities, so scheduling packs remain graph compilers
(report-07). Prompt preprocessing is a pure compilation step whose output
participates in behavior identity - not an arbitrary mutable callback
(Impact's prompt handlers, report-04). The plan compiler is distinct from
frontend graph transactions: one determines execution and caching; the other
edits the user's document.

### 3.9 Model families, latent schemas, codecs, artifacts
Runtime model-family bundles (detector + precedence, config, architecture
factory, conditioning schema, latent schema, VAE/text-encoder factories,
sampling defaults, memory estimator, versioned capabilities). Typed
multimodal latent schemas (named optional streams, layouts, mask alignment)
instead of `{"samples": tensor}` conventions. Codec plugins with
chunking/halo/temporal-state contracts and preview decoders (report-05).
**ArtifactStore** as an explicit contract (30% usage): logical artifact kinds
with aliases, validators (extension/magic-byte), role metadata, recursive
enumeration, duplicate policy, safe path handles (no raw path-table mutation),
and state-dict loading that accepts already-loaded data.

This section specifies the target architecture, not a running pack door.
`model-family-registration` is declared as capability metadata but has no
pack-facing registration path. Current capability status is listed in
[Pack routes, events, and frontend modules](supported/pack-routes-events-and-frontend-modules.md#extension-capability-status).

### 3.10 ControlNet pipeline contract (core)
Not just a `control-apply` wrapper (report-01): immutable/cloneable ordered
control chain; hint preparation; per-run lifecycle (configure hint, pre-run
with model/schedule/latent format, per-step evaluate, weighted residual merge,
cleanup); per-layer/per-frame/per-mask/cond-vs-uncond weighting; resize/dtype/
device normalization with source/target shape metadata; context-window index
propagation; decorator/interceptor mechanism so extensions never replace
methods on host controls. Dogfooded by a first-party image ControlNet pack -
"core contract + first-party implementation" is the only option consistent
with the no-private-hook rule.

### 3.11 Video/context windows
Window generation, scheduled length/stride/overlap, one standard frame-index
map slicing conditioning/masks/controls, overlap fusion policies,
ordered-state constraints - host-owned (ADE rebuilt all of it privately and
ACN interoperates through untyped `ad_params`, report-01). A later slice; the
image pipeline does not wait for it.

## 4. Server + frontend surface

Server (early - used by a third of the ecosystem): namespaced authenticated
routes with schema validation; typed JSON event bus with stable execution
events (prompt id, node identity incl. nested graphs, progress, cache state,
errors); lifecycle-managed background producers (no import-time daemon
threads, report-08); capability discovery. **Binary media channels
(versioned framing, MIME, correlation, backpressure) and upload/transcoding
are a later media-specific slice** (VHS's 24-byte header hack is the
cautionary tale). Current server capability status is listed in the
[capability status table](supported/pack-routes-events-and-frontend-modules.md#extension-capability-status).

Frontend design covers stable node schema, IDs, manifest fields, event
correlation, widget registration, render layers, menus, graph transactions,
app-shell mounts, workflow documents, and settings. This is a target surface,
not a claim that every contribution kind is wired. The generated vocabulary's
current per-kind status is listed in
[Pack routes, events, and frontend modules](supported/pack-routes-events-and-frontend-modules.md#extension-contribution-status).

## 5. Packaging, identity, compatibility

Extension factory allowlists pin every site by path, line, column, and owning
issue. Their ceilings are non-increasing; raising one is an explicit reviewed
decision, never an effect of regeneration. After merging main, run
`uv run --locked python scripts/check_extension_factories.py --write`, review
that only expected line or column coordinates changed and no ceiling changed,
then run `bash scripts/ci-fast.sh`. A ceiling lowered by `--write` is permanent;
reverting a removal requires both the re-added allowlist entry and an explicit
ceiling raise.

- Packs declare: id, version, required host API version, capabilities (routes,
  filesystem, downloads, background jobs, model-family registration,
  accelerator, artifacts), and provided services (per-scope, P7). A declared
  capability is not evidence of a runtime consumer; see the authoritative
  [capability status table](supported/pack-routes-events-and-frontend-modules.md#extension-capability-status).
- No import-time side effects; no runtime pip (17% of packs do it today -
  supply-chain risk); dependencies resolve at install time.
- One node/schema format from day one: stable internal IDs, typed custom
  socket types, expansion declarations.
- **Compatibility posture (decided): clean-break runtime API.** Provide a
  workflow/schema importer (stable-ID translation for graphs authored against
  ComfyUI - justified by V1's 75.9% usage share) and a migration SDK mapping
  common concepts (model clones, sampler descriptors, conditioning records,
  artifact lookup). NO ComfyUI Python module shim: packs rely on process
  globals, torch objects, private paths, import-time effects, and LiteGraph
  internals; emulating that would undermine the architecture. An out-of-core
  `dinkster-comfy-bridge` may come later, driven by measured pack ports.

## 6. Core vs extension packs

Core: sections 3.0-3.10, server surface, packaging/security. Optional
first-party packs dogfood the seams: samplers beyond the pinned catalog,
ControlNet implementation, regional prompting, detailer/tiled composition,
quantized storage backends, monitoring dashboard. Rule: if a first-party pack
needs a private hook, the seam is wrong - fix the seam, never add a private
channel.

## 7. Implementation slices (vertical, demo-first, dependency-ordered)

Every slice must prove: two extensions coexisting, deterministic ordering,
exception/cancellation cleanup, unload returns to baseline, behavior-identity
invalidation, golden output/event stability.

Repository guards that keep pack doors and core behavior aligned are tracked
in [comfy-vibe-station#120](https://github.com/Kosinkadink/comfy-vibe-station/issues/120).

- **S0. Activation + snapshot + behavior identity** (3.0). Proof: a trivial
  pack loads/unloads; goldens stable across snapshots.
- **S1. Minimal execution context + sampler registry** against the existing
  loop. Proof: one out-of-tree sampler pack, zero core edits.
- **S2. Normalized patch identity + aimdo adapter, then model-handle
  clone/lifecycle** (3.2 core, 3.3 descriptors). Proof: LoRA-style pack.
- **S3. Guidance phases - SHIPPED 2026-07-29** (3.4). Proved by two
  coexisting out-of-tree packs, a CFG-rescale contribution, native-arm and
  identity tests, and executed ComfyUI guidance goldens. PAG remains deferred
  to S6; all retained boundaries and triggers are ledgered in ROADMAP.
- **Conditioning records, graph compilation, and schedule-aware text
  encoding** (3.7, 3.8). Proof: prompt-scheduling pack.
- **S5. Server routes + event bus + managed jobs** (4, JSON only). Proof:
  monitoring pack. Parallelizable with guidance and conditioning scheduling.
- **S6. Semantic block/attention points + auxiliary prediction** (3.5, rest of
  3.4). Proof: PAG and a reference/attention-couple behavior on two different
  architectures.
- **S7. Re-entrant sampling + exact noise + nested progress** (3.6 minus
  driver/checkpoints). Proof: base/refiner continuation (not a full detailer).
- **S8. Image ControlNet pipeline** (3.10). Proof: first-party ControlNet pack.
- **S9. Storage providers + ArtifactStore + compile contracts** (3.3, 3.9
  artifacts). Proof: one quantized-storage pack.
- **S10. Model-family registration** (3.9), proposed and not available through
  a pack-facing door. Its proof target is the **simplest genuinely out-of-tree
  architecture** - a video model is an integration program, not a demo.
- **S11. Sampler checkpoints (envelope) + SamplingDriver + trajectory
  middleware** (rest of 3.6). Proof: stateful sampler resume; one full-loop
  driver under supervision.
- **S12. Video: latent/codec/context windows** (3.11, codec parts of 3.9).
- **S13. Media channels + uploads** (rest of 4 server).
- **S14. Frontend RFC -> implementation.** The implemented subset and the
  declared but unconsumed kinds are listed in the authoritative
  [contribution status table](supported/pack-routes-events-and-frontend-modules.md#extension-contribution-status).

Ordering rationale: S0 protects reproducibility before anything registers at
runtime; S1-S3 remove the largest monkeypatch classes and unblock the highest
usage-weighted backend demand (samplers/guidance); conditioning scheduling and
the following semantic attention work unlock the
conditioning/regional families; S7-S10 cover composition, control, storage,
and out-of-tree models; S11-S14 are capability completions.

## 8. Resolved questions (user decisions, 2026-07-28)

1. Frontend timeline: RESOLVED - the frontend RFC (S14) starts now, in
   parallel with backend planning/implementation. Drafting is in progress
   against the existing Dinkster-Frontend architecture (greenfield TS core,
   retained-canvas renderer, SolidJS shell, public widget registry).
2. Containment: RESOLVED - extensions run in-process initially, but all S0
   seam contracts are written RPC-clean from day one: no live Python objects,
   torch modules, or closures cross seam boundaries; everything is typed,
   serializable handles. Process isolation (pyisolate-style) then becomes a
   per-privilege-level deployment option later: server/dashboard extensions
   isolate first (I/O-bound), inference hot-path extensions stay in-process
   until RPC cost is proven acceptable. Capability declarations are
   authorization/audit controls, not containment, until isolation is enabled.
