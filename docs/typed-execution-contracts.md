# Typed execution contracts: interventions, timeline, mesh, manifest, media

Spec for the portable typed intervention plan, activation timeline
algebra, generic process mesh, and distributed execution manifest
(issue #174, step 1: spec only, no implementation). All type and
field names in chapters 1-10 are provisional; the semantics are the
contract. Chapter 11 defines the versioned VIDEO value contract for #1250.

Inputs this spec is drafted against (authoritative records):

- #174 issue body and its accepted-input comments (survey mesh/seam
  inputs; USP contract acceptance and drafts supplement).
- #185 governing comments: design, SOTA compatibility constraints,
  reconciliation, ring probe disposition.
- #189 distributed cache contract and its survey-derived sharpenings.
- #186 H3 capability contract targets (per-token AV noise strengths,
  motion guides).
- #219 native control audit and its recorded accepted recommendations.
- #121 multigpu survey report.
- #198 windowed-evaluation investigation (tiling/context windows)
  and its recorded user design input (wrap-around windows).

Scope: this document specifies contracts for the sampling engine, media
values, and distributed execution. It changes no wired behavior; it defines
the vocabulary that implementations must satisfy.

## 1. Typed intervention plan

### 1.1 Contribution values

An intervention enters the system as an immutable
`InterventionContribution` value:

- `site`: a `SiteId` from the declared site vocabulary (chapter 2),
  which is the union of the model family's declared sites and the
  engine's declared run sites. A contribution never names code
  locations, module paths, or layer indices directly; it names a
  declared site.
- `effect`: a typed effect kind drawn from a closed, versioned
  vocabulary. Initial kinds:
  - `WeightOverlay`: an ordered weight-overlay stack entry, named by
    content-addressed patch identity. The payload is the portable
    projection of the existing patch algebra in
    `packages/dinkster-inference/src/dinkster_inference/patches.py`: the
    structural data (patch kinds, content-addressed tensor leaves,
    strengths, offsets, nesting) travels as declared values, while
    every behavioral leaf in that algebra (a `WeightAdapter`
    implementation, a `NestedPatch.convert` converter, a
    `PatchEntry.function` delta hook) is named by a declared provider
    identity with a verified behavior version, never serialized as a
    callable. A patch value containing a behavioral leaf that no
    provider declaration names refuses at plan compilation with a
    named refusal (the existing unsupported-adapter reporting in the
    payload walk is the seam).
  - `AttentionTerm`: attention contribution or replacement for a
    declared site (chapter 2.1 selectors), below whole-component
    granularity.
  - `NoiseStrengthField`: per-token noise strengths on a declared
    token grid.
  - `ContextInjection`: typed context payloads at declared sites.
  - `ResidualTransform`: transforms of declared residual-site
    activations.
  - `SigmaScheduleOverride`: a directional sigma plan with an
    explicit initial-noise policy and explicit continuation state.
    Sentinel encodings (zero sigmas meaning "no initial noise" or
    "reverse direction") are forbidden; direction, initial-noise
    policy, and resume-from state are declared fields.
  - `NoiseSourceOverride`: a typed noise provider for initial, step,
    or substep noise, with its own declared RNG stream identity and
    serializable RNG state.
  - `GuidancePolicyOverride`: per-phase or per-stream guidance
    policy and post-combination (post-CFG) output transforms,
    consuming the lane vocabulary of chapter 1.3.
  - `ConditioningTransform`: a declared transform of conditioning,
    including deferred materialization against model or latent
    geometry resolved at plan compilation.
  - `SolverOverride`: selects a declared solver implementation by
    versioned provider identity (registered through the same
    descriptor/registry conventions as existing samplers) with a
    typed parameter payload; binds at the `run.solver` site
    (chapter 2.2). The solver owns its numerical loop
    (explicit/implicit stages, inner iterations, per-stage and
    per-substep parameters) and interacts with the engine only
    through the declared solver-stage sites (chapter 2.2) and the
    state contract of chapter 1.3.
  - `AuxiliaryResourceRef`: a content-addressed reference to an
    auxiliary model or resource a contribution needs.

  New kinds extend the vocabulary by revision; strings are never
  open-ended. Cache policy is NOT an effect kind: it is its own
  compiled value (chapter 7) so one fact has one carrier.
- `payload`: an immutable typed payload matching the effect kind.
  Payloads are declared data (shapes, spans, bounded scalars,
  content-addressed references), never callables that close over
  hidden state. No arbitrary Python callable is ever part of
  portable plan identity; worker-local implementations are selected
  through immutable provider declarations and verified behavior
  versions.
- `activation`: a timeline predicate (chapter 3) stating when the
  contribution applies.
- `gain` and `application_operator`: gain-capable effects carry the
  shared control-contribution values of chapter 10. Gain stays
  orthogonal to activation: activation decides whether the
  contribution is present, while gain and its typed operator decide
  how an active contribution combines with the declared site value.
  Effects whose resolved site does not declare a chapter 10 operator
  refuse these fields rather than ignoring them.
- `partition`: the contribution's partition-compatibility
  declaration (chapter 5.1), stating which sequence partition modes
  its payload semantics survive.
- `priority` and `conflict_policy`: explicit ordering among
  contributions targeting the same site. Supported policies:
  `refuse` (default: two same-site, overlapping-activation
  contributions without an explicit order are a compile error),
  `ordered` (apply in declared priority order), and `exclusive`
  (highest priority wins, others are dropped and recorded).
  Silent last-writer-wins does not exist.

Contributions are pure values: hashable, serializable, comparable.
Two runs given equal contribution sets compile equal plans.

### 1.2 Plan compilation

Contributions are compiled into an immutable `InterventionPlan`
before execution begins and, in distributed execution, before any
collective is issued. Compilation:

- resolves every `SiteId` against the declared site vocabulary -
  the union of the model family's sites and the engine's run sites
  (chapter 2.1) - and refuses unknown or type-incompatible sites
  (fail closed);
- validates payloads against the site's declared contract: a model
  site's declared geometry (for example a token-grid field must
  match the declared grid), or a run site's typed value contract;
- resolves priorities and conflict policies into a total, explicit
  application order per site;
- after the exact executed timeline is fixed, compiles every
  declared gain curve, keyframe hold, and effect mask into the
  realized values of chapter 10 and validates the selected
  application operator against the resolved site;
- in distributed execution, checks every contribution's partition
  compatibility declaration (chapter 5) against the chosen mesh and
  partition plan;
- produces a deterministic `plan_digest` (sha256 over the canonical
  serialized plan) that binds into invocation identity and receipts
  (chapter 6).

An empty contribution set compiles to the empty plan, whose
execution is byte-identical to today's un-intervened path.

### 1.3 State ownership

Interventions that need cross-step state (caches, running
statistics, chained-generation context) obtain it from the engine as
run-scoped, lane-keyed storage. The engine owns the lifecycle:
allocated at plan execution start, keyed by (plan, guidance lane),
dropped at run end. Contributions never stash state in module
attributes, globals, or closures. Lane keying follows the guidance
vocabulary in `packages/dinkster-inference/src/dinkster_inference/guidance.py`;
interventions consume lane semantics from that layer and never
redefine them.

Continuation state that must survive a run (solver state, RNG
streams, chained-generation context) is a typed, serializable value
returned by the engine, distinct from user latent metadata: it is
never smuggled inside latent dicts, and resuming from it is an
explicit declared input of the relevant contribution, not an
inference from tensor contents.

### 1.4 Per-call context

Model calls receive a typed, immutable per-call context value
compiled by the engine from the active plan: the contributions whose
activation predicates hold at the current step, resolved to the
current call's sites, lanes, and shard coordinates. There is no
mutable dict bus threaded through model options: families read
declared, typed fields, and a contribution that needs to reach a
model call does so by targeting a declared site, never by mutating
shared per-step state.

## 2. Site vocabulary

### 2.1 Sites are declared data

A `SiteId` is a stable, versioned identifier declared as data by its
owner - never a code branch, hook list, or monkey patch. Sites have
two owners:

- model sites, declared by a model family alongside its geometry,
  whose contract is a tensor contract (rank, layout, dtype class,
  sequence/token-grid geometry);
- run sites, declared by the engine itself, whose contract is a
  typed value contract at run scope (which effect kinds bind, at
  what timing they resolve: plan compilation, run start, or
  per-step).

Both kinds carry the identifier and its revision, the effect kinds
the site accepts, and - for model sites - the site's partition
behavior under sharded execution (chapter 5): token-local,
sequence-global, or head-scoped.

Site declarations may be parameterized by declared selectors:
component role, block index or block range, attention kind,
modality, and guidance lane. The attention-kind selector vocabulary
is versioned and carries `self`, `cross`, `joint`, `causal`, and an
any-kind selector (joint attention already exists in declared
geometry: `packages/dinkster-inference/src/dinkster_inference/qwen_image_layout.py`);
new kinds extend it by revision. Selectors are part of the
declaration, validated at plan compilation against the owner's
declared geometry; a selector value the owner did not declare is a
compile-time refusal, never a silent no-op.

An owner that does not declare a site does not support it; plans
targeting it refuse at compile time. Sites are additive: new
declarations never change the meaning of existing ones.

### 2.2 Initial site entries

Engine-owned run sites (typed value contracts, no tensor geometry):

- `run.solver`: solver selection for the run; accepts
  `SolverOverride`. Resolves at plan compilation, before any
  numerical loop exists; the compiled plan names exactly one solver
  identity (conflict policies of chapter 1.1 order competing
  contributions).
- `run.sigma_schedule`: the executed sigma plan; accepts
  `SigmaScheduleOverride`. Resolves at run start; the engine-owned
  timeline (chapter 3) reflects the executed result.
- `run.noise.initial`, `run.noise.step`, `run.noise.substep`: noise
  provider selection per draw class; accept `NoiseSourceOverride`.
  Initial-noise providers resolve at run start; step/substep
  providers resolve per step and may carry activation predicates.
- `run.guidance_policy`: guidance policy and post-combination
  transforms; accepts `GuidancePolicyOverride`, lane-scoped through
  the guidance-lane selector. Resolves at run start; per-phase and
  per-stream variation within the run is expressed through the
  resolved policy's chapter 3 activation predicates, never by
  re-resolving the site mid-run.
- `run.conditioning`: conditioning transforms, lane-scoped; accepts
  `ConditioningTransform`. Deferred materialization resolves at plan
  compilation against declared model/latent geometry.
- `run.resources`: auxiliary resource binding; accepts
  `AuxiliaryResourceRef`. Resolves at plan compilation;
  content-addressed identities bind into the manifest.

Model sites:

- `weights.<component>`: ordered weight-overlay stack sites per
  declared component, accepting `WeightOverlay` contributions.
  Overlay order is the compiled plan's explicit application order
  (chapter 1.1); activation windows make overlays step- or
  sigma-scoped without family code changes.
- `attention.<selector>`: attention route/replacement sites,
  selector-scoped (block range, attention kind, modality) below
  whole-component granularity, accepting `AttentionTerm`
  contributions. This carries regional attention masks, QKV
  injection, and style-injection routing as typed payloads on
  declared sites, never as module `__class__` replacement or
  object patches.
- `solver.pre_model_call`, `solver.post_model_call`,
  `solver.pre_stage`, `solver.post_stage`: solver-stage sites
  exposing the declared numerical-stage tensor contract (current
  latent, epsilon/denoised prediction, stage row, sigma, active
  masks) to `SolverOverride` implementations and stage-scoped
  contributions. Persistent solver state flows through chapter 1.3,
  never through these sites' tensors.
- `noise_strength.<modality>`: a per-token noise-strength field
  compiled from one source mask by the family's single declared
  mask-to-token-grid transform. Strengths are bounded fractional
  values in [0, 1], optionally quantized by family declaration;
  they are not booleans. The source mask and the compiled token
  grid are intentionally distinct representations of one fact:
  model timestep/modulation-row selection and sampler-side
  preserved-content (inpaint) correction consume the SAME compiled
  token strengths, while the sampler's final output blend applies
  the source mask at its full resolution. A second token transform
  or grid per modality is forbidden; so is requiring the final
  blend itself to use the pooled grid (that would regress to the
  superseded snap semantics). For MiniMax H3 (ComfyUI PR #15375
  executed semantics): video pools per 2x2 latent patch cell by
  max, audio pools per whole latent frame by max over feature
  channels, both quantized to 1/256 levels.
- `motion_guide.context`: typed context-injection sites for
  motion-guide and chained-generation interventions. Activation
  (when a guide applies) is a timeline predicate (chapter 3);
  placement (where its payload lands in the media) is a media
  placement coordinate (chapter 3.4). Neither is a family sampler
  branch.
- `cache.pre_block`, `cache.first_block_residual`,
  `cache.final_residual`, `cache.local_sequence_output`: the stable
  cache hook boundaries (chapter 7). `cache.pre_block` exposes the
  block-input indicator tensors; the residual sites expose the
  first-block and final-residual activations in the family's native
  layout; `cache.local_sequence_output` exposes each rank's local
  sequence output in its native shard layout.

### 2.3 Token layout vocabulary and padding

Token geometry is family-declared data, expressed in a shared typed
vocabulary (names provisional):

- `ModelTokenLayout`: the one declared ordered global model sequence
  for an invocation, with its valid semantic-row count. It carries
  semantic facts only - what each global row means. It is distinct
  from the existing `LatentPackLayout` (latent packing) and from
  shard-mechanics metadata (chapter 5.3).
- `ModelTokenSegment`: a segment of the layout with modality/role, a
  stable identity, a half-open global `[start, stop)` span, and its
  declared token-grid geometry.
- `TokenGridTransform`: the family-declared transform from declared
  source geometry to global model-token rows - exactly one per
  modality (chapter 2.2).
- `TokenRowTable`: dense per-row contributions or selectors over the
  global sequence (`[S_global, ...]`).
- `TokenRowSpan`: a compressed `(start, stop, row)` run over global
  rows.

Layouts are derived from declared packing geometry, never measured
from live tensors. `padded_rows` sit explicitly outside all
segments: they are structural only and are excluded from all site
semantics - they never receive noise-strength values, never map to
timestep/modulation rows, never participate in attention K/V, never
enter real-token digests, and never appear in the canonical output
gather.

Under sequence partitioning (chapter 5), dense `TokenRowTable` rows
slice by the same shard `[start, stop)` spans as hidden states and
RoPE; `TokenRowSpan` values localize by half-open intersection and
offset translation (`translate_segments` in
`packages/dinkster-inference/src/dinkster_inference/sequence_partition.py`).

## 3. Activation timeline algebra

### 3.1 Canonical anchors

The sampling engine owns one canonical timeline per run, exposing
three anchor coordinates for every denoising step:

- `step_index`: integer position in the executed step sequence;
- `sigma`: the step's noise level in the run's sigma space;
- `progress`: normalized [0, 1] position across the executed
  schedule.

All three are engine-computed facts. Families and interventions
never gate on privately counted step indices, wall time, or
call-counting heuristics.

### 3.2 Predicates

An activation predicate is an immutable expression over the anchors:
interval constraints on any anchor (`step_index in [a, b)`,
`sigma >= s`, `progress < p`), warmup windows (first N executed
steps), forced-full steps (explicit step indices where caches and
skips are disabled), and boolean combinations (and/or/not) of the
above. Predicates are total functions of the anchors: evaluating a
predicate at a step is deterministic and side-effect free.

### 3.3 One algebra, two consumers

The intervention plan (chapter 1) and the distributed cache contract
(chapter 7) share this algebra. A cache warmup window and an
intervention activation window are the same predicate type evaluated
on the same anchors. This is what makes distributed decisions
provable: because anchors are engine-owned facts identical on every
rank, any predicate evaluated on them is group-consistent by
construction. Input-dependent decisions (for example residual-drift
cache skips) are NOT predicates in this algebra; they are distributed
control flow and must go through the reduction contract in
chapter 7.

### 3.4 Media placement coordinates

Denoising activation (when a contribution applies) and media
placement (where its payload lands) are distinct typed coordinates.
A media placement is declared in media terms - frame anchors that
are positive (from the media start) or end-relative (from the media
end), with spans - and compiled at plan time through the declared
`ModelTokenLayout` (chapter 2.3) to global token rows/spans. Padded
rows are never part of a compiled placement. A contribution may
carry both coordinates (a motion guide active for early sigma on
the last N frames); neither substitutes for the other, and neither
is inferred from tensor contents.

## 4. Generic process mesh

### 4.1 Mesh values

A process mesh is an immutable, versioned value describing how the
ranks of one component's workgroup are arranged:

- an explicit ordered rank list `0..N-1`, validated exactly (no
  gaps, no permutation);
- ordered named axes with integer degrees; the product of degrees
  equals the rank count;
- a layout version identifying the coordinate convention.

Coordinates map to ranks in row-major order over the declared axis
order. Mesh instances are per-component: the sampling mesh and any
VAE mesh are independent values with independent lifetimes; VAE
parallelism is never an axis of the sampling mesh.

### 4.2 Sampling mesh axes

The sampling mesh declares the ordered axes
`("guidance", "tp", "sp_ulysses", "sp_ring")`:

- `guidance` (outermost): CFG/guidance lane parallelism.
- `tp`: tensor parallelism (reserved; degree 1 in #185's scope).
- `sp_ulysses`: Ulysses head-scatter sequence parallelism.
- `sp_ring` (innermost): Ring sequence parallelism. Innermost
  placement makes ring neighbors flat-rank adjacent for P2P
  locality.

Sequence parallelism is always the typed product
`sp_ulysses x sp_ring`, never an opaque SP degree and never inferred
from flat rank arithmetic.

### 4.3 Degree-1 elision and the degenerate mesh

Axes with degree 1 are present-but-elided: they are elided from
subgroup construction (no single-member transport groups) but NEVER
from identity facts. Canonical mesh identity always records the full
normalized form (for example `cfg1xtp1xu2xr2`), so U4R1, U1R4, and
single-GPU are the same type with degenerate degrees. The
single-GPU mesh (all degrees 1) is the degenerate mesh; execution on
it is byte-identical to today's non-mesh path. This is a mandatory
acceptance criterion for any implementation.

### 4.4 Subgroup derivation

Subgroup families are derived deterministically by axis name,
identically on every rank:

- `guidance_groups`: vary `guidance`, fix all other coordinates;
- `ulysses_groups`: vary `sp_ulysses`, fix others;
- `ring_groups`: vary `sp_ring`, fix others;
- `sequence_groups`: the full `sp_ulysses x sp_ring` block per
  (guidance, tp) coordinate;
- `model_parallel_groups`: the full `tp x sp_ulysses x sp_ring`
  block per guidance coordinate - every rank participating in one
  guidance lane's model forward and its collectives. This is the
  decision group for distributed control flow that skips
  collectives (chapter 7.1).

Each subgroup has a deterministic rank tuple and identifier. Mesh
identity facts (axis names and degrees, layout version, subgroup
digest) enter manifest consensus (chapter 6) before any collective.

### 4.5 Logical order vs transport groups

Transport-layer process-group membership is an unordered set:
PyTorch normalizes member order (probe-evidenced on
torch 2.13.0+cu130; #185 ring probe disposition). Logical rank order
is therefore mesh-owned. Any collective whose result depends on
participant order (ring traversal, ordered gathers) applies the
mesh-declared permutation explicitly around the collective; no code
may rely on transport-group member order.

### 4.6 Per-generation placement

Placement (which mesh, which ranks, which devices) is chosen per
generation. There is no process-wide startup mode: the same worker
pool serves replicated, guidance-parallel, and sequence-parallel
generations back to back. Feasibility predicates (chapter 5) are
torch-free and queryable before placement, so the planner selects or
falls back before any device work. The mesh and partition plan are
invocation facts, bound into identity per chapter 6.

Placement is topology-aware: rank-to-device binding and axis
assignment consume measured interconnect facts (link bandwidth and
latency, PCIe/NVLink hierarchy) as declared planner inputs. The
contracts never hard-code an axis-to-locality assumption (for
example Ulysses-local/Ring-remote): which axis maps to which links
is a placement decision, and the chosen binding is identity-visible
per chapter 6.2. The mesh itself stays logical (flat ranks exactly
`0..N-1`); topology changes only the spawn-time placement map.

## 5. Partition compatibility

### 5.1 Declarations

Every attention provider and every intervention contribution carries
an immutable, versioned partition-compatibility declaration,
checked at plan compilation, before manifest consensus, fail closed
at assembly. Execution-time refusal exists only as a backstop.

Sequence partition modes are a typed, versioned vocabulary
(dataclasses, not strings):

- `Replicated`: full sequence on every rank.
- `UlyssesHeadScatter`: heads scattered over `sp_ulysses` via
  equal-size all-to-all; each rank holds all tokens for its heads.
- `RingSequenceShard`: contiguous equal-chunk sequence shards over
  `sp_ring`; K/V circulate in fixed ring order.
- `UlyssesRingHybrid`: the composition over
  `sp_ulysses x sp_ring`.

The vocabulary is extensible by revision for future
context-parallel variants. Providers additionally declare tensor
expectations: `full_sequence` vs `contiguous_shard(seq_dim,
equal_chunk)`, supported dtypes, and device kinds. Full-sequence-only
providers refuse sequence-parallel placement at assembly.

### 5.2 Feasibility predicates

Feasibility is decided by pure, torch-free predicates over (declared
family geometry, candidate mesh, provider declaration):

- `head_count % ulysses_degree == 0`;
- `sequence_length >= sp_degree`, with at least one valid row per
  shard under equal-chunk padding (no empty shards);
- provider dtype/device-kind support holds;
- ring participation requires an LSE-exposing local kernel
  (online-softmax merge needs per-chunk log-sum-exp).

The placement planner queries these predicates and falls back BEFORE
choosing the mesh.

### 5.3 Shard metadata

Sharded tensors carry first-class layout metadata rather than
implied conventions: global sequence span, valid-row count, padding
plan reference, original (pre-redistribution) token order, head
layout and KV-head ownership after any Ulysses scatter, the RoPE
position offset for the shard, and the mesh generation that
produced the layout. Compatibility questions (can this contribution
apply to this shard? does this site see global or local
coordinates?) are decidable at plan time from metadata alone.

Shard metadata is mechanics, not semantics: it joins the semantic
`ModelTokenLayout` (chapter 2.3) by global row index and never
redefines what a row means. The two remain distinct types.

### 5.4 Sequence and padding plan

The sequence partition plan is derived from declared packing
geometry (chapter 2.3), never from measured tensors:

- contiguous equal-chunk shards; padding appended only at the
  global tail; explicit `padded_rows` accounting;
- the planner refuses layouts that leave any shard empty;
- padded rows are excluded from attention K/V participation, from
  per-token mask and timestep/modulation-row semantics, and from
  the canonical output gather;
- RoPE and other position-dependent transforms are applied before
  redistribution using globally sliced positions from the plan.

### 5.5 Equivalence contract

Full-sequence attention is the semantic reference. For token-local
operations, partition-then-apply equals apply-then-partition
exactly. For attention, sequence-parallel execution is
mathematically exact dense attention with a deterministic,
documented reduction order (fixed-order fp32 LSE merge in ascending
ring-step order). Per-mode equivalence classification (bit-exact vs
numerically exact, with the mechanism named) is measured, not
assumed, and recorded in receipts. Parity is enforced by CPU/gloo
tests first and NCCL/GPU parity gates after, per the validation
discipline in this repo (no tolerance widening; see chapter 6.4).

### 5.6 Exchange backend seams

Sequence-parallel attention kernels never call the transport layer
directly; they call an injected exchange backend behind a stable
interface. The interface contract:

- submissions are non-atomic: the API accepts per-chunk Q/K/V
  submissions ordered by events, even when an implementation
  submits one chunk, so overlap schedules (exchange V while
  computing QK) substitute the backend without changing attention
  math or the kernel-facing contract;
- the backend exposes completion events so weight-paging prefetch
  (Aimdo) can order against attention exchange instead of
  contending with it;
- backend identity (and for any private/vendored adapter its
  version, signature, and ATen schema pins) binds into invocation
  identity (chapter 6.2).

## 6. Distributed execution manifest and receipts

### 6.1 One canonical manifest

Each distributed invocation has exactly one canonical manifest.
Subsystems contribute compiled-plan slots (the USP slot, the
intervention-plan slot, the cache-policy slot, the
windowed-evaluation slot of chapter 9); no subsystem mints a
parallel identity scheme. Identity construction follows the existing
fact-based conventions in
`packages/dinkster-inference/src/dinkster_inference/identity.py`
(`build_runtime_identity_from_facts`, sha256 digests over canonical
serialized facts).

Ordering is strict:

1. plan compilation (interventions, partition plan, cache policy,
   windowed-evaluation plan) - all fail-closed checks from
   chapters 1, 5, 7, 9, and 10 run here;
2. manifest consensus and authentication - every rank derives the
   manifest independently and the group proves digest equality
   (cryptographic consensus over the sha256 manifest digest);
3. collectives - no collective is issued before consensus succeeds.

A rank that derives a different manifest digest refuses; there is no
majority override.

### 6.2 Identity contents

Invocation identity binds, at minimum:

- runtime and reconstruction identity per the existing conventions
  in identity.py: model/artifact identities (content-addressed),
  provider generation, and the identity of every extension whose
  code can affect the invocation;
- sampler/solver and scheduler identity plus the exact executed
  sigma table (not just the schedule name and parameters);
- seed and noise derivation: RNG stream identities and derivation
  facts for initial, step, and substep noise, including any
  `NoiseSourceOverride` stream state;
- conditioning and lane assignment: conditioning content digests
  and the lane-assignment mapping (which payload feeds which
  guidance lane);
- mesh identity facts: axis names/degrees (full normalized form,
  degree-1 axes included), layout version, subgroup digest;
- physical placement: the rank-to-device binding map alongside the
  logical mesh, so topology-aware placement changes are
  identity-visible;
- the partition plan: global sequence length, chunk size, per-rank
  shard spans, `padded_rows`, packing-geometry digest, RoPE slicing
  plan;
- collective schedule: ring traversal order (ascending ring-step
  peer schedule per rank), per-chunk accumulation order (sequential
  in ring-step order), merge algorithm and version (for USP:
  `ring-lse-fp32.v1`), accumulation dtype (fp32);
- provider evidence: routed attention backend identity, exchange
  backend identity, and for any private/vendored adapter its
  version, signature, and ATen schema pins;
- the intervention `plan_digest`, including the declared and
  realized control-contribution facts of chapter 10,
  mask/token-grid geometry digests, and the cache policy digest
  (chapter 7);
- the expected rank-local compiled-plan digests: every rank's
  expected plan digests are part of the shared manifest, so
  consensus proves each rank compiled the same plans, not merely
  that each rank has some plan.

Ranks that agree on this manifest cannot silently execute different
weights, schedules, noise streams, conditioning, or lane payloads;
any such difference changes the digest and refuses at consensus.

### 6.3 Two proof lineages and receipt lineage (user directive, 2026-08-18)

Correctness of distributed execution rests on two separate,
separately receipted proof obligations:

- **Family math identity (single-GPU).** Each family's numerical
  behavior is proven by its own single-GPU parity goldens and gates.
  and prefixes runtime identity strings; rotating it invalidates
  caches and manifests, never proofs. A model-code change - kernels,
  solvers, schedules, assembly, conditioning, identity composition -
  bumps the ledger and re-runs the family's own parity gates. It
  never requires a multi-GPU run.
- **Distributed mechanics.** That multi-GPU guidance execution
  equals single-GPU execution is a property of the distribution
  machinery, not of any model's math: both sides of the equality run
  the same model function, so a family math change preserves it by
  construction. What can break it is partitioning, exchange,
  collective scheduling, admission, or a family's integration
  surface - and only changes to those re-prove it, through the
  dual-GPU mint pipeline.

Distributed receipts therefore do NOT bind runtime identity strings
changelog docstring) counts float-visible or admission-visible
changes to the shared distribution machinery, and the digest is
sha256 over canonical facts: the family's declared
distributed-integration facts (shard recipe identities, rank-fencing
and reconstruction behavior, the exchange/plan vocabulary the family
executes), compute dtype, world size, and device capability.

This decoupling applies only to the standing mechanics receipts.
Per-invocation manifests and execution receipts (6.2, chapters 8-10)
continue to bind full runtime identity strings and plan digests.

Re-mint triggers:

- shared machinery change (partitioning, exchange, collective
  schedule, distributed admission or distributed guidance paths):
- family integration-surface change: that family's integration facts
  change; only its receipts re-mint;
- anything else - family kernels, solver loops, schedules, assembly,
  identity composition: no re-mint and no multi-GPU run.

Each minted receipt records as provenance (never as admission key)
the exact runtime identity strings, hardware, driver, and torch
versions of its proof runs. Admission remains fail-closed exactly as
before the split: a registry keyed by family, receipt identity, and
live environment; builtin-behavior, extension, callback, and
cancellation refusals are unchanged.

A family's integration facts also bind any float-visible reference shape. SD1.5
guidance binds one batch-1 lane per rank and is proved against the same
single-device split shape; changing ordinary single-device lane fusion alone does
not rotate that receipt.

New float-visible distributed routes take NEW receipts; they never
claim existing single-device receipt lineages. In particular, USP
execution records its own receipt family with an instrumented
single-GPU equivalence bound; it does not inherit H3's existing
receipts.

### 6.4 Tolerances and identity

Existing parity tolerances are never widened to admit distributed
execution (see the numerical parity discipline in AGENTS.md).
A new distributed route gets a new receipt when routing or refusal changes
float-visible behavior. Adding mesh or manifest routes alongside untouched
single-device paths does not change single-device receipts. Distributed
receipt-schema changes re-mint every distributed receipt through the dual-GPU
pipeline. The target for all #174/#185 implementations is that single-device
behavior does not move.

## 7. Distributed cache contract

Incorporates #189 and its survey sharpenings; the cache contract is
a consumer of chapters 3-6, not a parallel mechanism.

Sampling caches are approximate, float-visible acceleration and are
explicit opt-ins: no cache policy is ever part of the default exact
path, and an implementation must not enable one by default. The
opted-in policy and its parameters bind into the manifest and cache
keys (chapter 6.2, 7.2), so which policy ran - including none - is
an identity fact.

### 7.1 Decision semantics

- Cache hit/skip decisions are group-consistent across all ranks of
  the relevant subgroup. Timeline-predicate decisions (chapter 3)
  are group-consistent by construction. Input-dependent decisions
  (residual drift, similarity thresholds) are distributed control
  flow: the deciding quantity is computed via an explicit global
  reduction over the model-parallel group (chapter 4.4) - every
  rank whose later collective the decision may skip, across both SP
  and TP axes - then the identical scalar drives the identical
  branch on every rank. A rank never branches on shard-local
  values, and no rank outside the reduction may hold a collective
  the decision skips.
- Decisions and state are guidance-lane-local: CFG lanes cache
  independently; a hit on the conditional lane implies nothing
  about the unconditional lane. Agreement is required within each
  lane's model-parallel group, never artificially across
  independent lanes.
- When guidance lanes are fused into one batch, divergent per-lane
  hits are resolved with masks and a fixed collective order: every
  rank issues the same collective sequence regardless of which
  lanes hit, so divergent decisions cannot deadlock.
- Cache reset, warmup entry/exit, and cancellation follow the same
  contract: each is a group-consistent decision with a fixed,
  declared decision and collective order, never a rank-local
  fast path.

### 7.2 State layout and keys

- Cached state is stored in the native shard layout of the
  execution that produced it (SP shard, TP shard, or replicated).
  There is no gather-to-cache; a cache entry is only valid for the
  partition plan that wrote it.
- Cache keys include: model identity, conditioning identity,
  scheduler/sigma-schedule identity, partition/layout digest, and
  cache policy digest. Any change to the partition plan or policy
  invalidates by key mismatch, not by flush logic.
- Cached shards stay on their owning rank and are accounted against
  the same memory planner as attention buffers and the weight
  pager's disposable device cache; they are never invisible memory.
  CPU spill is permitted only where measured transfer time beats
  recomputation, and the spill decision is part of the declared
  cache policy, not an ad-hoc runtime choice.

### 7.3 Hooks and placement

Stable hook boundaries are the declared cache sites of chapter 2.2:
pre-block (indicator inputs), first-block residual, final residual,
and local sequence output. Cache implementations never assume full
replicated activations. Decision placement is paging-aware: under
offloading (Aimdo), the decision point must be reachable without
faulting in weights the skip would have avoided (probe-block
policies page only the probe first and prefetch the tail on a
miss); a cache design whose decision requires the full forward is
rejected at plan compilation.

### 7.4 Refusal

Cache modes compose with distributed execution only where the
combination has recorded parity evidence. An unproven combination
(for example an input-dependent skip policy under a new partition
mode) refuses at plan compilation, fail closed, rather than running
unvalidated.

## 8. Validation chapters (portability proofs)

Each case below must be expressible with chapters 1-7 as specified,
without registry monkey patching, untyped event buses, or family
sampler branches. These are paper proofs at spec stage; each becomes
an executable acceptance case when its implementation lands.

### 8.1 RES4LYF custom samplers and sigma manipulation

Each seam RES4LYF hacks into ComfyUI has a named carrier here:

- sampler/scheduler registration: the existing descriptor/registry
  conventions (no registry mutation);
- sentinel zero-sigma semantics (no-initial-noise, unsampling,
  resampling): `SigmaScheduleOverride` at `run.sigma_schedule`, with
  explicit direction, initial-noise policy, and resume-from fields;
- full RK solver ownership (explicit/implicit stages, Newton
  iterations, runtime sampler swapping): `SolverOverride` at
  `run.solver` selecting a declared solver implementation, with the
  solver-stage sites of chapter 2.2 as the pre/post model-call and
  pre/post numerical-stage boundaries;
- its noise subsystem (fractal/pyramid/Student-t, independent
  step/substep RNG streams): `NoiseSourceOverride` at
  `run.noise.initial`/`run.noise.step`/`run.noise.substep`, with
  declared RNG stream identity and serializable state;
- channelwise CFG via repurposed negative scales and post-CFG
  output transforms: `GuidancePolicyOverride` at
  `run.guidance_policy`;
- the `transformer_options` mutable event bus (regional data, guide
  latents, QKV controls): the typed per-call context of
  chapter 1.4, with guide-latent content carried as
  `AuxiliaryResourceRef` at `run.resources`;
- callables in conditioning metadata for deferred mask
  materialization: `ConditioningTransform` at `run.conditioning`
  with compile-time geometry resolution;
- module `__class__` replacement for regional attention and style
  injection: `AttentionTerm` and `WeightOverlay` contributions on
  the block-scoped `attention.<selector>` and `weights.<component>`
  sites;
- solver/guider/RNG continuation state persisted in latent dicts:
  the typed continuation state of chapter 1.3.

The engine remains the only owner of the executed schedule;
compilation validates overrides against the run's sigma space, and
the timeline anchors reflect the executed result. A RES4LYF
capability that maps to none of these carriers is a named
compile-time refusal, not a silent degradation.

### 8.2 Per-token AV latent noise strengths (H3, ComfyUI PR #15375)

Video/audio mask strengths are `NoiseStrengthField` payloads
compiled from the source masks by the H3-declared transforms
(chapter 2.2). The single transform per modality guarantees model
timestep/modulation-row selection and sampler-side preserved-content
correction consume the same compiled token strengths, while the
final output blend keeps the source mask's full resolution. Under
sequence parallelism, the field is partitioned by the plan's shard
spans; padded rows receive no strengths. Mask and geometry digests
bind into the manifest, receipts, and cache keys.

### 8.3 Motion guides and chained generation (H3, #186)

Guide context enters through `motion_guide.context` sites with
timeline-anchored activation and media placement coordinates
(chapter 3.4): a guide's frame anchors (positive or end-relative)
compile to global token rows/spans through the declared token
layout, with padded rows excluded. Chained-generation state lives
in run-scoped lane-keyed storage (chapter 1.3). `GuidedDenoiser`
and the shared lane evaluator retain sole ownership of lane fan-out
and combination; guides contribute payloads, not control flow.

### 8.4 EasyCache-class acceleration (#189)

An EasyCache-style policy is a cache policy value: timeline warmup
window (chapter 3), input-dependent skip via one reduction over the
model-parallel group (chapter 7.1) covering every SP and TP rank
whose collectives the skip removes, state at the declared residual
sites, lane-local and shard-native. The same policy value works
replicated, under SP, and under TP because the decision group,
group consistency, and layout are contract properties, not policy
code.

### 8.5 Exact USP execution (#185)

The USP runtime consumes the mesh (chapter 4), partition
declarations and padding plan (chapter 5), the exchange backend
seams (chapter 5.6), and the manifest slot (chapter 6) exactly as
specified. Its `SequenceParallelAttentionKernel` implements the
existing rank-4 BHSD `AttentionKernel` contract
(dinkster_inference_torch/attention.py), parameterized by mesh,
partition plan, exchange backend, and an inner route-selected
kernel. U1R1 collapses to the existing local provider route: the
provider-route identity is the existing one, while invocation
identity still binds the full degenerate mesh in normalized form
per chapter 4.3 - route identity and invocation identity are
distinct facts. This chapter is satisfied when implementations of
#185 validate against these contracts with no #185-private identity
or timeline mechanism.

## 9. Windowed evaluation plans

Windowed evaluation covers the technique family that slices model
inputs along declared dimensions, evaluates the model per slice, and
re-merges the outputs (#198). Two distinct compositions exist and
only one needs an engine seam:

- Graph-level composition: each slice, or batch of slices, is a
  complete sampling invocation (crop, encode, full sampling pass,
  decode, composite). Ultimate SD Upscale works this way
  (chapter 9.7). The contracts of chapters 1-8 already carry it;
  nothing in this chapter applies.
- Within-step windowed evaluation: one logical denoiser evaluation
  becomes W model calls on windows of the declared geometry, merged
  before sampling continues. ComfyUI core context windows
  (`comfy/context_windows.py`) and MultiDiffusion-class spatial
  tiling work this way. This is the seam this chapter defines.

A windowed evaluation plan is a compiled plan value with its own
manifest slot (chapter 6.1), like the cache policy. It is not an
intervention effect kind: it does not contribute payloads at sites;
it multiplies and merges model calls. Windowed evaluation is an
explicit opt-in and never part of the default single-call path.

A windowed evaluation plan is compiled from an ordered collection of
window-plan layers, so independently contributed window sets
compose: one contributor can declare spatial tiling while another
declares temporal context windows. Layers resolve at plan
compilation into one canonical composite plan (chapter 9.2); the
manifest slot always carries the composite (chapter 9.5), and a
single-layer plan is the degenerate composite. "The plan" in this
chapter always means the composite.

### 9.1 Declared media axes

Window slicing is declared in media terms, never as raw tensor dim
indices. The family geometry vocabulary (chapter 2.3) is extended
with declared media axes:

- every latent kind and every conditioning kind that can be sliced
  carries a declared axis map: which of its dimensions realize
  which media axis (temporal, height, width, or another declared
  role), together with a versioned index-map profile from primary
  axis indices to that kind's indices (the kind declares its own
  sampling relation, not the consumer). The initial profile
  vocabulary carries the two known relations, extensible by
  revision: an integer affine profile (integer scale and offset)
  and a rational proportional-range profile - primary index `i`
  maps to the half-open range
  `[round(i * kind_extent / primary_extent),
  round((i + 1) * kind_extent / primary_extent))` with declared
  rounding, extent-clamping, forced-nonempty, and deduplication
  semantics, in that order - the range is clamped to
  `[0, kind_extent)` bounds before being forced nonempty, so a
  forced range can never name an out-of-range index - covering
  non-integral ratios and many-to-one mappings. The profile
  identity and parameters bind into the window plan digest
  (chapter 9.5). A kind whose sampling relation fits no declared
  profile is a named compile-time refusal;
- each declared axis states its extent source (from the declared
  packing geometry) and whether the family declares it wrappable
  (chapter 9.2);
- a kind with no declared mapping for a sliced axis is not sliced
  silently: the plan compiler either replicates it per window
  (when its declaration says it is window-invariant) or refuses at
  compile time (when it is neither mapped nor declared invariant).

This is the "right slices on the right dimensions" requirement made
typed: slice correctness is decided at plan compilation from
declarations, not discovered at runtime from tensor shapes.

### 9.2 Window sets

The plan declares window multiplicity for one denoiser evaluation
as a deterministic pure function of (declared family geometry,
timeline anchors of chapter 3). Because anchors are engine-owned
facts identical on every rank, window sets are group-consistent by
construction and may vary per step.

A window is an ordered index list over one or more declared axes:

- contiguous spans are the common case; non-contiguous ordered
  index lists are first-class (looped temporal schedules produce
  them);
- wrap-around windows are first-class: an index list may wrap
  modulo the declared axis extent, on any declared media axis,
  spatial or temporal. Wrapping is valid only on axes the family
  declares wrappable; a wrapped window on a non-wrappable axis is
  a compile-time refusal. Wrapped spatial windows are how tileable
  (seamless, edge-rolling) generation is expressed;
- an index list may name the same axis index more than once: a
  modular strided walk whose cycle is shorter than the requested
  list length legally revisits an index, and core looped schedules
  produce such lists. Each list entry is therefore a distinct
  ordered occurrence: a window contributes once per occurrence,
  each occurrence carries the weight declared at its local
  position, and occurrences accumulate in ascending local position
  within the window's position in the declared traversal order,
  while totality (chapter 9.3) sums those occurrence weights per
  semantic output coordinate;
- structural rows a window carries beyond its semantic payload
  (causal anchor rows, injected guide rows) are explicitly
  accounted per window; their model outputs are dropped - excluded
  from the merge, from real-row digests, and from the canonical
  output gather - the same discipline as `padded_rows`
  (chapter 2.3). Dropping a structural row's model output is
  distinct from restoring its declared content: when carried state
  must retain guide rows across steps (chained generation), the
  plan declares a passthrough restoration that re-appends the
  declared guide payload - never the model's output for those
  rows - so post-merge shape and state reconstruction are defined
  facts of the plan, not implementation choices.

Layers compose window sets by Cartesian product:

- each layer declares its own window set over a declared set of
  media axes, derived by the same deterministic pure-function rule
  as above;
- canonical layer order is a deterministic function of layer
  identity (ascending layer digest, chapter 9.5), never graph or
  contribution arrival order, so every rank derives the same
  composite;
- layers must claim disjoint axis sets. Two layers slicing the same
  declared axis is a named compile-time refusal (overlapping axis
  claim); there is no precedence or last-writer-wins rule.
  Replicating a window-invariant kind (chapter 9.1) is not an axis
  claim;
- the composite window set is the Cartesian product of the layers'
  window sets: one joint window per product element, carrying each
  component window's index list on that component's axes. Each
  joint window is one model call. The composite window schedule
  enumerates joint windows lexicographically in canonical layer
  order, and that enumeration is the declared traversal order of
  chapter 9.3;
- layer-local structural-row and passthrough declarations lift into
  the composite: a joint window's structural rows are the union of
  each component window's declared structural rows, projected
  through the other layers' index lists (a temporal guide row
  appears in every joint window whose temporal component carries
  it, on every spatial tile of that row). Lifted structural model
  outputs are dropped per joint window exactly as in the
  single-layer rule above, and each declared passthrough
  restoration applies exactly once, after the composite's single
  merge (chapter 9.3), in the full output geometry. Two layers
  whose lifted passthrough restorations intersect on any output
  coordinate are a named compile-time refusal (conflicting
  structural claim);
- composition is flat: no layer's outputs are merged before another
  layer's model calls, and no nested or staged evaluation order
  exists. A composite plan performs its model calls on joint
  windows followed by exactly one merge (chapter 9.3).

### 9.3 Deterministic merge

The merge from per-window outputs back to the full geometry is a
declared, versioned algorithm bound into invocation identity, the
same pattern as the collective schedule (`ring-lse-fp32.v1`,
chapter 6.2). The declaration carries:

- the weight profile per window position (versioned vocabulary:
  flat, pyramid, overlap-linear, and revisions), realized as
  accumulate-then-normalize over window contributions;
- order-dependent merges (running-average style) are permitted
  only with an explicitly declared traversal order; otherwise
  traversal order is still fixed and declared so distributed and
  single-device execution accumulate identically;
- the accumulation dtype;
- wrap-seam semantics: a wrapped window's contribution accumulates
  into both edge regions of the axis under the same weight and
  normalization rule as interior overlaps, so the seam is
  indistinguishable from the interior by construction. Merge
  algorithms that cannot honor this under wrapping refuse at
  compile time when paired with wrapped windows.

The merge must be total over the semantic geometry, checked at plan
compilation: because window sets and weight profiles are declared
and enumerable at plan time (chapter 9.2), the compiler proves that
every semantic output coordinate receives a strictly positive,
finite aggregate weight, summed over every occurrence of every
window (chapter 9.2) - no uncovered coordinate, no zero or
non-finite denominator at normalization. Order-dependent merges
must likewise write every semantic coordinate at least once. A
window set or weight profile that leaves any coordinate uncovered
or zero-weighted is a compile-time refusal, never a runtime
divide-by-zero or undefined output row.

Under layer composition the composite performs exactly one merge,
so the composite-global merge facts are agreement-checked at plan
compilation: every layer must declare the same merge algorithm
identity and version and the same accumulation dtype, and a
multi-layer composite whose layers disagree is a named compile-time
refusal (conflicting merge declaration). Weight profiles remain
per-layer (a flat spatial layer composes with a pyramid temporal
layer). Wrap-seam capability is checked against the resulting
composite merge and the union of the layers' wrapped axes. A joint
window's occurrences are the Cartesian product of its component
windows' occurrences (chapter 9.2), and a joint occurrence's weight
is the product of its component occurrences' declared weights, each
taken at that component's local position; the factors are evaluated
and multiplied sequentially in canonical layer order in the
declared accumulation dtype, so every rank computes identical
weights. Within a joint window, joint occurrences accumulate in
lexicographic component local-position order in canonical layer
order. The composite merges in exactly one
accumulate-then-normalize pass over all joint occurrences; layers
never normalize separately. The totality proof above runs against
the composite window set with these per-occurrence product weights.
Order-dependent merge profiles are valid only in single-layer
plans: a multi-layer composite that declares one is a compile-time
refusal.

### 9.4 Composition with the mesh and partition plans

Windowed evaluation composes with sequence parallelism by nesting,
never interleaving:

- each windowed model call is an inner model call with its own
  `ModelTokenLayout` (chapter 2.3), derived from the window's
  sliced geometry, and its own partition plan (chapter 5). Inner
  model calls are not separate distributed invocations: the outer
  invocation remains the unit of manifest and consensus
  (chapter 9.5);
- the partition planner shards the window's sequence. Because the
  mesh is fixed per generation while window lengths vary within a
  step, mesh selection for a windowed plan runs the chapter 5.2
  feasibility predicates over every window the plan can produce -
  for a layered plan, every joint window of the composite
  (chapter 9.2). Window sets are pure functions of declared
  geometry and the timeline anchors, so all windows are enumerable
  at plan time - equivalently, the predicates hold for the minimum
  window length. Any infeasible window fails or falls back at mesh
  selection, never at execution, preserving chapter 5's ordering
  (planner-side feasibility primary, execution refusal backstop);
- the window plan owns media-axis multiplicity; the partition plan
  owns intra-call sharding. Neither reaches into the other: window
  index lists never appear in shard metadata, and shard spans never
  appear in window declarations.

Interventions compile against each inner model call's layout:
media placement coordinates (chapter 3.4) resolve through the
window's `ModelTokenLayout`, so a contribution placed on media
frames lands only in windows that contain those frames.

### 9.5 Identity and cache composition

A windowed invocation has exactly one canonical manifest and one
consensus - the outer invocation's (chapter 6.1). Inner model
calls never run their own consensus; because window sets, per-kind
index maps, per-window layouts, and per-window partition plans are
all plan-time facts, the outer consensus proves them all before
any collective for any window is issued.

The windowed-evaluation manifest slot binds, at minimum: the
ordered per-layer plan digests and the composite plan digest (each
layer digest is a sha256 over that layer's canonical serialized
facts, per the identity.py conventions of chapter 6.1; a
single-layer plan binds its one layer digest plus the composite).
The composite plan digest's canonical preimage begins with a
versioned composite-domain tag and includes the layer count and the
ordered layer-digest vector ahead of the compiled composite facts,
so two plans that compile to the same joint schedule and weights
from different layerings never share a composite digest - and
therefore never share the cache identity below. The slot further
binds the window-set derivation identity and parameters, per-axis
wrap topology (which axes wrap), the full window schedule (or its
deterministic derivation facts), the per-kind index-map profile
identities and parameters (chapter 9.1), the ordered per-window
`ModelTokenLayout` digests and partition-plan digests,
structural-row accounting including any declared passthrough
restoration (chapter 9.2), and the merge declaration of chapter
9.3 (algorithm, version, weights, traversal order, accumulation
dtype). A wrapped run and a non-wrapped run never share invocation
identity, and a layered composite and any of its layers compiled
alone never share invocation identity.

Cache identity (chapter 7.2) extends under windowed evaluation:
cache keys additionally include window identity - the joint window
index-list digest plus the composite plan digest. A cache entry is
valid only for the window that wrote it. Cache decisions are per
(window, lane), reduced over the inner call's model-parallel group
per chapter 7.1; a decision for one window implies nothing about
another window at the same step.

### 9.6 Semantics already carried elsewhere

Windowed evaluation does not duplicate existing carriers:

- FreeNoise-style noise rescheduling is `NoiseSourceOverride`
  (chapter 2.2), not window-plan behavior;
- guide-frame content and placement are `ContextInjection` with
  media placement coordinates (chapter 3.4); the window plan only
  accounts injected rows as structural (chapter 9.2);
- activation windows over the denoising timeline remain chapter 3
  predicates; a window plan whose window set varies per step
  consumes the same anchors.

### 9.7 Validation: the named cases

Core context windows (temporal). ComfyUI core's index-list context
windows map onto this chapter directly: windows are ordered index
lists on the declared temporal axis (chapter 9.2), including looped
schedules as wrapped lists whose strided modular walks may revisit
an index, carried as repeated occurrences with per-occurrence
weights and accumulation (chapter 9.2) - the intended scatter
semantics, not a description of current core: core's
accumulate-then-normalize fuse profiles use a non-accumulating
advanced update whose duplicate-index behavior is undefined (the
tested CPU path drops all but one occurrence; only the
running-average branch iterates occurrences), an upstream defect
recorded in
`docs/comfyui-issues/context-window-duplicate-index-fuse-drops-occurrences.md`;
per-cond temporal slicing - including secondary modalities derived
by proportional ranges over a different extent - is the declared
axis map of chapter 9.1 under
the rational proportional-range profile; injected guide frames and
causal anchor rows are structural rows whose model outputs are
dropped from the merge, with the re-appended guide latents carried
as the declared passthrough restoration of chapter 9.2; the
flat/pyramid/overlap-linear fuse profiles are chapter 9.3 weight
profiles subject to the totality check, and the relative
running-average fuse is an order-dependent merge with declared
traversal order that must write every semantic coordinate. A
context-window capability that maps to none of these carriers is a
named compile-time refusal.

Spatial tiling and tileable generation (MultiDiffusion-class).
Within-step spatial tiling is the same mechanism on declared
spatial axes: window index lists over height/width, weighted merge
across overlaps. Wrap-around spatial windows (index lists modulo
the axis extent on family-declared wrappable axes) express
tileable/seamless generation; chapter 9.3's wrap-seam rule makes
the seam follow the interior merge semantics. This confirms the
relation between tiling and context windows: one declared
mechanism - ordered, possibly non-contiguous, possibly modular
index-list windows over declared media axes with deterministic
weighted merge - covers both, differing only in the declared axes.

Stacked spatial and temporal windowing. Independently contributed
layers compose: one contribution declares spatial tiling
(height/width axes) while another declares temporal context windows
(temporal axis). The axis sets are disjoint, so the layers compose
by Cartesian product (chapter 9.2) into joint spatio-temporal
windows - each model call evaluates one spatial tile of one
temporal window - and one product-weighted merge (chapter 9.3)
reconstructs the full geometry. Two contributions claiming the same
axis (two spatial tiling layers) are the overlapping-axis-claim
refusal, decided at plan compilation.

Ultimate SD Upscale (graph-level composition). Each spatial tile -
or batch of tile coordinates, when tiles are batched into one
pass - is cropped, encoded, run through a complete sampling pass,
decoded, and composited; seam-fix modes are additional independent
masked passes. Each such pass is a graph-level complete sampling
invocation with its own plan and manifest under chapters 1-8 as
they stand; per-tile conditioning and ControlNet cropping is
node-layer coordinate transformation on typed conditioning. No
engine seam is needed, and this chapter's plan value deliberately
does not model it.

## 10. Shared control-contribution gain, masks, and application

This chapter defines the common application contract for controls,
adapters, and other gain-bearing interventions (#219). It adds no
control family and no callback seam. `WeightOverlay`, `AttentionTerm`,
`ContextInjection`, and `ResidualTransform` use these values when
they expose a gain-bearing application at a declared site.
`ConditioningTransform`, `AuxiliaryResourceRef`, and run overrides
refuse gain and application-operator fields: they remain source
preparation, resource bindings, or policies. Strength-like provider
parameters on those inputs never become the shared contribution gain.

### 10.1 Declared gain schedule

A `ContributionGain` is immutable declared data with these factors:

- one timeline schedule, expressed as a constant, a canonical
  keyframe curve, or a table over the exact executed timeline;
- one finite `global_gain`;
- finite gains keyed by each resolved concrete site, defaulting to
  1 only when the declaration explicitly selects that default;
- finite gains keyed by the existing guidance-lane identities,
  likewise with an explicit default;
- an ordered tuple of effect masks (chapter 10.4), empty for the
  all-one field.

Site selectors must resolve before the site-gain map is normalized.
An entry that resolves to no declared site, or a lane key outside the
engine guidance vocabulary, is a compile-time refusal. No family may
interpret an unknown key as a no-op.

A canonical keyframe declaration carries a stable keyframe ID, one
anchor in exactly one chapter 3 coordinate (`progress`, `sigma`, or
`step_index`), the timeline gain at that anchor, optional site-gain
and lane-gain replacements, an optional replacement effect-mask
tuple, and a positive `minimum_realized_steps` that defaults to 1.
Values above one request additional contiguous hold rows. The
schedule lists keyframes in execution order (sigma anchors therefore
normally descend) and declares, from a closed versioned vocabulary:

- interpolation between anchors (initial profiles: `hold.v1`,
  `linear.v1`, and `smoothstep.v1`);
- endpoint behavior before the first and after the last anchor
  (initial profiles: `clamp.v1` and `zero.v1`);
- for each omitted site, lane, or mask value, either `inherit` the
  preceding normalized keyframe state or `reset` to the schedule's
  declared default;
- the keyframe-row assignment profile and whether requested minimum
  rows are `best_effort` or `hard`.

Interpolation applies only to scalar gain values. Effect-mask tuples
are piecewise-held according to anchor segments and the declared
inherit/reset state; an assigned hold row forces the complete
normalized keyframe state. Providers cannot interpolate mask tensors
with a private rule. Duplicate or unordered anchors, mixed
coordinates, unknown profiles, non-finite values, and implicit
omission behavior refuse during compilation.

For a resolved site and guidance lane at one executed step, the
engine computes:

`effective_gain = timeline_gain * global_gain * site_gain * lane_gain * effect_mask`

The first four factors form the scalar gain. `effect_mask` is the
compiled token or region value in [0, 1]. This multiplication order
is canonical and float-visible. Providers receive the resolved
effective gain and cannot reorder factors, omit factors, or perform
a second hidden strength calculation. Lane gain broadcasts across
the applying site's rows within its guidance lane and remains
lane-local under chapter 7; independent lanes never share or reduce
this factor.

### 10.2 Exact realization and keyframe holds

Gain realization runs after `run.sigma_schedule` has fixed the exact
executed sigma table and before manifest consensus or any model
collective. Every contribution receives a `RealizedGainTable` with
one row per executed denoising step. Each row carries the exact
chapter 3 anchors, the analytic segment/profile, any keyframe ID
assigned to that row, the resolved timeline/global/site/lane scalar
values, and the ordered compiled effect-mask digests. A directly
declared table must match the exact executed rows; it cannot be
silently clipped, padded, or resampled.

An assigned hold row uses the normalized values at that keyframe,
so a requested keyframe is actually executed rather than merely
crossed by interpolation. The initial assignment profile is
`monotone-nearest.v1`:

1. Treat each requested minimum row as an ordered occurrence of its
   keyframe. Assign each keyframe at most one contiguous run of
   distinct executed rows, with runs preserving keyframe and
   occurrence order.
2. Choose an assignment by, in order: maximizing the number of
   keyframes receiving at least one row, maximizing the number of
   requested occurrences assigned, minimizing total absolute
   distance in the schedule's declared coordinate, and choosing the
   lexicographically earliest `(keyframe declaration index,
   step_index)` assignment on a remaining tie.
3. Record every keyframe's requested and realized row count in the
   table, including zero.

The profile's named state-monotonicity invariant is: if keyframe A
precedes keyframe B in declared execution order, every realized row
assigned to A precedes every realized row assigned to B. The
contiguous-run rule and the distance and tie objectives above are
part of the versioned profile semantics, not implementation choices.

When requested rows exceed available rows, `best_effort` uses that
deterministic partial assignment. This is the default compatibility
behavior: the realized table makes every unassigned keyframe
inspectable and runtime code does not reinterpret it. `hard` refuses
with `unsatisfied_keyframe_minimum` if any keyframe receives fewer
than its requested minimum. A hard at-least-once request therefore
refuses an over-subscribed run instead of making an impossible
promise.

The compiled plan retains both the declared analytic curve/keyframes
and the exact realized table. Their content digests, together with
the assignment-profile identity and parameters, bind into the
intervention `plan_digest` and canonical manifest slot; receipts
expose both declared and realized artifacts. Execution reads only the
realized row for the current engine-owned `step_index`; there are no
mutable `guarantee_steps` counters and no per-window, per-rank, or
per-model-call resampling.

### 10.3 Provider selectors

An SD control provider with an execution selector binds that selector on the
torch-free `ControlApplication`; it is part of invocation identity rather than
resolved provider state. `None` means that the provider has no selector and
preserves the legacy application identity. For xinsir SDXL ControlNet Union,
the closed semantic vocabulary and provider-owned execution mapping are:

| Semantic token | Execution index |
|---|---:|
| `openpose` | 0 |
| `depth` | 1 |
| `hed`, `pidi`, `scribble`, `ted` | 2 |
| `canny`, `lineart`, `anime_lineart`, `mlsd` | 3 |
| `normal` | 4 |
| `segment` | 5 |
| `tile` | 6 |
| `repaint` | 7 |

Detection establishes an artifact capacity of exactly 6 or 8 as a config
fact. Union without a mode, non-Union control with a mode, and a mapped index
outside the detected capacity refuse during conditioning resolution before
execution. Identity binds the semantic token, not only its execution index,
so distinct semantic modes that share an index remain distinct invocations.

Z-Image Fun control resolves `ControlApplication` through the family-specific
`ZImageControlConditioning` carrier. The 16-channel VAE control latent is the
application hint: its canonical pre-cast, pre-batch content digest must match
the hint reference, while its spatial geometry must match the sample latent.
The model digest is derived from the asset-system BLAKE3 identity and assembly
configuration; direct model construction has no provenance and refuses.

The declared residual-site vocabulary is
`z_image.dit.layer.{00,05,10,15,20,25}.control_residual.v1`. Timeline, site,
and active guidance-lane gains are realized before execution and multiply in
chapter 10.1 order. Z-Image Fun currently admits exactly one application;
application chains are an explicit admission limit until another official
provider establishes composition semantics. Effect masks are represented on
the resolved carrier for contract parity but remain refused until the family
has a `ModelTokenLayout`-backed compiled field. Source masks remain a distinct,
unsupported role.

### 10.4 Source masks and effect masks

`SourceMaskInput` and `EffectMaskInput` are distinct typed inputs:

- a source mask is data consumed by a conditioning transform or an
  auxiliary branch, such as an inpaint/control source;
- an effect mask limits where the resulting contribution is applied
  to the host site.

The same bytes may be bound in both roles only as two explicit typed
bindings. A provider cannot reinterpret a source mask as an effect
mask, or vice versa. Each binding carries its role in the canonical
plan.

An effect mask declares its source geometry, media placement and
axis identity, target `ModelTokenSegment`, source content digest,
and the existing family `TokenGridTransform` that maps that modality
to global model-token rows (chapter 2.3). Resampling, pooling,
rounding, clamping, and optional family-declared quantization are
properties of that transform, not of the control provider. Missing
or competing transforms refuse. Values are finite fractions in
[0, 1] before and after compilation; scalar contribution gains may
have wider domains, but masks never do.

Each mask compiles to a dense `TokenRowTable` in ascending semantic-row
order containing exactly the target segment's rows. The compiled artifact
contains no rows from other segments and no padded or structural rows. The
artifact is digested in full; an execution tensor for wider geometry is
derived from the artifact and layout rather than represented in its identity.
`TokenRowSpan` is only a storage or execution optimization and never changes
field identity. All-zero means no effect. All-one is identical to the
unmasked full effect.

The compiled field digest is SHA-256 over compact, sorted-key ASCII JSON with
domain tag `dinkster.effect-mask-field.v1`. Its facts bind the input digest,
source BLAKE3 digest, semantic target layout digest, transform digest, target
segment, and dense row values. The semantic layout digest covers every
declared segment but excludes `padded_rows`. The input digest binds source
geometry, axis identity, media placement, target segment, and transform
identity. Values are validated as finite fractions in [0, 1] before encoding,
negative zero is normalized to positive zero, and JSON uses shortest
round-trip decimal float spelling with non-finite values disabled. Row indices
are omitted because target segment, layout, and ascending order determine them.

Per-frame, per-latent, spatial, and spatiotemporal weights are effect
masks in their declared media geometry, not additional gain factors.
Multiple effect masks on one contribution are compiled separately
and multiplied element by element in declared tuple order. This is
the only initial composition rule; max, union, provider callbacks,
and implicit source/effect-mask combination are compile-time
refusals.

An effect mask targets the applying site's declared rows: for
example residual/output rows, attention query rows, or injected
context rows. It does not reach backward to mask an auxiliary source,
prepared attention K/V, or conditioning input; those need their own
typed source-mask binding or a separate applying contribution at a
declared site. Global attention can propagate consequences outside
directly masked query rows; this does not change the meaning of the
applied mask.

### 10.5 Windows and sequence partitioning

Timeline gain is realized once against the outer denoising timeline.
It is not evaluated again for a chapter 9 window. Effect masks are
evaluated in full-domain media coordinates and compiled once through
the family `TokenGridTransform` against the full-domain
`ModelTokenLayout`. Each window then gathers that compiled token-row
field through its ordered index lists and the existing per-kind axis
maps of chapter 9.1 into the window's inner `ModelTokenLayout`.
Repeated and wrap-around occurrences remain repeated ordered gathers.
The token transform is never rerun against a window slice; no
contiguous-slice shortcut or window-relative keyframe coordinate
exists.

The resulting token field partitions by the same global row spans as
the inner hidden state, RoPE, and other `TokenRowTable` values under
chapter 5. Token-local application is partition-then-apply exact.
Attention-term providers consume the full-sequence, contiguous-shard,
or head-local view already declared by their chapter 5 compatibility
value. An auxiliary input remains window-invariant, is gathered by a
declared media-axis map, or is recomputed per window exactly as
chapter 9.1 requires; no control-specific slicing mode is added.

Activation and scalar gain come from common outer anchors, so all
ranks take the same active/inactive branch. A contribution that
cannot preserve its application semantics for any selected inner
window layout or partition mode refuses during plan assembly before
collectives.

### 10.6 Typed application operators

An `ApplicationOperator` is a closed, versioned value selected by
the contribution and accepted by the target site's declaration.
Each operator declaration includes its finite `ScalarGainDomain`;
compilation validates every scalar factor as finite and every
realized scalar product against the domain. A site may narrow the
operator's domain but cannot widen it or change its formula.

The initial operators are:

- `Additive.v1`: `output = base + effective_gain * contribution`.
  Its base domain is any finite real scalar, subject to a site's
  declared narrowing. It applies only where the site's tensor
  contract defines `base` and `contribution` with compatible layout,
  shape, and dtype.
- `LerpReplacement.v1`:
  `output = base * (1 - effective_gain) + replacement * effective_gain`.
  Its realized effective gain domain is [0, 1]. Extrapolation is not
  this operator; a future named operator is required if a concrete
  use case needs it.
- `AttentionTerm.v1`:
  `output = ordinary_attention + effective_gain * extra_attention_term`.
  Its base domain is any finite real scalar, subject to site
  narrowing. The mask addresses the declared output/query rows;
  K/V preparation remains a separate typed input contract.
- `Overlay.v1`: first evaluates the existing chapter 1.1 typed
  weight-overlay algebra at its declared payload strengths, then
  applies
  `output = base_weight + scalar_gain * (typed_overlay(base_weight) - base_weight)`.
  Its scalar domain is any finite real value, subject to site
  narrowing. This fixes gain behavior for every structural patch
  kind, including replacement, while leaving behavioral-leaf
  identity and version in the `WeightOverlay` payload. The initial
  operator refuses effect masks and non-default lane gains because
  model weights have neither token rows nor lane-local ownership.

`Additive.v1` and `AttentionTerm.v1` guarantee strength zero returns
the base value and strength one adds the unscaled contribution.
`LerpReplacement.v1` guarantees zero returns base and one returns
replacement. `Overlay.v1` guarantees zero returns base weights and
one returns the typed overlay result. A provider that needs another
blend formula must add a reviewed operator vocabulary revision; it
cannot hide the formula in provider code.

### 10.7 Manifest, receipts, and cache identity

Control contributions extend the existing intervention-plan manifest
slot of chapter 6.1. They do not create a control manifest, gain
identity, mask identity, or receipt lineage beside the canonical
invocation manifest. Declared curve/keyframe and mask artifacts are
content-addressed resource bindings referenced by digest; artifact
bytes never enter the manifest slot. The intervention-plan slot
binds, at minimum:

- the declared schedule artifact digest; interpolation, endpoint,
  omission, and assignment-profile identities and parameters; and
  the hard/best-effort policy;
- the exact `RealizedGainTable` digest;
- global, resolved-site, and guidance-lane gains;
- each source-mask digest and compiled token-field digest, the
  `TokenGridTransform` profile identity, target `ModelTokenLayout`
  digest, and composition order;
- selected sites, typed application operator and version, and scalar
  gain domain;
- the contribution's existing partition compatibility declaration.

When windowing is active, the intervention-plan slot references the
chapter 9 composite plan digest and binds each derived per-window
field digest against that chapter's window identity and inner-layout
digest. It does not duplicate or redefine the window schedule. It
likewise references the chapter 5 partition-plan digest rather than
minting a control shard identity.

Receipts expose the declared schedule and the exact realized table,
or a content-addressed artifact containing that table, plus the
operator and mask-field digests. The existing `plan_digest` therefore
changes whenever a resource, curve, realized row, keyframe
assignment, mask, site, lane, operator, window, or supported
partition behavior changes. Chapter 7 cache keys already include
that digest, and chapter 9 adds window identity; no control-specific
cache key or invalidation channel is permitted.

### 10.7 Named refusals and conformance cases

At minimum, compilation reports these stable refusal reasons instead
of a generic provider error or runtime fallback:

- `gain_not_applicable` for gain/operator fields on a non-applying
  effect, including `ConditioningTransform`, `AuxiliaryResourceRef`,
  and run overrides;
- `invalid_gain_schedule` for mixed coordinates, invalid ordering,
  implicit state, or a non-finite value;
- `unsupported_gain_profile` for an unknown interpolation,
  endpoint, omission, or assignment profile;
- `unsatisfied_keyframe_minimum` for an impossible hard hold;
- `gain_domain_mismatch` for a realized scalar product outside the
  operator/site domain;
- `control-resource-binding-mismatch` when a declared ControlNet model or
  hint digest does not match its loaded or materialized resource, including
  an unresolved hint reference;
- `mask_role_mismatch`, `mask_value_out_of_range`,
  `missing_token_grid_transform`, `mask_layout_mismatch`, and
  `unsupported_mask_composition` for the mask contracts above;
- `unsupported_application_operator` and
  `operator_site_mismatch` for a provider formula or site pairing
  outside the closed vocabulary;
- `overlay_field_unsupported` for an effect mask or lane gain on the
  initial overlay operator;
- `unsupported_control_window_mapping` and
  `unsupported_control_partition` when chapters 9 and 5 cannot
  preserve the declared semantics.

The paper conformance set is: a ControlNet and a T2I-Adapter sharing
`Additive.v1` residual-site application despite different auxiliary
execution timing; an IP-Adapter-style `AttentionTerm.v1`; an explicit
feature replacement using `LerpReplacement.v1`; and a
`WeightOverlay` using `Overlay.v1`. Each case must represent strength
zero and one, a nontrivial curve, all-zero/all-one/partial effect
masks where the operator supports them, lane gains, overlapping and
wrap-around windows, and every partition mode it declares. An
unsupported cross-product must produce the named compile-time
refusal, never a provider-specific blend, slice, or fallback.

## 11. Portable VIDEO values

`comfy.VIDEO` codec v2 is immutable source/probe/ordered-edits data, or
deferred IMAGE/AUDIO components with the same edit semantics. The exact
fields, node surface, precision, compatibility differences, and save rules
are defined in [First-class VIDEO values](video-io-design.md). That document
replaces PR #108's frame-batch value model; frame materialization is an
explicit consumer action, never a prerequisite for trim or crop.

### 11.1 Portable source and wire identity

A source is an immutable asset reference or inline encoded bytes of at most
256 KiB. Asset references serialize identity and metadata using the existing
asset wire shape (`digest`, `name`, `size`, `mediaType`, `virtualPath`), never
a resolver, producer path, arbitrary URL, or serialized callable. The
receiving process binds its own authorized resolver. Larger bytes must be
published to the host's content-addressed asset store before a new v2 value
crosses a process or machine boundary. Legacy v1 migration may temporarily
hold larger bytes until that publication boundary; it cannot write them back
as oversized v2 inline data.

The v2 payload uses the `DINKSTER-VIDEO` magic followed by version byte `2`,
an unsigned little-endian 64-bit JSON-header length, the canonical header,
and length-declared binary chunks. JSON is UTF-8 with sorted keys, compact
separators, no NaN/Infinity, and no duplicate keys. Rational values serialize
as reduced `[numerator, denominator]` integer pairs with positive denominator.
The header contains source or component descriptors, probe, ordered edits,
and chunk lengths/types. Binary chunks contain inline source bytes or
canonical IMAGE/AUDIO codec payloads, never base64 or pickle. No implicit
padding, trailing bytes, overlapping ranges, or undeclared chunks are legal.
Component codec identity and metadata accompany each component chunk.

Header size is at most 1 MiB; the entire edit tree has at most 256 operations
and 64 source clips, with nesting depth at most 16. Existing encoded-media
and decoded-array bounds still apply. Validate lengths and the recursive
structure before allocating, resolving assets, or invoking a codec. Unknown
versions and operations fail closed; readers never reinterpret them as v1.

Fingerprinting includes the codec version, canonical source content identity
or component fingerprints, source probe, and ordered edits. Names, mounted
paths, resolver objects, location, and transfer handles are not semantic
identity. The same source bytes have the same content identity whether inline
or asset-backed. Presentation-only raw widget spelling belongs to the graph,
not the media fingerprint; normalized execution parameters determine media
identity. Encode, fingerprint, and metadata queries never encode a video or
materialize its pixels as a side effect.

### 11.2 Metadata, accounting, and trust

Envelope metadata includes codec version, the source probe, effective facts,
and a deduplicated `asset_refs` list containing every source reference in the
value, including concat children. The payload remains authoritative: encoded
validation cross-checks metadata against its declared sources and pure edit
projection. Metadata cannot introduce an asset absent from the payload or
conceal one needed for execution. An asserted probe never grants access;
source admission verifies it against the authorized bytes. Receivers preserve
that admitted binding and verify the content digest through the asset layer.

`COST_META_KEY` reports resident container bytes, not decoded-frame estimates.
An asset-backed descriptor does not claim the source file is resident in RAM;
inline data reports its actual encoded-byte residency. Deferred components
report their actual component storage cost without claiming they are an
encoded container. Source byte size and transfer cost are separate facts.
The value has no GPU residency until a materializing consumer creates it.
Host-local residency accounting belongs to #1251, not the serialized edits.

### 11.3 Process, machine, and cloud boundaries

Asset discovery traverses VIDEO references in addition to ordinary asset
inputs and list descendants. The transport validates `asset_refs`, stages
missing authorized content through the existing vault/value-store/P2P chain,
and deduplicates by digest. A value referencing an unavailable source fails
with that source identity; it never falls back to a producer-local path.
Admission includes nested references for authorization, not just transfer.

The host that accepts a producer's asset-backed output must make its source
available to downstream placement before releasing that producer. Publication
or a retained source lease is part of output acceptance. Forwarding only the
descriptor and forgetting a producer-private file is not a successful output.
Staged source bytes remain available under the existing job/store lifetime;
they are not fetched again for each lazy node.

Trimming on host A changes metadata only. Saving on host B fetches a missing
source once, then opens its locally verified content for packet-copy or
bounded transcode. Shared memory carries the same canonical v2 payload, not
an alternate object layout. In-process, shared-memory workers, remote daemons,
and cloud workers use identical validation, edit arithmetic, and identity.
No codec-specific remote filesystem or separate media execution protocol is
permitted. Components use the existing value transport; frame-tensor streams
are a separate contract in #1252.

### 11.4 Conformance

The eleven criteria in #1250 govern implementation acceptance. Tests must
prove both source portability and output equivalence: zero frame decodes at
lazy edit producers, one missing-source fetch at the destination, unchanged
serialized edits across hops, exact deterministic output bytes, and declared
codec tolerances only for re-encodes. The 200 MiB CPU RSS-growth gate includes
the process doing the save. Missing LAN/cloud/GPU lanes are recorded and
requested explicitly; an unexecuted lane never inherits a local pass.

## 12. Image alpha and mask semantics

IMAGE metadata declares `channels: {layout, alpha}` and `color` without
allocating channels. Layout is `gray`, `gray_alpha`, `rgb`, `rgba`, or
`planes`, derived from the channels-last shape. Alpha is `none` for images
without alpha, otherwise `straight` (default) or `premultiplied`. Color
uses nonnegative FFmpeg integer enums and defaults to
`{primaries: 1, transfer: 13, range: 2}` (sRGB/sRGB/full). Unknown and reserved
integers are preserved. Decoded RGB uses full range (2); optional `matrix`
and `bit_depth` preserve source provenance. `image_color` owns validation and
defaults for both IMAGE and layer documents. Carrying color metadata does
not perform color conversion or tone mapping.

MASK metadata declares `polarity: coverage | transparency` and
`semantic: alpha | selection | other`. Coverage means 1 selects; transparency
means 1 is see-through. Unannotated masks default to coverage/selection.
`annotate_image`, `annotate_mask`, and `copy_media_semantics` return annotated
array/tensor views. Callers must return the helper result. Arbitrary array
operations do not infer new metadata. At the worker boundary, an unannotated
output inherits matching input semantics only when they are unambiguous.

InputSpec and OutputSpec have `alpha_policy`: `preserve` (default), `require`,
`create_if_missing`, or `drop`. Require reports an ordinary typed node error
when alpha is absent. Create-if-missing appends opaque alpha; preserve and
drop do not change the input array. Drop declares intentional loss, rather
than asking the engine to truncate channels before node code runs. Policies
apply to list elements too. Optional `mask_polarity` and `mask_semantic`
declare mask expectations. The schema wire exposes these as `alphaPolicy`,
`maskPolarity`, and `maskSemantic`. Nondefault policies affect schema identity.

The `value_diagnostics` event carries `detail.diagnostics`. Each entry has
`code` and `nodeId`. Unexpected alpha loss includes `outputId` and the
alpha-bearing `inputIds`; asset-coercion loss also names `inputId`.
`mask_polarity_mismatch` names `inputId`, `actual`, and `expected`. These are
nonblocking diagnostics, not execution admission rules. The engine checks
fresh, coalesced, and cached output envelopes. Workers retain observed
diagnostics in output metadata as `valueDiagnostics`, including losses after
asset decoding, so cache replay does not need to reopen assets. Image header
inspection is best-effort for custom decoders; unknown formats remain executable.

The image/mask codec uses the ordinary npy payload. Nondefault semantics add
a bounded `DINKSTER-MEDIA` version-1 trailer with a little-endian 32-bit JSON
length and at most 4096 bytes of canonical JSON. Default semantics leave
legacy bytes and fingerprints unchanged. Payload validation checks semantic
metadata and shape before allocating pixels. Shared memory uses the same
encoding and fingerprint as in-process values. PNG renditions convert
premultiplied samples to straight alpha without modifying the source.

Future mesh, splat, or texture values can declare channels on texture maps
and color on albedo. They can reuse these diagnostic and inspector contracts;
this declaration does not introduce a 3D renderer or texture type.

## 13. Asset-backed audio

`comfy.AUDIO` carries a source, probe, and ordered edits. Sources are asset
references, inline encoded bytes, or inline int16/float32 PCM up to 256 KiB.
Probe fields are `sample_rate`, `channels`, `layout`, `duration`, `codec`,
`frames`, `batch`, and `stream_index`; duration and frames may be unknown.
The stream index is an ordinal among audio streams, not the container-global
stream index.
Legacy waveform carriers retain int16/float32 storage and normalize other
numeric dtypes to float32 before envelope metadata is computed. Typed
consumer windows convert int16 PCM to float32 by dividing by 32768.
Edits are one-key objects: `trim: {start_sample, sample_count}`, `gain`
(linear scalar), `resample` (target Hz), `channel_map: {matrix, layout}`,
or `concat: [child AUDIO, ...]`. Matrix rows are output channels and columns
are input channels. Concat appends each child's effective timeline and requires
matching sample rate, channels, layout, and batch, with known frame counts
(trim unknown durations first). The probe remains measured facts of the real
root stream; reads recheck it, with declared layout retained for npy PCM.

`audio_from_source` probes headers without decoding samples. `audio_window`
reads sample-indexed windows from the effective edited timeline, returning
float32 `[B,C,T]` samples and `sample_rate`. Its optional `batch_index`
selects one batch; otherwise batches are preserved. Windows and intermediate
arrays are bounded at 32 MiB; streaming consumers request successive windows.
`effective_audio_facts` and `append_audio_edit` do not materialize samples.
Legacy `waveform` access is an explicit bounded materialization boundary.
Unknown-duration sources need an explicit window or trim at that boundary.

`AudioWindowReader` is a consumer-owned context manager for sequential
windows. Its `read` method accepts the same sample coordinates as
`audio_window`, reuses one live decoder, and retains bounded overlap caches
across concat sources. Large caches spill to temporary files; closing the
reader releases decoders and caches. Backward cache misses reopen the source.
AAC windows decode from the stream origin to preserve codec noise state;
saving, muxing and onset detection reuse a reader rather than repeatedly
decoding that prefix. Readers are not thread-safe and do not mutate values.

The wire is `DINKSTER-AUDIO` version 2 followed by a little-endian uint32 header
length, canonical source/probe/edits JSON, and inline binary payloads in
preorder: root source, then each concat child's subtree in edit order.
Each source is `{inline_pcm: length}`, `{inline_encoded: length}`, or
`{asset: descriptor}`; concat children contain full source/probe/edits headers.
PCM lengths include npy headers, but the 256 KiB PCM limit counts sample data.
Headers are bounded at 1 MiB, depth 16, 64 records, and 256 total edits.
Asset references never serialize local paths or resolver bindings.
Persisted v1 uint64-rate plus float32 npy remains readable. Admission
validates the entire tree, shape, dtype, and exact total
payload consumption before allocating samples, without probing encoded media.
VIDEO components use this same AUDIO codec.

Metadata includes effective facts, stable preorder digest-deduplicated
`asset_refs`, and `cost.ram` summed across resident PCM and encoded bytes,
not unopened assets. Receiving hosts recursively bind their own resolver;
remote staging discovers references by metadata, not a media-type list.
Native/compat registration publishes large legacy PCM, including concat
children, to the configured asset
vault without losing edits or layout. WAV previews select the first batch
and at most ten seconds. Onset output uses the foundation
CURVE schema, whose points are `{position: seconds, value: strength}` objects.
