# Single-job multi-GPU architecture

## Goal

A sampling job can use multiple selected GPUs in one of four ways:

- evaluate independent guidance lanes in parallel;
- run independent batch shards in parallel;
- shard H3 sequence rows and attention heads across devices; or
- scatter a windowed Flux plan's per-step window evaluations across ranks.

These modes complement whole-job replicas. Replica mode improves queue throughput,
while the modes here reduce one job's latency or make an over-VRAM model runnable.
Cross-machine execution is not part of this architecture.

## Device and process boundary

The user supplies an ordered list of visible CUDA indices. The list is interpreted
after `CUDA_VISIBLE_DEVICES`, so a process started with
`CUDA_VISIBLE_DEVICES=1,2` can safely select logical devices `0,1` while physical
device 0 remains unavailable. No mode discovers or adds devices outside that list.

Whole-job replicas keep their current one-worker-per-device topology. Single-job
execution uses a workgroup with one authenticated rank per selected device. The
parent engine owns admission, cancellation, progress, and final gather. Each rank
owns model residency and a local CUDA stream. Rank 0 owns the sampler state and is
the semantic owner of the final result.

The workgroup protocol remains the control plane:

```text
engine -> prepare/admit/commit -> all ranks
engine -> run sampling unit     -> all participating ranks
ranks  -> unit result/failure   -> engine gather
engine -> cancel/release        -> all ranks
```

The v1 protocol has one `SINGLE` unit per replica and allows that unit to run once.
Single-job execution additionally negotiates a v2 capability that gates each rank's
native invocation on the committed workgroup. Stable rank-local resource mapping
binds each invocation input to the corresponding rank's resident model. The sampling
unit runs the complete trajectory and is not recreated for each diffusion step.

Per-evaluation guidance assignments and sequence collectives use NCCL inside that active
unit. A control tensor fences each monotonic evaluation ordinal and tensor shape in a
communicator unique to the workgroup attempt. This keeps high-frequency
synchronization out of the parent control plane while retaining one cancellation and
release authority.

Tensors do not make per-layer round trips through the parent protocol. Guidance
ranks start from corresponding rank-local invocation inputs. Sequence ranks use a
local `torch.distributed` process group with NCCL collectives. The workgroup fences
the rank identities and lifecycle; NCCL carries the hot-path tensors.

## Mode selection

Serve adds a separate single-job device list and mode. Existing
`--multi-gpu-devices` continues to mean whole-job replicas.

```text
--single-job-multi-gpu-devices 0,1[,2...]
--single-job-multi-gpu-mode auto|guidance|sequence|window
```

The two device-list flags are initially mutually exclusive. Combining queue replicas
with an inner multi-GPU workgroup would multiply residency and complicate admission;
it can be enabled later without changing either execution contract.

`auto` chooses only an eligible mode:

1. receipted guidance parallelism when at least two model-evaluated guidance lanes exist;
2. window scattering when a windowed Flux plan has at least two joint windows
   and a registered window receipt;
3. otherwise refusal before sampling, not duplicate single-device work.

An explicit ineligible mode fails before sampling. It never silently consumes extra
VRAM while running on one GPU. The selected arm identity includes the mode and rank
count.

Explicit `sequence` mode assigns every selected rank to pure Ulysses sequence
parallelism (`U=rank count`, `R=1`, guidance degree 1). It is not part of `auto`.
The runtime admits only exact registered family, provider, dtype, rank-geometry,
and device-environment receipts before process-group setup.

After the initial mode-specific live proofs, request-level selection adds mode and
logical device fields to the enqueue API. Server startup configuration becomes an
allowlist and capacity boundary; each run may choose a permitted subset and eligible
mode without restarting serve. Admission still prevents overlapping runs from
claiming the same exclusive devices. This runtime selector is a follow-up to the
fixed-mode implementation, not a prerequisite for its first performance proof.

## Guidance-lane parallelism

### Work decomposition

`GuidanceEvaluationPlan` is the source of work. Synthetic-zero lanes remain local.
Each model-evaluated lane becomes a rank-data-plane assignment containing:

- evaluation ordinal and sigma;
- the current latent input;
- the lane ID, role, and conditioning authority;
- the exact sampling execution context.

The lane planner assigns units by measured relative throughput while preserving the
plan's canonical lane order. A rank holds a full model replica and immutable
conditioning residency. It evaluates its assigned lanes and returns predictions
tagged by lane ID. Rank 0 restores canonical order, runs pre-guidance transforms,
the guidance reducer, post-guidance transforms, and then advances the solver.

### Synchronization and transfer

There is one scatter/gather barrier per denoiser evaluation. Solvers that evaluate
the denoiser more than once per step therefore have more than one barrier per step.
For a latent of `X` bytes and `R` active ranks, the worst-case rank-0 traffic per
evaluation is approximately `2 * X * (R - 1)`: one latent copy out and one prediction
back per remote rank. Conditioning transfers once when residency is prepared, not
on every step.

The reducer never runs until all required predictions for the same evaluation
ordinal arrive. A stale ordinal, duplicate lane, missing lane, or rank failure fails
the workgroup and cancels every rank. Partial predictions are never reused.

### Correctness

Each rank evaluates one batch-1 lane with the same full-model forward used by the
single-device split reference. Gather order matches `GuidanceEvaluationPlan`.
CFG++ receives the exact unconditional prediction. Guidance wrappers and custom
reducers remain rank-0-owned and see the same request and prediction sequence as
the split reference.

Ordinary single-device SD1.5 fuses conditional and unconditional lanes into one
batch-2 forward for performance. Convolution accumulation is batch-shape-visible,
so the distributed receipt compares against the equivalent single-device batch-1
lane execution rather than the fused result. Its fixed 512/20 and 1024/30 goldens
match exactly. Instrumentation at 1024/30 additionally proves exact per-rank lane,
gather, combine, per-step state, and final-output seams. This is a reference-shape
distinction, not a distributed tolerance.

The ComfyUI `worksplit-multigpu` branch is the functional reference for full model
copies, per-device conditioning work, and heterogeneous load balancing. Dinkster does
not copy live Python model graphs or coordinate CUDA work with ad hoc threads;
replicas are rebuilt from stable recipes inside fenced workers.

## Batch sharding

Batch sharding remains unexposed until a family-specific numerical receipt proves
that changing each denoiser call from the canonical full batch shape to rank-local
shard shapes preserves the existing single-device result.

### Work decomposition

The batch planner partitions original batch indices into stable, ordered shards,
weighted by measured rank throughput. Every tensor whose leading dimension follows
the latent batch is sharded by the same index set: latent, prepared noise, masks,
conditioning, ADM, controls, and batch-shaped extension data.

Each rank runs the complete sampler trajectory for its shard. This is preferable to
gathering every step because ordinary diffusion batch elements are independent.
Rank 0 gathers final latent shards once in original index order.

### Synchronization and transfer

Deterministic solvers have one barrier after preparation and one final gather.
Progress reports carry rank, shard indices, and local step; rank 0 publishes a step
only after every active shard has reached it. Cancellation remains all-rank.

For total latent size `X`, approximately `X * (R - 1) / R` leaves rank 0 at scatter
and the same amount returns at gather under equal shards. Model and conditioning
residency are prepared once per rank.

Rank 0 creates the canonical full-batch initial noise and scatters it by original
index. Stochastic samplers add one scatter at each noise draw: rank 0 advances the
unchanged full-batch noise authority, then sends each rank its slice. This preserves
the existing generator call shape and stream instead of inventing per-shard seeds.
Brownian/SDE samplers remain ineligible until their tree construction, bounds, and
draw sequence are proven identical through this authority. Extensions that declare
cross-batch behavior also make the mode ineligible.

### Correctness

Gather is by original index, never completion order. A parity receipt compares every
batch element against the corresponding single-device element at the schedule,
noise, per-step latent, and final output seams. Batch fusion can be float-visible, so
hash equality is required only after the seam evidence proves the selected family is
batch-independent; otherwise the mode remains disabled for that family.

## Residency, identity, and admission

A stable shard recipe digest includes:

- model family and artifact digests;
- mode and algorithm version;
- rank count and ordered tensor/block partitions;
- storage and compute dtypes;
- attention provider and collective backend versions;
- all float-visible wiring choices.

Physical PCI indices, PIDs, process tokens, and transient free-memory readings are
not computation identity. Authenticated worker instance tokens still fence live
resident handles. Cache identity includes the stable recipe and execution arm, so
single-device, replicated, and sequence-parallel results never alias accidentally.

Admission serializes the selected-device pool and atomically reserves every declared
rank-local VRAM input cost plus checkpoint-load RAM before commit. Child ranks do not
acquire overlapping leases after parent admission. A refusal releases the complete
reservation batch before any rank runs. Runtime release retains healthy rank-local
resources for later graph nodes. Parent-visible resident envelopes claim every selected
device so engine admission excludes other arms from hidden ranks. A failed unit
terminates every rank, releases its residency, and replaces the workers before retry.

## Expected scaling

These are engineering targets for GPU-bound denoising, not guarantees for complete
workflows with serial conditioning or decode:

| Mode and eligible work | 2 GPUs | 3 GPUs | Limiter |
| --- | ---: | ---: | --- |
| Two balanced guidance lanes | 1.7x-1.95x | 1.7x-1.95x | Only two model lanes |
| Three or more guidance lanes | 1.7x-1.95x | 2.3x-2.75x | Per-evaluation gather |
| Batch >= rank count | 1.7x-1.9x | 2.3x-2.7x | Slowest shard, final gather |

## Validation

Every enabled family/mode/device-count combination requires:

- protocol and lifecycle tests for prepare, refusal, cancellation, rank failure,
  stale attempts, duplicate sideband ordinals, gather order, and full release;
- CPU fake-rank tests covering uneven 2- and 3-way partitions;
- CUDA tests proving selected-device exclusion and per-rank residency;
- seam-level numerical receipts followed by same-seed final output comparison;
- sustained GPU-bound timing with utilization, memory, transfer volume, and
  collective timing;
- 2x RTX 4090 evidence for eligible light-model guidance and batch modes;
- comparison against single-device and whole-job replica modes so throughput and
  latency claims cannot be confused.

No mode is enabled by `auto` until its family-specific correctness and performance
receipt is registered.
