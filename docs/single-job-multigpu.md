# Single-job multi-GPU architecture

Single-job multi-GPU execution uses one isolated worker process per selected
CUDA device. It is separate from whole-job replicas: replicas improve queue
throughput, while a single-job workgroup coordinates one graph execution across
all of its ranks.

## Device and process boundary

The user supplies an ordered list of logical CUDA indices. Indices are relative
to `CUDA_VISIBLE_DEVICES`; Dinkster never discovers or adds devices outside that
list. Each worker receives one visible device and owns its model residency,
CUDA state, and attention registry.

The parent uses the workgroup protocol as the control plane:

```text
engine -> prepare/admit/commit -> all ranks
engine -> run sampling unit     -> all ranks
ranks  -> unit result/failure   -> engine
engine -> cancel/release        -> all ranks
```

The committed attempt activates distributed attention for one invocation on
each rank. A communicator derived from the attempt identity prevents stale or
foreign ranks from joining it. Release destroys the process group and clears
all attempt-local state.

NCCL is the tensor data plane. A control collective checks that every rank has
reached the same attention call index and tensor shape before output exchange.
A rank failure fails the workgroup rather than allowing partial output.

## Attention attachment

The worker owns a named attention registry and resolves its route without
changing the fork's default registry. At a sampling boundary, Dinkster clones
the resident `ModelPatcher` and attaches the invocation's optimized-attention
callable to the clone. The resident model is never mutated.

This attachment applies to direct KSampler execution and to imported
`BasicGuider` graphs that feed `SamplerCustomAdvanced`. Both paths therefore
use the same worker-owned route and workgroup lifecycle.

## Distributed execution modes

Guidance mode assigns model-evaluated conditioning lanes to ranks in canonical
lane order. Synthetic zero lanes remain local. The rank outputs are summed so
every rank observes the canonical conditioning result.

Window mode assigns contiguous joint-window ranges to ranks. Ranks evaluate
their ranges concurrently, broadcast each result in plan order, and run the
fork's canonical merge once. The four-rank path uses all ranks when the plan has
at least four joint windows.

Sequence mode supports the two-rank H3 U2R1 layout. Each DiT block retains one
contiguous sequence shard. Ulysses all-to-all exchanges transform local
sequence/full-head QKV into full-sequence/local-head attention and invert the
exchange for the residual and MLP. The final block gathers sequence rows before
the H3 output heads. Four-rank Ulysses is refused.

## Mode and admission

Serve exposes:

```text
--single-job-multi-gpu-devices 0,1[,2...]
--single-job-multi-gpu-mode auto|guidance|sequence|window
```

`auto` selects the two-rank sequence path. Guidance and window execution must
be selected explicitly. Sequence mode requires exactly two ranks; guidance and
window accept two or more ranks.

The startup list is an allowlist and capacity boundary. Native and
Comfy-compatible job submission may include a run-scoped selection:

```json
{"singleJobMultiGpu":{"cudaIndices":[1,3],"mode":"sequence"}}
```

The pool maps the ordered host indices to local ranks 0 through N-1 before the
worker boundary. Only those workers join the workgroup. The host selection is
never serialized to a rank; rank invocations carry only their local rank,
world size, and resolved mode. Mode and ordered indices partition execution
cache identity and rank-local resident-resource mappings.

The selected rank count must be at least two, device indices must be unique and
nonnegative, and the Ulysses attention head and packed sequence counts must
divide evenly across its two ranks.
Single-job devices and whole-job replica devices are mutually exclusive so one
serve process cannot create hidden overlapping residency.

## Numerical acceptance

An enabled graph and environment must be compared with stock ComfyUI at the
fork's recorded upstream commit using the same graph, artifacts, seed, sampler,
scheduler, steps, torch runtime, and GPU. Acceptance compares decoded output
bytes, not only successful execution. Provider versions, rank order, device
architecture, and the attention route are part of the evidence.

The supported combinations and accepted evidence are listed in
`docs/supported/standalone-native-generation.md`.
