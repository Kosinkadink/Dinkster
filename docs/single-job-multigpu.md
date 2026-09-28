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

## Distributed call shape

Every rank calls the selected attention backend with the same complete inputs,
head count, and tensor layout as stock ComfyUI. Each rank then retains its
ordered share of output heads. An all-gather reconstructs the complete output
on every rank.

Keeping the backend call shape unchanged is a numerical requirement. Splitting
the backend input by head changes CUDA accumulation for otherwise equivalent
attention and can change decoded output bytes. Output ownership distributes the
collective result without changing that backend computation.

This contract does not promise lower latency: all ranks currently perform the
full attention call. Performance work may partition computation only after an
oracle proves that the new call shape preserves the accepted stock output.

## Mode and admission

Serve exposes:

```text
--single-job-multi-gpu-devices 0,1[,2...]
--single-job-multi-gpu-mode auto|sequence
```

Both accepted modes use the worker-owned distributed attention route. `auto`
does not select retired guidance- or window-parallel implementations.
`guidance` and `window` are invalid arguments and fail before startup.

The selected rank count must be at least two, device indices must be unique and
nonnegative, and the attention head count must divide evenly across ranks.
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
