# Multi-GPU hosting

This guide covers same-host NVIDIA multi-GPU execution. Treat topology,
transport, and attention backend selection as measured host capabilities. GPU
model names and total VRAM do not prove that a distributed job is correct or
faster.

## Choose the execution model

Dinkster exposes two independent multi-GPU configurations:

- `--multi-gpu-devices` starts independent whole-job replicas. Use this for
  concurrent-job throughput.
- `--single-job-multi-gpu-devices` starts one isolated rank per device for a
  single graph execution. The ranks coordinate through the worker-owned
  attention route. Job submission can choose an ordered subset of this startup
  pool and an eligible guidance, sequence, or window mode.

Do not compare the latency of one distributed job with the throughput of
several replicas. Record wall seconds per job and completed jobs per hour as
separate results.

The supported single-job graphs and environments are listed in
`docs/supported/standalone-native-generation.md`. The implementation and its
failure boundaries are described in `docs/single-job-multigpu.md`.

## Isolate the device list

Use `CUDA_VISIBLE_DEVICES` to hide reserved GPUs before choosing logical device
indices. This exposes physical GPUs 1 and 2 as logical GPUs 0 and 1:

```sh
CUDA_VISIBLE_DEVICES=1,2 dinkster-serve \
  --single-job-multi-gpu-devices 0,1 \
  --single-job-multi-gpu-mode auto
```

Dinkster does not discover or add devices outside the visible list. Confirm the
mapping and topology before starting the service:

```sh
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total \
  --format=csv
```

Verify peer access in the exact torch environment used by Dinkster:

```python
import torch

for source in range(torch.cuda.device_count()):
    for destination in range(torch.cuda.device_count()):
        if source != destination:
            print(source, destination, torch.cuda.can_device_access_peer(source, destination))
```

Peer access is not required for correctness. When it is unavailable, NCCL may
use host-backed shared memory and distributed calls may be slower.

## Verify NCCL transport

Start one representative fresh process with NCCL logging enabled:

```sh
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH dinkster-serve ... 2>&1 | tee nccl.log
grep -E 'via (P2P/CUMEM|SHM/direct/direct|NET/Socket)' nccl.log
```

Remove inherited NCCL tuning variables for the baseline. Do not force P2P,
disable IOMMU, or add an ACS override because another host used it. If NCCL
selects an unexpected transport, inspect the host's topology, peer-access
result, BIOS settings, IOMMU groups, and driver logs before changing Dinkster.

Transport choices are workload- and topology-dependent. Re-run the transport
check after changing the driver, torch, CUDA, NCCL, GPU placement, or firmware.

## Process and memory hygiene

Use fresh processes for acceptance and performance measurements. A warm process
can retain models, CUDA contexts, NCCL communicators, pinned buffers, and
allocator state from an earlier arm.

Before and after every measured arm:

1. Confirm that every selected GPU is unclaimed and record its compute PIDs.
2. Record available host memory, GPU UUIDs, memory use, utilization, and driver.
3. Run the complete arm with a host-appropriate memory limit and no scope swap.
4. Check cgroup memory events for allocation failures or OOM kills.
5. Confirm that no compute PID remains on a selected GPU after release.
6. Check kernel logs for NVIDIA Xid errors or device inventory changes.

Any Xid error, unqueryable GPU, changed device inventory, OOM event, or foreign
compute PID invalidates the arm. Retain its logs as diagnostics, not acceptance
or performance evidence.

## Numerical acceptance

Compare each single-job distributed arm with stock ComfyUI at the fork's
recorded upstream commit. Keep these inputs identical:

- workflow graph and model artifacts;
- seed, sampler, scheduler, and step count;
- torch runtime, attention provider, dtype, and GPU architecture; and
- decoded-output procedure.

Require the accepted byte hashes on every rank and repeat. If output differs,
instrument the first divergent seam rather than changing a tolerance. A clean
run that never reaches the distributed attention hook is not distributed
acceptance.

## Performance measurements

Correctness does not imply speedup. Guidance divides model-evaluated
conditioning lanes, H3 Ulysses divides sequence rows and attention heads, and
window mode divides a joint-window plan. Verify that the selected decomposition
ran; process count alone is not evidence that work was sharded.

When evaluating later performance work, record cold complete-workflow wall
time, five warm complete-workflow walls, sampling-only time, peak memory, and
total GPU-seconds. Use a same-session one-GPU baseline, balance arm order when
temperature or clocks may drift, and save every repeat rather than only the
median.

## Receipt record

Retain enough information to reproduce both the accepted path and its failure
boundaries:

- exact Dinkster and `dinkster-inference` commits;
- stock ComfyUI reference commit;
- model and workflow artifact digests;
- GPU UUIDs, topology, peer access, and selected NCCL transport;
- driver, Python, torch, CUDA, NCCL, and attention provider versions;
- complete commands, environment overrides, logs, and output hashes; and
- rank count, logical rank order, memory events, and before/after process gauges.

Re-run a representative oracle after any source, model, runtime, provider,
driver, or topology change.
