# Multi-GPU hosting

This guide covers same-host NVIDIA multi-GPU execution. Treat topology,
transport, and attention-kernel selection as measured host capabilities. GPU
model names and total VRAM do not prove that a distributed mode will be faster.

## Choose the mode for the workload

Dinkster has two different ways to use multiple GPUs:

- `--multi-gpu-devices` runs independent whole-job replicas. It is the exposed
  multi-GPU option for concurrent jobs; measure workflow throughput on the
  exact host and artifacts before claiming a scaling factor.
- `--single-job-multi-gpu-devices` distributes one job with an eligible
  guidance, window, or explicit sequence route; lower latency is not
  guaranteed. See
  [serve-cli.md](serve-cli.md) for the currently exposed modes and receipts.

Pure Ulysses sequence parallelism is an explicit `dinkster-serve` mode and is never
selected by `auto`. It maps all selected ranks to Ulysses degree equal to the
rank count, Ring degree 1, and guidance degree 1. Current receipt admission is
limited to MiniMax H3 FL2VA BF16 on two SM 12.0 Blackwell GPUs under torch
2.13.0+cu130, with built-in SDPA or explicitly selected comfy-kitchen 0.2.31
or 0.2.32 INT8 attention, each with its own exact receipt. Other provider
versions refuse. Ring and hybrid sequence plans fail closed at compilation
with production attention providers. SDXL and Flux have no receipted general
sequence-parallel route. Flux window scattering is a separate, receipted
route for windowed plans, not ordinary full-frame sampling.

Guidance splitting and U2R1 are distinct. Guidance assigns conditional and
unconditional CFG lanes to full-model ranks. U2R1 is two-way Ulysses sequence
parallelism: it partitions sequence rows and attention heads for one model
evaluation, then exchanges attention data. A result or receipt for one method
does not authorize or predict the other.

Do not compare the latency of one distributed job with the throughput of
several replicas. Record both wall seconds per job and completed jobs per hour.
When a family has no receipted single-job route, replication is the supported
multi-GPU option, but its throughput factor still requires measurement.

Performance targets are 1.8x total-workflow speedup on two GPUs, with 1.5x an
attractive floor. General sampling-acceleration comparisons use one versus
two GPUs only, per the [comparison scope](https://github.com/Kosinkadink/Dinkster/issues/1266).
NVLink results may inform future designs but are not requirements for the
target consumer and workstation PCIe hosts.

Window- or tile-parallel acceleration applies only to workflows that contain
those structures. Report its results in a separate circumstantial category;
never count them toward the general-workload target above.

Useful comparison points, not inline porting requirements, are:

- the internal ComfyUI split-attention prototype at `9342fa8`, reported at
  1.5x on two RTX 4090 GPUs over PCIe 4.0 x8/x8;
- ComfyUI's `comfy/multigpu.py`, `nodes_multigpu.py`, and
  `_calc_cond_batch_multigpu` split-CFG path for like-for-like guidance tests;
- [ComfyUI-MiniMaxH3-Parallel](https://github.com/AesSedai/ComfyUI-MiniMaxH3-Parallel),
  which reports 1.50x, 1.82x, and 2.02x on two, three, and four workstation
  GPUs and is a data point rather than a design to copy; and
- comfy-aimdo PR 58, `dev/multi-gpu-ring-split`, for CFG-split offloading
  capability context.

Production single-job multi-GPU uses one OS process per rank with NCCL. No
single-process two-GPU sampling path has a performance receipt.
`multidevice_attention.py` is an opt-in direct-import correctness prototype,
not a production fast path. [Issue 275](https://github.com/Kosinkadink/Dinkster/issues/275)
evaluates single-process H3 execution, including head-partitioned attention and
a possible block pipeline that reduces full-copy paging. Single-process
full-copy CFG splitting is a separate candidate recorded on
[issue 266](https://github.com/Kosinkadink/Dinkster/issues/266). These are design
hypotheses, not hosting recommendations or results from this guide.

Current mode guidance:

| Goal and family | Recommendation |
| --- | --- |
| Concurrent jobs, any family | Whole-job replication is supported; measure complete-workflow throughput on the exact host |
| SD1.5 FP16 single-image latency on two SM 8.9 or two SM 12.0 GPUs | Receipted guidance with builtin samplers and schedulers; benchmark against one GPU on the same host |
| H3 single-job latency on two SM 8.9 Ada GPUs | Receipted guidance with builtin samplers and schedulers; benchmark against one GPU on the same host |
| H3 FL2VA BF16 single-job latency on two SM 12.0 Blackwell GPUs | Receipted guidance for CFG, or explicit `sequence` under torch 2.13.0+cu130 with SDPA or comfy-kitchen 0.2.31/0.2.32 INT8 attention; benchmark the exact workload |
| H3 single-job latency on other devices or rank counts | One GPU |
| SDXL or ordinary Flux single-image latency | One GPU; no distributed route is receipted |
| Flux Dev BF16 windowed plans | Receipted window scattering on two or four GPUs; compare to the same serial windowed plan, not full-frame sampling |
| SDXL or Flux host throughput | Whole-job replication; measure its factor |
| Three or more GPUs, ordinary sampling | Whole-job replication; qualify memory and throughput |

Guidance mode can use only the model-evaluating lanes, normally conditional and
unconditional. It is therefore a two-GPU technique for ordinary CFG and does not
scale to three or more GPUs. SD1.5 FP16 has registered production-route
guidance receipts on dual-Ada and dual-Blackwell, with exact encoded latent
bytes at 512x512/20 steps and 1024x1024/30 steps. The withdrawn single-device
proxy receipt does not admit distributed execution. The retained 1.53x-1.56x
guidance timings predate the parity fix and are not performance evidence for
the current receipt. See the `distributed-sd15-fp16-guidance-*` records under
[`dinkster-evidence`](https://github.com/Kosinkadink/dinkster-evidence/tree/main/inference-parity/records) and
[the parity fix](https://github.com/Kosinkadink/Dinkster/pull/278).

Separate [Blackwell guidance measurements](https://github.com/Kosinkadink/Dinkster/issues/302#issuecomment-5349815893)
report 1.974x sampling speedup for H3 BF16 672x384x56 and 1.802x for
1344x768x124, both at CFG2 with bit-exact fused-vs-split results. SD1.5
1280x1280 batch8 reached 1.979x, but fused-vs-split latents differ in both
Dinkster and ComfyUI; this is not bit-exactness against fused one-GPU CFG.
These are historical fit-in-VRAM measurements, not current-head workflow claims.

### Measurement scope of the retained results

The retained results below measure sampling calls, not complete workflows. They
remain valid for sampling comparisons but do not establish cold or warm
total-workflow speedup. New performance arms must report cold total workflow,
warm total workflow, and sampling-only time separately. Headline scaling also
requires a no-offload one-GPU baseline from the same session. Comparisons
against an offloading baseline qualify offload interoperability and crossover,
not the headline multi-GPU speedup. The commands, artifact manifests, raw arrays,
numerical classifications, and interrupted-cell dispositions are indexed on
[issue 264](https://github.com/Kosinkadink/Dinkster/issues/264).

### Consumer dual-Ada qualification

On two RTX 4090 GPUs, peer access was unavailable and NCCL selected
`SHM/direct/direct`. An H3 INT8 two-step sweep bounded the U2R1 crossover:

| Workload | One GPU | U2R1 | Speedup | U2R1 GPU-seconds/sample |
| --- | ---: | ---: | ---: | ---: |
| 32x32, 5 frames | 9.846 s | 10.071 s | 0.978x | 20.142 |
| 672x384, 56 frames | 4.939 s | 16.879 s | 0.293x | 33.758 |
| 1344x768, 124 frames | 76.684 s | 59.843 s | 1.281x | 119.686 |
| 1344x768, 243 frames | 245.406 s | 161.176 s | 1.523x | 322.352 |

Every pair was bit-exact across arms and ranks. The crossover lies between the
56-frame and 124-frame workloads at these geometries. This sweep uses only two
sampling steps to isolate transport and attention scaling; it is not a
production-latency recommendation. U2R1 also consumes more total GPU time at
every point. Its measured latency reduction at larger workloads motivates
future route work but is not deployment guidance. Every geometry used separate
fresh one-GPU and U2R1 processes with matching prompt, conditioning, seed,
receipt, and steps; no one-GPU result was reused across geometries. Each sweep
workload ran one GPU before U2R1, so the sweep was not order-balanced.

Ulysses shards sequence rows and attention heads, not model parameters. Every
rank owns a logical full H3 INT8 weight copy of 34,038,766,144 bytes (31.701
GiB), so a 24 GiB card must page part of that copy. A short one-GPU receipt
reported 7,416,695,872 loaded and 26,622,070,272 offloaded weight bytes. The
two-step sweep did not record per-rank residency. Separately verified 20-step
artifacts recorded:

| Arm/rank | Loaded weights | Offloaded weights | Torch peak allocated |
| --- | ---: | ---: | ---: |
| One GPU | 12,617,632,832 B | 21,421,133,312 B | 6,386,510,336 B |
| U2 rank 0 | 13,221,612,608 B | 20,817,153,536 B | 6,125,241,856 B |
| U2 rank 1 | 13,657,820,224 B | 20,380,945,920 B | 5,915,069,440 B |

These values are residency and torch-allocation counters, not peak board memory
or cumulative PCIe traffic. Neither evidence set records page faults, transfer
bytes, or transfer time. The timing crossover cannot apportion cost between SHM
collectives, weight paging, synchronization, and kernel work without new
instrumentation.

The verified consumer bottleneck facts are narrower: CUDA peer access is
unavailable, NCCL uses host-backed `SHM/direct/direct`, small U2R1 workloads
lose to the fixed distributed cost, and every measured U2R1 cell consumes more
GPU-seconds than one GPU. The receipts do not separately time synchronization,
communication, paging, or compute/communication overlap, so non-overlapped work
remains a profiling question rather than an established cause.

Separate raw artifacts independently verify a standard 1344x768, 124-frame
pair with 20 sampling steps:

| Arm | Sampling median | Sampling calls/hour (extrapolated) | GPU-seconds/sample |
| --- | ---: | ---: | ---: |
| One-GPU INT8 offload | 764.634 s | 4.71 | 764.634 |
| INT8 U2R1 | 576.304 s | 6.25 | 1152.608 |

The U2R1 output was bit-exact across ranks, repeats, and the one-GPU arm. It
reduced latency by 24.6% for a 1.327x speedup while consuming 1.507x the total
GPU time. On this no-P2P consumer host, the internal U2R1 arm reduced sampling
latency only at larger workloads and did not reach the 1.5x attractive floor in
this production-length cell. Because the route is not exposed and the baseline
required offloading, this is secondary offload-interaction evidence rather than
a deployment recommendation or headline scaling result.
U2R1 ran first to counterbalance the sweep, but an external claimant interrupted
the window before the one-GPU arm, which completed in a later clean slot. This
pair is therefore not a same-session headline comparison. Do not confuse it
with the 76.684 s versus 59.843 s standard cell above, which used only two
sampling steps and establishes crossover behavior only.

Whole-job replication is different: each process runs an independent job with
a complete model copy and no per-step rank gather. On 24 GiB cards, suitable
no-offload candidates include SD1.5, separately loaded SDXL base or refiner,
and only Flux artifact/dtype combinations whose measured complete copy plus
activation headroom fits. No dual-RTX-4090 replication factor or no-offload
memory receipt was completed here, so qualify the exact artifact before making
a throughput claim. Smaller H3 geometry does not reduce its fixed weight copy.

### Blackwell workstation qualification

MiniMax H3 BF16 sequence parallelism is sensitive to sequence length. On four
96 GiB RTX PRO 6000 Blackwell GPUs, a 1344x768, 124-frame, 20-step T2VA workload
measured as follows after the compacted flash-attention fix:

| Arm | Sampling median | Speedup | Sampling calls/hour (extrapolated) | GPU-seconds/sample |
| --- | ---: | ---: | ---: | ---: |
| One GPU | 208.179 s | 1.000x | 17.29 | 208.179 |
| Ulysses 2 | 142.957 s | 1.456x | 25.18 | 285.915 |

The distributed outputs were bit-exact against the one-GPU arm. Two-way
Ulysses was faster than one GPU for this workload. The 1.456x U2 result is below
the 1.5x attractive floor, so it is not deployment guidance. Degree 3 is not
legal because H3 has 56 attention heads, which is not divisible by 3.

A later full-residency run of this workload at source head `7ad9f33c` measured
203.7 s on one GPU and 132.3 s on Ulysses 2, a 1.54x speedup, with one identical
packed float32 fingerprint across every repeat and both distributed ranks. That
run is the evidence for the registered SM 12.0, world-size-2, Ulysses-2 built-in
SDPA receipt. The receipt admits only the measured provider, torch version,
model storage, dtype, rank geometry, and device environment. The explicit
product mode exposes only that receipted geometry.

The same production workload was also qualified with BF16 model weights and
comfy-kitchen 0.2.31 INT8 attention. Ulysses 2 produced bit-identical float32
packed outputs to the same INT8 attention provider on one GPU across a warmup
and three measured repeats, including identical outputs on both distributed
ranks. The correctness cell measured 156.7 s on one GPU and 122.2 s on Ulysses
2, a 1.28x speedup. The exact SM 12.0, world-size-2, Ulysses-2 receipt is
registered, but the result remains below the 1.5x attractive floor and is not a
performance recommendation. Both arms ran concurrently on disjoint GPUs of a
shared host, so these timings are correctness-cell context and are not comparable
to the table above. Ring stays excluded because the INT8 provider does not return
the log-sum-exp values required by the ring merge. The receipt pins
comfy-kitchen 0.2.31 and torch 2.13.0+cu130.

A separate [comfy-kitchen 0.2.32 receipt](https://github.com/Kosinkadink/dinkster-evidence/blob/main/inference-parity/records/distributed-minimax-h3-bf16-kitchen032-int8-sequence-u2-sm120-d2/README.md)
covers that provider under torch 2.13.0+cu130 with the same model, dtype,
device capability, and U2R1G1 geometry. Sequential full-ResidentWeights arms
with synthetic 32-token conditioning produced 12 bit-identical packed outputs
across warmup, three repeats and both ranks. Warm sampling medians were
157.039 s on one GPU and 109.298 s on U2, a 1.437x speedup, below the 1.5x
attractive floor. Peak allocated memory was 66.52 GiB single-GPU and 65.96 GiB
per U2 rank; weights remain replicated. This excludes text encoding, decode,
Aimdo, offload, cache and schedules, and is not a complete-workflow performance
or dense-SDPA quality claim. Different topology and concurrency prevent a
direct 0.2.31 speed comparison. Approximate attention remains explicit opt-in.

These are workstation Blackwell measurements, not consumer conclusions.
Architecture-specific receipt admission must match the measured compute
capability; do not transfer an Ada receipt to Blackwell or a Blackwell
measurement to an Ada deployment.

A longer-sequence diagnostic at 1536x864, 124 frames, and five steps measured
two-way Ulysses at 49.556 s against 76.788 s on one GPU, a bit-exact 1.550x
speedup. Four-way Ulysses was deterministic but had relative RMSE 0.012925
against the one-GPU arm, exceeding the fixed 0.01 stop threshold. Four-way
Ulysses is quarantined and its timing is not recommendation evidence. There is
no world-size-4 numerical receipt; the explicit product mode refuses it.
Re-measure and numerically qualify the intended sequence length instead of
assuming that a higher degree is better. No concurrent 2xU2 or four-replica
cell completed in this artifact, so do not infer throughput from sequential
latency.

A later concurrent 2xU2 attempt completed only two of five timed calls per job:
152.999593 and 160.030005 s on GPUs 0/1, and 172.228238 and 181.904540 s on
GPUs 2/3. Completed outputs were bit-exact, but the arm has no accepted median,
spread, workflow wall, jobs/hour, or GPU-seconds result. Four independent
replicas did not run.

Receipt audit found no world-size-4 H3 receipt. H3 guidance receipts are keyed
to world size 2 and reject a world-size mismatch. U4 used the direct internal
`sequence` environment path. The explicit production mode now derives the same
pure-Ulysses geometry, but no world-size-2 receipt admits U4 and the pre-group
gate refuses it. The internal path still lacks a world-size-4 numerical gate.
H3's 56 heads divide legally into 14 heads per U4 rank. For packed sequence
48,375, equal-chunk planning pads only the last shard by one row to 48,376;
U2 at the same length also has one padded tail row. Padding therefore needs
seam instrumentation rather than assumption as the divergence cause.

### Ring viability

Ring and Ulysses x Ring plans require a block-statistics attention kernel at
plan compilation. The production H3 providers supply tensor kernels, so these
plans fail closed before consensus on main. The internal float32 exchange
implementation is not a supported runtime route and has no product receipt.
The experimental fused provider on
[draft PR 1238](https://github.com/Kosinkadink/Dinkster/pull/1238) has operator-only
PCIe measurements, not a real-model speed claim or a hosting recommendation.
The [two-GPU comparison](https://github.com/Kosinkadink/Dinkster/issues/1266)
tracks NVLink evidence separately.

## Start with an isolated device list

Use `CUDA_VISIBLE_DEVICES` to hide reserved GPUs before choosing logical device
indices. For example, this makes physical GPUs 1 and 2 visible as logical GPUs
0 and 1:

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

Record whether each pair is `PIX`, `PXB`, `PHB`, `NODE`, `SYS`, or an NVLink
path. Also verify peer access in the exact torch environment used by Dinkster:

```python
import torch

for source in range(torch.cuda.device_count()):
    for destination in range(torch.cuda.device_count()):
        if source != destination:
            print(source, destination, torch.cuda.can_device_access_peer(source, destination))
```

All ordered GPU pairs on the measured AMD Threadripper PRO host reported
`NODE`, had no NVLink, and supported peer access. Raw directed copies reached
24.3-26.4 GB/s, but four simultaneous exchanges reached much less. A pairwise
bandwidth result is not a four-rank collective result.

The measured AMD host used one NUMA node on a Threadripper PRO 5995WX. Its
kernel command line contained `iommu=pt` and no ACS override. The four GPUs were
in separate IOMMU groups. These are receipt facts, not universal requirements:
do not disable IOMMU or force an ACS override merely to copy this host. First
record the consumer board's actual groups, topology, peer access, and transport.
If peer access is false or NCCL selects NET/Socket unexpectedly, inspect BIOS
IOMMU settings, PCIe bifurcation, ACS behavior, and driver logs before changing
the application.

The measured dual-RTX-4090 host reported a `PHB` path on one NUMA node, separate
IOMMU groups, and no explicit IOMMU kernel override. CUDA peer reads and writes
were unavailable for the chipset, and NCCL selected shared-memory channels.
That is a valid operating configuration, but its fixed host-memory exchange
cost makes small U2R1 workloads slower. Do not force P2P when CUDA reports that
the pair cannot support it. The measured device UUIDs were
`GPU-666d1242-9c20-341c-73ea-e63770947451` and
`GPU-5ac69527-f5f0-f6f0-1d46-24c6f401cdc6`; pin UUIDs rather than relying on
device indices when retaining a host receipt.

## NCCL transport selection

`NCCL_P2P_LEVEL` controls how far across the PCIe topology NCCL may use direct
GPU peer access. `NCCL_P2P_LEVEL=PHB` permits P2P across a PCIe host bridge.
That setting is not universally faster.

On the measured AMD host, `PHB` hurt four-rank Ulysses all-to-all at the
production payload:

| Four-rank all-to-all path | Transport | Bandwidth |
| --- | --- | ---: |
| NCCL default | SHM/direct/direct | 3.93 GB/s |
| `NCCL_P2P_LEVEL=PHB` | P2P/CUMEM | 1.38 GB/s |
| `NCCL_P2P_DISABLE=1` | SHM/direct/direct | 3.96 GB/s |

Dinkster therefore applies the AMD-host `PHB` default only to eligible 3+ rank
non-sequence modes. It does not set the override for two ranks or for
sequence-parallel Ulysses, Ring, or hybrid meshes. If the caller explicitly
sets `NCCL_P2P_LEVEL`, Dinkster preserves that value. An explicit environment
setting always wins, so remove old global tuning before trusting auto-config.

Do not force `NCCL_SHM_DISABLE=1` as a substitute. It selected NET/Socket and
measured only 1.43 GB/s in the same four-rank all-to-all test.

### Verify the selected transport

Start one representative fresh process with NCCL logging enabled:

```sh
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH dinkster-serve ... 2>&1 | tee nccl.log
```

Inspect the channel lines, not only the topology table:

```sh
grep -E 'via (P2P/CUMEM|SHM/direct/direct|NET/Socket)' nccl.log
```

Examples:

```text
Channel 00/0 : 0[1] -> 1[2] via P2P/CUMEM
Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
```

For two-GPU Ulysses on the measured host and payload, P2P/CUMEM was the faster
setting. For four-GPU Ulysses, SHM/direct/direct was faster. Treat an unexpected
transport as a qualification warning until a controlled A/B measures its effect.

## Verify the attention fast path

Sequence sharding can change the attention kernel even when the model and dtype
do not change. H3 Ulysses compacts valid sequence rows before attention so
padding does not create a boolean mask and select the CUTLASS memory-efficient
fallback. That fallback makes degree-normalized attention about 2.9 times more
expensive than PyTorch flash attention on the measured workload.

Use a short, separate Nsight Systems capture after the timed benchmark. Do not
profile the runs used for latency numbers:

```sh
nsys profile --trace=cuda,nvtx,osrt --sample=none -o dinkster-multigpu \
  dinkster-serve ...
nsys stats --report cuda_gpu_kern_sum dinkster-multigpu.nsys-rep
```

For the measured H3 BF16 Ulysses route, verify that the kernel summary contains
`pytorch_flash::flash_fwd_kernel` and no `fmha_cutlassF` attention fallback. If
the fallback appears, check for an attention mask, unsupported dtype, head
dimension, causal or grouped-query setting, and backend capability before
tuning NCCL. Faster transport cannot compensate for a slower attention family.

## Process and memory hygiene

Benchmark fresh detached processes. A warm resident process can retain models,
CUDA contexts, NCCL communicators, pinned host buffers, and allocator state from
an earlier arm. Start from a fresh, clean measurement checkout and retain its
exact commit and status. Before and after every arm:

1. Capture claim and release gauges for every selected GPU, including compute
   PIDs, and stop before launch if any selected GPU is occupied.
2. Record `MemAvailable`, swap totals, GPU UUIDs, memory use, utilization, and
   driver version.
3. Run the complete arm in a bounded systemd scope.
4. Verify `memory.events` has zero `max`, `oom`, and `oom_kill` events.
5. Verify that no compute PID remains on any selected GPU.

Check kernel logs after every arm. An NVIDIA Xid 79, Xid 154, unqueryable GPU,
or changed device inventory invalidates the arm, even when some timed calls
completed. Stop every related scope, retain the incomplete calls only as
diagnostic evidence, and do not publish a median or throughput claim until the
host has recovered and a fresh five-repeat arm completes.

The RipperPC evidence used:

```sh
systemd-run --user --scope \
  -p MemoryAccounting=yes \
  -p MemoryMax=480G \
  -p MemorySwapMax=0 \
  ./run-arm.sh
```

`MemorySwapMax=0` prevents the scope from hiding pressure in swap. Size
`MemoryMax` for the host instead of copying 480 GiB to smaller systems. Leave
headroom for the OS and stop before launching when available memory is below the
declared requirement. The measured host retained system swap for unrelated
processes; swap was disabled inside each benchmark scope. No manual CPU affinity,
NUMA pinning, `CUDA_LAUNCH_BLOCKING`, or NCCL tuning variable was supplied to the
reported post-fix latency cells. The consumer H3 scopes used `MemoryMax=112G`
and `MemorySwapMax=0`, with `CUDA_LAUNCH_BLOCKING` unset and clean compute-PID
gauges before claim and after release.

For each new performance arm, record the cold complete-workflow wall from
invocation before first load and conditioning through the first complete output.
Treat that cold execution as the discarded warmup for warm statistics, then
record five warm complete-workflow walls, with the sampling portion timed inside
each matching repeat. Run the one-GPU host
baseline in the same measurement session, balance arm order when thermal or
clock drift may matter, and report both warm medians and min-max spreads. Save
every per-repeat total and sampling wall rather than only the summaries.

Every distributed arm also needs a numerical classification against the
single-GPU output from the same source pin, model artifacts, prompt, geometry,
sampler, schedule, seed, and step count. Require bit-identical repeats within an
arm and across ranks. A cross-arm difference is accepted only under an existing
family-specific signature and stop threshold; otherwise stop and instrument the
first divergent seam.

## Pin and record the environment

Keep the source commit, model artifact SHA-256 values, and runtime versions with
the receipt. The RipperPC matrix above used:

```text
NVIDIA driver 595.84
Python 3.13.12
torch 2.10.0+cu130
CUDA 13.0
NCCL 2.28.9
comfy-aimdo 0.4.13
comfy-kitchen 0.2.31
```

The consumer dual-Ada sweep used driver 595.84 with maximum CUDA 13.2, Python
3.12.3, torch 2.13.0+cu130, and NCCL 2.29.7+cuda13.2.

Do not assume newer drivers or NCCL preserve transport selection. Repeat the
transport proof and a representative latency cell after upgrading the driver,
torch, CUDA, NCCL, comfy-aimdo, or comfy-kitchen.

## Hosting records and checklists

For managed hosting, retain enough information to reproduce both the fast path
and its failure boundaries:

- Pin the Dinkster source, model artifact digests, driver, torch, CUDA, and NCCL.
- Capture GPU UUIDs, PCIe topology, NVLink and peer-access state, IOMMU groups,
  kernel IOMMU/ACS options, and the selected NCCL channel transport.
- Record inherited NCCL variables and any CPU or NUMA affinity. The reported
  results used no manual affinity or pinning.
- Prove the attention kernel family in a separate short profile.
- Use fresh processes, bounded memory, no scope swap, and clean before/after
  gauges. Retain cgroup events and driver logs with each run.
- Compare one-GPU latency, distributed latency, total GPU-seconds, and
  whole-job replica throughput. Require a same-session host baseline, five
  timed calls, median and spread, and a numerical classification before
  publishing a cell.

For an end-user consumer board, start with observation rather than copied host
tuning:

- Reserve GPUs with `CUDA_VISIBLE_DEVICES` and verify logical-to-physical
  mapping, topology, peer access, and the actual NCCL channel lines.
- Remove inherited NCCL overrides for the baseline. Do not force P2P, disable
  IOMMU, or add an ACS override merely because another host used it.
- Use whole-job replication as the supported multi-GPU option unless the exact
  family and workload have a receipted single-job route that wins the required
  latency target; measure the replication factor before making a throughput claim.
- Repeat one representative latency and numerical cell after changing the
  driver, torch, CUDA, NCCL, model artifact, or Dinkster source.

For either class of host, retain raw logs, tensors or fingerprints, commands,
per-repeat walls, cgroup events, and a SHA-256 manifest for the receipt.
