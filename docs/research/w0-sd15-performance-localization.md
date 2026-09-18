# W0 canonical SD1.5 performance localization

Status: localized on 2026-07-29. No production or harness behavior, golden,
research slice.

The two remaining canonical reds have one shared source and one measurement
boundary issue:

- Dinkster materializes every planned tensor through a separate `file.read()` and
  `bytearray()` allocation. Reading and heap-backing the 2.132 GB payload takes
  1.478 seconds and leaves fragmented libc arenas resident after the CPU
  tensors move to CUDA.
- The adapters do not put the same work inside `cold_load_ns`. Dinkster transfers
  diffusion, CLIP-L, and VAE before ending cold load. ComfyUI transfers only
  CLIP there, then transfers diffusion and VAE inside warmup generation.

The reader dominates cold time. Its retained free heap blocks dominate the
real-phase RAM excess. comfy-aimdo, pinned host arenas, live CPU model tensors,
the codec lifecycle, and checkpoint mmap/page-cache accounting are not the RAM
cause in this workload.

## Pins, protocol, and evidence conditions

- The fresh Dinkster clone was at
  `786e8bc89a3fedd93ac98acf99c3815f24abbe0b`; its clone reflog, `HEAD`, and
  `origin/main` agree, and the worktree was clean. The first baseline-capture
  command accidentally queried the station13 wrapper repository; the corrected
  proof used the clone explicitly before any evidence run. This did not change
  either checkout.
- ComfyUI was the required
  `f4b99bc62389af315013dda85f24f2bbd262b686`, read-only and clean before and
  after execution.
- Checkpoint: `v1-5-pruned-emaonly-fp16.safetensors`, 2,132,696,762 file bytes,
  2,132,494,772 tensor payload bytes, sha256
  `e9476a13728cd75d8279f6ec8bad753a66a1957ca375a1464dc63b37db6e3916`.
- Canonical workload: seed `685468484323813`; Euler/normal; 20 steps; CFG 8;
  denoise 1; 512x512, batch 1; diffusion fp16, text fp32, codec fp32.
- Device: `GPU-666d1242-9c20-341c-73ea-e63770947451`, RTX 4090. Both adapters
  used `/home/kosin/ComfyUI/venv/bin/python`, torch `2.9.1+cu130`.
- The same fresh process performed one warmup then one real request per engine.
  Both instrumented engines produced the canonical raw tensor sha256
  `24bb0dd6fac42f49f4a5bfd9d27bd61692ac2bb202ce45c7b80fd74f542ae65b`
  and `.npy` sha256
  `799266ed84723a70c28874de478b867157e0e406df36d7cfb64a020fd2564d2c`
  in both phases.
- Every accepted launch followed an explicit coordinator all-clear and an
  immediately preceding fail-closed precheck. Each accepted precheck found no
  other pytest, inference-parity, or dinkster-workers process and zero NVIDIA
  compute processes. CPU, Dinkster, and ComfyUI MemFree values were 8,486,092,
  8,040,560, and 6,694,992 KiB, respectively, all above the 2,097,152 KiB pin
  floor. MemAvailable values were 124,202,052, 123,757,428, and 124,149,200
  KiB.

The harness never clears the OS page cache. The CPU and Dinkster profiles ran
before the ComfyUI profile. Two attempted ComfyUI launches correctly aborted
before execution: the first saw a crossing partner-node pytest process; the
second saw MemFree/`SC_AVPHYS_PAGES` at only 1,348,480 KiB. To restore the
required floor, the coordinator allocated, touched, and released 6 GiB of
anonymous memory (no `drop_caches`), forcing ordinary kernel cache reclaim.
The final ComfyUI run then passed its precheck. That intervention may have
changed checkpoint page-cache warmth, so the timing tables report each
engine's observed read/materialization phases and do not assume symmetric disk
state. Dinkster's independent CPU and GPU payload-read totals were 1,447.9 and
1,478.1 ms, a 2.1% difference.

Instrumentation was process-local and external to both repositories. Timers
wrapped the loader's ordered seams; 10 ms RSS sampling and `/proc` snapshots
captured memory ownership. Snapshot time was measured and subtracted from each
cold total. The adjusted totals, 2,047.8 ms for Dinkster and 623.9 ms for ComfyUI,
are within 0.4% and 1.6% of the committed canonical 2,040.0 and 634.0 ms.

## Defect 1: cold load

### Canonical observation

| Metric | ComfyUI | Dinkster | Limit | Result |
| --- | ---: | ---: | ---: | --- |
| `cold_load_ns` | 634,019,298 | 2,040,002,937 | 729,122,192.7 | Dinkster is 3.22x baseline and 1.311 s over limit. |
| Warmup `generation_ns` | 1,303,192,807 | 715,592,478 | 1,498,671,728.05 | Dinkster is faster; the red is confined to load accounting. |
| Real `generation_ns` | 507,945,662 | 520,053,955 | 584,137,511.3 | Comparable and passing. |

### Dinkster ordered attribution

The adjusted 2,047.8 ms total removes 271.9 ms spent taking memory snapshots.
Nested entries are shown as subrows and are not added twice.

| Ordered seam | Time (ms) | Evidence and ownership |
| --- | ---: | --- |
| Initial strict header parse | 6.2 | `load_safetensors_header` before `load_runtime`. Header-only. |
| `load_runtime` total | 1,602.5 | Probe, second plan, assembly, and runtime construction. |
| - probe plus load planning | 13.6 | Probe 6.2; second load plan 7.4. The second planning pass is intentional and cheap. |
| - three `load_tensors` calls | **1,478.1** | Diffusion 1,252.7; CLIP-L 147.1; VAE 78.3. Reads 2,132,470,614 bytes through 1,130 selected tensor allocations. Includes 15.0 ms of three repeated strict header parses. |
| - module construct, assign, component bookkeeping | 78.0 | Constructors 59.7; strict `assign=True` state loading 15.8; mapping/bookkeeping 2.5. |
| - runtime construction | 32.2 | SD runtime, tokenizer, sampler, and scheduler setup. |
| Diffusion host-to-device | 197.4 | 1,719,041,928 module bytes. |
| CLIP-L host-to-device | 34.6 | 248,480,256 module bytes. |
| VAE host-to-device | 21.7 | 167,307,726 module bytes. |
| Positive text encode | 54.9 | First CLIP forward and first-use CUDA kernels. |
| Negative text encode | 6.1 | Same CLIP forward after caches are warm. |
| Latent allocation | 1.4 | CUDA `[1,4,64,64]` float32. |
| Unwrapped first-use/orchestration residual | 122.8 | Principally first CUDA context initialization at the initial pre-transfer synchronize; also sub-millisecond wrapper gaps. Rows are independently rounded. |
| **Adjusted cold total** | **2,047.8** | Matches the canonical 2,040.0 ms. |

Payload ingestion alone is 72.2% of Dinkster cold load. The implementation at
`packages/dinkster-inference-torch/src/dinkster_inference_torch/sources.py:93-120`
re-parses the header once per component, then for each tensor performs
`file.seek`, `file.read`, and `bytearray(data)`. The bytes object returned by
`read` and the bytearray therefore coexist during each copy. Components are
loaded serially by
`packages/dinkster-inference-torch/src/dinkster_inference_torch/assemble.py:348-380`.

The checkpoint contains many heap-fragmenting allocations: diffusion has 686
tensors (406 below 128 KiB), CLIP-L has 196 (123 below 128 KiB), and VAE has
248 (184 below 128 KiB). The payload byte split is 1,719,041,928 diffusion,
246,120,960 CLIP-L, and 167,307,726 VAE.

### Where ComfyUI spends 623.9 ms

The adjusted total removes 336.3 ms of snapshot time. The coordinator cache
reclaim makes direct disk-warmth assumptions invalid, but the observed seams
still identify the algorithmic difference: `safe_open` exposes mapped tensor
views, while Dinkster creates a bytes object and heap copy per tensor.

| Ordered seam | Time (ms) | Evidence and ownership |
| --- | ---: | --- |
| Checkpoint loader total | 458.2 | Whole CPU checkpoint construction. |
| - `safe_open`/1,145 tensor views | 15.5 | `get_tensor` itself totals 13.4 ms for all 2,132,494,772 payload bytes. These are mmap-backed views, not 2.132 GB of Python heap copies. |
| - inspection | 10.4 | UNet prefix 0.1; model config 10.2. |
| - diffusion construction/load | 169.6 | Includes 169.0 ms `load_state_dict` copy from mapped source. |
| - VAE construction/load | 148.5 | Includes 28.7 ms state load. |
| - CLIP construction/load | 77.2 | Includes 28.7 ms state load. |
| - remaining assembly/orchestration | 37.0 | State-dict slicing, dtype policy, wrappers, and loader return. |
| Positive text encode | 156.3 | Includes the only cold-phase model transfer: 46.3 ms for CLIP-L, then first-use context/kernels and encode. |
| Negative text encode | 9.3 | CLIP is already resident. |
| Latent allocation | 0.1 | CPU latent, as owned by `EmptyLatentImage`. |
| **Adjusted cold total** | **623.9** | Matches the canonical 634.0 ms. |

ComfyUI uses `safetensors.safe_open` at `comfy/utils.py:110-129` and drains
component keys from the mapped state dict while constructing the model, VAE,
and CLIP at `comfy/sd.py:1508-1613`. No checkpoint mapping remained after the
loader returned. The sampled cold RSS did see a 5.191 GB transient while mapped
source pages and destination weights overlapped; the real phase does not.

### Boundary mismatch and first use

Dinkster's adapter moves all three modules at
`tools/inference_parity/dinkster_adapter.py:52-55` before ending cold load.
ComfyUI's adapter ends cold load after text encode at
`tools/inference_parity/comfyui_adapter.py:92-100`; upstream lazy residency
moves diffusion and VAE only when sampling and decode call
`load_models_gpu`. In the instrumented run, Dinkster charged 253.7 ms of all-model
transfer to cold. ComfyUI charged only 46.3 ms of CLIP transfer there and
deferred 308.2 ms of diffusion plus 70.3 ms of VAE transfer to warmup
generation. This accounts for 207.4 ms, or 14.6%, of the instrumented cold
gap. It does not explain the red alone: removing Dinkster diffusion and VAE
transfer still leaves roughly 1.83 seconds.

First-use generation confirms that nothing is hidden in the real phase:

| Engine | Warmup sample + decode (ms) | Real sample + decode (ms) | First-use premium (ms) |
| --- | ---: | ---: | ---: |
| Dinkster | 717.5 | 528.4 | 189.1 |
| ComfyUI | 1,262.9 | 504.8 | 758.1 |

ComfyUI's premium includes the deferred 378.5 ms model transfers. Dinkster's
warm generation remains faster even after doing all transfers during cold.
No torch compilation occurred in either adapter. The remaining premium is
first-use CUDA library/kernel/cache setup.

### Dominant cause and implementation-ready remediation

The dominant cold cause is Dinkster's 1,608.7 ms CPU checkpoint-load stage, of
which the payload reader is 1,478.1 ms, versus a 458.2 ms complete ComfyUI
checkpoint load. That produces 1,150.5 ms of the 1,423.9 ms instrumented engine
gap. The transfer boundary contributes another 207.4 ms.

Repair the source rather than adding a trim or timing exception:

1. In
   `packages/dinkster-inference-torch/src/dinkster_inference_torch/sources.py:77-121`,
   replace per-tensor `read()` plus `bytearray()` with one file-backed,
   lifetime-owned copy-on-write mmap and `torch.frombuffer` views over planned
   offsets. Copy-on-write preserves the existing "safe to mutate, never write
   the checkpoint" contract while avoiding eager payload copies. Tensor owners
   must keep the mapping alive after `load_tensors` returns, and the mapping
   must release when the last CPU source tensor is replaced or freed.
2. Preserve strict header validation, little-endian refusal, selected-key
   behavior, dtypes, shapes, `assign=True`, transforms, quant handling, and
   state identity. Add focused coverage in
   `packages/dinkster-inference-torch/tests/test_sources.py` for lifetime after
   return and GC, copy-on-write mutation without file mutation, selected and
   empty tensors, and mapping release. Keep assembly proofs in
   `packages/dinkster-inference-torch/tests/test_assemble.py`.
3. After the reader meets the existing budget, align the harness boundary in
   `tools/inference_parity/dinkster_adapter.py`: leave CLIP transfer before text
   encode, but defer diffusion and VAE transfer to the first measured warmup
   generation, matching ComfyUI. This is calibration, not the performance
   repair, and must not change a threshold.
4. Re-run these seam timers and regenerate the unchanged canonical gate in a
   fresh uncontended window. Require exact outputs and both existing budgets;
   do not infer acceptance from the causal measurements here.

## Defect 2: real-phase RAM

### What the harness measures

`tools/inference_parity/harness.py:223-246` sums `VmRSS` for the adapter process
and every descendant. A 50 ms sampler at lines 290-296 records an absolute
per-phase maximum. For these adapters there were no child processes.

`VmRSS` includes resident anonymous pages, resident file mappings, and resident
shared-memory pages. It is not live Python object size and not allocator
"bytes in use". It excludes ordinary checkpoint pages that exist only in the
global OS page cache, but includes those pages while they are mapped into and
resident in the process. It also excludes CUDA VRAM, which the harness samples
separately. Because one process survives warmup into real, real RSS includes
allocator pages retained from cold and warmup even if no live object owns
their contents.

### Phase-attributed RSS

The instrumented real peaks reproduce the records: Dinkster is 9.3 MB above its
2,493,513,728-byte canonical value (0.37%); ComfyUI is 12.9 MB above its
1,927,663,616-byte canonical value (0.67%).

| Phase snapshot | Dinkster RSS (bytes) | ComfyUI RSS (bytes) | Interpretation |
| --- | ---: | ---: | --- |
| Imports | 567,492,608 | 859,774,976 | Different framework/module import baselines. |
| CPU checkpoint loaded | 2,746,408,960 | 3,196,485,632 | Live CPU weights; ComfyUI has already released its checkpoint mapping at the recorded post-load snapshot. |
| Cold complete | 2,420,695,040 | 3,197,706,240 | Dinkster modules are all CUDA-resident but free heap arenas remain mapped. ComfyUI has moved only CLIP. |
| Warmup after sample | 2,497,892,352 | 2,218,291,200 | ComfyUI has moved diffusion and releases most CPU backing; Dinkster retained arenas remain. |
| Warmup complete | 2,502,795,264 | 1,929,617,408 | Both have sampled and decoded once. |
| Real after sample | 2,502,795,264 | 1,929,650,176 | No Dinkster real-phase rise; the high value predates real. |
| Real complete | **2,502,795,264** | **1,940,553,728** | Snapshot delta 562,241,536 bytes; canonical delta 565,850,112 bytes. |
| Sampled real peak | **2,502,795,264** | **1,940,566,016** | Profile peak delta 562,229,248 bytes. |

The Dinkster real-phase value is flat across sample, decode, output copy, and GC.
It is a persistent-process high residence level, not a real-request leak.

### Real ownership and retention

| Ownership at real complete | Dinkster (bytes) | ComfyUI (bytes) | Difference |
| --- | ---: | ---: | ---: |
| `RssAnon` | 2,040,987,648 | 1,420,976,128 | +620,011,520 |
| `RssFile` | 442,929,152 | 500,695,040 | -57,765,888 |
| `RssShmem` | 18,878,464 | 18,882,560 | -4,096 |
| libc allocated (`uordblks`) | 1,178,947,008 | 1,145,777,472 | +33,169,536 |
| libc free arena blocks (`fordblks`) | **650,162,752** | **18,240,192** | **+631,922,560** |
| Large mmap allocations (`hblkhd`) | 7,905,280 | 102,584,320 | -94,679,040 |
| RSS drop from post-real `malloc_trim(0)` | **647,843,840** | **15,794,176** | **+632,049,664** |

`gc.collect()` did not lower either RSS. A subsequent causal
`malloc_trim(0)` left live objects untouched but lowered Dinkster RSS from
2,502,795,264 to 1,854,951,424 bytes. The 647.8 MB RSS drop matches the 650.2
MB free-block inventory and is greater than both the canonical 565.9 MB
baseline delta and the 276.7 MB budget overrun. ComfyUI dropped only 15.8 MB.

An independent CPU-only causal run is stronger evidence that assembly creates
the retained arenas. It loaded the runtime to 2,748,133,376-byte peak RSS,
deleted the runtime and source, and ran GC. RSS remained 2,287,218,688 bytes.
Only `malloc_trim(0)` lowered it to 594,534,400 bytes, releasing 1,692,684,288
resident anonymous bytes. During the real GPU run, first-use libraries reuse
some of those free blocks; 650 MB remains free but resident.

### Ruled-out owners

- **Live model duplicates:** Dinkster's module traversal found diffusion
  1,719,041,928 bytes, CLIP-L 248,480,256 bytes, and VAE 167,307,726 bytes,
  all on `cuda:0`; zero module tensor bytes remained on CPU. ComfyUI likewise
  had all three modules on CUDA by real. Header mappings contain geometry only,
  not payload duplicates.
- **Pinned host / aimdo:** The canonical Dinkster adapter imports torch before
  any possible aimdo bootstrap and never enrolls a residency mechanism.
  `probe_aimdo` was importable but `initialized=false`; pinned-host owner count,
  `TOTAL_PINNED_MEMORY`, and bound residency-module count were all exactly
  zero. ComfyUI dynamic VRAM and its aimdo allocator were also false/unset.
  Thus the 2 GiB floor at
  `packages/dinkster-inference-torch/src/dinkster_inference_torch/pinned_host.py:19`
  and the doubled host-buffer sizing at lines 96-100 allocated no arena in
  this workload. The floor mattered to evidence admission, not measured RSS.
- **Cast arenas:** No aimdo mechanism means no cast arena was constructed.
- **Codec/VAE lifecycle:** The Dinkster VAE remained a single CUDA module; no CPU
  VAE tensors or new real-phase allocation appeared. ComfyUI's fp32 VAE is
  actually larger on CUDA (334,615,452 bytes), so codec ownership cannot
  explain Dinkster's host excess.
- **Checkpoint mmap/page cache:** Dinkster payloads are anonymous bytearrays, not
  mmap views. ComfyUI's source mapping was gone after checkpoint load and was
  absent in every warmup/real snapshot. Global page cache is outside `VmRSS`.
- **Application caches:** Output and tokenizer caches are small; real RSS was
  unchanged before and after the identical request. CUDA allocated/reserved
  values are VRAM and gate separately.

### Dominant cause and implementation-ready remediation

The dominant RAM cause is allocator retention created by the per-tensor
bytes-to-bytearray reader. It explains the entire budget overrun without any
live-object leak.

Use the same copy-on-write mmap reader remediation specified for cold load.
It removes the thousands of payload heap allocations and lets source pages
leave RSS when the last CPU view is replaced by the CUDA parameter. Verify the
repair with the same `/proc/self/status`, `smaps_rollup`, and `mallinfo2`
snapshots: after warmup, free arena blocks should no longer carry hundreds of
megabytes and post-real trim should be immaterial. Then regenerate the
canonical record under the unchanged budget.

Do **not** add `malloc_trim(0)` to the adapter or production inference path.
That process-global, libc-specific workaround happens to prove ownership but
masks the fragmented allocation source, adds unpredictable latency, and does
not improve the 1.478-second reader. Do not change the RAM budget or reinterpret
RSS as live-object bytes; persistent-process allocator retention is exactly
what the existing metric observes.

## Repair decision

No repair landed in this slice. A correct mapped-reader change alters payload
storage ownership, mutability, and lifetime and needs the focused correctness
coverage named above. A trim call would be narrow but wrong. The adapter
boundary calibration is safe only after the real reader defect is repaired,
because moving 219 ms out of cold does not make the current 1.478-second reader
acceptable.

Accordingly, the canonical record was not regenerated. The committed record at
[`w0-harness-sd15/verdict.json`](https://github.com/Kosinkadink/dinkster-evidence/blob/main/inference-parity/records/w0-harness-sd15/verdict.json)
remains the authoritative honest red with exact outputs, unchanged thresholds,
and these two implementation-ready repair triggers.
