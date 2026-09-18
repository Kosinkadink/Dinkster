# Inference parity gate

This directory owns the pinned cross-engine acceptance gate. Each engine starts
in a fresh process with empty application/model caches. The adapter receives
exactly one measured `warmup` and one measured `real` request; both outputs and
all declared metrics gate independently. The process stays alive between them
to retain intended model residency. The harness never clears OS page caches.

`workloads.json` pins canonical graph bytes, commits, artifacts, semantics,
precision, hardware, and cache state. `acceptance.json` pins tolerances derived
before candidate comparison. JSON is schema-versioned, ASCII-only, sorted-key,
and atomically persisted. The cross-engine layer samples complete process-tree
RSS and designated-device usage. Native engine event timing remains at the
existing `dinkster.benchmark` seam.

Records live in the sibling `dinkster-evidence/inference-parity/records` tree.
Set `DINKSTER_INFERENCE_PARITY_RECORDS` to that path, or pass
`--records /path/to/inference-parity/records` to `run`, `compare`, `gate`, or
`time`, to resolve relative record inputs and outputs there. Without either
override, harness paths are used as supplied. Absolute paths remain unchanged.

Each record binds to the whole-file acceptance revision in force when it ran.
Re-comparing a historical record requires the matching acceptance revision from
git history. A baseline produced for decode-path calibration may instead bind to
the one explicitly declared `pre_calibration_acceptance_digest` predecessor in
the active workload policy. Candidate records must always bind to the current
digest. Pending calibration policies permit baseline recording but fail closed
on compare and cannot emit an acceptance verdict.

## Chroma family end-to-end comparison

`chroma_e2e.py` compares Chroma and Chroma Radiance separately through their
pinned official workflows and artifacts. Each engine process records a cold
execution, discards one resident warmup, retains three warm executions, and
captures conditioning, noise, sigmas, first/middle/final sampling states,
latents, images, timing, process RSS/VRAM, Torch allocator peaks, cleanup
residuals, and residency/fallback facts. The comparison requires bit-exact
values and requires Dinkster's cold and warm median time and every memory median to
be no greater than ComfyUI's.

Run `preflight` immediately before each GPU process in
`comfyui,dinkster,dinkster,comfyui` order. Preflight verifies clean exact source
commits, workflow bytes, complete artifact digests, and the production-derived
runtime identities without initializing CUDA, then records the exact `run`
command. Pass the proven healthy physical device as `--gpu-index` and
`--gpu-uuid`, then set `CUDA_VISIBLE_DEVICES` to that index for the recorded
command. `run` verifies the UUID and acquires the corresponding GPU claim.
Keep each JSON record with its `.npy` sidecar directory, then pass all four
records and preflights to `compare`. It verifies their digests, pinned artifacts,
GPU identity, runtime identities, and production residency route before issuing
a verdict. Repeat the complete process independently for `--variant chroma` and
`--variant radiance`.

## Flux2 Mistral text encoding

`flux2_text_performance.py` compares the official pruned BF16 Mistral Small 3.1
24B text encoder through ComfyUI and Dinkster's production dynamic-residency paths.
It pins the source commits, complete artifact digest and revision, GPU UUID,
runtime versions, prompt, token IDs, dtype, and process order. Each fresh process
records one cold encode, discards one warmup, and retains five warm encodes. The
comparison requires bit-identical Dinkster hybrid and demand-paged outputs and a
strictly lower Dinkster warm median. Cross-engine numerical drift is reported, not
hidden behind an output tolerance.

Run four fresh processes in `comfyui, dinkster, dinkster, comfyui` order, then pass the
four JSON paths to `compare`. Every engine command requires `--text-encoder` and
its source root; Dinkster also requires `--dinkster-commit`. Set `CUDA_VISIBLE_DEVICES`
to the UUID pinned by the script. Engine records and their `.npy` sidecars must
stay together until comparison.

```text
CUDA_VISIBLE_DEVICES=GPU-666d1242-9c20-341c-73ea-e63770947451 \
  /path/to/comfyui/python /path/to/Dinkster/tools/inference_parity/flux2_text_performance.py comfyui \
  --comfyui-root /path/to/ComfyUI-725e6ec6 --text-encoder /path/to/encoder \
  --output /new/output/01-comfyui.json
CUDA_VISIBLE_DEVICES=GPU-666d1242-9c20-341c-73ea-e63770947451 \
  /path/to/dinkster/python /path/to/Dinkster/tools/inference_parity/flux2_text_performance.py dinkster \
  --dinkster-root /path/to/clean/Dinkster --dinkster-commit COMMIT \
  --text-encoder /path/to/encoder --output /new/output/02-dinkster.json
CUDA_VISIBLE_DEVICES=GPU-666d1242-9c20-341c-73ea-e63770947451 \
  /path/to/dinkster/python /path/to/Dinkster/tools/inference_parity/flux2_text_performance.py dinkster \
  --dinkster-root /path/to/clean/Dinkster --dinkster-commit COMMIT \
  --text-encoder /path/to/encoder --output /new/output/03-dinkster.json
CUDA_VISIBLE_DEVICES=GPU-666d1242-9c20-341c-73ea-e63770947451 \
  /path/to/comfyui/python /path/to/Dinkster/tools/inference_parity/flux2_text_performance.py comfyui \
  --comfyui-root /path/to/ComfyUI-725e6ec6 --text-encoder /path/to/encoder \
  --output /new/output/04-comfyui.json
/path/to/dinkster/python /path/to/Dinkster/tools/inference_parity/flux2_text_performance.py \
  compare --dinkster-commit COMMIT \
  --inputs /new/output/{01-comfyui,02-dinkster,03-dinkster,04-comfyui}.json \
  --output /new/output/verdict.json
```

## Flux spatial-window comparison

`flux_performance.py` runs the issue #260 plain native Flux baseline without
TiledDiffusion. Its `dinkster --window-mode` diagnostic also profiles native,
one-window, and two-window execution with the same model and settings. Every
timed process discards one resident warmup and records five runs; `compare`
requires the balanced `comfyui, dinkster, dinkster, comfyui` process order and the
pinned GPU, checkpoint, runtime, noise, schedule, precision, and source
receipts.

`flux_window_comparison.py` compares Dinkster's declared Flux windows with the
pinned ComfyUI-TiledDiffusion MultiDiffusion surface. It verifies the complete
checkpoint digest, exact Dinkster, ComfyUI, and custom-node commits, clean source
trees, matching CUDA runtime receipts, initial noise, normal schedule,
deterministic outputs, one-window controls, and the warm repeated-run
performance floor. A controlled Dinkster run replacing only global image position
IDs with local IDs verifies the reason its window output differs from the
custom-node reference. The custom node is a CC BY-NC-SA comparison reference
only; none of its code is ported into Dinkster.

Run the two engines as fresh processes, in the shown order and in one GPU
session, then compare their records:

```text
CUDA_VISIBLE_DEVICES=0 /path/to/comfyui/python \
  tools/inference_parity/flux_window_comparison.py comfyui \
  --comfyui-root /path/to/ComfyUI-947c2749 \
  --tiled-root /path/to/ComfyUI-TiledDiffusion-a155b1ba \
  --checkpoint /path/to/flux1-dev-fp8.safetensors \
  --output-dir /tmp/flux-window-comparison

CUDA_VISIBLE_DEVICES=0 /path/to/dinkster/gpu-python \
  tools/inference_parity/flux_window_comparison.py dinkster \
  --dinkster-root /path/to/Dinkster-ae0d5ae7 \
  --checkpoint /path/to/flux1-dev-fp8.safetensors \
  --output-dir /tmp/flux-window-comparison

/path/to/python tools/inference_parity/flux_window_comparison.py compare \
  --input-dir /tmp/flux-window-comparison \
  --output /tmp/flux-window-comparison/verdict.json
```

The fixed workload is Flux dev at 512x512, seed 424242, four Euler/normal
steps, guidance 3.5, and two horizontal 384x512 windows with a 256-pixel
overlap and flat float32 merge. One cold run is informational, one resident
warmup is discarded, and five resident runs supply medians and ranges. A Dinkster
median passes the performance floor only when it is at most the ComfyUI median
plus the larger of 1 percent or three ComfyUI median absolute deviations.
Engine outputs are consolidated under `records/w0-flux-window-comparison/`;
generated `.npy` arrays remain local and their hashes are bound by the receipt.

`W0-SDXL-INPAINT` has a prepared harness and artifact receipt, not an
acceptance result. Its pure standard-library preflight verifies the complete
checkpoint digest and size, safetensors bounds and canonical SDXL inpaint /
dual-CLIP / VAE component geometries, immutable RGBA fixture and workflow,
interpreter identity, and adapter/harness bytes before model, Torch, or ComfyUI
imports. Both adapters emit named sampled `latent` and decoded `image` arrays.
The separate `acceptance_sdxl_inpaint.json` policy preserves the byte identity
and record compatibility of the existing acceptance manifest while freezing
exact equality with the existing independent 15 percent metric budget.
Warmup/real execution, records, verdict, deployment, and product activation
remain open.

## Minimal wall-time records

The `time` command is a native inference-runtime comparison through the existing
adapters, not HTTP, queue, server, or product latency. It retains the pinned
ComfyUI CPU `normal`-sigma mapping used by the exact-work correctness contract.
Each dated attempt uses one designated GPU and exactly four fresh processes in
`comfyui,dinkster,dinkster,comfyui` order. Every process receives one recorded cold
request, one unrecorded resident warmup, and two recorded warm requests: 16 jobs
total, with two cold and four warm values retained per engine. `cold` means an
empty process/application/model cache after mandatory digest validation has
warmed the OS page cache. `timings.json` separately pins the exact RipperPC
four-GPU inventory for timing mode; this does not relax the canonical X570
hardware pin used by `run` and `gate`.

The synchronized external `perf_counter_ns` request-to-reply wall clock is the
only reported timing. Every output must still pass the workload's existing
correctness comparator, and both adapters must report actual float32 text
parameters before timing can be valid. There is no trimming, deletion, retry,
or replacement.
The Dinkster/ComfyUI cold and warm median ratios are report-only. A compatible
current-Dinkster/prior-Dinkster warm median ratio strictly greater than `1.10` is
`WARN`; equality passes. The first compatible record establishes the baseline.
`timings.json` keeps the compact accepted rows, while git history is the history.

## GGUF reference comparison

`workloads_gguf.json` pins the SDXL Q8_0 diffusion artifact, external CLIP-L,
CLIP-G and VAE, ComfyUI, and ComfyUI-GGUF. The CPU reference lane requires
exact tensor mapping and bit-exact float32 dequantization for every Q8_0 tensor,
then bounds the independently implemented SDXL modules' sampled-latent and
float-image accumulation drift. The GPU lane uses the same artifacts and a
512x512, 20-step Euler/simple workload. The pinned 16 GB GPU cannot run the
ComfyUI reference's regular VAE decode at 1024x1024 without falling back to
tiled decode, so both engines use 512x512.

The GPU lane runs in `comfyui,dinkster,dinkster,comfyui` process order. Each process
records cold load, sample, decode and request wall time, discards three resident
warm-ups, retains seven plain repeats, then records the first and second images
after changing CFG from 5 to 6. It reports median, MAD, min/max, sampling it/s,
process RSS, NVML process VRAM at a 50 ms polling interval, Torch allocator
peaks, actual dtypes, memory policy, attention backend, driver and GPU identity.
Warm Dinkster median wall time must be at parity or better within the measured MAD.

The accepted 2026-08-18 receipt on the pinned RTX 5060 Ti passed all 52 GPU
image comparisons exactly. Dinkster's 2.012 s warm median was 42.8 percent faster
than ComfyUI-GGUF's 3.518 s, and sampling was 10.897 versus 6.004 iterations/s.
Changing CFG did not add a second-image penalty: Dinkster's first/second medians
were 2.019/2.013 s versus 3.516/3.517 s. The tradeoff is eager dequantization:
Dinkster's cold request was 13.430 versus 9.055 s, primary NVML process VRAM was
11.07 versus 8.48 GB, and peak process RSS was 14.24 versus 5.17 GB.
`records/w0-sdxl-gguf-q8-parity/receipt.json` contains the pins, numerical
comparison, timing spread, runtime configuration, and memory measurements.

```text
.venv/bin/python tools/inference_parity/gguf_reference.py \
  --manifest tools/inference_parity/workloads_gguf.json \
  --acceptance tools/inference_parity/acceptance_gguf.json \
  --workload W0-SDXL-GGUF-Q8-PARITY \
  --comfyui-root /path/to/ComfyUI-947c2749 \
  --dinkster-root "$PWD" \
  --artifact-root /path/to/dinkster-gguf-e2e-240 \
  --output-dir /new/reference-directory

.venv/bin/python tools/inference_parity/harness.py time \
  --manifest tools/inference_parity/workloads_gguf.json \
  --acceptance tools/inference_parity/acceptance_gguf.json \
  --history tools/inference_parity/timings_gguf.json \
  --workload W0-SDXL-GGUF-Q8-PARITY \
  --device-uuid GPU-113db834-9cb8-8999-89e4-5a3148d37aec \
  --comfyui-root /path/to/ComfyUI-947c2749 \
  --dinkster-root "$PWD" \
  --template-root "$PWD" \
  --artifact-root /path/to/dinkster-gguf-e2e-240 \
  --output-dir /new/timing-directory
```

```text
.venv/bin/python tools/inference_parity/harness.py time \
  --manifest tools/inference_parity/workloads.json \
  --acceptance tools/inference_parity/acceptance.json \
  --history tools/inference_parity/timings.json \
  --workload W0-HARNESS-SD15-TXT2IMG \
  --device-uuid GPU-ff7692e5-eec5-36b4-4c72-e879f32c5e98 \
  --comfyui-root /path/to/pinned/ComfyUI \
  --dinkster-root "$PWD" \
  --template-root /path/to/workflow_templates-aa3661d9 \
  --artifact-root /home/kosin/ComfyUI-Shared/models \
  --output-dir /new/dated/attempt-directory
```

| Date | Workload | Dinkster | ComfyUI | GPU | Cold D/C | Warm D/C | D/prior D | Status |
|------|----------|-------|---------|-----|----------|----------|-----------|--------|
| 2026-08-06 | `W0-HARNESS-SD15-TXT2IMG` | 3.433973591s cold / 0.484856595s warm | 5.769788866s cold / 0.5924554265s warm | RTX PRO 6000 Blackwell, GPU0 | 0.595164515 | 0.818384934 | - | BASELINE |

All five canonical verdicts committed under `records/` are green. This table
reports committed-record truth; it does not claim a rerun at current HEAD.

| Workload | Candidate | Acceptance digest | `overall_pass` |
|----------|-----------|-------------------|----------------|
| `W0-HARNESS-SD15-TXT2IMG` | `c1e882e4a9fe25fc8a1237718ee5de92b21d36ae` | `sha256:df8852236fdd13e4c3ac56cfa9af5eff530fb81c153562a27d09c9dbef722d8a` | `true` |
| `W0-SD15-INPAINT` | `c415236b91589f9e3f2040419b28dd8cd9bf36e8` | `sha256:371c0c3aef1769bf517e003da5f19be06b4551137719e58bf546ef228dbc68cf` | `true` |
| `W0-SDXL-VPRED` | `61b926db8b5c5c0b70986ab6ba10dcd6d916d028` | `sha256:8d21fafb5320504c5ba3c18aa4069ede4e6ae75f4b680f5a94a4bf9156338f39` | `true` |
| `W0-SDXL-EDM-VPRED` | `073f8dc5d870eb47b0c4212c504691733f6e3c0e` | `sha256:b043f4e1e640197f6e023bff7312b21ad8395ba1a994b68f00882c360f22d740` | `true` |
| `W0-FLUX-GATED-OVIS-TXT2IMG` | `f893a964b34a9fa635c45dac2e05e5247a2bcd85` | `sha256:8763f0d08c3543ec14b478c35db631f35eac8c18aef703ccfdd1104d0c29120c` | `true` |

The earlier SD1.5 cold-load/RAM RED and Ovis latent/image RED diagnoses remain
historical evidence in `ROADMAP.md`; their repair policy and the rule against
weakening acceptance thresholds remain binding. The canonical records above
supersede those diagnoses as present status without asserting that current HEAD
was rerun.

Run the complete canonical proof from a clean Dinkster commit:

```text
.venv/bin/python tools/inference_parity/harness.py gate \
  --manifest tools/inference_parity/workloads.json \
  --acceptance tools/inference_parity/acceptance.json \
  --workload W0-HARNESS-SD15-TXT2IMG \
  --comfyui-root /home/kosin/ComfyUI \
  --dinkster-root "$PWD" \
  --template-root /path/to/workflow_templates-aa3661d9 \
  --artifact-root /home/kosin/ComfyUI/models \
  --output-dir /tmp/dinkster-inference-parity
```

`run` accepts the same roots plus `--engine comfyui|dinkster --output RECORD`.
`compare --baseline RECORD --candidate RECORD --acceptance FILE --manifest FILE
--workload ID --output FILE` is offline and exits nonzero if either phase fails.
It re-verifies each persisted output digest and the applied acceptance digest.
`gate` is exactly two `run` operations followed by `compare`; there is no
alternate execution path.

The canonical Ovis workload emits named `latent` and `image` arrays for each
phase. Latents gate exactly; images gate against the baseline-only regular versus
tiled VAE decode calibration. Performance is recorded without an Ovis budget.
Its policy is active and calibrated, and its committed canonical verdict is
green at the candidate and acceptance digest listed above.
The text artifact header declares all 310 accepted `model.*` tensors as BF16
storage. Both engines compute the text tower in float32; the Dinkster adapter
records this contract explicitly and leaves its two `inference_mode` regions
unchanged. The precision matrix found byte-identical positive and negative
embeddings under float32 compute (digests `0300e9f9...` and `b3b8c33f...`),
and identical values under inference-mode and no-grad contexts. The historical
sampled-latent divergence and its repair are tracked as
`W0-OVIS-DIFFUSION-FORWARD-DIVERGENCE` in `ROADMAP.md`. The committed evidence
is under `records/w0-harness-ovis/`.

## Qwen Image artifact receipts

`qwen_image_artifacts.json` pins the official base Qwen Image artifact graph at
an immutable `Comfy-Org/Qwen-Image_ComfyUI` revision and the upstream Qwen
Apache-2.0 license text. It records ordinary contained local paths, exact byte
sizes, SHA256 and BLAKE3 digests, exact DiT and scaled Qwen2.5-VL header layouts,
and the Wan21 header count, dtype, and structural anchors. All three files were
already present in the shared artifact root, so the receipt creates no duplicate
model bytes.

Verify the receipts without importing or executing a model:

```text
.venv/bin/python tools/inference_parity/qwen_image_receipts.py \
  --artifact-root /home/kosin/ComfyUI-Shared/models
```

This is provenance and header-role evidence only. It does not register, load,
execute, redistribute, or claim support for Qwen Image.

## MiniMax H3 codec source authority

`minimax_h3_codec_artifacts.json` pins the official video and audio VAE provider
paths, byte sizes, provider `lfs.sha256` values, and immutable URLs at one
`Comfy-Org/MiniMax-H3` revision. It separately pins the immutable upstream
MiniMax H3 community-license text and its territorial restriction statement,
without claiming acceptance. `minimax_h3_codec_receipts.py` validates only this
closed source authority and cross-checks its two component IDs against the
torch-free H3 profile.

This authority performs no artifact acquisition, local artifact-file inspection,
physical verification, model loading, execution, registration, or support
claim. Those gates remain in `ROADMAP.md` under "MiniMax H3 source sequence".
