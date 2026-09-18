# ComfyUI same-process warm-state divergence

Status: localized on 2026-07-29. The first divergence is the normal sigma
schedule, not noise, UNet state, VAE state, attention batching, or a
memory-dependent precision switch. No production inference behavior, golden,
slice.

## Pins and baseline

- The fresh Dinkster clone started at
  `fa9cd8f4c2be3973c40477df1e0b49076d760bb6`; `HEAD` and `origin/main` were
  exactly equal at clone time.
- ComfyUI was `f4b99bc62389af315013dda85f24f2bbd262b686`, read-only and
  clean before and after capture.
- Checkpoint: `v1-5-pruned-emaonly-fp16.safetensors`, sha256
  `e9476a13728cd75d8279f6ec8bad753a66a1957ca375a1464dc63b37db6e3916`.
- Canonical workload: seed `685468484323813`; Euler/normal; 20 steps; CFG 8;
  denoise 1; 512x512, batch 1; diffusion fp16, text fp32, codec fp32.
- Device: `GPU-666d1242-9c20-341c-73ea-e63770947451`, RTX 4090; torch
  `2.9.1+cu130`, cuDNN 9.13.0.
- Both accepted captures began after an explicit coordinator hold and an
  immediately preceding `nvidia-smi` check showing zero compute processes.
  The natural trace is retained at
  `/home/kosin/comfy-vibe-station/pr-tracker/stations/station9/delegates/
  comfyui-warm-state-divergence/evidence-natural/trace.json`; the causal replay
  is beside it under `evidence-cpu-sigmas/trace.json`.

## Method

The same fresh ComfyUI process received the canonical warmup request and then
the identical real request, matching the parity adapter. Temporary
process-local wrappers and forward hooks copied tensors to CPU after each
seam and recorded raw contiguous float32 sha256 digests, original dtype,
device, shape, and stride. Captures covered schedule construction, initial
noise, every Euler callback `x` and denoised result, every actual batched UNet
input/output, the first convolution input/output, sampled latent, VAE input
and output, model-load requests, loaded/offloaded bytes, free-memory values,
and CUDA allocated/reserved bytes. The temporary script was outside both
repositories and was removed after evidence extraction.

The trace reproduced the committed records byte-for-byte:

- warmup: raw digest
  `24bb0dd6fac42f49f4a5bfd9d27bd61692ac2bb202ce45c7b80fd74f542ae65b`,
  `.npy` sha256
  `799266ed84723a70c28874de478b867157e0e406df36d7cfb64a020fd2564d2c`;
- real: raw digest
  `a589b39e21bb07689a9d06af95c13c2e3bdc00a3cfd0e14938396b128b497251`,
  `.npy` sha256
  `28139ad4b266f732e60d8f686dfadcd8cbfba3087e0b5ae9b1f186a182ff8e7b`.

The output difference remained max_abs `0.027042925357818604`, mean_abs
`0.0004571652098093182`; the largest element was flat index 535414,
`0.6030497550964355` in run 1 and `0.6300926804542542` in run 2.

## Ordered seam comparison

| Seam | Run 1 (cold/warmup) | Run 2 (real) | Result |
| --- | --- | --- | --- |
| Normal sigma schedule | `88e30a19a489c6979796222b227749f4fb4aeb65ea7cfe62784261653919987e` | `2f4610efddd4df3b81ae6f9f6d912c1d6a4ae046255587dee364637ab7e5ee20` | **First divergence.** Six of 21 float32 values differ. |
| Initial noise | `191c57595c00f9edeb0b8d400b1a9584da992666d44a273dd03b71f8071cd24d` | Same | Bit-exact CPU float32 `[1,4,64,64]`; no per-step draw occurs because Euler has `s_churn=0`. |
| Euler steps 0-4 `x` and denoised | Identical at every step | Identical at every step | Float rounding absorbs the schedule difference until the update into step 5. |
| Euler step 5 `x` | `8f6c6c0aa97cb80d04205b92adb55ff225b26efb03b9bf8815d0e058225a9441` | `e1c760dee182c7f445e24810f9699bf07d740086c0f4dd146eb67dac9d98e462` | First downstream latent difference: max_abs `1.9073486328125e-06`, mean_abs `1.86884932418252e-07`; flat index 15783 is `-12.056079864501953` versus `-12.05607795715332`. |
| Step 5 actual UNet input | `00cbb008e8e4d174e7e252424d96fd48880c6d84796a93f38513d42954ec54d9` | `aaaf4dbbf93001f36ba6bdcc05a1b9bfe79cdefa1badf45637d3e09ef696f5d1` | CUDA fp16 `[2,4,64,64]`, CFG order `[1,0]`; max_abs `1.52587890625e-05`, mean_abs `9.458744898438454e-10`; flat index 10526 is `-0.0182647705078125` versus `-0.018280029296875`. |
| Step 5 first convolution output | `6108b1e3fd8ee2e3801a42800c853333eb6e2fb6bea183640747f0e7a8c35ecf` | `a2457ed5b6c0ecccde9e817e60add49fe4f5cdf5c8668fe136c4f1747d51bb7a` | Downstream of the differing input; max_abs `0.000244140625`. |
| Step 5 UNet output | `da1ffcf75c558f6c8959c3f3c2210107542f43873b807f438bb4463fe45e521b` | `d7c9e289a04fe4c7f7307364872c73eee3d3aded105561e6771fbd5f0ae0f09a` | Downstream; max_abs `0.001953125`, mean_abs `0.00009838369442149997`. |
| Step 5 CFG-denoised | `5304e011d20a22371fff23106c56e9cd0d42d2f2a33199dec7084fe9f122483d` | `220b55c385007fdbc69b16af2dbdb9be23802f8d66b8400f2829ad48149dff0b` | Downstream; max_abs `0.11323928833007812`, mean_abs `0.0051108840852975845`; flat index 4489 is `-0.47498607635498047` versus `-0.5882253646850586`. |
| Steps 6-19 | Different `x`, UNet input/output, and denoised digests at every step | Different | Propagation from step 5; there is no new independent seam. |
| Sampled latent / VAE input | `7324adbac83a18c791138d81f8c21093a4703a2009deff16f7b78418ed4d9a46` | `6943e016f8cde574774f7e9660306336960f8c8e7e81fd9f37f4afaae262fbe9` | max_abs `0.15195989608764648`, mean_abs `0.008627592585980892`; the VAE receives the already-different latent. |
| VAE output | `24bb0dd6fac42f49f4a5bfd9d27bd61692ac2bb202ce45c7b80fd74f542ae65b` | `a589b39e21bb07689a9d06af95c13c2e3bdc00a3cfd0e14938396b128b497251` | Final propagation, not a VAE-originated divergence. |

The schedule first differs at boundary 5: run 1 has
`3.8653781414031982` (`5b627740` little-endian bytes), while run 2 has
`3.865377902984619` (`5a627740`), exactly one float32 ULP apart. Boundaries
6 and 7 also differ by `2.384185791015625e-07`; boundaries 15, 17, and 18
differ by `5.960464477539063e-08`, `2.9802322387695312e-08`, and
`1.4901161193847656e-08`, respectively. All other boundaries are bit-exact.

## Mechanism

This is cached model-device state changing where schedule math executes.
ComfyUI `comfy/samplers.py:628-650` constructs the normal schedule by calling
`ModelSamplingDiscrete.sigma` for every interpolated timestep.
`comfy/model_sampling.py:163-174` explicitly moves the interpolation operand
to `self.log_sigmas.device`, performs the interpolation and `exp` there, then
moves the answer back to the timestep device.

Before run 1, the checkpoint's model-sampling buffers are CPU-resident, so
schedule interpolation and `exp` execute on CPU. Sampling then calls
`load_models_gpu`; `comfy/model_management.py:553-561` enters
`ModelPatcher.partially_load`, whose full-load branch recursively moves the
BaseModel to CUDA at `comfy/model_patcher.py:1078-1104` and
`comfy/model_patcher.py:920-925`. That move includes the registered `sigmas`
and `log_sigmas` buffers. Retained residency leaves those buffers on CUDA for
run 2, which therefore performs the same float32 interpolation and `exp` on
CUDA. The CPU and CUDA kernels round six boundaries differently.

Memory instrumentation rules out a memory-management mode switch:

- VRAM state stayed `NORMAL_VRAM`; minimum inference memory stayed
  `1,278,423,859.2` bytes.
- The BaseModel was fully loaded in both runs: `1,719,049,928` of
  `1,719,049,928` bytes, zero offloaded bytes, zero low-VRAM patches.
- The VAE was fully loaded in both runs: `334,615,452` of `334,615,452`
  bytes, and the text model remained fully loaded at `247,300,612` bytes.
- The sampling request was unchanged: estimated full-batch memory
  `171,798,691.84` bytes and minimum memory `85,899,345.92` bytes. The
  model-load allowance changed from `22,869,329,612.8` to
  `14,817,552,076.8` bytes as the CUDA allocator retained memory, but neither
  run entered partial loading.
- Both runs used one `[2,4,64,64]` fp16 UNet invocation per step with CFG
  order `[1,0]`, so free-memory batching did not change.
- Attention remained PyTorch SDPA; observed diffusion and VAE tensors stayed
  fp16 and fp32, respectively, and the process-wide TF32 and deterministic
  flags did not switch.

It is also not RNG drift: `comfy/sample.py:22-38` reseeds and draws the
initial tensor on CPU, and both draws are bit-exact. Plain Euler only draws
step noise when `gamma > 0`; `comfy/k_diffusion/sampling.py:190-211` keeps
`gamma=0` for this workload, so the per-draw list is empty.

## Causal replay

A second fresh process changed only schedule evaluation: immediately around
`comfy.samplers.calculate_sigmas`, it moved `model_sampling` to CPU, called
the unchanged upstream function, and restored the original device in a
`finally` block before sampling. Model residency and every other adapter path
were unchanged.

The replay made both schedule digests
`88e30a19a489c6979796222b227749f4fb4aeb65ea7cfe62784261653919987e`.
Every captured Euler `x`, UNet input/output, first convolution, denoised
result, sampled latent, and VAE output then became bit-exact across runs. Both
final raw digests were `24bb0dd6...`, both `.npy` sha256 values were
`799266ed...`, and max_abs and mean_abs were zero. This one-variable replay
proves the schedule-device transition is causal rather than merely
correlated with the retained VRAM increase.

## Single recommendation: harness calibration fix

Do not widen any tolerance and do not change Dinkster inference. In
`tools/inference_parity/comfyui_adapter.py:main`, after importing
`comfy.samplers` and before processing requests, install a process-local
wrapper around `comfy.samplers.calculate_sigmas` that returns the original
function unchanged unless `scheduler_name == "normal"`. For `normal` only,
the wrapper must:

1. record `original_device = model_sampling.sigmas.device`;
2. move `model_sampling` to `torch.device("cpu")`;
3. call the original `calculate_sigmas(model_sampling, scheduler, steps)`;
4. restore `model_sampling` to `original_device` in `finally`.

Keep this adapter-only and document it as enforcement of the canonical
CPU-derived `normal` sigma recipe already matched by Dinkster and the committed
cold record. The small model-sampling buffers move temporarily; BaseModel
weights, the model-manager residency record, the persistent process,
warm-generation measurement, UNet batching, VAE policy, and all acceptance
semantics remain unchanged. Add focused adapter tests proving both `sigmas`
and `log_sigmas` return to their original device after success and exception,
then regenerate the canonical record through the unchanged gate. The causal
replay establishes the expected outputs without deriving any tolerance from
error.
