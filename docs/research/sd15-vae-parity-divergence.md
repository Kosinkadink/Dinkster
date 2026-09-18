# Canonical SD1.5 VAE parity divergence

Status: fixed on 2026-07-29 by enforcing the workload codec precision in the
ComfyUI parity adapter. No golden, tolerance, or acceptance threshold changed.

## Pins

- Dinkster measurement baseline: `708624e2ae3772f1ab3f99d87483e6354c68de09`,
  equal to `origin/main` when the fresh checkout was created.
- ComfyUI: `f4b99bc62389af315013dda85f24f2bbd262b686`, read-only and clean
  before and after measurement.
- Checkpoint: `v1-5-pruned-emaonly-fp16.safetensors`, sha256
  `e9476a13728cd75d8279f6ec8bad753a66a1957ca375a1464dc63b37db6e3916`.
- Seed `685468484323813`; Euler/normal; 20 steps; CFG 8; denoise 1;
  512x512, batch 1; diffusion fp16, text fp32, codec fp32 as declared by
  `tools/inference_parity/workloads.json`.
- Device: `GPU-666d1242-9c20-341c-73ea-e63770947451`, NVIDIA GeForce
  RTX 4090; torch `2.9.1+cu130` in both fresh processes.

## Method

The ComfyUI process reproduced the sampled latent once through the same
`nodes.common_ksampler` object path as the committed parity adapter. That
unchanged float32 tensor was then decoded in separate fresh ComfyUI and Dinkster
processes through `nodes.VAEDecode().decode` and
`SDRuntime.decode_latent`, respectively.

Temporary process-local forward hooks detached and copied the decoder input,
post-quant convolution, decoder input convolution, both middle resnets, the
middle attention norm/q/k/v/projection/output, every resnet and upsample in
all four up levels, final group norm, final SiLU input to the output
convolution, final convolution, and processed output. Hooks never replaced an
input or output. Nothing was committed because the workload-specific probe is
not reusable public tooling.

All digests are sha256 over contiguous raw array bytes after CPU float32
conversion. Max/mean statistics use torch float32 reductions. Exact dtype,
device, shape, and stride at the first divergence were captured before
conversion. The scale-factor seam was also checked: both
samplers apply SD1.5 `latent_process_out` (`latent / 0.18215`) before returning
the sampled latent, and neither VAE wrapper scales it again. The common
sampled-latent digest proves that seam exact.

A causal replay set only ComfyUI's VAE selection flag to fp32 before
`CheckpointLoaderSimple` constructed the VAE, matching the codec precision
already requested explicitly by Dinkster's adapter. It reused the captured
latent and the same `VAEDecode` object path. This changed policy only in the
temporary process; no source file was modified.

## Ordered comparison

The default column compares the committed adapters as they actually execute.
The matched-fp32 column compares the temporary ComfyUI fp32 replay with
Dinkster. A matched digest means max_abs and mean_abs are both zero.

| Stage | Default adapters | Matched-fp32 causal replay |
| --- | --- | --- |
| Sampled latent after SD1.5 process-out | Bit-exact float32 `[1,4,64,64]`, digest `7324adbac83a18c791138d81f8c21093a4703a2009deff16f7b78418ed4d9a46`. | Same digest. |
| VAE decode input, before post-quant conv | **First divergence.** ComfyUI CUDA bf16 digest `aba26c0b0608cff1c2b984d3edb11291d17f4c87d95b5fa502e2e5c30b734c4c`; Dinkster CUDA float32 digest `7324adba...`; max_abs `0.11250686645507812`, mean_abs `0.006496191490441561`. | Bit-exact float32 digest `7324adba...`. |
| Post-quant conv | ComfyUI bf16 digest `08f40b6e04c0f9032d41b6f471e27f8a28b5b77a211094e7d3d70f5dbb70c3d6`; Dinkster float32 `fca1a84ac075ccc9c587fb62243bcb68e2bc98f84dc5705bb5c6294e7b11c854`; max_abs `0.11036872863769531`, mean_abs `0.006220065988600254`. | Bit-exact digest `fca1a84a...`. |
| Decoder input conv | Digests `5bdba42f2491954711a738913455617e7f4fc9b57c68c8c445b604a26ef93251` versus `c6e1aa39c45c915bb91f5477c7754f2c77deaf884a4897427a9cb66a82b8e155`; max_abs `0.08255958557128906`, mean_abs `0.004500493407249451`. | Bit-exact digest `c6e1aa39...`. |
| Mid resnet 1 | Digests `c00f540868396207e2a8e006e29c5147e29f654d09b8d9eeb00167e6efd9261c` versus `5703a5acfa9799c79682d5325e719cc710ba41746d35518a3c1a8256ab7fa437`. | Bit-exact digest `5703a5ac...`. |
| Mid attention norm, q, k, v, SDPA, projection, residual | All diverged downstream in bf16 versus float32; attention-block output digests `3681c6efa00432b1b2d7b376e6b56071434245f84634276ef3f3fa77a54df841` versus `ba88c53186b15e866886895d70e5c6ad8db781b9fef6dbef983213d6ddee0ef6`. | Every captured attention seam bit-exact; block digest `ba88c531...`. |
| Mid resnet 2 | Digests `35b4e207d689f77a164ea01ac5127b94fe2585e54ec20955fe2148ba2c6cf1f7` versus `5e0fcdea55f2bc36c1a60214342261fd49ab12422a40299e8147e5e01401dfec`. | Bit-exact digest `5e0fcdea...`. |
| Up level 3: three resnets, then 64->128 upsample | All four captures diverged; terminal upsample digests `3e3376782eecc033ae0cac06a19a7c5960333b664d45f1db6232fae80df231e5` versus `1be01fb144ba3de06310b1b019a2838330d4606f0e5ff666eefb76aa384927b7`. | All four captures bit-exact; terminal digest `1be01fb1...`. |
| Up level 2: three resnets, then 128->256 upsample | All four captures diverged; terminal digests `9820daf230a7bdd34a3cdfba756caf34be4e5ad2067c151b97757ddb3af49af7` versus `b8deba4ffb9f88aa04e18aaf8b8572451b4b8539694a2dfff6f2a0d279cfbd5d`. | All four captures bit-exact; terminal digest `b8deba4f...`. |
| Up level 1: three resnets, then 256->512 upsample | All four captures diverged; terminal digests `9554b421c8d20ea1cafd06f12eb5b63a269976e5c79ef04fa4ceed7017ba9c99` versus `173f55787711f5fd6f7e40980b847eb1a6bba37d03acf336726c426c7800a753`. | All four captures bit-exact; terminal digest `173f5578...`. |
| Up level 0: three 512x512 resnets | All three captures diverged; terminal digests `379b8742cea41e7768ec1a98ae3378b31065d53af87da674574ebe49b650f7b2` versus `0b1b64295469d2f4cd91cedb36e6fcce2b855c17437ad7b82ba8fd0de7bc7261`. | All three captures bit-exact; terminal digest `0b1b6429...`. |
| Final group norm, SiLU, convolution | All three diverged; final-conv digests `36afb2f91c0192228f901fe539f915e37f6d5b1a3bc379d0d43f4767be2f0091` versus `4ac9a2be7f57478b9ed2847666ccf838572b7cdd0aac4095353f64fd6eea992f`. | All three bit-exact; final-conv digest `4ac9a2be...`. |
| `(x+1)/2`, clamp, output move/layout | Reproduced predecessor output: ComfyUI NHWC digest `2166a7eadb70eb181c7bb2497b7bdeaff0298f77df702b476ac01ca91797dbb4`; Dinkster after the adapter's NHWC permutation `24bb0dd6fac42f49f4a5bfd9d27bd61692ac2bb202ce45c7b80fd74f542ae65b`; max_abs `0.08470785617828369`, predecessor mean_abs `0.0012725989496023733`. | Bit-exact NHWC digest `24bb0dd6...`; max_abs 0. |

## First divergence and cause

The first differing operation is not post-quant convolution. It is the
ComfyUI VAE wrapper's dtype cast immediately before that convolution. At flat
index 0, the common sampled latent is `1.5207901000976562`. Dinkster preserves it
as CUDA float32; ComfyUI casts it to CUDA bf16 `1.5234375`. Both are
contiguous `[1,4,64,64]` tensors with stride `[16384,4096,64,1]`.

This is a dtype-policy mismatch in the parity adapter:

- `tools/inference_parity/workloads.json` declares codec float32.
- `tools/inference_parity/dinkster_adapter.py` passes
  `vae_dtype=torch.float32` to `load_runtime`.
- `tools/inference_parity/comfyui_adapter.py` does not apply the declared
  codec precision. ComfyUI therefore uses its hardware-dependent automatic
  `model_management.vae_dtype` choice. On the pinned RTX 4090 process it chose
  bf16, and `comfy/sd.py:VAE.decode` cast the latent and ran the VAE in bf16.

The temporary fp32 selection made all captured operations and the complete
processed output bit-exact with Dinkster. That single policy replay rules out a
latent-scale mismatch, weight/layout departure, scalar-kernel difference,
attention-backend difference, group-norm implementation, nonlinearity,
clamp/movement ordering, and residual accumulation-order noise.

## Recommendation: enforce the pin, do not tolerate

Do not add a codec tolerance and do not change Dinkster inference. The observed
residual is entirely caused by the ComfyUI parity adapter failing to enforce
the workload's existing fp32 codec declaration.

In `tools/inference_parity/comfyui_adapter.py:main`, validate
`workload["precision"]["codec"]` and set exactly one of ComfyUI's
`comfy.cli_args.args.fp32_vae`, `bf16_vae`, or `fp16_vae` flags before
`CheckpointLoaderSimple().load_checkpoint` constructs the VAE. For this
canonical workload, set `fp32_vae=True` and the other two false. Keep this in
the adapter rather than production inference code: it is execution of an
already-versioned benchmark pin, not a new VAE policy.

After that adapter repair, regenerate the canonical comparison record through
the full harness. The expected fixed-latent VAE output is the matched digest
`24bb0dd6fac42f49f4a5bfd9d27bd61692ac2bb202ce45c7b80fd74f542ae65b`
with max_abs and mean_abs zero. The independent denoiser scalar-kernel defect
from `docs/research/sd15-parity-divergence.md` still required its separate
repair before complete pipeline output could match.

## Resolution

The ComfyUI parity adapter now validates `precision.codec` and sets exactly one
of `fp32_vae`, `bf16_vae`, or `fp16_vae` before checkpoint construction. The
canonical fp32 warmup output matches Dinkster exactly at raw tensor digest
`24bb0dd6fac42f49f4a5bfd9d27bd61692ac2bb202ce45c7b80fd74f542ae65b`
with max_abs and mean_abs 0 and SSIM 1. ComfyUI's second same-process run now
exposes a separate warm-state self-divergence, ledgered in ROADMAP as
`COMFYUI-WARM-STATE-DIVERGENCE`; it does not reopen this precision defect.
