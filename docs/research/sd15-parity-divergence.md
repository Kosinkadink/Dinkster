# Canonical SD1.5 parity divergence

No golden or acceptance threshold changed.

## Pins

- Dinkster baseline: `f4371777b19526ff72e0dcbb9681fa0078619219`, equal to
  `origin/main` before measurement.
- ComfyUI: `f4b99bc62389af315013dda85f24f2bbd262b686`, read-only and clean
  before and after measurement.
- Checkpoint: `v1-5-pruned-emaonly-fp16.safetensors`, sha256
  `e9476a13728cd75d8279f6ec8bad753a66a1957ca375a1464dc63b37db6e3916`.
- Seed `685468484323813`; Euler/normal; 20 steps; CFG 8; denoise 1;
  512x512, batch 1; diffusion fp16, text fp32, codec fp32.
- Device: `GPU-666d1242-9c20-341c-73ea-e63770947451`, NVIDIA GeForce
  RTX 4090; torch `2.9.1+cu130` in both processes.
- Prompts and graph/artifact pins are the committed
  `tools/inference_parity/workloads.json` values. The traced warmup outputs
  reproduced the committed `.npy` files byte-for-byte: ComfyUI
  `03d21a7fdfd16f84335d8773887c069f95786142ce3e31b0d64e78c0aa2da3d5`
  and Dinkster
  `a9d29bdac44f5af516d2c70dbcb7ea2f89a0d3bd67ea16a3e7c72ec48b7da597`.

## Method

The two engines ran in separate fresh processes through the same object-level
paths as their committed parity adapters. Temporary, process-local hooks saved
schedule values, initial noise, conditioning, Euler callback tensors, the
actual first UNet convolution input, selected module outputs, sampled latent,
and decode output. The hooks only detached and copied tensors after or
immediately before an operation. They did not alter production files and are
not committed because this one-workload probe is not a reusable public tool.

All tensor digests below are sha256 over contiguous raw array bytes, not the
`.npy` container. Captured floating tensors were converted to float32 on CPU
before hashing. Exact dtype/device at the first divergence was recorded before
that conversion.

## Ordered comparison

| Stage | Result | Exact evidence |
| --- | --- | --- |
| 1. Sigma schedule | Bit-exact | Both raw float32 digests `88e30a19a489c6979796222b227749f4fb4aeb65ea7cfe62784261653919987e`; max_abs 0. |
| 2. Initial latent noise | Bit-exact | Both raw float32 digests `191c57595c00f9edeb0b8d400b1a9584da992666d44a273dd03b71f8071cd24d`; CPU float32 `[1,4,64,64]`; max_abs 0. |
| 3a. Euler step 0 solver `x` | Bit-exact | Both raw float32 digests `e5695a64fc08ca226ef57ae75c406bcbc76936ccd32e584e424d31101e1201f0`; CUDA float32 `[1,4,64,64]`; max_abs 0. |
| 3b. Actual step 0 UNet input | **First divergence** | ComfyUI digest `c3b52c4ee5dd41b77f58a38c4e0a19e851385424932c30cef5baaac9e4b9a6ce`; Dinkster `5e2d8051558a727ffb82f7c8ffa48ca4c6b80348d1e809e934585bdba182a54a`; CUDA fp16, contiguous `[2,4,64,64]`; max_abs `0.00048828125`, mean_abs `2.9802322387695312e-08`. |
| 3c. Step 0 CFG-denoised output | Diverged downstream | At flat index 0 ComfyUI `0.6638044118881226`, Dinkster `0.6785225868225098`; max_abs `0.228363037109375`, mean_abs `0.01762481397827287`. |
| 4. CLIP conditioning | Bit-exact; checked to rule out the cause of 3b | Positive digest `c14e6a779b5d3e115704ec837c34d45e91734bdd9410acf4b1ab7b15a9e6eda9`; negative digest `801469eaa758986cb5116bb86a70239405cd6bf91e170c03bae994005076f805`; both engines, float32 `[1,77,768]`, max_abs 0. |
| 5. Fixed-latent VAE decode | Independent later divergence | Decoding the identical ComfyUI sampled latent produced raw float32 output digests ComfyUI `2166a7eadb70eb181c7bb2497b7bdeaff0298f77df702b476ac01ca91797dbb4`, Dinkster `24bb0dd6fac42f49f4a5bfd9d27bd61692ac2bb202ce45c7b80fd74f542ae65b`; max_abs `0.08470785617828369`, mean_abs `0.0012725989496023733`; first flat element `0.47705078125` versus `0.4774077832698822`. This is not the pipeline's first divergence and was not localized further under the stop rule. |

The 21 schedule boundaries, in order, were identical:

```text
14.614640235900879
10.746800422668457
8.08151912689209
6.204935073852539
4.855652332305908
3.8653781414031982
3.1237614154815674
2.55716609954834
2.1156575679779053
1.764822244644165
1.4805806875228882
1.245813250541687
1.0481420755386353
0.8784281611442566
0.7297186851501465
0.5964335799217224
0.47358471155166626
0.35554540157318115
0.23216423392295837
0.029167160391807556
0.0
```

The initial noise's first eight values were also identical:

```text
0.05262169614434242
0.1335056722164154
0.34370288252830505
1.915805459022522
-0.021359939128160477
0.16355028748512268
1.9042985439300537
-0.19655580818653107
```

## First divergence and cause

The first differing actual UNet-input element is flat index 10055 in each
duplicated CFG batch. ComfyUI has `-0.8828125`; Dinkster has
`-0.88232421875`; absolute error `0.00048828125`, exactly one fp16 ULP at
that magnitude. Both inputs are contiguous CUDA fp16 with shape
`[2,4,64,64]` and stride `[16384,4096,64,1]`.

This is not an attention-backend difference, a weight mismatch, CLIP drift,
CFG ordering, seeded-noise difference, sigma-table difference, or generic
fp16 accumulation-order noise:

- all 686 UNet state tensors had identical keys, shapes, dtypes, and raw
  digests;
- the step 0 Euler `x`, CLIP tensors, sigma, and timestep embedding were
  bit-exact;
- the divergence exists at the input of `input_blocks.0.0`, before the first
  convolution or attention operation.

The cause is an algorithmic kernel/order mismatch in EPS input
preconditioning. At sigma `14.614640235900879`, the solver value at the
differing position is `-12.92857837677002`:

- ComfyUI `comfy/model_sampling.py:EPS.calculate_input` evaluates
  `noise / (sigma ** 2 + 1.0 ** 2) ** 0.5` with a CUDA float32 sigma tensor.
  Its float32 reciprocal factor is `0.0682649165391922`, and the fp16 result
  is `-0.8828125`.
- Dinkster `dinkster_inference.parameterizations.calculate_input` evaluates
  `1.0 / math.sqrt(sigma * sigma + 1.0)` as a Python float64 scalar, then
  multiplies the tensor. Its factor is `0.0682649188358628`, and the fp16
  result is `-0.88232421875`.

A temporary process-local replacement of only Dinkster's input preconditioner
with ComfyUI's tensor square/root/divide expression made the first UNet input,
first convolution output, and complete step 0 CFG-denoised output bit-exact.
That causal check rules out an incidental hook or load-order effect.

After that temporary replacement, the next mismatch moved to Euler's update:
step 1 `x` max_abs `3.814697265625e-06`, mean_abs
`9.888026397675276e-09`. Dinkster's generic Euler implementation likewise uses
Python-float sigma arithmetic while ComfyUI keeps sigma as a device float32
tensor. Therefore the scalar-kernel mismatch is systematic, not a one-line
special case at preconditioning.

## Recommendation: fix, do not tolerate

Do not declare an output tolerance for this mechanism. The authoritative
inputs are exact, but the first differing operation is a known algorithmic
departure from ComfyUI, and a local tensor-kernel replay removes it exactly.
This fails the repository rule that a tolerance is allowed only after all
upstream seams are bit-exact and residual drift is pinned to denoiser
accumulation.

The repair should preserve the torch-free pure-value APIs for non-executed
math, but the executed torch path must use reference kernels:

1. In
   `packages/dinkster-inference-torch/src/dinkster_inference_torch/sd_denoise.py`,
   change the EPS preconditioning used by `SDDenoiser.__call__`,
   `call_with_uncond`, and `evaluate_guidance` to construct/use a float32
   scalar on `x.device` and evaluate ComfyUI's square/root/divide expression,
   rather than calling the Python-float
   `dinkster_inference.parameterizations.calculate_input` implementation.
   Expected behavior at ComfyUI `f4b99bc`: the captured step 0 UNet input,
   first convolution output, and CFG-denoised output above become bit-exact.
2. In the executed torch Euler path corresponding to
   `packages/dinkster-inference/src/dinkster_inference/solvers.py:euler`, keep sigma,
   `sigma_hat`, `_to_d` division, and `dt` arithmetic as device float32 tensor
   operations matching `comfy/k_diffusion/sampling.py:sample_euler`. Do not
   change the generic pure solver contract blindly; add or route through a
   torch reference-kernel implementation at the torch sampler registry seam.
   Expected behavior is no new step 1 `x` mismatch after fixing step 0.
3. Rerun this ordered trace after those two fixes before deciding whether any
   residual is accumulation-order noise. The fixed-latent VAE result above is
   a separate later divergence and must be localized before the canonical
   output gate can be declared numerically explained.

These changes were intentionally not implemented in the instrumentation-only
validation recorded below.

## Resolution

updates through device-float32 reference kernels while preserving the pure
torch-free APIs. The canonical ordered trace is now bit-exact at every pinned
seam (sha256 over contiguous CPU float32 bytes, ComfyUI then Dinkster):

- step-0 UNet input: both
  `c3b52c4ee5dd41b77f58a38c4e0a19e851385424932c30cef5baaac9e4b9a6ce`,
  max_abs and mean_abs 0;
- first convolution output: both
  `e8c85a04afb94723c7d7fc4e520988a0cc36191287f17eec4c06d5f293e043d7`,
  max_abs and mean_abs 0;
- complete step-0 CFG-denoised output: both
  `494eb2fa3be2fdbfa1d6a72e08448f41487a89563ac3729f5ab76bda81430606`,
  max_abs and mean_abs 0;
- step-1 Euler `x`: both
  `07fb97f741452ee059898e37f32c1b13f73ad5953068400fc0bf7580a080218b`,
  max_abs and mean_abs 0.

The unchanged step-0 Euler `x` remains
`e5695a64fc08ca226ef57ae75c406bcbc76936ccd32e584e424d31101e1201f0`.
