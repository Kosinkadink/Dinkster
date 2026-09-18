# TemporalScoreRescaling crashes on batch > 1 (tensor truth test on sigma)

Status: found 2026-08-27; not reported upstream; Dinkster's port is
immune (per-request sigma is a scalar in the guidance contract).

Baseline: ComfyUI b78cec879b9460d5cb25228a83a942fb78d2cd24.

## Symptom

Sampling with the `TemporalScoreRescaling` node patched onto a model
errors with `RuntimeError: Boolean value of Tensor with more than one
value is ambiguous` whenever the latent batch size is greater than 1.

## Root cause

`comfy_extras/nodes_eps.py` (`temporal_score_rescaling` post-cfg
function) truth-tests the batched sigma tensor:

```python
if tsr_k == 1 or sigma == 0:
    return denoised
```

`args["sigma"]` has shape `(B,)`, so for `B > 1` the `sigma == 0`
comparison is a multi-element bool tensor and raises. The later
`if snr == 0:` guard has the same defect, and the closed-form rescale
math broadcasts `(B,)`-shaped `snr`/`alpha` factors against `(B, C, H,
W)` latents without reshaping, which would mis-broadcast if execution
reached it.

## Repro

On the pinned checkout, register `TemporalScoreRescaling`
(`tsr_k=0.95`, `tsr_sigma=1.0`) on any model and run
`comfy.samplers.sampling_function` with a `(2, 4, 8, 8)` latent.
Executed on 2026-08-27 via the Dinkster golden generator harness
(`tools/gen_cfg_transform_goldens.py` `run_case` with
`shape=(2, 4, 8, 8)`, `pass_model=True`): crashes at
`if tsr_k == 1 or sigma == 0:`.

## Suggested upstream fix

Guard on scalars (`torch.all(sigma == 0)` or early-exit per sample) and
reshape the per-sample factors to `(B, 1, 1, 1)` before broadcasting
against the latent, mirroring how other post-cfg functions handle
batched sigma.

## Dinkster handling

`temporal_score_rescaling` in
`dinkster_inference_torch/guidance_transforms.py` collapses the request
sigma to a scalar (uniform per-request sigma is the guidance contract),
so its guards and broadcasts are scalar and any batch size works.
Goldens cover batch 1 only because the reference cannot execute larger
batches to mint against.
