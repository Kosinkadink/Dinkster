# Z-Image Q/K RMSNorm ignores the configured epsilon

- **Area:** ComfyUI `comfy/ldm/lumina/model.py` at
  `947c2749dd04c51ef0e21b069544d8b0b4f9b411`
- **Status:** found 2026-08-19; Dinkster follows official Z-Image

## Symptom

Z-Image BF16 execution diverges from the official model because query and key
RMSNorm use PyTorch's dtype-dependent default epsilon instead of 1e-5. On an
RTX 4090 seam comparison, identical QKV projection outputs first diverged at
the second context-refiner's Q/K normalization.

## Root cause

`JointTransformerBlock` receives `norm_eps` and uses it for its four hidden
state RMSNorm layers, but `JointAttention` constructs `q_norm` and `k_norm`
without passing an epsilon. `torch.nn.RMSNorm` therefore uses the input dtype's
machine epsilon. For BF16 that is 0.0078125, not the configured 1e-5.

Tongyi-MAI/Z-Image at
`e954755f5ced11262d5cb77e395b069005825a65` explicitly passes `norm_eps` into
both Q/K RMSNorm instances, with a default of 1e-5.

## Repro

Run a BF16 Z-Image attention block twice with identical projected Q/K values:
once through ComfyUI's `q_norm` and once through
`torch.nn.functional.rms_norm(..., eps=1e-5)`. The outputs differ. Repeating
with `eps=torch.finfo(torch.bfloat16).eps` matches ComfyUI.

## Suggested upstream fix

Pass `norm_eps` from `JointTransformerBlock` into `JointAttention`, then use it
when constructing `q_norm` and `k_norm`. Add BF16 coverage that compares these
normalizers with the official 1e-5 formula.

## Dinkster handling

Dinkster uses the official explicit 1e-5 epsilon. CUDA evidence checks the
official contract at the Q/K normalization seam instead of treating the
ComfyUI divergence as an acceptable tolerance.
