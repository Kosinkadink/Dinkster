# dinkster-vision-upscale

`dinkster-vision-upscale` is the separately installable execution provider for
`dinkster.image.upscale_model`. It runs in an isolated pack worker on CPU and
executes ESRGAN-family super-resolution checkpoints supplied as `dinkster.asset`
inputs - the model is chosen per graph, so the pack declares no artifacts of
its own.

Supported architectures, detected fail-closed from state-dict keys:

- ESRGAN-family RRDBNet: old-arch ESRGAN, ESRGAN+ (`conv1x1` residual paths),
  BSRGAN/RealSR, and Real-ESRGAN new-arch checkpoints (including the
  pixel-unshuffle x2/x1 variants). New-arch key layouts are converted to the
  flattened old-arch layout before loading. Network scales are powers of two.
- RealESRGAN Compact (SRVGGNetCompact), any integer pixel-shuffle scale.

Anything else is rejected with an error naming this list; state dicts always
load with `strict=True`. Detection, hyperparameter inference, and key
conversion reproduce spandrel 0.4.2 (commit
`724cca389f28c38e1050689d4862a452fd644484`), the loader ComfyUI itself uses.

Large images are processed as overlapping tiles with linearly feathered
blending, reproducing ComfyUI `comfy/utils.py` `tiled_scale_multidim` (commit
`a1079ba1`) specialized to two dimensions. The requested `tile_size` is
honored exactly: there is no out-of-memory halving retry, so a graph's
execution shape is deterministic. Each tile runs under spandrel's per-call
model contract: pad right/bottom to the architecture's size requirements
(minimum extent 2 for ESRGAN, rounded up to a multiple of 4 for
pixel-unshuffle variants; reflect, then replicate), forward, clamp to
`[0, 1]`, crop the padding off.
Clamping per tile before feather blending is required for parity - tiles
that overshoot `[0, 1]` blend differently if only the final image is
clamped. Outputs are `[0, 1]` float32 BHWC, matching ComfyUI's
`ImageUpscaleWithModel`.
