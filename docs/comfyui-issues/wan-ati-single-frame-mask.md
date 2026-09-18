# Wan ATI single-frame mask has zero temporal length

Status: found 2026-08-21; not reported upstream; fixed in Dinkster.

Baseline: ComfyUI `947c2749dd04c51ef0e21b069544d8b0b4f9b411`. The defect is also
present at `187eda8ef5e588c6a5765cad53e482765edae052` as of 2026-08-21.

## Symptom

`WanTrackToVideo` accepts `length=1`, but a nonempty track produces a mask
with shape `[B,4,0,H,W]`. Concatenating that mask with the one-frame motion
latent fails because their temporal dimensions differ.

## Root cause and reproduction

`_patch_motion_single` derives `out_weight` from `vid[:, 1:]`. For a
single-frame latent this tensor has no temporal entries, so
`torch.ones_like(out_weight[:1])` is also empty instead of creating the
conditioned first-frame mask.

Call `patch_motion` with processed one-frame tracks and a video latent shaped
`[1,16,1,H,W]`; the returned mask has temporal size zero.

## Suggested upstream fix

Create the first mask frame explicitly with shape `[1,H,W]`, using
`out_weight`'s dtype and device, before concatenating `out_weight`.

Dinkster applies that fix while preserving bit-exact ComfyUI output for
multi-frame inputs.
