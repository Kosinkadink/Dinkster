# MiniMax H3 official I2V output collapses to a dark tail

Status: found 2026-09-23 on current ComfyUI master; not reported upstream.

Baseline: ComfyUI `b5cc8830279eae909a59de030af1e50761c36751` and
workflow_templates `fc427f00097817d3f7d8099c5259837fa51e1267`.

## Symptom

The official MiniMax H3 I2V template completes on an RTX 5070 Ti, but its
last frame is nearly black for the no-LoRA and official 8-step variants and
severely underexposed for the official 4-step variant. The middle frame is
normally illuminated in all three variants, so this is a tail collapse rather
than an all-black decode.

Normalized encoded Rec.709 luma for first, middle, and last frames was:

| Variant | First mean | Middle mean | Last mean | Last pixels below 5% |
| --- | ---: | ---: | ---: | ---: |
| no-LoRA | 6.675% | 31.208% | 0.020% | 99.922% |
| 8-step | 6.659% | 31.776% | 0.097% | 99.977% |
| 4-step | 6.633% | 38.062% | 0.819% | 93.613% |

Four attempts per variant produced identical output hashes and identical
frame statistics. The graph used the pinned template example
`transparent_rgb_gaming_mouse.png`, SHA-256
`49696748d2fff0e8c9b63c7173c6d6282b70eac0195def5b39402f9564410e75`.

## Reproduction

Run the official I2V template unchanged at its 0.4 megapixel, 124-frame
setting with the pinned INT8 ConvRot diffusion model and video VAE. Test the
no-LoRA, 8-step, and 4-step variants. Decode and measure frames 0, 62, and
123. The middle frame is visible while frame 123 collapses toward black.

The retained evidence and exact output hashes are recorded in
Kosinkadink/comfy-vibe-station#407.

## Upstream status and next diagnostic

At discovery time `refs/heads/master` was still the baseline SHA, so there
was no later ComfyUI commit that could contain a fix. Open issues include an
all-black INT8 VAE report (#15524) and periodic 17-frame darkening (#15426),
but neither describes this repeatable bright-middle/dark-tail result from the
official 124-frame I2V example.

The cause is unresolved. An upstream investigation should compare the final
sampled latent with the final decoded frames to distinguish generation collapse
from VAE decode collapse before changing model or VAE code.

Dinkster does not use these ComfyUI output hashes as evidence of acceptable
I2V quality. Runtime parity measurements must retain the frame-level defect as
a reference finding rather than treating a matching dark tail as success.
