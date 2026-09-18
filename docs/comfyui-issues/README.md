# Upstream ComfyUI issues found during Dinkster work

One .md per issue. These are defects in ComfyUI itself, discovered
while porting/oracle-testing against the pinned reference checkout
(`../ComfyUI`, baseline commit recorded in each file). They are NOT
Dinkster work items - Dinkster's own handling of each is ledgered in
ROADMAP.md; this folder exists so the upstream fixes can be worked
on later (issue report, PR, or both) without re-deriving the
analysis.

Each file records: symptom, root cause, repro, suggested upstream
fix, how Dinkster handles it meanwhile, and a status line (found /
reported / PR open / fixed upstream). Update the status line when
acting on one; when an issue is fixed upstream, note the fixing
commit and regenerate any Dinkster goldens that pinned the old
behavior, then keep the file for provenance.

## Index

| Issue | Area | Status |
| --- | --- | --- |
| [t5-positive-end-layer-index.md](t5-positive-end-layer-index.md) | comfy/sd1_clip.py + comfy/text_encoders/t5.py | verified in pinned source; Dinkster rejects the invalid index |
| [load-image-small-animated-gif.md](load-image-small-animated-gif.md) | LoadImage / video alignment filter | found locally; native Pillow loading succeeds |
| [lokr-tucker-kron-noncontiguous.md](lokr-tucker-kron-noncontiguous.md) | comfy/weight_adapter/lokr.py | found 2026-07; fixed in Dinkster |
| [boft-fp16-partial-strength-dtype-mismatch.md](boft-fp16-partial-strength-dtype-mismatch.md) | comfy/weight_adapter/boft.py | found 2026-07; fixed in Dinkster |
| [comfy-kitchen-no-cuda-env-ignored.md](comfy-kitchen-no-cuda-env-ignored.md) | comfy-kitchen setup.py | fixed upstream; verified in 0.2.31 |
| [comfy-kitchen-missing-packaging-dependency.md](comfy-kitchen-missing-packaging-dependency.md) | comfy-kitchen PyPI metadata | present in 0.2.31; handled in Dinkster |
| [comfy-kitchen-cuda-stochastic-rounding-mutates-rng.md](comfy-kitchen-cuda-stochastic-rounding-mutates-rng.md) | comfy-kitchen CUDA backend | still present in 0.2.31; Dinkster immune |
| [comfy-kitchen-cuda-stochastic-rounding-wrong-device.md](comfy-kitchen-cuda-stochastic-rounding-wrong-device.md) | comfy-kitchen CUDA backend | still present in 0.2.31; fixed in Dinkster |
| [comfy-kitchen-triton-apply-rope-wrong-device.md](comfy-kitchen-triton-apply-rope-wrong-device.md) | comfy-kitchen Triton backend | still present in 0.2.31; fixed in Dinkster |
| [comfy-kitchen-nvfp4-biased-linear-dequantizes.md](comfy-kitchen-nvfp4-biased-linear-dequantizes.md) | comfy-kitchen NVFP4 tensor dispatch | retained by 0.2.31; Dinkster direct route unaffected |
| [asym-w4a8-checkpoint-contract-mismatch.md](asym-w4a8-checkpoint-contract-mismatch.md) | comfy/quant_ops.py + comfy/ops.py + comfy-kitchen W4A8 serialization | still present with Kitchen 0.2.31; Dinkster refuses it |
| [stochastic-rounding-fp16-log2-boundary.md](stochastic-rounding-fp16-log2-boundary.md) | comfy/float.py + comfy-kitchen eager backend | still present in 0.2.31; fixed in Dinkster |
| [convert-old-quants-ignores-marker-dtype.md](convert-old-quants-ignores-marker-dtype.md) | comfy/utils.py | found 2026-07; not reported upstream; handled in Dinkster |
| [sd1-clip-bare-embedding-directive-crash.md](sd1-clip-bare-embedding-directive-crash.md) | comfy/sd1_clip.py | found 2026-07; handled in Dinkster (reported-not-crash divergence) |
| [uni-pc-missing-no-grad.md](uni-pc-missing-no-grad.md) | comfy/extra_samplers/uni_pc.py | found 2026-07-30; Dinkster immune; parity adapter fixed |
| [kling-single-image-effect-duration-enum-mismatch.md](kling-single-image-effect-duration-enum-mismatch.md) | comfy_api_nodes/nodes_kling.py + apis | found 2026-07-30; node deferred to partner slice 3.2b |
| [image-aspect-lower-bound-comparator-inversion.md](image-aspect-lower-bound-comparator-inversion.md) | comfy_api_nodes/util/validation_utils.py | found 2026-07-31; fixed in Dinkster diagnostics |
| [video-duration-probe-exception-bypasses-validation.md](video-duration-probe-exception-bypasses-validation.md) | comfy_api_nodes/util/validation_utils.py | found 2026-07-31; Dinkster fails loudly |
| [video-dimension-probe-exception-bypasses-validation.md](video-dimension-probe-exception-bypasses-validation.md) | comfy_api_nodes/util/validation_utils.py | found 2026-08; Dinkster fails loudly |
| [context-window-duplicate-index-fuse-drops-occurrences.md](context-window-duplicate-index-fuse-drops-occurrences.md) | comfy/context_windows.py | found 2026-08-17; Dinkster contract specifies per-occurrence accumulation |
| [z-image-qk-rmsnorm-epsilon.md](z-image-qk-rmsnorm-epsilon.md) | comfy/ldm/lumina/model.py | found 2026-08-19; Dinkster follows official Z-Image |
| [wan-ati-single-frame-mask.md](wan-ati-single-frame-mask.md) | comfy_extras/nodes_wan.py | found 2026-08-21; fixed in Dinkster |
| [wan-humo-no-audio-latent-count.md](wan-humo-no-audio-latent-count.md) | comfy_extras/nodes_wan.py | found 2026-08-26; fixed in Dinkster |
| [flux2-tekken-tokenizer.md](flux2-tekken-tokenizer.md) | comfy/text_encoders/flux.py | found 2026-08; crash fixed upstream (bbb4b04c); Dinkster self-contained |
| [cosmos-predict2-retains-forward-activations.md](cosmos-predict2-retains-forward-activations.md) | comfy/ldm/cosmos/predict2.py | found 2026-08 (Dinkster #841); benchmark shim releases through ComfyUI free path |
| [renorm-cfg-batch-tensor-bool-crash.md](renorm-cfg-batch-tensor-bool-crash.md) | comfy_extras/nodes_lumina2.py | found 2026-08-27; Dinkster mirrors batch-1 contract |
| [temporal-score-rescaling-batch-tensor-bool-crash.md](temporal-score-rescaling-batch-tensor-bool-crash.md) | comfy_extras/nodes_eps.py | found 2026-08-27; Dinkster immune (scalar sigma contract) |
| [flux-normal-collapsed-brownian-interval.md](flux-normal-collapsed-brownian-interval.md) | comfy/k_diffusion/sampling.py + torchsde | executed reference failure on a narrow positive sigma interval |
| [video-matroska-frame-count.md](video-matroska-frame-count.md) | comfy_api/latest/_input_impl/video_types.py | found at 15eb748b3ec5; Dinkster uses marked container estimates |
| [ace15-generation-maximum-uses-minimum.md](ace15-generation-maximum-uses-minimum.md) | comfy/text_encoders/ace15.py | verified at 25dfc16f; Dinkster preserves the composer behavior |
