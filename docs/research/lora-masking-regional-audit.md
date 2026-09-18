# LoRA masking and regional execution audit

This audit distinguishes conditioning-scoped regional LoRA behavior from
tensor-level masking of LoRA weights. The interoperable behavior is separate
patched denoiser evaluation for each conditioning region, followed by spatial
output blending. None of the audited implementations defines a general mask
that multiplies arbitrary Linear or Conv2d LoRA deltas inside a layer.

## Audited revisions

| Project | Revision |
| --- | --- |
| ComfyUI product baseline | [`947c2749`](https://github.com/Comfy-Org/ComfyUI/tree/947c2749dd04c51ef0e21b069544d8b0b4f9b411) |
| ComfyUI current source | [`a7365071`](https://github.com/Comfy-Org/ComfyUI/tree/a7365071e47175fb06572d0a56d1bf4116c2f581) |
| Advanced-ControlNet | [`27a67fee`](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet/tree/27a67fee80cf46c198b10588a5651f23d67d1e95) |
| AnimateDiff-Evolved | [`92576512`](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/tree/9257651221002dcba0a12f9cff37e1944e58fb60) |
| VideoHelperSuite | [`4ee72c06`](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/tree/4ee72c065db22c9d96c2427954dc69e7b908444b) |
| Impact Pack | [`429d0159`](https://github.com/ltdrdata/ComfyUI-Impact-Pack/tree/429d0159ad429e64d2b3916e6e7be9c22d025c3c) |
| IPAdapter Plus | [`a0f451a5`](https://github.com/cubiq/ComfyUI_IPAdapter_plus/tree/a0f451a5113cf9becb0847b92884cb10cbdec0ef) |
| rgthree-comfy | [`6b76ee6f`](https://github.com/rgthree/rgthree-comfy/tree/6b76ee6f2c5a007710b5a16f97c94330d6ecc871) |

The earlier 829-pack ecosystem scan remains relevant: 39.2 percent of
usage-weighted packs use ModelPatcher while only 1.8 percent use the official
hooks API. Regional execution must therefore expose a simple native contract
rather than require packs to reproduce host internals.

## Behavior matrix

| System | Mask affects | Schedule | Composition | Native regional LoRA contract? |
| --- | --- | --- | --- | --- |
| ComfyUI hooks | Conditioning crop and denoiser-output multiplier | Hook keyframes and conditioning percent range | Group by hook identity, evaluate separately, normalize overlap, fill uncovered default region | Yes |
| AnimateDiff-Evolved LoRA hooks | ComfyUI conditioning hook regions | Percent keyframes, discrete or interpolated strength lists, guarantee steps | Cloned hook groups attached to conditioning | Yes, through the ComfyUI hook contract |
| Advanced-ControlNet | Control hint, attention-patch effect, latent keyframes | Percent range and timestep/latent keyframes | Previous control first, then current contribution | No; ControlNet and LLLite semantics are distinct |
| IPAdapter Plus | K/V attention patch influence | Start/end percent | Tiled patches accumulate in application order | No; attention-adapter semantics are distinct |
| Impact Pack | Latent noise mask, regional sampler pass, compositing | Sampler/pass specific | Sequential passes and explicit overlap/restoration policy | No |
| VideoHelperSuite | Batch/frame mask transformations and aligned latent properties | Frame selection | Split/concat/select order | No |
| rgthree-comfy | No mask; global LoRA stack only | None | Sequential UI/input order, separate model and CLIP strengths | No |

## ComfyUI reference semantics

`ConditioningSetMask` stores a mask, mask strength, and optional mask-bounds
flag. `get_area_and_mult` resizes and crops the mask, applies the conditioning
strength, and evaluates the model on the resulting area. Regional outputs and
their multipliers are accumulated and normalized. Hooked conditions are
partitioned by hook identity so each region runs under its own LoRA patch
state. The default condition fills only the uncovered residual.

Relevant current-source locations:

- [`nodes.py`](https://github.com/Comfy-Org/ComfyUI/blob/a7365071e47175fb06572d0a56d1bf4116c2f581/nodes.py#L210-L270) defines explicit-area and mask metadata.
- [`hooks.py`](https://github.com/Comfy-Org/ComfyUI/blob/a7365071e47175fb06572d0a56d1bf4116c2f581/comfy/hooks.py#L639-L755) binds ordered LoRA hooks, masks, and timestep ranges to conditioning.
- [`samplers.py`](https://github.com/Comfy-Org/ComfyUI/blob/a7365071e47175fb06572d0a56d1bf4116c2f581/comfy/samplers.py#L33-L92) materializes regional crops and multipliers.
- [`samplers.py`](https://github.com/Comfy-Org/ComfyUI/blob/a7365071e47175fb06572d0a56d1bf4116c2f581/comfy/samplers.py#L238-L355) groups hook states and accumulates normalized regional outputs.

The pinned product baseline uses the same ordering in
`comfy/hooks.py::set_conds_props`: attach hooks, apply mask metadata, then apply
the timestep range. A two-dimensional mask gains a leading batch dimension.
`set_cond_area="mask bounds"` derives the execution crop from the resized mask.

## Custom-node boundaries

AnimateDiff-Evolved is the direct compatibility target. Its LoRA hook loader
does not accept a mask itself; masking becomes effective when the hook is
attached to masked conditioning. Its motion-model `scale_multival` and
`effect_multival` masks are separate per-frame motion controls and must not be
reinterpreted as LoRA masks.

Advanced-ControlNet combines effect masks, timestep keyframes, latent
keyframes, per-layer weights, conditional/unconditional multipliers, and
control chaining. LLLite uses token-space attention masks and explicitly does
not mask Conv2d patches. IPAdapter Plus also masks attention adapter influence,
including tiled overlap. Impact Pack performs sequential masked sampling and
compositing. VideoHelperSuite establishes frame/batch alignment operations.
These are explicit future extension contracts, not aliases for regional LoRA.

## Dinkster contract

Dinkster's native contract carries an optional float32 mask, mask strength,
mask-bounds policy, closed conditioning percent range, and ordered LoRA hook
schedule on each canonical conditioning record. Sampling partitions records by
patch identity, evaluates the selected crop under that patch state, blends by
the materialized mask, normalizes overlap, and preserves ordinary default-region
coverage. The same mask payload is content-addressed and reused across hook
keyframe segments.

The contract deliberately excludes arbitrary per-layer spatial LoRA
modulation, ControlNet effect masks, motion-model multivals, and sequential
masked sampler passes. Standard SD1.5 IP-Adapter output masks use their own
typed attention-contribution surface; they are not conditioning regions or
LoRA modulation. Other behaviors require their own typed extension surfaces
and must fail explicitly rather than be approximated as conditioning regions.
