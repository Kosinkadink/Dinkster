# T5 accepts a hidden-layer index past its final block

Status: verified in pinned source; upstream has not been contacted.

Reference: [ComfyUI 25dfc16f](https://github.com/Comfy-Org/ComfyUI/commit/25dfc16f9ac0a87991d34fbf5f02d6c25c844639).

## Symptom and cause

Selecting a positive hidden-layer index equal to the T5 layer count
produces no intermediate tensor and fails when encoding calls `.float()`
on that missing output.

[`SDClipModel.set_clip_options`](https://github.com/Comfy-Org/ComfyUI/blob/25dfc16f9ac0a87991d34fbf5f02d6c25c844639/comfy/sd1_clip.py#L152-L164)
resets to the final-output policy only when `abs(layer_idx) > num_layers`.
Equality therefore selects hidden mode. However,
[`T5Stack.forward`](https://github.com/Comfy-Org/ComfyUI/blob/25dfc16f9ac0a87991d34fbf5f02d6c25c844639/comfy/text_encoders/t5.py#L202-L223)
enumerates blocks from zero through `num_layers - 1`; no block matches
the positive end index. The stack leaves its intermediate result as None.
The [encoding path](https://github.com/Comfy-Org/ComfyUI/blob/25dfc16f9ac0a87991d34fbf5f02d6c25c844639/comfy/sd1_clip.py#L272-L284)
uses `outputs[1].float()` in hidden mode without a fallback.

## Reproduction boundary

For a three-block T5 tower, apply `set_clip_options({"layer": 3})` and
encode a prompt. The setter chooses hidden mode, the stack captures no
intermediate, and the output conversion raises AttributeError. Index 2
selects the final block; index -3 selects block zero. Indices -4 and 4
instead trigger the setter's final-output fallback.

The upstream setter should distinguish valid zero-based and negative
indices with `-num_layers <= layer_idx < num_layers`, and explicitly
handle indices outside that range before selecting hidden mode.

## Dinkster behavior

`T5Stack` and `T5TextModel` reject out-of-range per-call indices with
ValueError. `T5TextEncoder` preserves the reference's oversized-index
fallback when `abs(hidden_layer) > num_layers`, but positive equality
raises ValueError instead of failing on a missing tensor. The negative
boundary remains valid. Valid captures are normalized post-block states;
omitting the selection preserves final output. This makes the invalid
boundary explicit without changing valid outputs or default encoding.
