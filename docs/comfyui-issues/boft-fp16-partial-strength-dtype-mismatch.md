# BOFT patches never apply to fp16 weights at strength != 1 (einsum dtype mismatch, silently swallowed)

Status: found 2026-07 (Dinkster stage-4b slice-1 golden generation). Not
yet reported upstream.

Baseline: ComfyUI @ 947c2749dd04c51ef0e21b069544d8b0b4f9b411, torch
2.9.1+cu130 (also reproduced on torch 2.13.0+cpu).

## Symptom

A BOFT adapter applied to a float16 (or bfloat16) weight with any
strength other than exactly 1.0 NEVER applies. `calculate_weight`
raises internally, the `except` logs `ERROR boft <key> expected
scalar type Float but found Half` and returns the weight UNPATCHED -
silently missing LoRA, normal-looking run. Strength 1.0 works, so
the bug surfaces exactly when a user drags the strength slider.

## Root cause

`comfy/weight_adapter/boft.py calculate_weight`:

```python
I = torch.eye(boft_b, device=blocks.device, dtype=blocks.dtype)  # intermediate dtype (fp32)
...
r = r.to(weight)              # r becomes weight dtype (fp16)
...
bi = r[i]                     # fp16
if strength != 1:
    bi = bi * strength + (1 - strength) * I   # fp16 + fp32 -> promotes bi to fp32
...
inp = torch.einsum("b i j, b j ...-> b i ...", bi, inp)  # fp32 x fp16 -> RuntimeError
```

The strength interpolation mixes the fp16 `bi` with the fp32
intermediate-dtype identity `I`; type promotion makes `bi` fp32, and
`torch.einsum` requires matching operand dtypes, so it raises for
every fp16 weight when `strength != 1`. The sibling OFT adapter does
not have this bug because it builds a second identity in the
weight's dtype (`I_w`) for its einsum.

## Suggested upstream fix

Interpolate with an identity in `bi`'s dtype (mirroring what oft.py
already does):

```python
if strength != 1:
    bi = bi * strength + (1 - strength) * I.to(bi)
```

(or hoist `I_w = I.to(weight)` once before the loop). Numerics at
fp16 change only in the sense that the currently-impossible path
starts working.

A test that would have caught it: any BOFT fixture on an fp16 weight
with strength 0.5.

## Repro

```python
import torch
blocks = torch.randn(2, 2, 4, 4) * 0.1   # fp32 intermediate
weight = torch.zeros(8, 6, dtype=torch.float16)
I = torch.eye(4, dtype=blocks.dtype)
q = blocks - blocks.transpose(-1, -2)
r = ((I + q) @ (I - q).float().inverse()).to(weight)
bi = r[0] * 0.5 + 0.5 * I                 # promotes to fp32
inp = weight.unflatten(0, (-1, 4))
torch.einsum("b i j, b j ...-> b i ...", bi, inp)  # RuntimeError: expected Float, found Half
```

## Dinkster handling

FIXED in Dinkster (2026-07): `BOFTAdapter.calculate` interpolates
partial strength with an identity in `bi`'s dtype (mirroring
oft.py's `eye_w`), so fp16 + partial strength applies correctly;
float32 behavior is numerically unchanged (goldens still pass).
Since no upstream oracle golden can exist while ComfyUI is broken,
correctness is verified independently -
`test_boft_fp16_partial_strength_fixed_beyond_reference` in
`packages/dinkster-inference-torch/tests/test_adapters.py` cross-checks
the fp16 result against the same inputs run at float32 (a
golden-covered path) within fp16 tolerance. Ledgered in ROADMAP
"Upstream-broken adapter paths: oracle cross-check pending". When
upstream fixes this, regenerate `tools/gen_adapter_goldens.py`
goldens including this case (fp16 + partial strength) and
cross-check Dinkster against the fixed reference.
