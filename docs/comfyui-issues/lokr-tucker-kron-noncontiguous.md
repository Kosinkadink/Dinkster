# LoKr Tucker+conv patches never apply (kron view error, silently swallowed)

Status: found 2026-07 (Dinkster stage-4b slice-1 golden generation). Not
yet reported upstream.

Baseline: ComfyUI @ 947c2749dd04c51ef0e21b069544d8b0b4f9b411, torch
2.9.1+cu130 (also reproduced on torch 2.13.0+cpu).

## Symptom

A LoKr LoRA that uses Tucker decomposition (`lokr_t2` present) on a
conv weight NEVER applies. `calculate_weight` raises internally, the
`except` logs `ERROR lokr <key> view size is not compatible with
input tensor's size and stride ...` and returns the weight
UNPATCHED - the user gets a normal-looking run with the LoRA
silently missing on every affected key.

## Root cause

`comfy/weight_adapter/lokr.py calculate_weight`: when `w2` is
rebuilt from the Tucker factors,

```python
w2 = torch.einsum("i j k l, j r, i p -> p r k l", t2, w2_b, w2_a)
...
lora_diff = torch.kron(w1, w2).reshape(weight.shape)
```

the einsum output is non-contiguous (permuted view), and
`torch.kron` internally uses `view`-based reshapes that require
contiguous inputs, so it raises the view error for every such
tensor. This is not shape-dependent: probed multiple (i, j, p, r)
combinations, the einsum result is never contiguous, so the Tucker
path cannot succeed at all at this pin.

## Suggested upstream fix

Make the operand contiguous before the kron:

```python
lora_diff = torch.kron(w1, w2.contiguous()).reshape(weight.shape)
```

(or `.contiguous()` on the einsum result where it is built). One
line; adds a copy only on the path that currently cannot run at all.

A test that would have caught it: any lokr fixture with `t2` set on
a 4-D weight.

## Repro

```python
import torch
g = torch.Generator().manual_seed(1)
t = lambda *s: torch.randn(s, generator=g)
t2, w2b, w2a = t(3, 2, 3, 3), t(2, 2), t(3, 4)
w2 = torch.einsum("i j k l, j r, i p -> p r k l", t2, w2b, w2a)
w1 = t(2, 3).unsqueeze(2).unsqueeze(2)
torch.kron(w1, w2)  # RuntimeError: view size is not compatible ...
```

## Dinkster handling

FIXED in Dinkster (2026-07): `LoKrAdapter.calculate` makes the Tucker
einsum output contiguous before `torch.kron`, so the path applies
correctly. Since no upstream oracle golden can exist while ComfyUI is
broken, correctness is verified independently -
`test_lokr_tucker_conv_fixed_beyond_reference` in
`packages/dinkster-inference-torch/tests/test_adapters.py` checks the
result against a float64 explicit-loop reconstruction (Tucker +
Kronecker from their definitions). Ledgered in ROADMAP
"Upstream-broken adapter paths: oracle cross-check pending". When
upstream fixes this, regenerate `tools/gen_adapter_goldens.py`
goldens including this case and cross-check Dinkster against the fixed
reference.
