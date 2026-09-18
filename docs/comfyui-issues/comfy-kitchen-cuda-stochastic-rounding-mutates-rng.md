# comfy-kitchen: CUDA stochastic_rounding_fp8 mutates its rng argument in place

- **Area:** comfy-kitchen `comfy_kitchen/backends/cuda` compiled
  extension (repo `Comfy-Org/comfy-kitchen`, observed through 0.2.31,
  PyPI wheel with prebuilt `_C.abi3.so`)
- **Status:** still present in 0.2.31; Dinkster immune by construction (rng
  allocated fresh per call); canary test pins the behavior

## Symptom

Calling `comfy_kitchen.stochastic_rounding_fp8(x, rng, output_type)`
on CUDA tensors overwrites the caller's `rng` tensor. The eager
backend does not (its `calc_mantissa` does `rng.to(dtype=...)`, a
copy). A caller that reuses one rng tensor across calls - or, as in
a naive A/B comparison, calls the CUDA backend and then the eager
backend with the "same" rng - silently gets different randomness on
the second use.

This cost real debugging time here: a CUDA-vs-eager parity probe that
passed the same rng tensor to both backends reported ~31% byte
divergence and an apparent statistical bias in the second backend's
output. Both artifacts vanish when each call gets its own copy; on
fresh inputs the two backends are bit-identical.

## Root cause

The compiled CUDA kernel treats the rng buffer as scratch space (it
is handed over via DLPack, `_wrap_for_dlpack(rng)`, with no copy),
whereas the eager reference treats it as read-only input. The two
backends implement different contracts for the same registry
operation.

## Repro

```python
import torch, comfy_kitchen as ck
x = torch.randn(64, 64, device="cuda:0")
rng = torch.randint(0, 256, x.shape, dtype=torch.uint8, device="cuda:0")
snap = rng.clone()
ck.stochastic_rounding_fp8(x, rng, output_type=torch.float8_e4m3fn)
assert torch.equal(rng, snap)  # fails: rng was overwritten
```

## Suggested upstream fix

Either make the CUDA kernel consume rng read-only (copy internally if
scratch space is needed), or document the argument as consumed and
make the eager backend match. Backends registered under one operation
name should share one contract.

## Dinkster interim handling

`dinkster_inference_torch/rounding.py` allocates the rng tensor fresh
from a seeded generator on every call and never reuses it, so Dinkster's
outputs are unaffected. The quirk is pinned by
`tests/test_gpu.py::test_kitchen_cuda_rng_mutation_canary` - if that
test fails after a kitchen upgrade, upstream fixed the kernel: update
this status line and drop the defensive `rng.clone()` calls in
`test_kitchen_cuda_kernel_matches_eager_bitwise`.
