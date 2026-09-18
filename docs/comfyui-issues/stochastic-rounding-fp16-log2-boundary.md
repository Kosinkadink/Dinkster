# Stochastic fp8 rounding: fp16 log2 exponent defect at power-of-two boundaries

- **Area:** ComfyUI `comfy/float.py` `manual_stochastic_round_to_float8`
  (reference checkout @ 947c2749) and its port in comfy-kitchen's
  eager backend, `comfy_kitchen/backends/eager/quantization.py`
  `stochastic_rounding_fp8` (observed at 0.2.22). The comfy-kitchen
  compiled CUDA kernel is NOT affected - which means upstream's two
  backends disagree bitwise for the same registry operation.
- **Status:** still present in comfy-kitchen 0.2.31; fixed in Dinkster
  (frexp exponent in
  `packages/dinkster-inference-torch/src/dinkster_inference_torch/rounding.py`,
  a documented deliberate divergence); canary test pins the upstream
  eager-vs-CUDA divergence
  (`test_gpu.py::test_kitchen_eager_log2_boundary_divergence_canary`)

## Symptom

For inputs whose fp16 value sits just below a power of two (within
half an fp16 ulp, e.g. `7.99614334` -> fp16 `7.99609375`, just below
`8.0`), the manual/eager stochastic rounding path produces, for low
RNG draws, a value one fp8 ulp BELOW the correct lower neighbor -
off the adjacent-value grid entirely. Example (e4m3fn, rng byte 0):

| input (fp16)   | legal results | eager/manual result | kitchen CUDA result |
| -------------- | ------------- | ------------------- | ------------------- |
| 7.99609375     | 7.5 or 8.0    | **7.0**             | 7.5                 |
| -0.2499...     | -0.234375 or -0.25 | **-0.21875**   | -0.234375           |
| 0.0624428...   | 0.05859375 or 0.0625 | **0.0546875** | 0.05859375          |

For all higher RNG draws the affected inputs round UP with
probability ~255/256 instead of the correct fractional probability -
a bias, but at least on-grid. The off-grid case is what makes the
CUDA kernel and eager backend bitwise-divergent; a randomized parity
test between them flakes with probability proportional to the chance
of drawing a boundary value (this is how the defect was found).

## Root cause

The manual path computes the fp8 exponent as
`torch.floor(torch.log2(abs_x))` with `abs_x` in float16. For
`abs_x` just below `2**k`, the true `log2` is fractionally below `k`,
but fp16 has too little precision near integers: the result rounds
to exactly `k` (fp16 ulp near 3.0 is ~0.002, while
`log2(7.99609375) ~ 2.99930`). `floor` then keeps `k` instead of
`k-1`, the normalized mantissa `abs_x / 2**k - 1` comes out slightly
NEGATIVE, and `floor(mantissa_scaled + rng/256)` yields `-1` for
draws below the deficit - decoding to one ulp below `2**k * 1.0`,
i.e. below the lower neighbor.

The comfy-kitchen CUDA kernel computes the exponent exactly (no fp16
log2), so it truncates onto the grid, and upstream's two backends
implement different numerics for the same operation.

## Repro

```python
import torch, comfy_kitchen as ck
x = torch.tensor([7.99614334], device="cuda:0")
rng = torch.zeros(1, dtype=torch.uint8, device="cuda:0")
cuda = ck.stochastic_rounding_fp8(x, rng.clone(), output_type=torch.float8_e4m3fn)
ck.disable_backend("cuda")
eager = ck.stochastic_rounding_fp8(x, rng.clone(), output_type=torch.float8_e4m3fn)
ck.enable_backend("cuda")
print(cuda.item(), eager.item())  # 7.5 vs 7.0 - eager is off-grid
```

Pure ComfyUI repro (no kitchen): call
`comfy.float.manual_stochastic_round_to_float8` with a generator
seeded so the first draw is < 1/256 and the same input; the result is
7.0 where only 7.5/8.0 are legal.

## Suggested upstream fix

Compute the exponent exactly instead of through fp16 log2, e.g.
`torch.frexp(abs_x).exponent - 1` (exact for every finite nonzero
float: `abs_x = m * 2**e`, `m in [0.5, 1)`), cast back to fp16 for
the downstream arithmetic. Fix BOTH copies (comfy/float.py and
comfy-kitchen eager) or the kitchen eager/CUDA parity stays broken.

## Dinkster handling

`rounding.py::_manual_stochastic_round_to_float8` uses the frexp
exponent (documented deliberate divergence at the code site). All
non-boundary inputs still round bit-identically to the reference -
the executed-reference goldens in `test_patches.py` stay green on the
golden torch build - and boundary inputs now stay on the adjacent
grid, proven by
`test_patches.py::test_rounding_boundary_values_stay_on_grid` (which
the reference algorithm fails). The kitchen-accelerated path is
unaffected (the CUDA kernel was already correct); kitchen's eager
backend is upstream's code and stays defective until fixed there,
pinned by the canary test.

When upstream fixes this, verify bit agreement of the fixed reference
against Dinkster's implementation on boundary values, then fold boundary
inputs into the seeded kernel-vs-eager bitwise test and retire the
canary.
