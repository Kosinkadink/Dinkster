# comfy-kitchen: biased NVFP4 Linear silently dequantizes on Blackwell

- **Area:** comfy-kitchen `comfy_kitchen/tensor/nvfp4.py` tensor dispatch,
  reached by ComfyUI `comfy/ops.py` mixed-precision Linear
- **Status:** retained by comfy-kitchen 0.2.31; Dinkster's direct NVFP4
  kernel route is unaffected

Baseline: ComfyUI
`2eb609766a749e3104485979615e062e401bab97`, comfy-kitchen 0.2.26,
torch 2.10.0+cu130, CUDA 13.0, NVIDIA RTX PRO 6000 Blackwell (SM 12.0).
The 0.2.31 source and its `test_nvfp4_addmm_fallback` still explicitly
dequantize `addmm`; a fresh Blackwell runtime receipt is tracked by Dinkster #77.

## Symptom

ComfyUI loads the official FLUX.1-dev-NVFP4 artifact and quantizes each Linear
input, but every official-model NVFP4 Linear silently dequantizes both operands
instead of calling the available Blackwell `scaled_mm_nvfp4` kernel. A full
bounded warmup over all 152 NVFP4 Linears recorded:

```
quantize_nvfp4:   304
scaled_mm_nvfp4:    0
dequantize_nvfp4: 608
```

All 152 NVFP4 Linears in that artifact have a bias. A minimal isolated dispatch
diagnosis under the same interpreter and device records:

```
biasless NVFP4 Linear: scaled_mm_nvfp4=1, dequantize_nvfp4=0
biased NVFP4 Linear:   scaled_mm_nvfp4=0, dequantize_nvfp4=2
```

The workflow still computes finite output, so users receive no error or warning
that the advertised accelerated NVFP4 route was lost.

## Root cause

In this torch execution path, biasless `torch.nn.functional.linear` lowers to
`aten.mm`, while the biased form lowers to `aten.addmm`. comfy-kitchen 0.2.26
registers `TensorCoreNVFP4Layout` handlers for `aten.linear` and `aten.mm`, but
not `aten.addmm`. `QuantizedTensor.__torch_dispatch__` therefore takes its
generic unhandled-op fallback for the biased form and recursively dequantizes
the quantized input and weight before running `addmm`.

The existing NVFP4 `aten.linear` handler already accepts a bias and passes it to
`ck.scaled_mm_nvfp4`; it is simply bypassed by the decomposition that production
ComfyUI reaches. This also explains why a biasless synthetic Linear appears to
prove acceleration while the official model does not.

## Repro

Run this from the pinned ComfyUI checkout with the acceptance interpreter on an
SM10+ CUDA device. Importing `comfy.quant_ops` installs ComfyUI's NVFP4 layout;
the wrapper counters then prove which Kitchen path each call reaches:

```python
import comfy.quant_ops  # noqa: F401
import comfy_kitchen as kitchen
import torch
from comfy_kitchen.tensor import QuantizedTensor

x = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
w = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
bias = torch.randn(16, device="cuda", dtype=torch.bfloat16)
x_nvfp4 = QuantizedTensor.from_float(x, "TensorCoreNVFP4Layout")
w_nvfp4 = QuantizedTensor.from_float(w, "TensorCoreNVFP4Layout")

counts = {"dequantize": 0, "scaled_mm": 0}
original_dequantize = kitchen.dequantize_nvfp4
original_scaled_mm = kitchen.scaled_mm_nvfp4

def counted_dequantize(*args, **kwargs):
    counts["dequantize"] += 1
    return original_dequantize(*args, **kwargs)

def counted_scaled_mm(*args, **kwargs):
    counts["scaled_mm"] += 1
    return original_scaled_mm(*args, **kwargs)

kitchen.dequantize_nvfp4 = counted_dequantize
kitchen.scaled_mm_nvfp4 = counted_scaled_mm

torch.nn.functional.linear(x_nvfp4, w_nvfp4, None)
assert counts == {"dequantize": 0, "scaled_mm": 1}

counts.update(dequantize=0, scaled_mm=0)
torch.nn.functional.linear(x_nvfp4, w_nvfp4, bias)
assert counts == {"dequantize": 2, "scaled_mm": 0}
```

The retained diagnosis and full-model record are sealed under
`/home/kosin/ComfyUI-Shared/inference-parity-evidence/`
`nvfp4-flux-v1-20260807/`. The diagnosis SHA256 is
`ab5db6a0755ff904d12efa9ea703fee2e1c0cf03f925db76af6444bac700d664`.

## Suggested upstream fix

Register an NVFP4 `aten.addmm` handler that preserves the supported
`bias + input @ weight.T` semantics and dispatches the compatible quantized
operands to `ck.scaled_mm_nvfp4`, following the existing NVFP4 Linear handler
and the addmm coverage already present for other Kitchen layouts. Alternatively,
ensure the ComfyUI mixed-precision Linear invokes a non-decomposed NVFP4 handler
for the biased case.

The regression test must use a non-None bias on SM10+ and prove selected backend
identity, not merely finite output. It should assert one accelerated scaled
matmul and zero dequantization calls. A full official-artifact rerun should then
prove all 152 biased NVFP4 Linears stay on the accelerated route before
cross-engine acceptance is called GREEN.

## Dinkster interim handling

Dinkster's package-internal `Nvfp4Linear` calls Kitchen input quantization and
`scaled_mm_nvfp4` directly, including bias, after an explicit SM10+ CUDA backend
capability proof. Its 2026-08-07 official-artifact gate completed two bounded
phases with 304 CUDA quantizations and 304 CUDA scaled matmuls per phase, zero
NVFP4 dequantizations, byte-identical repeated outputs, exact packed-state
offload/reload, and clean process/device teardown.

CPU/non-SM10 execution remains a deliberate dequantize-plus-`F.linear`
fallback. Missing capability chooses that fallback; corruption, a selected
kernel failure, OOM, and cancellation remain loud. The upstream defect blocked
only accelerated cross-engine comparison; it did not invalidate Dinkster's GREEN
direct-runtime result.
