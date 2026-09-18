# comfy-kitchen: Triton apply_rope fails on tensors off the current device

- **Area:** comfy-kitchen `comfy_kitchen/backends/triton/rope.py`
  (repo `Comfy-Org/comfy-kitchen`, observed at 0.2.22 and 0.2.31)
- **Status:** still present in 0.2.31; fixed in Dinkster (device-context
  pin in flux.py apply_rope)

## Symptom

On a multi-GPU machine, calling `comfy_kitchen.apply_rope` on
`cuda:1` tensors AFTER any prior call on `cuda:0` raises:

```
ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
```

A fresh process whose first call is on `cuda:1` works, which makes
the failure order-dependent: it only appears once the Triton
launcher has been cached from a launch on another device. ComfyUI
itself hits this path unconditionally for every non-training Flux
forward (comfy/ldm/flux/math.py apply_rope @ 947c2749 calls
`comfy.quant_ops.ck.apply_rope` directly), so any upstream multi-GPU
setup that runs Flux models on a device other than the current one
inherits the same crash.

## Root cause

The Triton kernel is launched on the CURRENT CUDA device's stream
(`torch.cuda.current_device()`), not the device of its tensor
arguments. Once the JIT launcher is compiled and cached for cuda:0,
a launch with cuda:1 pointers passes pointers that the cuda:0
context cannot access, and Triton's pointer validation rejects them.
Same family as the DLPack current-device constraint in the CUDA
backend (comfy-kitchen-cuda-stochastic-rounding-wrong-device.md),
different backend and failure mode.

## Repro

```python
import torch, comfy_kitchen as ck            # >= 2 GPUs
def mk(device):
    q = torch.randn(1, 2, 8, 16, device=device)
    k = torch.randn(1, 2, 8, 16, device=device)
    freqs = torch.randn(1, 1, 8, 8, 2, 2, device=device)
    return q, k, freqs
ck.apply_rope(*mk("cuda:0"))   # OK, caches the launcher
ck.apply_rope(*mk("cuda:1"))   # ValueError: Pointer argument ...
```

Wrapping the second call in `with torch.cuda.device("cuda:1"):`
succeeds and matches the eager math (max |diff| ~1e-7 fp32).

## Suggested upstream fix

Enter `torch.cuda.device(xq.device)` (or launch on an explicit
stream of the argument's device) inside the Triton backend wrapper
before kernel launch, so callers are not required to manage
current-device state around every op.

## Dinkster interim handling

`dinkster_inference_torch/flux.py` `apply_rope` pins the device context
(`with torch.cuda.device(xq.device)`) around the kitchen call for
CUDA tensors. The direct `test_kitchen_triton_rope_wrong_device_canary`
launches Kitchen's Triton backend on cuda:0 and then calls it with cuda:1
tensors while cuda:0 remains current, pinning the 0.2.31 failure. Dinkster's Flux
tests prove the guarded eager route separately. When upstream fixes the
launcher, the direct canary will fail and the guard can be removed.
