# comfy-kitchen: CUDA stochastic_rounding_fp8 fails on tensors off the current device

- **Area:** comfy-kitchen `comfy_kitchen/backends/cuda/__init__.py`
  `_wrap_for_dlpack` (repo `Comfy-Org/comfy-kitchen`, observed at
  0.2.22 and 0.2.31 PyPI wheels)
- **Status:** still present in 0.2.31; fixed in Dinkster (device-context
  pin in rounding.py)

## Symptom

On a multi-GPU machine, calling
`comfy_kitchen.stochastic_rounding_fp8` on a `cuda:1` tensor while
`cuda:0` is the current device raises:

```
BufferError: Can't export tensors on a different CUDA device index.
Expected: 1. Current device: 0.
```

Any registry-dispatched CUDA op that goes through `_wrap_for_dlpack`
has the same constraint.

## Root cause

`_wrap_for_dlpack` exports arguments with `tensor.__dlpack__(stream=-1)`.
torch refuses DLPack export of a CUDA tensor whose device index is
not `torch.cuda.current_device()` (the stream argument is ambiguous
otherwise). The kernel wrapper never sets the device context to the
argument's device before exporting.

## Repro

```python
import torch, comfy_kitchen as ck            # >= 2 GPUs
x = torch.randn(64, 64, device="cuda:1")
rng = torch.randint(0, 256, x.shape, dtype=torch.uint8, device="cuda:1")
ck.stochastic_rounding_fp8(x, rng, output_type=torch.float8_e4m3fn)
# BufferError (current device is cuda:0)
```

## Suggested upstream fix

Wrap the kernel invocation in `torch.cuda.device(x.device)` (or pass
an explicit stream for the tensor's own device) inside the CUDA
backend, so callers are not required to manage the current-device
state around every op.

## Dinkster interim handling

`dinkster_inference_torch/rounding.py` pins the device context
(`with torch.cuda.device(value.device)`) around the kitchen kernel
call for CUDA tensors. The direct
`test_kitchen_cuda_stochastic_rounding_wrong_device_canary` keeps cuda:0
current while calling Kitchen with cuda:1 tensors and pins the 0.2.31 failure;
`test_rounding_dispatches_kitchen_cuda_backend` proves Dinkster's guarded route
on every GPU. When upstream fixes the wrapper, the direct canary will fail and
the guard can be removed.
