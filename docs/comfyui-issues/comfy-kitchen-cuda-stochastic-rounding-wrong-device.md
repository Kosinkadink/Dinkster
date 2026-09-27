# comfy-kitchen: CUDA stochastic_rounding_fp8 fails on tensors off the current device

- **Area:** comfy-kitchen `comfy_kitchen/backends/cuda/__init__.py`
  `_wrap_for_dlpack` (repo `Comfy-Org/comfy-kitchen`, observed at
  0.2.22 and 0.2.31 PyPI wheels)
- **Status:** still present in 0.2.31

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

## Dinkster handling

Dinkster delegates stochastic rounding to `dinkster_comfy`, whose current
Kitchen dispatch matches the upstream behavior described above.
