# Legacy scaled-fp8 conversion ignores the marker dtype

- **Area:** ComfyUI `comfy/utils.py` `convert_old_quants`, legacy
  `scaled_fp8` branch (reference checkout @
  947c2749dd04c51ef0e21b069544d8b0b4f9b411)
- **Status:** found 2026-07-24; not reported upstream yet; handled in
  Dinkster by honoring the marker dtype

## Symptom

A legacy scaled-fp8 checkpoint whose `scaled_fp8` marker has dtype
`float8_e5m2` is converted to layer metadata claiming that its weights
use `float8_e4m3fn`. Loading therefore silently changes the stored
weight numerics instead of preserving the checkpoint's fp8 format.

## Root cause

`convert_old_quants` correctly computes `scaled_fp8_dtype` from the
marker tensor: a float32 marker means `float8_e4m3fn`, while any other
marker retains its own dtype. The computed variable is never used.
Every converted layer config instead hardcodes:

```python
{"format": "float8_e4m3fn"}
```

For an e5m2 marker, `_load_quantized_module` consequently calls
`weight.to(storage_t)` with e4m3fn as the destination. That operation
value-converts the e5m2 qdata to e4m3fn and silently changes its
numerics.

## Repro

Construct a legacy state dict containing a `scaled_fp8` marker tensor
with dtype `torch.float8_e5m2`, an e5m2 `<layer>.weight`, and a scalar
float32 `<layer>.scale_weight`. Pass it through `convert_old_quants`.
The resulting `<layer>.comfy_quant` configuration reports
`float8_e4m3fn`, despite the marker and weight both being e5m2.

## Suggested upstream fix

Use the already-derived `scaled_fp8_dtype` when constructing each
layer config, writing the format name represented by the marker dtype
instead of hardcoding `float8_e4m3fn`.

## Dinkster handling

`dinkster_inference.quantization._split_legacy` honors the marker dtype:
float32 markers retain the historical e4m3fn meaning, while e5m2
markers produce `float8_e5m2` layer descriptors. The code documents
this as a deliberate divergence from ComfyUI and validates that each
stored weight dtype matches the derived format rather than silently
value-converting it.
