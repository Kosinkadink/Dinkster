# Asymmetric W4A8 checkpoint contract is internally inconsistent

- **Area:** ComfyUI `comfy/quant_ops.py` and `comfy/ops.py` at
  `8f37cf8c833a8f2d3c62e2adbccebfd165623481`; comfy-kitchen
  0.2.31 at `7c6ca3a5b63857d42c2d49777d6afb69de23f13f`
- **Status:** still present with comfy-kitchen 0.2.31 and current ComfyUI;
  typed and refused by Dinkster

## Symptom

The three pinned authorities disagree about the payload keys for
`asym_w4a8_int8`. A checkpoint contract derived from the registry cannot be
loaded, while a complete Kitchen-produced state dict may contain correction
data that ComfyUI silently has no path to restore.

## Root cause

`QUANT_ALGOS["asym_w4a8_int8"]["parameters"]` declares only
`{"weight_scale"}`. The `_load_quantized_module` W4A8 branch never consumes
that name; it reads `weight_s_rel`, `weight_s_channel`, and optional
`weight_codebook` instead.

Kitchen's `QuantizedTensor.state_dict(prefix="weight")` is the producer
authority for checkpoint bytes. It emits `weight`, `weight_s_rel`,
`weight_s_channel`, optional `weight_correction`, and optional
`weight_codebook`. ComfyUI's loader never reads or passes
`weight_correction`, so correction-bearing producer payloads are not fully
load-compatible with the pinned ComfyUI revision.

## Repro

Serialize an asymmetric W4A8 Kitchen tensor with correction data under the
`weight` prefix. Observe the five keys above. Compare them with the ComfyUI
registry's declared `weight_scale`, then load through `_load_quantized_module`:
the loader has no correction-key branch and cannot reconstruct the complete
Kitchen parameter state.

## Suggested upstream fix

Make the registry declaration, state-dict producer, and loader share one
authoritative payload vocabulary. Include explicit correction handling or
reject correction-bearing payloads loudly; do not silently omit the state.
Add a producer-to-loader round-trip test covering both optional tensors.

## Dinkster handling

Dinkster models the exact Kitchen producer vocabulary as checkpoint identity and
does not accept the unrelated `weight_scale` spelling. Correction and codebook
presence rotate structural identity and are never dropped or normalized.
W4A8 remains a typed, strict unsupported outcome: assembly refuses it before
provider or runtime selection. Execution revives only after upstream loader
reconciliation and a real licensed consumer/provider acceptance plan.
