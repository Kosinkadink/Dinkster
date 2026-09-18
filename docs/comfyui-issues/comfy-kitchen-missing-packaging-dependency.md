# comfy-kitchen: 0.2.31 wheel omits its packaging dependency

- **Area:** comfy-kitchen PyPI metadata and `scaled_mm_v2.py`
- **Status:** present in the 0.2.31 PyPI wheel; handled in Dinkster

## Symptom

Installing comfy-kitchen 0.2.31 into a fresh torch environment and importing
`comfy_kitchen` fails with `ModuleNotFoundError: No module named 'packaging'`.

## Root cause

`comfy_kitchen/scaled_mm_v2.py` imports `packaging.version`, but the wheel does
not declare `packaging` as a runtime dependency.

## Suggested upstream fix

Declare `packaging` in comfy-kitchen's runtime dependencies and publish a new
wheel.

## Dinkster handling

The inference-torch extra and validation environments install `packaging`
alongside the pinned comfy-kitchen 0.2.31 wheel.
