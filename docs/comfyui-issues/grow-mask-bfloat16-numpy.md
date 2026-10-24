# GrowMask rejects bfloat16 masks

Status: found locally; not reported upstream.

Reference: ComfyUI bc5f880dbdd43e426077f42ccec53bb577ffee9f,
`comfy_extras/nodes_mask.py`, `GrowMask.expand_mask`.

The node converts the input tensor directly to NumPy before applying SciPy
morphology. Torch cannot expose a bfloat16 tensor through `.numpy()`, so a live
CPU bfloat16 MASK fails with `TypeError: Got unsupported ScalarType BFloat16`.
This was reproduced in both the AST-extracted reference method and the native
node with a CPU tensor of shape `(1, 3, 5)`, expand=1, tapered_corners=True.

Minimal boundary reproduction:

```python
import torch

torch.ones((1, 3, 5), dtype=torch.bfloat16).numpy()
```

An upstream fix could convert unsupported NumPy dtypes to float32 before the
morphology operation and explicitly specify the output dtype contract. Dinkster
currently preserves the reference behavior; normal decoded float32 MASK values
are unaffected. CUDA behavior was not exercised, and no device-specific crash
is claimed by this CPU observation.
