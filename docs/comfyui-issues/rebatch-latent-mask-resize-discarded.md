# Latent rebatching discards the resized noise mask

Reference: ComfyUI `bc5f880dbdd43e426077f42ccec53bb577ffee9f`,
`comfy_extras/nodes_rebatch.py`, `LatentRebatch.get_batch`.

The mask-size branch calls `torch.nn.functional.interpolate` but does not
assign its result. Rebatching therefore retains the supplied mask's spatial
size even when the branch requests a different size. The height comparison
also uses the latent height rather than eight times that height.

Reproduce in the reference CPU environment:

```python
import torch
from comfy_extras.nodes_rebatch import LatentRebatch

latent = {
    "samples": torch.zeros((2, 4, 2, 3)),
    "noise_mask": torch.full((1, 1, 2, 3), 0.25),
}
output = LatentRebatch.execute([latent], [2]).result[0][0]
assert output["noise_mask"].shape == (2, 1, 2, 3)
```

Assigning the interpolation result and comparing both dimensions against
the intended pixel size would correct the reference. Native rebatching
preserves the fixed reference's mask shape and values instead of silently
changing saved-workflow behavior.

Status: reproduced locally; not reported upstream.
