"""Deterministic LTX video VAE golden weight fill, shared with the
generator (tools/gen_ltx_vae_goldens.py).

Wraps the unet_fill rank rule with two documented exceptions:

- ``per_channel_statistics.std-of-means`` is a rank-1 buffer without a
  ``.weight`` suffix, so the rank rule would center it near zero - and
  the encoder DIVIDES normalized latents by it. It borrows the norm
  fill (1.0 +- 0.05, strictly positive) under a ``.weight`` pseudo-key.
- ``decoder.timestep_scale_multiplier`` is a scalar the decoder
  multiplies into the conditioning timestep before the sinusoidal
  embedding; a near-zero hash value would collapse every frequency to
  its linear regime. It pins the reference checkpoints' published
  value, 1000.0.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from unet_fill import fill_value


def fill_vae_state_dict(
    entries: Sequence[tuple[str, Sequence[int]]],
) -> dict[str, torch.Tensor]:
    """Fill every (key, shape) entry; order-independent by design."""
    filled: dict[str, torch.Tensor] = {}
    for key, shape in entries:
        if key.endswith("std-of-means"):
            filled[key] = fill_value(key + ".weight", shape)
        elif key.endswith("timestep_scale_multiplier"):
            filled[key] = torch.full(tuple(shape), 1000.0, dtype=torch.float32)
        else:
            filled[key] = fill_value(key, shape)
    return filled
