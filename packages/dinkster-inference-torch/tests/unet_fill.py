"""Deterministic UNet golden weight fill, shared with the generator.

Same scheme as kl_fill.py (which documents the hash): every tensor
fills from a pure 32-bit integer hash of its KEY NAME, bit-identical
across torch versions and platforms and independent of module
construction order.

Norm detection differs from the KL rule (".norm" in the key): the
UNet's GroupNorm scales live at keys like ``in_layers.0.weight``,
``out_layers.0.weight``, and ``out.0.weight`` with no "norm" in the
name. In this architecture a rank-1 ``.weight`` IS a norm scale
(conv/linear weights are rank >= 2, biases are ``.bias``), so the
rule keys on rank: rank-1 weights center on 1.0, everything else is
uniform in [-0.2, 0.2].
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Sequence

import torch
from kl_fill import hash_uniform


def fill_value(key: str, shape: Sequence[int]) -> torch.Tensor:
    """The deterministic float32 tensor for one state-dict entry."""
    count = math.prod(shape) if shape else 1
    uniform = hash_uniform(zlib.crc32(key.encode("utf-8")), count)
    if key.endswith(".weight") and len(shape) == 1:
        values = 1.0 + (uniform - 0.5) * 0.1
    else:
        values = (uniform - 0.5) * 0.4
    return values.reshape(tuple(shape))


def fill_state_dict(
    entries: Sequence[tuple[str, Sequence[int]]],
) -> dict[str, torch.Tensor]:
    """Fill every (key, shape) entry; order-independent by design."""
    return {key: fill_value(key, shape) for key, shape in entries}


def hashed_input(key: str, shape: Sequence[int]) -> torch.Tensor:
    """Deterministic activations from the same hash, shifted to
    roughly unit scale under an ``input:`` pseudo-key namespace -
    the golden generator and the replay tests build bit-identical
    inputs from the case name."""
    return fill_value(f"input:{key}", shape) * 5.0
