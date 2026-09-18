"""Deterministic CLIP text-model golden fill, shared with the generator.

Same scheme as kl_fill.py (which documents the hash): every tensor is
filled from a pure 32-bit integer hash of its KEY NAME, bit-identical
across torch versions and platforms and independent of module
construction order. Layer-norm scale weights center on 1.0 so
activations survive deep pre-norm stacks; everything else is uniform
in [-0.2, 0.2].

Textual-inversion vectors for golden cases come from the same hash
under an ``embedding:{name}`` pseudo-key, so the reference run and
the Dinkster replay substitute bit-identical rows.
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
    if key.endswith(".weight") and "layer_norm" in key:
        values = 1.0 + (uniform - 0.5) * 0.1
    else:
        values = (uniform - 0.5) * 0.4
    return values.reshape(tuple(shape))


def fill_state_dict(
    entries: Sequence[tuple[str, Sequence[int]]],
) -> dict[str, torch.Tensor]:
    """Fill every (key, shape) entry; order-independent by design."""
    return {key: fill_value(key, shape) for key, shape in entries}


def embedding_vectors(name: str, rows: int, dim: int) -> torch.Tensor:
    """The deterministic textual-inversion tensor for one golden
    embedding reference."""
    return fill_value(f"embedding:{name}", (rows, dim))
