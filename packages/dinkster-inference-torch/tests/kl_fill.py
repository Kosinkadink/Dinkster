"""Deterministic KL golden weight fill, shared with the generator.

The KL golden architectures cannot be tiny: the reference hardcodes
GroupNorm(num_groups=32), so every width is a multiple of 32 and the
smallest faithful models carry millions of parameters - too large to
store in the repo. Instead of storing weights, the golden pins the
reference's sorted (key, shape) listing and both sides (the reference
run in tools/gen_kl_goldens.py and the replay tests) fill each tensor
from this function: a pure 32-bit integer hash of the KEY NAME, so
the values are bit-identical across torch versions and platforms
(no torch RNG, no libm) and never depend on module construction
order.

Norm scale weights center on 1.0 so activations stay in a sane range
through deep stacks; running variances center on 1.0 too (they feed
sqrt(var + eps), so they must stay positive); everything else is
uniform in [-0.2, 0.2].
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Sequence

import torch

_MASK32 = 0xFFFFFFFF


def hash_uniform(seed: int, count: int) -> torch.Tensor:
    """``count`` uniforms in [0, 1) from a vectorized 32-bit mix.

    Every intermediate product stays below 2**53 (constants < 2**21,
    state masked to 32 bits), so the arithmetic is exact in int64 on
    any backend.
    """
    x = torch.arange(count, dtype=torch.int64) + (seed & _MASK32)
    x = (x * 1664525 + 1013904223) & _MASK32
    x = x ^ (x >> 13)
    x = (x * 214013 + 2531011) & _MASK32
    x = x ^ (x >> 17)
    x = (x * 69069 + 1) & _MASK32
    x = x ^ (x >> 5)
    return (x.to(torch.float64) / float(_MASK32 + 1)).to(torch.float32)


def fill_value(key: str, shape: Sequence[int]) -> torch.Tensor:
    """The deterministic float32 tensor for one state-dict entry."""
    count = math.prod(shape) if shape else 1
    uniform = hash_uniform(zlib.crc32(key.encode("utf-8")), count)
    if key.endswith(".weight") and ".norm" in f".{key}":
        values = 1.0 + (uniform - 0.5) * 0.1
    elif key.endswith("running_var"):
        values = 1.0 + (uniform - 0.5) * 0.5
    else:
        values = (uniform - 0.5) * 0.4
    return values.reshape(tuple(shape))


def fill_state_dict(
    entries: Sequence[tuple[str, Sequence[int]]],
) -> dict[str, torch.Tensor]:
    """Fill every (key, shape) entry; order-independent by design."""
    return {key: fill_value(key, shape) for key, shape in entries}
