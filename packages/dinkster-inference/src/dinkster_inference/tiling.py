"""Tiled codec execution planning: the index math as typed data.

Port of the index/geometry half of comfy/utils.py
tiled_scale_multidim @ b78cec87 - tile positions, edge clamping,
output placement, and feather widths - as a pure, torch-free
``TilePlan``. The tensor half (allocate, accumulate, feather-mask,
divide) lives in dinkster-inference-torch tiling.py and consumes this
plan; codec plugins (stage 5) declare tile/overlap defaults and the
executor plans against them.

Scale rules are data, not callables. The reference passes per-dim
``upscale_amount``/``index_formulas`` entries that are either numbers
or lambdas; every lambda in the tree @ b78cec87 is the causal-video
pair ``lambda a: max(0, a * f - (f - 1))`` (decode length) /
``lambda a: max(0, floor((a + f - 1) / f))`` (encode length) with a
plain-number index formula ``f`` - captured here as ``CausalScale``.
Plain numbers are ``LinearScale``. The reference's ``downscale`` flag
becomes the plan's ``downscale`` argument, flipping multiply to
divide exactly like get_upscale/get_downscale.

Deliberate deviations (documented, tested):

- Where ``size > tile`` and ``tile <= overlap`` the reference's
  ``range(0, size - overlap, tile - overlap)`` raises a bare
  ValueError (step 0) or silently plans NOTHING (negative step -
  uncovered output, NaNs after the divide). Planning refuses loudly
  with ``TilePlanError`` instead.
- ``get_tiled_scale_steps`` (a ceil approximation used only for
  progress bars) is not ported: ``TilePlan.tiles`` has the exact
  count.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

__all__ = [
    "CausalScale",
    "LinearScale",
    "PlannedTile",
    "Scale",
    "TilePlan",
    "TilePlanError",
    "TileSlice",
    "plan_tiles",
]


class TilePlanError(ValueError):
    """A tile plan that cannot cover the input (or nonsense sizes)."""


@dataclass(frozen=True)
class LinearScale:
    """A plain-number scale: output extents and positions are
    ``factor * value`` (or ``value / factor`` when the plan is a
    downscale) - the reference's non-callable upscale_amount /
    index_formulas entries. ``factor`` may be fractional: the
    reference's encode_tiled_1d passes ``1 / downscale_ratio`` as an
    upscale amount."""

    factor: float

    def __post_init__(self) -> None:
        if not self.factor > 0:
            raise TilePlanError(f"scale factor must be > 0, got {self.factor}")

    def size(self, value: int, *, downscale: bool) -> float:
        return value / self.factor if downscale else self.factor * value

    def index(self, value: int, *, downscale: bool) -> float:
        return value / self.factor if downscale else self.factor * value


@dataclass(frozen=True)
class CausalScale:
    """The causal-video time axis (first frame not compressed):
    decode maps ``t`` latent frames to ``max(0, t * f - (f - 1))``
    pixels frames, encode maps ``t`` pixel frames to
    ``max(0, floor((t + f - 1) / f))`` latents, while POSITIONS scale
    by plain ``f`` - the reference's lambda upscale_amount paired with
    a numeric index_formulas entry @ b78cec87 (f = 4, 6, 8 in the
    tree)."""

    factor: int

    def __post_init__(self) -> None:
        if self.factor < 1:
            raise TilePlanError(f"causal scale factor must be >= 1, got {self.factor}")

    def size(self, value: int, *, downscale: bool) -> int:
        if downscale:
            return max(0, (value + self.factor - 1) // self.factor)
        return max(0, value * self.factor - (self.factor - 1))

    def index(self, value: int, *, downscale: bool) -> float:
        if downscale:
            return value / self.factor
        return value * self.factor


Scale = LinearScale | CausalScale


@dataclass(frozen=True)
class TileSlice:
    """One dimension of one planned tile: read ``length`` source
    elements starting at ``pos``, write the tile's output starting at
    output position ``dst`` (cropping to the output bounds is the
    executor's runtime job, as in the reference)."""

    pos: int
    length: int
    dst: int


@dataclass(frozen=True)
class PlannedTile:
    """One tile: a slice per content dimension."""

    dims: tuple[TileSlice, ...]


@dataclass(frozen=True)
class TilePlan:
    """The full traversal for one input geometry.

    ``input_shape`` / ``output_shape`` are content dimensions only
    (no batch/channel). ``feather`` is the per-dim blend width in
    OUTPUT elements (``round(scale.size(overlap))``, the reference's
    feather). When ``single_tile`` is true the whole input fits in
    one tile: the executor runs the function once per batch item with
    no accumulation and ``tiles`` is empty."""

    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    feather: tuple[int, ...]
    tiles: tuple[PlannedTile, ...]
    single_tile: bool


def _check_entries(count: int, dims: int, name: str) -> None:
    if count != dims:
        raise TilePlanError(f"{name} has {count} entries for {dims} dimensions")


def plan_tiles(
    shape: tuple[int, ...],
    tile: tuple[int, ...],
    *,
    overlap: int | tuple[int, ...],
    scale: Scale | tuple[Scale, ...],
    downscale: bool = False,
) -> TilePlan:
    """Plan the tiled traversal of content ``shape`` (no
    batch/channel dims) - tiled_scale_multidim's position loop
    @ b78cec87 as data.

    Per dimension, positions step by ``tile - overlap`` up to (never
    reaching) ``size - overlap``; each position clamps into
    ``[0, size - overlap]`` and reads ``min(tile, size - pos)``
    elements; the output start is ``round(scale.index(pos))``. A
    dimension no larger than its tile contributes the single position
    0. ``downscale`` flips the scale rules' multiply to divide (the
    reference's encode direction).
    """
    dims = len(shape)
    if dims == 0:
        raise TilePlanError("shape needs at least one content dimension")
    _check_entries(len(tile), dims, "tile")
    tiles_d = tile
    if isinstance(overlap, tuple):
        _check_entries(len(overlap), dims, "overlap")
        overlaps = overlap
    else:
        overlaps = (overlap,) * dims
    if isinstance(scale, tuple):
        _check_entries(len(scale), dims, "scale")
        scales = scale
    else:
        scales = (scale,) * dims

    for d in range(dims):
        if shape[d] < 1:
            raise TilePlanError(f"dimension {d} has size {shape[d]}")
        if tiles_d[d] < 1:
            raise TilePlanError(f"dimension {d} has tile size {tiles_d[d]}")
        if overlaps[d] < 0:
            raise TilePlanError(f"dimension {d} has overlap {overlaps[d]}")
        if shape[d] > tiles_d[d] and tiles_d[d] <= overlaps[d]:
            # the reference range()s with step tile - overlap here:
            # zero step raises bare, negative step covers nothing and
            # NaNs the output after the divide
            raise TilePlanError(
                f"dimension {d}: tile ({tiles_d[d]}) must exceed overlap"
                f" ({overlaps[d]}) when the input ({shape[d]}) needs tiling"
            )

    output_shape = tuple(round(scales[d].size(shape[d], downscale=downscale)) for d in range(dims))
    feather = tuple(round(scales[d].size(overlaps[d], downscale=downscale)) for d in range(dims))

    if all(shape[d] <= tiles_d[d] for d in range(dims)):
        return TilePlan(
            input_shape=shape,
            output_shape=output_shape,
            feather=feather,
            tiles=(),
            single_tile=True,
        )

    positions = [
        range(0, shape[d] - overlaps[d], tiles_d[d] - overlaps[d]) if shape[d] > tiles_d[d] else [0]
        for d in range(dims)
    ]

    planned: list[PlannedTile] = []
    for it in itertools.product(*positions):
        slices: list[TileSlice] = []
        for d in range(dims):
            pos = max(0, min(shape[d] - overlaps[d], it[d]))
            length = min(tiles_d[d], shape[d] - pos)
            dst = round(scales[d].index(pos, downscale=downscale))
            slices.append(TileSlice(pos=pos, length=length, dst=dst))
        planned.append(PlannedTile(dims=tuple(slices)))

    return TilePlan(
        input_shape=shape,
        output_shape=output_shape,
        feather=feather,
        tiles=tuple(planned),
        single_tile=False,
    )
