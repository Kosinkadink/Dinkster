"""Tiled application: the tensor half of tiled codec execution.

Port of the accumulation half of comfy/utils.py tiled_scale_multidim
@ b78cec87 - allocate output + weight accumulator, run the function
per tile, feather-mask the tile result, accumulate, divide - driven
by a torch-free ``TilePlan`` (dinkster_inference.tiling) instead of the
reference's inline position loop. Byte-for-byte the same arithmetic:
edge tiles crop at the output bounds, the feather ramp is
``(t + 1) / feather`` multiplied onto both ends, dims whose feather
covers the whole tile output skip feathering, and every output
element is normalized by the accumulated mask weight.

Deliberate deviations (documented, tested):

- No ``@torch.inference_mode()``: the reference pins it on the
  utility; Dinkster's training program forbids baking no-grad into the
  substrate (docs/native-inference-plan.md 3.1). Callers that want it
  wrap the call.
- ``function`` output that misses the planned output region (leaving
  accumulated weight at zero) would divide 0/0 into NaNs in the
  reference; here it raises ``TileApplyError``.
- ``function`` output with the wrong rank, batch size, or channel
  count would silently broadcast in the reference (a singleton
  channel spreads across all output channels; on the single-tile
  path a singleton content dim can fill the whole output); here it
  raises ``TileApplyError``.
- Progress is an optional ``on_tile`` callback (the reference takes a
  pbar object).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from dinkster_inference.tiling import TilePlan

__all__ = [
    "TileApplyError",
    "tiled_apply",
]


class TileApplyError(ValueError):
    """Input geometry disagreeing with the plan, or a function whose
    tile outputs cannot cover the planned output."""


def tiled_apply(
    samples: torch.Tensor,
    function: Callable[[torch.Tensor], torch.Tensor],
    plan: TilePlan,
    *,
    out_channels: int,
    output_device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
    on_tile: Callable[[], None] | None = None,
) -> torch.Tensor:
    """Apply ``function`` (one batch item in, one batch item out - a
    codec encode or decode) tile by tile per ``plan``.

    ``samples`` is ``[batch, channels, *plan.input_shape]``; the
    result is ``[batch, out_channels, *plan.output_shape]`` on
    ``output_device`` in ``dtype`` (default: the process default
    dtype, matching the reference's bare ``torch.empty``). ``on_tile``
    fires once per function call, ``len(plan.tiles)`` (or one, when
    ``plan.single_tile``) times per batch item.
    """
    if tuple(samples.shape[2:]) != plan.input_shape:
        raise TileApplyError(
            f"samples content shape {tuple(samples.shape[2:])} does not"
            f" match the plan's input shape {plan.input_shape}"
        )
    dims = len(plan.input_shape)
    if dtype is None:
        dtype = torch.get_default_dtype()

    def check_result(ps: torch.Tensor) -> torch.Tensor:
        # the reference assigns/broadcasts whatever the function
        # returns; a singleton channel (or, single-tile, singleton
        # content dim) would silently broadcast wrong codec output -
        # refuse loudly instead
        if ps.ndim != dims + 2 or ps.shape[0] != 1 or (ps.shape[1] != out_channels):
            raise TileApplyError(
                f"function returned shape {tuple(ps.shape)}; expected"
                f" [1, {out_channels}, ...] with {dims} content"
                " dimensions"
            )
        return ps

    output = torch.empty(
        [samples.shape[0], out_channels, *plan.output_shape],
        device=output_device,
        dtype=dtype,
    )

    for b in range(samples.shape[0]):
        s = samples[b : b + 1]

        if plan.single_tile:
            ps = check_result(function(s).to(output_device))
            if tuple(ps.shape[2:]) != plan.output_shape:
                raise TileApplyError(
                    f"function returned content shape {tuple(ps.shape[2:])};"
                    f" the plan's single tile expects {plan.output_shape}"
                )
            output[b : b + 1] = ps
            if on_tile is not None:
                on_tile()
            continue

        out = output[b : b + 1].zero_()
        out_div = torch.zeros([1, 1, *plan.output_shape], device=output_device, dtype=dtype)

        for planned in plan.tiles:
            s_in = s
            for d in range(dims):
                sl = planned.dims[d]
                s_in = s_in.narrow(d + 2, sl.pos, sl.length)

            ps = check_result(function(s_in).to(output_device))
            mask = torch.ones([1, 1, *ps.shape[2:]], device=output_device, dtype=dtype)

            for d in range(2, dims + 2):
                feather = plan.feather[d - 2]
                if feather >= mask.shape[d]:
                    continue
                for t in range(feather):
                    a = (t + 1) / feather
                    mask.narrow(d, t, 1).mul_(a)
                    mask.narrow(d, mask.shape[d] - 1 - t, 1).mul_(a)

            o = out
            o_d = out_div
            ps_view = ps
            mask_view = mask
            for d in range(dims):
                dst = planned.dims[d].dst
                length = min(ps_view.shape[d + 2], o.shape[d + 2] - dst)
                if length < 1:
                    raise TileApplyError(
                        f"tile output at dst {dst} lies outside the planned"
                        f" output extent {o.shape[d + 2]} in dimension {d}"
                    )
                o = o.narrow(d + 2, dst, length)
                o_d = o_d.narrow(d + 2, dst, length)
                if length < ps_view.shape[d + 2]:
                    ps_view = ps_view.narrow(d + 2, 0, length)
                    mask_view = mask_view.narrow(d + 2, 0, length)

            o.add_(ps_view * mask_view)
            o_d.add_(mask_view)

            if on_tile is not None:
                on_tile()

        if bool((out_div == 0).any()):
            raise TileApplyError(
                "tile outputs left part of the planned output uncovered"
                " (zero accumulated weight); the function returned less"
                " content than the plan's scale rules predict"
            )
        out.div_(out_div)
    return output
