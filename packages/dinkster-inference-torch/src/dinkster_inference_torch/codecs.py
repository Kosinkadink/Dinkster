"""Codec plugins: the executing layer over torch tensors.

Binds a torch-free ``CodecDescriptor`` (dinkster_inference.codecs) to
concrete encoder/decoder implementations and runs direct or tiled
encode/decode. Plugins are plain descriptors-with-code: they satisfy
the ``Registrable`` protocol, so a ``Registry[CodecPlugin]`` gives
namespaced registration with loud collisions - no central family
switch (the reference's comfy/sd.py VAE __init__ @ b78cec87).

Tiled 2D decode/encode averages three tile aspect ratios exactly like
the reference (comfy/sd.py decode_tiled_/encode_tiled_ @ b78cec87) to
hide seams, in the reference's per-direction accumulation order, with
in-place accumulation (two full outputs alive at peak, the
reference's encode behavior); 1D and 3D content runs a single pass,
as upstream. All passes are planned before the first encoder/decoder
call runs, so invalid tile geometry refuses up front. No
``inference_mode``/``no_grad`` is baked in (training program,
docs/native-inference-plan.md 3.1); callers that want it wrap the
call.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from dinkster_inference.codecs import (
    CodecDecoder,
    CodecDescriptor,
    CodecEncoder,
    CodecMemoryEstimator,
    plan_codec_decode,
    plan_codec_encode,
)
from dinkster_inference.tiling import TilePlan, TilePlanError

from .tiling import tiled_apply

__all__ = [
    "CodecPlugin",
]


def _tile_variants(
    tile: tuple[int, ...], dims: int, *, decode: bool
) -> tuple[tuple[int, ...], ...]:
    """The reference's seam-hiding aspect sweep for 2D content; other
    dimensionalities tile once. Pass order follows the reference's
    per-direction accumulation order (float addition is not
    associative): decode sums (2t0, t1/2), (t0/2, 2t1), (t0, t1);
    encode sums (t0, t1), (t0/2, 2t1), (2t0, t1/2)."""
    if dims != 2:
        return (tile,)
    t0, t1 = tile
    if decode:
        return ((t0 * 2, t1 // 2), (t0 // 2, t1 * 2), (t0, t1))
    return ((t0, t1), (t0 // 2, t1 * 2), (t0 * 2, t1 // 2))


@dataclass(frozen=True)
class CodecPlugin:
    """One executable codec: descriptor + encoder/decoder (+ optional
    memory estimator, consumed by placement/OOM policy).

    ``id``/``aliases`` delegate to the descriptor, so a plugin
    registers directly in a ``Registry[CodecPlugin]``.

    ``content_in``/``content_out`` are the codec's content-boundary
    transforms (the reference's VAE.process_input/process_output,
    comfy/sd.py @ b78cec87), applied by every encode/decode entry
    point so direct and tiled execution normalize at identical
    semantic boundaries. Both must be POINTWISE
    (position-independent): ``content_in`` runs once on the whole
    input before the tile sweep, which is equivalent to the
    reference's per-tile application exactly because slicing commutes
    with pointwise transforms. ``content_out`` runs after the
    complete tiled average, like the reference - its clamp timing is
    observable across tile seams, so it must NEVER run per tile.
    ``content_out`` may mutate its argument in place; it only ever
    receives codec-owned output tensors.

    ``content_crop`` is the codec's geometry-fitting transform (the
    reference's VAE.vae_encode_crop_pixels, comfy/sd.py @ b78cec87),
    run FIRST on both encode entry points. Unlike ``content_in`` it
    may change spatial extents (center-cropping to the codec's
    downscale grid), so it must run once on the whole input, before
    tile planning ever sees the shape - never per tile.
    """

    descriptor: CodecDescriptor
    encoder: CodecEncoder[torch.Tensor]
    decoder: CodecDecoder[torch.Tensor]
    memory: CodecMemoryEstimator | None = None
    content_crop: Callable[[torch.Tensor], torch.Tensor] | None = None
    content_in: Callable[[torch.Tensor], torch.Tensor] | None = None
    content_out: Callable[[torch.Tensor], torch.Tensor] | None = None
    compute_dtype: torch.dtype | None = None

    @property
    def id(self) -> str:
        return self.descriptor.id

    @property
    def aliases(self) -> tuple[str, ...]:
        return self.descriptor.aliases

    def encode(self, content: torch.Tensor) -> torch.Tensor:
        """Direct (untiled) encode: content in, latent out."""
        if self.content_crop is not None:
            content = self.content_crop(content)
        if self.content_in is not None:
            content = self.content_in(content)
        if self.compute_dtype is not None:
            content = content.to(dtype=self.compute_dtype)
        return self.encoder.encode(content)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Direct (untiled) decode: latent in, content out."""
        if self.compute_dtype is not None:
            latent = latent.to(dtype=self.compute_dtype)
        output = self.decoder.decode(latent)
        if self.content_out is not None:
            output = self.content_out(output)
        return output

    def _resolve(
        self,
        tile: tuple[int, ...] | None,
        overlap: tuple[int, ...] | None,
        *,
        decode: bool,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if not self.descriptor.supports_tiling:
            direction = "decode" if decode else "encode"
            raise TilePlanError(f"codec {self.descriptor.id} does not support tiled {direction}")
        tiling = self.descriptor.tiling
        if tile is None or overlap is None:
            if tiling is None:
                raise TilePlanError(
                    f"codec {self.descriptor.id} declares no tiling defaults;"
                    " pass tile and overlap explicitly"
                )
            if tile is None:
                tile = tiling.decode_tile if decode else tiling.encode_tile
            if overlap is None:
                overlap = tiling.decode_overlap if decode else tiling.encode_overlap
        return tile, overlap

    def _run_tiled(
        self,
        samples: torch.Tensor,
        function: Callable[[torch.Tensor], torch.Tensor],
        tile: tuple[int, ...] | None,
        overlap: tuple[int, ...] | None,
        *,
        decode: bool,
        out_channels: int,
        output_device: torch.device | str,
        dtype: torch.dtype | None,
        on_tile: Callable[[], None] | None,
    ) -> torch.Tensor:
        tile, overlap = self._resolve(tile, overlap, decode=decode)
        if self.compute_dtype is not None:
            samples = samples.to(dtype=self.compute_dtype)
        shape = tuple(samples.shape[2:])
        plan_fn = plan_codec_decode if decode else plan_codec_encode
        variants = _tile_variants(tile, self.descriptor.latent.dimensions, decode=decode)
        # plan every pass before running the first: a variant whose
        # halved tile no longer exceeds the overlap must refuse before
        # any encoder/decoder work happens, not between passes
        plans = tuple(
            plan_fn(self.descriptor, shape, tile=variant, overlap=overlap) for variant in variants
        )

        def one_pass(plan: TilePlan) -> torch.Tensor:
            return tiled_apply(
                samples,
                function,
                plan,
                out_channels=out_channels,
                output_device=output_device,
                dtype=dtype,
                on_tile=on_tile,
            )

        output = one_pass(plans[0])
        if len(plans) == 1:
            return output
        for plan in plans[1:]:
            # in-place accumulation like the reference's encode
            # (samples += pass; samples /= 3.0): two full outputs
            # alive at peak instead of three; add_/div_ on the
            # non-leaf accumulator stay autograd-safe
            output.add_(one_pass(plan))
        return output.div_(len(plans))

    def encode_tiled(
        self,
        content: torch.Tensor,
        *,
        tile: tuple[int, ...] | None = None,
        overlap: tuple[int, ...] | None = None,
        output_device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        on_tile: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        """Tiled encode. ``tile``/``overlap`` are in content units per
        content dimension; omitted values come from the descriptor's
        tiling defaults (no defaults and no explicit sizes, or a codec
        with ``supports_tiling=False``, refuses with TilePlanError)."""
        if self.content_crop is not None:
            # before planning: the plan must see the cropped shape,
            # or plan/output geometry disagree on non-grid inputs
            content = self.content_crop(content)
        if self.content_in is not None:
            # once on the whole input, not per tile: pointwise
            # transforms commute with the sweep's input slicing
            content = self.content_in(content)
        return self._run_tiled(
            content,
            self.encoder.encode,
            tile,
            overlap,
            decode=False,
            out_channels=self.descriptor.latent.channels,
            output_device=output_device,
            dtype=dtype,
            on_tile=on_tile,
        )

    def decode_tiled(
        self,
        latent: torch.Tensor,
        *,
        tile: tuple[int, ...] | None = None,
        overlap: tuple[int, ...] | None = None,
        output_device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        on_tile: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        """Tiled decode. ``tile``/``overlap`` are in latent units per
        content dimension; defaults as in :meth:`encode_tiled`."""
        output = self._run_tiled(
            latent,
            self.decoder.decode,
            tile,
            overlap,
            decode=True,
            out_channels=self.descriptor.content_channels,
            output_device=output_device,
            dtype=dtype,
            on_tile=on_tile,
        )
        if self.content_out is not None:
            # after the COMPLETE tiled average, never per tile: the
            # reference clamps the composed output (comfy/sd.py
            # decode_tiled_ @ b78cec87), and clamp-before-average
            # would be observable across tile seams
            output = self.content_out(output)
        return output
