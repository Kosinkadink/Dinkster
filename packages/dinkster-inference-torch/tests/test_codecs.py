"""Stage 5 slice 1: executable codec plugins.

A deterministic toy codec (nearest-upsample decode / block-mean
encode) proves the plugin seam: registration through the generic
Registry, direct encode/decode, and tiled encode/decode. The toy
codec is spatially local and shift-invariant, so tiled output must
EQUAL direct output - an oracle independent of the tiler's own math
(the tiler itself is pinned against executed-reference goldens in
test_tiling.py).

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from dinkster_inference import FLOAT32, Registry
from dinkster_inference.codecs import (
    CodecDescriptor,
    CodecTiling,
)
from dinkster_inference.latents import LatentDescriptor
from dinkster_inference.tiling import TilePlanError
from dinkster_inference.weights import TensorGeometry
from dinkster_inference_torch import CodecPlugin, TileApplyError

# --------------------------------------------------------- toy codec
#
# latent (4ch) <-> content (3ch) at x8 spatial scale. decode maps each
# latent element to an 8x8 content block (nearest upsample of a fixed
# channel mix); encode block-means and lifts through a second fixed
# matrix. Both are local: each output element depends only on its own
# source block, so any tile traversal must reproduce the direct
# result (overlapping tiles compute identical values, and the feather
# average of equal values is the value).

_MIX = torch.tensor(
    [
        [0.5, 0.25, 0.25, 0.0],
        [0.0, 0.5, 0.25, 0.25],
        [0.25, 0.0, 0.25, 0.5],
    ]
)

_LIFT = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, -0.5, 0.25],
    ]
)


class ToyDecoder:
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        mixed = torch.einsum("oc,bchw->bohw", _MIX.to(latent.dtype), latent)
        return mixed.repeat_interleave(8, dim=-2).repeat_interleave(8, dim=-1)


class ToyEncoder:
    def encode(self, content: torch.Tensor) -> torch.Tensor:
        pooled = torch.nn.functional.avg_pool2d(content, 8)
        return torch.einsum("co,bohw->bchw", _LIFT.to(content.dtype), pooled)


class DtypeRecordingDecoder:
    def __init__(self) -> None:
        self.seen: list[torch.dtype] = []

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.seen.append(latent.dtype)
        return torch.zeros(
            latent.shape[0],
            3,
            latent.shape[2] * 8,
            latent.shape[3] * 8,
            device=latent.device,
            dtype=latent.dtype,
        )


class DtypeRecordingEncoder:
    def __init__(self) -> None:
        self.seen: list[torch.dtype] = []

    def encode(self, content: torch.Tensor) -> torch.Tensor:
        self.seen.append(content.dtype)
        return torch.zeros(
            content.shape[0],
            4,
            content.shape[2] // 8,
            content.shape[3] // 8,
            device=content.device,
            dtype=content.dtype,
        )


def toy_plugin(**overrides: object) -> CodecPlugin:
    fields: dict[str, object] = dict(
        id="dinkster.toy_vae",
        display_name="Toy VAE",
        kind="image",
        latent=LatentDescriptor(channels=4, dimensions=2, spatial_downscale=8),
        supported_dtypes=frozenset({FLOAT32}),
        tiling=CodecTiling(
            decode_tile=(8, 8),
            decode_overlap=(2, 2),
            encode_tile=(64, 64),
            encode_overlap=(16, 16),
        ),
    )
    fields.update(overrides)
    descriptor = CodecDescriptor(**fields)  # type: ignore[arg-type]
    return CodecPlugin(
        descriptor=descriptor,
        encoder=ToyEncoder(),
        decoder=ToyDecoder(),
    )


# -------------------------------------------------------- registration


def test_plugin_registers_in_generic_registry() -> None:
    registry: Registry[CodecPlugin] = Registry()
    plugin = toy_plugin(aliases=("legacy.toy_vae",))
    registry.register(plugin)
    assert registry.get("dinkster.toy_vae") is plugin
    assert registry.get("legacy.toy_vae") is plugin
    assert plugin.id == "dinkster.toy_vae"
    assert registry.ids() == ("dinkster.toy_vae",)


# ------------------------------------------------------ direct paths


def test_direct_decode_hand_computed() -> None:
    # one latent element [1, 2, 3, 4] -> mix rows dotted with it,
    # broadcast to an 8x8 block per content channel
    latent = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 4, 1, 1)
    content = toy_plugin().decode(latent)
    assert content.shape == (1, 3, 8, 8)
    expected = torch.tensor([1.75, 2.75, 3.0])
    torch.testing.assert_close(
        content,
        expected.reshape(1, 3, 1, 1).expand(1, 3, 8, 8),
    )


def test_direct_encode_hand_computed() -> None:
    # constant content [2, 4, 8] block-means to itself, then lifts:
    # [2, 4, 8, 0.5*2 - 0.5*4 + 0.25*8] = [2, 4, 8, 1]
    content = torch.tensor([2.0, 4.0, 8.0]).reshape(1, 3, 1, 1)
    latent = toy_plugin().encode(content.expand(1, 3, 8, 8))
    torch.testing.assert_close(
        latent,
        torch.tensor([2.0, 4.0, 8.0, 1.0]).reshape(1, 4, 1, 1),
    )


@pytest.mark.parametrize("compute_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tiled", [False, True])
def test_codec_inputs_use_compute_dtype(compute_dtype: torch.dtype, tiled: bool) -> None:
    encoder = DtypeRecordingEncoder()
    decoder = DtypeRecordingDecoder()
    base = toy_plugin()
    plugin = CodecPlugin(
        descriptor=base.descriptor,
        encoder=encoder,
        decoder=decoder,
        compute_dtype=compute_dtype,
    )

    if tiled:
        plugin.encode_tiled(torch.randn(1, 3, 112, 112))
        plugin.decode_tiled(torch.randn(1, 4, 14, 14))
    else:
        plugin.encode(torch.randn(1, 3, 112, 112))
        plugin.decode(torch.randn(1, 4, 14, 14))

    assert encoder.seen and set(encoder.seen) == {compute_dtype}
    assert decoder.seen and set(decoder.seen) == {compute_dtype}


def test_bfloat16_decode_output_keeps_storage_until_float_consumer() -> None:
    import numpy as np
    from dinkster_values.image_codec import (
        decode_image_array,
        encode_image_array,
        image_array_fingerprint,
        image_array_meta,
    )
    from dinkster_values.storage import image_input

    plugin = toy_plugin()
    latent = torch.ones(1, 4, 2, 2, dtype=torch.bfloat16)
    output = plugin.decode(latent).permute(0, 2, 3, 1).contiguous()
    assert output.dtype == torch.bfloat16
    decoded = np.asarray(decode_image_array(encode_image_array(output)))
    assert decoded.nbytes == output.numel() * 2
    assert image_array_meta(output) == image_array_meta(decoded)
    fingerprint = image_array_fingerprint("comfy.IMAGE")
    assert fingerprint(output) == fingerprint(decoded)
    normalized = np.asarray(image_input(decoded))
    assert normalized.dtype == np.float32
    np.testing.assert_array_equal(normalized, output.float().numpy())


@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.uint16, torch.float16, torch.bfloat16, torch.float32]
)
def test_image_storage_metadata_reads_retained_device_allocation_without_host_copy(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_values import COST_META_KEY
    from dinkster_values.image_codec import image_array_meta

    def reject_host_copy(self: torch.Tensor) -> torch.Tensor:
        pytest.fail("metadata copied to host")

    images = torch.zeros(2, 4, 4, 3, dtype=dtype)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", reject_host_copy)
        metadata = image_array_meta(images[:1])
    assert metadata[COST_META_KEY] == {"ram": images.untyped_storage().nbytes()}
    assert metadata["shape"] == (1, 4, 4, 3)


def test_numpy_view_cost_covers_its_retained_torch_allocation() -> None:
    from dinkster_values import COST_META_KEY, array_storage_meta

    backing = torch.zeros(1024, dtype=torch.float32)
    view = backing[:1].numpy()
    assert view.nbytes == 4
    assert array_storage_meta(view)[COST_META_KEY] == {"ram": backing.untyped_storage().nbytes()}


@pytest.mark.parametrize(
    "dtype", [torch.int16, torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_audio_envelope_canonicalizes_legacy_tensor_storage(dtype: torch.dtype) -> None:
    from dinkster_compat_comfy.audio import register_audio_type
    from dinkster_values import COST_META_KEY, TypeRegistry

    registry = TypeRegistry()
    register_audio_type(registry, "comfy.AUDIO")
    original = torch.arange(24, dtype=dtype).reshape(1, 2, 12)
    value = registry.wrap("comfy.AUDIO", {"waveform": original, "sample_rate": 48000})
    stored: Any = value.resolve()
    waveform = stored["waveform"]
    expected_dtype = torch.int16 if dtype == torch.int16 else torch.float32
    assert waveform.dtype == expected_dtype
    assert (waveform is original) == (dtype in (torch.int16, torch.float32))
    assert original.dtype == dtype
    assert value.meta.get("storage_dtype") == ("int16" if dtype == torch.int16 else "fp32")
    assert value.meta.get(COST_META_KEY) == {"ram": waveform.untyped_storage().nbytes()}
    spec = registry.spec("comfy.AUDIO")
    decoded: Any = spec.decode(spec.encode(stored))
    assert decoded["waveform"].dtype == expected_dtype
    torch.testing.assert_close(decoded["waveform"], waveform, rtol=0, atol=0)


# ------------------------------------------------------- tiled paths


def test_tiled_decode_matches_direct() -> None:
    # 14 latent > 8 tile in both dims: real multi-tile traversal,
    # including the reference's three-aspect 2D average
    latent = torch.randn(2, 4, 14, 14)
    plugin = toy_plugin()
    direct = plugin.decode(latent)
    tiled = plugin.decode_tiled(latent)
    assert tiled.shape == direct.shape
    torch.testing.assert_close(tiled, direct, rtol=1e-5, atol=1e-5)


def test_tiled_encode_matches_direct() -> None:
    # content 112 = 2*(64-16) + 16: positions 0, 48, 96 clamp inside
    # bounds and stay 8-aligned, so block pooling agrees with direct
    content = torch.randn(1, 3, 112, 112)
    plugin = toy_plugin()
    direct = plugin.encode(content)
    tiled = plugin.encode_tiled(content)
    assert tiled.shape == direct.shape
    torch.testing.assert_close(tiled, direct, rtol=1e-4, atol=1e-5)


def test_tiled_decode_single_tile_path() -> None:
    latent = torch.randn(1, 4, 4, 4)
    plugin = toy_plugin()
    tiled = plugin.decode_tiled(latent, tile=(16, 16), overlap=(2, 2))
    torch.testing.assert_close(tiled, plugin.decode(latent))


def test_tiled_2d_runs_three_aspect_passes() -> None:
    # the reference's seam-hiding sweep: (t0,t1), (2*t0,t1//2),
    # (t0//2,2*t1) - tile counts 4 + 2 + 2 with these sizes, per
    # batch item
    latent = torch.randn(1, 4, 14, 14)
    plugin = toy_plugin()
    calls = 0

    def counted() -> None:
        nonlocal calls
        calls += 1

    plugin.decode_tiled(latent, on_tile=counted)
    # (8,8): 2x2 tiles; (16,4): 1x4 -> [0] x range(0,12,2)=6 -> 6;
    # (4,16): 6x1 -> 6
    assert calls == 4 + 6 + 6


def test_tiled_explicit_sizes_override_defaults() -> None:
    latent = torch.randn(1, 4, 14, 14)
    plugin = toy_plugin()
    out = plugin.decode_tiled(latent, tile=(12, 12), overlap=(4, 4))
    torch.testing.assert_close(out, plugin.decode(latent), rtol=1e-5, atol=1e-5)


def test_tiled_refuses_without_defaults() -> None:
    plugin = toy_plugin(tiling=None)
    latent = torch.randn(1, 4, 14, 14)
    with pytest.raises(TilePlanError, match="no tiling defaults"):
        plugin.decode_tiled(latent)
    out = plugin.decode_tiled(latent, tile=(8, 8), overlap=(2, 2))
    torch.testing.assert_close(out, plugin.decode(latent), rtol=1e-5, atol=1e-5)


def test_tiled_refuses_when_unsupported() -> None:
    plugin = toy_plugin(supports_tiling=False, tiling=None)
    latent = torch.randn(1, 4, 14, 14)
    with pytest.raises(TilePlanError, match="does not support"):
        plugin.decode_tiled(latent, tile=(8, 8), overlap=(2, 2))


def test_tiled_1d_single_pass() -> None:
    # non-2D content skips the aspect sweep: one pass only
    class Decoder1D:
        def decode(self, latent: torch.Tensor) -> torch.Tensor:
            return latent.repeat_interleave(4, dim=-1)

    class Encoder1D:
        def encode(self, content: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.avg_pool1d(content, 4)

    descriptor = CodecDescriptor(
        id="dinkster.toy_audio",
        display_name="Toy audio codec",
        kind="audio",
        latent=LatentDescriptor(channels=2, dimensions=1, spatial_downscale=4),
        supported_dtypes=frozenset({FLOAT32}),
        content_channels=2,
        tiling=CodecTiling(
            decode_tile=(16,),
            decode_overlap=(4,),
            encode_tile=(64,),
            encode_overlap=(16,),
        ),
    )
    plugin = CodecPlugin(descriptor=descriptor, encoder=Encoder1D(), decoder=Decoder1D())
    latent = torch.randn(1, 2, 40)
    calls = 0

    def counted() -> None:
        nonlocal calls
        calls += 1

    tiled = plugin.decode_tiled(latent, on_tile=counted)
    torch.testing.assert_close(tiled, plugin.decode(latent), rtol=1e-5, atol=1e-5)
    # single pass: range(0, 36, 12) = 0, 12, 24 -> 3 tiles
    assert calls == 3


def test_gradients_flow_through_tiled_codec() -> None:
    latent = torch.randn(1, 4, 14, 14, requires_grad=True)
    plugin = toy_plugin()
    plugin.decode_tiled(latent).sum().backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()


def test_memory_estimator_is_carried() -> None:
    class Estimator:
        def encode_bytes(self, content: TensorGeometry) -> int:
            return 123

        def decode_bytes(self, latent: TensorGeometry) -> int:
            return 456

    plugin = CodecPlugin(
        descriptor=toy_plugin().descriptor,
        encoder=ToyEncoder(),
        decoder=ToyDecoder(),
        memory=Estimator(),
    )
    geometry = TensorGeometry(shape=(1, 4, 8, 8), dtype=FLOAT32)
    assert plugin.memory is not None
    assert plugin.memory.decode_bytes(geometry) == 456


# --------------------------------------- content-boundary transforms
#
# content_in/content_out placement is generic plugin contract, not a
# KL detail: content_in runs ONCE on the whole input before any
# encoder call (direct or sweep), content_out runs ONCE on the whole
# output after the complete tiled average. The recording transforms
# below pin call counts and the shapes each transform actually saw.


class _Recorder:
    """A pointwise affine transform that records every call's shape."""

    def __init__(self, scale: float, shift: float) -> None:
        self.scale = scale
        self.shift = shift
        self.shapes: list[tuple[int, ...]] = []

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        self.shapes.append(tuple(tensor.shape))
        return tensor * self.scale + self.shift


def _transform_plugin(
    content_in: Callable[[torch.Tensor], torch.Tensor] | None = None,
    content_out: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> CodecPlugin:
    return CodecPlugin(
        descriptor=toy_plugin().descriptor,
        encoder=ToyEncoder(),
        decoder=ToyDecoder(),
        content_in=content_in,
        content_out=content_out,
    )


def test_content_in_applies_before_direct_encode() -> None:
    recorder = _Recorder(2.0, -1.0)
    content = torch.randn(1, 3, 16, 16)
    latent = _transform_plugin(content_in=recorder).encode(content)
    torch.testing.assert_close(latent, toy_plugin().encode(content * 2.0 - 1.0))
    assert recorder.shapes == [(1, 3, 16, 16)]


def test_content_out_applies_after_direct_decode() -> None:
    recorder = _Recorder(0.5, 0.5)
    latent = torch.randn(1, 4, 2, 2)
    out = _transform_plugin(content_out=recorder).decode(latent)
    torch.testing.assert_close(out, toy_plugin().decode(latent) * 0.5 + 0.5)
    assert recorder.shapes == [(1, 3, 16, 16)]


def test_content_in_runs_once_before_tiled_sweep() -> None:
    # 112 content = multi-tile traversal; the transform must still
    # see exactly one call, with the FULL input shape (pointwise
    # transforms commute with the sweep's slicing, so once-up-front
    # equals the reference's per-tile application)
    recorder = _Recorder(2.0, -1.0)
    content = torch.randn(1, 3, 112, 112)
    tiled = _transform_plugin(content_in=recorder).encode_tiled(content)
    assert recorder.shapes == [(1, 3, 112, 112)]
    torch.testing.assert_close(
        tiled,
        toy_plugin().encode_tiled(content * 2.0 - 1.0),
        rtol=1e-4,
        atol=1e-5,
    )


def test_content_out_runs_once_after_tiled_average() -> None:
    # multi-tile decode: one call, on the complete composed output
    # (never per tile - clamp-like transforms are observable across
    # seams if applied before the average)
    recorder = _Recorder(0.5, 0.5)
    latent = torch.randn(1, 4, 14, 14)
    tiled = _transform_plugin(content_out=recorder).decode_tiled(latent)
    assert recorder.shapes == [(1, 3, 112, 112)]
    torch.testing.assert_close(
        tiled,
        toy_plugin().decode_tiled(latent) * 0.5 + 0.5,
        rtol=1e-4,
        atol=1e-5,
    )


def test_content_out_may_mutate_in_place() -> None:
    # the documented contract: content_out receives the codec-owned
    # output and may mutate it in place
    def clamp_(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.clamp_(-0.5, 0.5)

    latent = torch.randn(1, 4, 4, 4)
    plugin = _transform_plugin(content_out=clamp_)
    torch.testing.assert_close(plugin.decode(latent), toy_plugin().decode(latent).clamp(-0.5, 0.5))
    tiled = plugin.decode_tiled(latent, tile=(16, 16), overlap=(2, 2))
    torch.testing.assert_close(tiled, toy_plugin().decode(latent).clamp(-0.5, 0.5))


def test_gradients_flow_through_content_transforms() -> None:
    content = torch.randn(1, 3, 16, 16, requires_grad=True)
    plugin = _transform_plugin(
        content_in=lambda t: t * 2.0 - 1.0,
        content_out=lambda t: (t + 1.0) / 2.0,
    )
    plugin.decode(plugin.encode(content)).sum().backward()
    assert content.grad is not None
    assert torch.isfinite(content.grad).all()


# ---------------------------------------- executed-reference sweeps
#
# The toy-codec equality tests above only prove tiled == direct for a
# tile-LOCAL codec, where every aspect pass computes identical values.
# The sweep goldens pin the full three-pass composition against the
# EXECUTED reference (comfy/sd.py decode_tiled_/encode_tiled_ via
# comfy.utils.tiled_scale @ 947c2749, run by
# tools/gen_tiling_goldens.py) using tile-globally-sensitive stand-ins
# (test_tiling.up2_tilenorm / down4_tilenorm), so each pass differs
# and the per-direction accumulation order and /3 average fold into
# the expected tensor.


class _FnDecoder:
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self._fn = fn

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self._fn(latent)


class _FnEncoder:
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self._fn = fn

    def encode(self, content: torch.Tensor) -> torch.Tensor:
        return self._fn(content)


def _sweep_case(name: str) -> dict[str, Any]:
    # Imported here: test_tiling loads a platform golden at import, and an
    # unminted tuple must skip only the tests that need it.
    import test_tiling

    for case in test_tiling.GOLDENS["sweep_cases"]:
        if case["name"] == name:
            return case
    raise KeyError(name)


def _sweep_plugin(case: dict[str, Any]) -> CodecPlugin:
    import test_tiling

    fn = test_tiling.FUNCTIONS[case["function"]]
    if case["decode"]:
        latent_channels = case["samples"]["shape"][1]
        content_channels = case["out_channels"]
    else:
        latent_channels = case["out_channels"]
        content_channels = case["samples"]["shape"][1]
    descriptor = CodecDescriptor(
        id=f"dinkster.golden.{case['name']}",
        display_name=f"Golden sweep {case['name']}",
        kind="image",
        latent=LatentDescriptor(
            channels=latent_channels,
            dimensions=2,
            spatial_downscale=case["factor"],
        ),
        supported_dtypes=frozenset({FLOAT32}),
        content_channels=content_channels,
    )
    return CodecPlugin(
        descriptor=descriptor,
        encoder=_FnEncoder(fn),
        decoder=_FnDecoder(fn),
    )


def test_sweep_decode_matches_executed_reference() -> None:
    import test_tiling

    case = _sweep_case("sweep_decode_2d")
    plugin = _sweep_plugin(case)
    samples = test_tiling.dec(case["samples"])
    out = plugin.decode_tiled(
        samples,
        tile=tuple(case["tile"]),
        overlap=tuple(case["overlap"]),
    )
    expected = test_tiling.dec(case["expected"])
    assert out.shape == expected.shape
    torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)


def test_sweep_encode_matches_executed_reference() -> None:
    import test_tiling

    case = _sweep_case("sweep_encode_2d")
    plugin = _sweep_plugin(case)
    samples = test_tiling.dec(case["samples"])
    out = plugin.encode_tiled(
        samples,
        tile=tuple(case["tile"]),
        overlap=tuple(case["overlap"]),
    )
    expected = test_tiling.dec(case["expected"])
    assert out.shape == expected.shape
    torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)


def test_sweep_pass_order_is_direction_specific() -> None:
    # independent of the goldens: record the tile geometry each codec
    # call receives and assert the per-direction variant order the
    # reference uses (decode (2t0,t1/2),(t0/2,2t1),(t0,t1); encode
    # (t0,t1),(t0/2,2t1),(2t0,t1/2))
    def grouped_shapes(seen: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
        groups: list[tuple[int, ...]] = []
        for shape in seen:
            if not groups or groups[-1] != shape:
                groups.append(shape)
        return groups

    seen: list[tuple[int, ...]] = []

    class SpyDecoder(ToyDecoder):
        def decode(self, latent: torch.Tensor) -> torch.Tensor:
            seen.append(tuple(latent.shape[2:]))
            return super().decode(latent)

    plugin = CodecPlugin(
        descriptor=toy_plugin().descriptor,
        encoder=ToyEncoder(),
        decoder=SpyDecoder(),
    )
    plugin.decode_tiled(torch.randn(1, 4, 14, 14), tile=(8, 8), overlap=(2, 2))
    # (16,4) clamps to (14,4); (4,16) clamps to (4,14); (8,8) exact
    assert grouped_shapes(seen) == [(14, 4), (4, 14), (8, 8)]

    seen.clear()

    class SpyEncoder(ToyEncoder):
        def encode(self, content: torch.Tensor) -> torch.Tensor:
            seen.append(tuple(content.shape[2:]))
            return super().encode(content)

    plugin = CodecPlugin(
        descriptor=toy_plugin().descriptor,
        encoder=SpyEncoder(),
        decoder=ToyDecoder(),
    )
    plugin.encode_tiled(torch.randn(1, 3, 112, 112), tile=(64, 64), overlap=(16, 16))
    # (64,64) exact; (32,128) clamps to (32,112); (128,32) to (112,32)
    assert grouped_shapes(seen) == [(64, 64), (32, 112), (112, 32)]


def test_invalid_sweep_variant_refuses_before_any_codec_call() -> None:
    # tile (8,8) with overlap (4,4): the first decode variant is
    # (16,4), whose second dim no longer exceeds the overlap - the
    # whole sweep must refuse in planning, before the decoder runs
    calls = 0

    class CountingDecoder(ToyDecoder):
        def decode(self, latent: torch.Tensor) -> torch.Tensor:
            nonlocal calls
            calls += 1
            return super().decode(latent)

    plugin = CodecPlugin(
        descriptor=toy_plugin().descriptor,
        encoder=ToyEncoder(),
        decoder=CountingDecoder(),
    )
    with pytest.raises(TilePlanError):
        plugin.decode_tiled(torch.randn(1, 4, 14, 14), tile=(8, 8), overlap=(4, 4))
    assert calls == 0


# ------------------------------------- malformed codec result guards


def _bad_result_plugin(
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> CodecPlugin:
    return CodecPlugin(
        descriptor=toy_plugin().descriptor,
        encoder=ToyEncoder(),
        decoder=_FnDecoder(fn),
    )


def test_tiled_decode_refuses_wrong_channel_count() -> None:
    # descriptor promises 3 content channels; return 1 - the
    # reference would silently broadcast the singleton
    plugin = _bad_result_plugin(lambda x: ToyDecoder().decode(x).mean(1, keepdim=True))
    with pytest.raises(TileApplyError, match="expected \\[1, 3"):
        plugin.decode_tiled(torch.randn(1, 4, 14, 14))


def test_tiled_decode_refuses_wrong_rank() -> None:
    plugin = _bad_result_plugin(lambda x: ToyDecoder().decode(x)[0])
    with pytest.raises(TileApplyError, match="content"):
        plugin.decode_tiled(torch.randn(1, 4, 14, 14))


def test_tiled_decode_refuses_wrong_batch() -> None:
    plugin = _bad_result_plugin(lambda x: ToyDecoder().decode(x).expand(2, -1, -1, -1))
    with pytest.raises(TileApplyError, match="expected \\[1, 3"):
        plugin.decode_tiled(torch.randn(1, 4, 14, 14))


def test_single_tile_refuses_wrong_content_shape() -> None:
    # single-tile fast path: content shape must equal the plan's
    # output exactly (a singleton content dim would broadcast)
    plugin = _bad_result_plugin(lambda x: ToyDecoder().decode(x)[:, :, :1, :])
    with pytest.raises(TileApplyError, match="single tile"):
        plugin.decode_tiled(torch.randn(1, 4, 4, 4), tile=(16, 16), overlap=(2, 2))


def test_gradients_flow_through_sweep_accumulation() -> None:
    # the in-place add_/div_ sweep accumulator must stay
    # autograd-safe end to end with a tile-globally-sensitive codec
    # (three genuinely different passes, like a real VAE)
    latent = torch.randn(1, 2, 13, 11, requires_grad=True)
    plugin = _sweep_plugin(_sweep_case("sweep_decode_2d"))
    plugin.decode_tiled(latent, tile=(8, 6), overlap=(2, 2)).sum().backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    assert bool((latent.grad != 0).any())
