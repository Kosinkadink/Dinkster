"""Proving tests for the torch-free tile planner and codec planning.

The oracle for the geometry is hand-derivation of comfy/utils.py
tiled_scale_multidim's position loop @ 947c2749 (positions, clamps,
lengths, output starts written out explicitly per case - not the
implementation's own formulas); the tensor-level end-to-end pin
against the EXECUTED reference lives in
packages/dinkster-inference-torch/tests/test_tiling.py (golden replay).
"""

from __future__ import annotations

import pytest
from dinkster_inference import (
    FLOAT32,
    CausalScale,
    CodecDescriptor,
    CodecTiling,
    LatentDescriptor,
    LinearScale,
    TilePlanError,
    TileSlice,
    latent_scales,
    plan_codec_decode,
    plan_codec_encode,
    plan_tiles,
)

# ------------------------------------------------------------ scales


def test_linear_scale_multiplies_and_divides() -> None:
    s = LinearScale(8)
    assert s.size(21, downscale=False) == 168
    assert s.size(21, downscale=True) == 21 / 8
    assert s.index(4, downscale=False) == 32
    assert s.index(4, downscale=True) == 0.5


def test_linear_scale_accepts_fractional_factor() -> None:
    # the reference's encode_tiled_1d passes 1/downscale_ratio
    s = LinearScale(0.125)
    assert s.size(16, downscale=False) == 2.0


def test_linear_scale_rejects_nonpositive() -> None:
    with pytest.raises(TilePlanError):
        LinearScale(0)
    with pytest.raises(TilePlanError):
        LinearScale(-4)


def test_causal_scale_matches_reference_lambda_pair() -> None:
    # decode: max(0, t*f - (f-1)); encode: max(0, floor((t+f-1)/f));
    # positions scale by plain f both ways (comfy/sd.py @ 947c2749)
    s = CausalScale(4)
    assert s.size(5, downscale=False) == 17
    assert s.size(1, downscale=False) == 1
    assert s.size(0, downscale=False) == 0
    assert s.size(12, downscale=True) == 3
    assert s.size(13, downscale=True) == 4
    assert s.size(0, downscale=True) == 0
    assert s.index(3, downscale=False) == 12
    assert s.index(12, downscale=True) == 3.0


def test_causal_scale_rejects_factor_below_one() -> None:
    with pytest.raises(TilePlanError):
        CausalScale(0)


# ------------------------------------------------------- plan_tiles


def test_plan_2d_positions_hand_derived() -> None:
    # size 21, tile 8, overlap 4: range(0, 17, 4) = 0,4,8,12,16;
    # lengths min(8, 21-pos); dst = 2*pos. size 17 dim: range(0, 13, 4)
    # = 0,4,8,12; lengths 8,8,8,5.
    plan = plan_tiles(
        (21, 17),
        (8, 8),
        overlap=4,
        scale=LinearScale(2),
    )
    assert plan.input_shape == (21, 17)
    assert plan.output_shape == (42, 34)
    assert plan.feather == (8, 8)
    assert not plan.single_tile
    assert len(plan.tiles) == 5 * 4
    first_dim = [t.dims[0] for t in plan.tiles[::4]]
    assert first_dim == [
        TileSlice(pos=0, length=8, dst=0),
        TileSlice(pos=4, length=8, dst=8),
        TileSlice(pos=8, length=8, dst=16),
        TileSlice(pos=12, length=8, dst=24),
        TileSlice(pos=16, length=5, dst=32),
    ]
    second_dim = [t.dims[1] for t in plan.tiles[:4]]
    assert second_dim == [
        TileSlice(pos=0, length=8, dst=0),
        TileSlice(pos=4, length=8, dst=8),
        TileSlice(pos=8, length=8, dst=16),
        TileSlice(pos=12, length=5, dst=24),
    ]


def test_plan_iterates_rows_before_columns() -> None:
    # itertools.product order: last dimension varies fastest
    plan = plan_tiles((12, 12), (8, 8), overlap=4, scale=LinearScale(1))
    seen = [(t.dims[0].pos, t.dims[1].pos) for t in plan.tiles]
    assert seen == [(0, 0), (0, 4), (4, 0), (4, 4)]


def test_plan_clamps_edge_position_into_bounds() -> None:
    # size 10, tile 8, overlap 4: range(0, 6, 4) = 0, 4;
    # pos 4 clamps to min(10-4, 4) = 4, length min(8, 6) = 6
    plan = plan_tiles((10,), (8,), overlap=4, scale=LinearScale(2))
    assert [t.dims[0] for t in plan.tiles] == [
        TileSlice(pos=0, length=8, dst=0),
        TileSlice(pos=4, length=6, dst=8),
    ]


def test_plan_single_tile_fast_path() -> None:
    plan = plan_tiles((6, 6), (8, 8), overlap=4, scale=LinearScale(2))
    assert plan.single_tile
    assert plan.tiles == ()
    assert plan.output_shape == (12, 12)


def test_plan_mixed_dimension_uses_position_zero() -> None:
    # dim 0 fits its tile -> the reference's [0] positions branch
    plan = plan_tiles((6, 20), (8, 8), overlap=4, scale=LinearScale(2))
    assert not plan.single_tile
    assert all(t.dims[0] == TileSlice(pos=0, length=6, dst=0) for t in plan.tiles)
    assert [t.dims[1].pos for t in plan.tiles] == [0, 4, 8, 12]


def test_plan_downscale_direction() -> None:
    # encode: sizes and positions divide; 56/8 = 7, pos 24 -> dst 3
    plan = plan_tiles(
        (56,),
        (32,),
        overlap=8,
        scale=LinearScale(8),
        downscale=True,
    )
    assert plan.output_shape == (7,)
    assert plan.feather == (1,)
    assert [t.dims[0] for t in plan.tiles] == [
        TileSlice(pos=0, length=32, dst=0),
        TileSlice(pos=24, length=32, dst=3),
    ]


def test_plan_causal_video_geometry() -> None:
    # 4 latent frames, causal f=4: output 4*4-3 = 13 frames; spatial
    # 6 @ x2 -> 12. time tile 3 > overlap 1: range(0, 3, 2) = 0, 2;
    # pos 2 -> length min(3, 2) = 2, dst 2*4 = 8
    plan = plan_tiles(
        (4, 6, 6),
        (3, 4, 4),
        overlap=(1, 2, 2),
        scale=(CausalScale(4), LinearScale(2), LinearScale(2)),
    )
    assert plan.output_shape == (13, 12, 12)
    assert plan.feather == (1, 4, 4)
    time_slices = sorted({t.dims[0] for t in plan.tiles}, key=lambda s: s.pos)
    assert time_slices == [
        TileSlice(pos=0, length=3, dst=0),
        TileSlice(pos=2, length=2, dst=8),
    ]


def test_plan_scalar_and_tuple_arguments_agree() -> None:
    a = plan_tiles((21, 17), (8, 8), overlap=4, scale=LinearScale(2))
    b = plan_tiles(
        (21, 17),
        (8, 8),
        overlap=(4, 4),
        scale=(LinearScale(2), LinearScale(2)),
    )
    assert a == b


def test_plan_rejects_bad_geometry() -> None:
    with pytest.raises(TilePlanError):
        plan_tiles((), (), overlap=4, scale=LinearScale(2))
    with pytest.raises(TilePlanError):
        plan_tiles((16,), (8, 8), overlap=4, scale=LinearScale(2))
    with pytest.raises(TilePlanError):
        plan_tiles((16, 16), (8, 8), overlap=(4,), scale=LinearScale(2))
    with pytest.raises(TilePlanError):
        plan_tiles((16,), (8,), overlap=4, scale=(LinearScale(2),) * 2)
    with pytest.raises(TilePlanError):
        plan_tiles((0,), (8,), overlap=4, scale=LinearScale(2))
    with pytest.raises(TilePlanError):
        plan_tiles((16,), (0,), overlap=4, scale=LinearScale(2))
    with pytest.raises(TilePlanError):
        plan_tiles((16,), (8,), overlap=-1, scale=LinearScale(2))


def test_plan_rejects_tile_not_exceeding_overlap() -> None:
    # the reference range()s with step tile - overlap: zero step
    # raises bare ValueError, negative step silently covers nothing
    # and NaNs the output - planning refuses loudly instead
    with pytest.raises(TilePlanError, match="must exceed overlap"):
        plan_tiles((16,), (4,), overlap=4, scale=LinearScale(2))
    with pytest.raises(TilePlanError, match="must exceed overlap"):
        plan_tiles((16,), (4,), overlap=6, scale=LinearScale(2))
    # legal when the input fits the tile (the reference never
    # range()s in that case)
    plan = plan_tiles((4,), (4,), overlap=6, scale=LinearScale(2))
    assert plan.single_tile


# ---------------------------------------------------- codec planning


def _image_codec(**overrides: object) -> CodecDescriptor:
    fields: dict[str, object] = dict(
        id="dinkster.test_image_vae",
        display_name="Test image VAE",
        kind="image",
        latent=LatentDescriptor(channels=4, dimensions=2, spatial_downscale=8),
        supported_dtypes=frozenset({FLOAT32}),
        tiling=CodecTiling(
            decode_tile=(64, 64),
            decode_overlap=(16, 16),
            encode_tile=(512, 512),
            encode_overlap=(64, 64),
        ),
    )
    fields.update(overrides)
    return CodecDescriptor(**fields)  # type: ignore[arg-type]


def test_latent_scales_per_dimensionality() -> None:
    assert latent_scales(LatentDescriptor(channels=2, dimensions=1, spatial_downscale=2048)) == (
        LinearScale(2048),
    )
    assert latent_scales(LatentDescriptor(channels=4, dimensions=2, spatial_downscale=8)) == (
        LinearScale(8),
        LinearScale(8),
    )
    assert latent_scales(
        LatentDescriptor(
            channels=16,
            dimensions=3,
            spatial_downscale=8,
            temporal_downscale=4,
            temporal_causal=True,
        )
    ) == (CausalScale(4), LinearScale(8), LinearScale(8))
    assert latent_scales(
        LatentDescriptor(channels=16, dimensions=3, spatial_downscale=8, temporal_downscale=4)
    ) == (LinearScale(4), LinearScale(8), LinearScale(8))


def test_temporal_causal_requires_three_dimensions() -> None:
    with pytest.raises(ValueError, match="temporal_causal"):
        LatentDescriptor(channels=4, dimensions=2, temporal_causal=True)


def test_codec_decode_plan_uses_descriptor_defaults() -> None:
    plan = plan_codec_decode(_image_codec(), (96, 96))
    assert plan.output_shape == (768, 768)
    assert plan.feather == (128, 128)
    assert len(plan.tiles) == 4  # range(0, 80, 48) = 0, 48 per dim


def test_codec_encode_plan_divides() -> None:
    plan = plan_codec_encode(_image_codec(), (1024, 1024))
    assert plan.output_shape == (128, 128)
    assert plan.feather == (8, 8)


def test_codec_plan_explicit_sizes_override_defaults() -> None:
    plan = plan_codec_decode(
        _image_codec(),
        (96, 96),
        tile=(96, 96),
        overlap=(16, 16),
    )
    assert plan.single_tile


def test_codec_plan_refuses_without_defaults_or_sizes() -> None:
    bare = _image_codec(tiling=None)
    with pytest.raises(TilePlanError, match="no tiling defaults"):
        plan_codec_decode(bare, (96, 96))
    plan = plan_codec_decode(bare, (96, 96), tile=(64, 64), overlap=(16, 16))
    assert len(plan.tiles) == 4


def test_codec_plan_refuses_when_tiling_unsupported() -> None:
    codec = _image_codec(supports_tiling=False, tiling=None)
    with pytest.raises(TilePlanError, match="does not support"):
        plan_codec_decode(codec, (96, 96), tile=(64, 64), overlap=(16, 16))
    with pytest.raises(TilePlanError, match="does not support"):
        plan_codec_encode(codec, (768, 768), tile=(512, 512), overlap=(64, 64))


def test_codec_descriptor_validates_tiling_shape() -> None:
    with pytest.raises(ValueError, match="entries for 2"):
        _image_codec(
            tiling=CodecTiling(
                decode_tile=(64,),
                decode_overlap=(16, 16),
                encode_tile=(512, 512),
                encode_overlap=(64, 64),
            )
        )
    with pytest.raises(ValueError, match="supports_tiling is False"):
        _image_codec(supports_tiling=False)
    with pytest.raises(ValueError, match="content_channels"):
        _image_codec(content_channels=0)
    with pytest.raises(ValueError, match="decode_tile"):
        CodecTiling(
            decode_tile=(0, 64),
            decode_overlap=(16, 16),
            encode_tile=(512, 512),
            encode_overlap=(64, 64),
        )
    with pytest.raises(ValueError, match="encode_overlap"):
        CodecTiling(
            decode_tile=(64, 64),
            decode_overlap=(16, 16),
            encode_tile=(512, 512),
            encode_overlap=(-1, 64),
        )


def test_codec_causal_video_plan_end_to_end() -> None:
    codec = CodecDescriptor(
        id="dinkster.test_video_vae",
        display_name="Test video VAE",
        kind="video",
        latent=LatentDescriptor(
            channels=16,
            dimensions=3,
            spatial_downscale=8,
            temporal_downscale=4,
            temporal_causal=True,
        ),
        supported_dtypes=frozenset({FLOAT32}),
        tiling=CodecTiling(
            decode_tile=(999, 32, 32),
            decode_overlap=(1, 8, 8),
            encode_tile=(9999, 512, 512),
            encode_overlap=(1, 64, 64),
        ),
    )
    decode = plan_codec_decode(codec, (5, 64, 64))
    assert decode.output_shape == (17, 512, 512)
    encode = plan_codec_encode(codec, (17, 512, 512))
    assert encode.output_shape == (5, 64, 64)
