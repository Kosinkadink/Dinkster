from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import pytest
from dinkster_nodes_image import IMAGE_NODES, REFINE_NODES, TileBlendMask, TileRefinePlan
from dinkster_nodes_image.types import Region


def _regions(result: Mapping[str, object], output: str = "regions") -> list[Region]:
    return cast("list[Region]", result[output])


def _region_tuples(regions: list[Region]) -> list[tuple[int, int, int, int]]:
    return [(int(item.x), int(item.y), int(item.width), int(item.height)) for item in regions]


def _plan(**overrides: object) -> Mapping[str, object]:
    inputs: dict[str, object] = {
        "width": 64,
        "height": 64,
        "tile_width": 32,
        "tile_height": 32,
        "tile_padding": 0,
    }
    inputs.update(overrides)
    return TileRefinePlan.execute(**inputs)  # pyright: ignore[reportArgumentType]


def _mask(**overrides: object) -> np.ndarray:
    inputs: dict[str, object] = {
        "width": 5,
        "height": 5,
        "region": Region(0, 0, 5, 5),
    }
    inputs.update(overrides)
    result = TileBlendMask.execute(**inputs)  # pyright: ignore[reportArgumentType]
    return np.asarray(result["mask"], dtype=np.float32)


def test_refine_nodes_are_registered_with_the_declared_surface() -> None:
    assert REFINE_NODES == (TileRefinePlan, TileBlendMask)
    assert all(node in IMAGE_NODES for node in REFINE_NODES)
    assert TileRefinePlan.schema().node_type == "dinkster.image.tile_refine_plan"
    assert TileBlendMask.schema().node_type == "dinkster.mask.tile_blend"


def test_redraw_linear_and_chess_orders_cover_overflowing_edge_tiles() -> None:
    linear = _plan(width=20, height=20, tile_width=8, tile_height=8)
    expected_linear = [(x, y, 9, 9) for y in (0, 8, 16) for x in (0, 8, 16)]
    assert _region_tuples(_regions(linear)) == expected_linear
    assert _region_tuples(_regions(linear))[-1] == (16, 16, 9, 9)

    chess = _plan(width=20, height=20, tile_width=8, tile_height=8, mode="chess")
    expected_chess = [
        (0, 0, 9, 9),
        (16, 0, 9, 9),
        (8, 8, 9, 9),
        (0, 16, 9, 9),
        (16, 16, 9, 9),
        (8, 0, 9, 9),
        (0, 8, 9, 9),
        (16, 8, 9, 9),
        (8, 16, 9, 9),
    ]
    assert _region_tuples(_regions(chess)) == expected_chess


@pytest.mark.parametrize(
    "overrides",
    (
        {"mode": "none"},
        {"phase": "seam_fix", "seam_fix_mode": "none"},
    ),
)
def test_disabled_phase_emits_empty_equal_length_outputs(overrides: dict[str, object]) -> None:
    result = _plan(**overrides)
    assert result["count"] == 0
    for output in (
        "regions",
        "crops",
        "sample_widths",
        "sample_heights",
        "mask_kinds",
        "mask_blurs",
    ):
        assert result[output] == []


@pytest.mark.parametrize(
    ("overrides", "expected_count"),
    (
        ({}, 4),
        ({"mode": "chess"}, 4),
        ({"phase": "seam_fix", "seam_fix_mode": "band_pass"}, 2),
        ({"phase": "seam_fix", "seam_fix_mode": "half_tile"}, 4),
        ({"phase": "seam_fix", "seam_fix_mode": "half_tile_intersections"}, 5),
    ),
)
def test_plan_outputs_share_the_reported_count(
    overrides: dict[str, object], expected_count: int
) -> None:
    result = _plan(**overrides)
    assert result["count"] == expected_count
    lengths = {
        len(cast("list[object]", result[output]))
        for output in (
            "regions",
            "crops",
            "sample_widths",
            "sample_heights",
            "mask_kinds",
            "mask_blurs",
        )
    }
    assert lengths == {expected_count}


@pytest.mark.parametrize(
    ("tile_width", "expected"),
    ((508, 544), (500, 528)),
)
def test_uniform_sample_size_uses_python_ties_to_even_rounding(
    tile_width: int, expected: int
) -> None:
    result = _plan(
        width=1024,
        height=512,
        tile_width=tile_width,
        tile_height=480,
        tile_padding=32,
    )
    assert cast("list[int]", result["sample_widths"])[0] == expected


def test_crop_padding_and_interior_correction_match_reference_arithmetic() -> None:
    result = _plan(
        width=72,
        height=40,
        tile_width=24,
        tile_height=40,
        tile_padding=8,
        force_uniform_tiles=False,
    )
    assert _region_tuples(_regions(result, "crops")) == [
        (0, 0, 32, 40),
        (16, 0, 40, 40),
        (40, 0, 32, 40),
    ]


def test_redraw_crops_include_the_reference_inclusive_endpoint_pixel() -> None:
    result = _plan(
        width=1024,
        height=1024,
        tile_width=512,
        tile_height=512,
        tile_padding=32,
        force_uniform_tiles=False,
    )
    assert _region_tuples(_regions(result))[0] == (0, 0, 513, 513)
    assert _region_tuples(_regions(result, "crops"))[0] == (0, 0, 544, 544)


def test_uniform_and_minimal_edge_crops_expand_in_reference_preference_order() -> None:
    uniform = _plan(
        width=50,
        height=30,
        tile_width=32,
        tile_height=16,
        force_uniform_tiles=True,
    )
    minimal = _plan(
        width=50,
        height=30,
        tile_width=32,
        tile_height=16,
        force_uniform_tiles=False,
    )
    assert _region_tuples(_regions(uniform, "crops"))[1] == (18, 0, 32, 16)
    assert cast("list[int]", uniform["sample_widths"])[1] == 32
    assert cast("list[int]", uniform["sample_heights"])[1] == 16
    assert _region_tuples(_regions(minimal, "crops"))[1] == (26, 0, 24, 16)
    assert cast("list[int]", minimal["sample_widths"])[1] == 24
    assert cast("list[int]", minimal["sample_heights"])[1] == 16


def test_band_pass_orders_vertical_before_horizontal_strips() -> None:
    result = _plan(
        phase="seam_fix",
        seam_fix_mode="band_pass",
        seam_fix_width=16,
        seam_fix_padding=0,
    )
    assert _region_tuples(_regions(result)) == [(24, 0, 16, 64), (0, 24, 64, 16)]
    assert result["mask_kinds"] == ["tent_horizontal", "tent_vertical"]
    assert result["mask_blurs"] == [0, 0]
    assert result["sample_widths"] == [16, 64]
    assert result["sample_heights"] == [64, 16]


def test_half_tile_and_intersection_jobs_have_reference_order_and_offsets() -> None:
    half_tile = _plan(
        phase="seam_fix",
        seam_fix_mode="half_tile",
        seam_fix_padding=8,
        seam_fix_mask_blur=3,
    )
    assert _region_tuples(_regions(half_tile)) == [
        (0, 16, 32, 32),
        (32, 16, 32, 32),
        (16, 0, 32, 32),
        (16, 32, 32, 32),
    ]
    assert half_tile["mask_kinds"] == [
        "tent_vertical",
        "tent_vertical",
        "tent_horizontal",
        "tent_horizontal",
    ]
    assert half_tile["mask_blurs"] == [3, 3, 3, 3]

    intersections = _plan(
        phase="seam_fix",
        seam_fix_mode="half_tile_intersections",
        seam_fix_padding=8,
        seam_fix_mask_blur=3,
    )
    assert _region_tuples(_regions(intersections))[:-1] == _region_tuples(_regions(half_tile))
    assert _region_tuples(_regions(intersections))[-1] == (16, 16, 32, 32)
    assert _region_tuples(_regions(intersections, "crops"))[-1] == (16, 16, 31, 31)
    assert cast("list[str]", intersections["mask_kinds"])[-1] == "radial_inverted"
    assert cast("list[int]", intersections["mask_blurs"])[-1] == 3


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"width": 0}, "width"),
        ({"tile_width": 8193}, "tile_width"),
        ({"tile_height": 7}, "tile_height"),
        ({"tile_padding": -1}, "tile_padding"),
        ({"force_uniform_tiles": 1}, "force_uniform_tiles"),
        ({"phase": "other"}, "phase"),
        ({"mode": "other"}, "mode"),
        ({"seam_fix_mode": "other"}, "seam fix mode"),
        ({"seam_fix_width": 7}, "seam_fix_width"),
    ),
)
def test_plan_rejects_invalid_geometry(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _plan(**overrides)


def test_rectangle_mask_clips_nominal_region_at_canvas_corner() -> None:
    mask = _mask(width=4, height=3, region=Region(-2, -1, 4, 3))
    expected = np.zeros((1, 3, 4), dtype=np.float32)
    expected[:, :2, :2] = 1.0
    np.testing.assert_array_equal(mask, expected)


def test_tent_masks_follow_half_pixel_profiles_and_transpose() -> None:
    horizontal = _mask(width=4, height=3, region=Region(0, 0, 4, 3), kind="tent_horizontal")
    expected_horizontal = np.asarray([0.25, 0.75, 0.75, 0.25], dtype=np.float32)
    np.testing.assert_array_equal(horizontal[0], np.broadcast_to(expected_horizontal, (3, 4)))

    vertical = _mask(width=3, height=4, region=Region(0, 0, 3, 4), kind="tent_vertical")
    expected_vertical = expected_horizontal[:, None]
    np.testing.assert_array_equal(vertical[0], np.broadcast_to(expected_vertical, (4, 3)))


def test_clipped_tent_preserves_the_nominal_rectangle_profile() -> None:
    clipped = _mask(
        width=2,
        height=2,
        region=Region(-2, 0, 4, 2),
        kind="tent_horizontal",
    )
    np.testing.assert_array_equal(
        clipped,
        np.asarray([[[0.75, 0.25], [0.75, 0.25]]], dtype=np.float32),
    )


def test_radial_mask_has_exact_center_edge_midpoints_and_zero_corners() -> None:
    mask = _mask(kind="radial_inverted")
    assert mask.shape == (1, 5, 5)
    assert mask.dtype == np.float32
    assert mask[0, 2, 2] == 1.0
    assert mask[0, 0, 0] == 0.0
    assert mask[0, 0, 2] == pytest.approx(0.2)
    assert mask[0, 2, 0] == pytest.approx(0.2)
    assert np.all((0.0 <= mask) & (mask <= 1.0))


def test_blur_zero_is_exact_and_nonzero_blur_matches_pillow_quantization() -> None:
    region = Region(4, 4, 1, 1)
    exact = _mask(width=9, height=9, region=region, blur=0)
    blurred = _mask(width=9, height=9, region=region, blur=1)
    expected = np.zeros((1, 9, 9), dtype=np.float32)
    expected[0, 4, 4] = 1.0
    np.testing.assert_array_equal(exact, expected)
    assert float(blurred.sum()) == pytest.approx(244.0 / 255.0, abs=1e-7)
    np.testing.assert_array_equal(blurred, np.flip(blurred, axis=1))
    np.testing.assert_array_equal(blurred, np.flip(blurred, axis=2))


def test_refine_nodes_are_deterministic() -> None:
    plan_inputs = {
        "phase": "seam_fix",
        "seam_fix_mode": "half_tile_intersections",
        "seam_fix_padding": 8,
    }
    first_plan = _plan(**plan_inputs)
    second_plan = _plan(**plan_inputs)
    assert first_plan == second_plan

    mask_inputs = {
        "width": 9,
        "height": 7,
        "region": Region(-1, 1, 8, 5),
        "kind": "radial_inverted",
        "blur": 2,
    }
    np.testing.assert_array_equal(_mask(**mask_inputs), _mask(**mask_inputs))


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    (
        ({"width": 0}, ValueError, "width"),
        ({"region": Region(0, 0, 0, 1)}, ValueError, "dimensions"),
        ({"region": Region(0.5, 0, 2, 2)}, ValueError, "integers"),
        ({"kind": "other"}, ValueError, "kind"),
        ({"blur": -1}, ValueError, "blur"),
        ({"region": object()}, TypeError, "Region"),
    ),
)
def test_tile_blend_mask_rejects_invalid_inputs(
    overrides: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        _mask(**overrides)
