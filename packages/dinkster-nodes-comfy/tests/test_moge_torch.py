"""Camera measurements preserve normalized-intrinsics and geometry selection."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from dinkster_nodes_comfy.moge import MoGeGeometryToFOV
from dinkster_values import decode_latent, encode_latent


@pytest.mark.parametrize(
    ("axis", "radians"),
    (
        ("horizontal", math.pi / 2),
        ("vertical", 2 * math.atan(0.5)),
        ("diagonal", 2 * math.atan(math.sqrt(1.25))),
    ),
)
@pytest.mark.parametrize("unit", ("degrees", "radians"))
@pytest.mark.parametrize("portable", (False, True))
def test_fov_uses_first_intrinsics_and_image_height(
    axis: str, radians: float, unit: str, portable: bool
) -> None:
    # Hand-derived: fx=.5 -> tan(fov_x/2)=1; fy=1 -> tan(fov_y/2)=.5.
    # The second camera and other geometry heights must not influence either result.
    intrinsics = torch.tensor(
        [[[0.5, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]], [[4.0, 0, 0.5], [0, 8.0, 0.5], [0, 0, 1.0]]]
    )
    geometry = {
        "intrinsics": intrinsics,
        "image": torch.zeros(2, 7, 11, 3),
        "points": torch.zeros(2, 13, 19, 3),
        "depth": torch.zeros(2, 23, 29),
    }
    before = intrinsics.clone()
    if portable:
        geometry = decode_latent(encode_latent(geometry), require_samples=False)
    result = MoGeGeometryToFOV.execute(moge_geometry=geometry, axis=axis, unit=unit)
    assert result["fov"] == pytest.approx(math.degrees(radians) if unit == "degrees" else radians)
    assert result["focal_pixels"] == 7.0
    assert torch.equal(intrinsics, before)


@pytest.mark.parametrize("source", ("points", "depth"))
def test_unbatched_intrinsics_and_height_fallback(source: str) -> None:
    geometry = {
        "intrinsics": torch.tensor([[1.0, 0, 0.5], [0, 0.25, 0.5], [0, 0, 1.0]]),
        source: torch.zeros(1, 12, 17, 3) if source == "points" else torch.zeros(1, 12, 17),
    }
    result = MoGeGeometryToFOV.execute(moge_geometry=geometry, axis="vertical", unit="radians")
    assert result["fov"] == pytest.approx(2 * math.atan(2))
    assert result["focal_pixels"] == 3.0


@pytest.mark.parametrize("geometry", (None, {}, {"intrinsics": None}))
def test_missing_intrinsics(geometry: Any) -> None:
    with pytest.raises(ValueError, match="no intrinsics"):
        MoGeGeometryToFOV.execute(moge_geometry=geometry, axis="vertical", unit="degrees")


def test_missing_geometry_height() -> None:
    with pytest.raises(ValueError, match="no image/points/depth"):
        MoGeGeometryToFOV.execute(
            moge_geometry={"intrinsics": torch.eye(3)}, axis="horizontal", unit="degrees"
        )
