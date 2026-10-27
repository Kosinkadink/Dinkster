"""Asymmetric geometry visualization and portable tensor consumption."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_nodes_comfy.moge_render import MoGeRender
from dinkster_values import decode_latent, encode_latent


def portable_geometry(geometry: Any, portable: bool) -> Any:
    return decode_latent(encode_latent(geometry), require_samples=False) if portable else geometry


@pytest.mark.parametrize("portable", (False, True))
@pytest.mark.parametrize("output", ("normal_opengl", "normal_directx", "mask"))
def test_predicted_normals_mask_and_source_preservation(portable: bool, output: str) -> None:
    normals = torch.tensor(
        [[[[2.0, -0.6, 0.8], [-0.2, 0.4, -2.0]]], [[[0.0, 1.0, -1.0], [0.6, -0.8, 0.0]]]],
        dtype=torch.float64,
    )
    mask = torch.tensor([[[False, True]], [[True, False]]])
    geometry = {"normal": normals, "mask": mask, "points": torch.full_like(normals, float("nan"))}
    before = normals.clone()
    result: Any = MoGeRender.execute(
        moge_geometry=portable_geometry(geometry, portable), output=output
    )
    # Hand-derived channel signs: OpenGL flips Y, both conventions flip Z.
    expected = {
        "normal_opengl": [
            [[[1.0, 0.8, 0.1], [0.4, 0.3, 1.0]]],
            [[[0.5, 0.0, 1.0], [0.8, 0.9, 0.5]]],
        ],
        "normal_directx": [
            [[[1.0, 0.2, 0.1], [0.4, 0.7, 1.0]]],
            [[[0.5, 1.0, 1.0], [0.8, 0.1, 0.5]]],
        ],
        "mask": [[[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], [[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]]],
    }
    torch.testing.assert_close(result["IMAGE"], torch.tensor(expected[output]))
    assert result["IMAGE"].dtype == torch.float32 and result["IMAGE"].device.type == "cpu"
    torch.testing.assert_close(normals, before)


@pytest.mark.parametrize("portable", (False, True))
@pytest.mark.parametrize("output", ("depth", "depth_colored"))
def test_depth_percentiles_batch_and_invalid_pixels(portable: bool, output: str) -> None:
    depth = torch.tensor(
        [
            [[1.0, 2.0, 4.0, 0.0, -1.0, float("inf"), float("nan")]],
            [[2.0, 4.0, 8.0, 0.0, -1.0, float("inf"), float("nan")]],
        ],
        dtype=torch.float64,
    )
    result: Any = MoGeRender.execute(
        moge_geometry=portable_geometry({"depth": depth}, portable), output=output
    )
    if output == "depth":
        # Three valid disparities [1, .5, .25]: q.001=.2505, q.999=.999.
        middle = (0.5 - 0.2505) / (0.999 - 0.2505)
        expected = torch.tensor([1.0, middle, 0.0, 0.0, 0.0, 0.0, 0.0])
        torch.testing.assert_close(
            result["IMAGE"], expected[None, None, :, None].expand(2, 1, 7, 3)
        )
    else:
        # Turbo's polynomial constant is the black-end color for invalid/zero disparity.
        torch.testing.assert_close(
            result["IMAGE"][:, :, 2:],
            torch.tensor([0.13572138, 0.09140261, 0.10667330]).expand(2, 1, 5, 3),
        )
        assert torch.isfinite(result["IMAGE"]).all()
        torch.testing.assert_close(result["IMAGE"][0], result["IMAGE"][1])
    torch.testing.assert_close(
        depth[:, :, :5],
        torch.tensor(
            [[[1.0, 2.0, 4.0, 0.0, -1.0]], [[2.0, 4.0, 8.0, 0.0, -1.0]]], dtype=torch.float64
        ),
    )


@pytest.mark.parametrize("portable", (False, True))
def test_all_invalid_depth_renders_black(portable: bool) -> None:
    result: Any = MoGeRender.execute(
        moge_geometry=portable_geometry(
            {"depth": torch.tensor([[[0.0, -1.0, float("nan")]]])}, portable
        )
    )
    assert torch.count_nonzero(result["IMAGE"]) == 0


@pytest.mark.parametrize("output", ("normal_opengl", "normal_directx"))
def test_point_normal_fallback_cross_orientation_and_invalid_mask(output: str) -> None:
    # Hand-derived plane z=x+2y: dy x dx gives (1,2,-1)/sqrt(6).
    yy, xx = torch.meshgrid(torch.arange(5.0), torch.arange(5.0), indexing="ij")
    points = torch.stack((xx, yy, xx + 2 * yy), dim=-1)[None]
    points[0, 0, 0] = float("nan")
    result: Any = MoGeRender.execute(moge_geometry={"points": points}, output=output)
    sign = -1 if output == "normal_opengl" else 1
    expected = (torch.tensor([1.0, 2.0 * sign, 1.0]) / (6**0.5)) * 0.5 + 0.5
    torch.testing.assert_close(result["IMAGE"][0, 2, 2], expected)
    torch.testing.assert_close(result["IMAGE"][0, 0, 0], torch.full((3,), 0.5))


@pytest.mark.parametrize(
    "output, message",
    [
        ("depth", "has no depth output"),
        ("depth_colored", "has no depth output"),
        ("normal_opengl", "has neither normals nor points"),
        ("normal_directx", "has neither normals nor points"),
        ("mask", "has no mask output"),
        ("unknown", "Unknown output mode: unknown"),
    ],
)
def test_missing_fields_and_unknown_mode(output: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MoGeRender.execute(moge_geometry={}, output=output)
