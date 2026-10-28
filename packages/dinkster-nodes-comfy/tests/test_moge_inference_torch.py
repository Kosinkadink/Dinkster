"""Chunked geometry output, image boundaries and optional fields."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_api.v1 import materialize_image
from dinkster_nodes_comfy.moge_inference import MoGeInference
from dinkster_values import decode_image_array, encode_image_array


class GeometryModel:
    def __init__(self, expected_options: dict[str, Any], partial: bool) -> None:
        self.options = expected_options
        self.partial = partial

    def infer(self, pixels: Any, **options: Any) -> Any:
        assert options == self.options
        assert pixels.is_contiguous() and pixels.shape[1:] == (3, 3, 7)
        assert pixels.dtype == torch.float64
        # Each channel carries a distinct offset; alpha must never enter inference.
        red, green, blue = pixels.unbind(1)
        torch.testing.assert_close(green, red + 10)
        torch.testing.assert_close(blue, red + 20)
        ids = red[:, 0, 0]
        packet: dict[str, Any] = {"depth": red + 100, "points": pixels.movedim(1, -1)}
        if not self.partial:
            packet.update(
                intrinsics=ids[:, None, None].expand(-1, 3, 3).clone(),
                mask=red > 2,
                normal=pixels.movedim(1, -1) * -1,
            )
        elif ids[0] < 2:
            packet["mask"] = red > 2
        return packet


@pytest.mark.parametrize("portable", (False, True))
@pytest.mark.parametrize("fov", (-5.0, 0.0, 71.5))
@pytest.mark.parametrize("partial", (False, True))
def test_chunked_rgb_order_options_optional_fields_and_nonmutation(
    portable: bool, fov: float, partial: bool
) -> None:
    # Hand-derived: five image IDs, three different channel offsets, ignored
    # alpha 99. A two-image chunk size leaves an asymmetric one-image tail.
    ids = torch.arange(5.0, dtype=torch.float64)
    pixels = torch.stack((ids, ids + 10, ids + 20, ids + 99), -1)[:, None, None, :]
    pixels = pixels.expand(5, 3, 7, 4).clone()
    before = pixels.clone()
    image: Any = pixels
    if portable:
        image = decode_image_array(encode_image_array(pixels))
    model = GeometryModel(
        {
            "resolution_level": 6,
            "fov_x": None if fov <= 0 else fov,
            "force_projection": False,
            "apply_mask": True,
            "refine_steps": 2,
        },
        partial,
    )
    result: Any = MoGeInference.execute(
        moge_model=model,
        image=image,
        resolution_level=6,
        fov_x_degrees=fov,
        batch_size=2,
        force_projection=False,
        apply_mask=True,
        refine_steps=2,
    )
    geometry = result["moge_geometry"]
    rgb = before[..., :3]
    torch.testing.assert_close(geometry["image"], rgb)
    torch.testing.assert_close(geometry["points"], rgb)
    torch.testing.assert_close(geometry["depth"], rgb[..., 0] + 100)
    assert geometry["image"].device.type == "cpu"
    assert geometry["depth"].dtype == torch.float64
    if partial:
        assert set(geometry) == {"image", "points", "depth", "mask"}
        # F concatenates only chunks that supplied each optional field.
        torch.testing.assert_close(geometry["mask"], rgb[:2, ..., 0] > 2)
    else:
        torch.testing.assert_close(geometry["intrinsics"], ids[:, None, None].expand(5, 3, 3))
        torch.testing.assert_close(geometry["mask"], rgb[..., 0] > 2)
        torch.testing.assert_close(geometry["normal"], -rgb)
    torch.testing.assert_close(pixels, before)


def test_defaults_empty_predictions_and_numpy_input() -> None:
    image = torch.full((1, 3, 7, 4), 0.75, dtype=torch.float32).numpy()

    class NoGeometry:
        def infer(self, pixels: Any, **options: Any) -> Any:
            assert options == {
                "resolution_level": 9,
                "fov_x": None,
                "force_projection": True,
                "apply_mask": True,
                "refine_steps": 3,
            }
            torch.testing.assert_close(pixels, torch.full((1, 3, 3, 7), 0.75))
            return {"ignored": torch.tensor([99])}

    result: Any = MoGeInference.execute(moge_model=NoGeometry(), image=image)
    assert set(result["moge_geometry"]) == {"image"}
    torch.testing.assert_close(result["moge_geometry"]["image"], materialize_image(image)[..., :3])
