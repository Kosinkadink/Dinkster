"""Panorama radial depth, masks, image boundaries and inference options."""

from __future__ import annotations

import importlib
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_api.v1 import initialize_inference_process
from dinkster_nodes_comfy.moge_panorama import MoGePanoramaInference
from dinkster_values import decode_image_array, encode_image_array


def controlled_runtime(monkeypatch: Any, mask_kind: str) -> Any:
    initialize_inference_process()
    panorama = importlib.import_module("dinkster_inference.ldm.moge.panorama")
    management = importlib.import_module("dinkster_inference.model_management")
    monkeypatch.setattr(management, "load_model_gpu", lambda patcher: None)

    def split(image: Any, extrinsics: Any, intrinsics: Any, resolution: int) -> Any:
        assert image.shape == (3, 64, 96) and image.dtype == torch.float64
        assert resolution == 256 and len(extrinsics) == len(intrinsics) == 12
        torch.testing.assert_close(
            image[:, 0, 0], torch.tensor([0.125, 0.25, 0.5], dtype=torch.float64)
        )
        return torch.arange(1.0, 13.0, dtype=torch.float64)[:, None, None, None].expand(12, 3, 2, 3)

    def merge(
        width: int,
        height: int,
        distances: Any,
        masks: Any,
        extrinsics: Any,
        intrinsics: Any,
        **callbacks: Any,
    ) -> Any:
        assert len(distances) == len(masks) == len(extrinsics) == len(intrinsics) == 12
        for index, distance in enumerate(distances):
            np.testing.assert_array_equal(distance, np.full((2, 3), 13 * (index + 1), np.float32))
            np.testing.assert_array_equal(masks[index], np.full((2, 3), index % 2 == 0))
            callbacks["on_view"]()
        callbacks["on_solve_start"](width, height)
        callbacks["on_solve_end"](width, height)
        mask = np.ones((height, width), dtype=bool)
        if mask_kind == "mixed":
            mask[:, width // 2 :] = False
        elif mask_kind == "none":
            mask[:] = False
        return np.broadcast_to(
            np.arange(1, width + 1, dtype=np.float32), (height, width)
        ).copy(), mask

    monkeypatch.setattr(panorama, "split_panorama_image", split)
    monkeypatch.setattr(panorama, "merge_panorama_depth", merge)
    return panorama


class PanoramaModel:
    patcher = object()
    load_device = torch.device("cpu")
    dtype = torch.float64

    def infer(self, images: Any, **options: Any) -> Any:
        assert options == dict(
            resolution_level=6,
            fov_x=90.0,
            force_projection=True,
            apply_mask=False,
            apply_metric_scale=False,
            refine_steps=2,
        )
        ids = images[:, 0, 0, 0]
        points = ids[:, None, None, None] * torch.tensor([3.0, 4.0, 12.0], dtype=torch.float64)
        mask = (ids.to(torch.int64) % 2 == 1)[:, None, None].expand(-1, 2, 3)
        return {
            "points": points.expand(-1, 2, 3, -1),
            "mask": mask,
            "normal": torch.full_like(points, 999.0),
        }


@pytest.mark.parametrize("portable", (False, True))
@pytest.mark.parametrize("resize", (False, True))
@pytest.mark.parametrize("mask_kind", ("mixed", "all", "none"))
def test_panorama_geometry_and_source_preservation(
    monkeypatch: Any, portable: bool, resize: bool, mask_kind: str
) -> None:
    controlled_runtime(monkeypatch, mask_kind)
    image = (
        torch.tensor([0.125, 0.25, 0.5, 0.875], dtype=torch.float64).expand(1, 64, 96, 4).clone()
    )
    before = image.clone()
    value: Any = decode_image_array(encode_image_array(image)) if portable else image
    result: Any = MoGePanoramaInference.execute(
        moge_model=PanoramaModel(),
        image=value,
        resolution_level=6,
        split_resolution=256,
        merge_resolution=48 if resize else 1920,
        batch_size=5,
        refine_steps=2,
    )
    geometry = result["moge_geometry"]
    assert set(geometry) == {"image", "depth", "mask", "points"}
    expected_mask = torch.ones((1, 64, 96), dtype=torch.bool)
    if mask_kind == "mixed":
        expected_mask[:, :, 48:] = False
    elif mask_kind == "none":
        expected_mask[:] = False
    # The merge returns a horizontal ramp. Bilinear pixel-center sampling
    # maps each output column to x/2 - 1/4 on the resized path.
    row = torch.arange(96, dtype=torch.float32) * 0.5 - 0.25 if resize else torch.arange(96.0)
    row = row.clamp(min=0, max=47 if resize else 95) + 1
    expected_depth = row.expand(1, 64, 96).clone()
    if mask_kind == "mixed":
        # Each valid column repeats 64 times. The 95th-percentile rank
        # lies wholly in column 45, unlike the maximum or median.
        expected_depth[:, :, 48:] = row[45] * 5
    torch.testing.assert_close(geometry["depth"], expected_depth)
    torch.testing.assert_close(geometry["mask"], expected_mask)
    torch.testing.assert_close(geometry["image"], before[..., :3])
    torch.testing.assert_close(image, before)
    torch.testing.assert_close(geometry["points"].norm(dim=-1), expected_depth)
    # Pixel-center equirectangular directions, independently derived at an
    # off-axis pixel to distinguish component swaps and spherical handedness.
    y, x = 13, 57
    theta, phi = (1 - (x + 0.5) / 96) * 2 * math.pi, (y + 0.5) / 64 * math.pi
    direction = torch.tensor(
        [math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi)]
    )
    torch.testing.assert_close(geometry["points"][0, y, x], direction * expected_depth[0, y, x])


def test_panorama_rejects_batched_source_before_model_access() -> None:
    with pytest.raises(ValueError, match=r"takes a single image \(got batch of 2\)"):
        MoGePanoramaInference.execute(moge_model=SimpleNamespace(), image=torch.zeros(2, 8, 12, 3))


def test_actual_cpu_split_and_merge_normalizes_constant_radial_depth(monkeypatch: Any) -> None:
    initialize_inference_process()
    management = importlib.import_module("dinkster_inference.model_management")
    monkeypatch.setattr(management, "load_model_gpu", lambda patcher: None)

    class ConstantModel(PanoramaModel):
        def infer(self, images: Any, **options: Any) -> Any:
            batch, _, height, width = images.shape
            return {
                "points": torch.tensor([0.0, 0.0, 3.0]).expand(batch, height, width, 3),
                "mask": torch.ones((batch, height, width), dtype=torch.bool),
            }

    result: Any = MoGePanoramaInference.execute(
        moge_model=ConstantModel(),
        image=torch.ones(1, 32, 64, 3),
        split_resolution=256,
    )
    geometry = result["moge_geometry"]
    assert geometry["mask"].all()
    # The gradient-only log-depth solve has no absolute scale constraint:
    # a constant radial prediction gives zero gradients and log depth zero.
    torch.testing.assert_close(geometry["depth"], torch.ones(1, 32, 64))
    torch.testing.assert_close(geometry["points"].norm(dim=-1), torch.ones(1, 32, 64))
