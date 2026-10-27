"""CPU contracts for camera-conditioned Wan video inputs."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    IMAGE_TYPE,
    WAN_CAMERA_EMBEDDING_TYPE,
    TypeRegistry,
    register_inference_types,
)
from dinkster_nodes_comfy.wan_camera import WanCameraEmbedding
from dinkster_nodes_comfy.wan_camera_conditioning import WanCameraImageToVideo


class VAE:
    def __init__(self) -> None:
        self.image: torch.Tensor | None = None

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self.image = image.clone()
        return image[::4, ::8, ::8, 0].unsqueeze(0).unsqueeze(0).repeat(1, 16, 1, 1, 1)


def _conditioning(label: str) -> list[list[Any]]:
    return [[torch.tensor([[1.0, 3.0]]), {"label": label}]]


def test_camera_schema_and_absent_inputs_preserve_conditioning() -> None:
    schema = WanCameraImageToVideo.schema()
    assert [item.id for item in schema.inputs] == [
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "clip_vision_output",
        "start_image",
        "camera_conditions",
    ]
    assert schema.inputs[-1].type.types == (WAN_CAMERA_EMBEDDING_TYPE,)
    assert [item.required for item in schema.inputs[-3:]] == [False, False, False]
    assert [item.id for item in schema.outputs] == ["positive", "negative", "latent"]
    positive, negative = _conditioning("p"), _conditioning("n")
    result = WanCameraImageToVideo.execute(
        positive=positive, negative=negative, vae=VAE(), width=24, height=16, length=9, batch_size=2
    )
    assert result["positive"] is positive
    assert result["negative"] is negative
    samples = result["latent"]["samples"]
    assert samples.shape == (2, 16, 3, 2, 3)
    assert samples.dtype is torch.float32
    assert not samples.count_nonzero()


def test_camera_and_image_decode_public_values_without_mutating_sources() -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    image = torch.stack([torch.full((16, 24, 4), float(index)) for index in range(6)])
    camera = torch.arange(72, dtype=torch.float16).reshape(1, 3, 4, 2, 3).transpose(-1, -2)
    image_before, camera_before = image.clone(), camera.clone()
    encoded_image = registry.spec(IMAGE_TYPE).decode(registry.spec(IMAGE_TYPE).encode(image))
    encoded_camera = registry.spec(WAN_CAMERA_EMBEDDING_TYPE).decode(
        registry.spec(WAN_CAMERA_EMBEDDING_TYPE).encode(camera)
    )
    positive, negative = _conditioning("p"), _conditioning("n")
    vae = VAE()
    result = WanCameraImageToVideo.execute(
        positive=positive,
        negative=negative,
        vae=vae,
        width=24,
        height=16,
        length=9,
        batch_size=2,
        start_image=encoded_image,
        camera_conditions=encoded_camera,
    )
    assert vae.image is not None
    assert vae.image.shape == (6, 16, 24, 3)
    expected_encoded = torch.tensor([0.0, 4.0]).reshape(1, 1, 2, 1, 1).expand(2, 16, 2, 2, 3)
    metadata = result["positive"][0][1]
    assert torch.equal(metadata["concat_latent_image"][:, :, :2], expected_encoded)
    assert metadata["concat_mask"].shape == (1, 4, 3, 2, 3)
    chronological_mask = metadata["concat_mask"].transpose(1, 2).reshape(12, 2, 3)
    assert not chronological_mask[:9].count_nonzero()
    assert chronological_mask[9:].all()
    assert torch.equal(metadata["camera_conditions"], camera)
    assert metadata["camera_conditions"].dtype is torch.float16
    assert result["negative"][0][1]["camera_conditions"] is metadata["camera_conditions"]
    assert metadata["label"] == "p"
    assert result["negative"][0][1]["label"] == "n"
    assert set(positive[0][1]) == {"label"}
    assert set(negative[0][1]) == {"label"}
    assert torch.equal(image, image_before)
    assert torch.equal(camera, camera_before)


def test_camera_embedding_preserves_ray_geometry_and_temporal_packing() -> None:
    parameters = dict(width=32, height=16, length=5, fx=0.5, fy=0.25, cx=0.3, cy=0.6)
    static = WanCameraEmbedding.execute(camera_pose="Static", **parameters)
    embedding = static["camera_embedding"]
    assert (static["width"], static["height"], static["length"]) == (32, 16, 5)
    assert embedding.shape == (1, 24, 2, 16, 32)
    assert embedding.dtype is torch.float32
    assert not embedding[:, :12].count_nonzero()
    x = (0.5 - 0.3 * 32) / (0.5 * 32)
    y = (0.5 - 0.6 * 16) / (32 / (1280 / 720) * 0.25)
    norm = math.sqrt(x * x + y * y + 1.0)
    direction = (x / norm, y / norm, 1.0 / norm)
    for channel, expected in zip((12, 16, 20), direction, strict=True):
        assert embedding[0, channel, 0, 0, 0].item() == pytest.approx(expected, abs=2e-7)
        assert torch.equal(
            embedding[:, channel : channel + 4, 0], embedding[:, channel : channel + 4, 1]
        )
    moving = WanCameraEmbedding.execute(camera_pose="Pan Right", **parameters)["camera_embedding"]
    assert not moving[:, :12, 0].count_nonzero()
    # The last packed subframe is source frame four, translated by 4/5 * 1.5.
    assert moving[0, 7, 1, 0, 0].item() == pytest.approx(-1.2 * direction[2], abs=2e-7)
    assert moving[0, 11, 1, 0, 0].item() == pytest.approx(1.2 * direction[1], abs=2e-7)
