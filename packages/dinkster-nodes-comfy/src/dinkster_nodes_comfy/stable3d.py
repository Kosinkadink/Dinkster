"""Native Stable Zero123 and Stable Video 3D conditioning nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CLIP_VISION_TYPE,
    CONDITIONING_TYPE,
    CORE_FLOAT,
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_image,
)

_CLIP_VISION = TypeExpr.concrete(CLIP_VISION_TYPE)
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)
_ZERO123_CATEGORY = "model/conditioning/stable zero123"


def camera_embeddings(elevation: float, azimuth: float) -> Any:
    torch = importlib.import_module("torch")
    elevation_value = torch.as_tensor([elevation])
    azimuth_value = torch.as_tensor([azimuth])
    return torch.stack(
        [
            torch.deg2rad((90 - elevation_value) - 90),
            torch.sin(torch.deg2rad(azimuth_value)),
            torch.cos(torch.deg2rad(azimuth_value)),
            torch.deg2rad(90 - torch.full_like(elevation_value, 0)),
        ],
        dim=-1,
    ).unsqueeze(1)


def _integer(
    name: str, default: int, minimum: int, maximum: int, step: int | None = None
) -> InputSpec:
    return InputSpec(
        name,
        _INT,
        default=default,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


def _angle(name: str, minimum: float, maximum: float, *, advanced: bool = False) -> InputSpec:
    return InputSpec(
        name,
        _FLOAT,
        default=0.0,
        widget=NumberWidget(min=minimum, max=maximum, step=0.1, round=False),
        advanced=advanced,
    )


def _conditioning_outputs() -> tuple[OutputSpec, ...]:
    return (
        OutputSpec("positive", _CONDITIONING),
        OutputSpec("negative", _CONDITIONING),
        OutputSpec("latent", _LATENT),
    )


def _encode_image(
    clip_vision: Any, init_image: object, vae: Any, width: int, height: int
) -> tuple[Any, Any]:
    image = materialize_image(init_image)
    output = clip_vision.encode_image(image)
    utils = importlib.import_module("dinkster_inference.utils")
    pixels = utils.common_upscale(
        image.movedim(-1, 1), width, height, "bilinear", "center"
    ).movedim(1, -1)
    return output.image_embeds.unsqueeze(0), vae.encode(pixels[:, :, :, :3])


class StableZero123_Conditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="StableZero123_Conditioning",
            category=_ZERO123_CATEGORY,
            inputs=(
                InputSpec("clip_vision", _CLIP_VISION),
                InputSpec("init_image", _IMAGE),
                InputSpec("vae", _VAE),
                _integer("width", 256, 16, 16384, 8),
                _integer("height", 256, 16, 16384, 8),
                _integer("batch_size", 1, 1, 4096),
                _angle("elevation", -180.0, 180.0),
                _angle("azimuth", -180.0, 180.0),
            ),
            outputs=_conditioning_outputs(),
        )

    @classmethod
    def execute(
        cls,
        *,
        clip_vision: Any,
        init_image: object,
        vae: Any,
        width: int,
        height: int,
        batch_size: int,
        elevation: float,
        azimuth: float,
    ) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        pooled, encoded = _encode_image(clip_vision, init_image, vae, width, height)
        camera = camera_embeddings(elevation, azimuth)
        conditioning = torch.cat(
            [pooled, camera.to(pooled.device).repeat((pooled.shape[0], 1, 1))], dim=-1
        )
        return cls.outputs(
            positive=[[conditioning, {"concat_latent_image": encoded}]],
            negative=[
                [torch.zeros_like(pooled), {"concat_latent_image": torch.zeros_like(encoded)}]
            ],
            latent={"samples": torch.zeros([batch_size, 4, height // 8, width // 8])},
        )


class StableZero123_Conditioning_Batched(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="StableZero123_Conditioning_Batched",
            category=_ZERO123_CATEGORY,
            inputs=(
                InputSpec("clip_vision", _CLIP_VISION),
                InputSpec("init_image", _IMAGE),
                InputSpec("vae", _VAE),
                _integer("width", 256, 16, 16384, 8),
                _integer("height", 256, 16, 16384, 8),
                _integer("batch_size", 1, 1, 4096),
                _angle("elevation", -180.0, 180.0),
                _angle("azimuth", -180.0, 180.0),
                _angle("elevation_batch_increment", -180.0, 180.0, advanced=True),
                _angle("azimuth_batch_increment", -180.0, 180.0, advanced=True),
            ),
            outputs=_conditioning_outputs(),
        )

    @classmethod
    def execute(
        cls,
        *,
        clip_vision: Any,
        init_image: object,
        vae: Any,
        width: int,
        height: int,
        batch_size: int,
        elevation: float,
        azimuth: float,
        elevation_batch_increment: float,
        azimuth_batch_increment: float,
    ) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        pooled, encoded = _encode_image(clip_vision, init_image, vae, width, height)
        cameras: list[Any] = []
        for _ in range(batch_size):
            cameras.append(camera_embeddings(elevation, azimuth))
            elevation += elevation_batch_increment
            azimuth += azimuth_batch_increment
        utils = importlib.import_module("dinkster_inference.utils")
        conditioning = torch.cat(
            [utils.repeat_to_batch_size(pooled, batch_size), torch.cat(cameras, dim=0)], dim=-1
        )
        return cls.outputs(
            positive=[[conditioning, {"concat_latent_image": encoded}]],
            negative=[
                [torch.zeros_like(pooled), {"concat_latent_image": torch.zeros_like(encoded)}]
            ],
            latent={
                "samples": torch.zeros([batch_size, 4, height // 8, width // 8]),
                "batch_index": [0] * batch_size,
            },
        )


class SV3D_Conditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SV3D_Conditioning",
            category="model/conditioning/stable video 3d",
            inputs=(
                InputSpec("clip_vision", _CLIP_VISION),
                InputSpec("init_image", _IMAGE),
                InputSpec("vae", _VAE),
                _integer("width", 576, 16, 16384, 8),
                _integer("height", 576, 16, 16384, 8),
                _integer("video_frames", 21, 1, 4096),
                _angle("elevation", -90.0, 90.0),
            ),
            outputs=_conditioning_outputs(),
        )

    @classmethod
    def execute(
        cls,
        *,
        clip_vision: Any,
        init_image: object,
        vae: Any,
        width: int,
        height: int,
        video_frames: int,
        elevation: float,
    ) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        pooled, encoded = _encode_image(clip_vision, init_image, vae, width, height)
        azimuth = 0.0
        azimuth_increment = 360 / (max(video_frames, 2) - 1)
        elevations: list[float] = []
        azimuths: list[float] = []
        for _ in range(video_frames):
            elevations.append(elevation)
            azimuths.append(azimuth)
            azimuth += azimuth_increment
        metadata: dict[str, Any] = {
            "concat_latent_image": encoded,
            "elevation": elevations,
            "azimuth": azimuths,
        }
        return cls.outputs(
            positive=[[pooled, metadata]],
            negative=[
                [
                    torch.zeros_like(pooled),
                    {**metadata, "concat_latent_image": torch.zeros_like(encoded)},
                ]
            ],
            latent={"samples": torch.zeros([video_frames, 4, height // 8, width // 8])},
        )


STABLE3D_NODES = (
    StableZero123_Conditioning,
    StableZero123_Conditioning_Batched,
    SV3D_Conditioning,
)
