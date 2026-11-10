"""Video latent schema and tensor creation helpers shared by node packs."""

from __future__ import annotations

import importlib

from dinkster_api.v1 import (
    CORE_INT,
    LATENT_TYPE,
    InputSpec,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
)

__all__ = ["video_latent_schema", "empty_video_latent"]

_LATENT = TypeExpr.concrete(LATENT_TYPE)


def _integer(
    name: str, default: int, minimum: int, maximum: int, step: int | None = None
) -> InputSpec:
    return InputSpec(
        name,
        TypeExpr.concrete(CORE_INT),
        default=default,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


def _schema(
    node_type: str,
    display_name: str,
    category: str,
    *,
    width: int,
    height: int,
    length: int,
    maximum: int = 16384,
    temporal_maximum: int | None = None,
    batch_maximum: int = 4096,
    spatial_minimum: int = 16,
    spatial_step: int = 16,
    temporal_step: int = 4,
) -> NodeSchema:
    return NodeSchema(
        node_type=node_type,
        display_name=display_name,
        category=category,
        inputs=(
            _integer("width", width, spatial_minimum, maximum, spatial_step),
            _integer("height", height, spatial_minimum, maximum, spatial_step),
            _integer("length", length, 1, temporal_maximum or maximum, temporal_step),
            _integer("batch_size", 1, 1, batch_maximum),
        ),
        outputs=(OutputSpec("LATENT", _LATENT),),
    )


def _empty_video_latent(
    batch_size: int,
    channels: int,
    length: int,
    temporal_ratio: int,
    height: int,
    width: int,
    spatial_ratio: int,
    downscale_ratio_spacial: int | None = None,
) -> dict[str, object]:
    torch = importlib.import_module("torch")
    management = importlib.import_module("dinkster_inference.model_management")
    latent: dict[str, object] = {
        "samples": torch.zeros(
            [
                batch_size,
                channels,
                ((length - 1) // temporal_ratio) + 1,
                height // spatial_ratio,
                width // spatial_ratio,
            ],
            device=management.intermediate_device(),
        )
    }
    if downscale_ratio_spacial is not None:
        latent["downscale_ratio_spacial"] = downscale_ratio_spacial
    return latent


video_latent_schema = _schema
empty_video_latent = _empty_video_latent
