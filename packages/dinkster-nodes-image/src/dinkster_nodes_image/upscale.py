"""Stable schema for model-backed image upscaling.

The node executes only through an installed vision provider pack
(dinkster-vision-upscale); this owner schema keeps the graph contract stable
while providers come and go.
"""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    ASSET_TYPE,
    CORE_COMBO,
    AssetWidget,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)

from .geometry import IMAGE, INT
from .support import number_input

ASSET = TypeExpr.concrete(ASSET_TYPE)
COMBO = TypeExpr.concrete(CORE_COMBO)

UPSCALE_MODEL_PROVIDER_CHOICE = "dinkster.image.upscale_model.providers"


class UpscaleWithModel(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.image.upscale_model",
            display_name="Upscale Image With Model",
            category="image/upscale",
            description=(
                "Upscales images with a trained super-resolution model "
                "(ESRGAN-family checkpoint asset). Large images are "
                "processed as overlapping tiles blended back together."
            ),
            inputs=(
                InputSpec("image", IMAGE),
                InputSpec(
                    "upscale_model",
                    ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",),
                        kind="model/upscale",
                    ),
                ),
                InputSpec(
                    "provider",
                    COMBO,
                    required=False,
                    widget=ComboWidget(
                        remote_route=f"/api/choices/{UPSCALE_MODEL_PROVIDER_CHOICE}",
                    ),
                    hidden=True,
                ),
                number_input("tile_size", INT, 512, minimum=128, maximum=16_384, step=64),
                number_input("overlap", INT, 32, minimum=0, maximum=2_048, step=8),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True),),
            search_terms=("upscale model", "ESRGAN", "super resolution", "upscaler"),
        )

    @classmethod
    def execute(cls, **_inputs: object) -> Mapping[str, object]:
        raise RuntimeError("model upscaling requires an installed provider")


def upscale_choices() -> dict[str, tuple[str, ...]]:
    return {UPSCALE_MODEL_PROVIDER_CHOICE: ()}


UPSCALE_NODES: tuple[type[Node], ...] = (UpscaleWithModel,)


__all__ = [
    "UPSCALE_MODEL_PROVIDER_CHOICE",
    "UPSCALE_NODES",
    "UpscaleWithModel",
    "upscale_choices",
]
