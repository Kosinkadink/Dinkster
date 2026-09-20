"""Stable node shape and boundary codecs for the upscale provider."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    ASSET_TYPE,
    CORE_COMBO,
    CORE_INT,
    AssetWidget,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    decode_image_array,
    encode_image_array,
    image_array_fingerprint,
    image_array_meta,
    prepare_image_array_encoding,
    register_asset_type,
    resolver_from_env,
)

IMAGE_TYPE = "dinkster.image"
IMAGE = TypeExpr.concrete(IMAGE_TYPE)
ASSET = TypeExpr.concrete(ASSET_TYPE)
INT = TypeExpr.concrete(CORE_INT)
COMBO = TypeExpr.concrete(CORE_COMBO)
PROVIDER_CHOICE = "dinkster.image.upscale_model.providers"


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
                    widget=ComboWidget(remote_route=f"/api/choices/{PROVIDER_CHOICE}"),
                    hidden=True,
                ),
                InputSpec(
                    "tile_size",
                    INT,
                    required=False,
                    default=512,
                    widget=NumberWidget(min=128, max=16_384, step=64),
                ),
                InputSpec(
                    "overlap",
                    INT,
                    required=False,
                    default=32,
                    widget=NumberWidget(min=0, max=2_048, step=8),
                ),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True),),
            search_terms=("upscale model", "ESRGAN", "super resolution", "upscaler"),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        upscale_model: object,
        provider: str,
        tile_size: int = 512,
        overlap: int = 32,
    ) -> Mapping[str, object]:
        del provider
        from .execute import execute_upscale

        return cls.outputs(
            image=execute_upscale(
                image,
                upscale_model,
                tile_size=tile_size,
                overlap=overlap,
            )
        )


def register_types(registry: TypeRegistry) -> None:
    if IMAGE_TYPE not in registry:
        registry.register(
            IMAGE_TYPE,
            encode=encode_image_array,
            decode=decode_image_array,
            prepare_buffer_encoding=prepare_image_array_encoding,
            fingerprint=image_array_fingerprint(IMAGE_TYPE),
            meta=image_array_meta,
        )
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())


UPSCALE_PROVIDER_NODES: tuple[type[Node], ...] = (UpscaleWithModel,)


__all__ = ["UPSCALE_PROVIDER_NODES", "UpscaleWithModel", "register_types"]
