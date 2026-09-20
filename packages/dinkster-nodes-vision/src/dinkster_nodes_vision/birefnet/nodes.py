"""Stable node shape and boundary codecs for the BiRefNet matte provider."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_COMBO,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    decode_image_array,
    encode_image_array,
    image_array_fingerprint,
    image_array_meta,
    mask_array_meta,
    prepare_image_array_encoding,
)

IMAGE_TYPE = "dinkster.image"
MASK_TYPE = "dinkster.mask"
IMAGE = TypeExpr.concrete(IMAGE_TYPE)
MASK = TypeExpr.concrete(MASK_TYPE)
COMBO = TypeExpr.concrete(CORE_COMBO)
PROVIDER_CHOICE = "dinkster.image.matte.providers"


class ImageMatte(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.image.matte",
            display_name="Image Matte",
            category="image/matte",
            description=(
                "Estimates a foreground alpha matte for each image with a "
                "matting model. The mask is the soft foreground coverage; "
                "combine it with the input through alpha join or compositing "
                "for background removal."
            ),
            inputs=(
                InputSpec("image", IMAGE),
                InputSpec(
                    "provider",
                    COMBO,
                    required=False,
                    widget=ComboWidget(remote_route=f"/api/choices/{PROVIDER_CHOICE}"),
                    hidden=True,
                ),
            ),
            outputs=(
                OutputSpec(
                    "mask", MASK, preview=True, mask_polarity="coverage", mask_semantic="alpha"
                ),
            ),
            search_terms=(
                "background removal",
                "BiRefNet",
                "rembg",
                "alpha matte",
                "remove background",
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        provider: str = "",
    ) -> Mapping[str, object]:
        del provider
        from .model import execute_matte

        return cls.outputs(mask=execute_matte(image))


def _register_array(registry: TypeRegistry, type_id: str) -> None:
    if type_id not in registry:
        registry.register(
            type_id,
            encode=encode_image_array,
            decode=decode_image_array,
            prepare_buffer_encoding=prepare_image_array_encoding,
            fingerprint=image_array_fingerprint(type_id),
            meta=mask_array_meta if type_id == MASK_TYPE else image_array_meta,
        )


def register_types(registry: TypeRegistry) -> None:
    _register_array(registry, IMAGE_TYPE)
    _register_array(registry, MASK_TYPE)


BIREFNET_PROVIDER_NODES: tuple[type[Node], ...] = (ImageMatte,)


__all__ = ["BIREFNET_PROVIDER_NODES", "ImageMatte", "register_types"]
