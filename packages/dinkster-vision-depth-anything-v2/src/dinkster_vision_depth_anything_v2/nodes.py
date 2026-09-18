"""Stable node shape and boundary codecs for the Depth Anything V2 provider."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_INT,
    ComboOption,
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
)

IMAGE_TYPE = "dinkster.image"
IMAGE = TypeExpr.concrete(IMAGE_TYPE)
INT = TypeExpr.concrete(CORE_INT)
COMBO = TypeExpr.concrete(CORE_COMBO)
PROVIDER_CHOICE = "dinkster.preprocess.model_depth.providers"


class ModelDepthPreprocessor(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preprocess.model_depth",
            display_name="Preprocess Model Depth",
            category="image/preprocess",
            inputs=(
                InputSpec("image", IMAGE),
                InputSpec(
                    "model",
                    COMBO,
                    required=False,
                    default="auto",
                    widget=ComboWidget(
                        options=(
                            ComboOption("auto", "Automatic"),
                            ComboOption("depth-anything-v3", "Depth Anything V3"),
                            ComboOption(
                                "depth-anything-v2-large",
                                "Depth Anything V2 Large",
                            ),
                        ),
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
                    "resolution",
                    INT,
                    required=False,
                    default=512,
                    widget=NumberWidget(min=64, max=16_384, step=64),
                ),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True, alpha_policy="drop"),),
            search_terms=("depth", "relative depth", "Depth Anything V2", "controlnet"),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: object,
        model: str = "auto",
        provider: str = "",
        resolution: int = 512,
    ) -> Mapping[str, object]:
        del model, provider
        from .model import execute_depth_anything_v2

        return cls.outputs(image=execute_depth_anything_v2(image, resolution=resolution))


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


DEPTH_ANYTHING_V2_PROVIDER_NODES: tuple[type[Node], ...] = (ModelDepthPreprocessor,)


__all__ = [
    "DEPTH_ANYTHING_V2_PROVIDER_NODES",
    "ModelDepthPreprocessor",
    "register_types",
]
