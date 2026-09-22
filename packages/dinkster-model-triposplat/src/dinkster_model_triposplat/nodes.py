"""TripoSplat model nodes."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import Node
from dinkster_nodes_generation import (
    LoadTripoSplatDecoderSchema,
    LoadTripoSplatVisionEncoderSchema,
    TripoSplatConditioningSchema,
    TripoSplatDecodeSchema,
    TripoSplatPreprocessImageSchema,
)


class LoadTripoSplatVisionEncoder(LoadTripoSplatVisionEncoderSchema):
    @classmethod
    def execute(cls, vision_encoder: object) -> Mapping[str, object]:
        from .provider import execute_load_triposplat_vision_encoder

        return execute_load_triposplat_vision_encoder(vision_encoder=vision_encoder)


class LoadTripoSplatDecoder(LoadTripoSplatDecoderSchema):
    @classmethod
    def execute(cls, decoder: object) -> Mapping[str, object]:
        from .provider import execute_load_triposplat_decoder

        return execute_load_triposplat_decoder(decoder=decoder)


class TripoSplatPreprocessImage(TripoSplatPreprocessImageSchema):
    @classmethod
    def execute(
        cls, image: object, mask: object, erode_radius: int = 1, size: int = 1024
    ) -> Mapping[str, object]:
        from .provider import execute_triposplat_preprocess_image

        return execute_triposplat_preprocess_image(
            image=image,
            mask=mask,
            erode_radius=erode_radius,
            size=size,
        )


class TripoSplatConditioning(TripoSplatConditioningSchema):
    @classmethod
    def execute(cls, vision: object, vae: object, image: object) -> Mapping[str, object]:
        from .provider import execute_triposplat_conditioning

        return execute_triposplat_conditioning(vision=vision, vae=vae, image=image)


class TripoSplatDecode(TripoSplatDecodeSchema):
    @classmethod
    def execute(
        cls, samples: object, decoder: object, num_gaussians: int = 262144, seed: int = 0
    ) -> Mapping[str, object]:
        from .provider import execute_triposplat_decode

        return execute_triposplat_decode(
            samples=samples,
            decoder=decoder,
            num_gaussians=num_gaussians,
            seed=seed,
        )


TRIPOSPLAT_MODEL_NODES: tuple[type[Node], ...] = (
    LoadTripoSplatVisionEncoder,
    LoadTripoSplatDecoder,
    TripoSplatPreprocessImage,
    TripoSplatConditioning,
    TripoSplatDecode,
)
TRIPOSPLAT_MODEL_NODE_IDS = tuple(node.schema().node_type for node in TRIPOSPLAT_MODEL_NODES)
TRIPOSPLAT_PACK_NODES: tuple[type[Node], ...] = ()

__all__ = [
    "TRIPOSPLAT_MODEL_NODE_IDS",
    "TRIPOSPLAT_MODEL_NODES",
    "TRIPOSPLAT_PACK_NODES",
]
