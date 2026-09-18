"""TripoSplat model nodes."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_INT,
    AssetWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
)

ASSET = TypeExpr.concrete("dinkster.asset")
VAE = TypeExpr.concrete("dinkster.vae")
CONDITIONING = TypeExpr.concrete("dinkster.conditioning")
LATENT = TypeExpr.concrete("dinkster.latent")
IMAGE = TypeExpr.concrete("dinkster.image")
MASK = TypeExpr.concrete("dinkster.mask")
SPLAT = TypeExpr.concrete("dinkster.splat")
TRIPOSPLAT_VISION = TypeExpr.concrete("dinkster.triposplat_vision")
TRIPOSPLAT_DECODER = TypeExpr.concrete("dinkster.triposplat_decoder")
INT = TypeExpr.concrete(CORE_INT)


class LoadTripoSplatVisionEncoder(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_triposplat_vision_encoder",
            display_name="Load TripoSplat Vision Encoder",
            category="model/loaders/triposplat",
            inputs=(
                InputSpec(
                    "vision_encoder",
                    ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",),
                        kind="model/clip-vision",
                    ),
                ),
            ),
            outputs=(OutputSpec("vision", TRIPOSPLAT_VISION),),
            search_terms=("triposplat", "dinov3", "vision encoder", "3d"),
        )

    @classmethod
    def execute(cls, vision_encoder: object) -> Mapping[str, object]:
        from .provider import execute_load_triposplat_vision_encoder

        return execute_load_triposplat_vision_encoder(vision_encoder=vision_encoder)


class LoadTripoSplatDecoder(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_triposplat_decoder",
            display_name="Load TripoSplat Decoder",
            category="model/loaders/triposplat",
            inputs=(
                InputSpec(
                    "decoder",
                    ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",),
                        kind="model/vae",
                    ),
                ),
            ),
            outputs=(OutputSpec("decoder", TRIPOSPLAT_DECODER),),
            search_terms=("triposplat", "gaussian decoder", "octree", "3d"),
        )

    @classmethod
    def execute(cls, decoder: object) -> Mapping[str, object]:
        from .provider import execute_load_triposplat_decoder

        return execute_load_triposplat_decoder(decoder=decoder)


class TripoSplatPreprocessImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.triposplat_preprocess_image",
            display_name="TripoSplat Preprocess Image",
            category="model/conditioning/triposplat",
            description=(
                "Masks the subject, crops to its square bounding box on a black "
                "background, and resizes for TripoSplat conditioning."
            ),
            inputs=(
                InputSpec("image", IMAGE),
                InputSpec("mask", MASK),
                InputSpec(
                    "erode_radius", INT, default=1, widget=NumberWidget(min=0, max=16, step=1)
                ),
                InputSpec(
                    "size", INT, default=1024, widget=NumberWidget(min=256, max=4096, step=16)
                ),
            ),
            outputs=(OutputSpec("image", IMAGE),),
            search_terms=("triposplat", "preprocess", "crop", "3d"),
        )

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


class TripoSplatConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.triposplat_conditioning",
            display_name="TripoSplat Conditioning",
            category="model/conditioning/triposplat",
            description=(
                "Encodes a preprocessed image into TripoSplat conditioning and an "
                "empty splat latent."
            ),
            inputs=(
                InputSpec("vision", TRIPOSPLAT_VISION),
                InputSpec("vae", VAE),
                InputSpec("image", IMAGE),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
                OutputSpec("latent", LATENT),
            ),
            search_terms=("triposplat", "conditioning", "image to 3d"),
        )

    @classmethod
    def execute(cls, vision: object, vae: object, image: object) -> Mapping[str, object]:
        from .provider import execute_triposplat_conditioning

        return execute_triposplat_conditioning(vision=vision, vae=vae, image=image)


class TripoSplatDecode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.triposplat_decode",
            display_name="TripoSplat Decode",
            category="model/latent/triposplat",
            description="Decodes a sampled TripoSplat latent into a Gaussian splat.",
            inputs=(
                InputSpec("samples", LATENT),
                InputSpec("decoder", TRIPOSPLAT_DECODER),
                InputSpec(
                    "num_gaussians",
                    INT,
                    default=262144,
                    widget=NumberWidget(min=32768, max=1048576, step=32),
                ),
                InputSpec(
                    "seed",
                    INT,
                    default=0,
                    widget=NumberWidget(min=0, control_after_generate="randomize"),
                ),
            ),
            outputs=(OutputSpec("splat", SPLAT, preview=True),),
            search_terms=("triposplat", "decode", "gaussian splat", "3d"),
        )

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

__all__ = ["TRIPOSPLAT_MODEL_NODE_IDS", "TRIPOSPLAT_MODEL_NODES"]
