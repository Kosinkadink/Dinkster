"""Universal TRELLIS.2 and Pixal3D generation schemas."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
)

VISION = TypeExpr.concrete("dinkster.clip-vision")
VAE = TypeExpr.concrete("dinkster.vae")
CONDITIONING = TypeExpr.concrete("dinkster.conditioning")
LATENT = TypeExpr.concrete("dinkster.latent")
IMAGE = TypeExpr.concrete("dinkster.image")
VOXEL = TypeExpr.concrete("comfy.VOXEL")
MESH = TypeExpr.concrete("comfy.MESH")
SHAPE_SUBDIVIDES = TypeExpr.concrete("comfy.SHAPE_SUBDIVIDES")
COMBO = TypeExpr.concrete(CORE_COMBO)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
INT = TypeExpr.concrete(CORE_INT)


class _SchemaOnlyNode(Node):
    @classmethod
    def execute(cls, **_inputs: object) -> Mapping[str, object]:
        raise RuntimeError(f"{cls.schema().node_type} requires an execution provider")


class EmptyTrellis2LatentStructure(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.empty_trellis2_latent_structure",
            display_name="Empty TRELLIS.2 Latent Structure",
            category="model/latent/trellis",
            inputs=(InputSpec("batch_size", INT, default=1, widget=NumberWidget(min=1, max=4096)),),
            outputs=(OutputSpec("latent", LATENT),),
            aliases=("EmptyTrellis2LatentStructure",),
            search_terms=("trellis2", "trellis.2", "pixal3d", "3d latent"),
        )


class Trellis2Conditioning(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.trellis2_conditioning",
            display_name="TRELLIS.2 Conditioning",
            category="model/conditioning/trellis2",
            inputs=(InputSpec("clip_vision_model", VISION), InputSpec("image", IMAGE)),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
            ),
            aliases=("Trellis2Conditioning",),
            search_terms=("trellis2", "trellis.2", "image to 3d"),
        )


class Pixal3DConditioning(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.pixal3d_conditioning",
            display_name="Pixal3D Conditioning",
            category="model/conditioning/trellis2",
            inputs=(
                InputSpec("clip_vision_model", VISION),
                InputSpec("image", IMAGE),
                InputSpec(
                    "camera_angle_x",
                    FLOAT,
                    default=49.13,
                    widget=NumberWidget(min=1.0, max=170.0, step=0.01),
                ),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
            ),
            aliases=("Pixal3DConditioning",),
            search_terms=("pixal3d", "trellis2", "image to 3d"),
        )


class VaeDecodeStructureTrellis2(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.vae_decode_structure_trellis2",
            display_name="Decode TRELLIS.2 Structure",
            category="model/latent/trellis",
            inputs=(
                InputSpec("samples", LATENT),
                InputSpec("vae", VAE),
                InputSpec(
                    "resolution",
                    COMBO,
                    default="32",
                    widget=ComboWidget(options=("32", "64")),
                ),
            ),
            outputs=(OutputSpec("voxel", VOXEL),),
            aliases=("VaeDecodeStructureTrellis2",),
            search_terms=("trellis2", "structure", "voxel"),
        )


class Trellis2ShapeStage(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.trellis2_shape_stage",
            display_name="TRELLIS.2 Shape Stage",
            category="model/conditioning/trellis2",
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("voxel", VOXEL),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
                OutputSpec("latent", LATENT),
            ),
            aliases=("Trellis2ShapeStage",),
            search_terms=("trellis2", "shape stage", "sparse"),
        )


class Trellis2UpsampleStage(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.trellis2_upsample_stage",
            display_name="TRELLIS.2 Upsample Stage",
            category="model/conditioning/trellis2",
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("shape_latent", LATENT),
                InputSpec("vae", VAE),
                InputSpec(
                    "target_resolution",
                    INT,
                    default=1024,
                    widget=NumberWidget(min=1024, max=2048, step=128),
                ),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
                OutputSpec("latent", LATENT),
            ),
            aliases=("Trellis2UpsampleStage",),
            search_terms=("trellis2", "cascade", "upsample", "3d"),
        )


class VaeDecodeShapeTrellis(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.vae_decode_shape_trellis",
            display_name="Decode TRELLIS.2 Shape",
            category="model/latent/trellis",
            inputs=(InputSpec("samples", LATENT), InputSpec("vae", VAE)),
            outputs=(
                OutputSpec("mesh", MESH),
                OutputSpec("shape_subdivides", SHAPE_SUBDIVIDES),
            ),
            aliases=("VaeDecodeShapeTrellis",),
            search_terms=("trellis2", "shape decode", "mesh"),
        )


class Trellis2TextureStage(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.trellis2_texture_stage",
            display_name="TRELLIS.2 Texture Stage",
            category="model/conditioning/trellis2",
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("shape_latent", LATENT),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
                OutputSpec("latent", LATENT),
            ),
            aliases=("Trellis2TextureStage",),
            search_terms=("trellis2", "texture stage", "pbr"),
        )


class VaeDecodeTextureTrellis(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.vae_decode_texture_trellis",
            display_name="Decode TRELLIS.2 Texture",
            category="model/latent/trellis",
            inputs=(
                InputSpec("samples", LATENT),
                InputSpec("vae", VAE),
                InputSpec("shape_subdivides", SHAPE_SUBDIVIDES),
            ),
            outputs=(OutputSpec("voxel_colors", VOXEL),),
            aliases=("VaeDecodeTextureTrellis",),
            search_terms=("trellis2", "texture decode", "pbr voxel"),
        )


TRELLIS2_NODES: tuple[type[Node], ...] = (
    EmptyTrellis2LatentStructure,
    Trellis2Conditioning,
    Pixal3DConditioning,
    VaeDecodeStructureTrellis2,
    Trellis2ShapeStage,
    Trellis2UpsampleStage,
    VaeDecodeShapeTrellis,
    Trellis2TextureStage,
    VaeDecodeTextureTrellis,
)
TRELLIS2_NODE_IDS = tuple(node.schema().node_type for node in TRELLIS2_NODES)

__all__ = ["TRELLIS2_NODE_IDS", "TRELLIS2_NODES"]
