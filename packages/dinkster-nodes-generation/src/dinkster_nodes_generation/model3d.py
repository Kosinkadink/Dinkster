"""Generic geometry-estimation and mesh-processing schemas."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    AssetWidget,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
)

ASSET = TypeExpr.concrete("dinkster.asset")
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
COMBO = TypeExpr.concrete(CORE_COMBO)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
INT = TypeExpr.concrete(CORE_INT)
STRING = TypeExpr.concrete(CORE_STRING)
IMAGE = TypeExpr.concrete("dinkster.image")
MASK = TypeExpr.concrete("dinkster.mask")
MODEL3D = TypeExpr.concrete("dinkster.model3d")
MOGE_MODEL = TypeExpr.concrete("comfy.MOGE_MODEL")
MOGE_GEOMETRY = TypeExpr.concrete("comfy.MOGE_GEOMETRY")
BACKGROUND_REMOVAL = TypeExpr.concrete("comfy.BACKGROUND_REMOVAL")
VOXEL = TypeExpr.concrete("comfy.VOXEL")
MESH = TypeExpr.concrete("comfy.MESH")


class _SchemaOnlyNode(Node):
    @classmethod
    def execute(cls, **_inputs: object) -> Mapping[str, object]:
        raise RuntimeError(f"{cls.schema().node_type} requires an execution provider")


def _model_asset(kind: str) -> AssetWidget:
    return AssetWidget(accept=("application/octet-stream",), kind=kind)


class LoadGeometryModel(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_geometry_model",
            display_name="Load Geometry Model",
            category="model/loaders",
            inputs=(
                InputSpec(
                    "model",
                    ASSET,
                    widget=_model_asset("model/geometry-estimation"),
                ),
            ),
            outputs=(OutputSpec("model", MOGE_MODEL),),
            search_terms=("MoGe", "depth", "geometry estimation"),
        )


class EstimateGeometry(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.estimate_geometry",
            display_name="Estimate Geometry",
            category="image/geometry estimation",
            inputs=(
                InputSpec("model", MOGE_MODEL),
                InputSpec("image", IMAGE),
                InputSpec(
                    "resolution_level",
                    INT,
                    default=9,
                    widget=NumberWidget(min=0, max=9),
                ),
                InputSpec(
                    "fov_x_degrees",
                    FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=170.0, step=0.1),
                ),
                InputSpec(
                    "batch_size",
                    INT,
                    default=4,
                    widget=NumberWidget(min=1, max=64),
                ),
                InputSpec("force_projection", BOOLEAN, default=True),
                InputSpec("apply_mask", BOOLEAN, default=True),
            ),
            outputs=(OutputSpec("geometry", MOGE_GEOMETRY),),
            search_terms=("MoGe", "depth", "camera", "intrinsics"),
        )


class GeometryToFOV(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.geometry_to_fov",
            display_name="Geometry to Field of View",
            category="image/geometry estimation",
            inputs=(
                InputSpec("geometry", MOGE_GEOMETRY),
                InputSpec(
                    "axis",
                    COMBO,
                    default="vertical",
                    widget=ComboWidget(options=("vertical", "horizontal", "diagonal")),
                ),
                InputSpec(
                    "unit",
                    COMBO,
                    default="degrees",
                    widget=ComboWidget(options=("degrees", "radians")),
                ),
            ),
            outputs=(OutputSpec("fov", FLOAT), OutputSpec("focal_pixels", FLOAT)),
            search_terms=("camera", "field of view", "focal length", "MoGe"),
        )


class LoadBackgroundRemoval(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_background_removal",
            display_name="Load Background Removal Model",
            category="model/loaders",
            inputs=(
                InputSpec(
                    "model",
                    ASSET,
                    widget=_model_asset("model/background-removal"),
                ),
            ),
            outputs=(OutputSpec("model", BACKGROUND_REMOVAL),),
            search_terms=("BiRefNet", "background", "mask"),
        )


class RemoveBackground(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.remove_background",
            display_name="Remove Background",
            category="image/background removal",
            inputs=(InputSpec("model", BACKGROUND_REMOVAL), InputSpec("image", IMAGE)),
            outputs=(OutputSpec("mask", MASK),),
            search_terms=("BiRefNet", "background", "foreground mask"),
        )


class ImageCropToMask(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.image_crop_to_mask",
            display_name="Crop Image to Mask",
            category="image/transform",
            inputs=(
                InputSpec("images", IMAGE),
                InputSpec("masks", MASK),
                InputSpec("width", INT, default=1024, widget=NumberWidget(min=64, max=4096)),
                InputSpec("height", INT, default=1024, widget=NumberWidget(min=64, max=4096)),
                InputSpec(
                    "pad_factor",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=1.0, max=2.0, step=0.01),
                ),
                InputSpec(
                    "grow_mask",
                    INT,
                    default=0,
                    widget=NumberWidget(min=-32, max=32),
                ),
                InputSpec(
                    "background",
                    STRING,
                    default="#000000",
                    widget=StringWidget(multiline=False),
                ),
            ),
            outputs=(OutputSpec("images", IMAGE),),
            search_terms=("crop to mask", "subject crop", "3d input"),
        )


class PreviewMask(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preview_mask",
            display_name="Preview Mask",
            category="image/mask",
            inputs=(InputSpec("mask", MASK),),
            outputs=(OutputSpec("mask", MASK, preview=True),),
            output_node=True,
            search_terms=("mask", "preview", "inspect"),
        )


class VoxelToMesh(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.voxel_to_mesh",
            display_name="Voxel to Mesh",
            category="3d/mesh",
            inputs=(
                InputSpec("voxel", VOXEL),
                InputSpec(
                    "algorithm",
                    COMBO,
                    default="surface net",
                    widget=ComboWidget(options=("surface net", "basic")),
                ),
                InputSpec(
                    "threshold",
                    FLOAT,
                    default=0.6,
                    widget=NumberWidget(min=-1.0, max=1.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("mesh", MESH),),
            search_terms=("voxel", "mesh", "surface net"),
        )


class GetMeshInfo(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.get_mesh_info",
            display_name="Get Mesh Info",
            category="3d/mesh",
            inputs=(InputSpec("mesh", MESH),),
            outputs=(OutputSpec("mesh", MESH), OutputSpec("info", STRING)),
        )


class RemeshMesh(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.remesh_mesh",
            display_name="Remesh Mesh",
            category="3d/mesh",
            inputs=(
                InputSpec("mesh", MESH),
                InputSpec("resolution", INT, default=512, widget=NumberWidget(min=32, max=2048)),
                InputSpec(
                    "sign_mode",
                    COMBO,
                    default="udf",
                    widget=ComboWidget(options=("udf", "sdf")),
                ),
                InputSpec("qef", BOOLEAN, default=False),
                InputSpec("drop_inverted_components", BOOLEAN, default=False),
                InputSpec("drop_enclosed_components", BOOLEAN, default=False),
                InputSpec("manifold", BOOLEAN, default=False),
                InputSpec("band", FLOAT, default=1.0, widget=NumberWidget(min=0.5, max=4.0)),
                InputSpec(
                    "project_back",
                    FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0),
                ),
                InputSpec("fix_poles", BOOLEAN, default=False),
                InputSpec("smooth_iters", INT, default=0, widget=NumberWidget(min=0, max=20)),
                InputSpec(
                    "drop_small_components",
                    FLOAT,
                    default=0.01,
                    widget=NumberWidget(min=0.0, max=0.5, step=0.005),
                ),
                InputSpec(
                    "precluster_max_verts",
                    INT,
                    default=20_000_000,
                    widget=NumberWidget(min=0, max=100_000_000),
                ),
            ),
            outputs=(OutputSpec("mesh", MESH),),
        )


class DecimateMesh(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.decimate_mesh",
            display_name="Decimate Mesh",
            category="3d/mesh",
            inputs=(
                InputSpec("mesh", MESH),
                InputSpec(
                    "target_face_count",
                    INT,
                    default=200_000,
                    widget=NumberWidget(min=0, max=50_000_000),
                ),
                InputSpec(
                    "placement_mode",
                    COMBO,
                    default="midpoint",
                    widget=ComboWidget(options=("midpoint", "qem")),
                ),
            ),
            outputs=(OutputSpec("mesh", MESH),),
        )


class SmoothMeshNormals(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.smooth_mesh_normals",
            display_name="Smooth Mesh Normals",
            category="3d/mesh",
            inputs=(
                InputSpec("mesh", MESH),
                InputSpec(
                    "crease_angle",
                    FLOAT,
                    default=180.0,
                    widget=NumberWidget(min=0.0, max=180.0),
                ),
            ),
            outputs=(OutputSpec("mesh", MESH),),
        )


class UnwrapMesh(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.unwrap_mesh",
            display_name="Unwrap Mesh UVs",
            category="3d/texturing",
            inputs=(
                InputSpec("mesh", MESH),
                InputSpec(
                    "segmenter",
                    COMBO,
                    default="pec",
                    widget=ComboWidget(options=("pec", "adaptive")),
                ),
                InputSpec("resolution", INT, default=1024, widget=NumberWidget(min=0, max=8192)),
                InputSpec("padding", INT, default=1, widget=NumberWidget(min=0, max=16)),
                InputSpec(
                    "weld_distance",
                    FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.0001),
                ),
            ),
            outputs=(OutputSpec("mesh", MESH),),
        )


class PaintMesh(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.paint_mesh",
            display_name="Paint Mesh",
            category="3d/texturing",
            inputs=(InputSpec("mesh", MESH), InputSpec("voxel_colors", VOXEL)),
            outputs=(OutputSpec("mesh", MESH),),
        )


class BakeTextureFromVoxel(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.bake_texture_from_voxel",
            display_name="Bake Texture from Voxel",
            category="3d/texturing",
            inputs=(
                InputSpec("mesh", MESH),
                InputSpec("voxel_colors", VOXEL),
                InputSpec(
                    "texture_size",
                    INT,
                    default=2048,
                    widget=NumberWidget(min=64, max=8192),
                ),
                InputSpec("reference_mesh", MESH, required=False),
            ),
            outputs=(
                OutputSpec("base_color", IMAGE),
                OutputSpec("metallic", IMAGE),
                OutputSpec("roughness", IMAGE),
            ),
        )


class BakeNormalMapFromMesh(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.bake_normal_map_from_mesh",
            display_name="Bake Normal Map from Mesh",
            category="3d/texturing",
            inputs=(
                InputSpec("low_poly", MESH),
                InputSpec("high_poly", MESH),
                InputSpec("resolution", INT, default=1024, widget=NumberWidget(min=64, max=8192)),
                InputSpec(
                    "cage_distance",
                    FLOAT,
                    default=0.05,
                    widget=NumberWidget(min=0.001, max=0.5, step=0.001),
                ),
                InputSpec("ignore_backfaces", BOOLEAN, default=True),
            ),
            outputs=(OutputSpec("normal_map", IMAGE),),
        )


class BakeAmbientOcclusion(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.bake_ambient_occlusion",
            display_name="Bake Ambient Occlusion",
            category="3d/texturing",
            inputs=(
                InputSpec("low_poly", MESH),
                InputSpec("high_poly", MESH),
                InputSpec("resolution", INT, default=1024, widget=NumberWidget(min=64, max=8192)),
                InputSpec("samples", INT, default=64, widget=NumberWidget(min=4, max=1024)),
                InputSpec(
                    "max_distance",
                    FLOAT,
                    default=0.5,
                    widget=NumberWidget(min=0.01, max=2.0, step=0.01),
                ),
                InputSpec(
                    "strength",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=2.0, step=0.05),
                ),
                InputSpec(
                    "bias",
                    FLOAT,
                    default=0.01,
                    widget=NumberWidget(min=0.0001, max=0.2, step=0.0005),
                ),
            ),
            outputs=(OutputSpec("occlusion", IMAGE),),
        )


class RenderUVAtlas(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.render_uv_atlas",
            display_name="Render UV Atlas",
            category="3d/texturing",
            inputs=(
                InputSpec("mesh", MESH),
                InputSpec("resolution", INT, default=1024, widget=NumberWidget(min=64, max=4096)),
            ),
            outputs=(OutputSpec("image", IMAGE),),
        )


class ApplyTextureToMesh(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.apply_texture_to_mesh",
            display_name="Apply Texture to Mesh",
            category="3d/texturing",
            inputs=(
                InputSpec("mesh", MESH),
                InputSpec("base_color", IMAGE),
                InputSpec("metallic", IMAGE, required=False),
                InputSpec("roughness", IMAGE, required=False),
                InputSpec("occlusion", IMAGE, required=False),
                InputSpec("normal_map", IMAGE, required=False),
            ),
            outputs=(OutputSpec("mesh", MESH),),
        )


class MeshToModel3D(_SchemaOnlyNode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.mesh_to_model3d",
            display_name="Mesh to 3D Model",
            category="3d",
            inputs=(InputSpec("mesh", MESH),),
            outputs=(OutputSpec("model", MODEL3D, preview=True),),
            search_terms=("mesh", "GLB", "glTF", "3d model"),
        )


MODEL3D_GENERATION_NODES: tuple[type[Node], ...] = (
    LoadGeometryModel,
    EstimateGeometry,
    GeometryToFOV,
    LoadBackgroundRemoval,
    RemoveBackground,
    ImageCropToMask,
    PreviewMask,
    VoxelToMesh,
    GetMeshInfo,
    RemeshMesh,
    DecimateMesh,
    SmoothMeshNormals,
    UnwrapMesh,
    PaintMesh,
    BakeTextureFromVoxel,
    BakeNormalMapFromMesh,
    BakeAmbientOcclusion,
    RenderUVAtlas,
    ApplyTextureToMesh,
    MeshToModel3D,
)

MODEL3D_GENERATION_NODE_IDS = tuple(node.schema().node_type for node in MODEL3D_GENERATION_NODES)

__all__ = ["MODEL3D_GENERATION_NODE_IDS", "MODEL3D_GENERATION_NODES"]
