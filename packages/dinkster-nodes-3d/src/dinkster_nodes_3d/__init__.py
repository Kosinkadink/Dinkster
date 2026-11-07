"""ComfyUI-origin nodes with independent first-party identities."""

from dinkster_api.v1 import (
    ASSET_TYPE,
    CAMERA_INFO_STATE_TYPE,
    COLOR_TYPE,
    LOAD3D_CAMERA_TYPE,
    LOAD3D_MODEL_INFO_TYPE,
    VIDEO_TYPE,
    TypeRegistry,
    inference_nodes,
    register_asset_type,
    register_file3d_types,
    register_load3d_camera_type,
    register_load3d_model_info_type,
    register_load3d_type,
    register_media_types,
    register_mesh_type,
    register_splat_type,
    register_video_value_type,
    register_voxel_type,
    resolver_from_env,
)

from .camera_angle import CameraAngle
from .camera_info import CreateCameraInfo
from .gaussian_splat import GetSplatCount, MergeSplat, TransformSplat
from .load3d import Load3DAdvanced
from .load3d_captures import Load3D
from .mesh import GetMeshInfo, MergeMeshes, RotateMesh
from .mesh_bake import BakeAmbientOcclusion, BakeNormalMapFromMesh
from .mesh_decimate import DecimateMesh
from .mesh_holes import FillHoles
from .mesh_io import Get3DComponents, MeshToFile3D, SaveGLB
from .mesh_normals import MeshSmoothNormals
from .mesh_paint import PaintMesh
from .mesh_remesh import RemeshMesh
from .mesh_render import RenderMesh
from .mesh_texture import ApplyTextureToMesh, MeshTextureToImage
from .mesh_unwrap import UnwrapMesh
from .mesh_weld import WeldVertices
from .preview3d import Preview3D, Preview3DAdvanced, PreviewGaussianSplat, PreviewPointCloud
from .save3d import Save3DAdvanced, SaveGaussianSplat, SavePointCloud
from .splat_files import File3DToSplat, SplatToFile3D
from .splat_mesh import SplatToMesh
from .splat_render import RenderSplat
from .uv_atlas import RenderUVAtlas
from .voxel_mesh import VoxelToMesh, VoxelToMeshBasic
from .voxel_texture import BakeTextureFromVoxel

NODES = (
    ApplyTextureToMesh,
    BakeAmbientOcclusion,
    BakeNormalMapFromMesh,
    BakeTextureFromVoxel,
    CameraAngle,
    CreateCameraInfo,
    DecimateMesh,
    File3DToSplat,
    FillHoles,
    Get3DComponents,
    GetMeshInfo,
    GetSplatCount,
    Load3D,
    Load3DAdvanced,
    MergeMeshes,
    MergeSplat,
    MeshSmoothNormals,
    MeshTextureToImage,
    MeshToFile3D,
    PaintMesh,
    Preview3D,
    Preview3DAdvanced,
    PreviewGaussianSplat,
    PreviewPointCloud,
    RemeshMesh,
    RenderMesh,
    RenderSplat,
    RenderUVAtlas,
    RotateMesh,
    Save3DAdvanced,
    SaveGLB,
    SaveGaussianSplat,
    SavePointCloud,
    SplatToFile3D,
    SplatToMesh,
    TransformSplat,
    UnwrapMesh,
    VoxelToMesh,
    VoxelToMeshBasic,
    WeldVertices,
)


def register_types(registry: TypeRegistry) -> None:
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    register_media_types(registry)
    register_mesh_type(registry)
    register_splat_type(registry)
    register_voxel_type(registry)
    for type_id in (CAMERA_INFO_STATE_TYPE, COLOR_TYPE):
        if type_id not in registry:
            registry.register(type_id)
    if LOAD3D_CAMERA_TYPE not in registry:
        register_load3d_camera_type(registry)
    if LOAD3D_MODEL_INFO_TYPE not in registry:
        register_load3d_model_info_type(registry)
    if VIDEO_TYPE not in registry:
        register_video_value_type(registry, VIDEO_TYPE, resolver_from_env())
    register_file3d_types(registry, resolver_from_env())
    register_load3d_type(registry)


inference_nodes(NODES)
