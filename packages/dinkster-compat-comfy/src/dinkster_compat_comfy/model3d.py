"""ComfyUI-backed mesh and voxel execution bodies."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any, cast

from dinkster_native.native_arm import (
    GenerationApplyTextureToMesh as _ApplyTextureToMesh,
)
from dinkster_native.native_arm import (
    GenerationBakeAmbientOcclusion as _BakeAmbientOcclusion,
)
from dinkster_native.native_arm import (
    GenerationBakeNormalMapFromMesh as _BakeNormalMapFromMesh,
)
from dinkster_native.native_arm import (
    GenerationBakeTextureFromVoxel as _BakeTextureFromVoxel,
)
from dinkster_native.native_arm import GenerationDecimateMesh as _DecimateMesh
from dinkster_native.native_arm import GenerationPaintMesh as _PaintMesh
from dinkster_native.native_arm import GenerationRemeshMesh as _RemeshMesh
from dinkster_native.native_arm import GenerationRenderUVAtlas as _RenderUVAtlas
from dinkster_native.native_arm import GenerationSmoothMeshNormals as _SmoothMeshNormals
from dinkster_native.native_arm import GenerationUnwrapMesh as _UnwrapMesh
from dinkster_native.native_arm import GenerationVoxelToMesh as _VoxelToMesh
from dinkster_native.native_residency import default_native_residency, select_current_device


def _torch() -> Any:
    return importlib.import_module("torch")


def _triangle_mesh_batch(value: object) -> object:
    inference = importlib.import_module("dinkster_inference")
    if type(value) is inference.TriangleMeshBatch:
        return value
    if any(not hasattr(value, name) for name in ("vertices", "faces")):
        raise TypeError("mesh operation did not return a triangle mesh")
    mesh = cast("Any", value)
    return inference.TriangleMeshBatch(
        vertices=mesh.vertices,
        faces=mesh.faces,
        uvs=getattr(mesh, "uvs", None),
        vertex_colors=getattr(mesh, "vertex_colors", None),
        texture=getattr(mesh, "texture", None),
        metallic_roughness=getattr(mesh, "metallic_roughness", None),
        vertex_counts=getattr(mesh, "vertex_counts", None),
        face_counts=getattr(mesh, "face_counts", None),
        unlit=bool(getattr(mesh, "unlit", False)),
        normals=getattr(mesh, "normals", None),
        tangents=getattr(mesh, "tangents", None),
        normal_map=getattr(mesh, "normal_map", None),
        occlusion_in_mr=bool(getattr(mesh, "occlusion_in_mr", False)),
        material=getattr(mesh, "material", None),
        emissive=getattr(mesh, "emissive", None),
    )


def _node_outputs(module_name: str, class_name: str, **inputs: object) -> tuple[object, ...]:
    node_class = getattr(importlib.import_module(module_name), class_name)
    prepare = getattr(node_class, "PREPARE_CLASS_CLONE", None)
    prepared = prepare({"hidden_inputs": {}}) if callable(prepare) else node_class
    execute = cast("Callable[..., object]", cast("Any", prepared).execute)
    output = execute(**inputs)
    result = getattr(output, "result", None)
    if not isinstance(result, tuple):
        raise TypeError(f"ComfyUI {class_name} did not return a tuple result")
    return cast("tuple[object, ...]", result)


def _model3d_outputs(
    module_name: str,
    class_name: str,
    *,
    offload_models: bool = False,
    **inputs: object,
) -> tuple[object, ...]:
    coordinator = default_native_residency()
    with coordinator.placement_pass():
        if offload_models:
            device = select_current_device(_torch())
            if device.type != "cpu":
                manager = coordinator.manager
                memory = manager.policy_memory(device).free_total
                loaded = sum(
                    mechanism.loaded_bytes()
                    for mechanism in manager.registered()
                    if mechanism.load_device == device
                )
                manager.free(memory + loaded, device)
                manager.empty_cache(device)
        return _node_outputs(module_name, class_name, **inputs)


class GenerationVoxelToMesh(_VoxelToMesh):
    @classmethod
    def execute(
        cls, *, voxel: object, algorithm: str = "surface net", threshold: float = 0.6
    ) -> Mapping[str, object]:
        (mesh,) = _model3d_outputs(
            "comfy_extras.nodes_hunyuan3d",
            "VoxelToMesh",
            voxel=voxel,
            algorithm=algorithm,
            threshold=threshold,
        )
        return cls.outputs(mesh=_triangle_mesh_batch(mesh))


class GenerationRemeshMesh(_RemeshMesh):
    @classmethod
    def execute(
        cls,
        *,
        mesh: object,
        resolution: int = 512,
        sign_mode: str = "udf",
        qef: bool = False,
        drop_inverted_components: bool = False,
        drop_enclosed_components: bool = False,
        manifold: bool = False,
        band: float = 1.0,
        project_back: float = 0.0,
        fix_poles: bool = False,
        smooth_iters: int = 0,
        drop_small_components: float = 0.01,
        precluster_max_verts: int = 20_000_000,
    ) -> Mapping[str, object]:
        sign_options = {
            "sign_mode": sign_mode,
            "qef": qef,
            "drop_inverted_components": drop_inverted_components,
            "drop_enclosed_components": drop_enclosed_components,
            "manifold": manifold,
        }
        (result,) = _model3d_outputs(
            "comfy_extras.nodes_mesh_postprocess",
            "RemeshMesh",
            offload_models=True,
            mesh=mesh,
            resolution=resolution,
            sign_mode=sign_options,
            band=band,
            project_back=project_back,
            fix_poles=fix_poles,
            smooth_iters=smooth_iters,
            drop_small_components=drop_small_components,
            precluster_max_verts=precluster_max_verts,
        )
        return cls.outputs(mesh=_triangle_mesh_batch(result))


class GenerationDecimateMesh(_DecimateMesh):
    @classmethod
    def execute(
        cls, *, mesh: object, target_face_count: int = 200_000, placement_mode: str = "midpoint"
    ) -> Mapping[str, object]:
        (result,) = _model3d_outputs(
            "comfy_extras.nodes_mesh_postprocess",
            "DecimateMesh",
            mesh=mesh,
            target_face_count=target_face_count,
            placement_mode={"placement_mode": placement_mode},
        )
        return cls.outputs(mesh=_triangle_mesh_batch(result))


class GenerationSmoothMeshNormals(_SmoothMeshNormals):
    @classmethod
    def execute(cls, *, mesh: object, crease_angle: float = 180.0) -> Mapping[str, object]:
        (result,) = _model3d_outputs(
            "comfy_extras.nodes_mesh_postprocess",
            "MeshSmoothNormals",
            mesh=mesh,
            crease_angle=crease_angle,
        )
        return cls.outputs(mesh=_triangle_mesh_batch(result))


class GenerationUnwrapMesh(_UnwrapMesh):
    @classmethod
    def execute(
        cls,
        *,
        mesh: object,
        segmenter: str = "pec",
        resolution: int = 1024,
        padding: int = 1,
        weld_distance: float = 0.0,
    ) -> Mapping[str, object]:
        (result,) = _model3d_outputs(
            "comfy_extras.nodes_mesh_postprocess",
            "UnwrapMesh",
            offload_models=True,
            mesh=mesh,
            segmenter=segmenter,
            resolution=resolution,
            padding=padding,
            weld_distance=weld_distance,
        )
        return cls.outputs(mesh=_triangle_mesh_batch(result))


class GenerationPaintMesh(_PaintMesh):
    @classmethod
    def execute(cls, *, mesh: object, voxel_colors: object) -> Mapping[str, object]:
        (result,) = _model3d_outputs(
            "comfy_extras.nodes_mesh_postprocess",
            "PaintMesh",
            mesh=mesh,
            voxel_colors=voxel_colors,
        )
        return cls.outputs(mesh=_triangle_mesh_batch(result))


class GenerationBakeTextureFromVoxel(_BakeTextureFromVoxel):
    @classmethod
    def execute(
        cls,
        *,
        mesh: object,
        voxel_colors: object,
        texture_size: int = 2048,
        reference_mesh: object | None = None,
    ) -> Mapping[str, object]:
        base_color, metallic, roughness = _model3d_outputs(
            "comfy_extras.nodes_mesh_postprocess",
            "BakeTextureFromVoxel",
            mesh=mesh,
            voxel_colors=voxel_colors,
            texture_size=texture_size,
            reference_mesh=reference_mesh,
        )
        return cls.outputs(base_color=base_color, metallic=metallic, roughness=roughness)


class GenerationBakeNormalMapFromMesh(_BakeNormalMapFromMesh):
    @classmethod
    def execute(
        cls,
        *,
        low_poly: object,
        high_poly: object,
        resolution: int = 1024,
        cage_distance: float = 0.05,
        ignore_backfaces: bool = True,
    ) -> Mapping[str, object]:
        (normal_map,) = _model3d_outputs(
            "comfy_extras.nodes_mesh_postprocess",
            "BakeNormalMapFromMesh",
            low_poly=low_poly,
            high_poly=high_poly,
            resolution=resolution,
            cage_distance=cage_distance,
            ignore_backfaces=ignore_backfaces,
        )
        return cls.outputs(normal_map=normal_map)


class GenerationBakeAmbientOcclusion(_BakeAmbientOcclusion):
    @classmethod
    def execute(
        cls,
        *,
        low_poly: object,
        high_poly: object,
        resolution: int = 1024,
        samples: int = 64,
        max_distance: float = 0.5,
        strength: float = 1.0,
        bias: float = 0.01,
    ) -> Mapping[str, object]:
        (occlusion,) = _model3d_outputs(
            "comfy_extras.nodes_mesh_postprocess",
            "BakeAmbientOcclusion",
            low_poly=low_poly,
            high_poly=high_poly,
            resolution=resolution,
            samples=samples,
            max_distance=max_distance,
            strength=strength,
            bias=bias,
        )
        return cls.outputs(occlusion=occlusion)


class GenerationRenderUVAtlas(_RenderUVAtlas):
    @classmethod
    def execute(cls, *, mesh: object, resolution: int = 1024) -> Mapping[str, object]:
        (image,) = _model3d_outputs(
            "comfy_extras.nodes_mesh_postprocess",
            "RenderUVAtlas",
            mesh=mesh,
            resolution=resolution,
        )
        return cls.outputs(image=image)


class GenerationApplyTextureToMesh(_ApplyTextureToMesh):
    @classmethod
    def execute(
        cls,
        *,
        mesh: object,
        base_color: object,
        metallic: object | None = None,
        roughness: object | None = None,
        occlusion: object | None = None,
        normal_map: object | None = None,
    ) -> Mapping[str, object]:
        (result,) = _model3d_outputs(
            "comfy_extras.nodes_mesh_postprocess",
            "ApplyTextureToMesh",
            mesh=mesh,
            base_color=base_color,
            metallic=metallic,
            roughness=roughness,
            occlusion=occlusion,
            normal_map=normal_map,
        )
        return cls.outputs(mesh=_triangle_mesh_batch(result))


COMFY_MODEL3D_NODES = (
    GenerationVoxelToMesh,
    GenerationRemeshMesh,
    GenerationDecimateMesh,
    GenerationSmoothMeshNormals,
    GenerationUnwrapMesh,
    GenerationPaintMesh,
    GenerationBakeTextureFromVoxel,
    GenerationBakeNormalMapFromMesh,
    GenerationBakeAmbientOcclusion,
    GenerationRenderUVAtlas,
    GenerationApplyTextureToMesh,
)
