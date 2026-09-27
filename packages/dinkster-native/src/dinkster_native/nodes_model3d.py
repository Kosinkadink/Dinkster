"""Native mesh operations retained until dinkster-comfy provides them."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_inference import TriangleMeshBatch
from dinkster_nodes_generation import MODEL3D_GENERATION_NODES
from dinkster_schema import Node, NodeSchema
from dinkster_workers import current_execution_context

_SCHEMAS = {node.schema().node_type: node.schema() for node in MODEL3D_GENERATION_NODES}


def _schema(node_type: str) -> NodeSchema:
    try:
        return _SCHEMAS[node_type]
    except KeyError as error:
        raise RuntimeError(f"unknown model3d provider schema {node_type!r}") from error


def _mesh(value: object) -> TriangleMeshBatch[Any]:
    if type(value) is TriangleMeshBatch:
        return cast("TriangleMeshBatch[Any]", value)
    if not hasattr(value, "vertices") or not hasattr(value, "faces"):
        raise TypeError("mesh operation did not return a triangle mesh")
    source = cast("Any", value)
    return TriangleMeshBatch(
        vertices=source.vertices,
        faces=source.faces,
        uvs=getattr(source, "uvs", None),
        vertex_colors=getattr(source, "vertex_colors", None),
        texture=getattr(source, "texture", None),
        metallic_roughness=getattr(source, "metallic_roughness", None),
        vertex_counts=getattr(source, "vertex_counts", None),
        face_counts=getattr(source, "face_counts", None),
        unlit=bool(getattr(source, "unlit", False)),
        normals=getattr(source, "normals", None),
        tangents=getattr(source, "tangents", None),
        normal_map=getattr(source, "normal_map", None),
        occlusion_in_mr=bool(getattr(source, "occlusion_in_mr", False)),
        material=getattr(source, "material", None),
        emissive=getattr(source, "emissive", None),
    )


def _operation(operation: str, *, unload_models: bool = False, **inputs: object) -> object:
    if unload_models:
        importlib.import_module("dinkster_comfy.model_management").unload_all_models()
    mesh_operations = importlib.import_module("dinkster_native.model3d.mesh_operations")
    return getattr(mesh_operations, operation)(**inputs)


class GenerationImageCropToMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("dinkster.image_crop_to_mask")

    @classmethod
    def execute(
        cls,
        *,
        images: object,
        masks: object,
        width: int = 1024,
        height: int = 1024,
        pad_factor: float = 1.0,
        grow_mask: int = 0,
        background: str = "#000000",
    ) -> Mapping[str, object]:
        crop = importlib.import_module("dinkster_native.model3d.image_crop")
        result = crop.crop_images_to_masks(
            images=images,
            masks=masks,
            width=width,
            height=height,
            pad_factor=pad_factor,
            grow_mask=grow_mask,
            background=background,
        )
        return cls.outputs(images=result)


class GenerationPreviewMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("dinkster.preview_mask")

    @classmethod
    def execute(cls, *, mask: object) -> Mapping[str, object]:
        return cls.outputs(mask=mask)


class GenerationVoxelToMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("dinkster.voxel_to_mesh")

    @classmethod
    def execute(
        cls,
        *,
        voxel: object,
        algorithm: str = "surface net",
        threshold: float = 0.6,
    ) -> Mapping[str, object]:
        result = _operation(
            "voxel_grid_to_mesh",
            voxel=voxel,
            algorithm=algorithm,
            threshold=threshold,
        )
        return cls.outputs(mesh=result)


class GenerationGetMeshInfo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("dinkster.get_mesh_info")

    @classmethod
    def execute(cls, *, mesh: object) -> Mapping[str, object]:
        value = _mesh(mesh)
        mesh_module = importlib.import_module("dinkster_native.model3d.mesh")
        return cls.outputs(mesh=value, info=mesh_module.mesh_info(value))


class GenerationRemeshMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("dinkster.remesh_mesh")

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
        context = current_execution_context()
        result = _operation(
            "remesh_mesh",
            unload_models=True,
            cancelled=(lambda: False) if context is None else context.cancelled,
            mesh=_mesh(mesh),
            resolution=resolution,
            sign_mode=sign_mode,
            qef=qef,
            drop_inverted_components=drop_inverted_components,
            drop_enclosed_components=drop_enclosed_components,
            manifold=manifold,
            band=band,
            project_back=project_back,
            fix_poles=fix_poles,
            smooth_iters=smooth_iters,
            drop_small_components=drop_small_components,
            precluster_max_verts=precluster_max_verts,
        )
        return cls.outputs(mesh=result)


class GenerationDecimateMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("dinkster.decimate_mesh")

    @classmethod
    def execute(
        cls,
        *,
        mesh: object,
        target_face_count: int = 200_000,
        placement_mode: str = "midpoint",
    ) -> Mapping[str, object]:
        result = _operation(
            "decimate_mesh",
            mesh=_mesh(mesh),
            target_face_count=target_face_count,
            placement_mode=placement_mode,
        )
        return cls.outputs(mesh=result)


class GenerationSmoothMeshNormals(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("dinkster.smooth_mesh_normals")

    @classmethod
    def execute(cls, *, mesh: object, crease_angle: float = 180.0) -> Mapping[str, object]:
        result = _operation(
            "smooth_mesh_normals",
            mesh=_mesh(mesh),
            crease_angle=crease_angle,
        )
        return cls.outputs(mesh=result)


class GenerationUnwrapMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("dinkster.unwrap_mesh")

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
        result = _operation(
            "unwrap_mesh",
            unload_models=True,
            mesh=_mesh(mesh),
            segmenter=segmenter,
            resolution=resolution,
            padding=padding,
            weld_distance=weld_distance,
        )
        return cls.outputs(mesh=result)


class GenerationPaintMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("dinkster.paint_mesh")

    @classmethod
    def execute(cls, *, mesh: object, voxel_colors: object) -> Mapping[str, object]:
        result = _operation("paint_mesh", mesh=_mesh(mesh), voxel_colors=voxel_colors)
        return cls.outputs(mesh=result)


class GenerationBakeTextureFromVoxel(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("dinkster.bake_texture_from_voxel")

    @classmethod
    def execute(
        cls,
        *,
        mesh: object,
        voxel_colors: object,
        texture_size: int = 2048,
        reference_mesh: object | None = None,
    ) -> Mapping[str, object]:
        result = _operation(
            "bake_texture_from_voxel",
            unload_models=True,
            mesh=_mesh(mesh),
            voxel_colors=voxel_colors,
            texture_size=texture_size,
            reference_mesh=None if reference_mesh is None else _mesh(reference_mesh),
        )
        base_color, metallic, roughness = cast("tuple[object, object, object]", result)
        return cls.outputs(base_color=base_color, metallic=metallic, roughness=roughness)


class GenerationBakeNormalMapFromMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("dinkster.bake_normal_map_from_mesh")

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
        result = _operation(
            "bake_normal_map_from_mesh",
            low_poly=_mesh(low_poly),
            high_poly=_mesh(high_poly),
            resolution=resolution,
            cage_distance=cage_distance,
            ignore_backfaces=ignore_backfaces,
        )
        return cls.outputs(normal_map=result)


class GenerationBakeAmbientOcclusion(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("dinkster.bake_ambient_occlusion")

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
        result = _operation(
            "bake_ambient_occlusion",
            low_poly=_mesh(low_poly),
            high_poly=_mesh(high_poly),
            resolution=resolution,
            samples=samples,
            max_distance=max_distance,
            strength=strength,
            bias=bias,
        )
        return cls.outputs(occlusion=result)


class GenerationRenderUVAtlas(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("dinkster.render_uv_atlas")

    @classmethod
    def execute(cls, *, mesh: object, resolution: int = 1024) -> Mapping[str, object]:
        result = _operation("render_uv_atlas", mesh=_mesh(mesh), resolution=resolution)
        return cls.outputs(image=result)


class GenerationApplyTextureToMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("dinkster.apply_texture_to_mesh")

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
        result = _operation(
            "apply_texture_to_mesh",
            mesh=_mesh(mesh),
            base_color=base_color,
            metallic=metallic,
            roughness=roughness,
            occlusion=occlusion,
            normal_map=normal_map,
        )
        return cls.outputs(mesh=result)


class GenerationMeshToModel3D(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("dinkster.mesh_to_model3d")

    @classmethod
    def execute(cls, *, mesh: object) -> Mapping[str, object]:
        mesh_module = importlib.import_module("dinkster_native.model3d.mesh")
        glb = mesh_module.mesh_item_to_glb_bytes(_mesh(mesh), 0)
        if glb is None:
            raise ValueError("mesh is empty")
        return cls.outputs(model={"format": "glb", "bytes": glb})
