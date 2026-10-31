"""Import and export mesh geometry through the public FILE_3D asset boundary."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    FILE_3D_GLB_TYPE,
    FILE_3D_GLTF_TYPE,
    FILE_3D_OBJ_TYPE,
    FILE_3D_STL_TYPE,
    FILE_3D_TYPE,
    MESH_TYPE,
    AssetRef,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    TriangleMeshBatch,
    TypeExpr,
    digest_bytes,
    materialize_mesh,
    produced_output_scratch,
)


class MeshToFile3D(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MeshToFile3D",
            display_name="Create 3D File (from Mesh)",
            search_terms=("mesh to glb", "mesh to file", "export mesh"),
            category="3d",
            description=(
                "Serialize a mesh to a GLB File3D object for Save / Preview 3D nodes, "
                "carrying its UVs, colors, normals, texture, normal/occlusion/emissive "
                "maps and material. Supports one item per batch only."
            ),
            inputs=(InputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),),
            outputs=(OutputSpec("model_3d", TypeExpr.concrete(FILE_3D_GLB_TYPE)),),
        )

    @classmethod
    def execute(cls, *, mesh: object) -> Mapping[str, object]:
        value = materialize_mesh(mesh)
        if value.vertices.shape[0] > 1:
            logging.warning(
                "MeshToFile3D supports one item per batch only. Got %d; using first.",
                value.vertices.shape[0],
            )
        serializer = importlib.import_module("dinkster_native.model3d.mesh")
        glb = serializer.mesh_item_to_glb_bytes(value, 0)
        if glb is None:
            raise ValueError("MeshToFile3D: mesh is empty (no vertices/faces).")
        vault = produced_output_scratch()
        if vault is None:
            raise RuntimeError("MeshToFile3D requires a produced-output store")
        digest = digest_bytes(glb)
        with vault.writer(digest) as writer:
            writer.write(glb)
            writer.commit()
        source = AssetRef(digest, "mesh.glb", len(glb), "model/gltf-binary", resolver=vault)
        return cls.outputs(model_3d={"format": "glb", "source": source})


def _sniff_format(data: bytes) -> str:
    if data[:4] == b"glTF":
        return "glb"
    head = data[:512].lstrip()
    if head[:1] == b"{":
        return "gltf"
    if head[:5].lower() == b"solid":
        return "stl"
    return ""


def _merge_primitives(primitives: list[dict[str, Any]]) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    any_uv = any(primitive["uvs"] is not None for primitive in primitives)
    any_color = any(primitive["colors"] is not None for primitive in primitives)
    all_normals = all(primitive["normals"] is not None for primitive in primitives)
    all_tangents = all_normals and all(
        primitive["tangents"] is not None for primitive in primitives
    )
    channels = max(
        (
            primitive["colors"].shape[1]
            for primitive in primitives
            if primitive["colors"] is not None
        ),
        default=3,
    )
    if not all_normals and any(primitive["normals"] is not None for primitive in primitives):
        logging.warning(
            "Get3DComponents: some primitives lack normals; normals dropped "
            "(MeshSmoothNormals can regenerate them)"
        )
    vertices: list[Any] = []
    faces: list[Any] = []
    uvs: list[Any] = []
    colors: list[Any] = []
    normals: list[Any] = []
    tangents: list[Any] = []
    offset = 0
    for primitive in primitives:
        positions = primitive["positions"]
        count = positions.shape[0]
        vertices.append(positions)
        faces.append(primitive["faces"] + offset)
        offset += count
        if any_uv:
            uvs.append(
                primitive["uvs"]
                if primitive["uvs"] is not None
                else np.zeros((count, 2), np.float32)
            )
        if any_color:
            values = (
                primitive["colors"]
                if primitive["colors"] is not None
                else np.ones((count, channels), np.float32)
            )
            if values.shape[1] < channels:
                values = np.concatenate(
                    [values, np.ones((count, channels - values.shape[1]), np.float32)], axis=1
                )
            colors.append(values)
        if all_normals:
            normals.append(primitive["normals"])
        if all_tangents:
            tangents.append(primitive["tangents"])
    return {
        "vertices": np.concatenate(vertices, axis=0),
        "faces": np.concatenate(faces, axis=0),
        "uvs": np.concatenate(uvs, axis=0) if any_uv else None,
        "colors": np.concatenate(colors, axis=0) if any_color else None,
        "normals": np.concatenate(normals, axis=0) if all_normals else None,
        "tangents": np.concatenate(tangents, axis=0) if all_tangents else None,
    }


class Get3DComponents(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="Get3DComponents",
            display_name="Get 3D Components",
            category="3d",
            description=(
                "Parse a 3D model file (GLB, GLTF, OBJ, STL) into an editable MESH for the "
                "mesh-processing nodes (decimate, remesh, UV unwrap, bake, ...). All scene "
                "nodes/primitives are merged into one mesh with their transforms applied; "
                "textures and material factors come from the first material. "
                "Counterpart of MeshToFile3D."
            ),
            inputs=(
                InputSpec(
                    "model_3d",
                    TypeExpr.union(
                        FILE_3D_GLB_TYPE,
                        FILE_3D_GLTF_TYPE,
                        FILE_3D_OBJ_TYPE,
                        FILE_3D_STL_TYPE,
                        FILE_3D_TYPE,
                    ),
                    doc="3D model file from Load 3D or another 3D node. "
                    "FBX/USDZ are not supported - convert to GLB first.",
                ),
            ),
            outputs=(OutputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),),
        )

    @classmethod
    def execute(cls, *, model_3d: Mapping[str, Any]) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        from . import mesh_file_read, mesh_gltf_read

        source = model_3d["source"]
        if not isinstance(source, (AssetRef, LocalAssetRef)):
            raise TypeError("FILE_3D source must be a public asset identity")
        data = source.read_bytes()
        format_id = (model_3d["format"] or _sniff_format(data)).lower()
        if format_id in ("fbx", "usdz"):
            raise ValueError(
                f"Get3DComponents: .{format_id} parsing is not supported; "
                "convert the model to GLB/GLTF first"
            )
        resources = model_3d.get("resources", {})

        def read_resource(name: str) -> bytes:
            ref = resources.get(name)
            if not isinstance(ref, (AssetRef, LocalAssetRef)):
                raise ValueError(f"glTF references resource {name!r} not granted by FILE_3D")
            return ref.read_bytes()

        warned: set[str] = set()

        def warn_once(key: str, message: str) -> None:
            if key not in warned:
                warned.add(key)
                logging.warning("Get3DComponents: %s", message)

        material: dict[str, Any] = {}
        if format_id in ("glb", "gltf"):
            gltf, buffers, primitives = mesh_gltf_read.load_gltf(data, read_resource, warn_once)
            if not primitives:
                raise ValueError("Get3DComponents: no triangle geometry found in the glTF scene")
            indices = [item["material"] for item in primitives if item["material"] is not None]
            if len(set(indices)) > 1:
                warn_once(
                    "multimat",
                    f"{len(set(indices))} materials found; "
                    "keeping textures/factors of the first only",
                )
            material = mesh_gltf_read.extract_material(
                gltf, buffers, read_resource, indices[0] if indices else None, warn_once
            )
        elif format_id == "obj":
            primitives = [mesh_file_read.load_obj(data)]
        elif format_id == "stl":
            primitives = [mesh_file_read.load_stl(data)]
        else:
            raise ValueError(
                f"Get3DComponents: unsupported or unrecognized format {format_id!r} "
                "(supported: glb, gltf, obj, stl)"
            )
        merged = _merge_primitives(primitives)
        count = merged["vertices"].shape[0]
        max_face = int(merged["faces"].max())
        if max_face >= count:
            raise ValueError(
                f"Get3DComponents: face index {max_face} out of range "
                f"for {count} vertices (corrupt file?)"
            )

        def batch(array: Any) -> Any:
            return torch.from_numpy(array)[None] if array is not None else None

        mesh = TriangleMeshBatch(
            vertices=batch(merged["vertices"]),
            faces=batch(merged["faces"]),
            uvs=batch(merged["uvs"]),
            vertex_colors=batch(merged["colors"]),
            normals=batch(merged["normals"]),
            tangents=batch(merged["tangents"]),
            texture=batch(material.get("texture")),
            metallic_roughness=batch(material.get("metallic_roughness")),
            normal_map=batch(material.get("normal_map")),
            emissive=batch(material.get("emissive")),
            unlit=bool(material.get("unlit", False)),
            occlusion_in_mr=bool(material.get("occlusion_in_mr", False)),
            material=material.get("material") or None,
        )
        logging.info(
            "Get3DComponents: %s -> %d vertices, %d faces (%d primitives)",
            format_id,
            count,
            merged["faces"].shape[0],
            len(primitives),
        )
        return cls.outputs(mesh=mesh)
