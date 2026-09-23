"""Parse standard 3D file containers into Dinkster triangle meshes."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from dinkster_inference import TriangleMeshBatch

from . import gltf_read, mesh_file_read


def sniff_mesh_format(data: bytes) -> str:
    if data[:4] == b"glTF":
        return "glb"
    head = data[:512].lstrip()
    if head[:1] == b"{":
        return "gltf"
    if head[:5].lower() == b"solid":
        return "stl"
    return ""


def _merge_primitives(primitives: list[dict[str, object]]) -> dict[str, np.ndarray | None]:
    any_uv = any(primitive["uvs"] is not None for primitive in primitives)
    any_color = any(primitive["colors"] is not None for primitive in primitives)
    all_normals = all(primitive["normals"] is not None for primitive in primitives)
    all_tangents = all_normals and all(
        primitive["tangents"] is not None for primitive in primitives
    )
    color_channels = max(
        (
            primitive["colors"].shape[1]  # type: ignore[union-attr]
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

    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    tangents: list[np.ndarray] = []
    offset = 0
    for primitive in primitives:
        positions = primitive["positions"]
        primitive_faces = primitive["faces"]
        if not isinstance(positions, np.ndarray) or not isinstance(primitive_faces, np.ndarray):
            raise TypeError("mesh primitive positions and faces must be numpy arrays")
        count = positions.shape[0]
        vertices.append(positions)
        faces.append(primitive_faces + offset)
        offset += count
        if any_uv:
            value = primitive["uvs"]
            uvs.append(value if isinstance(value, np.ndarray) else np.zeros((count, 2), np.float32))
        if any_color:
            value = primitive["colors"]
            color = (
                value
                if isinstance(value, np.ndarray)
                else np.ones((count, color_channels), np.float32)
            )
            if color.shape[1] < color_channels:
                color = np.concatenate(
                    [color, np.ones((count, color_channels - color.shape[1]), np.float32)], axis=1
                )
            colors.append(color)
        if all_normals:
            normals.append(primitive["normals"])  # type: ignore[arg-type]
        if all_tangents:
            tangents.append(primitive["tangents"])  # type: ignore[arg-type]

    return {
        "vertices": np.concatenate(vertices, axis=0),
        "faces": np.concatenate(faces, axis=0),
        "uvs": np.concatenate(uvs, axis=0) if any_uv else None,
        "colors": np.concatenate(colors, axis=0) if any_color else None,
        "normals": np.concatenate(normals, axis=0) if all_normals else None,
        "tangents": np.concatenate(tangents, axis=0) if all_tangents else None,
    }


def _tensor(value: np.ndarray | None) -> torch.Tensor | None:
    return torch.from_numpy(value)[None] if value is not None else None


def parse_mesh_file(
    data: bytes,
    format_name: str = "",
    *,
    base_dir: Path | None = None,
) -> TriangleMeshBatch[torch.Tensor]:
    fmt = (format_name or sniff_mesh_format(data)).lower().lstrip(".")
    if fmt in ("fbx", "usdz"):
        raise ValueError(
            f"Get3DComponents: .{fmt} parsing is not supported; convert the model to GLB/GLTF first"
        )

    warned: set[str] = set()

    def warn_once(key: str, message: str) -> None:
        if key not in warned:
            warned.add(key)
            logging.warning("Get3DComponents: %s", message)

    material_info: dict[str, object] | None = None
    if fmt in ("glb", "gltf"):
        gltf, buffers, primitives = gltf_read.load_gltf(
            data, None if base_dir is None else str(base_dir), warn_once
        )
        if not primitives:
            raise ValueError("Get3DComponents: no triangle geometry found in the glTF scene")
        material_indices = [
            primitive["material"] for primitive in primitives if primitive["material"] is not None
        ]
        if len(set(material_indices)) > 1:
            warn_once(
                "multimat",
                f"{len(set(material_indices))} materials found; keeping "
                "textures/factors of the first only",
            )
        first_material = material_indices[0] if material_indices else None
        material_info = gltf_read.extract_material(
            gltf,
            buffers,
            None if base_dir is None else str(base_dir),
            first_material,
            warn_once,
        )
    elif fmt == "obj":
        primitives = [mesh_file_read.load_obj(data)]
    elif fmt == "stl":
        primitives = [mesh_file_read.load_stl(data)]
    else:
        raise ValueError(
            f"Get3DComponents: unsupported or unrecognized format {fmt!r} "
            "(supported: glb, gltf, obj, stl)"
        )

    merged = _merge_primitives(primitives)
    vertices = merged["vertices"]
    faces = merged["faces"]
    if vertices is None or faces is None:
        raise ValueError("Get3DComponents: mesh contains no geometry")
    max_face = int(faces.max())
    if max_face >= vertices.shape[0]:
        raise ValueError(
            f"Get3DComponents: face index {max_face} out of range for "
            f"{vertices.shape[0]} vertices (corrupt file?)"
        )

    material_info = material_info or {}
    return TriangleMeshBatch(
        vertices=torch.from_numpy(vertices)[None],
        faces=torch.from_numpy(faces)[None],
        uvs=_tensor(merged["uvs"]),
        vertex_colors=_tensor(merged["colors"]),
        normals=_tensor(merged["normals"]),
        tangents=_tensor(merged["tangents"]),
        texture=_tensor(material_info.get("texture")),  # type: ignore[arg-type]
        metallic_roughness=_tensor(material_info.get("metallic_roughness")),  # type: ignore[arg-type]
        normal_map=_tensor(material_info.get("normal_map")),  # type: ignore[arg-type]
        emissive=_tensor(material_info.get("emissive")),  # type: ignore[arg-type]
        unlit=bool(material_info.get("unlit", False)),
        occlusion_in_mr=bool(material_info.get("occlusion_in_mr", False)),
        material=material_info.get("material") or None,
    )


__all__ = ["parse_mesh_file", "sniff_mesh_format"]
