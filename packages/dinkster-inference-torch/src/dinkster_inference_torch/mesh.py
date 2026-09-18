"""Mesh inspection and GLB bytes matching ComfyUI 25dfc16f9ac0 nodes_save_3d."""

from __future__ import annotations

import json
import logging
import struct
from io import BytesIO
from typing import Any

import numpy as np
import torch
from dinkster_inference import TriangleMeshBatch
from PIL import Image


def mesh_info(mesh: TriangleMeshBatch[torch.Tensor]) -> str:
    """Describe padded batch counts and the source-visible attributes."""

    def fmt(n: int) -> str:
        text = f"{n:,}"
        if n >= 1_000_000:
            text += f" ({n / 1_000_000:.2f}M)"
        elif n >= 10_000:
            text += f" ({n / 1_000:.1f}K)"
        return text

    batch = mesh.vertices.shape[0]
    if mesh.vertex_counts is not None:
        assert mesh.face_counts is not None
        v_counts = [int(x) for x in mesh.vertex_counts.tolist()]
        f_counts = [int(x) for x in mesh.face_counts.tolist()]
    else:
        v_counts = [int(mesh.vertices.shape[1])] * batch
        f_counts = [int(mesh.faces.shape[1])] * batch
    attrs: list[str] = []
    for name in (
        "uvs",
        "vertex_colors",
        "normals",
        "tangents",
        "texture",
        "metallic_roughness",
        "normal_map",
    ):
        tensor = getattr(mesh, name)
        if tensor is not None:
            if name in ("texture", "metallic_roughness", "normal_map"):
                attrs.append(f"{name} {int(tensor.shape[-3])}\u00d7{int(tensor.shape[-2])}")
            else:
                attrs.append(name)
    lines: list[str] = []
    if batch > 1:
        lines.extend(
            [
                f"Batch:      {batch} meshes",
                f"Vertices:   {fmt(sum(v_counts))} total",
                f"Faces:      {fmt(sum(f_counts))} total",
            ]
        )
        for i in range(batch):
            lines.append(f"  [{i}]  {v_counts[i]:>10,} verts  \u00b7  {f_counts[i]:>10,} faces")
    else:
        lines.extend([f"Vertices:   {fmt(v_counts[0])}", f"Faces:      {fmt(f_counts[0])}"])
    lines.append(f"Attributes: {', '.join(attrs) if attrs else 'none'}")
    info = "\n".join(lines)
    logging.info("[GetMeshInfo]\n%s", info)
    return info


def mesh_item_to_glb_bytes(mesh: TriangleMeshBatch[torch.Tensor], index: int) -> bytes | None:
    """Serialize a batch item, including PBR attributes; empty items return None."""
    vertex_count = mesh.vertices.shape[1]
    face_count = mesh.faces.shape[1]
    if mesh.vertex_counts is not None:
        assert mesh.face_counts is not None
        vertex_count = int(mesh.vertex_counts[index].item())
        face_count = int(mesh.face_counts[index].item())
        if not 0 <= vertex_count <= mesh.vertices.shape[1]:
            raise ValueError(
                "save_glb: vertex count must be in "
                f"[0, {mesh.vertices.shape[1]}], got {vertex_count}"
            )
        if not 0 <= face_count <= mesh.faces.shape[1]:
            raise ValueError(
                f"save_glb: face count must be in [0, {mesh.faces.shape[1]}], got {face_count}"
            )
    vertices = mesh.vertices[index, :vertex_count]
    faces = mesh.faces[index, :face_count]
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return None
    n_verts = vertices.shape[0]
    vertices_np = vertices.cpu().numpy().astype(np.float32)
    faces_signed = faces.cpu().numpy().astype(np.int64)
    fmin, fmax = int(faces_signed.min()), int(faces_signed.max())
    if fmin < 0 or fmax >= n_verts:
        raise ValueError(
            f"save_glb: face index out of range [0, {n_verts}): min={fmin}, max={fmax}"
        )
    faces_np = faces_signed.astype(np.uint32)
    blobs: dict[str, bytes] = {"vertices": vertices_np.tobytes(), "indices": faces_np.tobytes()}
    attributes: list[tuple[str, str, str, int]] = []
    for name, semantic, kind in (
        ("uvs", "TEXCOORD_0", "VEC2"),
        ("vertex_colors", "COLOR_0", "VEC3"),
        ("normals", "NORMAL", "VEC3"),
        ("tangents", "TANGENT", "VEC4"),
    ):
        tensor = getattr(mesh, name)
        if tensor is None:
            blobs[name] = b""
            continue
        # Tangents are always truncated to the real vertex count in the source.
        item = tensor[index]
        if name == "tangents":
            item = item[:n_verts]
        elif mesh.vertex_counts is not None:
            item = item[:vertex_count]
        array = item.cpu().numpy().astype(np.float32)
        widths = {"uvs": (2,), "vertex_colors": (3, 4), "normals": (3,), "tangents": (4,)}
        if name == "tangents" and array.shape != (n_verts, 4):
            raise ValueError(
                f"save_glb: tangents must be (N, 4) with N={n_verts}, got {tuple(array.shape)}"
            )
        if name != "tangents" and (array.ndim != 2 or array.shape[0] != n_verts):
            entries = array.shape[0] if array.ndim else 0
            raise ValueError(
                f"save_glb: {name} has {entries} entries but vertex count is {n_verts}"
            )
        if array.shape[1] not in widths[name]:
            expected_width = " or ".join(str(width) for width in widths[name])
            raise ValueError(
                f"save_glb: {name} must be (N, {expected_width}) with N={n_verts}, "
                f"got {tuple(array.shape)}"
            )
        if name == "vertex_colors":
            array = np.clip(array, 0.0, 1.0)
            kind = "VEC3" if array.shape[1] == 3 else "VEC4"
        blobs[name] = array.tobytes()
        attributes.append((name, semantic, kind, len(array)))

    for name in ("texture", "metallic_roughness", "normal_map", "emissive"):
        tensor = getattr(mesh, name)
        if tensor is None:
            blobs[name] = b""
            continue
        array = (tensor[index].clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
        assert array.ndim == 3 and array.shape[-1] == 3, (
            f"{name} must be (B, H, W, 3), got {tuple(tensor.shape)}"
        )
        buffer = BytesIO()
        Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
        blobs[name] = buffer.getvalue()

    offsets: dict[str, int] = {}
    parts: list[bytes] = []
    offset = 0
    for name, blob in blobs.items():
        offsets[name] = offset
        padded = blob + b"\x00" * (-len(blob) % 4)
        parts.append(padded)
        offset += len(padded)
    buffer_data = b"".join(parts)
    views: list[dict[str, Any]] = [
        {
            "buffer": 0,
            "byteOffset": offsets["vertices"],
            "byteLength": len(blobs["vertices"]),
            "target": 34962,
        },
        {
            "buffer": 0,
            "byteOffset": offsets["indices"],
            "byteLength": len(blobs["indices"]),
            "target": 34963,
        },
    ]
    accessors: list[dict[str, Any]] = [
        {
            "bufferView": 0,
            "byteOffset": 0,
            "componentType": 5126,
            "count": n_verts,
            "type": "VEC3",
            "max": vertices_np.max(axis=0).tolist(),
            "min": vertices_np.min(axis=0).tolist(),
        },
        {
            "bufferView": 1,
            "byteOffset": 0,
            "componentType": 5125,
            "count": faces_np.size,
            "type": "SCALAR",
        },
    ]
    primitive_attributes = {"POSITION": 0}
    for name, semantic, kind, count in attributes:
        views.append(
            {
                "buffer": 0,
                "byteOffset": offsets[name],
                "byteLength": len(blobs[name]),
                "target": 34962,
            }
        )
        primitive_attributes[semantic] = len(accessors)
        accessors.append(
            {
                "bufferView": len(views) - 1,
                "byteOffset": 0,
                "componentType": 5126,
                "count": count,
                "type": kind,
            }
        )
    primitive: dict[str, Any] = {"attributes": primitive_attributes, "indices": 1, "mode": 4}
    images: list[dict[str, Any]] = []
    textures: list[dict[str, int]] = []
    samplers: list[dict[str, int]] = []
    extensions_used: list[str] = []

    def add_texture(name: str) -> int:
        views.append({"buffer": 0, "byteOffset": offsets[name], "byteLength": len(blobs[name])})
        images.append({"bufferView": len(views) - 1, "mimeType": "image/png"})
        if not samplers:
            samplers.append({"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071})
        textures.append({"source": len(images) - 1, "sampler": 0})
        return len(textures) - 1

    has_uv = "TEXCOORD_0" in primitive_attributes
    if mesh.unlit and not blobs["texture"]:
        if (
            blobs["normal_map"]
            or blobs["emissive"]
            or mesh.occlusion_in_mr
            or mesh.material is not None
        ):
            logging.warning(
                "Unlit material ignores normal/occlusion/emissive maps and PBR overrides."
            )
        material: dict[str, Any] = {
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "extensions": {"KHR_materials_unlit": {}},
            "doubleSided": True,
        }
        extensions_used.append("KHR_materials_unlit")
    else:
        pbr: dict[str, Any] = {
            "metallicFactor": 0.0,
            "roughnessFactor": 0.5,
            "baseColorFactor": [0.22, 0.22, 0.22, 1.0],
        }
        if blobs["texture"] and has_uv:
            pbr["baseColorTexture"] = {"index": add_texture("texture"), "texCoord": 0}
        if (blobs["texture"] and has_uv) or "COLOR_0" in primitive_attributes:
            pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
            pbr["roughnessFactor"] = 1.0
        mr_texture_index = 0
        if blobs["metallic_roughness"] and has_uv:
            mr_texture_index = add_texture("metallic_roughness")
            pbr["metallicRoughnessTexture"] = {"index": mr_texture_index, "texCoord": 0}
            pbr["metallicFactor"] = 1.0
            pbr["roughnessFactor"] = 1.0
        mat: dict[str, Any] = mesh.material if isinstance(mesh.material, dict) else {}
        if mat.get("base_color_factor") is not None:
            pbr["baseColorFactor"] = [float(x) for x in mat["base_color_factor"]]
        if mat.get("metallic_factor", -1.0) >= 0.0:
            pbr["metallicFactor"] = float(mat["metallic_factor"])
        if mat.get("roughness_factor", -1.0) >= 0.0:
            pbr["roughnessFactor"] = float(mat["roughness_factor"])
        material = {"pbrMetallicRoughness": pbr, "doubleSided": bool(mat.get("double_sided", True))}
        if mesh.occlusion_in_mr and blobs["metallic_roughness"] and has_uv:
            material["occlusionTexture"] = {
                "index": mr_texture_index,
                "texCoord": 0,
                "strength": float(mat.get("occlusion_strength", 1.0)),
            }
        if blobs["normal_map"] and has_uv:
            material["normalTexture"] = {
                "index": add_texture("normal_map"),
                "texCoord": 0,
                "scale": float(mat.get("normal_scale", 1.0)),
            }
        emissive_factor = [float(x) for x in mat.get("emissive_factor", [0.0, 0.0, 0.0])]
        emissive_strength = float(mat.get("emissive_strength", 1.0))
        has_em_tex = bool(blobs["emissive"]) and has_uv
        if any(c > 0.0 for c in emissive_factor) or has_em_tex:
            if has_em_tex and not any(c > 0.0 for c in emissive_factor):
                emissive_factor = [1.0, 1.0, 1.0]
            material["emissiveFactor"] = [min(1.0, c) for c in emissive_factor]
            if has_em_tex:
                material["emissiveTexture"] = {"index": add_texture("emissive"), "texCoord": 0}
            if emissive_strength != 1.0:
                material.setdefault("extensions", {})["KHR_materials_emissive_strength"] = {
                    "emissiveStrength": emissive_strength
                }
                extensions_used.append("KHR_materials_emissive_strength")
    primitive["material"] = 0
    gltf: dict[str, Any] = {
        # Retain source metadata so serialization remains byte-identical.
        "asset": {"version": "2.0", "generator": "ComfyUI"},
        "buffers": [{"byteLength": len(buffer_data)}],
        "bufferViews": views,
        "accessors": accessors,
        "meshes": [{"primitives": [primitive]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    if images:
        gltf["images"] = images
    if samplers:
        gltf["samplers"] = samplers
    if textures:
        gltf["textures"] = textures
    gltf["materials"] = [material]
    if extensions_used:
        gltf["extensionsUsed"] = extensions_used
    gltf_json = json.dumps(gltf).encode("utf8")
    gltf_json += b" " * (-len(gltf_json) % 4)
    return b"".join(
        [
            struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(gltf_json) + 8 + len(buffer_data)),
            struct.pack("<II", len(gltf_json), 0x4E4F534A),
            gltf_json,
            struct.pack("<II", len(buffer_data), 0x004E4942),
            buffer_data,
        ]
    )
