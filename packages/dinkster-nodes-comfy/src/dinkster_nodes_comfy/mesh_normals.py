"""Area-weighted smooth normals and edge-connected crease splitting."""

from __future__ import annotations

import copy
import importlib
import math
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_FLOAT,
    MESH_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_mesh,
)


def _smooth_normals(vertices: Any, faces: Any) -> Any:
    np = importlib.import_module("numpy")
    triangles = vertices[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    minimum = vertices.min(0)
    inverse_tolerance = 1.0 / (max(float((vertices.max(0) - minimum).max()), 1e-9) * 1e-5)
    quantized = np.round((vertices - minimum) * inverse_tolerance).astype(np.int64)
    _, groups = np.unique(quantized, axis=0, return_inverse=True)
    accumulated = np.zeros((int(groups.max()) + 1, 3), dtype=np.float64)
    for corner in range(3):
        np.add.at(accumulated, groups[faces[:, corner]], face_normals)
    normals = accumulated[groups]
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(lengths > 1e-12, lengths, 1.0)
    return normals.astype(np.float32)


def _split_normals(vertices: Any, faces: Any, crease_angle: float) -> tuple[Any, Any, Any, Any]:
    np = importlib.import_module("numpy")
    triangles = vertices[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_unit = face_normals / np.where(lengths > 1e-12, lengths, 1.0)
    threshold = math.cos(math.radians(crease_angle))
    parents = list(range(faces.shape[0]))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    edge_faces: dict[tuple[int, int], list[int]] = {}
    for index, face in enumerate(faces):
        a, b, c = (int(value) for value in face)
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces.setdefault((u, v) if u < v else (v, u), []).append(index)
    for adjacent in edge_faces.values():
        if (
            len(adjacent) == 2
            and float(np.dot(face_unit[adjacent[0]], face_unit[adjacent[1]])) >= threshold
        ):
            left, right = find(adjacent[0]), find(adjacent[1])
            if left != right:
                parents[left] = right

    # Face components, not just pairs of normals, define each duplicated vertex.
    indices: dict[tuple[int, int], int] = {}
    remap_list: list[int] = []
    output_faces = np.empty((faces.shape[0], 3), dtype=np.int64)
    for index, face in enumerate(faces):
        group = find(index)
        for corner in range(3):
            original = int(face[corner])
            key = (original, group)
            selected = indices.get(key)
            if selected is None:
                selected = len(remap_list)
                indices[key] = selected
                remap_list.append(original)
            output_faces[index, corner] = selected
    remap = np.asarray(remap_list, dtype=np.int64)
    normals = np.zeros((remap.shape[0], 3), dtype=np.float64)
    for corner in range(3):
        np.add.at(normals, output_faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(lengths > 1e-12, lengths, 1.0)
    return vertices[remap], output_faces, normals.astype(np.float32), remap


class MeshSmoothNormals(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MeshSmoothNormals",
            display_name="Smooth Mesh Normals",
            category="3d/mesh",
            description=(
                "Compute smooth per-vertex normals and attach them to the mesh. Meshes "
                "without normals are shaded flat (per-face) by glTF viewers; this makes "
                "them shade smoothly. With crease_angle below 180, edges sharper than the "
                "threshold are kept hard by splitting vertices along them."
            ),
            inputs=(
                InputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),
                InputSpec(
                    "crease_angle",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=180.0,
                    widget=NumberWidget(min=0.0, max=180.0, step=1.0),
                    doc="Edges whose dihedral angle exceeds this value (degrees) stay "
                    "hard (vertices are split). 180 = fully smooth; lower "
                    "preserves sharp edges (e.g. ~30-60 for hard-surface).",
                ),
            ),
            outputs=(OutputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),),
        )

    @classmethod
    def execute(cls, *, mesh: object, crease_angle: float) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        np = importlib.import_module("numpy")
        source = materialize_mesh(mesh)
        out: Any = copy.copy(source)
        smooth = crease_angle >= 180.0
        if smooth:
            out.normals = torch.zeros_like(source.vertices)
        items: dict[str, list[Any]] = {"vertices": [], "faces": [], "normals": []}
        for name in ("vertex_colors", "uvs", "tangents"):
            if getattr(source, name) is not None:
                items[name] = []
        for index in range(source.vertices.shape[0]):
            nv = (
                int(source.vertex_counts[index].item())
                if source.vertex_counts is not None
                else source.vertices.shape[1]
            )
            nf = (
                int(source.face_counts[index].item())
                if source.face_counts is not None
                else source.faces.shape[1]
            )
            vertices, faces = source.vertices[index, :nv], source.faces[index, :nf]
            if nv == 0 or nf == 0:
                continue
            v = vertices.cpu().numpy().astype(np.float32)
            f = faces.cpu().numpy().astype(np.int64)
            if smooth:
                out.normals[index, :nv] = torch.from_numpy(_smooth_normals(v, f)).to(
                    source.vertices
                )
                continue
            v, f, normals, remap = _split_normals(v, f, float(crease_angle))
            items["vertices"].append(torch.from_numpy(v).to(vertices))
            items["faces"].append(torch.from_numpy(f).to(faces))
            items["normals"].append(torch.from_numpy(normals).to(vertices))
            remap_tensor = torch.from_numpy(remap)
            for name in ("vertex_colors", "uvs", "tangents"):
                if name in items:
                    attribute = getattr(source, name)[index, :nv]
                    items[name].append(attribute[remap_tensor.to(attribute.device)])
        if smooth:
            return cls.outputs(mesh=out)
        if not items["vertices"]:
            return cls.outputs(mesh=source)
        maximum_vertices = max(value.shape[0] for value in items["vertices"])
        maximum_faces = max(value.shape[0] for value in items["faces"])
        for name, values in items.items():
            maximum = maximum_faces if name == "faces" else maximum_vertices
            packed = values[0].new_zeros((len(values), maximum, values[0].shape[1]))
            for index, value in enumerate(values):
                packed[index, : value.shape[0]] = value
            setattr(out, name, packed)
        out.vertex_counts = torch.tensor(
            [v.shape[0] for v in items["vertices"]], device=out.vertices.device, dtype=torch.int64
        )
        out.face_counts = torch.tensor(
            [f.shape[0] for f in items["faces"]], device=out.faces.device, dtype=torch.int64
        )
        return cls.outputs(mesh=out)
