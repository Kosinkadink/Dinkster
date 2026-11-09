"""Nearest-voxel mesh coloring through the portable geometry contracts."""

from __future__ import annotations

import copy
import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    MESH_TYPE,
    VOXEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TriangleMeshBatch,
    TypeExpr,
    materialize_mesh,
    materialize_voxel,
)


def _paint(mesh: Any, coords: Any, colors: Any, resolution: Any) -> Any:
    torch = importlib.import_module("torch")
    if coords.shape[0] == 0:
        out = copy.copy(mesh)
        out.vertex_colors = mesh.vertices.new_zeros((1, mesh.vertices.shape[1], 3))
        return out
    tree_type = importlib.import_module("scipy.spatial").cKDTree
    origin = torch.tensor([-0.5, -0.5, -0.5], device="cpu")
    positions = coords.to("cpu").float() * (1.0 / resolution) + origin
    colors = colors.to("cpu")
    tree = tree_type(positions.numpy())
    _, nearest = tree.query(mesh.vertices.to("cpu").squeeze(0).numpy(), k=1, workers=-1)
    selected = colors[torch.from_numpy(nearest).long().to(colors.device)]
    if selected.shape[-1] > 3:
        selected = selected[:, :3]
    out = copy.deepcopy(mesh)
    out.vertex_colors = torch.pow(selected.clamp(0, 1), 2.2).unsqueeze(0)
    return out


class PaintMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="PaintMesh",
            display_name="Paint Mesh",
            category="3d/mesh",
            description=(
                "Paints each mesh vertex with its nearest voxel color from the input voxel field."
            ),
            inputs=(
                InputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),
                InputSpec("voxel_colors", TypeExpr.concrete(VOXEL_TYPE)),
            ),
            outputs=(OutputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),),
        )

    @classmethod
    def execute(cls, *, mesh: object, voxel_colors: object) -> Mapping[str, object]:
        mesh = materialize_mesh(mesh)
        voxels = materialize_voxel(voxel_colors)
        coords, colors, resolution = voxels.data, voxels.voxel_colors, voxels.resolution
        batch_size = mesh.vertices.shape[0]
        if coords.shape[0] != 0 and coords.shape[-1] == 4 and batch_size > 1:
            torch = importlib.import_module("torch")
            vertices: list[Any] = []
            faces: list[Any] = []
            painted_colors: list[Any] = []
            for index in range(batch_size):
                vertex_count = (
                    int(mesh.vertex_counts[index].item())
                    if mesh.vertex_counts is not None
                    else mesh.vertices.shape[1]
                )
                face_count = (
                    int(mesh.face_counts[index].item())
                    if mesh.face_counts is not None
                    else mesh.faces.shape[1]
                )
                item = TriangleMeshBatch(
                    mesh.vertices[index : index + 1, :vertex_count],
                    mesh.faces[index : index + 1, :face_count],
                )
                selected = coords[:, 0].long() == index
                painted = _paint(item, coords[selected, 1:], colors[selected], resolution)
                vertices.append(painted.vertices.squeeze(0))
                faces.append(painted.faces.squeeze(0))
                painted_colors.append(painted.vertex_colors.squeeze(0))
            max_vertices = max(value.shape[0] for value in vertices)
            max_faces = max(value.shape[0] for value in faces)
            packed_vertices = vertices[0].new_zeros((batch_size, max_vertices, 3))
            packed_faces = faces[0].new_zeros((batch_size, max_faces, 3))
            packed_colors = painted_colors[0].new_zeros(
                (batch_size, max_vertices, painted_colors[0].shape[1])
            )
            for index, (v, f, c) in enumerate(zip(vertices, faces, painted_colors, strict=True)):
                packed_vertices[index, : v.shape[0]] = v
                packed_faces[index, : f.shape[0]] = f
                packed_colors[index, : c.shape[0]] = c
            return cls.outputs(
                mesh=TriangleMeshBatch(
                    packed_vertices,
                    packed_faces,
                    vertex_colors=packed_colors,
                    vertex_counts=torch.tensor(
                        [v.shape[0] for v in vertices], device=vertices[0].device, dtype=torch.int64
                    ),
                    face_counts=torch.tensor(
                        [f.shape[0] for f in faces], device=faces[0].device, dtype=torch.int64
                    ),
                )
            )
        if coords.shape[0] != 0 and coords.shape[-1] == 4:
            coords = coords[:, 1:]
        return cls.outputs(mesh=_paint(mesh, coords, colors, resolution))
