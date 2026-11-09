"""Native voxel surface extraction and padded triangle mesh batches."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_FLOAT,
    MESH_TYPE,
    VOXEL_TYPE,
    ComboOption,
    ComboWidget,
    Deprecation,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TriangleMeshBatch,
    TypeExpr,
    materialize_voxel,
    report_progress,
)


def _basic(voxels: Any, threshold: float) -> tuple[Any, Any]:
    torch = importlib.import_module("torch")
    device = torch.device("cpu")
    voxels = voxels.to(device)
    binary = (voxels > threshold).float()
    padded = torch.nn.functional.pad(binary, (1, 1, 1, 1, 1, 1), "constant", 0)
    depth, height, width = binary.shape
    neighbors = torch.tensor(
        [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]],
        device=device,
    )
    z, y, x = torch.meshgrid(
        torch.arange(depth, device=device),
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    voxel_indices = torch.stack([z.flatten(), y.flatten(), x.flatten()], dim=1)
    solid_indices = voxel_indices[binary.flatten() > 0]
    corner_offsets = [
        torch.tensor(corners, device=device)
        for corners in (
            [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]],
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]],
            [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
            [[1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0]],
            [[0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 0]],
        )
    ]
    all_vertices: list[Any] = []
    all_indices: list[Any] = []
    vertex_count = 0
    for face_index, offset in enumerate(neighbors):
        padded_indices = solid_indices + offset + 1
        exposed = padded[padded_indices[:, 0], padded_indices[:, 1], padded_indices[:, 2]] == 0
        if not exposed.any():
            continue
        exposed_indices = solid_indices[exposed]
        face_vertices = exposed_indices.unsqueeze(1) + corner_offsets[face_index].unsqueeze(0)
        all_vertices.append(face_vertices.reshape(-1, 3))
        count = exposed_indices.shape[0]
        indices = torch.arange(vertex_count, vertex_count + 4 * count, device=device).reshape(-1, 4)
        all_indices.append(torch.stack([indices[:, 0], indices[:, 1], indices[:, 2]], dim=1))
        all_indices.append(torch.stack([indices[:, 0], indices[:, 2], indices[:, 3]], dim=1))
        vertex_count += 4 * count
    if all_vertices:
        vertices = torch.cat(all_vertices, dim=0)
        faces = torch.cat(all_indices, dim=0)
    else:
        vertices = torch.zeros((1, 3))
        faces = torch.zeros((1, 3))
    maximum = max(voxels.shape)
    vertices = vertices - maximum / 2
    scale = maximum / 2
    if scale > 0:
        vertices = vertices / scale
    return torch.fliplr(vertices), faces


def _surface_net(voxels: Any, threshold: float) -> tuple[Any, Any]:
    torch = importlib.import_module("torch")
    device = torch.device("cpu")
    voxels = voxels.to(device)
    depth, height, width = voxels.shape
    padded = torch.nn.functional.pad(voxels, (1, 1, 1, 1, 1, 1), "constant", 0)
    z, y, x = torch.meshgrid(
        torch.arange(depth, device=device),
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    positions = torch.stack([z.flatten(), y.flatten(), x.flatten()], dim=1)
    corners = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        device=device,
    )
    sample_positions = positions.unsqueeze(1) + corners.unsqueeze(0)
    z_index, y_index, x_index = sample_positions.unbind(-1)
    corner_values = padded[z_index, y_index, x_index]
    signs = corner_values > threshold
    active = torch.any(signs, dim=1) & torch.any(~signs, dim=1)
    active_cells, active_signs, active_values = (
        positions[active],
        signs[active],
        corner_values[active],
    )
    if active_cells.shape[0] == 0:
        return torch.zeros((0, 3), device=device), torch.zeros(
            (0, 3), dtype=torch.long, device=device
        )
    edges = torch.tensor(
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [1, 3],
            [1, 5],
            [2, 3],
            [2, 6],
            [3, 7],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
        ],
        device=device,
    )
    cell_vertices: dict[int, list[Any]] = {}
    for edge_index, (first, second) in enumerate(edges):
        report_progress(edge_index + 1, 100)
        crossing = active_signs[:, first] != active_signs[:, second]
        if not crossing.any():
            continue
        indices = torch.nonzero(crossing, as_tuple=True)[0]
        first_value, second_value = active_values[indices, first], active_values[indices, second]
        fraction = torch.zeros_like(first_value, device=device)
        denominator = second_value - first_value
        valid = denominator != 0
        fraction[valid] = (threshold - first_value[valid]) / denominator[valid]
        fraction[~valid] = 0.5
        start, end = corners[first].float(), corners[second].float()
        intersection = start.unsqueeze(0) + fraction.unsqueeze(1) * (
            end.unsqueeze(0) - start.unsqueeze(0)
        )
        for index, point in zip(indices.tolist(), intersection, strict=True):
            cell_vertices.setdefault(index, []).append(point)
    vertices: list[Any] = []
    lookup: dict[tuple[int, ...], int] = {}
    for index, points in cell_vertices.items():
        if points:
            vertex = torch.stack(points).mean(dim=0) + active_cells[index].float()
            lookup[tuple(active_cells[index].tolist())] = len(vertices)
            vertices.append(vertex)
    if not vertices:
        return torch.zeros((0, 3), device=device), torch.zeros(
            (0, 3), dtype=torch.long, device=device
        )
    final_vertices = torch.stack(vertices)
    inside_count = active_signs.sum(dim=1, keepdim=True).float()
    outside_count = (~active_signs).sum(dim=1, keepdim=True).float()
    inside = torch.zeros((active_cells.shape[0], 3), device=device)
    outside = torch.zeros((active_cells.shape[0], 3), device=device)
    for index in range(8):
        inside += corners[index].float().unsqueeze(0) * active_signs[:, index].unsqueeze(1)
        outside += corners[index].float().unsqueeze(0) * (~active_signs)[:, index].unsqueeze(1)
    gradients = inside / inside_count - outside / outside_count
    directions = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], device=device)
    pairs = [(0, 1), (0, 2), (1, 2)]
    faces: list[Any] = []
    for first, second in pairs:
        direction_first, direction_second = directions[first], directions[second]
        cross = torch.linalg.cross(direction_first.float(), direction_second.float())
        alignments = torch.matmul(gradients, cross)
        neighbors_first = active_cells + direction_first
        neighbors_second = active_cells + direction_second
        diagonals = active_cells + direction_first + direction_second
        for index, cell in enumerate(active_cells):
            keys = [
                tuple(value.tolist())
                for value in (
                    cell,
                    neighbors_first[index],
                    neighbors_second[index],
                    diagonals[index],
                )
            ]
            if not all(key in lookup for key in keys):
                continue
            v0, v1, v2, v3 = (lookup[key] for key in keys)
            triangles = (
                ((v0, v1, v3), (v0, v3, v2))
                if alignments[index] > 0
                else ((v0, v3, v1), (v0, v2, v3))
            )
            faces.extend(torch.tensor(face, device=device, dtype=torch.long) for face in triangles)
    final_faces = (
        torch.stack(faces) if faces else torch.zeros((0, 3), dtype=torch.long, device=device)
    )
    maximum = max(depth, height, width)
    final_vertices = final_vertices - maximum / 2
    if maximum / 2 > 0:
        final_vertices = final_vertices / (maximum / 2)
    return torch.fliplr(final_vertices), final_faces


def _extract(voxel: object, algorithm: str, threshold: float) -> TriangleMeshBatch[Any]:
    torch = importlib.import_module("torch")
    voxel = materialize_voxel(voxel)
    mesh_function = _basic if algorithm == "basic" else _surface_net
    vertices: list[Any] = []
    faces: list[Any] = []
    for grid in voxel.data:
        vertex, face = mesh_function(grid, threshold)
        vertices.append(vertex)
        faces.append(face)
    if (
        vertices
        and all(v.shape == vertices[0].shape for v in vertices)
        and all(f.shape == faces[0].shape for f in faces)
    ):
        return TriangleMeshBatch(torch.stack(vertices), torch.stack(faces))
    batch = len(vertices)
    maximum_vertices, maximum_faces = (
        max(v.shape[0] for v in vertices),
        max(f.shape[0] for f in faces),
    )
    packed_vertices = vertices[0].new_zeros((batch, maximum_vertices, vertices[0].shape[1]))
    packed_faces = faces[0].new_zeros((batch, maximum_faces, faces[0].shape[1]))
    for index, (vertex, face) in enumerate(zip(vertices, faces, strict=True)):
        packed_vertices[index, : vertex.shape[0]] = vertex
        packed_faces[index, : face.shape[0]] = face
    return TriangleMeshBatch(
        packed_vertices,
        packed_faces,
        vertex_counts=torch.tensor(
            [v.shape[0] for v in vertices], device=vertices[0].device, dtype=torch.int64
        ),
        face_counts=torch.tensor(
            [f.shape[0] for f in faces], device=faces[0].device, dtype=torch.int64
        ),
    )


_VOXEL = TypeExpr.concrete(VOXEL_TYPE)
_MESH = TypeExpr.concrete(MESH_TYPE)
_THRESHOLD = InputSpec(
    "threshold",
    TypeExpr.concrete(CORE_FLOAT),
    default=0.6,
    widget=NumberWidget(min=-1.0, max=1.0, step=0.01),
)


class VoxelToMeshBasic(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="VoxelToMeshBasic",
            display_name="Voxel to Mesh (Basic) (DEPRECATED)",
            category="3d",
            description="Converts a voxel grid to a mesh.",
            deprecation=Deprecation(message="Use VoxelToMesh instead."),
            search_visibility="deprecated",
            inputs=(InputSpec("voxel", _VOXEL), _THRESHOLD),
            outputs=(OutputSpec("MESH", _MESH),),
        )

    @classmethod
    def execute(cls, *, voxel: object, threshold: float) -> Mapping[str, object]:
        return cls.outputs(MESH=_extract(voxel, "basic", threshold))


class VoxelToMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="VoxelToMesh",
            display_name="Voxel to Mesh",
            category="3d",
            description="Converts a voxel grid to a mesh.",
            inputs=(
                InputSpec("voxel", _VOXEL),
                InputSpec(
                    "algorithm",
                    TypeExpr.concrete(CORE_COMBO),
                    default="surface net",
                    widget=ComboWidget(options=(ComboOption("surface net"), ComboOption("basic"))),
                ),
                _THRESHOLD,
            ),
            outputs=(OutputSpec("MESH", _MESH),),
        )

    @classmethod
    def execute(cls, *, voxel: object, algorithm: str, threshold: float) -> Mapping[str, object]:
        return cls.outputs(MESH=_extract(voxel, algorithm, threshold))


VOXEL_MESH_NODES = (VoxelToMeshBasic, VoxelToMesh)
