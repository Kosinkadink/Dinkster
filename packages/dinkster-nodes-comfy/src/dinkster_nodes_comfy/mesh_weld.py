"""Grid-quantized vertex welding through the public mesh contract."""

from __future__ import annotations

import copy
import importlib
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


def _weld(
    vertices: Any, faces: Any, colors: Any, relative: float, absolute: float
) -> tuple[Any, Any, Any]:
    torch = importlib.import_module("torch")
    if vertices.shape[0] == 0:
        return vertices, faces, colors
    if absolute > 0:
        epsilon = absolute
    else:
        bounds = vertices.max(dim=0)[0] - vertices.min(dim=0)[0]
        epsilon = max(float((torch.norm(bounds) * relative).item()), 1e-12)
    scale = 1.0 / epsilon
    minimum = vertices.min(dim=0)[0]
    quantized = ((vertices - minimum) * scale).round().to(torch.int64)
    extent = ((vertices.max(dim=0)[0] - minimum) * scale).round().to(torch.int64) + 2
    keys = (quantized[:, 0] * extent[1] + quantized[:, 1]) * extent[2] + quantized[:, 2]
    unique, inverse = torch.unique(keys, return_inverse=True)
    count = unique.shape[0]
    if count == vertices.shape[0]:
        return vertices, faces, colors
    counts = vertices.new_zeros(count)
    counts.scatter_add_(0, inverse, vertices.new_ones(vertices.shape[0]))
    divisors = counts.unsqueeze(-1).clamp_min(1.0)
    welded = vertices.new_zeros((count, 3))
    welded.scatter_add_(0, inverse.unsqueeze(-1).expand_as(vertices), vertices)
    welded = welded / divisors
    if colors is not None:
        averaged = colors.new_zeros((count, colors.shape[1]))
        averaged.scatter_add_(0, inverse.unsqueeze(-1).expand_as(colors), colors)
        colors = averaged / divisors.to(colors.dtype)
    remapped = inverse[faces.long()].to(faces.dtype) if faces.numel() else faces
    return welded, remapped, colors


class WeldVertices(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WeldVertices",
            display_name="Weld Vertices",
            category="3d/mesh",
            description=(
                "Merge coincident vertices via L_inf grid quantization. Use when a mesh comes in "
                "unwelded (per-face verts, no shared edges) - pre-pass before FillHoles, "
                "DecimateMesh, or any topology-aware op. Colors averaged per cluster."
            ),
            inputs=(
                InputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),
                InputSpec(
                    "epsilon_rel",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1e-5,
                    widget=NumberWidget(min=0.0, step=1e-6),
                    doc="Weld tolerance (fraction of bbox diagonal). 1e-5 for float dedup; "
                    "1e-3 for visibly-close-but-distinct verts.",
                ),
                InputSpec(
                    "epsilon_abs",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.0,
                    widget=NumberWidget(min=0.0, step=1e-6),
                    doc="Absolute weld tolerance (overrides epsilon_rel when > 0).",
                ),
            ),
            outputs=(OutputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),),
        )

    @classmethod
    def execute(
        cls, *, mesh: object, epsilon_rel: float, epsilon_abs: float
    ) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        source: Any = materialize_mesh(mesh)
        vertices: list[Any] = []
        faces: list[Any] = []
        colors: list[Any] = []
        for index in range(source.vertices.shape[0]):
            nv = (
                int(source.vertex_counts[index])
                if source.vertex_counts is not None
                else source.vertices.shape[1]
            )
            nf = (
                int(source.face_counts[index])
                if source.face_counts is not None
                else source.faces.shape[1]
            )
            color = source.vertex_colors[index, :nv] if source.vertex_colors is not None else None
            v, f, c = _weld(
                source.vertices[index, :nv],
                source.faces[index, :nf],
                color,
                epsilon_rel,
                epsilon_abs,
            )
            vertices.append(v)
            faces.append(f)
            colors.append(c)
        maximum = max(v.shape[0] for v in vertices)
        maximum_faces = max(f.shape[0] for f in faces)
        out: Any = copy.copy(source)
        out.vertices = source.vertices.new_zeros((len(vertices), maximum, 3))
        out.faces = source.faces.new_zeros((len(faces), maximum_faces, 3))
        out.vertex_colors = (
            source.vertex_colors.new_zeros((len(vertices), maximum, source.vertex_colors.shape[-1]))
            if source.vertex_colors is not None
            else None
        )
        for index, (v, f, c) in enumerate(zip(vertices, faces, colors, strict=True)):
            out.vertices[index, : v.shape[0]] = v
            out.faces[index, : f.shape[0]] = f
            if c is not None:
                out.vertex_colors[index, : c.shape[0]] = c
        out.vertex_counts = torch.tensor(
            [v.shape[0] for v in vertices], dtype=torch.int64, device=out.vertices.device
        )
        out.face_counts = torch.tensor(
            [f.shape[0] for f in faces], dtype=torch.int64, device=out.faces.device
        )
        out.uvs = out.normals = out.tangents = None
        return cls.outputs(mesh=out)
