"""Boundary-component hole filling through the public mesh contract."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    MESH_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)

from .mesh_weld import process_mesh_batch, weld_vertices


def _connected_labels(edges: Any, count: int) -> Any:
    torch = importlib.import_module("torch")
    kitchen = getattr(importlib.import_module("dinkster_inference.quant_ops"), "ck", None)
    if kitchen is not None and hasattr(kitchen, "connected_components") and edges.is_cuda:
        cuda = kitchen.list_backends().get("cuda", {})
        if (
            cuda.get("available", False)
            and not cuda.get("disabled", True)
            and "connected_components" in cuda.get("capabilities", ())
        ):
            floor = kitchen.registry.get_constraints(
                "cuda", "connected_components"
            ).min_compute_capability
            if floor is None or torch.cuda.get_device_capability(edges.device) >= floor:
                return kitchen.connected_components(edges, count)
    labels = torch.arange(count, dtype=torch.int32, device=edges.device)
    a, b = edges[:, 0], edges[:, 1]
    while True:
        low = torch.minimum(labels[a], labels[b])
        updated = labels.clone()
        updated.scatter_reduce_(0, a, low, "amin")
        updated.scatter_reduce_(0, b, low, "amin")
        updated = updated[updated]
        if torch.equal(updated, labels):
            return labels.long()
        labels = updated


def _fill_boundaries(
    vertices: Any, faces: Any, colors: Any, perimeter: float, chains: bool, maximum: int
) -> tuple[Any, Any, Any]:
    torch = importlib.import_module("torch")
    size = vertices.shape[0]
    device, dtype = vertices.device, vertices.dtype
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
    sorted_edges, _ = edges.sort(dim=1)
    keys = sorted_edges[:, 0].long() * size + sorted_edges[:, 1].long()
    unique, counts = torch.unique(keys, return_counts=True)
    boundary = unique[counts == 1]
    if boundary.numel() == 0:
        return vertices, faces, colors
    directed = edges[torch.isin(keys, boundary)]
    src, tgt = directed[:, 0].long(), directed[:, 1].long()
    labels = _connected_labels(torch.stack([src, tgt], 1), size)
    components, edge_component = torch.unique(labels[src], return_inverse=True)
    count = components.shape[0]
    edge_count = torch.bincount(edge_component, minlength=count)
    lengths = (vertices[src] - vertices[tgt]).norm(dim=-1)
    perimeters = torch.zeros(count, dtype=dtype, device=device)
    perimeters.scatter_add_(0, edge_component, lengths)
    pairs = torch.unique(
        torch.cat([edge_component.long() * size + src, edge_component.long() * size + tgt])
    )
    pair_vertex, pair_component = pairs % size, pairs // size
    vertex_count = torch.bincount(pair_component, minlength=count)
    centroids = torch.zeros((count, 3), dtype=dtype, device=device)
    centroids.scatter_add_(0, pair_component[:, None].expand(-1, 3), vertices[pair_vertex])
    centroids = centroids / vertex_count.clamp_min(1).to(dtype).unsqueeze(-1)
    cycles = (vertex_count == edge_count) & (vertex_count > 0)
    keep = (vertex_count >= 3) & (vertex_count <= maximum) & (perimeters < perimeter)
    if not chains:
        keep = keep & cycles
    if not keep.any():
        return vertices, faces, colors
    centroid_component = keep & (vertex_count > 8)
    centroid_index = size + centroid_component.long().cumsum(0) - 1
    kept = keep[edge_component]
    component = edge_component[kept]
    kept_src, kept_tgt = src[kept], tgt[kept]
    centroid_edge = centroid_component[component]
    pieces: list[Any] = []
    if centroid_edge.any():
        pieces.append(
            torch.stack(
                [
                    kept_tgt[centroid_edge],
                    kept_src[centroid_edge],
                    centroid_index[component[centroid_edge]],
                ],
                dim=1,
            ).to(faces.dtype)
        )
    vertex_component = keep & (vertex_count <= 8)
    if vertex_component.any():
        apex = labels[components]
        mask = vertex_component[component]
        if mask.any():
            a, b, center = kept_src[mask], kept_tgt[mask], apex[component[mask]]
            non_apex = (a != center) & (b != center)
            pieces.append(
                torch.stack([b[non_apex], a[non_apex], center[non_apex]], dim=1).to(faces.dtype)
            )
    new_faces = (
        torch.cat(pieces, dim=0)
        if pieces
        else torch.empty((0, 3), dtype=faces.dtype, device=device)
    )
    if chains:
        degree = torch.zeros(size, dtype=torch.long, device=device)
        degree.scatter_add_(0, src, torch.ones_like(src))
        degree.scatter_add_(0, tgt, torch.ones_like(tgt))
        endpoints = (degree[pair_vertex] == 1) & centroid_component[pair_component]
        if endpoints.any():
            v, c = pair_vertex[endpoints], pair_component[endpoints]
            order = torch.argsort(c, stable=True)
            v, c = v[order], c[order]
            is_chain = (torch.bincount(c, minlength=count) == 2)[c]
            if is_chain.any():
                v, c = v[is_chain].view(-1, 2), c[is_chain].view(-1, 2)[:, 0]
                closing = torch.stack([v[:, 0], v[:, 1], centroid_index[c]], dim=1).to(faces.dtype)
                new_faces = torch.cat([new_faces, closing], dim=0)
    out_colors = colors
    if colors is not None:
        sums = torch.zeros((count, colors.shape[1]), dtype=colors.dtype, device=device)
        sums.scatter_add_(
            0, pair_component[:, None].expand(-1, colors.shape[1]), colors[pair_vertex]
        )
        average = sums / vertex_count.clamp_min(1).to(colors.dtype).unsqueeze(-1)
        out_colors = torch.cat([colors, average[centroid_component]], dim=0)
    return (
        torch.cat([vertices, centroids[centroid_component]], dim=0),
        torch.cat([faces, new_faces], dim=0),
        out_colors,
    )


class FillHoles(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="FillHoles",
            display_name="Fill Holes",
            category="3d/mesh",
            description="Fills holes up to a max perimeter, preserving existing geometry/UVs. "
            "GPU-vectorised with auto-corrected winding and loop-averaged centroid colors",
            inputs=(
                InputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),
                InputSpec(
                    "max_perimeter",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.03,
                    widget=NumberWidget(min=0.0, step=0.0001),
                    doc="Max hole perimeter to fill. 0 disables.",
                ),
                InputSpec(
                    "weld_epsilon_rel",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1e-5,
                    widget=NumberWidget(min=0.0, step=1e-6),
                    doc="Pre-weld tolerance (fraction of bounding box diagonal); "
                    "boundary detection needs welded vertices. 0 skips.",
                ),
                InputSpec(
                    "max_vertices",
                    TypeExpr.concrete(CORE_INT),
                    default=16,
                    widget=NumberWidget(min=3, max=1024),
                    doc="Cap boundary vertices per cycle; centroid-fan only works for small "
                    "near-planar holes. Keep <=16.",
                ),
                InputSpec(
                    "fill_chains",
                    TypeExpr.concrete(CORE_BOOLEAN),
                    default=False,
                    doc="Also fill open chains (not just cycles). Noisy; OFF matches cumesh.",
                ),
            ),
            outputs=(OutputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        mesh: object,
        max_perimeter: float,
        weld_epsilon_rel: float,
        max_vertices: int,
        fill_chains: bool,
    ) -> Mapping[str, object]:
        def fill(vertices: Any, faces: Any, colors: Any) -> tuple[Any, Any, Any]:
            if max_perimeter <= 0 or faces.numel() == 0:
                return vertices, faces, colors
            if weld_epsilon_rel > 0:
                epsilon = float(weld_epsilon_rel)
                while True:
                    vertices, faces, colors = weld_vertices(vertices, faces, colors, epsilon, 0.0)
                    if vertices.shape[0] / max(faces.shape[0], 1) < 1.0 or epsilon >= 1e-2:
                        break
                    epsilon = min(epsilon * 2.0, 1e-2)
            initialize_inference_process()
            device = importlib.import_module(
                "dinkster_inference.model_management"
            ).get_torch_device()
            return _fill_boundaries(
                vertices.to(device),
                faces.to(device),
                colors.to(device) if colors is not None else None,
                max_perimeter,
                fill_chains,
                max_vertices,
            )

        return cls.outputs(mesh=process_mesh_batch(mesh, fill))
