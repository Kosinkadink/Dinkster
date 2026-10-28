"""Textured triangle meshes from perspective or spherical MoGe point maps."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    MESH_TYPE,
    MOGE_GEOMETRY_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)
from dinkster_inference_wire import TriangleMeshBatch


class MoGePointMapToMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MoGePointMapToMesh",
            display_name="Convert MoGe Point Map to Mesh",
            description="Convert a MoGe point map into a 3D mesh.",
            category="image/geometry estimation",
            search_terms=("moge", "mesh", "geometry", "point map"),
            inputs=(
                InputSpec("moge_geometry", TypeExpr.concrete(MOGE_GEOMETRY_TYPE)),
                InputSpec(
                    "batch_index",
                    TypeExpr.concrete(CORE_INT),
                    default=0,
                    widget=NumberWidget(min=0, max=4096),
                    doc=(
                        "Which image of a batched MoGe geometry to mesh. Per-image vertex counts "
                        "differ, so batches can't be stacked into a single MESH."
                    ),
                ),
                InputSpec(
                    "decimation",
                    TypeExpr.concrete(CORE_INT),
                    default=1,
                    widget=NumberWidget(min=1, max=8),
                    doc="Vertex stride; 1 = full resolution.",
                ),
                InputSpec(
                    "discontinuity_threshold",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.04,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                    doc="Drop pixels whose 3x3 depth span exceeds this fraction. 0 = off.",
                ),
                InputSpec(
                    "texture",
                    TypeExpr.concrete(CORE_BOOLEAN),
                    default=True,
                    doc="Carry the source image through as the baseColor texture.",
                ),
            ),
            outputs=(OutputSpec("MESH", TypeExpr.concrete(MESH_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        moge_geometry: Any,
        batch_index: int = 0,
        decimation: int = 1,
        discontinuity_threshold: float = 0.04,
        texture: bool = True,
    ) -> Mapping[str, object]:
        geometry = materialize_latent(moge_geometry)
        if "points" not in geometry:
            raise ValueError("moge_geometry has no points output.")
        points = geometry["points"]
        batch = points.shape[0]
        if batch_index >= batch:
            raise ValueError(
                f"batch_index {batch_index} out of range; moge_geometry has batch size {batch}."
            )
        initialize_inference_process()
        torch = importlib.import_module("torch")
        helpers = importlib.import_module("dinkster_inference.ldm.moge.geometry")
        edge_depth = geometry["depth"][batch_index] if "depth" in geometry else None
        vertices, faces, uvs = helpers.triangulate_grid_mesh(
            points[batch_index],
            decimation=decimation,
            discontinuity_threshold=discontinuity_threshold,
            depth=edge_depth,
        )
        if vertices.shape[0] == 0 or faces.shape[0] == 0:
            raise ValueError(
                "MoGe produced an empty mesh; try discontinuity_threshold=0 or apply_mask=False."
            )
        if "intrinsics" not in geometry:
            vertices = vertices[:, [1, 2, 0]].contiguous()
        else:
            vertices = vertices * torch.tensor([1.0, -1.0, -1.0], dtype=vertices.dtype)
            faces = faces[:, [0, 2, 1]].contiguous()
        image = geometry["image"][batch_index : batch_index + 1] if texture else None
        mesh = TriangleMeshBatch(
            vertices=vertices.unsqueeze(0),
            faces=faces.unsqueeze(0),
            uvs=uvs.unsqueeze(0),
            texture=image,
        )
        return {"MESH": mesh}
