"""Native mesh simplification using independent quadric-error edge collapses."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    MESH_TYPE,
    DynamicComboOption,
    DynamicComboSpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    report_progress,
)

from .mesh_weld import process_mesh_batch


class DecimateMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="DecimateMesh",
            display_name="Decimate Mesh",
            category="3d/mesh",
            description=(
                "Simplifies a mesh to a target face count using QEM, on the active compute "
                "device. 'midpoint' is the cumesh-faithful preset (best quality, preserves thin "
                "features / hair); 'qem' places vertices at the QEM optimum with line/feature-edge "
                "controls. Output stays welded."
            ),
            inputs=(
                InputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),
                InputSpec(
                    "target_face_count",
                    TypeExpr.concrete(CORE_INT),
                    default=200_000,
                    widget=NumberWidget(min=0, max=50_000_000),
                    doc="Target max faces. 0 disables.",
                ),
            ),
            combos=(
                DynamicComboSpec(
                    "placement_mode",
                    (
                        DynamicComboOption("midpoint", ()),
                        DynamicComboOption(
                            "qem",
                            (
                                InputSpec(
                                    "line_quadric_weight",
                                    TypeExpr.concrete(CORE_FLOAT),
                                    default=0.0,
                                    widget=NumberWidget(min=0.0, max=100.0, step=0.1),
                                    doc="Per-edge line-quadric weight; preserves sharp "
                                    "ridges/valleys. 0 = off.",
                                ),
                                InputSpec(
                                    "feature_edge_quadric_weight",
                                    TypeExpr.concrete(CORE_FLOAT),
                                    default=0.0,
                                    widget=NumberWidget(min=0.0, max=1000.0, step=1.0),
                                    doc="Extra quadric weight on dihedral feature edges "
                                    "(creases). 0 = off.",
                                ),
                                InputSpec(
                                    "feature_edge_min_dihedral_deg",
                                    TypeExpr.concrete(CORE_FLOAT),
                                    default=30.0,
                                    widget=NumberWidget(min=0.0, max=180.0, step=1.0),
                                    doc="Minimum dihedral angle (degree) to count an edge "
                                    "as a feature edge.",
                                ),
                                InputSpec(
                                    "clamp_v_to_edge",
                                    TypeExpr.concrete(CORE_BOOLEAN),
                                    default=True,
                                    doc="Project the QEM-optimal position onto the "
                                    "collapsed edge segment.",
                                ),
                            ),
                        ),
                    ),
                    doc="midpoint: cumesh-faithful (recommended). qem: QEM-optimal placement.",
                ),
            ),
            input_order=("mesh", "target_face_count", "placement_mode"),
            outputs=(OutputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        mesh: object,
        target_face_count: int,
        placement_mode: str,
        **inputs: Any,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        management = importlib.import_module("dinkster_inference.model_management")
        from .mesh_qem import QEMConfig, qem_simplify

        if placement_mode == "qem":
            config = QEMConfig(
                placement_mode="qem",
                line_quadric_weight=float(inputs.get("placement_mode.line_quadric_weight", 0.0)),
                feature_edge_quadric_weight=float(
                    inputs.get("placement_mode.feature_edge_quadric_weight", 0.0)
                ),
                feature_edge_min_dihedral_deg=float(
                    inputs.get("placement_mode.feature_edge_min_dihedral_deg", 30.0)
                ),
                clamp_v_to_edge=bool(inputs.get("placement_mode.clamp_v_to_edge", True)),
            )
        else:
            config = QEMConfig()
        device = management.get_torch_device()
        counts = [0, 0]

        def simplify(vertices: Any, faces: Any, colors: Any) -> tuple[Any, Any, Any]:
            counts[0] += int(faces.shape[0])
            if target_face_count > 0 and faces.shape[0] > target_face_count:
                source_device = vertices.device
                v, f, c = qem_simplify(
                    vertices.to(device),
                    faces.to(device),
                    int(target_face_count),
                    colors=colors.to(device) if colors is not None else None,
                    config=config,
                )
                vertices, faces = v.to(source_device), f.to(source_device)
                if c is not None:
                    colors = c.to(source_device)
            counts[1] += int(faces.shape[0])
            return vertices, faces, colors

        output = process_mesh_batch(mesh, simplify)
        percent = f"  ({(counts[1] - counts[0]) / counts[0] * 100:+.0f}%)" if counts[0] else ""
        report_progress(1, 1, text=f"faces: {counts[0]:,} -> {counts[1]:,}{percent}")
        return cls.outputs(mesh=output)
