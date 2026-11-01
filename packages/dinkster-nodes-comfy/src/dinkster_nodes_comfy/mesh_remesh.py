"""Native narrow-band mesh re-extraction through the public mesh contract."""

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
    materialize_mesh,
    report_progress,
)

from .mesh_weld import process_mesh_batch


class RemeshMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="RemeshMesh",
            display_name="Remesh Mesh (Narrow-Band DC)",
            category="3d/mesh",
            description=(
                "Re-extracts a uniformly tessellated mesh via a narrow-band distance field + Dual "
                "Contouring, on the active compute device. Normalizes messy / non-manifold / "
                "self-intersecting topology; run before Decimate Mesh to hit an exact face count. "
                "Output stays welded."
            ),
            inputs=(
                InputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),
                InputSpec(
                    "resolution",
                    TypeExpr.concrete(CORE_INT),
                    default=512,
                    widget=NumberWidget(min=32, max=2048),
                ),
                InputSpec(
                    "band",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=0.5, max=4.0, step=0.1),
                    advanced=True,
                ),
                InputSpec(
                    "project_back",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.05),
                    advanced=True,
                ),
                InputSpec(
                    "fix_poles", TypeExpr.concrete(CORE_BOOLEAN), default=False, advanced=True
                ),
                InputSpec(
                    "smooth_iters",
                    TypeExpr.concrete(CORE_INT),
                    default=0,
                    widget=NumberWidget(min=0, max=20),
                ),
                InputSpec(
                    "drop_small_components",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.01,
                    widget=NumberWidget(min=0.0, max=0.5, step=0.005),
                    advanced=True,
                ),
                InputSpec(
                    "precluster_max_verts",
                    TypeExpr.concrete(CORE_INT),
                    default=20_000_000,
                    widget=NumberWidget(min=0, max=100_000_000),
                    advanced=True,
                ),
            ),
            combos=(
                DynamicComboSpec(
                    "sign_mode",
                    (
                        DynamicComboOption(
                            "udf",
                            (
                                InputSpec(
                                    "qef",
                                    TypeExpr.concrete(CORE_BOOLEAN),
                                    default=False,
                                    advanced=True,
                                ),
                                InputSpec(
                                    "drop_inverted_components",
                                    TypeExpr.concrete(CORE_BOOLEAN),
                                    default=False,
                                    advanced=True,
                                ),
                                InputSpec(
                                    "drop_enclosed_components",
                                    TypeExpr.concrete(CORE_BOOLEAN),
                                    default=False,
                                    advanced=True,
                                ),
                            ),
                        ),
                        DynamicComboOption(
                            "sdf",
                            (
                                InputSpec("qef", TypeExpr.concrete(CORE_BOOLEAN), default=True),
                                InputSpec(
                                    "manifold", TypeExpr.concrete(CORE_BOOLEAN), default=False
                                ),
                            ),
                        ),
                        DynamicComboOption(
                            "solid",
                            (
                                InputSpec(
                                    "fill",
                                    TypeExpr.concrete(CORE_FLOAT),
                                    default=20.0,
                                    widget=NumberWidget(min=0.0, max=1000.0, step=1.0),
                                ),
                                InputSpec(
                                    "qef",
                                    TypeExpr.concrete(CORE_BOOLEAN),
                                    default=False,
                                    advanced=True,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            input_order=(
                "mesh",
                "resolution",
                "sign_mode",
                "band",
                "project_back",
                "fix_poles",
                "smooth_iters",
                "drop_small_components",
                "precluster_max_verts",
            ),
            outputs=(OutputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        mesh: object,
        resolution: int,
        sign_mode: str,
        band: float,
        project_back: float,
        fix_poles: bool,
        smooth_iters: int,
        drop_small_components: float,
        precluster_max_verts: int,
        **inputs: Any,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        from dinkster_native.model3d.mesh_ops.qem_decimate import qem_cluster_decimate

        from .mesh_remesh_ops import remesh_narrow_band_dc

        source: Any = materialize_mesh(mesh)
        face_count = (
            int(source.face_counts.sum())
            if source.face_counts is not None
            else int(source.faces.shape[0] * source.faces.shape[1])
        )
        device = management.get_torch_device()
        management.free_memory(
            max(resolution**3 * 64, face_count * 512) + management.minimum_inference_memory(),
            device,
        )
        counts = [0, 0]

        def remesh(v: Any, f: Any, c: Any) -> tuple[Any, Any, Any]:
            counts[0] += int(f.shape[0])
            original_device = v.device
            vv, ff = v.to(device).float(), f.to(device=device, dtype=torch.int64)
            cc = c.to(device).float() if c is not None else None
            if precluster_max_verts > 0 and vv.shape[0] > precluster_max_verts:
                vv, ff, cc = qem_cluster_decimate(
                    vv, ff, target_verts=precluster_max_verts, colors=cc
                )
            rv, rf, rc = remesh_narrow_band_dc(
                vv,
                ff,
                resolution=resolution,
                band=band,
                project_back=project_back,
                qef=bool(inputs.get("sign_mode.qef", sign_mode == "sdf")),
                sign_mode=sign_mode,
                manifold=bool(inputs.get("sign_mode.manifold", False)),
                fill=float(inputs.get("sign_mode.fill", 20.0)),
                fix_poles=fix_poles,
                smooth_iters=smooth_iters,
                drop_small_components=drop_small_components,
                drop_inverted_components=bool(
                    inputs.get("sign_mode.drop_inverted_components", False)
                ),
                drop_enclosed_components=bool(
                    inputs.get("sign_mode.drop_enclosed_components", False)
                ),
                scale=(resolution + 3.0 * band) / resolution,
                center=torch.zeros(3, dtype=vv.dtype, device=device),
                colors=cc,
            )
            counts[1] += int(rf.shape[0])
            return (
                rv.to(original_device),
                rf.to(device=original_device, dtype=torch.int32),
                rc.to(original_device) if rc is not None else None,
            )

        result = process_mesh_batch(source, remesh)
        percent = f"  ({(counts[1] - counts[0]) / counts[0] * 100:+.0f}%)" if counts[0] else ""
        report_progress(1, 1, text=f"faces: {counts[0]:,} -> {counts[1]:,}{percent}")
        return cls.outputs(mesh=result)
