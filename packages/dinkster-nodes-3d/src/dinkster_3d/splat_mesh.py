"""Extract colored surfaces from gaussian splats."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import replace

from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_INT,
    MESH_TYPE,
    SPLAT_TYPE,
    InputSpec,
    MappingSource,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    ReplacementCase,
    ReplacementRule,
    TypeExpr,
    initialize_inference_process,
    materialize_splat,
    report_progress,
)


class SplatToMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.3d.splat_to_mesh",
            display_name="Extract Mesh from Splat",
            category="3d/splat",
            description="Extract a coloured mesh from a gaussian splat.",
            search_terms=(
                "SplatToMesh",
                "splat to mesh",
                "gaussian surface nets",
                "splat surface",
                "mesh splat",
            ),
            inputs=(
                InputSpec("splat", TypeExpr.concrete(SPLAT_TYPE)),
                InputSpec(
                    "resolution",
                    TypeExpr.concrete(CORE_INT),
                    default=384,
                    widget=NumberWidget(min=64, max=768, step=16),
                ),
                InputSpec(
                    "kernel",
                    TypeExpr.concrete(CORE_INT),
                    default=5,
                    widget=NumberWidget(min=1, max=8),
                ),
                InputSpec(
                    "smooth",
                    TypeExpr.concrete(CORE_INT),
                    default=0,
                    advanced=True,
                    widget=NumberWidget(min=0, max=60),
                ),
                InputSpec(
                    "level",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.4,
                    widget=NumberWidget(min=0.0, max=2.0, step=0.01),
                ),
                InputSpec(
                    "min_component",
                    TypeExpr.concrete(CORE_INT),
                    default=500,
                    advanced=True,
                    widget=NumberWidget(min=0, max=100000, step=50),
                ),
                InputSpec(
                    "min_opacity",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.02,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "color_sharpen",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=2.0,
                    widget=NumberWidget(min=1.0, max=8.0, step=0.5),
                ),
            ),
            outputs=(OutputSpec("mesh", TypeExpr.concrete(MESH_TYPE), display_name="mesh"),),
        )
        return replace(
            schema,
            replacements=(
                ReplacementRule(
                    from_type="SplatToMesh",
                    cases=(
                        ReplacementCase.build(
                            schema.node_type,
                            inputs={spec.id: MappingSource.copy(spec.id) for spec in schema.inputs},
                            outputs={spec.id: spec.id for spec in schema.outputs},
                        ),
                    ),
                ),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        splat: object,
        resolution: int = 384,
        kernel: int = 5,
        smooth: int = 0,
        level: float = 0.4,
        min_component: int = 500,
        min_opacity: float = 0.02,
        color_sharpen: float = 2.0,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        value = materialize_splat(splat)
        module = importlib.import_module("dinkster_3d.splat_mesh_ops")
        mesh = module.splat_to_mesh(
            value,
            resolution,
            kernel,
            smooth,
            level,
            min_component,
            min_opacity,
            color_sharpen,
            report_progress,
        )
        return cls.outputs(mesh=mesh)
