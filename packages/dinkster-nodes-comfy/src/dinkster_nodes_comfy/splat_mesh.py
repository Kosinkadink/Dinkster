"""Native colored surface extraction from counted Gaussian splats."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_INT,
    MESH_TYPE,
    SPLAT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TriangleMeshBatch,
    TypeExpr,
    initialize_inference_process,
    materialize_splat,
    report_progress,
)


class SplatToMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SplatToMesh",
            display_name="Extract Mesh from Splat",
            category="3d/splat",
            description="Extract a coloured mesh from a gaussian splat.",
            search_terms=("splat to mesh", "gaussian surface nets", "splat surface", "mesh splat"),
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
                    widget=NumberWidget(min=1, max=8, step=1),
                ),
                InputSpec(
                    "smooth",
                    TypeExpr.concrete(CORE_INT),
                    default=0,
                    advanced=True,
                    widget=NumberWidget(min=0, max=60, step=1),
                ),
                InputSpec(
                    "level",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.4,
                    widget=NumberWidget(min=0, max=2, step=0.01),
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
                    widget=NumberWidget(min=0, max=1, step=0.01),
                ),
                InputSpec(
                    "color_sharpen",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=2.0,
                    widget=NumberWidget(min=1, max=8, step=0.5),
                ),
            ),
            outputs=(OutputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        splat: object,
        resolution: int,
        kernel: int,
        smooth: int,
        level: float,
        min_component: int,
        min_opacity: float,
        color_sharpen: float,
    ) -> Mapping[str, object]:
        from .splat_mesh_ops import mesh_item

        torch = importlib.import_module("torch")
        initialize_inference_process()
        management = importlib.import_module("dinkster_inference.model_management")
        device = management.get_torch_device()
        source = materialize_splat(splat)
        batch = source.positions.shape[0]
        items: dict[str, list[Any]] = {"vertices": [], "faces": [], "vertex_colors": []}
        for index in range(batch):

            def progress(fraction: float, base: int = index * 1000) -> None:
                report_progress(base + int(min(max(fraction, 0.0), 1.0) * 1000), batch * 1000)

            result = mesh_item(
                source,
                index,
                resolution,
                kernel,
                smooth,
                level,
                min_component,
                min_opacity,
                color_sharpen,
                device,
                progress,
            )
            if result is None:
                logging.warning(
                    "SplatToMesh: splat %d produced no surface; emitting an empty mesh.", index
                )
                result = (
                    torch.zeros((0, 3)),
                    torch.zeros((0, 3), dtype=torch.int64),
                    torch.zeros((0, 3)),
                )
            for name, value in zip(items, result, strict=True):
                items[name].append(value)
            report_progress((index + 1) * 1000, batch * 1000)
        packed: dict[str, Any] = {}
        for name, values in items.items():
            maximum = max(value.shape[0] for value in values)
            output = values[0].new_zeros((batch, maximum, 3))
            for index, value in enumerate(values):
                output[index, : value.shape[0]] = value
            packed[name] = output
        if batch > 1:
            packed["vertex_counts"] = torch.tensor(
                [len(v) for v in items["vertices"]], dtype=torch.int64
            )
            packed["face_counts"] = torch.tensor(
                [len(f) for f in items["faces"]], dtype=torch.int64
            )
        return cls.outputs(mesh=TriangleMeshBatch(**packed, unlit=True))
