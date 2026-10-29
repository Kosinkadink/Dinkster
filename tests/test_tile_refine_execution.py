"""Engine coverage for tiled-refine fold composition."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import cast

import numpy as np
import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EngineEvent
from dinkster_graph import PORTS_NODE_ID, Graph, GraphNode, Link, RegionNode, RegionOutput
from dinkster_nodes_foundation import StringToCombo
from dinkster_nodes_image import IMAGE_NODES, register_image_types
from dinkster_nodes_image.geometry import IMAGE, INT, REGION
from dinkster_nodes_media_io import register_media_types
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_values import CORE_STRING, TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

STRING = TypeExpr.concrete(CORE_STRING)
WIDTH = 24
HEIGHT = 8


class StubTileRefine(Node):
    calls: list[tuple[str, np.ndarray]] = []

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.tile_refine_stub",
            inputs=(InputSpec("image", IMAGE), InputSpec("phase", STRING)),
            outputs=(OutputSpec("image", IMAGE),),
        )

    @classmethod
    def execute(cls, *, image: object, phase: str) -> Mapping[str, object]:
        array = np.asarray(image, dtype=np.float32)
        cls.calls.append((phase, np.array(array, copy=True)))
        increment = 0.25 if phase == "redraw" else 0.5
        return cls.outputs(image=np.ascontiguousarray(np.clip(array + increment, 0.0, 1.0)))


def _port(port_id: str) -> Link:
    return Link(PORTS_NODE_ID, port_id)


def _refine_fold(plan_id: str, canvas: Link, phase: str) -> RegionNode:
    body = Graph(
        nodes={
            "kind": GraphNode("dinkster.string_to_combo", {"string": _port("mask_kind")}),
            "mask": GraphNode(
                "dinkster.mask.tile_blend",
                {
                    "width": WIDTH,
                    "height": HEIGHT,
                    "region": _port("region"),
                    "kind": Link("kind", "choice"),
                    "blur": _port("mask_blur"),
                },
            ),
            "crop": GraphNode(
                "dinkster.image.crop",
                {
                    "image": _port("canvas"),
                    "source.region": _port("crop"),
                    "source.mask": Link("mask", "mask"),
                    "padding": 0,
                    "rounding": "expand",
                },
                slot_variants={"source": "region", "outside": "clip"},
            ),
            "crop_info": GraphNode(
                "dinkster.region.info",
                {"region": _port("crop"), "integer_rounding": "expand"},
            ),
            "sample": GraphNode(
                "dinkster.image.resize",
                {
                    "image": Link("crop", "image"),
                    "target.width": _port("sample_width"),
                    "target.height": _port("sample_height"),
                    "interpolation": "nearest-exact",
                },
                slot_variants={
                    "target": "dimensions",
                    "mode": "stretch",
                    "divisibility": "none",
                },
            ),
            "refine": GraphNode(
                StubTileRefine.schema().node_type,
                {"image": Link("sample", "image"), "phase": phase},
            ),
            "restore": GraphNode(
                "dinkster.image.resize",
                {
                    "image": Link("refine", "image"),
                    "target.width": Link("crop_info", "integer_width"),
                    "target.height": Link("crop_info", "integer_height"),
                    "interpolation": "nearest-exact",
                },
                slot_variants={
                    "target": "dimensions",
                    "mode": "stretch",
                    "divisibility": "none",
                },
            ),
            "composite": GraphNode(
                "dinkster.image.composite",
                {
                    "destination": _port("canvas"),
                    "source": Link("restore", "image"),
                    "x": Link("crop_info", "integer_x"),
                    "y": Link("crop_info", "integer_y"),
                    "blend_mode": "normal",
                    "factor": 1.0,
                    "mask": Link("crop", "mask"),
                },
                slot_variants={"source_resize": "none", "mask": "mask"},
            ),
        }
    )
    ports = {
        "region": REGION,
        "crop": REGION,
        "sample_width": INT,
        "sample_height": INT,
        "mask_kind": STRING,
        "mask_blur": INT,
        "canvas": IMAGE,
    }
    return RegionNode(
        kind="fold",
        body=body,
        ports=ports,
        inputs={
            "region": Link(plan_id, "regions"),
            "crop": Link(plan_id, "crops"),
            "sample_width": Link(plan_id, "sample_widths"),
            "sample_height": Link(plan_id, "sample_heights"),
            "mask_kind": Link(plan_id, "mask_kinds"),
            "mask_blur": Link(plan_id, "mask_blurs"),
            "canvas": canvas,
        },
        element_ports=(
            "region",
            "crop",
            "sample_width",
            "sample_height",
            "mask_kind",
            "mask_blur",
        ),
        state_ports=("canvas",),
        outputs={"canvas": RegionOutput(Link("composite", "image"), mode="state")},
    )


def _make_engine(events: list[EngineEvent]) -> Engine:
    nodes = (*IMAGE_NODES, StringToCombo, StubTileRefine)
    registry = TypeRegistry()
    register_core_types(registry)
    register_media_types(registry)
    register_image_types(registry)
    return Engine(
        schemas=build_schemas(nodes),
        registry=registry,
        worker=InProcessWorker(build_node_types(nodes), registry),
        cache=MemoryLRUCache(),
        on_event=events.append,
    )


def test_tiled_refine_plans_execute_as_chained_image_folds() -> None:
    async def scenario() -> None:
        StubTileRefine.calls.clear()
        events: list[EngineEvent] = []
        graph = Graph(
            nodes={
                "canvas": GraphNode(
                    "dinkster.image.generate",
                    {
                        "width": WIDTH,
                        "height": HEIGHT,
                        "batch_size": 1,
                        "channels": "rgb",
                        "color_source.color_a": "#000000",
                    },
                    slot_variants={"color_source": "hex", "operation": "solid"},
                ),
                "redraw_plan": GraphNode(
                    "dinkster.image.tile_refine_plan",
                    {
                        "width": WIDTH,
                        "height": HEIGHT,
                        "tile_width": 8,
                        "tile_height": 8,
                        "phase": "redraw",
                        "mode": "linear",
                        "mask_blur": 0,
                        "tile_padding": 2,
                        "force_uniform_tiles": False,
                    },
                ),
                "redraw": _refine_fold("redraw_plan", Link("canvas", "image"), "redraw"),
                "seam_plan": GraphNode(
                    "dinkster.image.tile_refine_plan",
                    {
                        "width": WIDTH,
                        "height": HEIGHT,
                        "tile_width": 8,
                        "tile_height": 8,
                        "phase": "seam_fix",
                        "seam_fix_mode": "band_pass",
                        "seam_fix_width": 8,
                        "seam_fix_padding": 2,
                        "force_uniform_tiles": False,
                    },
                ),
                "seam": _refine_fold("seam_plan", Link("redraw", "canvas"), "seam_fix"),
            }
        )

        result = await _make_engine(events).run(graph, ["seam"])
        output = cast("np.ndarray", result.outputs["seam"]["canvas"].resolve())

        assert [phase for phase, _ in StubTileRefine.calls] == [
            "redraw",
            "redraw",
            "redraw",
            "seam_fix",
            "seam_fix",
        ]
        first_redraw = StubTileRefine.calls[0][1]
        second_redraw = StubTileRefine.calls[1][1]
        np.testing.assert_array_equal(first_redraw, np.zeros_like(first_redraw))
        np.testing.assert_array_equal(
            second_redraw[0, 0, :, 0],
            np.array([0.25] * 5 + [0.0] * 11, dtype=np.float32),
        )
        first_seam = StubTileRefine.calls[3][1]
        assert first_seam[0, 0, 8, 0] == pytest.approx(0.5)

        expected_row = np.array(
            [
                0.25,
                0.25,
                0.25,
                0.25,
                0.3125,
                0.4375,
                0.5625,
                0.6875,
                0.9375,
                0.5625,
                0.4375,
                0.3125,
                0.3125,
                0.4375,
                0.5625,
                0.6875,
                0.9375,
                0.5625,
                0.4375,
                0.3125,
                0.25,
                0.25,
                0.25,
                0.25,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            output[0, :, :, 0], np.broadcast_to(expected_row, (HEIGHT, WIDTH)), atol=1e-7
        )
        np.testing.assert_array_equal(output[..., 0], output[..., 1])
        np.testing.assert_array_equal(output[..., 1], output[..., 2])
        assert output[0, 0, 7, 0] < 1.0
        assert output[0, 0, 7, 0] > output[0, 0, 4, 0]

        assert [event.node_id for event in events if event.kind == "region_finished"] == [
            "redraw",
            "seam",
        ]
        assert {
            "redraw[0]/refine",
            "redraw[1]/refine",
            "redraw[2]/refine",
            "seam[0]/refine",
            "seam[1]/refine",
        }.issubset(result.executed)

    asyncio.run(scenario())
