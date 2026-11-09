"""Canonical curves and histogram events through ordinary native hosts."""

from __future__ import annotations

import asyncio
import copy
import os
from pathlib import Path
from typing import Any

import pytest
from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CURVE_TYPE,
    Curve,
    CurveWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.curve_editor import CurveEditor
from dinkster_schema import use_reporter

from dinkster.compose import PackSpec, ServingComposer


@pytest.mark.parametrize("histogram", (None, [], [7, -3, 2]))
def test_curve_editor_preserves_curve_and_optional_histogram(histogram: list[int] | None) -> None:
    curve = Curve(((1.0, 0.25), (0.0, 0.5), (0.0, 0.75)))
    before = copy.deepcopy(histogram)
    events = []
    with use_reporter(lambda name, data, blob: events.append((name, data, blob))):
        result = CurveEditor.execute(curve=curve, histogram=histogram)
    assert result["curve"] is curve
    assert curve.points == ((0.0, 0.5), (0.0, 0.75), (1.0, 0.25))
    assert curve.interpolation == "monotone_cubic"
    assert histogram == before
    assert events == (
        [] if histogram is None else [("dinkster.curve.histogram", {"histogram": before}, None)]
    )
    if histogram is not None:
        histogram.append(99)
        assert events[0][1]["histogram"] == before


def test_curve_editor_public_schema_uses_canonical_types_and_source_default() -> None:
    schema = CurveEditor.define_schema()
    assert (schema.node_type, schema.display_name, schema.category) == (
        "CurveEditor",
        "Curve Editor",
        "utilities",
    )
    assert schema.editor_role == "curve"
    curve, histogram = schema.inputs
    assert isinstance(curve, InputSpec) and isinstance(histogram, InputSpec)
    assert curve.type == TypeExpr.concrete(CURVE_TYPE)
    assert isinstance(curve.widget, CurveWidget)
    assert curve.default == {
        "points": [[0.0, 0.0], [1.0, 1.0]],
        "interpolation": "monotone_cubic",
    }
    assert not histogram.required
    assert [output.id for output in schema.outputs] == ["curve"]


class CurveConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="curve-fixture.consumer",
            inputs=(InputSpec("curve", TypeExpr.concrete(CURVE_TYPE)),),
            outputs=(OutputSpec("valid", TypeExpr.concrete(CORE_BOOLEAN)),),
        )

    @classmethod
    def execute(cls, *, curve: Curve) -> dict[str, object]:
        assert isinstance(curve, Curve)
        # Hand-derived linear interpolation, stable duplicate ordering and empty evaluation.
        if curve.points:
            assert curve.points == ((0.0, 0.125), (0.0, 0.25), (1.0, 0.75))
            assert curve.interpolation == "linear"
            assert curve.evaluate(0.5) == 0.5
        else:
            assert curve.interpolation == "monotone_cubic"
            assert curve.evaluate(-3.0) == 0.0
        return cls.outputs(valid=True)


@pytest.mark.parametrize("in_process", (True, False))
def test_curve_editor_ordinary_hosts_coerce_values_report_histograms_and_cache(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "curve_fixture.py").write_text(
            "from test_curve_editor import CurveConsumer\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (CurveConsumer,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="curve-fixture"\npublisher="dinkster"\n'
            'namespaces=["curve-fixture"]\n[pack.entry]\n'
            'nodes="curve_fixture:NODES"\ntypes="curve_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(native.parent / "src"))
                ),
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest, in_process=True)
                    if in_process
                    else PackSpec(manifest, worker_group="curve", group_manifests=(native, fixture))
                )
            graph = Graph(
                nodes={
                    "curve": GraphNode(
                        "CurveEditor",
                        {
                            "curve": {
                                "points": [[1.0, 0.75], [0.0, 0.125], [0.0, 0.25]],
                                "interpolation": "linear",
                            },
                            "histogram": [7, -3, 2],
                        },
                    ),
                    "empty": GraphNode("CurveEditor", {"curve": [], "histogram": []}),
                    "consume": GraphNode(
                        "curve-fixture.consumer", {"curve": Link("curve", "curve")}
                    ),
                    "consume-empty": GraphNode(
                        "curve-fixture.consumer", {"curve": Link("empty", "curve")}
                    ),
                }
            )
            before = copy.deepcopy(graph)
            events: list[Any] = []
            engine = composer.composition.make_engine(events.append)
            for run in range(2):
                result = await engine.run(graph, ["consume", "consume-empty"])
                for node in ("consume", "consume-empty"):
                    assert result.outputs[node]["valid"].resolve() is True
                if run == 1:
                    assert set(result.cached) == set(graph.nodes)
            histogram_events = [
                event
                for event in events
                if event.kind == "node_event" and event.node_id in {"curve", "empty"}
            ]
            assert {
                (event.node_id, tuple(event.detail["data"]["histogram"]))
                for event in histogram_events
            } == {("curve", (7, -3, 2)), ("empty", ())}
            assert graph == before
        finally:
            await composer.close()

    asyncio.run(scenario())
