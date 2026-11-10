"""Curve editor identity and histogram event behavior through ordinary hosts."""

import asyncio
from pathlib import Path

import pytest
from dinkster_api.v1 import Curve
from dinkster_graph import Graph, GraphNode
from dinkster_schema import use_reporter
from dinkster_utilities.curve_editor import CurveEditor

from dinkster.compose import PackSpec, ServingComposer


def test_curve_editor_returns_original_curve_and_reports_histogram() -> None:
    curve = Curve(((0, 2), (1, 4)), "monotone_cubic")
    histogram = list(range(256))
    events = []
    with use_reporter(lambda name, data, blob: events.append((name, data, blob))):
        assert CurveEditor.execute(curve=curve, histogram=histogram)["curve"] is curve
    assert events == [("dinkster.curve.histogram", {"histogram": histogram}, None)]


@pytest.mark.parametrize("in_process", (True, False))
def test_curve_editor_ordinary_manifest_round_trip(in_process: bool) -> None:
    curve = Curve(((0, 0.2), (0.3, 0.8), (1, 0.4)), "monotone_cubic")

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            await composer.add_pack(
                PackSpec(
                    Path(__file__).resolve().parents[1] / "dinkster-pack.toml",
                    in_process=in_process,
                    trust_reserved=True,
                )
            )
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                {
                    "curve": GraphNode(
                        "dinkster.utilities.curve_editor", {"curve": curve.to_record()}
                    )
                }
            )
            for _ in range(2):
                result = await engine.run(graph, ["curve"])
                actual = result.outputs["curve"]["curve"].resolve()
                assert isinstance(actual, Curve)
                assert actual.to_record() == curve.to_record()
        finally:
            await composer.close()

    asyncio.run(scenario())
