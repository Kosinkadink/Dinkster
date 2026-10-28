"""Loss graph pixels and ordinary native preview delivery."""

from __future__ import annotations

import asyncio
import io
import os
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import pytest
from dinkster_api.v1 import LOSS_MAP_TYPE, Node, NodeSchema, OutputSpec, TypeExpr, schema_to_wire
from dinkster_engine import EngineEvent
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.loss_graph import LossGraphNode
from dinkster_schema import use_reporter
from PIL import Image

from dinkster.compose import PackSpec, ServingComposer


def test_loss_graph_serialized_prefix_is_single_line_and_required() -> None:
    inputs = schema_to_wire(LossGraphNode.schema())["interface"]
    assert [item["id"] for item in inputs] == ["loss", "filename_prefix"]
    assert all(item["required"] is True for item in inputs)
    assert inputs[1]["widget"] == {"type": "STRING", "multiline": False}


def _assert_plot(blob: bytes) -> None:
    with Image.open(io.BytesIO(blob)) as image:
        assert image.size == (840, 520) and image.mode == "RGB"
        # Five losses 1,5,2,9,3 yield x=40,200,360,520,680 (not an endpoint at 840).
        for point in ((120, 360), (280, 330), (440, 210), (600, 180), (680, 360)):
            assert image.getpixel(point) == (0, 0, 255)
        assert image.getpixel((40, 200)) == (0, 0, 0)
        assert image.getpixel((500, 480)) == (0, 0, 0)
        assert image.getpixel((839, 200)) == (255, 255, 255)
        assert image.getpixel((800, 360)) == (255, 255, 255)


def test_loss_graph_previews_asymmetric_points_and_ignores_prefix_without_mutation() -> None:
    loss = {"loss": [1.0, 5.0, 2.0, 9.0, 3.0], "other": {"step": 17}}
    original = deepcopy(loss)
    captured: list[tuple[str, Mapping[str, object], bytes | None]] = []

    def capture(name: str, data: Mapping[str, object], blob: bytes | None) -> None:
        captured.append((name, data, blob))

    for prefix in ("loss_graph", "../unused/prefix"):
        with use_reporter(capture):
            assert LossGraphNode.execute(loss=loss, filename_prefix=prefix) == {}
    assert loss == original
    assert len(captured) == 2
    assert captured[0] == captured[1]
    name, data, blob = captured[0]
    assert name == "preview"
    assert data == {"mime": "image/png", "width": 840, "height": 520}
    assert isinstance(blob, bytes)
    _assert_plot(blob)
    schema = LossGraphNode.schema()
    assert [item.id for item in schema.inputs] == ["loss", "filename_prefix"]
    assert schema.inputs[0].type == TypeExpr.concrete(LOSS_MAP_TYPE)
    assert schema.inputs[1].default == "loss_graph"
    assert schema.output_node and schema.emits_previews and schema.outputs == ()


@pytest.mark.parametrize(
    ("values", "error"),
    (([], ValueError), ([2.0], ZeroDivisionError), ([3.0, 3.0], ZeroDivisionError)),
)
def test_loss_graph_preserves_fixed_empty_and_constant_loss_errors(
    values: list[float], error: type[Exception]
) -> None:
    with pytest.raises(error):
        LossGraphNode.execute(loss={"loss": values}, filename_prefix="unused")


class LossSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "loss-fixture.source", outputs=(OutputSpec("loss", TypeExpr.concrete(LOSS_MAP_TYPE)),)
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(loss={"loss": [1.0, 5.0, 2.0, 9.0, 3.0]})


@pytest.mark.parametrize("in_process", (True, False))
def test_loss_graph_ordinary_native_hosts_deliver_preview_without_files(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        checkout = package.parents[1]
        native = package / "dinkster-pack.toml"
        (tmp_path / "loss_fixture.py").write_text(
            "from test_loss_graph import LossSource\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (LossSource,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="loss-fixture"\npublisher="dinkster"\n'
            'namespaces=["loss-fixture"]\n[pack.entry]\nnodes="loss_fixture:NODES"\n'
            'types="loss_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        paths = (
            tmp_path,
            Path(__file__).parent,
            checkout / "src",
            *sorted(checkout.glob("packages/*/src")),
            *(Path(path) for path in sys.path if path and Path(path).is_dir()),
        )
        output = tmp_path / "output"
        output.mkdir()
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(map(str, paths)),
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest, in_process=True)
                    if in_process
                    else PackSpec(manifest, worker_group="loss", group_manifests=(native, fixture))
                )
            events: list[EngineEvent] = []
            engine = composer.composition.make_engine(events.append)
            previews = []
            for prefix in ("loss_graph", str(output / "ignored")):
                events.clear()
                graph = Graph(
                    nodes={
                        "source": GraphNode("loss-fixture.source"),
                        "plot": GraphNode(
                            "LossGraphNode",
                            {"loss": Link("source", "loss"), "filename_prefix": prefix},
                        ),
                    }
                )
                result = await engine.run(graph, ["plot"])
                assert result.outputs["plot"] == {}
                reports = [
                    event
                    for event in events
                    if event.kind == "node_event"
                    and event.node_id == "plot"
                    and event.detail.get("name") == "preview"
                ]
                assert len(reports) == 1
                assert reports[0].run_id == result.run_id
                blob = reports[0].detail["blob"]
                assert isinstance(blob, bytes)
                _assert_plot(blob)
                previews.append(blob)
                assert list(output.iterdir()) == []
            assert previews[0] == previews[1]
        finally:
            await composer.close()
        assert list(output.iterdir()) == []

    asyncio.run(scenario())
