"""First-party primitive execution and portable color values across workers."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path

import dinkster_api.v1 as api
import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, Invocation
from dinkster_graph import Graph, GraphNode, Link
from dinkster_schema import build_node_types, build_schemas
from dinkster_utilities import NODES, register_types
from dinkster_utilities.utilities import PreviewAny
from dinkster_values import EncodedPayload, TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

from dinkster.compose import PackSpec, ServingComposer
from tests.async_scenarios import run_scenario


class ColorSource(api.Node):
    @classmethod
    def define_schema(cls) -> api.NodeSchema:
        return api.NodeSchema(
            node_type="color-fixture.source",
            outputs=(
                api.OutputSpec("color", api.TypeExpr.concrete(api.COLOR_TYPE)),
                api.OutputSpec("palette", api.TypeExpr.concrete(api.COLORS_TYPE)),
                api.OutputSpec("preview", api.TypeExpr.concrete(api.RESOLUTION_PREVIEW_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(
            color="#aB12f080",
            palette=["#1200ff", "red", "#1200ff"],
            preview={"width": 832, "height": 1216, "extra": {"enabled": False}},
        )


class ColorInspect(api.Node):
    @classmethod
    def define_schema(cls) -> api.NodeSchema:
        return api.NodeSchema(
            node_type="color-fixture.inspect",
            inputs=(
                api.InputSpec("color", api.TypeExpr.concrete(api.COLOR_TYPE)),
                api.InputSpec("palette", api.TypeExpr.concrete(api.COLORS_TYPE)),
                api.InputSpec(
                    "preview", api.TypeExpr.concrete(api.RESOLUTION_PREVIEW_TYPE), required=False
                ),
            ),
            outputs=(api.OutputSpec("text", api.TypeExpr.concrete(api.CORE_STRING)),),
        )

    @classmethod
    def execute(
        cls, *, color: str, palette: list[str], preview: dict[str, object] | None = None
    ) -> Mapping[str, object]:
        return cls.outputs(text=json.dumps([color, palette, preview]))


COLOR_FIXTURES = (ColorSource, ColorInspect)


def test_color_preview_registration_is_portable_and_idempotent() -> None:
    registry = TypeRegistry()
    api.register_inference_types(registry)
    api.register_inference_types(registry)
    assert (api.COLOR_TYPE, api.COLORS_TYPE, api.RESOLUTION_PREVIEW_TYPE) == (
        "COLOR",
        "COLORS",
        "RESOLUTION_PREVIEW",
    )
    for type_id, source in (
        (api.COLOR_TYPE, "#ff001280"),
        (api.COLORS_TYPE, []),
        (api.COLORS_TYPE, ["blue", "#00ff00", "blue"]),
        (api.RESOLUTION_PREVIEW_TYPE, {}),
        (api.RESOLUTION_PREVIEW_TYPE, {"width": 832, "height": 1216}),
    ):
        spec = registry.spec(type_id)
        encoded = spec.encode(source)
        assert encoded.startswith(b"json:")
        decoded = spec.decode(encoded)
        assert decoded == source
        assert (
            registry.wrap(type_id, decoded).fingerprint
            == registry.wrap(type_id, source).fingerprint
        )
    assert (
        registry.wrap(api.COLORS_TYPE, ["red", "blue"]).fingerprint
        != registry.wrap(api.COLORS_TYPE, ["blue", "red"]).fingerprint
    )


@pytest.mark.parametrize("in_process", (True, False))
def test_color_and_resolution_preview_types_cross_pack_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env={"PYTHONPATH": str(Path(__file__).parent)})
        try:
            (tmp_path / "color_fixture.py").write_text(
                "from test_primitives import COLOR_FIXTURES as NODES\n"
                "from dinkster_api.v1 import register_inference_types\n",
                encoding="ascii",
            )
            manifest = tmp_path / "dinkster-pack.toml"
            manifest.write_text(
                '[pack]\nname = "color-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["color-fixture"]\n'
                '[pack.sandbox]\n[pack.contracts]\nhost = "dinkster-pack-host/1"\n'
                'api = "dinkster-api/v1"\n[pack.entry]\nnodes = "color_fixture:NODES"\n'
                'types = "color_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
            engine = composer.composition.make_engine(lambda _event: None)
            for linked_preview in (True, False):
                inputs: dict[str, Link] = {
                    "color": Link("source", "color"),
                    "palette": Link("source", "palette"),
                }
                if linked_preview:
                    inputs["preview"] = Link("source", "preview")
                graph = Graph(
                    nodes={
                        "source": GraphNode("color-fixture.source"),
                        "inspect": GraphNode("color-fixture.inspect", inputs),
                    }
                )
                for _ in range(2):
                    result = await engine.run(graph, ["inspect"])
                    text = result.outputs["inspect"]["text"].resolve()
                    assert isinstance(text, str)
                    assert json.loads(text) == [
                        "#aB12f080",
                        ["#1200ff", "red", "#1200ff"],
                        {"width": 832, "height": 1216, "extra": {"enabled": False}}
                        if linked_preview
                        else None,
                    ]
        finally:
            await composer.close()

    run_scenario(scenario())


def test_first_party_primitives_execute_and_feed_preview() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        register_types(registry)
        engine = Engine(
            schemas=build_schemas(NODES),
            registry=registry,
            worker=InProcessWorker(build_node_types(NODES), registry),
            cache=MemoryLRUCache(),
        )
        cases = (
            ("i", "primitive_int", "INT", 7),
            ("f", "primitive_float", "FLOAT", 0.25),
            ("s", "primitive_string", "STRING", "hello"),
            ("m", "primitive_string_multiline", "STRING", "a\nb"),
            ("b", "primitive_boolean", "BOOLEAN", True),
        )
        nodes = {
            key: GraphNode(f"dinkster.utilities.{leaf}", {"value": value})
            for key, leaf, _output, value in cases
        }
        nodes["p"] = GraphNode("dinkster.utilities.preview_any", {"source": Link("i", "INT")})
        result = await engine.run(Graph(nodes=nodes), list(nodes))
        for key, _leaf, output, value in cases:
            actual = result.outputs[key][output]
            assert actual.type_id == output
            assert actual.resolve() == value
        assert result.outputs["p"]["STRING"].resolve() == "7"

    run_scenario(scenario())


def test_preview_text_formats_like_comfy_preview_any() -> None:
    for source, text in (
        (None, "None"),
        ("already text", "already text"),
        (7, "7"),
        (0.5, "0.5"),
        (True, "True"),
        ({"a": 1}, '{\n    "a": 1\n}'),
        ([1, 2], "[\n    1,\n    2\n]"),
    ):
        assert PreviewAny.execute(source=source) == {"STRING": text}

    class OnlyStr:
        def __str__(self) -> str:
            return "stringified"

    assert PreviewAny.execute(source=OnlyStr()) == {"STRING": "stringified"}

    class Hostile:
        def __str__(self) -> str:
            raise RuntimeError("no")

    assert PreviewAny.execute(source=Hostile()) == {
        "STRING": "source exists, but could not be serialized."
    }


def test_unresolvable_payload_fails_as_node_error() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        worker = InProcessWorker(build_node_types((PreviewAny,)), registry)
        donor = registry.wrap("STRING", "x")
        alien = dataclasses.replace(
            donor,
            type_id="alien.tensor",
            payload=EncodedPayload(type_id="alien.tensor", data=b"\x00", decoder=None),
        )
        schema = PreviewAny.schema()
        result = await worker.invoke(
            Invocation(
                invocation_id="i1",
                node_id="p",
                node_type=schema.node_type,
                inputs={"source": alien},
                effective_schema=schema,
            )
        )
        assert result.error is not None
        assert result.error.node_id == "p"
        assert "alien.tensor" in result.error.message
        assert "not registered" in result.error.message
        assert "Traceback" not in result.error.message

    run_scenario(scenario())
