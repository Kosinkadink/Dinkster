"""Native primitive value nodes and the text preview sink (wire v11).

What this proves: dinkster.int/float/string/string_multiline/boolean are
always-composed foundation nodes with semantic ports, legacy Primitive* aliases,
and first-class presentation metadata (NUMBER/STRING widgets, per-input
displayName); dinkster.preview_any stringifies any engine-decodable value
with ComfyUI PreviewAny's formatting; and a cross-worker value whose type
has no codec in this process fails as an ordinary node error, never a
worker crash.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import dinkster_api.v1 as api
import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, Invocation
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_foundation import (
    FOUNDATION_NODES,
    PRIMITIVE_NODES,
)
from dinkster_nodes_foundation.primitives import preview_text
from dinkster_nodes_media_io import (
    MEDIA_IO_NODES,
    SAVE_TARGET_NODES,
    SetSaveTargetPrefix,
    register_media_types,
)
from dinkster_schema import (
    InputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_values import (
    EncodedPayload,
    TypeRegistry,
    register_core_types,
)
from dinkster_workers import InProcessWorker

from dinkster.compose import PackSpec, ServingComposer

PRIMITIVE_TYPES = {node.schema().node_type for node in PRIMITIVE_NODES}
PRIMITIVE_TEST_NODES = [*FOUNDATION_NODES, *MEDIA_IO_NODES]


def make_engine() -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    register_media_types(registry)
    return Engine(
        schemas=build_schemas(PRIMITIVE_TEST_NODES),
        registry=registry,
        worker=InProcessWorker(build_node_types(PRIMITIVE_TEST_NODES), registry),
        cache=MemoryLRUCache(),
    )


def _entry(wire: dict[str, object], entry_id: str) -> dict[str, Any]:
    for item in cast("list[dict[str, Any]]", wire["interface"]):
        if item.get("id") == entry_id:
            return item
    raise AssertionError(f"{wire['nodeType']}: no interface entry {entry_id!r}")


def test_primitives_are_foundation_nodes_with_legacy_aliases() -> None:
    """The five value primitives + preview live in dinkster-nodes-foundation (the
    always-composed user surface, zero ComfyUI dependence) and each value
    primitive claims exactly its v1 class name as an alias. The preview
    node claims NO alias: translated comfy.PreviewAny keeps the name (it
    runs beside torch and stringifies resident values this one cannot)."""
    schemas = build_schemas(FOUNDATION_NODES)
    assert PRIMITIVE_TYPES <= set(schemas)
    expected_aliases = {
        "dinkster.int": ("PrimitiveInt",),
        "dinkster.float": ("PrimitiveFloat",),
        "dinkster.string": ("PrimitiveString",),
        "dinkster.string_multiline": ("PrimitiveStringMultiline",),
        "dinkster.boolean": ("PrimitiveBoolean",),
        "dinkster.preview_any": (),
    }
    for node_type, aliases in expected_aliases.items():
        assert schemas[node_type].aliases == aliases, node_type


def test_create_list_keeps_its_comfy_name_and_alias() -> None:
    schema = build_schemas(FOUNDATION_NODES)["std.list.make"]

    assert schema.display_name == "Create List"
    assert schema.aliases == ("CreateList",)


def test_append_to_list_declares_one_list_and_variadic_items() -> None:
    schema = build_schemas(FOUNDATION_NODES)["std.list.append"]

    assert schema.display_name == "Append to List"
    assert schema.inputs[0].type == TypeExpr.list_of(TypeExpr.variable("T"))
    family = schema.input_families[0]
    assert family.template == (InputSpec("value", TypeExpr.variable("T")),)
    assert family.min_members == 1
    assert schema.outputs[0].type == TypeExpr.list_of(TypeExpr.variable("T"))

    wire = schema_to_wire(schema)
    assert schema_from_wire(wire) == schema
    family_wire = next(
        entry
        for entry in cast("list[dict[str, Any]]", wire["interface"])
        if entry["role"] == "inputFamily"
    )
    assert family_wire["template"][0]["type"] == {
        "kind": "variable",
        "templateId": "T",
    }


def test_list_slice_and_range_schemas_preserve_element_types() -> None:
    schemas = build_schemas(FOUNDATION_NODES)

    slice_schema = schemas["std.list.slice"]
    assert tuple(item.id for item in slice_schema.inputs) == ("list", "start", "stop", "step")
    assert slice_schema.inputs[0].type == TypeExpr.list_of(TypeExpr.variable("T"))
    assert slice_schema.inputs[1].required is False
    assert slice_schema.inputs[2].required is False
    assert slice_schema.inputs[3].default == 1
    assert slice_schema.outputs[0].type == TypeExpr.list_of(TypeExpr.variable("T"))

    range_schema = schemas["std.list.range"]
    assert tuple(item.id for item in range_schema.inputs) == ("start", "stop", "step")
    assert range_schema.inputs[0].default == 0
    assert range_schema.inputs[1].required is True
    assert range_schema.inputs[1].default is None
    assert range_schema.inputs[2].default == 1
    assert range_schema.outputs[0].type == TypeExpr.list_of(TypeExpr.concrete("INT"))

    for schema in (slice_schema, range_schema):
        assert schema_from_wire(schema_to_wire(schema)) == schema


def test_scalar_or_list_match_inputs_share_the_unconstrained_t_contract() -> None:
    schemas = build_schemas(FOUNDATION_NODES)
    inputs = {
        "Create List": schemas["std.list.make"].input_families[0].template[0],
        "Repeat Item": schemas["std.list.repeat"].inputs[0],
        "Select Value false": schemas["dinkster.value.select"].inputs[1],
        "Select Value true": schemas["dinkster.value.select"].inputs[2],
        "Route Switch": schemas["dinkster.route.switch"].input_families[0].template[0],
        "Route Gate": schemas["dinkster.route.gate"].inputs[1],
    }

    for name, item in inputs.items():
        assert isinstance(item, InputSpec), name
        assert item.type == TypeExpr.variable("T"), name


def test_compat_translation_claims_match_always_composed_schema_aliases() -> None:
    from dinkster_compat_comfy.native import (
        FOUNDATION_CLAIMED_V1_NAMES,
        MEDIA_IO_CLAIMED_V1_NAMES,
    )

    foundation = {alias for node in FOUNDATION_NODES for alias in node.schema().aliases}
    media = {alias for node in MEDIA_IO_NODES for alias in node.schema().aliases}
    assert set(FOUNDATION_CLAIMED_V1_NAMES) == foundation
    assert set(MEDIA_IO_CLAIMED_V1_NAMES) == media


def test_primitive_wire_presentation() -> None:
    """Each primitive's single input is optional-with-default and carries
    its presentation contract: int gets the control-after-generate controller,
    float a fractional step, text exposes both string views, and every value
    input is labeled "Value"."""
    schemas = build_schemas(PRIMITIVE_TEST_NODES)

    int_wire = schema_to_wire(schemas["dinkster.int"])
    int_value = _entry(int_wire, "value")
    assert int_value["required"] is False
    assert int_value["default"] == 0
    assert int_value["displayName"] == "Value"
    assert int_value["widget"] == {
        "type": "NUMBER",
        "controlAfterGenerate": "fixed",
    }

    float_wire = schema_to_wire(schemas["dinkster.float"])
    assert _entry(float_wire, "value")["widget"] == {
        "type": "NUMBER",
        "step": 0.1,
    }

    string_wire = schema_to_wire(schemas["dinkster.string"])
    string_widget = _entry(string_wire, "value")["widget"]
    assert string_widget["type"] == "REPRESENTATIONS"
    assert string_widget["default"] == "single-line"
    assert string_widget["userSwitchable"] is True
    assert [item["id"] for item in string_widget["representations"]] == [
        "single-line",
        "multiline",
    ]

    multiline_wire = schema_to_wire(schemas["dinkster.string_multiline"])
    multiline_widget = _entry(multiline_wire, "value")["widget"]
    assert multiline_widget["type"] == "REPRESENTATIONS"
    assert multiline_widget["default"] == "multiline"
    assert schemas["dinkster.string_multiline"].search_visibility == "hidden"

    boolean_wire = schema_to_wire(schemas["dinkster.boolean"])
    assert "widget" not in _entry(boolean_wire, "value")

    for node_type in (
        "dinkster.int",
        "dinkster.float",
        "dinkster.string",
        "dinkster.string_multiline",
        "dinkster.boolean",
    ):
        output = next(
            entry
            for entry in cast(
                "list[dict[str, Any]]", schema_to_wire(schemas[node_type])["interface"]
            )
            if entry["role"] == "output"
        )
        assert output["knownValue"] == {"input": "value"}

    preview_wire = schema_to_wire(schemas["dinkster.preview_any"])
    assert preview_wire["outputNode"] is True
    source = _entry(preview_wire, "source")
    assert source["type"] == {"kind": "wildcard"}
    assert source["required"] is False

    target_wire = schema_to_wire(schemas["dinkster.set_save_target_prefix"])
    target = _entry(target_wire, "target")
    assert target["type"] == {
        "kind": "concrete",
        "types": ["dinkster.save_target"],
    }
    assert target["widget"] == {"type": "SAVE_TARGET"}
    prefix = _entry(target_wire, "prefix")
    assert prefix["type"] == {"kind": "concrete", "types": ["STRING"]}
    assert prefix["required"] is False
    assert "widget" not in prefix
    output = _entry(target_wire, "save_target")
    assert output["type"] == {
        "kind": "concrete",
        "types": ["dinkster.save_target"],
    }


def test_save_target_prefix_is_an_explicit_typed_constructor() -> None:
    """A computed STRING may replace only the relative prefix. The
    structured input supplies and preserves the mount id, the output remains
    dinkster.save_target, and omitting the replacement is an exact identity."""
    from dinkster_assets import AssetError, SaveTarget

    assert SAVE_TARGET_NODES == (SetSaveTargetPrefix,)
    base = SaveTarget(mount="project-output", prefix="manual/fallback")
    assert SetSaveTargetPrefix.execute(target=base) == {"save_target": base}

    computed = SetSaveTargetPrefix.execute(target=base, prefix="computed/scene")["save_target"]
    assert isinstance(computed, SaveTarget)
    assert computed == SaveTarget(mount="project-output", prefix="computed/scene")
    assert computed.to_wire() == {
        "mount": "project-output",
        "prefix": "computed/scene",
    }

    for unsafe in ("", "/absolute", "a/../b", "a\\b", "a//b"):
        with pytest.raises(AssetError):
            SetSaveTargetPrefix.execute(target=base, prefix=unsafe)

    async def scenario() -> None:
        engine = make_engine()
        graph = Graph(
            nodes={
                "prefix": GraphNode("dinkster.string", {"value": "computed/linked-scene"}),
                "target": GraphNode(
                    "dinkster.set_save_target_prefix",
                    {
                        "target": base.to_wire(),
                        "prefix": Link("prefix", "value"),
                    },
                ),
            }
        )
        result = await engine.run(graph, ["target"])
        wrapped = result.outputs["target"]["save_target"]
        assert wrapped.type_id == "dinkster.save_target"
        assert wrapped.resolve() == SaveTarget(
            mount="project-output", prefix="computed/linked-scene"
        )

    asyncio.run(scenario())


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
                '[pack]\nname = "color-fixture"\nnamespaces = ["color-fixture"]\n'
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

    asyncio.run(scenario())


def test_primitives_execute_with_values_and_defaults() -> None:
    """Every primitive echoes its value (explicit and default) through a
    torch-free engine, and preview_any renders a linked int as text."""

    async def scenario() -> None:
        engine = make_engine()
        graph = Graph(
            nodes={
                "i": GraphNode("dinkster.int", {"value": 7}),
                "f": GraphNode("dinkster.float", {"value": 0.25}),
                "s": GraphNode("dinkster.string", {"value": "hello"}),
                "m": GraphNode("dinkster.string_multiline", {"value": "a\nb"}),
                "b": GraphNode("dinkster.boolean", {"value": True}),
                "p": GraphNode("dinkster.preview_any", {"source": Link("i", "value")}),
            }
        )
        result = await engine.run(graph, ["i", "f", "s", "m", "b", "p"])
        assert result.outputs["i"]["value"].resolve() == 7
        assert result.outputs["i"]["value"].type_id == "INT"
        assert result.outputs["f"]["value"].resolve() == 0.25
        assert result.outputs["s"]["value"].resolve() == "hello"
        assert result.outputs["m"]["value"].resolve() == "a\nb"
        assert result.outputs["b"]["value"].resolve() is True
        assert result.outputs["p"]["text"].resolve() == "7"

        defaults = Graph(
            nodes={
                node_type.split(".", 1)[1]: GraphNode(node_type, {})
                for node_type in PRIMITIVE_TYPES
            }
        )
        result = await engine.run(defaults, sorted(defaults.nodes))
        assert result.outputs["int"]["value"].resolve() == 0
        assert result.outputs["float"]["value"].resolve() == 0.0
        assert result.outputs["string"]["value"].resolve() == ""
        assert result.outputs["string_multiline"]["value"].resolve() == ""
        assert result.outputs["boolean"]["value"].resolve() is False
        # An unlinked preview accepts absence and previews "None".
        assert result.outputs["preview_any"]["text"].resolve() == "None"

    asyncio.run(scenario())


def test_preview_text_formats_like_comfy_preview_any() -> None:
    """ComfyUI PreviewAny's formatting contract, torch-free: scalars and
    strings verbatim, JSON-able structures as pretty JSON, str() as the
    fallback, and a plain apology when even str() fails."""
    assert preview_text(None) == "None"
    assert preview_text("already text") == "already text"
    assert preview_text(7) == "7"
    assert preview_text(0.5) == "0.5"
    assert preview_text(True) == "True"
    assert preview_text({"a": 1}) == '{\n    "a": 1\n}'
    assert preview_text([1, 2]) == "[\n    1,\n    2\n]"

    class OnlyStr:
        def __str__(self) -> str:
            return "stringified"

    assert preview_text(OnlyStr()) == "stringified"

    class Hostile:
        def __str__(self) -> str:
            raise RuntimeError("no")

    assert preview_text(Hostile()) == "source exists, but could not be serialized."


def test_unresolvable_payload_fails_as_node_error() -> None:
    """A linked value produced in another worker whose type has no codec
    here (a wildcard input fed a resident/tensor type) fails THIS node
    with the type named - an ordinary InvocationResult error, never an
    exception escaping the worker (and never a leaked traceback)."""

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        worker = InProcessWorker(build_node_types(PRIMITIVE_TEST_NODES), registry)
        donor = registry.wrap("STRING", "x")
        alien = dataclasses.replace(
            donor,
            type_id="alien.tensor",
            payload=EncodedPayload(type_id="alien.tensor", data=b"\x00", decoder=None),
        )
        schema = build_schemas(FOUNDATION_NODES)["dinkster.preview_any"]
        result = await worker.invoke(
            Invocation(
                invocation_id="i1",
                node_id="p",
                node_type="dinkster.preview_any",
                inputs={"source": alien},
                effective_schema=schema,
            )
        )
        assert result.error is not None
        assert result.error.node_id == "p"
        assert "alien.tensor" in result.error.message
        assert "not registered" in result.error.message
        assert "Traceback" not in result.error.message

    asyncio.run(scenario())
