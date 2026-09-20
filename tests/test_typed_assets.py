"""Typed assets end to end (joint contract 2026-07-26, wire v12).

The pinned rules this file proves: the runtime grammar grows exactly one
constructor (``asset<id>``, recursive with ``list<...>``); an asset<T>
value is the base asset envelope with a parametric stamp; the ONLY
coercions are asset-originated (``asset<T> -> T``, ``list<asset<T>> ->
list<T>``, ``asset<list<T>> -> list<T>``, and ``list<asset<T>> -> T``
through a registered batch merge as the terminal step); plain list/scalar
bridging stays a structural error; coerced cache identity is source
digest(s) + provider identities, never decoded bytes; decode runs in the
worker, planning and cache identity in the engine, and both consume ONE
planner so they can never disagree.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from dinkster_assets import AssetRef, digest_bytes, register_asset_type
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, ExecutionError, GraphValidationError
from dinkster_graph import Graph, GraphNode, TypedLiteral, validate
from dinkster_schema import (
    AssetWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeSolveError,
    WidgetRepresentation,
    WidgetRepresentations,
    bind_type_variables,
    build_node_types,
    build_schemas,
    plan_asset_coercion,
    resolved_type_id,
    schema_from_wire,
    schema_to_wire,
    type_expr_from_wire,
    type_expr_to_wire,
)
from dinkster_values import (
    CORE_STRING,
    TypeRegistry,
    asset_type_id,
    parse_asset_type_id,
    register_core_types,
    runtime_type_atom,
)
from dinkster_workers import InProcessWorker

# ---------------------------------------------------------------------------
# Grammar


def test_asset_type_id_round_trips() -> None:
    for element in (
        "comfy.IMAGE",
        "list<comfy.IMAGE>",
        "asset<comfy.IMAGE>",  # grammatical even though registration bans it
    ):
        assert parse_asset_type_id(asset_type_id(element)) == element


def test_parse_asset_type_id_rejects_non_asset_spellings() -> None:
    for spelling in ("comfy.IMAGE", "list<comfy.IMAGE>", "asset<>", "asset<", "", "x>"):
        assert parse_asset_type_id(spelling) is None


def test_runtime_type_atom_peels_the_whole_grammar() -> None:
    assert runtime_type_atom("comfy.IMAGE") == "comfy.IMAGE"
    assert runtime_type_atom("asset<comfy.IMAGE>") == "comfy.IMAGE"
    assert runtime_type_atom("asset<list<comfy.IMAGE>>") == "comfy.IMAGE"
    assert runtime_type_atom("list<asset<comfy.IMAGE>>") == "comfy.IMAGE"
    assert runtime_type_atom("list<asset<list<comfy.IMAGE>>>") == "comfy.IMAGE"
    assert runtime_type_atom("stream<comfy.IMAGE>") == "comfy.IMAGE"
    assert runtime_type_atom("asset<stream<comfy.IMAGE>>") == "comfy.IMAGE"


def test_runtime_type_atom_rejects_malformed_ids() -> None:
    for bad in (
        "",
        "asset<>",
        "list<asset<>>",
        "asset<comfy.IMAGE",
        "list<comfy.IMAGE>>",
        "asset<comfy.IMAGE>x",
        "stream<>",
        "stream<comfy.IMAGE",
        "stream<comfy.IMAGE>>",
        "as<set>",
    ):
        assert runtime_type_atom(bad) is None, bad


# ---------------------------------------------------------------------------
# TypeExpr model + wire


IMAGE = TypeExpr.concrete("comfy.IMAGE")


def test_asset_expr_runtime_ids_and_cardinality() -> None:
    single = TypeExpr.asset_of(IMAGE)
    assert single.runtime_type_id() == "asset<comfy.IMAGE>"
    assert single.cardinality() == "scalar"

    one_asset_many_images = TypeExpr.asset_of(TypeExpr.list_of(IMAGE))
    assert one_asset_many_images.runtime_type_id() == "asset<list<comfy.IMAGE>>"
    assert one_asset_many_images.cardinality() == "scalar"  # ONE AssetRef

    many_assets = TypeExpr.list_of(TypeExpr.asset_of(IMAGE))
    assert many_assets.runtime_type_id() == "list<asset<comfy.IMAGE>>"
    assert many_assets.cardinality() == "list"

    assert TypeExpr.runtime_cardinality("asset<list<comfy.IMAGE>>") == "scalar"
    assert TypeExpr.runtime_cardinality("list<asset<comfy.IMAGE>>") == "list"


def test_asset_expr_accepts_only_matching_asset_ids() -> None:
    expr = TypeExpr.asset_of(IMAGE)
    assert expr.accepts_concrete("asset<comfy.IMAGE>")
    assert not expr.accepts_concrete("comfy.IMAGE")
    assert not expr.accepts_concrete("asset<comfy.LATENT>")
    assert not expr.accepts_concrete("list<asset<comfy.IMAGE>>")


def test_asset_ids_are_never_spelled_concrete() -> None:
    with pytest.raises(ValueError, match="asset types are spelled"):
        TypeExpr.concrete("asset<comfy.IMAGE>")
    with pytest.raises(ValueError, match="asset types are spelled"):
        TypeExpr.union("comfy.IMAGE", "asset<comfy.IMAGE>")


def test_asset_expr_wire_round_trip() -> None:
    for expr in (
        TypeExpr.asset_of(IMAGE),
        TypeExpr.asset_of(TypeExpr.list_of(IMAGE)),
        TypeExpr.list_of(TypeExpr.asset_of(IMAGE)),
    ):
        assert type_expr_from_wire(type_expr_to_wire(expr)) == expr


# ---------------------------------------------------------------------------
# Registry: parametric wrap + provider registration rules


def make_registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_core_types(registry)
    register_asset_type(registry)
    return registry


def ref_for(data: bytes, name: str = "note.txt") -> AssetRef:
    return AssetRef(digest=digest_bytes(data), name=name, size=len(data))


def test_parametric_asset_wrap_keeps_the_stamp() -> None:
    registry = make_registry()
    ref = ref_for(b"hello")
    typed = registry.wrap("asset<core.string>", ref)
    assert typed.type_id == "asset<core.string>"
    assert typed.fingerprint == ref.digest  # identity is the content digest
    untyped = registry.wrap("dinkster.asset", ref)
    assert untyped.fingerprint == typed.fingerprint

    assert "asset<core.string>" in registry
    assert "asset<list<core.string>>" in registry
    assert "asset<no.such.type>" not in registry
    assert "asset<>" not in registry
    # Parametric ids resolve through the base spec, never enumerate:
    assert "asset<core.string>" not in registry.type_ids()


def test_unregistered_asset_stamps_never_wrap_or_code() -> None:
    """spec()/wrap() hold the same bar as __contains__: a well-formed stamp
    over an unregistered atom is refused, never silently coded as the base
    asset."""
    registry = make_registry()
    ref = ref_for(b"hello")
    with pytest.raises(KeyError, match="no.such.type"):
        registry.spec("asset<no.such.type>")
    with pytest.raises(KeyError, match="no.such.type"):
        registry.wrap("asset<no.such.type>", ref)
    with pytest.raises(KeyError):
        registry.wrap("asset<>", ref)  # malformed: not even asset-shaped


def test_decoder_registration_rules() -> None:
    registry = make_registry()
    registry.register_asset_decoder("core.string", provider_id="t.text@1", decode=lambda ref: "x")
    with pytest.raises(ValueError, match="already registered"):
        registry.register_asset_decoder(
            "core.string", provider_id="t.text@2", decode=lambda ref: "y"
        )
    with pytest.raises(ValueError, match="provider_id must be non-empty"):
        registry.register_asset_decoder("core.int", provider_id="", decode=lambda ref: 1)
    with pytest.raises(ValueError, match="no provider chains"):
        registry.register_asset_decoder(
            "asset<core.string>", provider_id="t.chain@1", decode=lambda ref: ref
        )
    with pytest.raises(KeyError, match="unregistered decode target"):
        registry.register_asset_decoder("no.such.type", provider_id="t.err@1", decode=lambda ref: 1)
    # list targets are legal: one asset decoding to a list of values.
    registry.register_asset_decoder(
        "list<core.string>", provider_id="t.lines@1", decode=lambda ref: ["x"]
    )
    assert registry.asset_decoder_for("list<core.string>") is not None


def test_batch_merge_registration_rules() -> None:
    registry = make_registry()
    registry.register_batch_merge("core.string", provider_id="t.join@1", merge=lambda items: "")
    with pytest.raises(ValueError, match="already registered"):
        registry.register_batch_merge("core.string", provider_id="t.join@2", merge=lambda items: "")
    with pytest.raises(ValueError, match="atom type ids"):
        registry.register_batch_merge(
            "list<core.string>", provider_id="t.err@1", merge=lambda items: ""
        )
    with pytest.raises(KeyError, match="unregistered value type"):
        registry.register_batch_merge("no.such.type", provider_id="t.err@1", merge=lambda items: "")
    with pytest.raises(ValueError, match="provider_id must be non-empty"):
        registry.register_batch_merge("core.int", provider_id="", merge=lambda items: 0)


# ---------------------------------------------------------------------------
# Planner: the closed coercion vocabulary


def test_planner_covers_exactly_the_pinned_coercions() -> None:
    scalar = TypeExpr.concrete(CORE_STRING)
    listy = TypeExpr.list_of(scalar)

    decode = plan_asset_coercion("asset<core.string>", scalar)
    assert decode is not None and decode.kind == "decode"
    assert decode.target_type_id == "core.string"
    assert decode.result_type_id == "core.string"
    assert decode.merge_type_id is None

    packed = plan_asset_coercion("asset<list<core.string>>", listy)
    assert packed is not None and packed.kind == "decode"
    assert packed.target_type_id == "list<core.string>"

    lift = plan_asset_coercion("list<asset<core.string>>", listy)
    assert lift is not None and lift.kind == "lift"
    assert lift.target_type_id == "core.string"
    assert lift.result_type_id == "list<core.string>"

    merge = plan_asset_coercion("list<asset<core.string>>", scalar)
    assert merge is not None and merge.kind == "merge"
    assert merge.target_type_id == "core.string"
    assert merge.result_type_id == "core.string"
    assert merge.merge_type_id == "core.string"


def test_planner_refuses_everything_else() -> None:
    scalar = TypeExpr.concrete(CORE_STRING)
    listy = TypeExpr.list_of(scalar)
    # Plain values never coerce: no list/scalar bridging exists.
    assert plan_asset_coercion("core.string", listy) is None
    assert plan_asset_coercion("list<core.string>", scalar) is None
    # Wrong element type.
    assert plan_asset_coercion("asset<core.int>", scalar) is None
    # A list-decoding asset never merges (list-of-lists does not batch).
    assert plan_asset_coercion("list<asset<list<core.string>>>", listy) is None
    # No implicit flattening between asset list forms.
    assert (
        plan_asset_coercion("asset<list<core.string>>", TypeExpr.list_of(TypeExpr.asset_of(scalar)))
        is None
    )


def test_planner_missing_providers_names_what_is_absent() -> None:
    registry = make_registry()
    merge = plan_asset_coercion("list<asset<core.string>>", TypeExpr.concrete(CORE_STRING))
    assert merge is not None
    missing = merge.missing_providers(registry)
    assert len(missing) == 2  # decoder and merger both absent
    registry.register_asset_decoder("core.string", provider_id="t.text@1", decode=lambda ref: "x")
    missing = merge.missing_providers(registry)
    assert len(missing) == 1 and "batch merge" in missing[0]
    registry.register_batch_merge("core.string", provider_id="t.join@1", merge=lambda items: "")
    assert merge.missing_providers(registry) == ()


# ---------------------------------------------------------------------------
# Type-variable solving through the asset constructor


def test_asset_expr_binds_and_resolves_type_variables() -> None:
    """Generic asset nodes (asset<T> -> asset<T>) solve exactly like list
    generics: the constructor recurses, the variable binds to the element."""
    T = TypeExpr.variable("T")
    specs = (InputSpec("ref", TypeExpr.asset_of(T)),)
    bindings = bind_type_variables(specs, {"ref": "asset<core.int>"})
    assert bindings == {"T": "core.int"}
    assert resolved_type_id(TypeExpr.asset_of(T), bindings) == "asset<core.int>"
    assert (
        resolved_type_id(TypeExpr.list_of(TypeExpr.asset_of(T)), bindings)
        == "list<asset<core.int>>"
    )
    nested = bind_type_variables(
        (InputSpec("refs", TypeExpr.list_of(TypeExpr.asset_of(T))),),
        {"refs": "list<asset<core.string>>"},
    )
    assert nested == {"T": "core.string"}
    with pytest.raises(TypeSolveError, match="expected an asset value"):
        bind_type_variables(specs, {"ref": "core.int"})


# ---------------------------------------------------------------------------
# The test node set: plain string consumers fed by asset stamps


STRING = TypeExpr.concrete(CORE_STRING)


class Upper(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ta.upper",
            display_name="Upper",
            category="test",
            inputs=(InputSpec("text", STRING),),
            outputs=(OutputSpec("text", STRING),),
        )

    @classmethod
    def execute(cls, *, text: str) -> Mapping[str, object]:
        assert isinstance(text, str), f"coercion must deliver str, got {type(text)}"
        return cls.outputs(text=text.upper())


class AssetUpper(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ta.asset_upper",
            display_name="Asset Upper",
            category="test",
            inputs=(
                InputSpec(
                    "text",
                    STRING,
                    widget=WidgetRepresentations(
                        (
                            WidgetRepresentation("picker", AssetWidget()),
                            WidgetRepresentation("alternate-picker", AssetWidget()),
                        ),
                        default="picker",
                        user_switchable=True,
                    ),
                ),
            ),
            outputs=(OutputSpec("text", STRING),),
        )

    @classmethod
    def execute(cls, *, text: str) -> Mapping[str, object]:
        assert isinstance(text, str), f"coercion must deliver str, got {type(text)}"
        return cls.outputs(text=text.upper())


class Join(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ta.join",
            display_name="Join",
            category="test",
            inputs=(InputSpec("texts", TypeExpr.list_of(STRING)),),
            outputs=(OutputSpec("text", STRING),),
        )

    @classmethod
    def execute(cls, *, texts: Sequence[str]) -> Mapping[str, object]:
        assert all(isinstance(t, str) for t in texts)
        return cls.outputs(text="|".join(texts))


TA_NODES = (Upper, AssetUpper, Join)
SCHEMAS = build_schemas(TA_NODES)


class DirResolver:
    """Digest -> path over one directory of pre-hashed files."""

    def __init__(self, files: Mapping[str, Path]) -> None:
        self._files = dict(files)

    def resolve(self, digest: str) -> Path | None:
        return self._files.get(digest)


def store_assets(tmp_path: Path, *texts: str) -> tuple[DirResolver, list[dict[str, object]]]:
    """Write each text as a file; return a resolver plus wire descriptors
    (what an asset-typed literal carries: identity, never a path)."""
    files: dict[str, Path] = {}
    wires: list[dict[str, object]] = []
    for index, text in enumerate(texts):
        data = text.encode("utf-8")
        digest = digest_bytes(data)
        path = tmp_path / f"asset{index}.txt"
        path.write_bytes(data)
        files[digest] = path
        wires.append({"digest": digest, "name": path.name, "size": len(data)})
    return DirResolver(files), wires


def make_engine(
    resolver: DirResolver,
    *,
    decoder_id: str = "t.text@1",
    merger_id: str | None = "t.join@1",
    register_providers: bool = True,
    cache: MemoryLRUCache | None = None,
) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    register_asset_type(registry, resolver)
    if register_providers:
        registry.register_asset_decoder(
            "core.string",
            provider_id=decoder_id,
            decode=lambda ref: ref.read_bytes().decode("utf-8"),  # type: ignore[union-attr]
        )
        if merger_id is not None:
            registry.register_batch_merge(
                "core.string",
                provider_id=merger_id,
                merge=lambda items: "+".join(str(i) for i in items),
            )
    return Engine(
        schemas=SCHEMAS,
        registry=registry,
        worker=InProcessWorker(build_node_types(TA_NODES), registry),
        cache=cache if cache is not None else MemoryLRUCache(),
    )


# ---------------------------------------------------------------------------
# Graph validation


def registry_with_providers(*, merger: bool = True) -> TypeRegistry:
    registry = make_registry()
    registry.register_asset_decoder("core.string", provider_id="t.text@1", decode=lambda ref: "x")
    if merger:
        registry.register_batch_merge("core.string", provider_id="t.join@1", merge=lambda items: "")
    return registry


def asset_literal(*wires: object) -> TypedLiteral:
    if len(wires) == 1:
        return TypedLiteral("asset<core.string>", wires[0])
    return TypedLiteral("list<asset<core.string>>", list(wires))


WIRE = {"digest": digest_bytes(b"w"), "name": "w.txt", "size": 1}


def test_validation_accepts_coercible_asset_stamps() -> None:
    graph = Graph(
        nodes={
            "u": GraphNode("ta.upper", {"text": asset_literal(WIRE)}),
            "j": GraphNode("ta.join", {"texts": asset_literal(WIRE, WIRE)}),
            "m": GraphNode("ta.upper", {"text": asset_literal(WIRE, WIRE)}),
        }
    )
    diags = validate(graph, SCHEMAS, ["u", "j", "m"], known_types=registry_with_providers())
    assert diags == []


def test_validation_errors_on_missing_providers() -> None:
    graph = Graph(nodes={"m": GraphNode("ta.upper", {"text": asset_literal(WIRE, WIRE)})})
    # Merge plan with no merger registered: loud, named error.
    diags = validate(graph, SCHEMAS, ["m"], known_types=registry_with_providers(merger=False))
    assert [d.code for d in diags] == ["asset-coercion-unavailable"]
    assert "batch merge" in diags[0].message
    # No registry at hand (plain atom set): structural acceptance, silent -
    # the engine's own registry-backed validation is the enforcement point.
    quiet = validate(graph, SCHEMAS, ["m"], known_types={"core.string", "dinkster.asset"})
    assert quiet == []


def test_asset_widget_scalar_rejects_unstamped_asset_literal() -> None:
    graph = Graph(nodes={"u": GraphNode("ta.asset_upper", {"text": WIRE})})
    diags = validate(graph, SCHEMAS, ["u"], known_types=registry_with_providers())
    assert [d.code for d in diags] == ["unstamped-asset-widget-literal"]
    assert "must carry an asset<T> or list<asset<T>> type stamp" in diags[0].message


def test_validation_still_rejects_plain_shape_mismatches() -> None:
    graph = Graph(
        nodes={"m": GraphNode("ta.upper", {"text": TypedLiteral("list<core.string>", ["a"])})}
    )
    diags = validate(graph, SCHEMAS, ["m"], known_types=registry_with_providers())
    assert any(d.code == "list-into-scalar" for d in diags)


def test_asset_of_list_needs_its_own_list_target_decoder() -> None:
    """asset<list<T>> is ONE AssetRef whose decoder targets list<T>: the
    scalar-element decoder never covers it, and registering the list-target
    decoder makes it valid."""
    graph = Graph(
        nodes={"j": GraphNode("ta.join", {"texts": TypedLiteral("asset<list<core.string>>", WIRE)})}
    )
    registry = registry_with_providers()  # core.string decoder only
    diags = validate(graph, SCHEMAS, ["j"], known_types=registry)
    assert [d.code for d in diags] == ["asset-coercion-unavailable"]
    assert "list<core.string>" in diags[0].message
    registry.register_asset_decoder(
        "list<core.string>", provider_id="t.lines@1", decode=lambda ref: ["x"]
    )
    assert validate(graph, SCHEMAS, ["j"], known_types=registry) == []


# ---------------------------------------------------------------------------
# Execution: decode / lift / merge through the engine + worker


def test_decode_lift_and_merge_execute(tmp_path: Path) -> None:
    async def scenario() -> None:
        resolver, wires = store_assets(tmp_path, "alpha", "beta")
        engine = make_engine(resolver)

        decode = await engine.run(
            Graph(nodes={"u": GraphNode("ta.upper", {"text": asset_literal(wires[0])})}),
            ["u"],
        )
        assert decode.outputs["u"]["text"].resolve() == "ALPHA"

        lift = await engine.run(
            Graph(nodes={"j": GraphNode("ta.join", {"texts": asset_literal(wires[0], wires[1])})}),
            ["j"],
        )
        assert lift.outputs["j"]["text"].resolve() == "alpha|beta"

        merge = await engine.run(
            Graph(nodes={"u": GraphNode("ta.upper", {"text": asset_literal(wires[0], wires[1])})}),
            ["u"],
        )
        assert merge.outputs["u"]["text"].resolve() == "ALPHA+BETA"

    asyncio.run(scenario())


def test_asset_widget_scalar_decodes_stamped_asset(tmp_path: Path) -> None:
    async def scenario() -> None:
        resolver, wires = store_assets(tmp_path, "alpha")
        result = await make_engine(resolver).run(
            Graph(nodes={"u": GraphNode("ta.asset_upper", {"text": asset_literal(wires[0])})}),
            ["u"],
        )
        assert result.outputs["u"]["text"].resolve() == "ALPHA"

    asyncio.run(scenario())


def test_asset_widget_scalar_merges_stamped_asset_list(tmp_path: Path) -> None:
    async def scenario() -> None:
        resolver, wires = store_assets(tmp_path, "alpha", "beta")
        result = await make_engine(resolver).run(
            Graph(nodes={"u": GraphNode("ta.asset_upper", {"text": asset_literal(*wires)})}),
            ["u"],
        )
        assert result.outputs["u"]["text"].resolve() == "ALPHA+BETA"

    asyncio.run(scenario())


def test_asset_widget_scalar_merge_requires_registered_merger(tmp_path: Path) -> None:
    async def scenario() -> None:
        resolver, wires = store_assets(tmp_path, "alpha", "beta")
        graph = Graph(nodes={"u": GraphNode("ta.asset_upper", {"text": asset_literal(*wires)})})
        with pytest.raises(GraphValidationError) as err:
            await make_engine(resolver, merger_id=None).run(graph, ["u"])
        assert [d.code for d in err.value.diagnostics] == ["asset-coercion-unavailable"]
        assert "batch merge for type 'core.string'" in str(err.value)

    asyncio.run(scenario())


def test_asset_widget_scalar_unstamped_literal_fails_structurally(tmp_path: Path) -> None:
    async def scenario() -> None:
        resolver, wires = store_assets(tmp_path, "alpha")
        graph = Graph(nodes={"u": GraphNode("ta.asset_upper", {"text": wires[0]})})
        with pytest.raises(GraphValidationError) as err:
            await make_engine(resolver).run(graph, ["u"])
        assert [d.code for d in err.value.diagnostics] == ["unstamped-asset-widget-literal"]

    asyncio.run(scenario())


def test_asset_of_list_decodes_to_the_whole_list(tmp_path: Path) -> None:
    """One asset -> list<T> through its own list-target decoder."""

    async def scenario() -> None:
        resolver, wires = store_assets(tmp_path, "alpha\nbeta")
        registry = TypeRegistry()
        register_core_types(registry)
        register_asset_type(registry, resolver)
        registry.register_asset_decoder(
            "list<core.string>",
            provider_id="t.lines@1",
            decode=lambda ref: ref.read_bytes().decode("utf-8").splitlines(),  # type: ignore[union-attr]
        )
        engine = Engine(
            schemas=SCHEMAS,
            registry=registry,
            worker=InProcessWorker(build_node_types(TA_NODES), registry),
            cache=MemoryLRUCache(),
        )
        result = await engine.run(
            Graph(
                nodes={
                    "j": GraphNode(
                        "ta.join",
                        {"texts": TypedLiteral("asset<list<core.string>>", wires[0])},
                    )
                }
            ),
            ["j"],
        )
        assert result.outputs["j"]["text"].resolve() == "alpha|beta"

    asyncio.run(scenario())


def test_provider_identity_keys_the_cache(tmp_path: Path) -> None:
    """Same asset, same decoder identity -> cache hit across engines; a
    bumped decoder identity -> miss. Identity is the digest + provider ids,
    never the decoded bytes."""

    async def scenario() -> None:
        resolver, wires = store_assets(tmp_path, "alpha")
        graph = Graph(nodes={"u": GraphNode("ta.upper", {"text": asset_literal(wires[0])})})
        cache = MemoryLRUCache()

        first = await make_engine(resolver, cache=cache).run(graph, ["u"])
        assert first.executed == ("u",)
        same = await make_engine(resolver, cache=cache).run(graph, ["u"])
        assert same.cached == ("u",)
        bumped = await make_engine(resolver, decoder_id="t.text@2", cache=cache).run(graph, ["u"])
        assert bumped.executed == ("u",)

    asyncio.run(scenario())


def test_merger_identity_keys_the_cache(tmp_path: Path) -> None:
    async def scenario() -> None:
        resolver, wires = store_assets(tmp_path, "alpha", "beta")
        graph = Graph(
            nodes={"u": GraphNode("ta.upper", {"text": asset_literal(wires[0], wires[1])})}
        )
        cache = MemoryLRUCache()
        first = await make_engine(resolver, cache=cache).run(graph, ["u"])
        assert first.executed == ("u",)
        bumped = await make_engine(resolver, merger_id="t.join@2", cache=cache).run(graph, ["u"])
        assert bumped.executed == ("u",)

    asyncio.run(scenario())


def test_different_stamps_on_the_same_asset_never_share_a_cache_entry(
    tmp_path: Path,
) -> None:
    """The stamp is part of cache identity: the same AssetRef under
    asset<core.string> vs asset<core.int> shares a content fingerprint but
    must miss the cache and re-solve the output stamp (a generic
    asset<T> -> asset<T> node's output type depends on the stamp alone)."""

    class Identity(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            T = TypeExpr.variable("T")
            return NodeSchema(
                node_type="ta.identity",
                display_name="Identity",
                category="test",
                inputs=(InputSpec("ref", TypeExpr.asset_of(T)),),
                outputs=(OutputSpec("ref", TypeExpr.asset_of(T)),),
            )

        @classmethod
        def execute(cls, *, ref: AssetRef) -> Mapping[str, object]:
            return cls.outputs(ref=ref)

    async def scenario() -> None:
        resolver, wires = store_assets(tmp_path, "alpha")
        registry = TypeRegistry()
        register_core_types(registry)
        register_asset_type(registry, resolver)
        nodes = (Identity,)
        cache = MemoryLRUCache()
        engine = Engine(
            schemas=build_schemas(nodes),
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=cache,
        )

        def graph(stamp: str) -> Graph:
            return Graph(
                nodes={"i": GraphNode("ta.identity", {"ref": TypedLiteral(stamp, wires[0])})}
            )

        first = await engine.run(graph("asset<core.string>"), ["i"])
        assert first.executed == ("i",)
        assert first.outputs["i"]["ref"].type_id == "asset<core.string>"

        restamped = await engine.run(graph("asset<core.int>"), ["i"])
        assert restamped.executed == ("i",)  # different stamp, never a hit
        assert restamped.outputs["i"]["ref"].type_id == "asset<core.int>"

        again = await engine.run(graph("asset<core.string>"), ["i"])
        assert again.cached == ("i",)  # same stamp still hits

    asyncio.run(scenario())


def test_missing_providers_fail_validation_in_the_engine(tmp_path: Path) -> None:
    async def scenario() -> None:
        resolver, wires = store_assets(tmp_path, "alpha")
        engine = make_engine(resolver, register_providers=False)
        graph = Graph(nodes={"u": GraphNode("ta.upper", {"text": asset_literal(wires[0])})})
        with pytest.raises(GraphValidationError) as err:
            await engine.run(graph, ["u"])
        assert any(d.code == "asset-coercion-unavailable" for d in err.value.diagnostics)

    asyncio.run(scenario())


def test_decode_failure_is_a_node_error(tmp_path: Path) -> None:
    """A decoder that blows up fails THAT node with the coercion named -
    an ordinary node error, never a worker crash."""

    async def scenario() -> None:
        resolver, wires = store_assets(tmp_path, "alpha")
        registry = TypeRegistry()
        register_core_types(registry)
        register_asset_type(registry, resolver)
        registry.register_asset_decoder(
            "core.string",
            provider_id="t.broken@1",
            decode=lambda ref: (_ for _ in ()).throw(ValueError("corrupt")),
        )
        engine = Engine(
            schemas=SCHEMAS,
            registry=registry,
            worker=InProcessWorker(build_node_types(TA_NODES), registry),
            cache=MemoryLRUCache(),
        )
        graph = Graph(nodes={"u": GraphNode("ta.upper", {"text": asset_literal(wires[0])})})
        with pytest.raises(ExecutionError) as err:
            await engine.run(graph, ["u"])
        assert "coercing asset<core.string>" in str(err.value)
        assert "corrupt" in str(err.value)

    asyncio.run(scenario())


def test_declared_asset_inputs_still_receive_refs(tmp_path: Path) -> None:
    """A destination that accepts the asset AS-IS never coerces: an input
    declared asset_of(...) receives the AssetRef itself."""

    class Inspect(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="ta.inspect",
                display_name="Inspect",
                category="test",
                inputs=(InputSpec("ref", TypeExpr.asset_of(STRING)),),
                outputs=(OutputSpec("name", STRING),),
            )

        @classmethod
        def execute(cls, *, ref: AssetRef) -> Mapping[str, object]:
            assert isinstance(ref, AssetRef), type(ref)
            return cls.outputs(name=ref.name)

    async def scenario() -> None:
        resolver, wires = store_assets(tmp_path, "alpha")
        registry = TypeRegistry()
        register_core_types(registry)
        register_asset_type(registry, resolver)
        registry.register_asset_decoder(
            "core.string",
            provider_id="t.text@1",
            decode=lambda ref: ref.read_bytes().decode("utf-8"),  # type: ignore[union-attr]
        )
        nodes = (Inspect,)
        engine = Engine(
            schemas=build_schemas(nodes),
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
        )
        result = await engine.run(
            Graph(nodes={"i": GraphNode("ta.inspect", {"ref": asset_literal(wires[0])})}),
            ["i"],
        )
        assert result.outputs["i"]["name"].resolve() == "asset0.txt"

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Wire v12 negotiation


def test_wire_v12_round_trips_asset_schemas() -> None:
    class Loader(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="ta.load",
                display_name="Load",
                category="test",
                inputs=(InputSpec("source", TypeExpr.asset_of(IMAGE)),),
                outputs=(OutputSpec("images", TypeExpr.list_of(TypeExpr.asset_of(IMAGE))),),
            )

        @classmethod
        def execute(cls) -> Mapping[str, object]:  # pragma: no cover - schema only
            raise NotImplementedError

    wire = schema_to_wire(Loader.schema())
    assert wire["schemaVersion"] == 1
    assert schema_from_wire(wire) == Loader.schema()
