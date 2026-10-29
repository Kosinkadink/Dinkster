"""Worker-local graph compiler execution proofs."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from dinkster_graph import Graph, GraphNode
from dinkster_inference import (
    CompilerEmission,
    GeneratedNodeSpec,
    GraphCompilerDescriptor,
    GraphCompileView,
    InferenceGraphCompileError,
    InputRewrite,
    SamplerExtensionEntry,
    compile_inference_graph,
    graph_compiler_declaration,
    materialize_inference_generation,
    register_inference_types,
    release_inference_generation,
    write_sampler_catalog,
)
from dinkster_inference.graph_compilers import execute_graph_compilers
from dinkster_protocol import (
    GRAPH_COMPILE_ERROR_COMPILER_FAILURE,
    GRAPH_COMPILE_ERROR_GENERATED_LIMIT,
    GRAPH_COMPILE_ERROR_ID_COLLISION,
    GRAPH_COMPILE_ERROR_MALFORMED_REPLY,
    GRAPH_COMPILE_ERROR_NODE_LIMIT,
    GRAPH_COMPILE_ERROR_ORIGIN_COVERAGE,
    GRAPH_COMPILE_ERROR_PASS_LIMIT,
    GRAPH_COMPILE_ERROR_UNKNOWN_GENERATION,
    GRAPH_COMPILE_MAX_GENERATED_PER_PASS,
    GRAPH_COMPILE_MAX_NODES,
    GRAPH_COMPILE_RESULT_TYPE,
    GraphCompilerRegistrySnapshot,
    canonical_compile_reply_bytes,
    generated_node_id,
)
from dinkster_values import TypeRegistry
from test_engine_compile import _engine


def _graph() -> dict[str, object]:
    return {
        "nodes": {
            "source": {"nodeType": "test.source", "inputs": {"value": 1}},
            "sink": {"nodeType": "test.sink", "inputs": {"value": 0}},
        }
    }


def _empty(_view: GraphCompileView) -> CompilerEmission:
    return CompilerEmission()


def _descriptor(
    callback=_empty, *, compiler_id: str = "proof.compiler", order: int = 0
) -> GraphCompilerDescriptor:
    return GraphCompilerDescriptor(compiler_id, order, callback)


def _execute(*compilers: GraphCompilerDescriptor, cancelled=lambda: False):
    return execute_graph_compilers(
        "sha256:" + "a" * 64,
        _graph(),
        ("sink",),
        compilers,
        cancelled=cancelled,
    )


def _error(code: str, action) -> InferenceGraphCompileError:
    with pytest.raises(InferenceGraphCompileError) as caught:
        action()
    assert caught.value.error_name == code
    assert str(caught.value)
    return caught.value


def _write_pack(root: Path) -> SamplerExtensionEntry:
    (root / "c1_pack.py").write_text(
        "from dinkster_api.v1 import (CompilerEmission, GraphCompilerDescriptor, "
        "InferenceContribution, InputRewrite)\n"
        "def compile_graph(view):\n"
        "    generated = view.generated_id(('source',), 'adapter')\n"
        "    nodes = view.graph['nodes']\n"
        "    if generated in nodes:\n"
        "        return CompilerEmission()\n"
        "    spec = view.attempt_generated_node(\n"
        "        'adapter', ('source',), 'dev.image.gradient', "
        "{'width': 1, 'height': 1})\n"
        "    return CompilerEmission(\n"
        "        generated=(spec,),\n"
        "        rewrites=(InputRewrite('sink', 'value', generated, 'IMAGE'),),\n"
        "    )\n"
        "DESCRIPTOR = GraphCompilerDescriptor('proof.adapter', -2, compile_graph, "
        "(('mode', 'v1'),))\n"
        "def register():\n"
        "    return InferenceContribution(graph_compilers=(DESCRIPTOR,))\n"
        "def register_collision():\n"
        "    return InferenceContribution(graph_compilers=(DESCRIPTOR,))\n"
        "",
        encoding="utf-8",
    )
    return SamplerExtensionEntry("proof", "c1_pack:register")


def test_declaration_catalog_bidirectional_equality_and_cross_extension_collision(
    tmp_path: Path,
) -> None:
    sys.path.insert(0, str(tmp_path))
    catalog = tmp_path / "catalog.json"
    entry = _write_pack(tmp_path)
    first_key = "candidate:first"
    try:
        write_sampler_catalog(catalog, first_key, (entry,))
        first = materialize_inference_generation(first_key, catalog_path=catalog)
        declaration = first.graph_compiler_snapshot.contributions[0]
        assert declaration == graph_compiler_declaration(first.graph_compilers[0])
        assert dict(declaration.behavior_metadata) == {
            "config.mode": "v1",
            "contractVersion": 1,
            "order": -2,
        }
        compiled = compile_inference_graph(first_key, _graph(), ("sink",))
        generated = generated_node_id("proof.adapter", ("source",), "adapter")
        assert cast("dict[str, object]", compiled["graph"])["nodes"] == {
            "source": {"nodeType": "test.source", "inputs": {"value": 1}},
            "sink": {
                "nodeType": "test.sink",
                "inputs": {"value": {"$link": {"node": generated, "output": "IMAGE"}}},
            },
            generated: {
                "nodeType": "dev.image.gradient",
                "inputs": {"width": 1, "height": 1},
            },
        }
        checked_key = "candidate:checked"
        write_sampler_catalog(
            catalog,
            checked_key,
            (entry,),
            expected_extensions=first.extensions,
        )
        checked = materialize_inference_generation(checked_key, catalog_path=catalog)
        assert checked.extensions == first.extensions
        collision = (
            entry,
            SamplerExtensionEntry("proof2", "c1_pack:register_collision"),
        )
        write_sampler_catalog(catalog, "candidate:collision", collision)
        with pytest.raises(ValueError, match="globally unique"):
            materialize_inference_generation("candidate:collision", catalog_path=catalog)
    finally:
        for key in (first_key, "candidate:checked", "candidate:collision"):
            release_inference_generation(key)
        sys.modules.pop("c1_pack", None)
        sys.path.remove(str(tmp_path))


def test_unknown_generation_and_empty_registry_exact_identity(tmp_path: Path) -> None:
    _error(
        GRAPH_COMPILE_ERROR_UNKNOWN_GENERATION,
        lambda: compile_inference_graph("missing", _graph(), ("sink",)),
    )
    sys.path.insert(0, str(tmp_path))
    (tmp_path / "empty_pack.py").write_text(
        "from dinkster_api.v1 import InferenceContribution, SamplerDescriptor\n"
        "from dinkster_inference import NoiseKind\n"
        "def solve(denoiser, x, sigmas, **kwargs): return x\n"
        "def register(): return InferenceContribution(samplers=("
        "SamplerDescriptor('proof.empty', solve, NoiseKind.NONE),))\n",
        encoding="utf-8",
    )
    catalog = tmp_path / "catalog.json"
    key = "candidate:empty"
    graph = _graph()
    try:
        write_sampler_catalog(
            catalog, key, (SamplerExtensionEntry("proof", "empty_pack:register"),)
        )
        materialize_inference_generation(key, catalog_path=catalog)
        reply = compile_inference_graph(key, graph, ("sink",))
        assert reply["graph"] is graph
        assert reply["targets"] == ["sink"]
        assert reply["passCount"] == 0
        assert reply["attemptedGeneratedCounts"] == []
        assert reply["origins"] == []
    finally:
        release_inference_generation(key)
        sys.modules.pop("empty_pack", None)
        sys.path.remove(str(tmp_path))
    with pytest.raises(asyncio.CancelledError):
        execute_graph_compilers("key", graph, ("sink",), (), cancelled=lambda: True)


def test_retained_generations_are_exact_and_released_generation_refuses(tmp_path: Path) -> None:
    def write_pack(module: str, compiler_id: str, local_key: str) -> SamplerExtensionEntry:
        (tmp_path / f"{module}.py").write_text(
            "from dinkster_api.v1 import CompilerEmission, GraphCompilerDescriptor, "
            "InferenceContribution\n"
            "def compile_graph(view):\n"
            f"    node_id = view.generated_id(('source',), {local_key!r})\n"
            "    if node_id in view.graph['nodes']:\n"
            "        return CompilerEmission()\n"
            "    return CompilerEmission(generated=(view.attempt_generated_node(\n"
            f"        {local_key!r}, ('source',), 'test.generated', "
            f"{{'generation': {local_key!r}}}),))\n"
            f"DESCRIPTOR = GraphCompilerDescriptor({compiler_id!r}, 0, compile_graph)\n"
            "def register():\n"
            "    return InferenceContribution(graph_compilers=(DESCRIPTOR,))\n",
            encoding="utf-8",
        )
        return SamplerExtensionEntry(module, f"{module}:register")

    sys.path.insert(0, str(tmp_path))
    catalog = tmp_path / "catalog.json"
    first_key = "generation:first"
    second_key = "generation:second"
    first_compiler = "proof.first-generation"
    second_compiler = "proof.second-generation"
    try:
        write_sampler_catalog(
            catalog, first_key, (write_pack("generation_first", first_compiler, "first"),)
        )
        write_sampler_catalog(
            catalog, second_key, (write_pack("generation_second", second_compiler, "second"),)
        )
        first = materialize_inference_generation(first_key, catalog_path=catalog)
        second = materialize_inference_generation(second_key, catalog_path=catalog)
        assert tuple(item.id for item in first.graph_compiler_snapshot.contributions) == (
            first_compiler,
        )
        assert tuple(item.id for item in second.graph_compiler_snapshot.contributions) == (
            second_compiler,
        )

        first_reply = compile_inference_graph(first_key, _graph(), ("sink",))
        second_reply = compile_inference_graph(second_key, _graph(), ("sink",))
        first_id = generated_node_id(first_compiler, ("source",), "first")
        second_id = generated_node_id(second_compiler, ("source",), "second")
        first_nodes = cast(
            "Mapping[str, object]", cast("Mapping[str, object]", first_reply["graph"])["nodes"]
        )
        second_nodes = cast(
            "Mapping[str, object]", cast("Mapping[str, object]", second_reply["graph"])["nodes"]
        )
        assert first_id in first_nodes and second_id not in first_nodes
        assert second_id in second_nodes and first_id not in second_nodes
        assert (
            cast("list[dict[str, object]]", first_reply["origins"])[0]["compilerId"]
            == first_compiler
        )
        assert (
            cast("list[dict[str, object]]", second_reply["origins"])[0]["compilerId"]
            == second_compiler
        )

        release_inference_generation(first_key)
        _error(
            GRAPH_COMPILE_ERROR_UNKNOWN_GENERATION,
            lambda: compile_inference_graph(first_key, _graph(), ("sink",)),
        )
        retained = compile_inference_graph(second_key, _graph(), ("sink",))
        assert canonical_compile_reply_bytes(retained) == canonical_compile_reply_bytes(
            second_reply
        )
    finally:
        release_inference_generation(first_key)
        release_inference_generation(second_key)
        sys.modules.pop("generation_first", None)
        sys.modules.pop("generation_second", None)
        sys.path.remove(str(tmp_path))


def test_single_multiple_order_generated_identity_origin_and_canonical_bytes() -> None:
    calls: list[str] = []

    def add(view: GraphCompileView) -> CompilerEmission:
        calls.append(view.compiler_id)
        node_id = view.generated_id(("source",), "one")
        if node_id in cast("dict[str, object]", view.graph["nodes"]):
            return CompilerEmission()
        return CompilerEmission(
            generated=(view.attempt_generated_node("one", ("source",), "test.generated", {}),),
            rewrites=(InputRewrite("sink", "value", node_id, "out"),),
        )

    reply = _execute(_descriptor(_empty, compiler_id="proof.late", order=4), _descriptor(add))
    generated = generated_node_id("proof.compiler", ("source",), "one")
    assert calls == ["proof.compiler", "proof.compiler"]
    assert reply["passCount"] == 1
    assert reply["attemptedGeneratedCounts"] == [1]
    assert reply["origins"] == [
        {
            "nodeId": generated,
            "compilerId": "proof.compiler",
            "passIndex": 0,
            "sources": ["source"],
            "localKey": "one",
        }
    ]
    assert canonical_compile_reply_bytes(reply) == canonical_compile_reply_bytes(
        _execute(_descriptor(_empty, compiler_id="proof.late", order=4), _descriptor(add))
    )


def test_rewrite_only_and_multi_pass_fixpoint_accounting() -> None:
    def rewrite(view: GraphCompileView) -> CompilerEmission:
        sink = cast("dict[str, object]", cast("dict[str, object]", view.graph["nodes"])["sink"])
        if cast("dict[str, object]", sink["inputs"])["value"] == {
            "$link": {"node": "source", "output": "out"}
        }:
            return CompilerEmission()
        return CompilerEmission(rewrites=(InputRewrite("sink", "value", "source", "out"),))

    rewrite_reply = _execute(_descriptor(rewrite))
    assert rewrite_reply["passCount"] == 1
    assert rewrite_reply["attemptedGeneratedCounts"] == [0]

    def staged(view: GraphCompileView) -> CompilerEmission:
        nodes = cast("Mapping[str, object]", view.graph["nodes"])
        first = view.generated_id(("source",), "first")
        second = view.generated_id((first,), "second")
        if first not in nodes:
            return CompilerEmission(
                generated=(view.attempt_generated_node("first", ("source",), "test.generated", {}),)
            )
        if second not in nodes:
            return CompilerEmission(
                generated=(view.attempt_generated_node("second", (first,), "test.generated", {}),)
            )
        return CompilerEmission()

    staged_reply = _execute(_descriptor(staged))
    assert staged_reply["passCount"] == 2
    assert staged_reply["attemptedGeneratedCounts"] == [1, 1]


def test_host_mediated_attempt_accounting_and_callback_cancellation() -> None:
    def discard_then_rewrite(view: GraphCompileView) -> CompilerEmission:
        view.attempt_generated_node("discarded", ("source",), "test.generated", {})
        sink = cast("dict[str, object]", cast("dict[str, object]", view.graph["nodes"])["sink"])
        if cast("dict[str, object]", sink["inputs"])["value"] == {
            "$link": {"node": "source", "output": "out"}
        }:
            return CompilerEmission()
        return CompilerEmission(rewrites=(InputRewrite("sink", "value", "source", "out"),))

    discarded = _execute(_descriptor(discard_then_rewrite))
    assert discarded["passCount"] == 1
    assert discarded["attemptedGeneratedCounts"] == [1]
    assert discarded["origins"] == []

    def exact_limit(view: GraphCompileView) -> CompilerEmission:
        nodes = cast("Mapping[str, object]", view.graph["nodes"])
        generated = tuple(
            view.attempt_generated_node(str(index), ("source",), "test.generated", {})
            for index in range(GRAPH_COMPILE_MAX_GENERATED_PER_PASS)
            if view.generated_id(("source",), str(index)) not in nodes
        )
        return CompilerEmission(generated=generated)

    exact = _execute(_descriptor(exact_limit))
    assert exact["passCount"] == 1
    assert exact["attemptedGeneratedCounts"] == [GRAPH_COMPILE_MAX_GENERATED_PER_PASS]
    assert len(cast("list[object]", exact["origins"])) == GRAPH_COMPILE_MAX_GENERATED_PER_PASS

    host_polls = 0
    preprocessing_steps = 0

    def cancelled() -> bool:
        nonlocal host_polls
        host_polls += 1
        return host_polls >= 4

    def long_preprocessing(view: GraphCompileView) -> CompilerEmission:
        nonlocal preprocessing_steps
        while True:
            preprocessing_steps += 1
            view.check_cancelled()

    with pytest.raises(asyncio.CancelledError):
        _execute(_descriptor(long_preprocessing), cancelled=cancelled)
    assert preprocessing_steps == 3


def test_limits_collisions_malformed_failure_and_cancellation_retry() -> None:
    def duplicates(view: GraphCompileView) -> CompilerEmission:
        spec = view.attempt_generated_node("same", ("source",), "test.generated", {})
        return CompilerEmission(generated=(spec, spec))

    _error(
        GRAPH_COMPILE_ERROR_ID_COLLISION,
        lambda: _execute(_descriptor(duplicates)),
    )
    constructed = 0

    def too_many(view: GraphCompileView) -> CompilerEmission:
        nonlocal constructed
        specs: list[GeneratedNodeSpec] = []
        for index in range(GRAPH_COMPILE_MAX_GENERATED_PER_PASS + 1):
            try:
                spec = view.attempt_generated_node(str(index), ("source",), "test.generated", {})
            except Exception:
                return CompilerEmission(generated=tuple(specs))
            constructed += 1
            specs.append(spec)
        return CompilerEmission(generated=tuple(specs))

    _error(
        GRAPH_COMPILE_ERROR_GENERATED_LIMIT,
        lambda: _execute(_descriptor(too_many)),
    )
    assert constructed == GRAPH_COMPILE_MAX_GENERATED_PER_PASS
    huge = {
        "nodes": {
            str(index): {"nodeType": "test.node", "inputs": {}}
            for index in range(GRAPH_COMPILE_MAX_NODES + 1)
        }
    }
    _error(
        GRAPH_COMPILE_ERROR_NODE_LIMIT,
        lambda: execute_graph_compilers("key", huge, (), (_descriptor(),), cancelled=lambda: False),
    )

    def missing_source(view: GraphCompileView) -> CompilerEmission:
        return CompilerEmission(
            generated=(view.attempt_generated_node("missing", ("absent",), "test.generated", {}),)
        )

    _error(
        GRAPH_COMPILE_ERROR_ORIGIN_COVERAGE,
        lambda: _execute(_descriptor(missing_source)),
    )
    duplicate_rewrite = InputRewrite("sink", "value", "source", "out")
    _error(
        GRAPH_COMPILE_ERROR_MALFORMED_REPLY,
        lambda: _execute(
            _descriptor(
                lambda _view: CompilerEmission(rewrites=(duplicate_rewrite, duplicate_rewrite))
            )
        ),
    )
    _error(
        GRAPH_COMPILE_ERROR_MALFORMED_REPLY,
        lambda: _execute(
            _descriptor(
                lambda _view: CompilerEmission(
                    rewrites=(InputRewrite("sink", "absent", "source", "out"),)
                )
            )
        ),
    )

    def toggles(view: GraphCompileView) -> CompilerEmission:
        sink = cast("dict[str, object]", cast("dict[str, object]", view.graph["nodes"])["sink"])
        value = cast("dict[str, object]", sink["inputs"])["value"]
        source = "source" if value != {"$link": {"node": "source", "output": "a"}} else "sink"
        return CompilerEmission(rewrites=(InputRewrite("sink", "value", source, "a"),))

    _error(GRAPH_COMPILE_ERROR_PASS_LIMIT, lambda: _execute(_descriptor(toggles)))
    _error(
        GRAPH_COMPILE_ERROR_MALFORMED_REPLY,
        lambda: _execute(_descriptor(lambda _view: cast(CompilerEmission, object()))),
    )

    def raises(_view: GraphCompileView) -> CompilerEmission:
        raise LookupError("boom")

    _error(GRAPH_COMPILE_ERROR_COMPILER_FAILURE, lambda: _execute(_descriptor(raises)))

    spurious_calls = 0

    def spurious_cancel(_view: GraphCompileView) -> CompilerEmission:
        nonlocal spurious_calls
        spurious_calls += 1
        if spurious_calls == 1:
            raise asyncio.CancelledError
        return CompilerEmission()

    descriptor = _descriptor(spurious_cancel)
    _error(GRAPH_COMPILE_ERROR_COMPILER_FAILURE, lambda: _execute(descriptor))
    healthy = _execute(descriptor)
    assert healthy["passCount"] == 0
    assert spurious_calls == 2
    _error(
        GRAPH_COMPILE_ERROR_COMPILER_FAILURE,
        lambda: _execute(
            _descriptor(
                lambda _view: (_ for _ in ()).throw(
                    InferenceGraphCompileError(GRAPH_COMPILE_ERROR_UNKNOWN_GENERATION, "spoof")
                )
            )
        ),
    )
    _error(
        GRAPH_COMPILE_ERROR_MALFORMED_REPLY,
        lambda: _execute(
            _descriptor(
                lambda _view: CompilerEmission(
                    generated=(GeneratedNodeSpec("raw", ("source",), "test.generated", {}),)
                )
            )
        ),
    )

    def forged(view: GraphCompileView) -> CompilerEmission:
        issued = view.attempt_generated_node("issued", ("source",), "test.generated", {})
        return CompilerEmission(generated=(replace(issued, local_key="forged"),))

    _error(GRAPH_COMPILE_ERROR_MALFORMED_REPLY, lambda: _execute(_descriptor(forged)))

    issued_elsewhere: list[GeneratedNodeSpec] = []

    def issue_and_discard(view: GraphCompileView) -> CompilerEmission:
        issued_elsewhere.append(
            view.attempt_generated_node("cross-view", ("source",), "test.generated", {})
        )
        return CompilerEmission()

    def emit_from_other_view(_view: GraphCompileView) -> CompilerEmission:
        return CompilerEmission(generated=(issued_elsewhere[-1],))

    _error(
        GRAPH_COMPILE_ERROR_MALFORMED_REPLY,
        lambda: _execute(
            _descriptor(issue_and_discard, compiler_id="proof.first"),
            _descriptor(emit_from_other_view, compiler_id="proof.second"),
        ),
    )

    def malformed_link(view: GraphCompileView) -> CompilerEmission:
        return CompilerEmission(
            generated=(
                view.attempt_generated_node(
                    "bad-link",
                    ("source",),
                    "test.generated",
                    {"value": {"$link": {"node": "source"}}},
                ),
            )
        )

    _error(GRAPH_COMPILE_ERROR_MALFORMED_REPLY, lambda: _execute(_descriptor(malformed_link)))

    original = _graph()

    def extra_link_key(view: GraphCompileView) -> CompilerEmission:
        return CompilerEmission(
            generated=(
                view.attempt_generated_node(
                    "bad-link-extra",
                    ("source",),
                    "test.generated",
                    {"value": {"$link": {"node": "source", "output": "out", "extra": 1}}},
                ),
            )
        )

    _error(
        GRAPH_COMPILE_ERROR_MALFORMED_REPLY,
        lambda: execute_graph_compilers(
            "key", original, ("sink",), (_descriptor(extra_link_key),), cancelled=lambda: False
        ),
    )
    assert original == _graph()

    polls = 0

    def cancelled() -> bool:
        nonlocal polls
        polls += 1
        return polls >= 5

    def adds(_view: GraphCompileView) -> CompilerEmission:
        return CompilerEmission(
            generated=(_view.attempt_generated_node("same", ("source",), "test.generated", {}),)
        )

    with pytest.raises(asyncio.CancelledError):
        _execute(_descriptor(adds), cancelled=cancelled)
    retry = _execute(
        _descriptor(
            lambda view: (
                CompilerEmission()
                if view.generated_id(("source",), "same")
                in cast("dict[str, object]", view.graph["nodes"])
                else CompilerEmission(
                    generated=(
                        view.attempt_generated_node("same", ("source",), "test.generated", {}),
                    )
                )
            )
        )
    )
    assert retry["passCount"] == 1


def test_engine_validator_round_trip_uses_existing_public_execution_seam() -> None:
    engine = _engine()
    base = engine.pin_execution()
    calls = 0

    def compiler(view: GraphCompileView) -> CompilerEmission:
        node_id = view.generated_id(("g",), "extra")
        if node_id in cast("dict[str, object]", view.graph["nodes"]):
            return CompilerEmission()
        return CompilerEmission(
            generated=(
                view.attempt_generated_node(
                    "extra",
                    ("g",),
                    "dev.image.gradient",
                    {"width": 1, "height": 1},
                ),
            )
        )

    descriptor = _descriptor(compiler)

    async def transport(generation_key, graph_wire, targets):
        nonlocal calls
        calls += 1
        semantic = execute_graph_compilers(
            generation_key,
            graph_wire,
            targets,
            (descriptor,),
            cancelled=lambda: False,
        )
        return {
            "type": GRAPH_COMPILE_RESULT_TYPE,
            "requestId": "proof-request",
            "blobs": [],
            **semantic,
        }

    runtime = replace(
        base,
        graph_compiler_registry=GraphCompilerRegistrySnapshot(
            (graph_compiler_declaration(descriptor),)
        ),
        graph_compile_transport=transport,
    )
    compiled = asyncio.run(
        engine.compile_for_execution(
            Graph({"g": GraphNode("dev.image.gradient", {"width": 2, "height": 2})}),
            ("g",),
            execution=runtime,
        )
    )
    generated = generated_node_id("proof.compiler", ("g",), "extra")
    assert calls == 1
    assert set(compiled.graph.nodes) == {"g", generated}
    assert set(compiled.origins) == {generated}


def test_register_inference_types_is_idempotent() -> None:
    registry = TypeRegistry()
    first = register_inference_types(registry)
    second = register_inference_types(registry)
    expected_ids = (
        "dinkster.conditioning",
        "dinkster.latent",
        "dinkster.control",
        "dinkster.model",
        "dinkster.clip",
        "dinkster.clip-vision",
        "dinkster.vae",
        "dinkster.sampler",
        "dinkster.sigmas",
        "dinkster.guider",
        "dinkster.noise",
    )
    assert first == second == tuple(registry.spec(type_id) for type_id in expected_ids)
    assert registry.type_ids() == expected_ids
