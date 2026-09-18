from __future__ import annotations

import asyncio
import gc
import json
import os
from pathlib import Path

from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link, TypedLiteral, validate
from dinkster_schema import (
    InputSpec,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    plan_type_equivalence,
    schema_to_wire,
)
from dinkster_values import (
    EncodedPayload,
    TypeRegistry,
    Value,
    ValueMeta,
    decode_image_array,
    encode_image_array,
    image_array_fingerprint,
    image_array_meta,
    prepare_image_array_encoding,
    register_core_types,
)
from dinkster_workers import BoundaryDiagnostic, IsolatedWorker, RoutingWorker

from dinkster.comfy_compose import register_comfy_host_types

TESTS_DIR = Path(__file__).parent
COMPAT_IMAGE = "comfy.IMAGE"
COMPAT_MASK = "comfy.MASK"
NATIVE_IMAGE = "dinkster.image"
NATIVE_MASK = "dinkster.mask"
PROVIDER = "test.compat-wire@1"


def _register_image_type(registry: TypeRegistry, type_id: str) -> None:
    registry.register(
        type_id,
        encode=encode_image_array,
        decode=decode_image_array,
        prepare_buffer_encoding=prepare_image_array_encoding,
        fingerprint=image_array_fingerprint(type_id),
        meta=image_array_meta,
    )


def _image_registry(*, with_equivalences: bool = True) -> TypeRegistry:
    registry = TypeRegistry()
    register_core_types(registry)
    for type_id in (COMPAT_IMAGE, COMPAT_MASK, NATIVE_IMAGE, NATIVE_MASK):
        _register_image_type(registry, type_id)
    if with_equivalences:
        registry.register_type_equivalence(COMPAT_IMAGE, NATIVE_IMAGE, provider_id=PROVIDER)
        registry.register_type_equivalence(COMPAT_MASK, NATIVE_MASK, provider_id=PROVIDER)
    return registry


def test_registry_type_equivalence_is_pairwise_symmetric_and_staged() -> None:
    registry = TypeRegistry()
    registry.register("compat.value", encode=lambda value: str(value).encode(), decode=int)
    registry.register("native.value", encode=lambda value: str(value).encode(), decode=int)
    registry.register("other.value")

    registry.register_type_equivalence(
        "compat.value", "native.value", provider_id="test.value-wire@1"
    )
    registry.register_type_equivalence(
        "native.value", "compat.value", provider_id="test.value-wire@1"
    )
    assert registry.equivalent_type("compat.value") == "native.value"
    assert registry.equivalent_type("native.value") == "compat.value"
    assert registry.equivalent_type("other.value") is None

    copied = registry.copy()
    published = TypeRegistry()
    published.replace_from(copied)
    assert published.equivalent_type("compat.value") == "native.value"


def test_registry_type_equivalence_rejects_ambiguous_or_invalid_pairs() -> None:
    registry = TypeRegistry()
    for type_id in ("compat.value", "native.value", "other.value"):
        registry.register(type_id)

    for left, right, message in (
        ("compat.value", "compat.value", "different"),
        ("list<compat.value>", "native.value", "atom"),
    ):
        try:
            registry.register_type_equivalence(left, right, provider_id=PROVIDER)
        except (KeyError, ValueError) as exc:
            assert message in str(exc)
        else:
            raise AssertionError("invalid type equivalence was accepted")

    registry.register_type_equivalence("compat.value", "native.value", provider_id=PROVIDER)
    try:
        registry.register_type_equivalence("compat.value", "other.value", provider_id=PROVIDER)
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("ambiguous type equivalence was accepted")


def test_bridge_equivalent_reuses_encoded_bytes_and_has_target_decoder() -> None:
    registry = TypeRegistry()
    source_encodes: list[object] = []

    def encode_source(value: object) -> bytes:
        source_encodes.append(value)
        return f"wire:{value}".encode()

    registry.register("compat.value", encode=encode_source, decode=lambda data: data.decode())
    registry.register(
        "native.value",
        encode=lambda value: f"wire:{value}".encode(),
        decode=lambda data: int(data.decode().removeprefix("wire:")),
    )
    registry.register_type_equivalence(
        "compat.value", "native.value", provider_id="test.value-wire@1"
    )
    raw = registry.wrap("compat.value", 8)
    source_encodes.clear()
    bridged_raw = registry.bridge_equivalent(raw, "native.value")
    assert source_encodes == [8]
    assert bridged_raw.resolve() == 8

    source_encodes.clear()
    source = Value(
        type_id="compat.value",
        fingerprint="source-fingerprint",
        meta=ValueMeta({"kept": True}),
        payload=EncodedPayload("compat.value", b"wire:7", None, "shm"),
    )

    bridged = registry.bridge_equivalent(source, "native.value")
    assert source_encodes == []
    assert bridged.type_id == "native.value"
    assert bridged.meta == source.meta
    assert bridged.resolve() == 7
    assert isinstance(bridged.payload, EncodedPayload)
    assert bridged.payload.transport == "shm"
    assert bridged.payload.data == b"wire:7"
    assert registry.bridge_equivalent(source, "native.value").fingerprint == bridged.fingerprint


def test_restamped_encoded_buffer_keeps_shared_memory_alive() -> None:
    registry = TypeRegistry()
    registry.register(
        "compat.value", encode=lambda value: str(value).encode(), decode=lambda data: data.decode()
    )
    registry.register(
        "native.value", encode=lambda value: str(value).encode(), decode=lambda data: data.decode()
    )
    registry.register_type_equivalence(
        "compat.value", "native.value", provider_id="test.value-wire@1"
    )
    released: list[bool] = []
    source_payload = EncodedPayload.from_buffer(
        "compat.value",
        memoryview(b"shared bytes"),
        lambda data: data.decode(),
        "shm",
        lambda: released.append(True),
    )
    source = Value(
        type_id="compat.value",
        fingerprint="source-fingerprint",
        meta=ValueMeta(),
        payload=source_payload,
    )
    bridged = registry.bridge_equivalent(source, "native.value")

    del source, source_payload
    gc.collect()
    assert released == []
    assert isinstance(bridged.payload, EncodedPayload)
    with bridged.payload.borrow_data() as view:
        assert bytes(view) == b"shared bytes"
    assert released == []

    del bridged
    gc.collect()
    assert released == [True]


def test_type_equivalence_planning_preserves_exact_spelling_and_is_explicit() -> None:
    registry = _image_registry()
    native = TypeExpr.variable("T", allowed=(NATIVE_IMAGE, NATIVE_MASK))
    assert plan_type_equivalence(COMPAT_IMAGE, native, registry) == NATIVE_IMAGE
    assert plan_type_equivalence(NATIVE_IMAGE, native, registry) is None
    assert (
        plan_type_equivalence(
            COMPAT_IMAGE,
            TypeExpr.wildcard(),
            registry,
        )
        is None
    )

    without_pair = _image_registry(with_equivalences=False)
    assert plan_type_equivalence(COMPAT_IMAGE, native, without_pair) is None


def test_graph_validation_uses_registered_equivalence_for_links_and_literals() -> None:
    source = NodeSchema(
        "compat.source", outputs=(OutputSpec("image", TypeExpr.concrete(COMPAT_IMAGE)),)
    )
    sink = NodeSchema(
        "native.sink",
        inputs=(
            InputSpec(
                "image",
                TypeExpr.variable("input_type", allowed=(NATIVE_IMAGE, NATIVE_MASK)),
            ),
        ),
    )
    schemas = {source.node_type: source, sink.node_type: sink}
    linked = Graph(
        nodes={
            "source": GraphNode(source.node_type, {}),
            "sink": GraphNode(sink.node_type, {"image": Link("source", "image")}),
        }
    )
    stamped = Graph(
        nodes={
            "sink": GraphNode(
                sink.node_type,
                {"image": TypedLiteral(COMPAT_IMAGE, [[[[0.0, 0.0, 0.0]]]])},
            )
        }
    )

    registry = _image_registry()
    assert validate(linked, schemas, ["sink"], known_types=registry) == []
    assert validate(stamped, schemas, ["sink"], known_types=registry) == []

    without_pair = _image_registry(with_equivalences=False)
    assert [
        diag.code for diag in validate(linked, schemas, ["sink"], known_types=without_pair)
    ] == ["type-mismatch"]
    assert [
        diag.code for diag in validate(stamped, schemas, ["sink"], known_types=without_pair)
    ] == ["type-mismatch"]


def test_comfy_host_registration_activates_only_available_native_pairs() -> None:
    compat_only = TypeRegistry()
    register_core_types(compat_only)
    register_comfy_host_types(compat_only)
    assert compat_only.equivalent_type(COMPAT_IMAGE) is None
    assert compat_only.equivalent_type(COMPAT_MASK) is None

    registry = TypeRegistry()
    register_core_types(registry)
    _register_image_type(registry, NATIVE_IMAGE)
    _register_image_type(registry, NATIVE_MASK)
    register_comfy_host_types(registry)
    register_comfy_host_types(registry)
    assert registry.equivalent_type(COMPAT_IMAGE) == NATIVE_IMAGE
    assert registry.equivalent_type(NATIVE_IMAGE) == COMPAT_IMAGE
    assert registry.equivalent_type(COMPAT_MASK) == NATIVE_MASK
    assert registry.equivalent_type(NATIVE_MASK) == COMPAT_MASK


def _write_manifest(root: Path, *, compat: bool) -> Path:
    root.mkdir()
    pack = "compat-equivalence-test" if compat else "native-equivalence-test"
    namespace = "comfy" if compat else "dinkster"
    module = "type_equivalence_compat_nodes" if compat else "type_equivalence_native_nodes"
    manifest = root / "dinkster-pack.toml"
    manifest.write_text(
        f'[pack]\nname = "{pack}"\nnamespaces = ["{namespace}"]\n\n'
        f'[pack.entry]\nnodes = "{module}:NODES"\n'
        f'types = "{module}:register_types"\n',
        encoding="utf-8",
    )
    return manifest


def test_equivalent_image_and_mask_atoms_cross_isolated_workers_both_directions(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        _register_image_type(registry, NATIVE_IMAGE)
        _register_image_type(registry, NATIVE_MASK)
        register_comfy_host_types(registry)
        compat_diagnostics: list[BoundaryDiagnostic] = []
        native_diagnostics: list[BoundaryDiagnostic] = []
        extra_env = {
            "PYTHONPATH": os.pathsep.join((str(TESTS_DIR), os.environ.get("PYTHONPATH", "")))
        }
        compat = IsolatedWorker(
            _write_manifest(tmp_path / "compat", compat=True),
            registry,
            extra_env=extra_env,
            shm_threshold=1,
            on_diagnostic=compat_diagnostics.append,
        )
        native = IsolatedWorker(
            _write_manifest(tmp_path / "native", compat=False),
            registry,
            extra_env=extra_env,
            shm_threshold=1,
            on_diagnostic=native_diagnostics.append,
        )
        await asyncio.gather(compat.start(), native.start())
        try:
            schemas = {**compat.schemas, **native.schemas}
            resize_wire = json.dumps(
                schema_to_wire(schemas["dinkster.image.resize"]), sort_keys=True
            )
            assert COMPAT_IMAGE not in resize_wire
            assert COMPAT_MASK not in resize_wire
            routes = {node_type: compat for node_type in compat.schemas}
            routes.update({node_type: native for node_type in native.schemas})
            engine = Engine(
                schemas=schemas,
                registry=registry,
                worker=RoutingWorker(routes),
                cache=MemoryLRUCache(),
            )
            graph = Graph(
                nodes={
                    "compat_image": GraphNode("comfy.custom.ImageProducer", {}),
                    "compat_mask": GraphNode("comfy.custom.MaskProducer", {}),
                    "resize_image": GraphNode(
                        "dinkster.image.resize",
                        {
                            "image": Link("compat_image", "image"),
                            "target.width": 2,
                            "target.height": 2,
                            "interpolation": "nearest-exact",
                        },
                        slot_variants={
                            "target": "dimensions",
                            "mode": "stretch",
                            "divisibility": "none",
                        },
                    ),
                    "resize_mask": GraphNode(
                        "dinkster.image.resize",
                        {
                            "image": Link("compat_mask", "mask"),
                            "target.width": 2,
                            "target.height": 2,
                            "interpolation": "nearest-exact",
                        },
                        slot_variants={
                            "target": "dimensions",
                            "mode": "stretch",
                            "divisibility": "none",
                        },
                    ),
                    "native_image_sink": GraphNode(
                        "dinkster.test.image_consumer",
                        {"image": Link("resize_image", "image")},
                    ),
                    "native_mask_sink": GraphNode(
                        "dinkster.test.mask_consumer",
                        {"mask": Link("resize_mask", "image")},
                    ),
                    "native_image": GraphNode("dinkster.test.image_producer", {}),
                    "native_mask": GraphNode("dinkster.test.mask_producer", {}),
                    "compat_image_sink": GraphNode(
                        "comfy.custom.ImageConsumer",
                        {"image": Link("native_image", "image")},
                    ),
                    "compat_mask_sink": GraphNode(
                        "comfy.custom.MaskConsumer",
                        {"mask": Link("native_mask", "mask")},
                    ),
                }
            )
            targets = [
                "resize_image",
                "resize_mask",
                "native_image_sink",
                "native_mask_sink",
                "compat_image_sink",
                "compat_mask_sink",
            ]
            assert validate(graph, schemas, targets, known_types=registry) == []
            result = await engine.run(graph, targets)

            assert result.outputs["resize_image"]["image"].type_id == NATIVE_IMAGE
            assert result.outputs["resize_mask"]["image"].type_id == NATIVE_MASK
            assert result.outputs["native_image_sink"]["shape"].resolve() == "1x2x2x3"
            assert result.outputs["native_mask_sink"]["shape"].resolve() == "1x2x2"
            assert result.outputs["compat_image_sink"]["shape"].resolve() == "1x2x3x3"
            assert result.outputs["compat_mask_sink"]["shape"].resolve() == "1x2x3"

            native_by_node = {diagnostic.node_id: diagnostic for diagnostic in native_diagnostics}
            compat_by_node = {diagnostic.node_id: diagnostic for diagnostic in compat_diagnostics}
            for node_id, expected in (
                ("resize_image", NATIVE_IMAGE),
                ("resize_mask", NATIVE_MASK),
                ("native_image_sink", NATIVE_IMAGE),
                ("native_mask_sink", NATIVE_MASK),
            ):
                edge = native_by_node[node_id].inputs[0]
                assert edge.type_id == expected
                assert edge.transport == "shm"
                assert edge.reused is True
            assert compat_by_node["compat_image_sink"].inputs[0].type_id == COMPAT_IMAGE
            assert compat_by_node["compat_mask_sink"].inputs[0].type_id == COMPAT_MASK
            assert compat_by_node["compat_image_sink"].inputs[0].transport == "shm"
            assert compat_by_node["compat_mask_sink"].inputs[0].transport == "shm"
            assert compat_by_node["compat_image_sink"].inputs[0].reused is True
            assert compat_by_node["compat_mask_sink"].inputs[0].reused is True
        finally:
            await asyncio.gather(compat.close(), native.close())

    asyncio.run(scenario())
