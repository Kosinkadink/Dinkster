import asyncio
import io
import json
import struct
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_engine.events import EngineEvent
from dinkster_graph import Graph, GraphNode, Link
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
    schema_from_wire,
    schema_signature,
    schema_to_wire,
)
from dinkster_schema.media import media_diagnostics
from dinkster_server.events import engine_event_to_wire
from dinkster_values import (
    TypeRegistry,
    Value,
    ValueMeta,
    annotate_image,
    annotate_mask,
    decode_image_array,
    encode_image_array,
    image_array_fingerprint,
    image_array_meta,
    make_list_value,
    mask_array_meta,
    media_semantics,
    prepare_image_array_encoding,
    validate_image_encoded,
)
from dinkster_values.model import PyObjPayload
from dinkster_workers import InProcessWorker

IMAGE = TypeExpr.concrete("comfy.IMAGE")
MASK = TypeExpr.concrete("comfy.MASK")


def test_media_array_metadata_does_not_copy_tensor_pixels() -> None:
    class DeviceTensor:
        shape = (1, 32, 32, 4)
        dtype = "torch.bfloat16"

        def detach(self):
            raise AssertionError("metadata inspection must not read tensor pixels")

    image = image_array_meta(DeviceTensor())
    assert image["shape"] == (1, 32, 32, 4)
    assert image["dtype"] == "float32"
    assert image["channels"] == {"layout": "rgba", "alpha": "straight"}
    assert mask_array_meta(DeviceTensor())["polarity"] == "coverage"


def test_media_policy_schema_roundtrip_and_identity() -> None:
    schema = NodeSchema(
        node_type="test.media",
        inputs=(InputSpec("image", IMAGE, alpha_policy="create_if_missing"),),
        outputs=(OutputSpec("mask", MASK, mask_polarity="transparency", mask_semantic="alpha"),),
    )
    wire = schema_to_wire(schema)
    assert schema_from_wire(wire) == schema
    interface = cast("list[dict[str, Any]]", wire["interface"])
    assert interface[0]["alphaPolicy"] == "create_if_missing"
    assert interface[1]["maskPolarity"] == "transparency"
    assert interface[1]["maskSemantic"] == "alpha"
    default = replace(schema, inputs=(InputSpec("image", IMAGE),))
    assert schema_signature(schema) != schema_signature(default)
    with pytest.raises(ValueError, match="media policies.*40"):
        schema_to_wire(schema, wire_version=39)
    wire["schemaVersion"] = 39
    with pytest.raises(ValueError, match="media policies require"):
        schema_from_wire(wire)


@pytest.mark.parametrize("field", ["alpha_policy", "mask_polarity", "mask_semantic"])
@pytest.mark.parametrize("spec", [InputSpec, OutputSpec])
def test_invalid_media_policy_is_rejected(field, spec) -> None:
    with pytest.raises(ValueError, match="unknown"):
        spec("image", IMAGE, **{field: "invalid"})


def _value(type_id: str, **meta: object) -> Value:
    return Value(type_id, "fixture", ValueMeta(meta), PyObjPayload(None))


def test_alpha_and_mask_checks_are_metadata_only_and_declarations_control_loss() -> None:
    rgba = _value("comfy.IMAGE", shape=(1, 2, 2, 4))
    rgb = _value("comfy.IMAGE", shape=(1, 2, 2, 3))
    mask = _value("comfy.MASK", polarity="transparency", semantic="alpha")
    schema = NodeSchema(
        node_type="test.media",
        inputs=(InputSpec("image", IMAGE), InputSpec("mask", MASK, mask_polarity="coverage")),
        outputs=(OutputSpec("image", IMAGE),),
    )
    inputs = {"image": rgba, "mask": mask}
    diagnostics = media_diagnostics(schema, inputs, {"image": rgb})
    assert diagnostics == [
        {
            "code": "mask_polarity_mismatch",
            "inputId": "mask",
            "expected": "coverage",
            "actual": "transparency",
        },
        {"code": "alpha_dropped", "outputId": "image", "inputIds": ["image"]},
    ]
    declared_drop = replace(schema, outputs=(OutputSpec("image", IMAGE, alpha_policy="drop"),))
    assert media_diagnostics(declared_drop, inputs, {"image": rgb}) == diagnostics[:1]
    assert media_diagnostics(schema, inputs, {"image": rgba}) == diagnostics[:1]
    batched = make_list_value("comfy.IMAGE", (rgba, rgb))
    assert media_diagnostics(schema, inputs, {"image": batched}) == diagnostics


def test_mixed_alpha_passthrough_is_not_reported_as_loss() -> None:
    rgba = replace(_value("comfy.IMAGE", shape=(1, 2, 2, 4)), fingerprint="rgba")
    rgb = replace(_value("comfy.IMAGE", shape=(1, 2, 2, 3)), fingerprint="rgb")
    schema = NodeSchema(
        node_type="test.passthrough",
        inputs=(InputSpec("a", IMAGE), InputSpec("b", IMAGE)),
        outputs=(OutputSpec("a", IMAGE), OutputSpec("b", IMAGE)),
    )
    assert media_diagnostics(schema, {"a": rgba, "b": rgb}, {"a": rgba, "b": rgb}) == []
    mixed = make_list_value("comfy.IMAGE", (rgba, rgb))
    assert media_diagnostics(schema, {"a": mixed}, {"a": mixed}) == []
    dropped = replace(rgb, fingerprint="dropped")
    assert media_diagnostics(schema, {"a": rgba, "b": rgb}, {"a": dropped, "b": rgb}) == [
        {"code": "alpha_dropped", "outputId": "a", "inputIds": ["a"]}
    ]


class AlphaSource(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(node_type="test.alpha-source", outputs=(OutputSpec("image", IMAGE),))

    @classmethod
    def execute(cls):
        return cls.outputs(image=np.ones((1, 2, 2, 4), dtype=np.float32))


class AlphaDrop(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="test.alpha-drop",
            inputs=(InputSpec("image", IMAGE),),
            outputs=(OutputSpec("image", IMAGE),),
        )

    @classmethod
    def execute(cls, image):
        return cls.outputs(image=image[..., :3])


def test_unexpected_alpha_loss_is_nonblocking_and_replayed_on_cache_hits() -> None:
    async def scenario():
        registry = TypeRegistry()
        registry.register(
            "comfy.IMAGE",
            encode=encode_image_array,
            decode=decode_image_array,
            meta=image_array_meta,
        )
        nodes = (AlphaSource, AlphaDrop)
        events: list[EngineEvent] = []
        engine = Engine(
            schemas=build_schemas(nodes),
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
            on_event=events.append,
        )
        graph = Graph(
            nodes={
                "source": GraphNode("test.alpha-source"),
                "drop": GraphNode("test.alpha-drop", {"image": Link("source", "image")}),
            }
        )
        for cached in (False, True):
            events.clear()
            result = await engine.run(graph, ["drop"])
            assert np.asarray(result.outputs["drop"]["image"].resolve()).shape[-1] == 3
            assert ("drop" in result.cached) == cached
            (diagnostic,) = [event for event in events if event.kind == "value_diagnostics"]
            wire = engine_event_to_wire(diagnostic, client_id="client", job_id="job")
            assert wire["nodeId"] == "drop"
            assert wire["detail"] == {
                "diagnostics": [
                    {
                        "code": "alpha_dropped",
                        "nodeId": "drop",
                        "outputId": "image",
                        "inputIds": ["image"],
                    }
                ]
            }
            records = cast(list[dict[str, Any]], diagnostic.detail["diagnostics"])
            records[0]["inputIds"].append("observer-edit")
            assert result.outputs["drop"]["image"].meta.get("valueDiagnostics") == [
                {"code": "alpha_dropped", "outputId": "image", "inputIds": ["image"]}
            ]

    asyncio.run(scenario())


@pytest.mark.parametrize("mask", [False, True])
def test_semantic_codec_and_shared_memory_roundtrip(mask: bool, monkeypatch) -> None:
    from multiprocessing.shared_memory import SharedMemory

    import dinkster_workers.boundary as boundary
    from dinkster_workers.boundary import ValueCodec, release_segment

    # Both handles belong to this process, so its creator must stay tracked.
    monkeypatch.setattr(boundary, "_attach_segment", lambda name: SharedMemory(name=name))
    array = np.full((1, 2, 3) if mask else (1, 2, 3, 4), 0.5, dtype=np.float32)
    annotated = (
        annotate_mask(array, polarity="transparency", semantic="alpha")
        if mask
        else annotate_image(
            array,
            alpha="premultiplied",
            color={"primaries": 9, "transfer": 16, "matrix": 9, "range": 1, "bit_depth": 10},
        )
    )
    type_id = "comfy.MASK" if mask else "comfy.IMAGE"
    metadata = mask_array_meta if mask else image_array_meta
    encoded = encode_image_array(annotated)
    restored = decode_image_array(encoded)
    np.testing.assert_array_equal(restored, array)
    assert metadata(restored) == metadata(annotated)
    fingerprint = image_array_fingerprint(type_id)
    assert fingerprint(restored) == fingerprint(annotated) != fingerprint(array)
    prepared = prepare_image_array_encoding(annotated)
    buffer = bytearray(prepared.size)
    assert prepared.write(memoryview(buffer)) == len(encoded)
    assert bytes(buffer) == encoded
    registry = TypeRegistry()
    registry.register(
        type_id,
        encode=encode_image_array,
        decode=decode_image_array,
        fingerprint=fingerprint,
        meta=metadata,
        prepare_buffer_encoding=prepare_image_array_encoding,
        validate_encoded=validate_image_encoded,
        validate_encoded_buffer=validate_image_encoded,
    )
    codec = ValueCodec(registry, shm_threshold=1)
    blobs, segments = [], []
    wire, _ = codec.encode(registry.wrap(type_id, annotated), blobs, segments)
    try:
        consumed: list[str] = []
        received, stat = codec.decode(wire, blobs, consumed)
        assert stat.transport == "shm"
        assert consumed == [segment.name for segment in segments]
        received_meta = dict(received.meta.entries)
        expected_meta = dict(metadata(annotated))
        assert received_meta.keys() == expected_meta.keys()
        assert received_meta["cost"] == {"ram": len(encoded)}
        assert json.dumps(
            {key: value for key, value in received_meta.items() if key != "cost"}, sort_keys=True
        ) == json.dumps(
            {key: value for key, value in expected_meta.items() if key != "cost"}, sort_keys=True
        )
        assert media_semantics(received.resolve()) == media_semantics(annotated)
    finally:
        for segment in segments:
            release_segment(segment)


def test_default_semantics_do_not_change_legacy_bytes_or_identity() -> None:
    array = np.zeros((1, 2, 3, 4), dtype=np.float32)
    annotated = annotate_image(
        array, alpha="straight", color={"primaries": 1, "transfer": 13, "range": 2}
    )
    legacy = io.BytesIO()
    np.save(legacy, array, allow_pickle=False)
    assert encode_image_array(annotated) == encode_image_array(array) == legacy.getvalue()
    assert image_array_fingerprint("comfy.IMAGE")(annotated) == image_array_fingerprint(
        "comfy.IMAGE"
    )(array)
    assert media_semantics(array) == {}


def test_image_and_layer_color_preserve_unknown_ffmpeg_enums() -> None:
    from dinkster_image_document.document import empty_document
    from dinkster_values.image_codec import image_color

    color = {"primaries": 300, "transfer": 301, "range": 302, "matrix": 303, "bit_depth": 12}
    image = annotate_image(np.zeros((1, 1, 1, 3), np.float32), color=color)
    assert image_array_meta(decode_image_array(encode_image_array(image)))["color"] == color
    assert empty_document(1, 1, color=color).to_record()["canvas"]["color"] == color
    assert empty_document(1, 1).to_record()["canvas"]["color"] == image_color()


@pytest.mark.parametrize("image_first", [False, True])
def test_pack_registration_order_preserves_mask_contract(image_first: bool) -> None:
    from dinkster_nodes_image import register_image_types
    from dinkster_nodes_media_io import register_media_types

    registry = TypeRegistry()
    registrations = [register_image_types, register_media_types]
    for register in registrations if image_first else reversed(registrations):
        register(registry)
    mask = annotate_mask(np.zeros((1, 1, 1), np.float32), polarity="transparency", semantic="alpha")
    value = registry.wrap("dinkster.mask", mask)
    assert value.meta.get("polarity") == "transparency"
    assert value.meta.get("semantic") == "alpha"
    assert "channels" not in value.meta.entries
    for type_id in ("dinkster.mask", "dinkster.image"):
        assert registry.spec(type_id).validate_encoded is not None
        assert registry.spec(type_id).validate_encoded_buffer is not None


@pytest.mark.parametrize(
    "metadata",
    [
        {"alpha": "premultiplied"},
        {"color": {"primaries": "sRGB"}},
        {"color": {"primaries": True, "transfer": 13, "range": 2}},
        {"color": {"primaries": -1, "transfer": 13, "range": 2}},
        {"color": {"primaries": "sRGB", "transfer": "sRGB", "range": "full"}},
        {"polarity": "opacity"},
        {"unknown": "value"},
    ],
)
def test_invalid_semantics_reject_before_pixel_allocation(monkeypatch, metadata) -> None:
    data = encode_image_array(np.zeros((1, 2, 3, 3), dtype=np.float32))
    payload = json.dumps(metadata).encode()
    data += b"DINKSTER-MEDIA\x01" + struct.pack("<I", len(payload)) + payload
    monkeypatch.setattr(
        np, "load", lambda *_a, **_k: pytest.fail("allocated pixels before validation")
    )
    with pytest.raises(ValueError):
        decode_image_array(data)


def test_forged_layout_is_rejected_without_decoding() -> None:
    data = encode_image_array(np.zeros((1, 2, 3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="channels metadata"):
        validate_image_encoded(data, {"channels": {"layout": "rgba", "alpha": "straight"}})


@pytest.mark.parametrize("channels", [2, 4])
def test_png_rendition_unpremultiplies_without_mutating_input(channels) -> None:
    from dinkster_values import render_image_png
    from PIL import Image

    array = np.full((2, 1, 2, channels), 0.25, dtype=np.float32)
    array[..., -1] = 0.5
    array[:, :, 1, -1] = 0
    annotated = annotate_image(array, alpha="premultiplied")
    with Image.open(io.BytesIO(render_image_png(annotated))) as image:
        pixels = np.asarray(image)
    np.testing.assert_array_equal(pixels[0, 0], [128] * channels)
    np.testing.assert_array_equal(pixels[0, 1], [0] * channels)
    assert np.all(array[..., :-1] == 0.25)


def test_merge_preserves_semantics_and_normalizes_mixed_alpha_modes() -> None:
    from dinkster_values import merge_image_batches

    color = {"primaries": 9, "transfer": 16, "range": 1}
    image = annotate_image(
        np.full((1, 2, 2, 4), 0.5, dtype=np.float32), alpha="premultiplied", color=color
    )
    single = merge_image_batches([image])
    uniform = merge_image_batches([image, image])
    assert media_semantics(single) == media_semantics(uniform) == media_semantics(image)
    mixed = merge_image_batches(
        [image, annotate_image(np.ones((1, 2, 2, 3), dtype=np.float32), color=color)]
    )
    assert media_semantics(mixed) == {"color": color}
    np.testing.assert_array_equal(np.asarray(mixed)[0, ..., :3], 1)
    np.testing.assert_array_equal(np.asarray(mixed)[0, ..., 3], 0.5)
    np.testing.assert_array_equal(np.asarray(mixed)[1], 1)


@pytest.mark.parametrize("list_value", [False, True])
def test_worker_policies_preserve_require_and_create_only_when_declared(list_value) -> None:
    from dinkster_workers.media import apply_alpha_policy, prepare_media_output

    rgb = np.ones((1, 2, 3, 3), dtype=np.float32)
    obj = [rgb] if list_value else rgb
    type_id = "list<comfy.IMAGE>" if list_value else "comfy.IMAGE"
    with pytest.raises(ValueError, match="requires an alpha"):
        apply_alpha_policy(obj, type_id, "require")
    unchanged = cast("Any", apply_alpha_policy(obj, type_id, "preserve"))
    assert (unchanged[0] if list_value else unchanged) is rgb
    spec = OutputSpec(
        "image", TypeExpr.list_of(IMAGE) if list_value else IMAGE, alpha_policy="create_if_missing"
    )
    schema = NodeSchema(node_type="test.create-alpha", outputs=(spec,))
    created = cast("Any", prepare_media_output(obj, type_id, spec, schema, {}))
    result = created[0] if list_value else created
    assert result.shape == (1, 2, 3, 4)
    np.testing.assert_array_equal(result[..., :3], rgb)
    np.testing.assert_array_equal(result[..., 3], 1)


@pytest.mark.parametrize(
    "drop,create", [("none", False), ("decoder", False), ("node", False), ("decoder", True)]
)
def test_asset_coercion_loss_is_reported_and_cached_without_manufacturing_loss(
    tmp_path, drop, create
):
    from dinkster_assets import AssetRef, AssetVault, digest_bytes, register_asset_type
    from dinkster_graph import TypedLiteral
    from dinkster_values import decode_image_file
    from PIL import Image

    class Passthrough(AlphaDrop):
        @classmethod
        def define_schema(cls):
            schema = super().define_schema()
            return replace(
                schema,
                inputs=(
                    InputSpec(
                        "image", IMAGE, alpha_policy="create_if_missing" if create else "preserve"
                    ),
                ),
            )

        @classmethod
        def execute(cls, image):
            return cls.outputs(image=image[..., :3] if drop == "node" else image)

    source = io.BytesIO()
    Image.new("RGBA", (2, 2), (255, 0, 0, 80)).save(source, format="PNG")
    data = source.getvalue()
    digest = digest_bytes(data)
    vault = AssetVault(tmp_path)
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()
    ref = AssetRef(digest, "transparent.png", len(data), resolver=vault)

    def decoder(asset):
        decoded = decode_image_file(asset)
        return np.asarray(decoded)[..., :3] if drop == "decoder" else decoded

    async def scenario():
        registry = TypeRegistry()
        register_asset_type(registry, vault)
        registry.register(
            "comfy.IMAGE",
            encode=encode_image_array,
            decode=decode_image_array,
            meta=image_array_meta,
        )
        registry.register_asset_decoder("comfy.IMAGE", provider_id="test.alpha@1", decode=decoder)
        nodes = (Passthrough,)
        events = []
        engine = Engine(
            schemas=build_schemas(nodes),
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
            on_event=events.append,
        )
        graph = Graph(
            nodes={
                "node": GraphNode(
                    "test.alpha-drop", {"image": TypedLiteral("asset<comfy.IMAGE>", ref.to_wire())}
                )
            }
        )
        for cached in (False, True):
            events.clear()
            result = await engine.run(graph, ["node"])
            assert ("node" in result.cached) == cached
            diagnostics = [event for event in events if event.kind == "value_diagnostics"]
            assert len(diagnostics) == int(drop != "none")
            if drop != "none":
                assert diagnostics[0].detail["diagnostics"] == [
                    {
                        "code": "alpha_dropped",
                        "nodeId": "node",
                        **({"inputIds": ["image"]} if drop == "node" else {"inputId": "image"}),
                        "outputId": "image",
                    }
                ]
            assert np.asarray(result.outputs["node"]["image"].resolve()).shape[-1] == (
                3 if drop != "none" and not create else 4
            )

    asyncio.run(scenario())


class SemanticSource(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="test.semantic-source",
            outputs=(OutputSpec("image", IMAGE), OutputSpec("mask", MASK)),
        )

    @classmethod
    def execute(cls):
        return cls.outputs(
            image=annotate_image(
                np.full((1, 8, 8, 4), 0.25, dtype=np.float32),
                alpha="premultiplied",
                color={"primaries": 9, "transfer": 16, "range": 1},
            ),
            mask=annotate_mask(
                np.full((1, 8, 8), 0.75, dtype=np.float32),
                polarity="transparency",
                semantic="alpha",
            ),
        )


class SemanticPass(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="test.semantic-pass",
            inputs=(InputSpec("image", IMAGE), InputSpec("mask", MASK, mask_polarity="coverage")),
            outputs=(OutputSpec("image", IMAGE), OutputSpec("mask", MASK)),
        )

    @classmethod
    def execute(cls, image, mask):
        return cls.outputs(image=np.asarray(image).copy(), mask=np.asarray(mask).copy())


SEMANTIC_NODES = (SemanticSource, SemanticPass, AlphaDrop)


def register_semantic_types(registry: TypeRegistry) -> None:
    for type_id, meta in (("comfy.IMAGE", image_array_meta), ("comfy.MASK", mask_array_meta)):
        registry.register(
            type_id,
            encode=encode_image_array,
            decode=decode_image_array,
            fingerprint=image_array_fingerprint(type_id),
            meta=meta,
            prepare_buffer_encoding=prepare_image_array_encoding,
            validate_encoded=validate_image_encoded,
            validate_encoded_buffer=validate_image_encoded,
        )


@pytest.mark.parametrize("isolated", [False, True])
def test_media_contract_through_ordinary_worker_boundaries(tmp_path, isolated):
    from pathlib import Path

    from dinkster_workers import IsolatedWorker

    async def scenario():
        registry = TypeRegistry()
        register_semantic_types(registry)
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "media-contract"\n[pack.entry]\n'
            'nodes = "tests.test_media_contract:SEMANTIC_NODES"\n'
            'types = "tests.test_media_contract:register_semantic_types"\n'
        )
        boundary = (
            IsolatedWorker(
                manifest,
                registry,
                shm_threshold=1,
                extra_env={"PYTHONPATH": str(Path(__file__).resolve().parents[1])},
            )
            if isolated
            else None
        )
        if boundary is not None:
            await boundary.start()
        try:
            events = []
            worker = boundary or InProcessWorker(build_node_types(SEMANTIC_NODES), registry)
            schemas = boundary.schemas if boundary is not None else build_schemas(SEMANTIC_NODES)
            engine = Engine(
                schemas=schemas,
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
                on_event=events.append,
            )
            graph = Graph(
                nodes={
                    "source": GraphNode("test.semantic-source"),
                    "pass": GraphNode(
                        "test.semantic-pass",
                        {
                            "image": Link("source", "image"),
                            "mask": Link("source", "mask"),
                        },
                    ),
                    "drop": GraphNode("test.alpha-drop", {"image": Link("pass", "image")}),
                }
            )
            for cached in (False, True):
                events.clear()
                result = await engine.run(graph, ["pass", "drop"])
                assert ("pass" in result.cached) == cached
                for key in ("image", "mask"):
                    expected = SemanticSource.execute()[key]
                    value = result.outputs["pass"][key]
                    assert media_semantics(value.resolve()) == media_semantics(expected)
                    assert value.fingerprint == image_array_fingerprint(value.type_id)(expected)
                    assert encode_image_array(value.resolve()) == encode_image_array(expected)
                diagnostics = [
                    event.detail["diagnostics"]
                    for event in events
                    if event.kind == "value_diagnostics"
                ]
                assert diagnostics == [
                    [
                        {
                            "nodeId": "pass",
                            "code": "mask_polarity_mismatch",
                            "inputId": "mask",
                            "actual": "transparency",
                            "expected": "coverage",
                        }
                    ],
                    [
                        {
                            "nodeId": "drop",
                            "code": "alpha_dropped",
                            "outputId": "image",
                            "inputIds": ["image"],
                        }
                    ],
                ]
        finally:
            if boundary is not None:
                await boundary.close()

    asyncio.run(scenario())
