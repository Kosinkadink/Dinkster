"""Compact storage is converted at typed invocation, never at transport."""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from dinkster_api.v1 import audio_input, image_input
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_protocol import Invocation
from dinkster_schema import (
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    elaborate,
    schema_from_wire,
    schema_signature,
    schema_to_wire,
)
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker, IsolatedWorker

IMAGE = TypeExpr.concrete("test.image")
BF16 = np.dtype([("bfloat16", "<u2")])


def pixels(dtype: str) -> Any:
    if dtype == "bf16":
        return np.array([0, 0x3F80], dtype=np.uint16).reshape(1, 1, 2, 1).view(BF16)
    maximum = {"uint8": 255, "uint16": 65535}.get(dtype, 1)
    return np.array([0, maximum], dtype=dtype).reshape(1, 1, 2, 1)


def register_types(registry: TypeRegistry) -> None:
    # The fallback codec preserves arrays independently of media codec policy.
    registry.register("test.image", input_convert=image_input)
    registry.register("test.audio", input_convert=audio_input)
    registry.register("dinkster.asset")
    registry.register_asset_decoder(
        "test.image", provider_id="test.storage-decode", decode=lambda obj: obj
    )
    registry.register_asset_decoder(
        "list<test.image>", provider_id="test.storage-list-decode", decode=lambda obj: obj
    )
    registry.register_batch_merge(
        "test.image",
        provider_id="test.storage-merge",
        merge=lambda arrays: np.concatenate(cast(Any, arrays)),
    )


class FloatInput(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.float-input",
            inputs=(InputSpec("image", IMAGE),),
            outputs=(OutputSpec("dtype", TypeExpr.concrete("core.string")),),
        )

    @classmethod
    def execute(cls, image: Any) -> Mapping[str, object]:
        if isinstance(image, list):
            return cls.outputs(dtype=",".join(str(item.dtype) for item in image))
        if isinstance(image, dict):
            image = image["waveform"]
        return cls.outputs(dtype=str(image.dtype))


class StorageInput(FloatInput):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return replace(
            FloatInput.schema(),
            node_type="test.storage-input",
            inputs=(InputSpec("image", IMAGE, accepts_storage=True),),
        )


class FloatListInput(FloatInput):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return replace(
            FloatInput.schema(),
            node_type="test.float-list-input",
            inputs=(InputSpec("image", TypeExpr.list_of(IMAGE)),),
        )


class StorageListInput(FloatInput):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return replace(
            FloatListInput.schema(),
            node_type="test.storage-list-input",
            inputs=(InputSpec("image", TypeExpr.list_of(IMAGE), accepts_storage=True),),
        )


class FloatAudioInput(FloatInput):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return replace(
            FloatInput.schema(),
            node_type="test.float-audio-input",
            inputs=(InputSpec("image", TypeExpr.concrete("test.audio")),),
        )


class StorageAudioInput(FloatInput):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return replace(
            FloatAudioInput.schema(),
            node_type="test.storage-audio-input",
            inputs=(InputSpec("image", TypeExpr.concrete("test.audio"), accepts_storage=True),),
        )


NODES = [
    FloatInput,
    StorageInput,
    FloatListInput,
    StorageListInput,
    FloatAudioInput,
    StorageAudioInput,
]


def registry() -> TypeRegistry:
    result = TypeRegistry()
    register_core_types(result)
    register_types(result)
    return result


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float16", "bf16", "float32"])
@pytest.mark.parametrize("accepts_storage", [False, True])
@pytest.mark.parametrize("route", ["direct", "list", "decode", "decode-list", "lift", "merge"])
def test_typed_inputs_normalize_only_at_invocation(
    dtype: str, accepts_storage: bool, route: str
) -> None:
    types = registry()
    array = pixels(dtype)
    source_type, target = {
        "direct": ("test.image", IMAGE),
        "list": ("list<test.image>", TypeExpr.list_of(IMAGE)),
        "decode": ("asset<test.image>", IMAGE),
        "decode-list": ("asset<list<test.image>>", TypeExpr.list_of(IMAGE)),
        "lift": ("list<asset<test.image>>", TypeExpr.list_of(IMAGE)),
        "merge": ("list<asset<test.image>>", IMAGE),
    }[route]
    listed = route in ("list", "decode-list", "lift", "merge")
    value = types.wrap(source_type, [array, array] if listed else array)
    fingerprint = value.fingerprint
    schema = NodeSchema(
        node_type="test.input",
        inputs=(InputSpec("image", target, accepts_storage=accepts_storage),),
    )
    invocation = Invocation("invocation", "node", schema.node_type, {"image": value}, schema)
    received = InProcessWorker({}, types)._unwrap(invocation, "image", value)
    members = received if target.kind == "list" else [received]
    for member in cast(list[Any], members):
        assert member.dtype == (array.dtype if accepts_storage else np.dtype("float32"))
        expected = (
            array if accepts_storage else np.array([0, 1], dtype=np.float32).reshape(array.shape)
        )
        if route == "merge":
            expected = np.concatenate([expected, expected])
        np.testing.assert_array_equal(member, expected)
    np.testing.assert_array_equal(array, pixels(dtype))
    assert value.fingerprint == fingerprint
    if route == "direct" and (accepts_storage or dtype == "float32"):
        assert received is array


def test_mixed_storage_members_are_normalized_before_merging() -> None:
    types = registry()
    value = types.wrap("list<asset<test.image>>", [pixels("uint8"), pixels("uint16")])
    schema = FloatInput.schema()
    invocation = Invocation("invocation", "node", schema.node_type, {"image": value}, schema)
    received = InProcessWorker({}, types)._unwrap(invocation, "image", value)
    np.testing.assert_array_equal(received, np.tile(pixels("float32"), (2, 1, 1, 1)))


@pytest.mark.parametrize("accepts_storage", [False, True])
def test_audio_and_unknown_types(accepts_storage: bool) -> None:
    types = registry()
    types.register("test.opaque")
    worker = InProcessWorker({}, types)
    waveform = np.array([-32768, 0, 32767], dtype=np.int16).reshape(1, 1, 3)
    audio = {"waveform": waveform, "sample_rate": 48000}
    for type_id, obj in (("test.audio", audio), ("test.opaque", {"opaque": 3})):
        value = types.wrap(type_id, obj)
        schema = NodeSchema(
            node_type="test.input",
            inputs=(
                InputSpec("value", TypeExpr.concrete(type_id), accepts_storage=accepts_storage),
            ),
        )
        invocation = Invocation("invocation", "node", schema.node_type, {"value": value}, schema)
        received = cast(dict[str, Any], worker._unwrap(invocation, "value", value))
        if type_id == "test.audio" and not accepts_storage:
            assert received["waveform"].dtype == np.float32
            np.testing.assert_array_equal(received["waveform"], waveform.astype(np.float32) / 32768)
            assert received["sample_rate"] == 48000
        else:
            assert received is obj
    unknown = object()
    assert types.input_object("unregistered.type", unknown) is unknown


def test_storage_schema_roundtrip_elaboration_and_cache_identity() -> None:
    ordinary = FloatInput.schema()
    compact = replace(ordinary, inputs=(replace(ordinary.inputs[0], accepts_storage=True),))
    explicit_default = replace(
        ordinary, inputs=(replace(ordinary.inputs[0], accepts_storage=False),)
    )
    assert schema_signature(ordinary) == schema_signature(explicit_default)
    assert schema_signature(ordinary) != schema_signature(compact)
    types = registry()
    engine = Engine(
        schemas={ordinary.node_type: ordinary},
        worker=InProcessWorker({}, types),
        registry=types,
        cache=MemoryLRUCache(),
    )
    value = types.wrap("test.image", pixels("uint8"))
    assert engine._cache_key(ordinary, schema_signature(ordinary), {"image": value}) != (
        engine._cache_key(compact, schema_signature(compact), {"image": value})
    )
    wire = schema_to_wire(ordinary)
    assert "acceptsStorage" not in cast(list[dict[str, Any]], wire["interface"])[0]
    assert schema_from_wire(wire) == ordinary
    wire = schema_to_wire(compact)
    assert schema_from_wire(wire) == compact
    assert cast(list[dict[str, Any]], wire["interface"])[0]["acceptsStorage"] is True
    cast(list[dict[str, Any]], wire["interface"])[0]["acceptsStorage"] = "true"
    with pytest.raises(ValueError, match="must be a bool"):
        schema_from_wire(wire)
    with pytest.raises(ValueError, match="accepts_storage must be a bool"):
        InputSpec("image", IMAGE, accepts_storage=cast(bool, 1))
    family = NodeSchema(
        node_type="test.family",
        input_families=(InputFamilySpec("images", compact.inputs),),
    )
    assert schema_from_wire(schema_to_wire(family)) == family
    effective = elaborate(family, ["images.one"])
    assert effective.inputs[0].accepts_storage


@pytest.mark.parametrize("isolated", [False, True])
def test_invocation_storage_contract_crosses_worker_boundary(
    tmp_path: Path, isolated: bool
) -> None:
    async def scenario() -> None:
        types = registry()
        if isolated:
            manifest = tmp_path / "dinkster-pack.toml"
            manifest.write_text(
                '[pack]\nname = "storage-test"\n[pack.entry]\n'
                'nodes = "tests.test_storage_inputs:NODES"\n'
                'types = "tests.test_storage_inputs:register_types"\n'
            )
            worker = IsolatedWorker(
                manifest, types, extra_env={"PYTHONPATH": str(Path(__file__).parents[1])}
            )
            await worker.start()
        else:
            worker = InProcessWorker({node.schema().node_type: node for node in NODES}, types)
        try:
            for node in NODES:
                schema = node.schema()
                spec = schema.inputs[0]
                audio = spec.type == TypeExpr.concrete("test.audio")
                listed = spec.type.kind == "list"
                dtypes = (
                    ("int16", "float32")
                    if audio
                    else ("uint8", "uint16", "float16", "bf16", "float32")
                )
                sources = (
                    ("test.audio",)
                    if audio
                    else (("list<test.image>",) if listed else ("test.image",))
                    if isolated
                    else (
                        ("list<test.image>", "asset<list<test.image>>", "list<asset<test.image>>")
                        if listed
                        else ("test.image", "asset<test.image>", "list<asset<test.image>>")
                    )
                )
                for dtype in dtypes:
                    for source_type in sources:
                        array = pixels(dtype)
                        obj = (
                            {"waveform": array, "sample_rate": 48000}
                            if audio
                            else ([array, array] if "list<" in source_type else array)
                        )
                        value = types.wrap(source_type, obj)
                        result = await worker.invoke(
                            Invocation(
                                "invocation", "node", schema.node_type, {"image": value}, schema
                            )
                        )
                        assert result.error is None, result.error
                        assert result.outputs is not None
                        expected = str(array.dtype) if spec.accepts_storage else "float32"
                        if listed:
                            expected = ",".join([expected, expected])
                        assert result.outputs["dtype"].resolve() == expected
        finally:
            if isinstance(worker, IsolatedWorker):
                await worker.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "module,registration",
    [
        ("dinkster_nodes_dev.image", "register_dev_types"),
        ("dinkster_nodes_media_io", "register_media_types"),
        ("dinkster.comfy_compose", "register_comfy_host_types"),
        *[
            (f"dinkster_nodes_vision.{name}.nodes", "register_types")
            for name in (
                "birefnet",
                "depth_anything_v2",
                "depth_anything_v3",
                "detr",
                "efficient_sam",
                "hed",
                "rtdetr",
                "sam31",
                "upscale",
            )
        ],
    ],
)
def test_production_media_registrations_attach_input_hooks(module: str, registration: str) -> None:
    from dinkster_values import audio_meta, image_array_meta

    types = TypeRegistry()
    getattr(importlib.import_module(module), registration)(types)
    media = [types.spec(type_id) for type_id in types.type_ids()]
    media = [spec for spec in media if spec.meta in (image_array_meta, audio_meta)]
    assert media
    for spec in media:
        assert spec.input_convert is (audio_input if spec.meta is audio_meta else image_input)
