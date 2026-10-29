"""Public audio encoder schemas and governed resident identities without Torch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from dinkster_api.v1 import (
    ASSET_TYPE,
    AUDIO_ENCODER_OUTPUT_TYPE,
    AUDIO_ENCODER_TYPE,
    AUDIO_TYPE,
    CORE_STRING,
    AssetWidget,
    ComboWidget,
    TypeExpr,
    TypeRegistry,
    register_inference_types,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_memory.resident import ResidentPool
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.audio_encoder import AUDIO_ENCODER_NODES
from dinkster_values import RESOURCE_ID_META_KEY, ResidentLookupError, resident_resource_id
from dinkster_workers import load_manifest


def test_audio_encoder_interfaces_and_native_manifest() -> None:
    loader, encode, abc = (node.schema() for node in AUDIO_ENCODER_NODES)
    assert (loader.node_type, encode.node_type, abc.node_type) == (
        "AudioEncoderLoader",
        "AudioEncoderEncode",
        "SheetSage2AudioToABC",
    )
    assert loader.display_name == "Load Audio Encoder"
    assert loader.category == "model/loaders"
    assert [(i.id, i.type) for i in loader.inputs] == [
        ("audio_encoder_name", TypeExpr.concrete(ASSET_TYPE))
    ]
    assert loader.inputs[0].widget == AssetWidget(
        accept=("application/octet-stream",), kind="model/audio-encoder"
    )
    assert [(o.id, o.type) for o in loader.outputs] == [
        ("AUDIO_ENCODER", TypeExpr.concrete(AUDIO_ENCODER_TYPE))
    ]
    assert encode.category == "model/conditioning"
    for schema in (encode, abc):
        assert [(i.id, i.type) for i in schema.inputs[:2]] == [
            ("audio_encoder", TypeExpr.concrete(AUDIO_ENCODER_TYPE)),
            ("audio", TypeExpr.concrete(AUDIO_TYPE)),
        ]
    assert [(o.id, o.type) for o in encode.outputs] == [
        ("AUDIO_ENCODER_OUTPUT", TypeExpr.concrete(AUDIO_ENCODER_OUTPUT_TYPE))
    ]
    assert abc.display_name == "SheetSage2 Audio to ABC"
    assert abc.category == "model/conditioning/yue2"
    assert abc.inputs[2].widget == ComboWidget(options=("melody", "full"))
    assert [(o.id, o.type) for o in abc.outputs] == [
        ("abc", TypeExpr.list_of(TypeExpr.concrete(CORE_STRING)))
    ]
    for node in AUDIO_ENCODER_NODES:
        assert node in COMFY_NODES
        assert schema_from_wire(schema_to_wire(node.schema())) == node.schema()
    manifest = load_manifest("packages/dinkster-nodes-comfy/dinkster-pack.toml")
    assert ("devices", *(node.schema().node_type for node in COMFY_NODES)) == manifest.namespaces


def test_audio_encoder_resident_release_and_first_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_values import register_resident_type

    pool = ResidentPool()
    monkeypatch.setattr("dinkster_inference_wire.types.default_pool", lambda: pool)
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(AUDIO_ENCODER_TYPE)
    register_inference_types(registry)
    assert registry.spec(AUDIO_ENCODER_TYPE) is spec
    patcher = object()
    encoder = SimpleNamespace(_dinkster_resident_owner=patcher)
    value = registry.wrap(AUDIO_ENCODER_TYPE, encoder)
    encoded = spec.encode(encoder)
    assert spec.decode(encoded) is encoder
    assert spec.fingerprint is not None
    assert spec.fingerprint(encoder) == spec.fingerprint(spec.decode(encoded))
    assert value.meta.get(RESOURCE_ID_META_KEY) == resident_resource_id(pool.rid_for(patcher))
    assert pool.release_resident(patcher)
    with pytest.raises(ResidentLookupError):
        spec.decode(encoded)

    other = TypeRegistry()
    authority = register_resident_type(other, AUDIO_ENCODER_TYPE, table=ResidentPool())
    register_inference_types(other)
    assert other.spec(AUDIO_ENCODER_TYPE) is authority
    with pytest.raises(ResidentLookupError):
        authority.decode(encoded)
