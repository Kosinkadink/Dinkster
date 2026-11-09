"""Torch-free native audio latent interfaces."""

from dinkster_api.v1 import CORE_FLOAT, CORE_INT, LATENT_TYPE, schema_from_wire, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.audio_latent import AUDIO_LATENT_NODES, GetAudioLatentSize


def test_audio_latent_interfaces_roundtrip() -> None:
    assert tuple(node.schema().node_type for node in AUDIO_LATENT_NODES) == (
        "EmptyLatentAudio",
        "EmptyAceStepLatentAudio",
        "EmptyAceStep1.5LatentAudio",
        "EmptyYuE2LatentAudio",
    )
    for node in AUDIO_LATENT_NODES:
        schema = node.schema()
        assert node in COMFY_NODES
        assert tuple((item.id, item.type.types) for item in schema.inputs) == (
            ("seconds", (CORE_FLOAT,)),
            ("batch_size", (CORE_INT,)),
        )
        assert tuple((item.id, item.type.types) for item in schema.outputs) == (
            ("LATENT", (LATENT_TYPE,)),
        )
        assert schema_from_wire(schema_to_wire(schema)) == schema


def test_audio_latent_size_schema_and_cold_discovery() -> None:
    schema = GetAudioLatentSize.schema()
    assert GetAudioLatentSize in COMFY_NODES
    assert (schema.node_type, schema.display_name, schema.category) == (
        "GetAudioLatentSize",
        "Get Audio Latent Size",
        "model/latent",
    )
    assert [(item.id, item.type.types, item.required) for item in schema.inputs] == [
        ("latent", (LATENT_TYPE,), True)
    ]
    assert [(item.id, item.type.types) for item in schema.outputs] == [
        ("length", (CORE_INT,)),
        ("audio_channels", (CORE_INT,)),
        ("batch_size", (CORE_INT,)),
    ]
    assert schema_from_wire(schema_to_wire(schema)) == schema
