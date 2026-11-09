"""Torch-free native audio latent interfaces."""

from dinkster_api.v1 import CORE_FLOAT, CORE_INT, LATENT_TYPE, schema_from_wire, schema_to_wire
from dinkster_audio_models import NODES as AUDIO_MODEL_NODES
from dinkster_audio_models.audio_latent import (
    EmptyAceStep15LatentAudio,
    EmptyAceStepLatentAudio,
    EmptyYuE2LatentAudio,
)
from dinkster_latent import NODES as LATENT_NODES
from dinkster_latent.audio_latent import EmptyLatentAudio, GetAudioLatentSize


def test_audio_latent_interfaces_roundtrip() -> None:
    cases = (
        (EmptyLatentAudio, LATENT_NODES, "dinkster.latent.empty_latent_audio"),
        (
            EmptyAceStepLatentAudio,
            AUDIO_MODEL_NODES,
            "dinkster.audio.models.empty_ace_step_latent_audio",
        ),
        (
            EmptyAceStep15LatentAudio,
            AUDIO_MODEL_NODES,
            "dinkster.audio.models.empty_ace_step1_5_latent_audio",
        ),
        (EmptyYuE2LatentAudio, AUDIO_MODEL_NODES, "dinkster.audio.models.empty_yu_e2_latent_audio"),
    )
    for node, registered, node_type in cases:
        schema = node.schema()
        assert node in registered
        assert schema.node_type == node_type
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
    assert GetAudioLatentSize in LATENT_NODES
    assert (schema.node_type, schema.display_name, schema.category) == (
        "dinkster.latent.get_audio_latent_size",
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
