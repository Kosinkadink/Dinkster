"""Torch-free public schemas for the audio generator nodes."""

from dinkster_api.v1 import AUDIO_TYPE, LATENT_TYPE, schema_from_wire, schema_to_wire
from dinkster_audio import NODES as AUDIO_NODES
from dinkster_audio.audio_generators import EmptyAudio
from dinkster_minimax import NODES as MINIMAX_NODES
from dinkster_minimax.audio_generators import EmptyMiniMaxMusic3LatentAudio


def test_audio_generator_schemas() -> None:
    expected = (
        (
            EmptyAudio,
            AUDIO_NODES,
            "dinkster.audio.empty_audio",
            "Empty Audio",
            "audio",
            (("duration", "FLOAT", 60.0), ("sample_rate", "INT", 44100), ("channels", "INT", 2)),
            ("AUDIO", AUDIO_TYPE),
        ),
        (
            EmptyMiniMaxMusic3LatentAudio,
            MINIMAX_NODES,
            "dinkster.minimax.empty_mini_max_music3_latent_audio",
            "Empty MiniMax Music3 Latent Audio",
            "model/latent/minimax music",
            (("seconds", "FLOAT", 120.0), ("batch_size", "INT", 1)),
            ("LATENT", LATENT_TYPE),
        ),
    )
    for node, registered, node_id, display, category, inputs, output in expected:
        schema = node.schema()
        assert node in registered
        assert (schema.node_type, schema.display_name, schema.category) == (
            node_id,
            display,
            category,
        )
        assert (
            tuple((item.id, item.type.runtime_type_id(), item.default) for item in schema.inputs)
            == inputs
        )
        assert all(item.required for item in schema.inputs)
        assert tuple((item.id, item.type.runtime_type_id()) for item in schema.outputs) == (output,)
        assert schema_from_wire(schema_to_wire(schema)) == schema
