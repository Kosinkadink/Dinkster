"""Torch-free public schemas for the audio generator nodes."""

from dinkster_api.v1 import AUDIO_TYPE, LATENT_TYPE, schema_from_wire, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.audio_generators import AUDIO_GENERATOR_NODES


def test_audio_generator_schemas() -> None:
    expected = (
        (
            "EmptyAudio",
            "Empty Audio",
            "audio",
            (("duration", "FLOAT", 60.0), ("sample_rate", "INT", 44100), ("channels", "INT", 2)),
            ("AUDIO", AUDIO_TYPE),
        ),
        (
            "EmptyMiniMaxMusic3LatentAudio",
            "Empty MiniMax Music3 Latent Audio",
            "model/latent/minimax music",
            (("seconds", "FLOAT", 120.0), ("batch_size", "INT", 1)),
            ("LATENT", LATENT_TYPE),
        ),
    )
    for node, (node_id, display, category, inputs, output) in zip(
        AUDIO_GENERATOR_NODES, expected, strict=True
    ):
        schema = node.schema()
        assert node in COMFY_NODES
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
