"""Torch-free schemas for native LTXV audio VAE nodes."""

from dinkster_api.v1 import AUDIO_TYPE, LATENT_TYPE, VAE_TYPE, schema_from_wire, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.ltx_audio import LTX_AUDIO_NODES, LTXVAudioVAEDecode, LTXVAudioVAEEncode


def test_ltx_audio_interfaces_match_comfy_schemas() -> None:
    assert LTX_AUDIO_NODES[:2] == (LTXVAudioVAEEncode, LTXVAudioVAEDecode)
    assert all(node in COMFY_NODES for node in (LTXVAudioVAEEncode, LTXVAudioVAEDecode))
    expected = (
        (
            LTXVAudioVAEEncode,
            "LTXV Audio VAE Encode",
            (("audio", AUDIO_TYPE), ("audio_vae", VAE_TYPE)),
            ("LATENT", LATENT_TYPE),
        ),
        (
            LTXVAudioVAEDecode,
            "LTXV Audio VAE Decode",
            (("samples", LATENT_TYPE), ("audio_vae", VAE_TYPE)),
            ("AUDIO", AUDIO_TYPE),
        ),
    )
    for node, display_name, inputs, output in expected:
        schema = node.schema()
        assert (schema.node_type, schema.display_name, schema.category) == (
            node.__name__,
            display_name,
            "model/latent/ltxv",
        )
        assert tuple((item.id, item.type.types[0]) for item in schema.inputs) == inputs
        assert tuple((item.id, item.type.types[0]) for item in schema.outputs) == (output,)
        assert all(item.required for item in schema.inputs)
        assert schema_from_wire(schema_to_wire(schema)) == schema

    encode = LTXVAudioVAEEncode.schema()
    decode = LTXVAudioVAEDecode.schema()
    assert (encode.inputs[0].doc, encode.inputs[1].display_name, encode.inputs[1].doc) == (
        "The audio to be encoded.",
        "Audio VAE",
        "The Audio VAE model to use for encoding.",
    )
    assert (decode.inputs[0].doc, decode.inputs[1].display_name, decode.inputs[1].doc) == (
        "The latent to be decoded.",
        "Audio VAE",
        "The Audio VAE model used for decoding the latent.",
    )
    assert (encode.outputs[0].display_name, decode.outputs[0].display_name) == (
        "Audio Latent",
        "Audio",
    )
