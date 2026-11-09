"""Torch-free schemas for native ComfyUI audio VAE nodes."""

from dinkster_api.v1 import (
    AUDIO_TYPE,
    CORE_INT,
    LATENT_TYPE,
    VAE_TYPE,
    NumberWidget,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_latent import NODES
from dinkster_latent.audio_vae import (
    VAEDecodeAudio,
    VAEDecodeAudioTiled,
    VAEEncodeAudio,
)


def test_audio_vae_interfaces_match_comfy_schemas() -> None:
    expected = (
        (
            VAEEncodeAudio,
            "dinkster.latent.vae_encode_audio",
            "VAE Encode Audio",
            (("audio", AUDIO_TYPE), ("vae", VAE_TYPE)),
            ("LATENT", LATENT_TYPE),
        ),
        (
            VAEDecodeAudio,
            "dinkster.latent.vae_decode_audio",
            "VAE Decode Audio",
            (("samples", LATENT_TYPE), ("vae", VAE_TYPE)),
            ("AUDIO", AUDIO_TYPE),
        ),
        (
            VAEDecodeAudioTiled,
            "dinkster.latent.vae_decode_audio_tiled",
            "VAE Decode Audio (Tiled)",
            (
                ("samples", LATENT_TYPE),
                ("vae", VAE_TYPE),
                ("tile_size", CORE_INT),
                ("overlap", CORE_INT),
            ),
            ("AUDIO", AUDIO_TYPE),
        ),
    )
    for node, node_type, display_name, inputs, output in expected:
        assert node in NODES
        schema = node.schema()
        assert (schema.node_type, schema.display_name, schema.category) == (
            node_type,
            display_name,
            "model/latent",
        )
        assert tuple((item.id, item.type.types[0]) for item in schema.inputs) == inputs
        assert all(item.required for item in schema.inputs)
        assert tuple((item.id, item.type.types[0]) for item in schema.outputs) == (output,)
        assert schema_from_wire(schema_to_wire(schema)) == schema

    tiled = VAEDecodeAudioTiled.schema()
    for setting, default, minimum, maximum in zip(
        tiled.inputs[2:], (512, 64), (32, 0), (8192, 1024), strict=True
    ):
        assert setting.default == default
        assert isinstance(setting.widget, NumberWidget)
        assert (setting.widget.min, setting.widget.max, setting.widget.step) == (
            minimum,
            maximum,
            8,
        )
