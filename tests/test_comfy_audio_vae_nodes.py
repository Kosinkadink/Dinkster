"""Torch-free schemas for native ComfyUI audio VAE nodes."""

import subprocess
import sys

from dinkster_api.v1 import (
    AUDIO_TYPE,
    CORE_INT,
    LATENT_TYPE,
    VAE_TYPE,
    NumberWidget,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.audio_vae import (
    AUDIO_VAE_NODES,
    VAEDecodeAudio,
    VAEDecodeAudioTiled,
    VAEEncodeAudio,
)


def test_audio_vae_interfaces_match_comfy_schemas() -> None:
    assert AUDIO_VAE_NODES == (VAEEncodeAudio, VAEDecodeAudio, VAEDecodeAudioTiled)
    assert all(node in COMFY_NODES for node in AUDIO_VAE_NODES)
    expected = (
        (
            VAEEncodeAudio,
            "VAE Encode Audio",
            (("audio", AUDIO_TYPE), ("vae", VAE_TYPE)),
            ("LATENT", LATENT_TYPE),
        ),
        (
            VAEDecodeAudio,
            "VAE Decode Audio",
            (("samples", LATENT_TYPE), ("vae", VAE_TYPE)),
            ("AUDIO", AUDIO_TYPE),
        ),
        (
            VAEDecodeAudioTiled,
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
    for node, display_name, inputs, output in expected:
        schema = node.schema()
        assert (schema.node_type, schema.display_name, schema.category) == (
            node.__name__,
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


def test_audio_vae_cold_discovery() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dinkster_nodes_comfy.audio_vae "
            "import AUDIO_VAE_NODES; assert len([n.schema() for n in AUDIO_VAE_NODES]) == 3; "
            "assert 'torch' not in sys.modules; "
            "assert 'dinkster_inference.model_management' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
