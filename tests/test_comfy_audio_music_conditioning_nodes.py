"""Torch-free public audio music conditioning schemas."""

import subprocess
import sys

from dinkster_api.v1 import schema_from_wire, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.audio_music_conditioning import AUDIO_MUSIC_CONDITIONING_NODES


def test_audio_music_conditioning_schemas() -> None:
    music, reference = (node.schema() for node in AUDIO_MUSIC_CONDITIONING_NODES)
    assert all(node in COMFY_NODES for node in AUDIO_MUSIC_CONDITIONING_NODES)
    assert (music.node_type, music.display_name, music.category) == (
        "MiniMaxMusic3TextEncode",
        "MiniMax Music3 Text Encode",
        "model/conditioning/minimax music",
    )
    assert [(i.id, i.type.runtime_type_id(), i.default) for i in music.inputs] == [
        ("clip", "CLIP", None),
        ("caption", "STRING", None),
        ("lyrics", "STRING", None),
        ("seed", "INT", 0),
        ("max_duration", "FLOAT", 120.0),
        ("cfg_scale", "FLOAT", 1.5),
        ("top_k", "INT", 50),
    ]
    assert all(i.required for i in music.inputs)
    assert [(o.id, o.type.runtime_type_id()) for o in music.outputs] == [
        ("CONDITIONING", "CONDITIONING"),
        ("seconds", "FLOAT"),
    ]
    assert music.inputs[5].advanced and music.inputs[6].advanced
    assert (reference.node_type, reference.display_name, reference.category) == (
        "ReferenceTimbreAudio",
        "Set Reference Audio",
        "model/conditioning",
    )
    assert [(i.id, i.type.runtime_type_id(), i.required) for i in reference.inputs] == [
        ("conditioning", "CONDITIONING", True),
        ("latent", "LATENT", False),
    ]
    for schema in (music, reference):
        assert schema_from_wire(schema_to_wire(schema)) == schema


def test_audio_music_conditioning_cold_discovery() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dinkster_nodes_comfy.audio_music_conditioning "
            "import AUDIO_MUSIC_CONDITIONING_NODES; "
            "assert len([n.schema() for n in AUDIO_MUSIC_CONDITIONING_NODES]) == 2; "
            "assert 'torch' not in sys.modules; "
            "assert 'dinkster_inference.model_management' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
