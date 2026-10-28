"""Torch-free schemas for native ComfyUI audio effect nodes."""

import subprocess
import sys

from dinkster_api.v1 import (
    AUDIO_TYPE,
    CORE_FLOAT,
    CORE_INT,
    NumberWidget,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.audio_effects import (
    AUDIO_EFFECT_NODES,
    AudioEqualizer3Band,
    TrimAudioDuration,
)


def test_audio_effect_schemas_match_comfyui_without_torch() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from dinkster_nodes_comfy.audio_effects import AUDIO_EFFECT_NODES; "
            "[node.schema() for node in AUDIO_EFFECT_NODES]; "
            "import sys; assert 'torch' not in sys.modules",
        ],
        check=True,
        timeout=30,
    )
    assert AUDIO_EFFECT_NODES == (TrimAudioDuration, AudioEqualizer3Band)
    trim, equalizer = (node.schema() for node in AUDIO_EFFECT_NODES)
    assert all(node in COMFY_NODES for node in AUDIO_EFFECT_NODES)
    assert (trim.node_type, trim.display_name, trim.description, trim.category) == (
        "TrimAudioDuration",
        "Trim Audio Duration",
        "Trim audio tensor into chosen time range.",
        "audio",
    )
    assert tuple((input.id, input.type.types, input.default) for input in trim.inputs) == (
        ("audio", (AUDIO_TYPE,), None),
        ("start_index", (CORE_FLOAT,), 0.0),
        ("duration", (CORE_FLOAT,), 60.0),
    )
    assert isinstance(trim.inputs[1].widget, NumberWidget)
    assert trim.inputs[1].widget.min == -float(0xFFFFFFFFFFFFFFFF)
    assert trim.inputs[1].widget.max == float(0xFFFFFFFFFFFFFFFF)
    assert trim.inputs[1].widget.step == 0.01
    assert trim.inputs[2].widget == NumberWidget(min=0.0, step=0.01)
    assert tuple(output.id for output in trim.outputs) == ("AUDIO",)
    assert (
        equalizer.node_type,
        equalizer.display_name,
        equalizer.description,
        equalizer.category,
    ) == (
        "AudioEqualizer3Band",
        "Audio Equalizer (3-Band)",
        "",
        "audio",
    )
    assert tuple((input.id, input.type.types, input.default) for input in equalizer.inputs) == (
        ("audio", (AUDIO_TYPE,), None),
        ("low_gain_dB", (CORE_FLOAT,), 0.0),
        ("low_freq", (CORE_INT,), 100),
        ("mid_gain_dB", (CORE_FLOAT,), 0.0),
        ("mid_freq", (CORE_INT,), 1000),
        ("mid_q", (CORE_FLOAT,), 0.707),
        ("high_gain_dB", (CORE_FLOAT,), 0.0),
        ("high_freq", (CORE_INT,), 5000),
    )
    assert tuple(output.id for output in equalizer.outputs) == ("AUDIO",)
    assert all(
        schema_from_wire(schema_to_wire(node.schema())) == node.schema()
        for node in AUDIO_EFFECT_NODES
    )
