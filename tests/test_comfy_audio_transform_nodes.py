"""Torch-free schemas for native ComfyUI audio transform nodes."""

from dinkster_api.v1 import AUDIO_TYPE, CORE_INT, NumberWidget, schema_from_wire, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.audio_transform import (
    AUDIO_TRANSFORM_NODES,
    AudioAdjustVolume,
    SplitAudioChannels,
)


def test_audio_transform_interfaces_match_comfy_schemas() -> None:
    assert AUDIO_TRANSFORM_NODES == (SplitAudioChannels, AudioAdjustVolume)
    split = SplitAudioChannels.schema()
    volume = AudioAdjustVolume.schema()
    assert all(node in COMFY_NODES for node in AUDIO_TRANSFORM_NODES)
    assert (split.node_type, split.display_name, split.category) == (
        "SplitAudioChannels",
        "Split Audio Channels",
        "audio",
    )
    assert tuple((item.id, item.type.types) for item in split.inputs) == (("audio", (AUDIO_TYPE,)),)
    assert split.inputs[0].required
    assert tuple((item.id, item.type.types) for item in split.outputs) == (
        ("left", (AUDIO_TYPE,)),
        ("right", (AUDIO_TYPE,)),
    )
    assert (volume.node_type, volume.display_name, volume.category) == (
        "AudioAdjustVolume",
        "Adjust Audio Volume",
        "audio",
    )
    assert tuple((item.id, item.type.types) for item in volume.inputs) == (
        ("audio", (AUDIO_TYPE,)),
        ("volume", (CORE_INT,)),
    )
    assert all(item.required for item in volume.inputs)
    setting = volume.inputs[1]
    assert setting.default == 1
    assert isinstance(setting.widget, NumberWidget)
    assert setting.widget.min == -100
    assert setting.widget.max == 100
    assert setting.widget.step is None
    assert (
        setting.doc
        == "Volume adjustment in decibels (dB). 0 = no change, +6 = double, -6 = half, etc"
    )
    assert tuple((item.id, item.type.types) for item in volume.outputs) == (
        ("AUDIO", (AUDIO_TYPE,)),
    )
    for node in AUDIO_TRANSFORM_NODES:
        assert schema_from_wire(schema_to_wire(node.schema())) == node.schema()
