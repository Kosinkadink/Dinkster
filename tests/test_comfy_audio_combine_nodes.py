"""Torch-free public interfaces for native ComfyUI audio combine nodes."""

from typing import Any, cast

from dinkster_api.v1 import AUDIO_TYPE, CORE_COMBO, ComboWidget, schema_from_wire, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.audio_combine import AUDIO_COMBINE_NODES


def test_audio_combine_schemas_match_comfyui() -> None:
    expected = (
        ("JoinAudioChannels", ("audio_left", "audio_right"), ("audio",)),
        ("AudioConcat", ("audio1", "audio2", "direction"), ("AUDIO",)),
        ("AudioMerge", ("audio1", "audio2", "merge_method"), ("AUDIO",)),
    )
    assert (
        tuple(
            (
                node.schema().node_type,
                tuple(item.id for item in node.schema().inputs),
                tuple(item.id for item in node.schema().outputs),
            )
            for node in AUDIO_COMBINE_NODES
        )
        == expected
    )
    for node in AUDIO_COMBINE_NODES:
        schema = node.schema()
        assert node in COMFY_NODES
        assert all(item.required for item in schema.inputs)
        assert all(item.type.runtime_type_id() == AUDIO_TYPE for item in schema.inputs[:2])
        assert all(item.type.runtime_type_id() == AUDIO_TYPE for item in schema.outputs)
        assert schema_from_wire(schema_to_wire(schema)) == schema
    concat = AUDIO_COMBINE_NODES[1].schema().inputs[-1]
    merge = AUDIO_COMBINE_NODES[2].schema().inputs[-1]
    assert concat.type.runtime_type_id() == merge.type.runtime_type_id() == CORE_COMBO
    assert concat.default == "after"
    assert merge.default is None
    assert isinstance(concat.widget, ComboWidget)
    assert isinstance(merge.widget, ComboWidget)
    assert tuple(option.value for option in cast(Any, concat.widget).options) == ("after", "before")
    assert tuple(option.value for option in cast(Any, merge.widget).options) == (
        "add",
        "mean",
        "subtract",
        "multiply",
    )
