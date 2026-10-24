"""Torch-free public ACE Step text-conditioning schemas."""

import subprocess
import sys

from dinkster_api.v1 import ComboWidget, NumberWidget, schema_from_wire, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.ace_text_conditioning import ACE_TEXT_CONDITIONING_NODES


def test_ace_text_conditioning_schemas_and_registration() -> None:
    ace, ace15 = (node.schema() for node in ACE_TEXT_CONDITIONING_NODES)
    assert all(node in COMFY_NODES for node in ACE_TEXT_CONDITIONING_NODES)
    assert [(node.node_type, node.category) for node in (ace, ace15)] == [
        ("TextEncodeAceStepAudio", "model/conditioning/ace"),
        ("TextEncodeAceStepAudio1.5", "model/conditioning/ace"),
    ]
    assert [
        (item.id, item.type.runtime_type_id(), item.required, item.default) for item in ace.inputs
    ] == [
        ("clip", "CLIP", True, None),
        ("tags", "STRING", True, None),
        ("lyrics", "STRING", True, None),
        ("lyrics_strength", "FLOAT", True, 1.0),
    ]
    assert [
        (item.id, item.type.runtime_type_id(), item.required, item.default) for item in ace15.inputs
    ] == [
        ("clip", "CLIP", True, None),
        ("tags", "STRING", True, None),
        ("lyrics", "STRING", True, None),
        ("seed", "INT", True, 0),
        ("bpm", "INT", True, 120),
        ("duration", "FLOAT", True, 120.0),
        ("timesignature", "COMBO", True, "2"),
        ("language", "COMBO", True, "en"),
        ("keyscale", "COMBO", True, "C major"),
        ("generate_audio_codes", "BOOLEAN", True, True),
        ("cfg_scale", "FLOAT", True, 2.0),
        ("temperature", "FLOAT", True, 0.85),
        ("top_p", "FLOAT", True, 0.9),
        ("top_k", "INT", True, 0),
        ("min_p", "FLOAT", True, 0.0),
    ]
    seed_widget = ace15.inputs[3].widget
    timesignature_widget = ace15.inputs[6].widget
    language_widget = ace15.inputs[7].widget
    keyscale_widget = ace15.inputs[8].widget
    assert isinstance(seed_widget, NumberWidget)
    assert isinstance(timesignature_widget, ComboWidget)
    assert isinstance(language_widget, ComboWidget)
    assert isinstance(keyscale_widget, ComboWidget)
    assert seed_widget.control_after_generate == "randomize"
    assert seed_widget.max == 0xFFFFFFFFFFFFFFFF
    assert timesignature_widget.options == ("2", "3", "4", "6")
    assert len(language_widget.options) == 51
    assert len(keyscale_widget.options) == 34
    assert all(item.advanced for item in ace15.inputs[9:])
    for schema in (ace, ace15):
        assert [(item.id, item.type.runtime_type_id()) for item in schema.outputs] == [
            ("CONDITIONING", "CONDITIONING")
        ]
        assert schema_from_wire(schema_to_wire(schema)) == schema


def test_ace_text_conditioning_cold_discovery() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dinkster_nodes_comfy.ace_text_conditioning "
            "import ACE_TEXT_CONDITIONING_NODES; "
            "assert len([node.schema() for node in ACE_TEXT_CONDITIONING_NODES]) == 2; "
            "assert 'torch' not in sys.modules; "
            "assert 'dinkster_inference' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
