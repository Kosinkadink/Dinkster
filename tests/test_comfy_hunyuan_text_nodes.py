"""Torch-free Hunyuan text interfaces."""

import subprocess
import sys

from dinkster_api.v1 import (
    CLIP_TYPE,
    CLIP_VISION_OUTPUT_TYPE,
    CONDITIONING_TYPE,
    CORE_INT,
    CORE_STRING,
    NumberWidget,
    StringWidget,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.hunyuan_text import HUNYUAN_TEXT_NODES


def test_hunyuan_text_schemas() -> None:
    expected = (
        (
            "CLIPTextEncodeHunyuanDiT",
            "CLIP Text Encode (Hunyuan Image)",
            "model/conditioning/hunyuan image",
            (("clip", CLIP_TYPE), ("bert", CORE_STRING), ("mt5xl", CORE_STRING)),
        ),
        (
            "TextEncodeHunyuanVideo_ImageToVideo",
            "",
            "model/conditioning/hunyuan video",
            (
                ("clip", CLIP_TYPE),
                ("clip_vision_output", CLIP_VISION_OUTPUT_TYPE),
                ("prompt", CORE_STRING),
                ("image_interleave", CORE_INT),
            ),
        ),
    )
    for node, (name, display, category, inputs) in zip(HUNYUAN_TEXT_NODES, expected, strict=True):
        schema = node.schema()
        assert node in COMFY_NODES
        assert (schema.node_type, schema.display_name, schema.category) == (name, display, category)
        assert tuple((s.id, s.type.types[0]) for s in schema.inputs) == inputs
        assert all(s.required for s in schema.inputs)
        assert tuple((s.id, s.type.types[0]) for s in schema.outputs) == (
            ("CONDITIONING", CONDITIONING_TYPE),
        )
        assert schema_from_wire(schema_to_wire(schema)) == schema
        for item in schema.inputs:
            if item.type.types[0] == CORE_STRING:
                assert isinstance(item.widget, StringWidget)
                assert item.widget.multiline and item.widget.dynamic_prompts
    setting = HUNYUAN_TEXT_NODES[1].schema().inputs[-1]
    assert setting.default == 2 and setting.advanced
    assert isinstance(setting.widget, NumberWidget)
    assert (setting.widget.min, setting.widget.max) == (1, 512)


def test_hunyuan_text_cold_discovery() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dinkster_nodes_comfy.hunyuan_text "
            "import HUNYUAN_TEXT_NODES; assert len([n.schema() for n in HUNYUAN_TEXT_NODES]) == 2; "
            "assert 'torch' not in sys.modules; "
            "assert 'dinkster_inference.model_management' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
