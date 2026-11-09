"""Torch-free Hunyuan text interfaces."""

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
from dinkster_hunyuan import NODES
from dinkster_hunyuan.hunyuan_text import (
    CLIPTextEncodeHunyuanDiT,
    TextEncodeHunyuanVideo_ImageToVideo,
)


def test_hunyuan_text_schemas() -> None:
    expected = (
        (
            "dinkster.hunyuan.clip_text_encode_hunyuan_di_t",
            "CLIP Text Encode (Hunyuan Image)",
            "model/conditioning/hunyuan image",
            (("clip", CLIP_TYPE), ("bert", CORE_STRING), ("mt5xl", CORE_STRING)),
        ),
        (
            "dinkster.hunyuan.text_encode_hunyuan_video_image_to_video",
            "TextEncodeHunyuanVideo_ImageToVideo",
            "model/conditioning/hunyuan video",
            (
                ("clip", CLIP_TYPE),
                ("clip_vision_output", CLIP_VISION_OUTPUT_TYPE),
                ("prompt", CORE_STRING),
                ("image_interleave", CORE_INT),
            ),
        ),
    )
    for node, (name, display, category, inputs) in zip(
        (CLIPTextEncodeHunyuanDiT, TextEncodeHunyuanVideo_ImageToVideo), expected, strict=True
    ):
        schema = node.schema()
        assert node in NODES
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
    setting = TextEncodeHunyuanVideo_ImageToVideo.schema().inputs[-1]
    assert setting.default == 2 and setting.advanced
    assert isinstance(setting.widget, NumberWidget)
    assert (setting.widget.min, setting.widget.max) == (1, 512)
