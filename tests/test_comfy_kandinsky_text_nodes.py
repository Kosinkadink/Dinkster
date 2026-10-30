"""Torch-free Kandinsky 5 text-conditioning schema."""

from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_STRING,
    StringWidget,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.clip_model_specific import CLIPTextEncodeKandinsky5


def test_kandinsky_text_schema_and_registration() -> None:
    schema = CLIPTextEncodeKandinsky5.schema()
    assert CLIPTextEncodeKandinsky5 in COMFY_NODES
    assert (schema.node_type, schema.display_name, schema.category, schema.search_terms) == (
        "CLIPTextEncodeKandinsky5",
        "CLIP Text Encode (Kandinsky 5)",
        "model/conditioning/kandinsky",
        ("kandinsky prompt",),
    )
    assert [
        (item.id, item.type.types[0], item.required, item.default) for item in schema.inputs
    ] == [
        ("clip", CLIP_TYPE, True, None),
        ("clip_l", CORE_STRING, True, None),
        ("qwen25_7b", CORE_STRING, True, None),
    ]
    assert [(item.id, item.type.types[0]) for item in schema.outputs] == [
        ("CONDITIONING", CONDITIONING_TYPE)
    ]
    for item in schema.inputs[1:]:
        assert isinstance(item.widget, StringWidget)
        assert item.widget.multiline and item.widget.dynamic_prompts
    assert schema_from_wire(schema_to_wire(schema)) == schema
