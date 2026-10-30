"""Torch-free schemas for native LTXV preprocessing."""

from dinkster_api.v1 import CORE_INT, IMAGE_TYPE, NumberWidget, schema_from_wire, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.ltx_video_preprocess import LTX_VIDEO_PREPROCESS_NODES, LTXVPreprocess


def test_ltx_video_preprocess_schema() -> None:
    schema = LTXVPreprocess.schema()
    assert schema.node_type == "LTXVPreprocess"
    assert schema.display_name == "LTXV Preprocess"
    assert schema.category == "video/preprocessors"
    assert tuple((item.id, item.type.types[0]) for item in schema.inputs) == (
        ("image", IMAGE_TYPE),
        ("img_compression", CORE_INT),
    )
    assert schema.inputs[1].default == 35
    assert isinstance(schema.inputs[1].widget, NumberWidget)
    assert schema.inputs[1].widget.min == 0
    assert schema.inputs[1].widget.max == 100
    assert schema.outputs[0].id == "IMAGE"
    assert schema.outputs[0].display_name == "output_image"
    assert schema_from_wire(schema_to_wire(schema)) == schema
    assert LTX_VIDEO_PREPROCESS_NODES == (LTXVPreprocess,)
    assert LTXVPreprocess in COMFY_NODES
