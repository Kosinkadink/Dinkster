"""Torch-free WanDancer scalar and list socket contracts."""

from dinkster_api.v1 import (
    AUDIO_TYPE,
    IMAGE_TYPE,
    MASK_TYPE,
    NumberWidget,
    TypeExpr,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_wan import NODES
from dinkster_wan.wandancer_keyframes import WanDancerPadKeyframes, WanDancerPadKeyframesList


def test_wandancer_keyframe_schemas_and_cold_discovery() -> None:
    nodes = (WanDancerPadKeyframes, WanDancerPadKeyframesList)
    assert tuple(node.schema().node_type for node in nodes) == (
        "dinkster.wan.wan_dancer_pad_keyframes",
        "dinkster.wan.wan_dancer_pad_keyframes_list",
    )
    for node, list_output in zip(nodes, (False, True), strict=True):
        schema = node.schema()
        assert schema.category == "image/video"
        assert [i.id for i in schema.inputs] == [
            "images",
            "segment_length",
            "num_segments" if list_output else "segment_index",
            "audio",
        ]
        assert all(i.required for i in schema.inputs)
        assert schema.inputs[1].default == 149
        assert schema.inputs[1].widget == NumberWidget(min=1, max=10000)
        assert schema.inputs[2].default == (1 if list_output else 0)
        assert schema.inputs[2].widget == NumberWidget(min=1 if list_output else 0, max=100)
        outputs = [TypeExpr.concrete(kind) for kind in (IMAGE_TYPE, MASK_TYPE, AUDIO_TYPE)]
        assert [o.id for o in schema.outputs] == [
            "keyframes_sequence",
            "keyframes_mask",
            "audio_segment",
        ]
        assert [o.type for o in schema.outputs] == [
            TypeExpr.list_of(t) if list_output else t for t in outputs
        ]
        assert schema_from_wire(schema_to_wire(schema)) == schema
        assert NODES.count(node) == 1
