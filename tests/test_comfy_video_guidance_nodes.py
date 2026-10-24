"""Torch-free interfaces for native video CFG guidance nodes."""

from dinkster_api.v1 import CORE_FLOAT, MODEL_TYPE, NumberWidget, TypeExpr, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.video_guidance import VIDEO_GUIDANCE_NODES


def test_video_cfg_guidance_schemas_are_registered_with_canonical_types() -> None:
    expected = (
        ("VideoLinearCFGGuidance", "Video Linear CFG Guidance"),
        ("VideoTriangleCFGGuidance", "Video Triangle CFG Guidance"),
    )
    assert (
        tuple(
            (node.schema().node_type, node.schema().display_name) for node in VIDEO_GUIDANCE_NODES
        )
        == expected
    )
    for node in VIDEO_GUIDANCE_NODES:
        assert node in COMFY_NODES
        schema = node.schema()
        assert schema.category == "model/sampling/guiders"
        assert [(item.id, item.type) for item in schema.inputs] == [
            ("model", TypeExpr.concrete(MODEL_TYPE)),
            ("min_cfg", TypeExpr.concrete(CORE_FLOAT)),
        ]
        assert schema.inputs[1].default == 1.0
        assert schema.inputs[1].advanced
        assert schema.inputs[1].widget == NumberWidget(min=0.0, max=100.0, step=0.5, round=0.01)
        assert [(item.id, item.type) for item in schema.outputs] == [
            ("MODEL", TypeExpr.concrete(MODEL_TYPE))
        ]
        assert schema_to_wire(schema)
