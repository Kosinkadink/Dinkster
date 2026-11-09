"""Torch-free interfaces for native video CFG guidance nodes."""

from dinkster_api.v1 import CORE_FLOAT, MODEL_TYPE, NumberWidget, TypeExpr, schema_to_wire
from dinkster_sampling import NODES
from dinkster_sampling.video_guidance import VideoLinearCFGGuidance, VideoTriangleCFGGuidance


def test_video_cfg_guidance_schemas_are_registered_with_canonical_types() -> None:
    nodes = (VideoLinearCFGGuidance, VideoTriangleCFGGuidance)
    expected = (
        ("dinkster.sampling.video_linear_cfg_guidance", "Video Linear CFG Guidance"),
        ("dinkster.sampling.video_triangle_cfg_guidance", "Video Triangle CFG Guidance"),
    )
    assert (
        tuple((node.schema().node_type, node.schema().display_name) for node in nodes) == expected
    )
    for node in nodes:
        assert NODES.count(node) == 1
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
