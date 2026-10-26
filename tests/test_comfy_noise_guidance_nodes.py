"""Public native schemas for noise and frequency guidance."""

from dinkster_api.v1 import CORE_FLOAT, CORE_INT, MODEL_TYPE, TypeExpr
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.noise_guidance import NOISE_GUIDANCE_NODES


def test_noise_guidance_schemas_preserve_core_sockets_and_controls() -> None:
    expected = (
        ("Epsilon Scaling", ["scaling_factor"], [1.005]),
        ("TemporalScoreRescaling", ["tsr_k", "tsr_sigma"], [0.95, 1.0]),
        ("FreSca", ["scale_low", "scale_high", "freq_cutoff"], [1.0, 1.25, 20]),
    )
    for node, (node_type, names, defaults) in zip(NOISE_GUIDANCE_NODES, expected, strict=True):
        assert node in COMFY_NODES
        schema = node.define_schema()
        assert schema.node_type == node_type
        assert [item.id for item in schema.inputs] == ["model", *names]
        assert schema.inputs[0].type == TypeExpr.concrete(MODEL_TYPE)
        assert [item.default for item in schema.inputs[1:]] == defaults
        for item in schema.inputs[1:]:
            assert item.advanced
            assert item.type == TypeExpr.concrete(
                CORE_INT if item.id == "freq_cutoff" else CORE_FLOAT
            )
        assert [(item.id, item.type) for item in schema.outputs] == [
            ("MODEL", TypeExpr.concrete(MODEL_TYPE))
        ]
