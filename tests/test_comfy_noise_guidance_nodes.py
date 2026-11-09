"""Public native schemas for noise and frequency guidance."""

from dinkster_api.v1 import CORE_FLOAT, CORE_INT, MODEL_TYPE, TypeExpr
from dinkster_sampling import NODES
from dinkster_sampling.noise_guidance import EpsilonScaling, FreSca, TemporalScoreRescaling


def test_noise_guidance_schemas_preserve_core_sockets_and_controls() -> None:
    expected = (
        ("dinkster.sampling.epsilon_scaling", ["scaling_factor"], [1.005]),
        ("dinkster.sampling.temporal_score_rescaling", ["tsr_k", "tsr_sigma"], [0.95, 1.0]),
        ("dinkster.sampling.fre_sca", ["scale_low", "scale_high", "freq_cutoff"], [1.0, 1.25, 20]),
    )
    for node, (node_type, names, defaults) in zip(
        (EpsilonScaling, TemporalScoreRescaling, FreSca), expected, strict=True
    ):
        assert NODES.count(node) == 1
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
