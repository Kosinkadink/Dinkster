"""Torch-free interfaces for native CFG guidance nodes."""

from dinkster_api.v1 import CORE_BOOLEAN, CORE_FLOAT, MODEL_TYPE, TypeExpr, schema_to_wire
from dinkster_sampling import NODES
from dinkster_sampling.cfg_guidance import (
    APG,
    TCFG,
    CFGNorm,
    CFGZeroStar,
    Mahiro,
    NAGuidance,
    RenormCFG,
)


def test_cfg_guidance_schemas_are_registered_with_canonical_types() -> None:
    nodes = (CFGZeroStar, CFGNorm, APG, TCFG, Mahiro, NAGuidance, RenormCFG)
    expected = (
        ("dinkster.sampling.cfg_zero_star", ("model",)),
        ("dinkster.sampling.cfg_norm", ("model", "strength", "pre_cfg")),
        ("dinkster.sampling.apg", ("model", "eta", "norm_threshold", "momentum")),
        ("dinkster.sampling.tcfg", ("model",)),
        ("dinkster.sampling.mahiro", ("model",)),
        ("dinkster.sampling.na_guidance", ("model", "nag_scale", "nag_alpha", "nag_tau")),
        ("dinkster.sampling.renorm_cfg", ("model", "cfg_trunc", "renorm_cfg")),
    )
    assert (
        tuple(
            (node.schema().node_type, tuple(port.id for port in node.schema().inputs))
            for node in nodes
        )
        == expected
    )
    for node in nodes:
        assert node in NODES
        assert node.schema().inputs[0].type == TypeExpr.concrete(MODEL_TYPE)
        assert node.schema().outputs[0].type == TypeExpr.concrete(MODEL_TYPE)
        assert schema_to_wire(node.schema())
    cfg_norm = CFGNorm.schema()
    assert cfg_norm.inputs[1].type == TypeExpr.concrete(CORE_FLOAT)
    assert cfg_norm.inputs[2].type == TypeExpr.concrete(CORE_BOOLEAN)
    assert not cfg_norm.inputs[2].required
