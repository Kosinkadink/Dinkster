"""Torch-free interfaces for native CFG guidance nodes."""

from dinkster_api.v1 import CORE_BOOLEAN, CORE_FLOAT, MODEL_TYPE, TypeExpr, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.cfg_guidance import CFG_GUIDANCE_NODES


def test_cfg_guidance_schemas_are_registered_with_canonical_types() -> None:
    expected = (
        ("CFGZeroStar", ("model",)),
        ("CFGNorm", ("model", "strength", "pre_cfg")),
        ("APG", ("model", "eta", "norm_threshold", "momentum")),
        ("TCFG", ("model",)),
        ("Mahiro", ("model",)),
        ("NAGuidance", ("model", "nag_scale", "nag_alpha", "nag_tau")),
        ("RenormCFG", ("model", "cfg_trunc", "renorm_cfg")),
    )
    assert (
        tuple(
            (node.schema().node_type, tuple(port.id for port in node.schema().inputs))
            for node in CFG_GUIDANCE_NODES
        )
        == expected
    )
    for node in CFG_GUIDANCE_NODES:
        assert node in COMFY_NODES
        assert node.schema().inputs[0].type == TypeExpr.concrete(MODEL_TYPE)
        assert node.schema().outputs[0].type == TypeExpr.concrete(MODEL_TYPE)
        assert schema_to_wire(node.schema())
    cfg_norm = CFG_GUIDANCE_NODES[1].schema()
    assert cfg_norm.inputs[1].type == TypeExpr.concrete(CORE_FLOAT)
    assert cfg_norm.inputs[2].type == TypeExpr.concrete(CORE_BOOLEAN)
    assert not cfg_norm.inputs[2].required
