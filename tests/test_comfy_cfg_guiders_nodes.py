"""Torch-free schemas and manifest claims for custom-sampling guiders."""

import sys

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    GUIDER_TYPE,
    MODEL_TYPE,
    TypeExpr,
)
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.cfg_guiders import CFG_GUIDER_NODES
from dinkster_workers import load_manifest


def test_cfg_guider_schemas_are_cold_and_registered_in_source_order() -> None:
    assert "torch" not in sys.modules
    expected = (
        ("CFGGuider", ("model", "positive", "negative", "cfg")),
        ("BasicGuider", ("model", "conditioning")),
        (
            "DualCFGGuider",
            ("model", "cond1", "cond2", "negative", "cfg_conds", "cfg_cond2_negative", "style"),
        ),
        ("DualModelGuider", ("model", "model_negative", "positive", "cfg", "negative")),
        ("CFGOverride", ("model", "cfg", "start_percent", "end_percent")),
    )
    assert (
        tuple(
            (node.schema().node_type, tuple(port.id for port in node.schema().inputs))
            for node in CFG_GUIDER_NODES
        )
        == expected
    )
    assert all(node in COMFY_NODES for node in CFG_GUIDER_NODES)
    for node in CFG_GUIDER_NODES[:-1]:
        assert node.schema().inputs[0].type == TypeExpr.concrete(MODEL_TYPE)
        assert node.schema().outputs[0].type == TypeExpr.concrete(GUIDER_TYPE)
    assert CFG_GUIDER_NODES[-1].schema().category == "model/sampling/guiders"
    assert CFG_GUIDER_NODES[-1].schema().outputs[0].type == TypeExpr.concrete(MODEL_TYPE)
    assert CFG_GUIDER_NODES[0].schema().inputs[1].type == TypeExpr.concrete(CONDITIONING_TYPE)


def test_cfg_guider_manifest_claims_match_the_complete_registry() -> None:
    manifest = load_manifest("packages/dinkster-nodes-comfy/dinkster-pack.toml")
    assert tuple(node.schema().node_type for node in COMFY_NODES) == manifest.namespaces
