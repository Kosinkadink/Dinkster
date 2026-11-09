"""Torch-free schemas and manifest claims for custom-sampling guiders."""

import subprocess
import sys

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    GUIDER_TYPE,
    MODEL_TYPE,
    TypeExpr,
)
from dinkster_sampling import NODES
from dinkster_sampling.cfg_guiders import (
    BasicGuider,
    CFGGuider,
    CFGOverride,
    DualCFGGuider,
    DualModelGuider,
)


def test_cfg_guider_schemas_are_cold_and_registered_in_source_order() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from dinkster_sampling import NODES; "
            "[node.schema() for node in NODES]; "
            "import sys; assert 'torch' not in sys.modules",
        ],
        check=True,
        timeout=30,
    )
    expected = (
        ("dinkster.sampling.cfg_guider", ("model", "positive", "negative", "cfg")),
        ("dinkster.sampling.basic_guider", ("model", "conditioning")),
        (
            "dinkster.sampling.dual_cfg_guider",
            ("model", "cond1", "cond2", "negative", "cfg_conds", "cfg_cond2_negative", "style"),
        ),
        (
            "dinkster.sampling.dual_model_guider",
            ("model", "model_negative", "positive", "cfg", "negative"),
        ),
        ("dinkster.sampling.cfg_override", ("model", "cfg", "start_percent", "end_percent")),
    )
    nodes = (CFGGuider, BasicGuider, DualCFGGuider, DualModelGuider, CFGOverride)
    assert (
        tuple(
            (node.schema().node_type, tuple(port.id for port in node.schema().inputs))
            for node in nodes
        )
        == expected
    )
    assert all(node in NODES for node in nodes)
    for node in nodes[:-1]:
        assert node.schema().inputs[0].type == TypeExpr.concrete(MODEL_TYPE)
        assert node.schema().outputs[0].type == TypeExpr.concrete(GUIDER_TYPE)
    assert CFGOverride.schema().category == "model/sampling/guiders"
    assert CFGOverride.schema().outputs[0].type == TypeExpr.concrete(MODEL_TYPE)
    assert CFGGuider.schema().inputs[1].type == TypeExpr.concrete(CONDITIONING_TYPE)
