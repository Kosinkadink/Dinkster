"""Torch-free schema and registration coverage for compile compatibility nodes."""

from __future__ import annotations

import subprocess
import sys

from dinkster_api.v1 import CORE_COMBO, MODEL_TYPE, TypeExpr
from dinkster_models import NODES as MODEL_NODES
from dinkster_models.compile import TorchCompileModel
from dinkster_wan import NODES as WAN_NODES
from dinkster_wan.compile import wanBlockSwap


def test_compile_nodes_are_registered_once_with_their_public_schemas() -> None:
    compile_schema, nop_schema = TorchCompileModel.schema(), wanBlockSwap.schema()

    assert MODEL_NODES.count(TorchCompileModel) == 1
    assert WAN_NODES.count(wanBlockSwap) == 1
    assert TorchCompileModel not in WAN_NODES
    assert wanBlockSwap not in MODEL_NODES
    assert compile_schema.node_type == "dinkster.models.torch_compile_model"
    assert compile_schema.category == "experimental"
    assert compile_schema.inputs[0].type == TypeExpr.concrete(MODEL_TYPE)
    assert compile_schema.inputs[1].type == TypeExpr.concrete(CORE_COMBO)
    assert compile_schema.inputs[1].advanced is True
    assert compile_schema.inputs[1].required is True
    assert nop_schema.node_type == "dinkster.wan.wan_block_swap"
    assert nop_schema.deprecation is not None
    assert nop_schema.search_visibility == "deprecated"


def test_compile_schemas_serialize_without_importing_torch() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from dinkster_api.v1 import schema_to_wire; "
            "from dinkster_models.compile import TorchCompileModel; "
            "from dinkster_wan.compile import wanBlockSwap; "
            "assert len([schema_to_wire(node.schema()) "
            "for node in (TorchCompileModel, wanBlockSwap)]) == 2; "
            "assert 'torch' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
