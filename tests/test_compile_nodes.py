"""Torch-free schema and registration coverage for compile compatibility nodes."""

from __future__ import annotations

import subprocess
import sys

from dinkster_api.v1 import CORE_COMBO, MODEL_TYPE, TypeExpr
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.compile import COMPILE_NODES, TorchCompileModel, wanBlockSwap


def test_compile_nodes_are_registered_once_with_their_public_schemas() -> None:
    compile_schema, nop_schema = (node.schema() for node in COMPILE_NODES)

    assert COMPILE_NODES == (TorchCompileModel, wanBlockSwap)
    assert all(node in COMFY_NODES for node in COMPILE_NODES)
    assert tuple(node for node in COMFY_NODES if node in COMPILE_NODES) == (
        TorchCompileModel,
        wanBlockSwap,
    )
    assert compile_schema.node_type == "TorchCompileModel"
    assert compile_schema.category == "experimental"
    assert compile_schema.inputs[0].type == TypeExpr.concrete(MODEL_TYPE)
    assert compile_schema.inputs[1].type == TypeExpr.concrete(CORE_COMBO)
    assert compile_schema.inputs[1].advanced is True
    assert compile_schema.inputs[1].required is True
    assert nop_schema.node_type == "wanBlockSwap"
    assert nop_schema.deprecation is not None
    assert nop_schema.search_visibility == "deprecated"


def test_compile_schemas_serialize_without_importing_torch() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from dinkster_api.v1 import schema_to_wire; "
            "from dinkster_nodes_comfy.compile import COMPILE_NODES; "
            "assert len([schema_to_wire(node.schema()) for node in COMPILE_NODES]) == 2; "
            "assert 'torch' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
