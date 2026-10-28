"""MoGe loader registration and schema stay independent of Torch."""

import subprocess
import sys


def test_moge_loader_cold_schema_and_registration() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_api.v1 import ASSET_TYPE, MOGE_MODEL_TYPE, AssetWidget, TypeExpr
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.moge_loader import LoadMoGeModel
from dinkster_workers import load_manifest
schema = LoadMoGeModel.define_schema()
manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')
assert COMFY_NODES.count(LoadMoGeModel) == 1
assert manifest.namespaces.count('LoadMoGeModel') == 1
assert schema.display_name == 'Load MoGe Model'
assert schema.category == 'model/loaders'
assert len(schema.inputs) == len(schema.outputs) == 1
assert schema.inputs[0].id == 'model_name'
assert schema.inputs[0].type == TypeExpr.concrete(ASSET_TYPE)
assert isinstance(schema.inputs[0].widget, AssetWidget)
assert schema.inputs[0].widget.kind == 'model/geometry-estimation'
assert schema.outputs[0].id == 'MOGE_MODEL'
assert schema.outputs[0].type == TypeExpr.concrete(MOGE_MODEL_TYPE)
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
