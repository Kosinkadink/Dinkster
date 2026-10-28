"""Torch-free schema coverage for ModelPatchLoader."""

from __future__ import annotations

import subprocess
import sys


def test_model_patch_loader_schema_is_torch_free_and_uses_bound_name_asset() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _COLD_SCHEMA], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


_COLD_SCHEMA = """
import sys
from dinkster_api.v1 import ASSET_TYPE, MODEL_PATCH_TYPE, AssetWidget, TypeExpr
from dinkster_nodes_comfy.model_patch_loader import ModelPatchLoader

schema = ModelPatchLoader.schema()
assert "torch" not in sys.modules
# ModelPatchLoader, F bc5f880d, define_schema() without inputs; ASSET is the
# native bound-file substitution for the source name combo.
assert schema.display_name == "Load Model Patch"
assert schema.inputs[0].id == "name"
assert schema.inputs[0].type == TypeExpr.concrete(ASSET_TYPE)
assert schema.inputs[0].widget == AssetWidget(kind="model/patch")
assert schema.outputs[0].id == "MODEL_PATCH"
assert schema.outputs[0].type == TypeExpr.concrete(MODEL_PATCH_TYPE)
"""
