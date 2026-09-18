"""Host-side utility schemas stay Torch-free and call native execution helpers."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_utility_execution_targets_are_native() -> None:
    path = ROOT / "packages/dinkster-native/src/dinkster_native/native_arm.py"
    targets = {
        "GenerationImageCropToMask": "dinkster_inference_torch.image_crop",
        "GenerationGetMeshInfo": "dinkster_inference_torch.mesh",
        "GenerationMeshToModel3D": "dinkster_inference_torch.mesh",
    }
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name in targets:
            imports = [
                call.args[0].value
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "import_module"
                and isinstance(call.args[0], ast.Constant)
            ]
            assert imports == [targets.pop(node.name)]
    assert not targets


def test_native_utility_registration_does_not_import_torch_or_upstream() -> None:
    script = """
import importlib.abc
import os
import sys
os.environ['DINKSTER_COMFY_NATIVE_ONLY'] = '1'
os.environ['DINKSTER_COMFYUI_ROOT'] = ''
class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'comfy', 'comfy_extras', 'comfy_api',
                                     'nodes', 'folder_paths', 'server'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Forbidden())
from dinkster_compat_comfy.entry import COMFY_NODES, ARM_NODES
from dinkster_compat_comfy.native_arm import (
    GenerationImageCropToMask, GenerationGetMeshInfo, GenerationMeshToModel3D,
)
nodes = {node.schema().node_type: node for node in COMFY_NODES}
for node in (GenerationImageCropToMask, GenerationGetMeshInfo, GenerationMeshToModel3D):
    assert nodes[node.schema().node_type] is node
    assert node in ARM_NODES['native']
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
