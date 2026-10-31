"""The native weld declaration remains usable without a tensor runtime."""

import subprocess
import sys


def test_weld_vertices_cold_schema() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.mesh_weld import WeldVertices
from dinkster_workers import load_manifest
assert COMFY_NODES.count(WeldVertices) == 1
schema = WeldVertices.schema()
assert [slot.id for slot in schema.inputs] == ['mesh', 'epsilon_rel', 'epsilon_abs']
assert schema.inputs[0].type.types == ('MESH',)
assert schema.outputs[0].type.types == ('MESH',)
assert schema.inputs[1].default == 1e-5 and schema.inputs[2].default == 0.0
manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')
assert manifest.namespaces.count('WeldVertices') == 1
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
