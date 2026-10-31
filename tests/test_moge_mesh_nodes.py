"""MoGe mesh declaration and imports remain cold."""

import subprocess
import sys


def test_moge_mesh_cold_schema_and_registration() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.moge_mesh import MoGePointMapToMesh
from dinkster_workers import load_manifest
schema = MoGePointMapToMesh.define_schema()
manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')
assert COMFY_NODES.count(MoGePointMapToMesh) == 1
assert manifest.namespaces.count('MoGePointMapToMesh') == 1
assert [i.id for i in schema.inputs] == [
    'moge_geometry', 'batch_index', 'decimation', 'discontinuity_threshold', 'texture'
]
assert [i.default for i in schema.inputs[1:]] == [0, 1, 0.04, True]
assert [o.id for o in schema.outputs] == ['MESH']
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
