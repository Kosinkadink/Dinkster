"""Normal-generation declarations remain available without tensor dependencies."""

import subprocess
import sys


def test_mesh_normals_cold_schema() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_3d import NODES
from dinkster_3d.mesh_normals import MeshSmoothNormals
from dinkster_workers import load_manifest
assert NODES.count(MeshSmoothNormals) == 1
schema = MeshSmoothNormals.schema()
assert [slot.id for slot in schema.inputs] == ['mesh', 'crease_angle']
assert schema.inputs[0].type.types == ('MESH',)
assert schema.outputs[0].type.types == ('MESH',)
assert schema.inputs[1].default == 180.0
manifest = load_manifest('packages/dinkster-nodes-3d/dinkster-pack.toml')
assert 'dinkster.3d' in manifest.namespaces
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
