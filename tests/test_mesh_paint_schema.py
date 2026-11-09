"""PaintMesh declares public geometry without loading tensor runtimes."""

import subprocess
import sys


def test_paint_mesh_cold_registration_and_schema() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['torch'] = None
sys.modules['scipy'] = None
sys.modules['dinkster_inference'] = None
from dinkster_3d import NODES
from dinkster_3d.mesh_paint import PaintMesh
from dinkster_workers import load_manifest
schema = PaintMesh.schema()
assert [(i.id, i.type.types) for i in schema.inputs] == [
    ('mesh', ('MESH',)), ('voxel_colors', ('VOXEL',))
]
assert [(o.id, o.type.types) for o in schema.outputs] == [('mesh', ('MESH',))]
assert NODES.count(PaintMesh) == 1
manifest = load_manifest('packages/dinkster-nodes-3d/dinkster-pack.toml')
assert 'dinkster.3d' in manifest.namespaces
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
