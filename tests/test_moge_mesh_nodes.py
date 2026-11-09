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
from dinkster_vision_moge import NODES
from dinkster_vision_moge.moge_mesh import MoGePointMapToMesh
from dinkster_workers import load_manifest
schema = MoGePointMapToMesh.define_schema()
manifest = load_manifest('packages/dinkster-nodes-vision-moge/dinkster-pack.toml')
assert NODES.count(MoGePointMapToMesh) == 1
assert manifest.namespaces.count('dinkster.vision.moge') == 1
assert schema.node_type == 'dinkster.vision.moge.mo_ge_point_map_to_mesh'
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
