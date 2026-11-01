"""UV atlas declarations do not import tensor or inference dependencies."""

import subprocess
import sys


def test_uv_atlas_cold_schema() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.uv_atlas import RenderUVAtlas
from dinkster_nodes_comfy.mesh_unwrap import UnwrapMesh
from dinkster_workers import load_manifest
assert COMFY_NODES.count(RenderUVAtlas) == 1
schema = RenderUVAtlas.schema()
assert [slot.id for slot in schema.inputs] == ['mesh', 'resolution']
assert schema.inputs[0].type.types == ('MESH',)
assert schema.outputs[0].type.types == ('IMAGE',)
assert schema.inputs[1].default == 1024
manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')
assert manifest.namespaces.count('RenderUVAtlas') == 1
assert COMFY_NODES.count(UnwrapMesh) == manifest.namespaces.count('UnwrapMesh') == 1
schema = UnwrapMesh.schema()
assert [slot.id for slot in schema.inputs] == [
    'mesh', 'segmenter', 'resolution', 'padding', 'weld_distance'
]
assert schema.inputs[0].type.types == schema.outputs[0].type.types == ('MESH',)
assert schema.inputs[1].widget.options == ('pec', 'adaptive')
assert [slot.default for slot in schema.inputs[1:]] == ['pec', 1024, 1, 0.0]
assert 'dinkster_native.model3d.mesh_operations' not in sys.modules
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
