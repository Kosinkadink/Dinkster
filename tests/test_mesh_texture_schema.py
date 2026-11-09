"""Baked texture declarations do not require a tensor runtime."""

import subprocess
import sys


def test_mesh_texture_cold_schema() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_3d import NODES
from dinkster_3d.mesh_texture import ApplyTextureToMesh, MeshTextureToImage
from dinkster_workers import load_manifest
assert NODES.count(MeshTextureToImage) == 1
schema = MeshTextureToImage.schema()
assert [slot.id for slot in schema.inputs] == ['mesh']
assert schema.inputs[0].type.types == ('MESH',)
assert [slot.id for slot in schema.outputs] == [
    'base_color', 'metallic', 'roughness', 'occlusion', 'normal_map'
]
assert all(slot.type.types == ('IMAGE',) for slot in schema.outputs)
manifest = load_manifest('packages/dinkster-nodes-3d/dinkster-pack.toml')
assert 'dinkster.3d' in manifest.namespaces
assert NODES.count(ApplyTextureToMesh) == 1
schema = ApplyTextureToMesh.schema()
assert [slot.id for slot in schema.inputs] == [
    'mesh', 'base_color', 'metallic', 'roughness', 'occlusion', 'normal_map'
]
assert all(not slot.required for slot in schema.inputs[2:])
assert schema.outputs[0].type.types == ('MESH',)
assert 'dinkster.3d' in manifest.namespaces
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
