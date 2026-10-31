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
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.mesh_texture import MeshTextureToImage
from dinkster_workers import load_manifest
assert COMFY_NODES.count(MeshTextureToImage) == 1
schema = MeshTextureToImage.schema()
assert [slot.id for slot in schema.inputs] == ['mesh']
assert schema.inputs[0].type.types == ('MESH',)
assert [slot.id for slot in schema.outputs] == [
    'base_color', 'metallic', 'roughness', 'occlusion', 'normal_map'
]
assert all(slot.type.types == ('IMAGE',) for slot in schema.outputs)
manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')
assert manifest.namespaces.count('MeshTextureToImage') == 1
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
