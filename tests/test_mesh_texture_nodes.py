"""Mesh texture discovery remains available without the tensor runtime."""

import subprocess
import sys


def test_mesh_texture_schema_is_cold_and_registered() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.mesh_textures import MeshTextureToImage
from dinkster_workers import load_manifest
schema = MeshTextureToImage.define_schema()
assert COMFY_NODES.count(MeshTextureToImage) == 1
manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')
assert manifest.namespaces.count(schema.node_type) == 1
assert [i.id for i in schema.inputs] == ['mesh']
assert [o.id for o in schema.outputs] == [
    'base_color', 'metallic', 'roughness', 'occlusion', 'normal_map'
]
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
