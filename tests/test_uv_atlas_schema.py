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
from dinkster_workers import load_manifest
assert COMFY_NODES.count(RenderUVAtlas) == 1
schema = RenderUVAtlas.schema()
assert [slot.id for slot in schema.inputs] == ['mesh', 'resolution']
assert schema.inputs[0].type.types == ('MESH',)
assert schema.outputs[0].type.types == ('IMAGE',)
assert schema.inputs[1].default == 1024
manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')
assert manifest.namespaces.count('RenderUVAtlas') == 1
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
