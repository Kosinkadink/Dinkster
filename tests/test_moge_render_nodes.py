"""MoGe rendering declarations stay public and Torch-free."""

import subprocess
import sys


def test_moge_render_schema_is_cold_and_registered_once() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.moge_render import MoGeRender
from dinkster_workers import load_manifest
manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')
schema = MoGeRender.define_schema()
assert COMFY_NODES.count(MoGeRender) == 1
assert manifest.namespaces.count('MoGeRender') == 1
assert [i.id for i in schema.inputs] == ['moge_geometry', 'output']
assert [o.id for o in schema.outputs] == ['IMAGE']
assert schema.inputs[1].default == 'depth'
assert schema.inputs[1].widget.options == (
    'depth', 'depth_colored', 'normal_opengl', 'normal_directx', 'mask'
)
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
