"""Panorama geometry declarations stay cold and registered once."""

import subprocess
import sys


def test_moge_panorama_cold_schema_and_registration() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.moge_panorama import MoGePanoramaInference
from dinkster_workers import load_manifest
schema = MoGePanoramaInference.define_schema()
manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')
assert COMFY_NODES.count(MoGePanoramaInference) == 1
assert manifest.namespaces.count('MoGePanoramaInference') == 1
assert [i.id for i in schema.inputs] == [
    'moge_model', 'image', 'resolution_level', 'split_resolution',
    'merge_resolution', 'batch_size', 'refine_steps'
]
assert [i.default for i in schema.inputs[2:]] == [9, 512, 1920, 4, 3]
assert schema.inputs[-1].advanced
assert [o.id for o in schema.outputs] == ['moge_geometry']
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
