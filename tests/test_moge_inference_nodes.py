"""MoGe inference declarations remain cold and registered once."""

import subprocess
import sys


def test_moge_inference_cold_schema_and_registration() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.moge_inference import MoGeInference
from dinkster_workers import load_manifest
schema = MoGeInference.define_schema()
manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')
assert COMFY_NODES.count(MoGeInference) == 1
assert manifest.namespaces.count('MoGeInference') == 1
assert [i.id for i in schema.inputs] == [
    'moge_model', 'image', 'resolution_level', 'fov_x_degrees', 'batch_size',
    'force_projection', 'apply_mask', 'refine_steps'
]
assert [i.default for i in schema.inputs[2:]] == [9, 0.0, 4, True, True, 3]
assert [o.id for o in schema.outputs] == ['moge_geometry']
assert schema.inputs[3].advanced and schema.inputs[5].advanced
assert schema.inputs[6].advanced and schema.inputs[7].advanced
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
