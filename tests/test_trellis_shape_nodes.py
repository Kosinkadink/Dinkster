"""Trellis shape stage declarations are registered and Torch-free."""

import subprocess
import sys


def test_shape_stage_schema_is_cold_and_registered_once() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['dinkster_inference'] = None\n"
            "from dinkster_nodes_comfy import COMFY_NODES\n"
            "from dinkster_nodes_comfy.trellis_shape import Trellis2ShapeStage\n"
            "from dinkster_workers import load_manifest\n"
            "manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')\n"
            "schema = Trellis2ShapeStage.define_schema()\n"
            "assert COMFY_NODES.count(Trellis2ShapeStage) == 1\n"
            "assert manifest.namespaces.count('Trellis2ShapeStage') == 1\n"
            "assert [i.id for i in schema.inputs] == ['positive', 'negative', 'voxel']\n"
            "assert [o.id for o in schema.outputs] == ['positive', 'negative', 'LATENT']\n",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
