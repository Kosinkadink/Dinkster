"""Trellis upsample declarations remain registered and Torch-free."""

import subprocess
import sys


def test_upsample_stage_schema_is_cold_and_registered_once() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['dinkster_inference'] = None\n"
            "from dinkster_nodes_comfy import COMFY_NODES\n"
            "from dinkster_nodes_comfy.trellis_upsample import Trellis2UpsampleStage\n"
            "from dinkster_workers import load_manifest\n"
            "manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')\n"
            "schema = Trellis2UpsampleStage.define_schema()\n"
            "assert COMFY_NODES.count(Trellis2UpsampleStage) == 1\n"
            "assert manifest.namespaces.count('Trellis2UpsampleStage') == 1\n"
            "assert [i.id for i in schema.inputs] == "
            "['positive', 'negative', 'shape_latent', 'vae', 'target_resolution']\n"
            "assert [o.id for o in schema.outputs] == ['positive', 'negative', 'LATENT']\n"
            "assert schema.inputs[-1].default == 1024\n"
            "assert schema.inputs[-1].widget.min == 1024\n"
            "assert schema.inputs[-1].widget.max == 2048\n"
            "assert schema.inputs[-1].widget.step == 128\n",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
