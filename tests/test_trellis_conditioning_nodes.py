"""Trellis vision conditioning declares native types without a tensor runtime."""

import subprocess
import sys


def test_conditioning_schema_and_native_manifest_are_cold() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['dinkster_inference'] = None\n"
            "from dinkster_nodes_comfy import COMFY_NODES\n"
            "from dinkster_nodes_comfy.trellis_conditioning import Trellis2Conditioning\n"
            "from dinkster_workers import load_manifest\n"
            "schema = Trellis2Conditioning.define_schema()\n"
            "manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')\n"
            "assert COMFY_NODES.count(Trellis2Conditioning) == 1\n"
            "assert manifest.namespaces.count(schema.node_type) == 1\n"
            "assert [i.id for i in schema.inputs] == ['clip_vision_model', 'image']\n"
            "assert [o.id for o in schema.outputs] == ['positive', 'negative']\n",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
