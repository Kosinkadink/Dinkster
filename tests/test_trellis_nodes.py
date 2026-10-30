"""Trellis stage declarations stay independent of the tensor runtime."""

from __future__ import annotations

import subprocess
import sys


def test_texture_stage_schema_is_cold_and_registered() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['dinkster_inference'] = None\n"
            "from dinkster_nodes_comfy import COMFY_NODES\n"
            "from dinkster_nodes_comfy.trellis import Trellis2TextureStage\n"
            "from dinkster_workers import load_manifest\n"
            "schema = Trellis2TextureStage.define_schema()\n"
            "manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')\n"
            "assert COMFY_NODES.count(Trellis2TextureStage) == 1\n"
            "assert manifest.namespaces.count(schema.node_type) == 1\n"
            "assert [i.id for i in schema.inputs] == ['positive','negative','shape_latent']\n"
            "assert [o.id for o in schema.outputs] == ['positive','negative','LATENT']\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
