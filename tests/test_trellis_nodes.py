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


def test_texture_decoder_schema_is_cold_and_registered() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['dinkster_inference'] = None\n"
            "from dinkster_api.v1 import LATENT_TYPE, VAE_TYPE, SHAPE_SUBDIVIDES_TYPE, VOXEL_TYPE\n"
            "from dinkster_nodes_comfy import COMFY_NODES\n"
            "from dinkster_nodes_comfy.trellis import VaeDecodeTextureTrellis\n"
            "from dinkster_workers import load_manifest\n"
            "schema = VaeDecodeTextureTrellis.define_schema()\n"
            "manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')\n"
            "assert COMFY_NODES.count(VaeDecodeTextureTrellis) == 1\n"
            "assert manifest.namespaces.count(schema.node_type) == 1\n"
            "assert [i.id for i in schema.inputs] == ['samples','vae','shape_subdivides']\n"
            "from dinkster_api.v1 import TypeExpr\n"
            "assert [i.type for i in schema.inputs] == [TypeExpr.concrete(t) "
            "for t in (LATENT_TYPE,VAE_TYPE,SHAPE_SUBDIVIDES_TYPE)]\n"
            "assert [(o.id,o.type) for o in schema.outputs] == "
            "[('voxel_colors',TypeExpr.concrete(VOXEL_TYPE))]\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
