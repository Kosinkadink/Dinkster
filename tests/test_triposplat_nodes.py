"""TripoSplat preprocessing declares its schema without loading tensor runtimes."""

from __future__ import annotations

import subprocess
import sys


def test_preprocess_schema_is_cold_and_registered() -> None:
    # Source contract: TripoSplatPreprocessImage at F bc5f880d, define_schema().
    # These are declarations, not executed ComfyUI output constants or goldens.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['dinkster_inference'] = None\n"
            "from dinkster_nodes_comfy import COMFY_NODES\n"
            "from dinkster_nodes_comfy.triposplat import TripoSplatPreprocessImage\n"
            "from dinkster_workers import load_manifest\n"
            "from dinkster_schema import schema_to_wire\n"
            "schema = TripoSplatPreprocessImage.define_schema()\n"
            "manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')\n"
            "assert TripoSplatPreprocessImage in COMFY_NODES\n"
            "assert schema.node_type in manifest.namespaces\n"
            "assert [i.id for i in schema.inputs] == ['image','mask','erode_radius','size']\n"
            "assert [o.id for o in schema.outputs] == ['image']\n"
            "assert schema_to_wire(schema)['interface'][-1]['type']['types'] == ['IMAGE']\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_conditioning_schema_is_cold_and_registered() -> None:
    # Source contract: TripoSplatConditioning at F bc5f880d, define_schema().
    # These declarations are not executed ComfyUI output constants or goldens.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['dinkster_inference'] = None\n"
            "from dinkster_nodes_comfy import COMFY_NODES\n"
            "from dinkster_nodes_comfy.triposplat import TripoSplatConditioning\n"
            "from dinkster_workers import load_manifest\n"
            "schema = TripoSplatConditioning.define_schema()\n"
            "manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')\n"
            "assert TripoSplatConditioning in COMFY_NODES\n"
            "assert schema.node_type in manifest.namespaces\n"
            "assert [i.id for i in schema.inputs] == ['clip_vision','vae','image']\n"
            "assert [o.id for o in schema.outputs] == ['positive','negative','latent']\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_decode_schema_is_cold_and_registered() -> None:
    # Source contract: VAEDecodeTripoSplat at F bc5f880d, define_schema().
    # These declarations are not executed ComfyUI output constants or goldens.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['dinkster_inference'] = None\n"
            "from dinkster_nodes_comfy import COMFY_NODES\n"
            "from dinkster_nodes_comfy.triposplat import VAEDecodeTripoSplat\n"
            "from dinkster_workers import load_manifest\n"
            "schema = VAEDecodeTripoSplat.define_schema()\n"
            "manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')\n"
            "assert VAEDecodeTripoSplat in COMFY_NODES\n"
            "assert schema.node_type in manifest.namespaces\n"
            "assert [i.id for i in schema.inputs] == ['samples','vae','num_gaussians','seed']\n"
            "assert [o.id for o in schema.outputs] == ['splat']\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
