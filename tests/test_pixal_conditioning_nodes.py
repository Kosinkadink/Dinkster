"""Pixal3D declarations stay cold and registered exactly once."""

import subprocess
import sys


def test_pixal_conditioning_schemas_are_cold() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['dinkster_inference'] = None\n"
            "from dinkster_nodes_comfy import COMFY_NODES\n"
            "from dinkster_nodes_comfy.pixal_conditioning import (\n"
            " Pixal3DConditioning, Pixal3DMultiViewConditioning)\n"
            "from dinkster_workers import load_manifest\n"
            "manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')\n"
            "for cls in (Pixal3DConditioning, Pixal3DMultiViewConditioning):\n"
            " schema = cls.define_schema()\n"
            " assert COMFY_NODES.count(cls) == 1\n"
            " assert manifest.namespaces.count(schema.node_type) == 1\n"
            " assert [o.id for o in schema.outputs] == ['positive','negative']\n"
            " assert schema.inputs[0].id == 'clip_vision_model'\n",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
