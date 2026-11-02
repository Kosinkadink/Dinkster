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
            "from dinkster_nodes_comfy.trellis_shape import VaeDecodeShapeTrellis\n"
            "from dinkster_workers import load_manifest\n"
            "manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')\n"
            "schema = Trellis2ShapeStage.define_schema()\n"
            "assert COMFY_NODES.count(Trellis2ShapeStage) == 1\n"
            "assert manifest.namespaces.count('Trellis2ShapeStage') == 1\n"
            "assert [i.id for i in schema.inputs] == ['positive', 'negative', 'voxel']\n"
            "assert [o.id for o in schema.outputs] == ['positive', 'negative', 'LATENT']\n"
            "assert COMFY_NODES.count(VaeDecodeShapeTrellis) == 1\n"
            "assert manifest.namespaces.count('VaeDecodeShapeTrellis') == 1\n"
            "schema = VaeDecodeShapeTrellis.schema()\n"
            "assert [i.type.types for i in schema.inputs] == [('LATENT',), ('VAE',)]\n"
            "assert [o.id for o in schema.outputs] == ['mesh', 'shape_subdivides']\n"
            "assert [o.type.types for o in schema.outputs] == [('MESH',), ('SHAPE_SUBDIVIDES',)]\n",
        ],
        capture_output=True,
        text=True,
        check=True,
    )


def test_structure_decode_schema_is_cold_and_registered_once() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['dinkster_inference'] = None\n"
            "from dinkster_nodes_comfy import COMFY_NODES\n"
            "from dinkster_nodes_comfy.trellis_structure import VaeDecodeStructureTrellis2\n"
            "from dinkster_workers import load_manifest\n"
            "manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')\n"
            "schema = VaeDecodeStructureTrellis2.define_schema()\n"
            "assert COMFY_NODES.count(VaeDecodeStructureTrellis2) == 1\n"
            "assert manifest.namespaces.count('VaeDecodeStructureTrellis2') == 1\n"
            "assert [i.id for i in schema.inputs] == ['samples', 'vae', 'resolution']\n"
            "assert schema.inputs[-1].default == '32'\n"
            "assert schema.inputs[-1].widget.options == ('32', '64')\n"
            "assert [o.id for o in schema.outputs] == ['voxel']\n",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
