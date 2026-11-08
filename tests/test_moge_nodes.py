"""Native MoGe camera declarations do not initialize the tensor runtime."""

import subprocess
import sys


def test_moge_fov_schema_is_cold_and_registered() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['dinkster_inference'] = None\n"
            "from dinkster_vision_moge import NODES\n"
            "from dinkster_vision_moge.moge import MoGeGeometryToFOV\n"
            "from dinkster_workers import load_manifest\n"
            "schema = MoGeGeometryToFOV.define_schema()\n"
            "manifest = load_manifest('packages/dinkster-nodes-vision-moge/dinkster-pack.toml')\n"
            "assert NODES.count(MoGeGeometryToFOV) == 1\n"
            "assert manifest.namespaces.count('dinkster.vision.moge') == 1\n"
            "assert schema.node_type == 'dinkster.vision.moge.mo_ge_geometry_to_fov'\n"
            "assert [i.id for i in schema.inputs] == ['moge_geometry', 'axis', 'unit']\n"
            "assert [o.id for o in schema.outputs] == ['fov', 'focal_pixels']\n",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
