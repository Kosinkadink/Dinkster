"""SAM3D Body schemas remain discoverable without the tensor runtime."""

import json
import subprocess
import sys
from pathlib import Path

from dinkster_api.v1 import AssetWidget, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.sam3d_body import SAM3DBody_Loader, SAM3DBody_Smooth
from dinkster_workers import load_manifest

from tools.compare_nodes import compare


def test_sam3d_smoothing_schema_matches_reference() -> None:
    reference = json.loads(Path("tools/data/comfy_core_reference.json").read_text())
    manifest = load_manifest("packages/dinkster-nodes-comfy/dinkster-pack.toml")
    for node in (SAM3DBody_Loader, SAM3DBody_Smooth):
        schema = node.define_schema()
        row = next(row for row in reference["nodes"] if row["class_name"] == schema.node_type)
        assert compare(row, schema_to_wire(schema)) == []
        assert COMFY_NODES.count(node) == 1
        assert manifest.namespaces.count(schema.node_type) == 1
    widget = SAM3DBody_Loader.define_schema().inputs[0].widget
    assert isinstance(widget, AssetWidget) and widget.kind == "model/detection"


def test_sam3d_discovery_is_tensor_runtime_free() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['scipy'] = None\n"
            "sys.modules['dinkster_inference'] = None\n"
            "from dinkster_nodes_comfy.sam3d_body import SAM3DBody_Loader, SAM3DBody_Smooth\n"
            "assert SAM3DBody_Loader.define_schema().node_type == 'SAM3DBody_Loader'\n"
            "assert SAM3DBody_Smooth.define_schema().node_type == 'SAM3DBody_Smooth'\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
