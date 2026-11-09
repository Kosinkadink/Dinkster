"""The native DA3 declarations retain the fixed reference interface without Torch."""

import json
import subprocess
import sys
from pathlib import Path

from dinkster_api.v1 import AssetWidget, schema_to_wire
from dinkster_vision_depth_anything_3 import NODES
from dinkster_vision_depth_anything_3.depth_anything_3 import (
    DA3GeometryToMesh,
    DA3Inference,
    DA3Render,
    LoadDA3Model,
)
from dinkster_workers import load_manifest

from tools.compare_nodes import compare


def test_da3_schema_agrees_with_committed_reference() -> None:
    reference = json.loads(Path("tools/data/comfy_core_reference.json").read_text())
    manifest = load_manifest("packages/dinkster-nodes-vision-depth-anything-3/dinkster-pack.toml")
    for node, node_type in (
        (LoadDA3Model, "dinkster.vision.depth.anything.3.load_da3_model"),
        (DA3Inference, "dinkster.vision.depth.anything.3.da3_inference"),
        (DA3Render, "dinkster.vision.depth.anything.3.da3_render"),
        (DA3GeometryToMesh, "dinkster.vision.depth.anything.3.da3_geometry_to_mesh"),
    ):
        schema = node.define_schema()
        row = next(row for row in reference["nodes"] if row["class_name"] == node.__name__)
        assert compare({**row, "class_name": node_type}, schema_to_wire(schema)) == []
        assert schema.node_type == node_type
        assert NODES.count(node) == 1
        assert manifest.namespaces == ("dinkster.vision.depth.anything.3",)
    widget = LoadDA3Model.define_schema().inputs[0].widget
    assert isinstance(widget, AssetWidget) and widget.kind == "model/geometry-estimation"


def test_da3_discovery_does_not_import_tensor_runtime() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['dinkster_inference'] = None\n"
            "from dinkster_vision_depth_anything_3 import NODES\n"
            "nodes = [n for n in NODES if n.__module__.endswith('.depth_anything_3')]\n"
            "assert len(nodes) == 4\n"
            "assert all(n.define_schema() for n in nodes)\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
