"""Native MediaPipe declarations preserve the fixed reference without Torch."""

import json
import subprocess
import sys
from pathlib import Path

from dinkster_api.v1 import schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.mediapipe import MediaPipeFaceMask, MediaPipeFaceMeshVisualize
from dinkster_nodes_comfy.mediapipe_detection import (
    LoadMediaPipeFaceLandmarker,
    MediaPipeFaceLandmarker,
)
from dinkster_workers import load_manifest

from tools.compare_nodes import compare


def test_mediapipe_schema_and_registration() -> None:
    reference = json.loads(Path("tools/data/comfy_core_reference.json").read_text())
    manifest = load_manifest("packages/dinkster-nodes-comfy/dinkster-pack.toml")
    for node in (
        MediaPipeFaceMeshVisualize,
        MediaPipeFaceMask,
        LoadMediaPipeFaceLandmarker,
        MediaPipeFaceLandmarker,
    ):
        schema = node.define_schema()
        row = next(row for row in reference["nodes"] if row["class_name"] == schema.node_type)
        assert compare(row, schema_to_wire(schema)) == []
        assert COMFY_NODES.count(node) == 1
        assert manifest.namespaces.count(schema.node_type) == 1


def test_mediapipe_discovery_is_torch_free() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['dinkster_inference'] = None\n"
            "from dinkster_nodes_comfy import COMFY_NODES\n"
            "modules = ('.mediapipe', '.mediapipe_detection')\n"
            "nodes = [n for n in COMFY_NODES if n.__module__.endswith(modules)]\n"
            "assert len(nodes) == 4\n"
            "assert all(n.define_schema() for n in nodes)\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
