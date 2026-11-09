"""Native MediaPipe declarations preserve the fixed reference without Torch."""

import json
import subprocess
import sys
from pathlib import Path

from dinkster_api.v1 import schema_to_wire
from dinkster_schema import claim_covers
from dinkster_vision_mediapipe import NODES
from dinkster_vision_mediapipe.mediapipe import MediaPipeFaceMask, MediaPipeFaceMeshVisualize
from dinkster_vision_mediapipe.mediapipe_detection import (
    LoadMediaPipeFaceLandmarker,
    MediaPipeFaceLandmarker,
)
from dinkster_workers import load_manifest

from tools.compare_nodes import compare


def test_mediapipe_schema_and_registration() -> None:
    reference = json.loads(Path("tools/data/comfy_core_reference.json").read_text())
    manifest = load_manifest("packages/dinkster-nodes-vision-mediapipe/dinkster-pack.toml")
    for node, suffix in (
        (MediaPipeFaceMeshVisualize, "media_pipe_face_mesh_visualize"),
        (MediaPipeFaceMask, "media_pipe_face_mask"),
        (LoadMediaPipeFaceLandmarker, "load_media_pipe_face_landmarker"),
        (MediaPipeFaceLandmarker, "media_pipe_face_landmarker"),
    ):
        schema = node.define_schema()
        expected_type = f"dinkster.vision.mediapipe.{suffix}"
        row = next(row for row in reference["nodes"] if row["class_name"] == node.__name__)
        assert compare(row, schema_to_wire(schema), expected_node_type=expected_type) == []
        assert NODES.count(node) == 1
        assert sum(claim_covers(claim, schema.node_type) for claim in manifest.namespaces) == 1


def test_mediapipe_discovery_is_torch_free() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['dinkster_inference'] = None\n"
            "from dinkster_vision_mediapipe import NODES\n"
            "modules = ('.mediapipe', '.mediapipe_detection')\n"
            "nodes = [n for n in NODES if n.__module__.endswith(modules)]\n"
            "assert len(nodes) == 4\n"
            "assert all(n.define_schema() for n in nodes)\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
