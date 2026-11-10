"""SAM3D Body schemas remain discoverable without the tensor runtime."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from dinkster_api.v1 import AssetWidget, schema_to_wire
from dinkster_vision_sam3d_body import NODES
from dinkster_vision_sam3d_body.sam3d_body import SAM3DBody_Loader, SAM3DBody_Smooth
from dinkster_vision_sam3d_body.sam3d_export import BuildPoseFile
from dinkster_vision_sam3d_body.sam3d_expression import SAM3DBody_FaceExpression
from dinkster_vision_sam3d_body.sam3d_prediction import SAM3DBody_Predict
from dinkster_workers import load_manifest

from tools.compare_nodes import compare


@pytest.mark.parametrize(
    ("node", "node_type"),
    (
        (SAM3DBody_Loader, "dinkster.vision.sam3d.body.sam3_d_body_loader"),
        (SAM3DBody_Predict, "dinkster.vision.sam3d.body.sam3_d_body_predict"),
        (SAM3DBody_FaceExpression, "dinkster.vision.sam3d.body.sam3_d_body_face_expression"),
        (SAM3DBody_Smooth, "dinkster.vision.sam3d.body.sam3_d_body_smooth"),
        (BuildPoseFile, "dinkster.vision.sam3d.body.build_pose_file"),
    ),
)
def test_sam3d_schema_matches_reference(node: type, node_type: str) -> None:
    reference = json.loads(Path("tools/data/comfy_core_reference.json").read_text())
    schema = node.define_schema()
    row = next(row for row in reference["nodes"] if row["class_name"] == node.__name__)
    assert compare({**row, "class_name": node_type}, schema_to_wire(schema)) == []
    assert schema.node_type == node_type
    assert NODES.count(node) == 1
    manifest = load_manifest("packages/dinkster-nodes-vision-sam3d-body/dinkster-pack.toml")
    assert manifest.namespaces == ("dinkster.vision.sam3d.body",)


def test_sam3d_loader_uses_detection_model_asset() -> None:
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
            "from dinkster_vision_sam3d_body import (\n"
            "    sam3d_body, sam3d_export, sam3d_expression, sam3d_prediction\n"
            ")\n"
            "prefix = 'dinkster.vision.sam3d.body.'\n"
            "classes = (sam3d_body.SAM3DBody_Loader, sam3d_prediction.SAM3DBody_Predict,\n"
            "           sam3d_expression.SAM3DBody_FaceExpression, sam3d_body.SAM3DBody_Smooth,\n"
            "           sam3d_export.BuildPoseFile)\n"
            "assert tuple(node.define_schema().node_type for node in classes) == (\n"
            "    prefix + 'sam3_d_body_loader', prefix + 'sam3_d_body_predict',\n"
            "    prefix + 'sam3_d_body_face_expression', prefix + 'sam3_d_body_smooth',\n"
            "    prefix + 'build_pose_file')\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
