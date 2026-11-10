"""Canonical extraction discovery stays free of tensor runtime imports."""

import subprocess
import sys


def test_extractor_schema_is_torch_free() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_api.v1 import POSE_KEYPOINT_TYPE, TypeExpr
from dinkster_vision_sdpose import NODES
from dinkster_vision_sdpose.sdpose_extract import SDPoseKeypointExtractor
from dinkster_workers import load_manifest
s = SDPoseKeypointExtractor.define_schema()
assert SDPoseKeypointExtractor in NODES
assert s.node_type == 'dinkster.vision.sdpose.sd_pose_keypoint_extractor'
assert load_manifest(
    'packages/dinkster-nodes-vision-sdpose/dinkster-pack.toml'
).namespaces == ('dinkster.vision.sdpose',)
assert [i.id for i in s.inputs] == ['model', 'vae', 'image', 'batch_size', 'bboxes']
assert not s.inputs[-1].required and s.inputs[-1].force_input
assert s.outputs[0].type == TypeExpr.concrete(POSE_KEYPOINT_TYPE)
""",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
