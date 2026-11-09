"""Drawing discovery preserves canonical types without tensor imports."""

import subprocess
import sys


def test_sdpose_drawing_schema_is_torch_free() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_api.v1 import IMAGE_TYPE, POSE_KEYPOINT_TYPE, TypeExpr
from dinkster_vision_sdpose import NODES
from dinkster_vision_sdpose.sdpose import SDPoseDrawKeypoints
from dinkster_workers import load_manifest
s = SDPoseDrawKeypoints.define_schema()
assert SDPoseDrawKeypoints in NODES
assert s.node_type == 'dinkster.vision.sdpose.sd_pose_draw_keypoints'
assert load_manifest(
    'packages/dinkster-nodes-vision-sdpose/dinkster-pack.toml'
).namespaces == ('dinkster.vision.sdpose',)
assert [p.id for p in s.inputs] == ['keypoints', 'draw_body', 'draw_hands', 'draw_face',
    'draw_feet', 'stick_width', 'face_point_size', 'score_threshold', 'draw_head']
assert s.inputs[0].type == TypeExpr.concrete(POSE_KEYPOINT_TYPE)
assert s.outputs[0].type == TypeExpr.concrete(IMAGE_TYPE)
assert 'dinkster_vision_sdpose.pose_draw' not in sys.modules
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
