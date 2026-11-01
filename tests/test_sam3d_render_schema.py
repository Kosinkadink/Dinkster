"""SAM3D renderer discovery preserves cold imports and ordered native claims."""

import json
import subprocess
import sys
from pathlib import Path

from dinkster_api.v1 import schema_to_wire
from dinkster_nodes_comfy.sam3d_render import SAM3DBody_Render

from tools.compare_nodes import compare


def test_render_schema_matches_committed_reference() -> None:
    reference = json.loads(Path("tools/data/comfy_core_reference.json").read_text())
    row = next(row for row in reference["nodes"] if row["class_name"] == "SAM3DBody_Render")
    assert compare(row, schema_to_wire(SAM3DBody_Render.define_schema())) == []


def test_render_schema_is_cold_and_registered() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_api.v1 import IMAGE_TYPE, MHR_POSE_DATA_TYPE, KIMODO_POSE_DATA_TYPE, TypeExpr
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.sam3d_render import SAM3DBody_Render
from dinkster_workers import load_manifest
schema = SAM3DBody_Render.define_schema()
assert SAM3DBody_Render in COMFY_NODES
manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')
assert tuple(node.define_schema().node_type for node in COMFY_NODES) == tuple(
    name for name in manifest.namespaces if name != 'devices')
assert [entry.id for entry in schema.input_entries] == [
    'pose_data', 'background', 'width', 'height', 'camera_info', 'render_style']
assert schema.inputs[0].type == TypeExpr.union(MHR_POSE_DATA_TYPE, KIMODO_POSE_DATA_TYPE)
assert schema.outputs[0].type == TypeExpr.concrete(IMAGE_TYPE)
assert schema.outputs[0].id == 'image'
assert [option.key for option in schema.combos[0].options] == [
    'mesh', 'silhouette', 'openpose_2d', 'openpose_3d', 'scail']
shader = schema.combos[0].options[0].inputs[0]
assert [option.key for option in shader.options] == [
    'default', 'normals', 'rainbow', 'rainbow_face_normal', 'rainbow_face_semantic', 'depth']
assert 'dinkster_nodes_comfy.sam3d_rasterizer' not in sys.modules
assert 'dinkster_nodes_comfy.sam3d_openpose' not in sys.modules
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
