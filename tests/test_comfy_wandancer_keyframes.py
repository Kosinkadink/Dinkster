"""Torch-free WanDancer scalar and list socket contracts."""

import subprocess
import sys

from dinkster_api.v1 import (
    AUDIO_TYPE,
    IMAGE_TYPE,
    MASK_TYPE,
    NumberWidget,
    TypeExpr,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.wandancer_keyframes import WANDANCER_KEYFRAME_NODES
from dinkster_workers import load_manifest


def test_wandancer_keyframe_schemas_and_cold_discovery() -> None:
    for node, list_output in zip(WANDANCER_KEYFRAME_NODES, (False, True), strict=True):
        schema = node.schema()
        assert schema.category == "image/video"
        assert [i.id for i in schema.inputs] == [
            "images",
            "segment_length",
            "num_segments" if list_output else "segment_index",
            "audio",
        ]
        assert all(i.required for i in schema.inputs)
        assert schema.inputs[1].default == 149
        assert schema.inputs[1].widget == NumberWidget(min=1, max=10000)
        assert schema.inputs[2].default == (1 if list_output else 0)
        assert schema.inputs[2].widget == NumberWidget(min=1 if list_output else 0, max=100)
        outputs = [TypeExpr.concrete(kind) for kind in (IMAGE_TYPE, MASK_TYPE, AUDIO_TYPE)]
        assert [o.id for o in schema.outputs] == [
            "keyframes_sequence",
            "keyframes_mask",
            "audio_segment",
        ]
        assert [o.type for o in schema.outputs] == [
            TypeExpr.list_of(t) if list_output else t for t in outputs
        ]
        assert schema_from_wire(schema_to_wire(schema)) == schema
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dinkster_nodes_comfy import COMFY_NODES; "
            "names = {n.schema().node_type for n in COMFY_NODES}; "
            "assert {'WanDancerPadKeyframes','WanDancerPadKeyframesList'} <= names; "
            "assert 'torch' not in sys.modules; assert 'dinkster_inference' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_wandancer_complete_registry_matches_manifest() -> None:
    manifest = load_manifest("packages/dinkster-nodes-comfy/dinkster-pack.toml")
    assert tuple(node.schema().node_type for node in COMFY_NODES) == manifest.namespaces
