"""Torch-free autoregressive sampler schema and discovery."""

import subprocess
import sys

from dinkster_api.v1 import (
    CORE_INT,
    SAMPLER_TYPE,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.ar_video import SamplerARVideo
from dinkster_workers import load_manifest


def test_ar_video_sampler_schema_and_cold_discovery() -> None:
    schema = SamplerARVideo.schema()
    assert schema.node_type == "SamplerARVideo"
    assert schema.display_name == "Sampler AR Video"
    assert schema.category == "model/sampling/samplers"
    (input_,) = schema.inputs
    assert input_.id == "num_frame_per_block"
    assert input_.type == TypeExpr.concrete(CORE_INT)
    assert input_.default == 1
    assert input_.widget == NumberWidget(min=1, max=64)
    assert input_.doc == (
        "Frames per autoregressive block. 1 = framewise, "
        "3 = chunkwise. Must match the checkpoint's training mode."
    )
    assert schema.outputs == (OutputSpec("SAMPLER", TypeExpr.concrete(SAMPLER_TYPE)),)
    assert schema_from_wire(schema_to_wire(schema)) == schema
    assert SamplerARVideo in COMFY_NODES
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dinkster_nodes_comfy import COMFY_NODES; "
            "assert any(n.schema().node_type == 'SamplerARVideo' for n in COMFY_NODES); "
            "assert 'torch' not in sys.modules; assert 'dinkster_inference' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_ar_video_sampler_complete_registry_matches_manifest() -> None:
    manifest = load_manifest("packages/dinkster-nodes-comfy/dinkster-pack.toml")
    assert tuple(node.schema().node_type for node in COMFY_NODES) == manifest.namespaces
