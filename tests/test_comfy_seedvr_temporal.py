"""Torch-free SeedVR2 temporal schemas and discovery."""

import subprocess
import sys

from dinkster_api.v1 import LATENT_TYPE, NumberWidget, TypeExpr, schema_from_wire, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.seedvr_temporal import SeedVR2TemporalChunk, SeedVR2TemporalMerge
from dinkster_workers import load_manifest


def test_seedvr_temporal_schemas_are_registered_with_bounded_merge_overlap() -> None:
    chunk = SeedVR2TemporalChunk.schema()
    merge = SeedVR2TemporalMerge.schema()
    assert chunk.node_type == "SeedVR2TemporalChunk"
    assert merge.node_type == "SeedVR2TemporalMerge"
    assert chunk.outputs[0].type == TypeExpr.list_of(TypeExpr.concrete(LATENT_TYPE))
    overlap = merge.inputs[1]
    assert overlap.id == "temporal_overlap"
    assert overlap.force_input
    assert overlap.widget == NumberWidget(min=0, max=16384)
    assert schema_from_wire(schema_to_wire(chunk)) == chunk
    assert schema_from_wire(schema_to_wire(merge)) == merge
    assert (SeedVR2TemporalChunk, SeedVR2TemporalMerge) == tuple(
        node for node in COMFY_NODES if node in {SeedVR2TemporalChunk, SeedVR2TemporalMerge}
    )


def test_seedvr_temporal_manifest_claims_match_the_complete_registry() -> None:
    manifest = load_manifest("packages/dinkster-nodes-comfy/dinkster-pack.toml")
    assert tuple(node.schema().node_type for node in COMFY_NODES) == manifest.namespaces


def test_seedvr_temporal_cold_discovery() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from dinkster_nodes_comfy.seedvr_temporal import "
            "SeedVR2TemporalChunk, SeedVR2TemporalMerge; "
            "SeedVR2TemporalChunk.schema(); SeedVR2TemporalMerge.schema(); "
            "assert 'torch' not in sys.modules; assert 'dinkster_inference' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
