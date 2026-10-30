"""Torch-free SeedVR2 temporal schemas and discovery."""

from dinkster_api.v1 import LATENT_TYPE, NumberWidget, TypeExpr, schema_from_wire, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.seedvr_temporal import SeedVR2TemporalChunk, SeedVR2TemporalMerge


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
