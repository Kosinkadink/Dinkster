"""Torch-free SeedVR2 preprocessing schema and discovery."""

from dinkster_api.v1 import schema_from_wire, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.seedvr_preprocess import SeedVR2Preprocess


def test_seedvr_preprocess_schema() -> None:
    schema = SeedVR2Preprocess.schema()
    assert schema.node_type == "SeedVR2Preprocess"
    assert schema.inputs[0].id == "resized_images"
    assert schema.inputs[0].type.types == ("IMAGE",)
    assert schema.outputs[0].id == "images"
    assert schema.outputs[0].type.types == ("IMAGE",)
    assert schema_from_wire(schema_to_wire(schema)) == schema
    assert SeedVR2Preprocess in COMFY_NODES
