"""Torch-free LTXV duration schema and native discovery."""

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_FLOAT,
    CORE_INT,
    MODEL_PATCH_TYPE,
    MODEL_TYPE,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.ltx_duration import LTXVDurationPredictor
from dinkster_workers import load_manifest


def test_duration_predictor_schema_and_complete_manifest() -> None:
    assert LTXVDurationPredictor in COMFY_NODES
    schema = LTXVDurationPredictor.schema()
    assert schema.node_type == "LTXVDurationPredictor"
    assert schema.display_name == "LTXV Duration Predictor"
    assert schema.category == "model/conditioning/ltxv"
    assert tuple((i.id, i.type.types) for i in schema.inputs) == (
        ("model", (MODEL_TYPE,)),
        ("positive", (CONDITIONING_TYPE,)),
        ("duration_head", (MODEL_PATCH_TYPE,)),
        ("frame_rate", (CORE_FLOAT,)),
        ("min_seconds", (CORE_FLOAT,)),
        ("max_seconds", (CORE_FLOAT,)),
    )
    assert tuple(i.default for i in schema.inputs[3:]) == (24.0, 1.0, 20.0)
    assert tuple((o.id, o.type.types) for o in schema.outputs) == (
        ("num_frames", (CORE_INT,)),
        ("seconds", (CORE_FLOAT,)),
    )
    assert schema_from_wire(schema_to_wire(schema)) == schema
    assert ("devices", *(n.schema().node_type for n in COMFY_NODES)) == load_manifest(
        "packages/dinkster-nodes-comfy/dinkster-pack.toml"
    ).namespaces
