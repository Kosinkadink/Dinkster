"""Public Gaussian splat inspection contracts without Torch execution."""

import tomllib
from pathlib import Path

from dinkster_api.v1 import CORE_FLOAT, CORE_INT, SPLAT_TYPE, schema_from_wire, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.gaussian_splat import GetSplatCount, TransformSplat


def test_splat_count_schema_and_production_claim_are_canonical() -> None:
    schema = GetSplatCount.schema()
    assert schema_from_wire(schema_to_wire(schema)) == schema
    assert schema.node_type == "GetSplatCount" and schema.category == "3d/splat"
    assert [(port.id, port.type.types, port.required) for port in schema.inputs] == [
        ("splat", (SPLAT_TYPE,), True)
    ]
    assert [(port.id, port.type.types) for port in schema.outputs] == [
        ("splat", (SPLAT_TYPE,)),
        ("count", (CORE_INT,)),
    ]
    assert sum(node is GetSplatCount for node in COMFY_NODES) == 1
    manifest = (
        Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
    )
    assert tomllib.loads(manifest.read_text())["pack"]["namespaces"].count("GetSplatCount") == 1


def test_transform_splat_public_schema_and_production_claim() -> None:
    schema = TransformSplat.schema()
    assert schema_from_wire(schema_to_wire(schema)) == schema
    assert schema.node_type == "TransformSplat" and schema.category == "3d/splat"
    assert schema.inputs[0].type.types == (SPLAT_TYPE,) and schema.inputs[0].required
    assert [(port.id, port.default, port.type.types) for port in schema.inputs[1:]] == [
        (f"{operation}_{axis}", default, (CORE_FLOAT,))
        for operation, default in (("translate", 0.0), ("rotate", 0.0), ("scale", 1.0))
        for axis in "xyz"
    ]
    assert [(port.id, port.type.types) for port in schema.outputs] == [("splat", (SPLAT_TYPE,))]
    assert sum(node is TransformSplat for node in COMFY_NODES) == 1
    manifest = (
        Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
    )
    assert tomllib.loads(manifest.read_text())["pack"]["namespaces"].count("TransformSplat") == 1
