"""Public Gaussian splat inspection contracts without Torch execution."""

import tomllib
from pathlib import Path

from dinkster_api.v1 import CORE_INT, SPLAT_TYPE, schema_from_wire, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.gaussian_splat import GetSplatCount


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
