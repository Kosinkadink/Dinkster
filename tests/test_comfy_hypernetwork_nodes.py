"""Torch-free public hypernetwork schema and registration."""

import tomllib
from pathlib import Path

from dinkster_api.v1 import ASSET_TYPE, CORE_FLOAT, MODEL_TYPE, AssetWidget, NumberWidget, TypeExpr
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.hypernetwork import HypernetworkLoader


def test_hypernetwork_schema_and_cold_registration() -> None:
    schema = HypernetworkLoader.schema()
    assert schema.node_type == "HypernetworkLoader"
    assert schema.category == "model/loaders"
    assert schema.display_name == "Load Hypernetwork"
    assert tuple(port.id for port in schema.inputs) == ("model", "hypernetwork_name", "strength")
    assert tuple(port.type for port in schema.inputs) == tuple(
        TypeExpr.concrete(kind) for kind in (MODEL_TYPE, ASSET_TYPE, CORE_FLOAT)
    )
    assert all(port.required for port in schema.inputs)
    assert schema.inputs[1].widget == AssetWidget(
        kind="model/hypernetwork", accept=("application/octet-stream",)
    )
    assert schema.inputs[2].default == 1.0
    assert schema.inputs[2].widget == NumberWidget(min=-10.0, max=10.0, step=0.01)
    assert [(port.id, port.type) for port in schema.outputs] == [
        ("MODEL", TypeExpr.concrete(MODEL_TYPE))
    ]
    root = Path(__file__).resolve().parents[1]
    manifest = tomllib.loads(
        (root / "packages/dinkster-nodes-comfy/dinkster-pack.toml").read_text()
    )
    assert [node.schema().node_type for node in COMFY_NODES] == manifest["pack"]["namespaces"]
    assert HypernetworkLoader in COMFY_NODES
