"""The complete manifest preserves the public node registry's order."""

from dataclasses import replace

import pytest
from dinkster_api.v1 import schema_from_wire, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_schema import schema_signature
from dinkster_workers import load_manifest


def test_manifest_claims_match_the_complete_registry() -> None:
    manifest = load_manifest("packages/dinkster-nodes-comfy/dinkster-pack.toml")
    assert ("devices", *(node.schema().node_type for node in COMFY_NODES)) == manifest.namespaces


@pytest.mark.parametrize(
    ("node_type", "family", "template", "flat_port", "inputs", "outputs"),
    [
        ("BatchImagesNode", "images", "image", "", (), ("IMAGE",)),
        ("BatchMasksNode", "masks", "mask", "", (), ("MASK",)),
        ("BatchLatentsNode", "latents", "latent", "", (), ("LATENT",)),
        ("ComfyAndNode", "values", "value", "values.values", (), ("BOOLEAN",)),
        ("ComfyOrNode", "values", "value", "values.values", (), ("BOOLEAN",)),
        ("CreateList", "inputs", "input", "", (), ("list",)),
        ("ComfyMathExpression", "values", "value", "", ("expression",), ("FLOAT", "INT", "BOOL")),
    ],
)
def test_canonical_families_ship_complete_historical_migrations(
    node_type: str,
    family: str,
    template: str,
    flat_port: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
) -> None:
    node = next(node for node in COMFY_NODES if node.schema().node_type == node_type)
    schema = node.schema()
    assert schema_to_wire(schema).get("replacements") == [
        {
            "from": node_type,
            "migration": {"historicalInputs": [family]},
            "cases": [
                {
                    "to": node_type,
                    **(
                        {"inputs": {name: {"kind": "copy", "input": name} for name in inputs}}
                        if inputs
                        else {}
                    ),
                    "inputFamilies": {
                        family: {
                            "kind": "copy",
                            "sourceFamily": family,
                            "sourceTemplate": {
                                "inputs": [template],
                                **({"memberPort": flat_port} if flat_port else {}),
                            },
                            "memberIds": "targetPositions",
                            "inputs": {template: {"kind": "copy", "input": template}},
                        },
                    },
                    "outputs": {name: name for name in outputs},
                },
            ],
        },
    ]
    assert schema_from_wire(schema_to_wire(schema)) == schema
    assert schema_signature(schema) == schema_signature(replace(schema, replacements=()))
