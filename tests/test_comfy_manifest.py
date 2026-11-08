"""ComfyUI family conversion targets the owning first-party pack."""

from dataclasses import replace

import pytest
from dinkster_api.v1 import schema_from_wire, schema_to_wire
from dinkster_image.ported.post_processing import BatchImagesNode, BatchMasksNode
from dinkster_latent.post_processing import BatchLatentsNode
from dinkster_schema import Node, schema_signature, validate_replacement_references
from dinkster_utilities.list_utilities import CreateList
from dinkster_utilities.logic import ComfyAndNode, ComfyOrNode
from dinkster_utilities.math import ComfyMathExpression


@pytest.mark.parametrize(
    ("node", "target", "family", "template", "inputs", "outputs"),
    [
        (BatchImagesNode, "dinkster.image.batch_images_node", "images", "image", (), ("IMAGE",)),
        (BatchMasksNode, "dinkster.image.batch_masks_node", "masks", "mask", (), ("MASK",)),
        (
            BatchLatentsNode,
            "dinkster.latent.batch_latents_node",
            "latents",
            "latent",
            (),
            ("LATENT",),
        ),
        (ComfyAndNode, "dinkster.utilities.comfy_and_node", "values", "value", (), ("BOOLEAN",)),
        (ComfyOrNode, "dinkster.utilities.comfy_or_node", "values", "value", (), ("BOOLEAN",)),
        (CreateList, "dinkster.utilities.create_list", "inputs", "input", (), ("list",)),
        (
            ComfyMathExpression,
            "dinkster.utilities.comfy_math_expression",
            "values",
            "value",
            ("expression",),
            ("FLOAT", "INT", "BOOL"),
        ),
    ],
)
def test_comfy_families_convert_to_first_party_ids(
    node: type[Node],
    target: str,
    family: str,
    template: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
) -> None:
    schema = node.schema()
    assert schema.node_type == target
    assert schema_to_wire(schema).get("replacements") == [
        {
            "from": node.__name__,
            "cases": [
                {
                    "to": target,
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
                            },
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
    assert validate_replacement_references({schema.node_type: schema}) == ()
