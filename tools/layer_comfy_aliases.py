"""Pinned compositor interfaces and executable layer migration records."""

from __future__ import annotations

import json
import math
from pathlib import Path

from dinkster_schema import InputSpec, NodeSchema, OutputSpec, TypeExpr, schema_to_wire
from dinkster_schema.replace import rule_from_wire, rule_to_wire
from dinkster_values import (
    BOUNDING_BOX_TYPE,
    COMFY_ARRAY_TYPE,
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
)

REFERENCE = "f00bfd610cb001381603669e2cc01160ae37aaf3"
SOURCE_IMAGE = "comfy.IMAGE"
SOURCE_MASK = "comfy.MASK"
SOURCE_LAYERS = "comfy.LAYERS"
SOURCE_COMPOSITOR = "comfy.COMPOSITOR"


def layer_alias_data() -> tuple[list[NodeSchema], list[dict[str, object]]]:
    def port(name: str, type_id: str, default: object = None) -> InputSpec:
        return InputSpec(name, TypeExpr.concrete(type_id), required=False, default=default)

    image = InputSpec("image", TypeExpr.concrete(SOURCE_IMAGE))
    layers = port("layers", SOURCE_LAYERS)
    mask = port("mask", SOURCE_MASK)
    definitions = [
        (
            "AddLayer",
            "dinkster.layers.add",
            (
                layers,
                image,
                mask,
                port("name", CORE_STRING, ""),
                port("x", CORE_INT, 0),
                port("y", CORE_INT, 0),
                port("opacity", CORE_FLOAT, 1.0),
                port("blend_mode", CORE_COMBO, "normal"),
                port("rotation", CORE_FLOAT, 0.0),
                port("width", CORE_INT, 0),
                port("height", CORE_INT, 0),
                port("z_index", CORE_INT, 0),
                port("flip_h", CORE_BOOLEAN, False),
                port("flip_v", CORE_BOOLEAN, False),
            ),
            (OutputSpec("layers", TypeExpr.concrete(SOURCE_LAYERS)),),
        ),
        (
            "LayersFromBoundingBoxes",
            "dinkster.layers.from_bounding_boxes",
            (
                image,
                InputSpec(
                    "bboxes",
                    TypeExpr.union(BOUNDING_BOX_TYPE, COMFY_ARRAY_TYPE, CORE_STRING),
                ),
                mask,
                layers,
                port("crop_to_content", CORE_BOOLEAN, True),
                port("canvas_width", CORE_INT, 0),
                port("canvas_height", CORE_INT, 0),
            ),
            (OutputSpec("layers", TypeExpr.concrete(SOURCE_LAYERS)),),
        ),
        (
            "ImageCompositor",
            "dinkster.image.create_layered",
            (
                InputSpec("layers", TypeExpr.concrete(SOURCE_LAYERS)),
                port("compositor", SOURCE_COMPOSITOR, {}),
            ),
            (
                OutputSpec("image", TypeExpr.concrete(SOURCE_IMAGE)),
                OutputSpec("mask", TypeExpr.concrete(SOURCE_MASK)),
            ),
        ),
    ]
    schemas, records = [], []
    for name, carrier, inputs, outputs in definitions:
        source_type = f"comfy.{name}"
        schemas.append(
            NodeSchema(
                node_type=source_type,
                display_name=name,
                category="comfy/image",
                inputs=inputs,
                outputs=outputs,
                aliases=(name,),
            )
        )
        mapping = {item.id: {"kind": "copy", "input": item.id} for item in inputs}
        if name == "AddLayer":
            mapping["flip_horizontal"] = mapping.pop("flip_h")
            mapping["flip_vertical"] = mapping.pop("flip_v")
            mapping["rotation"] = {
                "kind": "value",
                "input": "rotation",
                "transform": {"kind": "scale", "factor": math.pi / 180, "offset": 0.0},
            }
            mapping["blend_mode"] = {
                "kind": "value",
                "input": "blend_mode",
                "transform": {
                    "kind": "enumRename",
                    "map": {
                        name: name.replace("-", "_")
                        for name in (
                            "normal",
                            "multiply",
                            "screen",
                            "overlay",
                            "soft-light",
                            "hard-light",
                            "color-dodge",
                            "linear-dodge",
                            "color-burn",
                            "linear-burn",
                            "vivid-light",
                            "linear-light",
                            "difference",
                            "darken",
                            "lighten",
                            "hue",
                            "saturation",
                            "luminosity",
                            "color",
                            "exclusion",
                            "pin-light",
                            "hard-mix",
                            "subtract",
                            "divide",
                            "grain-extract",
                            "grain-merge",
                        )
                    },
                },
            }
        output_mapping = {item.id: item.id for item in outputs}
        if name == "ImageCompositor":
            mapping["color_space"] = {"kind": "constant", "value": "linear"}
            output_mapping = {"image": "image", "transparency_mask": "mask"}
        records.append(
            {
                "id": f"comfy_alias:comfy-core/{name}",
                "mappingKind": "op",
                "carrier": carrier,
                "source": {
                    "pack": "comfy-core",
                    "nodeClass": name,
                    "nodeType": source_type,
                    "revision": REFERENCE,
                },
                "replacement": rule_to_wire(
                    rule_from_wire(
                        {
                            "from": source_type,
                            "cases": [
                                {"to": carrier, "inputs": mapping, "outputs": output_mapping}
                            ],
                        }
                    )
                ),
                "confidence": {
                    "tier": "parametric",
                    "evidence": ["tests/test_layer_documents.py::test_layer_aliases_execute"],
                    "tolerances": [{"metric": "max_abs", "operator": "<=", "value": 1 / 255}],
                },
            }
        )
    return schemas, records


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-image/comfy-aliases.json"
    content = json.loads(path.read_text())
    schemas, records = layer_alias_data()
    names = {schema.node_type for schema in schemas}
    content["sourceSchemas"] = [
        schema for schema in content["sourceSchemas"] if schema["nodeType"] not in names
    ] + [schema_to_wire(schema) for schema in schemas]
    content["records"] = [
        record for record in content["records"] if record["source"]["nodeType"] not in names
    ] + records
    path.write_text(json.dumps(content, indent=2) + "\n")
