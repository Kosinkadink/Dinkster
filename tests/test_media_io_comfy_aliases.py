from __future__ import annotations

import json
import tomllib
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from dinkster_nodes_media_io import MEDIA_IO_NODES
from dinkster_schema import (
    comfy_alias_registry_from_wire,
    comfy_alias_registry_to_wire,
    schema_from_wire,
    schema_to_wire,
    validate_replacement_references,
)
from dinkster_schema.replace import rule_from_wire, rule_to_wire

ROOT = Path(__file__).parent.parent
ALIAS_PATH = ROOT / "packages" / "dinkster-nodes-media-io" / "comfy-aliases.json"
IMAGE_ALIAS_CLASSES = {
    "LoadImage",
    "LoadImageMask",
    "LoadImageOutput",
    "SaveImage",
    "PreviewImage",
    "SaveAnimatedPNG",
    "SaveAnimatedWEBP",
}


def _registry() -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(ALIAS_PATH.read_text(encoding="utf-8")))


def test_media_io_comfy_aliases_use_the_canonical_wire_contract() -> None:
    registry = _registry()
    assert set(registry) == {"format", "sourceSchemas", "records"}
    assert registry["format"] == "dinkster-comfy-alias/1"
    assert comfy_alias_registry_to_wire(comfy_alias_registry_from_wire(registry)) == registry

    source_schemas = [schema_from_wire(wire) for wire in registry["sourceSchemas"]]
    assert len(source_schemas) == len({schema.node_type for schema in source_schemas})
    assert all(not schema.replacements for schema in source_schemas)
    assert [schema_to_wire(schema) for schema in source_schemas] == registry["sourceSchemas"]

    records = [
        record
        for record in registry["records"]
        if record["source"]["nodeClass"] in IMAGE_ALIAS_CLASSES
    ]
    assert len(records) == 7
    assert len({record["id"] for record in records}) == len(records)
    image_source_types = {record["source"]["nodeType"] for record in records}
    assert {record["source"]["nodeType"] for record in records} == {
        schema.node_type for schema in source_schemas if schema.node_type in image_source_types
    }
    for record in records:
        rule = rule_from_wire(record["replacement"])
        assert record["mappingKind"] == "op"
        assert record["source"]["nodeType"] == rule.from_type
        assert rule_to_wire(rule) == record["replacement"]
        assert record["carrier"] in {case.to for case in rule.cases}


def test_media_io_comfy_alias_references_are_executable() -> None:
    registry = _registry()
    native_schemas = {node.schema().node_type: node.schema() for node in MEDIA_IO_NODES}
    records_by_carrier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in registry["records"]:
        records_by_carrier[record["carrier"]].append(record)

    schemas = {
        schema.node_type: schema
        for schema in (schema_from_wire(wire) for wire in registry["sourceSchemas"])
    }
    for carrier, records in records_by_carrier.items():
        schema = native_schemas[carrier]
        schemas[carrier] = replace(
            schema,
            replacements=tuple(rule_from_wire(record["replacement"]) for record in records),
        )
    schemas["dinkster.set_save_target_prefix"] = native_schemas["dinkster.set_save_target_prefix"]
    assert validate_replacement_references(schemas) == ()
    assert {node_type for node_type, schema in native_schemas.items() if schema.replacements} == {
        "dinkster.save_video",
        "dinkster.load_mask",
        "dinkster.save_mask",
    }


def test_media_io_comfy_aliases_preserve_core_image_io_contracts() -> None:
    records = {record["id"]: record for record in _registry()["records"]}

    load_mask = records["comfy_alias:comfy-core/LoadImageMask"]["replacement"]["cases"][0]
    assert load_mask["inputs"]["mask"] == {"kind": "copy", "input": "image"}
    assert load_mask["inputs"]["channel"] == {"kind": "copy", "input": "channel"}
    assert load_mask["inputs"]["mask_polarity"] == {
        "kind": "value",
        "input": "channel",
        "transform": {
            "kind": "enumRename",
            "map": {
                "alpha": "transparency",
                "blue": "coverage",
                "green": "coverage",
                "red": "coverage",
            },
        },
    }

    load_output = records["comfy_alias:comfy-core/LoadImageOutput"]
    assert load_output["carrier"] == "dinkster.load_image_output"
    assert load_output["replacement"]["cases"][0]["inputs"] == {
        "image": {"kind": "copy", "input": "image"}
    }

    preview = records["comfy_alias:comfy-core/PreviewImage"]["replacement"]["cases"][0]
    assert preview["inputs"] == {"images": {"kind": "copy", "input": "images"}}
    assert preview["outputs"] == {"images": "images"}


def test_media_io_save_aliases_build_mounted_targets_and_copy_controls() -> None:
    records = {record["id"]: record for record in _registry()["records"]}
    expected_inputs = {
        "SaveImage": {
            "images": {"kind": "copy", "input": "images"},
            "format": {"kind": "constant", "value": "png"},
            "compression": {"kind": "constant", "value": 4},
        },
        "SaveAnimatedPNG": {
            "images": {"kind": "copy", "input": "images"},
            "format": {"kind": "constant", "value": "png"},
            "fps": {"kind": "copy", "input": "fps"},
            "compression": {"kind": "copy", "input": "compress_level"},
            "loop": {"kind": "constant", "value": 0},
        },
        "SaveAnimatedWEBP": {
            "images": {"kind": "copy", "input": "images"},
            "format": {"kind": "constant", "value": "webp"},
            "fps": {"kind": "copy", "input": "fps"},
            "lossless": {"kind": "copy", "input": "lossless"},
            "quality": {"kind": "copy", "input": "quality"},
            "method": {"kind": "copy", "input": "method"},
            "loop": {"kind": "constant", "value": 0},
        },
    }
    for source, expected in expected_inputs.items():
        case = records[f"comfy_alias:comfy-core/{source}"]["replacement"]["cases"][0]
        assert case["nodes"] == {
            "save_target": {
                "type": "dinkster.set_save_target_prefix",
                "values": {"target": {"mount": "comfy-output", "prefix": "ComfyUI"}},
            }
        }
        assert case["links"] == [{"from": "save_target:save_target", "to": "target"}]
        assert case["inputs"] == {
            **expected,
            "save_target:prefix": {"kind": "copy", "input": "filename_prefix"},
        }


def test_media_io_comfy_alias_evidence_and_wheel_bundle_are_pinned() -> None:
    records = (
        record
        for record in _registry()["records"]
        if record["source"]["nodeClass"] in IMAGE_ALIAS_CLASSES
    )
    for record in records:
        assert record["source"]["revision"] == "b78cec87"
        assert record["confidence"]["tier"] == "parametric"
        for selector in record["confidence"]["evidence"]:
            path_text, separator, test_name = selector.partition("::")
            path = ROOT / path_text
            assert separator and test_name
            assert path.is_file()
            assert f"def {test_name}(" in path.read_text(encoding="utf-8")

    configuration = cast(
        "dict[str, Any]",
        tomllib.loads(
            (ROOT / "packages" / "dinkster-nodes-media-io" / "pyproject.toml").read_text(
                encoding="utf-8"
            )
        ),
    )
    force_include = configuration["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert force_include["comfy-aliases.json"] == (
        "dinkster_nodes_media_io_pack/comfy-aliases.json"
    )
