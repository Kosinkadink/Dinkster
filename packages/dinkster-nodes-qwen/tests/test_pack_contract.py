"""Frozen first-party identities and ordinary ComfyUI conversion contracts."""

import json
from importlib.resources import files
from pathlib import Path

from dinkster_api.v1 import schema_from_wire, schema_to_wire
from dinkster_qwen import NODES
from dinkster_workers import load_manifest


def test_pack_identity_and_ordinary_conversion() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(root / "dinkster-pack.toml")
    table = json.loads(files("dinkster_schema").joinpath("comfy_node_ids.json").read_text())
    expected = {
        row["nodeType"]: row["from"] for row in table["nodes"] if row["pack"] == "dinkster-qwen"
    }
    assert manifest.name == "dinkster-qwen"
    assert {node.schema().node_type for node in NODES} == set(expected)
    assert len(NODES) == len(expected)
    locale = json.loads((root / "locales/en.json").read_text())
    assert set(locale["nodes"]) == set(expected)
    for node in NODES:
        schema = node.schema()
        assert not schema.aliases
        assert expected[schema.node_type] in schema.search_terms
        assert len(schema.replacements) == 1
        rule = schema.replacements[0]
        assert rule.from_type == expected[schema.node_type]
        assert rule.migration is None
        assert all(case.to == schema.node_type for case in rule.cases)
        assert schema_from_wire(schema_to_wire(schema)) == schema
