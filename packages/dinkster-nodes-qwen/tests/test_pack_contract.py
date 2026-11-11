"""Pack-authored identities and ordinary ComfyUI conversion contracts."""

import json
from pathlib import Path

from dinkster_api.v1 import schema_from_wire, schema_to_wire
from dinkster_qwen import NODES
from dinkster_workers import load_manifest


def test_pack_identity_and_ordinary_conversion() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(root / "dinkster-pack.toml")
    assert manifest.name == "dinkster-qwen"
    locale = json.loads((root / "locales/en.json").read_text())
    expected = set(locale["nodes"])
    assert {node.schema().node_type for node in NODES} == set(expected)
    assert len(NODES) == len(expected)
    for node in NODES:
        schema = node.schema()
        assert not schema.aliases
        assert len(schema.replacements) == 1
        rule = schema.replacements[0]
        assert rule.from_type != schema.node_type
        assert rule.from_type in schema.search_terms
        assert rule.migration is None
        assert all(case.to == schema.node_type for case in rule.cases)
        assert schema_from_wire(schema_to_wire(schema)) == schema
