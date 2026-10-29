"""Golden fixtures stay encoder-authored (goldens/replacements/).

Dinkster-Frontend locks its wire decoder against these files, so their whole
value is that they are EXACTLY what the current encoders produce. This test
regenerates every fixture in-memory through the real builder and fails when
a committed file drifts - after an intentional encoder change, refresh with:

    uv run python scripts/generate_replacement_goldens.py

The shape assertions below are the coverage contract agreed with the
frontend: if a refactor quietly drops a union member from the fixtures, the
files could still "match" while covering less, so coverage is asserted
independently of file equality.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from dinkster_schema import schema_from_wire

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDENS_DIR = REPO_ROOT / "goldens" / "replacements"
EXTENSION_GOLDEN = REPO_ROOT / "goldens" / "extensions" / "snapshot-v1.json"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_replacement_goldens",
        REPO_ROOT / "scripts" / "generate_replacement_goldens.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_extension_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_extension_snapshot_golden",
        REPO_ROOT / "scripts" / "generate_extension_snapshot_golden.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extension_snapshot_golden_matches_canonical_serializer() -> None:
    generator = _load_extension_generator()
    built = generator.build_golden()
    on_disk_bytes = EXTENSION_GOLDEN.read_bytes()
    assert on_disk_bytes == generator.render_golden()
    on_disk = json.loads(on_disk_bytes)
    assert on_disk == built, (
        "extension snapshot golden drifted; regenerate with "
        "scripts/generate_extension_snapshot_golden.py"
    )

    canonical = on_disk["canonicalSerialization"]
    assert json.loads(canonical) == on_disk["snapshot"]
    assert on_disk["behaviorSha256"] == generator.extension_behavior_hash(
        generator.proof_snapshot()
    )


def test_committed_goldens_match_current_encoders() -> None:
    generator = _load_generator()
    built = generator.build_goldens()
    assert set(built) == {"vocabulary.json", "chain.json", "combo.json", "dynamic.json"}
    for name, content in built.items():
        on_disk = json.loads((GOLDENS_DIR / name).read_text(encoding="utf-8"))
        assert on_disk == content, (
            f"{name} drifted from the current encoders; regenerate with "
            "scripts/generate_replacement_goldens.py"
        )


def test_goldens_round_trip_through_the_decoder() -> None:
    """Every fixture schema decodes back through schema_from_wire and
    re-encodes byte-identically - the same loop the frontend's decoder
    mirrors."""
    from dinkster_schema import schema_to_wire

    for name in ("vocabulary.json", "chain.json", "combo.json", "dynamic.json"):
        content = json.loads((GOLDENS_DIR / name).read_text(encoding="utf-8"))
        for wire in content["schemas"]:
            assert schema_to_wire(schema_from_wire(wire)) == wire


def _rules_of(content: dict[str, Any]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for schema in content["schemas"]:
        rules.extend(schema.get("replacements", []))
    return rules


def _walk_predicates(predicate: dict[str, Any]):
    yield predicate
    of = predicate.get("of")
    if isinstance(of, dict):
        yield from _walk_predicates(of)
    elif isinstance(of, list):
        for child in of:
            yield from _walk_predicates(child)


def test_vocabulary_fixture_covers_every_union_member() -> None:
    content = json.loads((GOLDENS_DIR / "vocabulary.json").read_text(encoding="utf-8"))
    modern = next(schema for schema in content["schemas"] if schema["nodeType"] == "fixture.modern")
    intensity = next(entry for entry in modern["interface"] if entry.get("id") == "intensity")
    assert intensity["widget"] == {"type": "NUMBER", "display": "slider"}
    cases = [case for rule in _rules_of(content) for case in rule["cases"]]

    predicate_kinds = {
        p["kind"] for case in cases if "when" in case for p in _walk_predicates(case["when"])
    }
    assert predicate_kinds >= {
        "inputConnected",
        "valuePresent",
        "valueEquals",
        "not",
        "all",
        "any",
    }
    # `always` is representable as the omitted `when` on the fallback case.
    assert any("when" not in case for case in cases)

    mapping_kinds = {source["kind"] for case in cases for source in case.get("inputs", {}).values()}
    assert mapping_kinds == {"copy", "value", "link", "constant"}

    transforms = [
        source["transform"]
        for case in cases
        for source in case.get("inputs", {}).values()
        if "transform" in source
    ]
    kinds = {t["kind"] for t in transforms}
    assert kinds == {"enumRename", "scale"}
    scales = [t for t in transforms if t["kind"] == "scale"]
    assert any("offset" in t for t in scales)
    assert any("offset" not in t for t in scales)

    # Multi-successor fan-out: one rule, more than one distinct target.
    fan_out = {case["to"] for rule in _rules_of(content) for case in rule["cases"]}
    assert len(fan_out) > 1


def test_dynamic_fixture_pins_nested_slot_variants_and_paths() -> None:
    content = json.loads((GOLDENS_DIR / "dynamic.json").read_text(encoding="utf-8"))
    target = next(
        schema for schema in content["schemas"] if schema["nodeType"] == "fixture.dynamic-target"
    )
    (case,) = target["replacements"][0]["cases"]
    assert case["slotVariants"] == {
        "policy": "tolerance_color",
        "policy.color_source": "integer",
    }
    assert set(case["inputs"]) == {
        "policy.color_source.color_value",
        "policy.tolerance",
        "policy.metric",
    }


def test_chain_fixture_ships_the_agreed_chain_pair() -> None:
    content = json.loads((GOLDENS_DIR / "chain.json").read_text(encoding="utf-8"))
    by_type = {schema["nodeType"]: schema for schema in content["schemas"]}

    # A->B rides B, B->C rides C - each hop with its own transform.
    assert by_type["fixture.chain-b"]["replacements"][0]["from"] == "fixture.chain-a"
    assert by_type["fixture.chain-c"]["replacements"][0]["from"] == "fixture.chain-b"

    # Deprecation + searchVisibility + replacements on ONE schema (B).
    middle = by_type["fixture.chain-b"]
    assert middle["deprecation"]["replacement"] == "fixture.chain-c"
    assert middle["deprecation"]["since"]
    assert middle["searchVisibility"] == "hidden"
    assert middle["replacements"]
    assert by_type["fixture.chain-a"]["deprecation"]["replacement"] == ("fixture.chain-b")
    assert by_type["fixture.chain-a"]["searchVisibility"] == "deprecated"


def test_combo_fixture_pins_current_wire_widget_contract() -> None:
    content = json.loads((GOLDENS_DIR / "combo.json").read_text(encoding="utf-8"))
    (schema,) = content["schemas"]
    assert schema["nodeType"] == "fixture.combo-contract"
    assert schema["schemaVersion"] == 1
    by_key = {(entry["role"], entry["id"]): entry for entry in schema["interface"]}
    combo_input = by_key[("input", "choice")]
    combo_output = by_key[("output", "choice")]
    multicombo_input = by_key[("input", "providers")]
    multicombo_output = by_key[("output", "providers")]
    assert combo_input["type"] == {
        "kind": "concrete",
        "types": ["core.combo"],
    }
    assert combo_input["widget"] == {
        "type": "COMBO",
        "options": [
            {
                "value": "alpha",
                "label": "Alpha",
                "info": "Primary choice",
                "folder": "Featured",
            },
            "beta",
        ],
        "controlAfterGenerate": "randomize",
        "remote": {
            "route": "/api/choices/fixture.combo",
            "refreshButton": True,
            "controlAfterRefresh": "last",
            "timeoutMs": 4096,
            "maxRetries": 2,
            "refreshMs": 0,
        },
    }
    assert combo_output["type"] == {
        "kind": "concrete",
        "types": ["core.combo"],
    }
    assert multicombo_input["type"] == {
        "kind": "list",
        "element": {"kind": "concrete", "types": ["core.combo"]},
    }
    assert multicombo_input["default"] == ["beta", "alpha", "beta"]
    assert multicombo_input["widget"] == {
        "type": "MULTI_COMBO",
        "options": [
            {
                "value": "beta",
                "label": "Beta provider",
                "info": "Preferred provider",
                "folder": "Providers/Featured",
            },
            "alpha",
            "beta",
        ],
        "remote": {
            "route": "/api/choices/fixture.providers",
            "refreshButton": True,
            "controlAfterRefresh": "last",
            "timeoutMs": 4096,
            "maxRetries": 2,
            "refreshMs": 0,
        },
        "placeholder": "Select providers",
        "chip": False,
    }
    assert multicombo_output["type"] == multicombo_input["type"]
    assert "widget" not in multicombo_output
    prompt = by_key[("input", "prompt")]
    assert prompt["type"] == {
        "kind": "concrete",
        "types": ["core.string"],
    }
    assert prompt["widget"] == {
        "type": "REPRESENTATIONS",
        "default": "multiline",
        "userSwitchable": True,
        "representations": [
            {
                "id": "single-line",
                "displayName": "Single line",
                "widget": {
                    "type": "STRING",
                    "multiline": False,
                    "placeholder": "Describe an image",
                    "dynamicPrompts": False,
                },
            },
            {
                "id": "multiline",
                "displayName": "Multiline",
                "widget": {
                    "type": "STRING",
                    "multiline": True,
                    "placeholder": "Describe an image",
                    "dynamicPrompts": True,
                },
            },
        ],
    }
    assert by_key[("input", "scale")]["widget"] == {
        "type": "NUMBER",
        "round": 0.001,
    }
    assert by_key[("input", "color")]["widget"] == {"type": "COLOR"}
    assert all("comboSource" not in entry for entry in schema["interface"])
