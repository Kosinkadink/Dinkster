"""Maintained ComfyUI aliases for string operations."""

from __future__ import annotations

import dataclasses
import json
from itertools import combinations
from pathlib import Path
from typing import Any, cast

from dinkster_api.v1 import CORE_BOOLEAN, BooleanWidget, TypeExpr
from dinkster_nodes_foundation import FOUNDATION_NODES
from dinkster_schema import (
    build_schemas,
    comfy_alias_registry_from_wire,
    comfy_alias_registry_problems,
    comfy_alias_registry_to_wire,
    schema_from_wire,
    validate_replacement_references,
)
from dinkster_schema.replace import rule_from_wire, rule_to_wire

from tools.generate_foundation_comfy_aliases import STRING_EVIDENCE, build_aliases

ALIAS_PATH = (
    Path(__file__).parents[1] / "packages" / "dinkster-nodes-foundation" / "comfy-aliases.json"
)


def _payload() -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(ALIAS_PATH.read_text(encoding="utf-8")))


def _records_by_class(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        cast("str", record["source"]["nodeClass"]): record
        for record in cast("list[dict[str, Any]]", payload["records"])
    }


def _string_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        record
        for record in cast("list[dict[str, Any]]", payload["records"])
        if record["confidence"]["evidence"] == list(STRING_EVIDENCE)
    ]


def _operation_values(record: dict[str, Any]) -> list[str | dict[str, str]]:
    values: list[str | dict[str, str]] = []
    for case in cast("list[dict[str, Any]]", record["replacement"]["cases"]):
        operation = cast("dict[str, Any]", case["inputs"])["operation"]
        if operation["kind"] == "constant":
            values.append(cast("str", operation["value"]))
        else:
            values.append(cast("dict[str, str]", operation["transform"]["map"]))
    return values


def test_string_alias_registry_is_canonical_and_valid() -> None:
    encoded = ALIAS_PATH.read_text(encoding="utf-8")
    assert encoded == json.dumps(build_aliases(), ensure_ascii=True, separators=(",", ":")) + "\n"
    payload = _payload()
    assert payload == build_aliases()

    registry = comfy_alias_registry_from_wire(payload)
    assert comfy_alias_registry_to_wire(registry) == payload

    native_schemas = build_schemas(FOUNDATION_NODES)
    assert comfy_alias_registry_problems(registry, native_schemas) == ()

    source_schemas = {
        schema.node_type: schema
        for schema in (
            schema_from_wire(cast("dict[str, Any]", wire))
            for wire in cast("list[dict[str, object]]", payload["sourceSchemas"])
        )
    }
    for record in _string_records(payload):
        rule = rule_from_wire(record["replacement"])
        assert rule_to_wire(rule) == record["replacement"]
        carrier = native_schemas[record["carrier"]]
        schemas = {
            **native_schemas,
            **source_schemas,
            carrier.node_type: dataclasses.replace(carrier, replacements=(rule,)),
        }
        assert validate_replacement_references(schemas) == ()


def test_string_alias_operation_mappings() -> None:
    payload = _payload()
    records = _records_by_class(payload)
    string_records = _string_records(payload)
    assert len(string_records) == 23

    expected_carriers = {
        "StringConcatenate": "std.string.concat",
        "StringSubstring": "dinkster.string.transform",
        "StringLength": "dinkster.string.length",
        "CaseConverter": "dinkster.string.transform",
        "StringTrim": "dinkster.string.transform",
        "StringReplace": "dinkster.string.transform",
        "StringContains": "dinkster.string.test",
        "StringCompare": "dinkster.string.test",
        "RegexMatch": "dinkster.string.regex",
        "RegexReplace": "dinkster.string.regex",
        "JsonExtractString": "dinkster.string.json",
        "ConvertDictionaryToString": "dinkster.string.json_emit",
        "ConvertArrayToString": "dinkster.string.json_emit",
        "StringConstant": "dinkster.string",
        "StringConstantMultiline": "dinkster.string.transform",
        "JoinStrings": "std.string.concat",
        "easy string": "dinkster.string",
        "easy mathString": "dinkster.string.test",
        "easy stringJoinLines": "dinkster.string.transform",
        "ImpactStringSelector": "dinkster.string.transform",
        "Text Concatenate": "dinkster.string.join",
        "Text Contains": "dinkster.string.test",
        "String Replace (mtb)": "dinkster.string.transform",
    }
    assert {
        record["source"]["nodeClass"]: record["carrier"] for record in string_records
    } == expected_carriers

    concatenate = records["StringConcatenate"]["replacement"]["cases"][0]["inputs"]
    assert {key: value["input"] for key, value in concatenate.items()} == {
        "a": "string_a",
        "b": "string_b",
        "separator": "delimiter",
    }
    substring = records["StringSubstring"]["replacement"]["cases"][0]["inputs"]
    assert substring["text"]["input"] == "string"
    assert substring["start"]["input"] == "start"
    assert substring["end"]["input"] == "end"
    json_extract = records["JsonExtractString"]["replacement"]["cases"][0]["inputs"]
    assert json_extract["text"]["input"] == "json_string"
    assert json_extract["selector"]["input"] == "key"
    for source_class, source_input in (
        ("ConvertDictionaryToString", "dictionary"),
        ("ConvertArrayToString", "array"),
    ):
        inputs = records[source_class]["replacement"]["cases"][0]["inputs"]
        assert inputs["value"]["input"] == source_input
        assert inputs["indent"]["input"] == "indent"

    assert _operation_values(records["StringSubstring"]) == ["slice"]
    assert _operation_values(records["CaseConverter"]) == [
        {
            "UPPERCASE": "upper",
            "lowercase": "lower",
            "Capitalize": "capitalize",
            "Title Case": "title",
        }
    ]
    assert _operation_values(records["StringTrim"]) == [
        {"Both": "trim", "Left": "trim_left", "Right": "trim_right"}
    ]
    assert _operation_values(records["StringReplace"]) == ["replace_literal"]
    assert _operation_values(records["StringContains"]) == ["contains", "contains"]
    assert _operation_values(records["StringCompare"]) == [
        {"Starts With": "starts_with", "Ends With": "ends_with", "Equal": "equals"},
        {"Starts With": "starts_with", "Ends With": "ends_with", "Equal": "equals"},
    ]
    assert _operation_values(records["RegexMatch"]) == ["search", "search"]
    assert _operation_values(records["RegexReplace"]) == ["replace", "replace"]
    assert _operation_values(records["JsonExtractString"]) == ["extract_string_legacy"]
    assert _operation_values(records["ConvertDictionaryToString"]) == ["legacy"]
    assert _operation_values(records["ConvertArrayToString"]) == ["legacy"]
    assert _operation_values(records["easy mathString"]) == [
        "match",
        "match",
        "contains",
        "contains",
        {
            "a == b": "equals",
            "a != b": "not_equals",
            "a BEGINSWITH b": "starts_with",
            "a ENDSWITH b": "ends_with",
        },
        {
            "a == b": "equals",
            "a != b": "not_equals",
            "a BEGINSWITH b": "starts_with",
            "a ENDSWITH b": "ends_with",
        },
    ]
    assert _operation_values(records["easy stringJoinLines"]) == ["join_nonempty_lines"]
    assert _operation_values(records["ImpactStringSelector"]) == [
        "select_hash_section",
        "select_line",
    ]
    assert _operation_values(records["Text Contains"]) == ["contains", "contains"]
    assert _operation_values(records["String Replace (mtb)"]) == [
        "replace",
        "replace_literal",
    ]

    assert [
        case["inputs"]["case_mode"]["value"]
        for case in records["easy mathString"]["replacement"]["cases"]
    ] == ["lower_both", "sensitive", "lower_both", "sensitive", "lower_both", "sensitive"]
    easy_math_cases = records["easy mathString"]["replacement"]["cases"]
    assert [case["to"] for case in easy_math_cases] == [
        "dinkster.string.regex",
        "dinkster.string.regex",
        "dinkster.string.test",
        "dinkster.string.test",
        "dinkster.string.test",
        "dinkster.string.test",
    ]
    assert easy_math_cases[2]["inputs"]["text"]["input"] == "b"
    assert easy_math_cases[2]["inputs"]["query"]["input"] == "a"

    join_cases = records["JoinStrings"]["replacement"]["cases"]
    assert len(join_cases) == 4
    assert join_cases[-1]["inputs"]["a"]["value"] == ""
    assert join_cases[-1]["inputs"]["b"]["value"] == ""
    assert "when" not in join_cases[-1]

    multiline_cases = records["StringConstantMultiline"]["replacement"]["cases"]
    assert [case["to"] for case in multiline_cases] == [
        "dinkster.string_multiline",
        "dinkster.string.transform",
    ]
    assert multiline_cases[0]["when"] == {
        "kind": "valueEquals",
        "input": "strip_newlines",
        "value": False,
    }
    chained = multiline_cases[1]
    assert chained["nodes"]["without_newlines"] == {
        "type": "dinkster.string.transform",
        "values": {"operation": "replace_literal", "value": "\n", "replacement": ""},
    }
    assert chained["links"] == [{"from": "without_newlines:text", "to": "text"}]

    concatenate_cases = records["Text Concatenate"]["replacement"]["cases"]
    assert len(concatenate_cases) == 16
    assert all(case["to"] == "dinkster.string.join" for case in concatenate_cases)
    assert all(case["inputs"]["skip_empty"]["value"] is True for case in concatenate_cases)
    assert all(
        case["inputs"]["delimiter_escape"]["value"] == "newline" for case in concatenate_cases
    )
    assert all(
        case["inputs"]["trim"] == {"kind": "copy", "input": "clean_whitespace"}
        for case in concatenate_cases
    )
    source_inputs = ("text_a", "text_b", "text_c", "text_d")
    expected_members = tuple(
        members
        for count in range(len(source_inputs), -1, -1)
        for members in combinations(source_inputs, count)
    )
    for case, expected in zip(concatenate_cases, expected_members, strict=True):
        assert case["inputs"] == {
            "separator": {"kind": "copy", "input": "delimiter"},
            "trim": {"kind": "copy", "input": "clean_whitespace"},
            "skip_empty": {"kind": "constant", "value": True},
            "delimiter_escape": {"kind": "constant", "value": "newline"},
        }
        members = case.get("inputFamilies", {}).get("pieces", {}).get("members", [])
        assert tuple(member["suffix"] for member in members) == expected
        assert all(member["inputs"]["value"]["input"] == member["suffix"] for member in members)
    assert "when" not in concatenate_cases[-1]
    assert "inputFamilies" not in concatenate_cases[-1]

    impact_cases = records["ImpactStringSelector"]["replacement"]["cases"]
    assert all(case["inputs"]["count"]["input"] == "select" for case in impact_cases)
    assert [case["to"] for case in records["String Replace (mtb)"]["replacement"]["cases"]] == [
        "dinkster.string.regex",
        "dinkster.string.transform",
    ]


def test_was_text_concatenate_source_schema_is_normalized_boolean() -> None:
    payload = _payload()
    source_schemas = {
        schema.node_type: schema
        for schema in (
            schema_from_wire(cast("dict[str, Any]", wire))
            for wire in cast("list[dict[str, object]]", payload["sourceSchemas"])
        )
    }
    source = source_schemas["comfy.was-node-suite-comfyui.Text Concatenate"]
    assert [item.id for item in source.inputs] == [
        "delimiter",
        "clean_whitespace",
        "text_a",
        "text_b",
        "text_c",
        "text_d",
    ]
    assert source.inputs[0].default == ", "
    assert source.inputs[1].type == TypeExpr.concrete(CORE_BOOLEAN)
    assert source.inputs[1].default is True
    assert source.inputs[1].widget == BooleanWidget(label_on="true", label_off="false")
    assert all(not item.required and item.force_input for item in source.inputs[2:])


def test_string_alias_revisions_and_held_nodes() -> None:
    payload = _payload()
    string_records = _string_records(payload)
    revisions = {
        record["source"]["pack"]: record["source"]["revision"] for record in string_records
    }
    assert revisions == {
        "comfy-core": "b78cec87",
        "comfyui-kjnodes": "3f20054214fec9f9234fd3841ae6f1e4287948f6",
        "comfyui-easy-use": "4de1ab3b66e48da916b6f263bacd001df53a2720",
        "comfyui-impact-pack": "429d0159ad429e64d2b3916e6e7be9c22d025c3c",
        "was-node-suite-comfyui": "ea935d1044ae5a26efa54ebeb18fe9020af49a45",
        "comfy-mtb": "b35b5d8a17c0d59e80a8b3627b679c2c1003d04f",
    }

    classes = {record["source"]["nodeClass"] for record in string_records}
    assert classes.isdisjoint(
        {
            "StringFormat",
            "RegexExtract",
            "JoinStringMulti",
        }
    )
    family_classes = {
        record["source"]["nodeClass"]
        for record in string_records
        if any("inputFamilies" in case for case in record["replacement"]["cases"])
    }
    assert family_classes == {"Text Concatenate"}
    assert all(
        case["to"] not in {"dinkster.string.format", "dinkster.string.split"}
        for record in string_records
        for case in record["replacement"]["cases"]
    )
