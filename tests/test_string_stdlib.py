"""String and structured-text standard library coverage."""

from __future__ import annotations

import asyncio
import math
import re
from typing import Any, cast

import pytest
from dinkster_api.v1 import CORE_BOOLEAN, CORE_FLOAT, CORE_INT, CORE_STRING, TypeExpr
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_foundation import (
    FOUNDATION_NODES,
    StringCsvEmit,
    StringCsvParse,
    StringFormat,
    StringJoin,
    StringJson,
    StringJsonEmit,
    StringLength,
    StringRegex,
    StringSplit,
    StringTest,
)
from dinkster_nodes_foundation.string_ops import (
    MAX_FORMAT_WIDTH,
    MAX_REGEX_PATTERN_BYTES,
    MAX_SPLIT_PARTS,
    MAX_TEXT_BYTES,
    format_string,
    transform_string,
)
from dinkster_schema import build_node_types, build_schemas
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker


def _transform(text: str, operation: str, **changes: Any) -> str:
    options: dict[str, Any] = {
        "text": text,
        "operation": operation,
        "value": "",
        "replacement": "",
        "start": 0,
        "end": (1 << 53) - 1,
        "count": 0,
        "fill": " ",
        "unit": "characters",
        "side": "start",
    }
    options.update(changes)
    return transform_string(**options)


def _regex(text: str, pattern: str, operation: str, **changes: Any) -> dict[str, object]:
    options: dict[str, Any] = {
        "text": text,
        "pattern": pattern,
        "operation": operation,
        "replacement": "",
        "group_index": 1,
        "count": 0,
        "case_mode": "sensitive",
        "multiline": False,
        "dotall": False,
    }
    options.update(changes)
    return dict(StringRegex.execute(**options))


def _json(operation: str, text: str, **changes: Any) -> str:
    options: dict[str, Any] = {
        "operation": operation,
        "text": text,
        "selector": "",
        "indent": 2,
        "key_order": "preserve",
    }
    options.update(changes)
    return str(StringJson.execute(**options)["text"])


def test_string_format_uses_bounded_scalar_grammar() -> None:
    assert format_string("{{{a:^5}}} {b:04d} {c:.2f}", {"a": "x", "b": 7, "c": 1.25}) == (
        "{  x  } 0007 1.25"
    )
    assert format_string("{a} {b} {c}", {"a": True, "b": None, "c": -2}) == "True None -2"
    assert StringFormat.execute(template="{a:.3s}", slots={"a": "string"}) == {"text": "str"}

    for template, message in (
        ("{missing}", "one lowercase"),
        ("{a.x}", "one lowercase"),
        ("{a[0]}", "one lowercase"),
        ("{a!r}", "conversions"),
        ("{a:{b}}", "nested"),
        (f"{{a:{MAX_FORMAT_WIDTH + 1}}}", "width"),
    ):
        with pytest.raises(ValueError, match=message):
            format_string(template, {"a": "x", "b": 1})
    with pytest.raises(ValueError, match="missing format slot"):
        format_string("{a}", {})
    with pytest.raises(TypeError, match="JSON scalar"):
        format_string("{a}", {"a": object()})
    with pytest.raises(ValueError, match="finite"):
        format_string("{a}", {"a": math.nan})
    with pytest.raises(ValueError, match="formatted text"):
        format_string("{a:65536}{a:65536}" * 9, {"a": "x"})

    family = StringFormat.schema().input_families[0]
    assert family.type == TypeExpr.union(CORE_STRING, CORE_INT, CORE_FLOAT, CORE_BOOLEAN)


@pytest.mark.parametrize(
    ("operation", "text", "changes", "expected"),
    (
        ("identity", "Text", {}, "Text"),
        ("upper", "Stra\u00dfe", {}, "STRASSE"),
        ("lower", "\u0130", {}, "i\u0307"),
        ("capitalize", "hELLO wORLD", {}, "Hello world"),
        ("title", "hello world", {}, "Hello World"),
        ("trim", " \ttext\n", {}, "text"),
        ("trim_left", "  text  ", {}, "text  "),
        ("trim_right", "  text  ", {}, "  text"),
        ("prefix", "text", {"value": "pre-"}, "pre-text"),
        ("suffix", "text", {"value": "-post"}, "text-post"),
        ("replace_literal", "aba", {"value": "", "replacement": "-"}, "-a-b-a-"),
        ("slice", "abcdef", {"start": -4, "end": -1}, "cde"),
        ("take", "abcdef", {"count": 3}, "abc"),
        ("take", "abcdef", {"count": 0, "side": "end"}, "abcdef"),
        ("take", "one  two three", {"count": 2, "unit": "words"}, "one two"),
        ("pad_left", "x", {"count": 3, "fill": "0"}, "00x"),
        ("pad_right", "x", {"count": 3, "fill": "0"}, "x00"),
        ("pad_center", "x", {"count": 4, "fill": "-"}, "-x--"),
        ("join_nonempty_lines", " a \n\n b ", {"value": "|"}, "a|b"),
        ("select_line", "a\nb", {"count": 3}, "b"),
        ("select_hash_section", "# one\na\n## two\nb", {"count": 1}, "# two\nb"),
    ),
)
def test_string_transform_operations(
    operation: str, text: str, changes: dict[str, object], expected: str
) -> None:
    assert _transform(text, operation, **changes) == expected


def test_string_transform_rejects_bad_modes_and_bounds() -> None:
    with pytest.raises(ValueError, match="one Unicode"):
        _transform("x", "pad_left", count=3, fill="ab")
    with pytest.raises(ValueError, match="padding width"):
        _transform("x", "pad_left", count=MAX_FORMAT_WIDTH + 1)
    with pytest.raises(ValueError, match="take unit"):
        _transform("x", "take", unit="bytes")
    assert _transform("", "select_hash_section") == ""
    assert _transform("\n# heading", "select_hash_section", count=0) == ""
    with pytest.raises(ValueError, match="exceeds"):
        _transform("x" * (MAX_TEXT_BYTES + 1), "identity")
    with pytest.raises(ValueError, match="transformed text"):
        _transform("a" * 1024, "replace_literal", value="a", replacement="x" * 2048)
    with pytest.raises(ValueError, match="transformed text"):
        _transform("a\nb\nc", "join_nonempty_lines", value="x" * (MAX_TEXT_BYTES // 2))


@pytest.mark.parametrize(
    ("operation", "expected"),
    (
        ("equals", False),
        ("not_equals", True),
        ("contains", True),
        ("starts_with", True),
        ("ends_with", True),
    ),
)
def test_string_test_and_length(operation: str, expected: bool) -> None:
    query = {
        "equals": "other",
        "not_equals": "other",
        "contains": "ELL",
        "starts_with": "HELL",
        "ends_with": "LLO",
    }[operation]
    assert StringTest.execute(
        text="Hello", query=query, operation=operation, case_mode="lower_both"
    ) == {"result": expected}
    assert StringLength.execute(text="e\u0301\U0001f642") == {"length": 3}


def test_regex_search_extract_and_replace_semantics() -> None:
    assert _regex(
        "Abc\ndef", "^DEF$", "search", case_mode="unicode_ignorecase", multiline=True
    ) == {
        "matched": True,
        "text": "",
        "count": 1,
    }
    assert _regex("xx abc", "abc", "match") == {"matched": False, "text": "", "count": 0}
    assert _regex("a1 b22", r"\w\d+", "extract_first") == {
        "matched": True,
        "text": "a1",
        "count": 1,
    }
    assert _regex("a1 b22", r"(\w)(\d+)", "extract_all_legacy") == {
        "matched": True,
        "text": "a\nb",
        "count": 2,
    }
    assert _regex("a1 b22", r"(\w)(\d+)", "extract_group_first", group_index=2)["text"] == "1"
    assert _regex("a1 b22", r"(\w)(\d+)", "extract_group_all", group_index=2)["text"] == ("1\n22")
    assert _regex("aaaa", "a", "replace", replacement="x", count=2) == {
        "matched": True,
        "text": "xxaa",
        "count": 2,
    }
    assert _regex("ABC", "abc", "match", case_mode="lower_both")["matched"] is True


def test_regex_invalid_patterns_and_bounds() -> None:
    assert _regex("text", "(", "search") == {"matched": False, "text": "", "count": 0}
    with pytest.raises(re.error):
        _regex("text", "(", "replace")
    with pytest.raises(ValueError, match="regex pattern"):
        _regex("text", "x" * (MAX_REGEX_PATTERN_BYTES + 1), "search")
    with pytest.raises(ValueError, match="replacement result"):
        _regex("a" * 1024, "a", "replace", replacement="x" * 2048)
    with pytest.raises(ValueError, match="replacement result"):
        _regex("a" * (MAX_TEXT_BYTES // 2), "(a+)", "replace", replacement=r"\1\1\1")
    with pytest.raises(ValueError, match="regex results"):
        _regex("a" * 20_000, "a", "replace", replacement="")


def test_join_and_split_cover_typed_and_dynamic_inputs() -> None:
    assert StringJoin.execute(
        items=[" one ", ""],
        pieces={"a": " two ", "b": "three"},
        separator="|",
        trim=True,
        skip_empty=True,
        delimiter_escape="raw",
    ) == {"text": "one|two|three"}
    assert StringJoin.execute(
        items=None,
        pieces={"a": "one", "b": "two"},
        separator=r"\n",
        trim=False,
        skip_empty=False,
        delimiter_escape="newline",
    ) == {"text": "one\ntwo"}
    assert StringSplit.execute(
        text="a::b:c",
        mode="literal_any",
        delimiter=":",
        keep_empty=True,
        max_parts=0,
        additional_delimiters={"long": "::"},
    ) == {"parts": ["a", "b", "c"]}
    assert StringSplit.execute(
        text=" a\t b  c ",
        mode="whitespace",
        delimiter=" ",
        keep_empty=False,
        max_parts=2,
        additional_delimiters={},
    ) == {"parts": ["a", "b  c "]}
    assert StringSplit.execute(
        text="a\r\n\r\nb",
        mode="lines",
        delimiter=" ",
        keep_empty=True,
        max_parts=0,
        additional_delimiters={},
    ) == {"parts": ["a", "", "b"]}
    assert StringSplit.execute(
        text="abc",
        mode="characters",
        delimiter=" ",
        keep_empty=True,
        max_parts=10,
        additional_delimiters={},
    ) == {"parts": ["a", "b", "c"]}
    assert StringSplit.execute(
        text="a,b",
        mode="literal_any",
        delimiter=",",
        keep_empty=True,
        max_parts=1,
        additional_delimiters={},
    ) == {"parts": ["a,b"]}


def test_join_and_split_bound_string_inputs() -> None:
    with pytest.raises(ValueError, match="join separator"):
        StringJoin.execute(
            items=["value"],
            pieces={},
            separator="x" * (MAX_TEXT_BYTES + 1),
            trim=False,
            skip_empty=False,
            delimiter_escape="raw",
        )
    with pytest.raises(ValueError, match="joined text"):
        StringJoin.execute(
            items=["x" * (MAX_TEXT_BYTES // 2), "x" * (MAX_TEXT_BYTES // 2)],
            pieces={},
            separator="--",
            trim=False,
            skip_empty=False,
            delimiter_escape="raw",
        )
    with pytest.raises(ValueError, match="split delimiter"):
        StringSplit.execute(
            text="value",
            mode="literal_any",
            delimiter="x" * (MAX_REGEX_PATTERN_BYTES + 1),
            keep_empty=False,
            max_parts=0,
            additional_delimiters={},
        )


def test_split_rejects_ambiguous_or_excessive_shapes() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        StringSplit.execute(
            text="a",
            mode="literal_any",
            delimiter="",
            keep_empty=False,
            max_parts=0,
            additional_delimiters={},
        )
    with pytest.raises(ValueError, match="keep_empty"):
        StringSplit.execute(
            text="a",
            mode="whitespace",
            delimiter=" ",
            keep_empty=True,
            max_parts=0,
            additional_delimiters={},
        )
    with pytest.raises(ValueError, match="exceed"):
        StringSplit.execute(
            text="x" * (MAX_SPLIT_PARTS + 1),
            mode="characters",
            delimiter=" ",
            keep_empty=False,
            max_parts=0,
            additional_delimiters={},
        )


@pytest.mark.parametrize(
    ("mode", "delimiter", "keep_empty", "additional_delimiters", "message"),
    [
        ("unknown", " ", False, {}, "unknown split mode"),
        ("literal_any", "", False, {}, "non-empty"),
        ("whitespace", " ", True, {}, "keep_empty"),
        ("characters", ",", False, {}, "does not use delimiters"),
    ],
)
def test_split_validates_mode_controls_when_limited_to_one_part(
    mode: str,
    delimiter: str,
    keep_empty: bool,
    additional_delimiters: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        StringSplit.execute(
            text="a b",
            mode=mode,
            delimiter=delimiter,
            keep_empty=keep_empty,
            max_parts=1,
            additional_delimiters=additional_delimiters,
        )


def test_strict_json_format_select_and_emit() -> None:
    source = '{"b": [true, null], "a/b": {"~key": "\u00e9"}}'
    assert _json("minify", source, key_order="sorted") == (
        '{"a/b":{"~key":"\u00e9"},"b":[true,null]}'
    )
    assert _json("select", source, selector="/a~1b/~0key", indent=0) == '"\u00e9"'
    assert _json("pretty", "[1,2]", indent=2) == "[\n  1,\n  2\n]"
    assert StringJsonEmit.execute(
        value={"\u00e9": [1, True, None]},
        operation="strict",
        indent=0,
        key_order="preserve",
    ) == {"text": '{"\u00e9":[1,true,null]}'}

    with pytest.raises(ValueError, match="duplicate"):
        _json("minify", '{"a":1,"a":2}')
    with pytest.raises(ValueError, match="non-finite"):
        _json("minify", "NaN")
    with pytest.raises(ValueError, match="does not exist"):
        _json("select", "{}", selector="/missing")
    with pytest.raises(ValueError, match="depth"):
        _json("minify", "[" * 66 + "0" + "]" * 66)
    with pytest.raises(TypeError, match="keys"):
        StringJsonEmit.execute(value={1: "bad"}, operation="strict", indent=0, key_order="preserve")
    with pytest.raises(ValueError, match="JSON output"):
        StringJsonEmit.execute(
            value=["x" * 65_536] * 17,
            operation="strict",
            indent=0,
            key_order="preserve",
        )


def test_legacy_json_operations_match_pinned_core_shapes() -> None:
    source = '{"boolean":true,"array":[1,2],"object":{"a":1},"null":null}'
    assert _json("extract_string_legacy", source, selector="boolean") == "True"
    assert _json("extract_string_legacy", source, selector="array") == "[1, 2]"
    assert _json("extract_string_legacy", source, selector="object") == "{'a': 1}"
    assert _json("extract_string_legacy", source, selector="null") == ""
    assert _json("extract_string_legacy", "bad", selector="key") == ""


def test_json_legacy_extraction_finds_first_valid_object_in_surrounding_text() -> None:
    assert (
        _json(
            "extract_string_legacy",
            'prefix {not json}\n```json\n{"answer": "yes", "count": 3}\n``` suffix',
            selector="answer",
        )
        == "yes"
    )
    assert (
        _json(
            "extract_string_legacy",
            '{"other": 1} then {"answer": "later"}',
            selector="answer",
        )
        == ""
    )
    assert _json("extract_string_legacy", "prefix [1, 2, 3] suffix", selector="0") == ""
    assert StringJsonEmit.execute(
        value={"a": 1, "b": 2}, operation="legacy", indent=0, key_order="preserve"
    ) == {"text": '{"a": 1, "b": 2}'}


def test_csv_parse_emit_round_trip_is_platform_independent() -> None:
    source = '\ufeffname,note\r\nAlice,"hello, world"\r\nBob,"line 1\r\nline 2"\r\n'
    parsed = StringCsvParse.execute(
        text=source, delimiter=",", quote='"', header=True, strip_bom=True
    )
    assert parsed == {
        "header": ["name", "note"],
        "rows": [["Alice", "hello, world"], ["Bob", "line 1\r\nline 2"]],
    }
    rows = cast("list[list[str]]", parsed["rows"])
    header = cast("list[str]", parsed["header"])
    emitted = StringCsvEmit.execute(rows=rows, header=header, delimiter=",", quote='"')
    assert emitted == {"text": 'name,note\nAlice,"hello, world"\nBob,"line 1\r\nline 2"\n'}


def test_csv_rejects_ragged_duplicate_and_invalid_dialects() -> None:
    with pytest.raises(ValueError, match="same number"):
        StringCsvParse.execute(
            text="a,b\n1\n", delimiter=",", quote='"', header=True, strip_bom=False
        )
    with pytest.raises(ValueError, match="unique"):
        StringCsvParse.execute(
            text="a,a\n1,2\n", delimiter=",", quote='"', header=True, strip_bom=False
        )
    with pytest.raises(ValueError, match="distinct"):
        StringCsvEmit.execute(rows=[["a"]], delimiter=",", quote=",", header=None)
    with pytest.raises(ValueError, match="at least one column"):
        StringCsvEmit.execute(rows=[[]], delimiter=",", quote='"', header=None)
    with pytest.raises(ValueError, match="CSV output"):
        StringCsvEmit.execute(
            rows=[["x" * 65_536]] * 17,
            delimiter=",",
            quote='"',
            header=None,
        )
    with pytest.raises(ValueError, match="CSV rows"):
        StringCsvParse.execute(
            text="x\n" * 10_001,
            delimiter=",",
            quote='"',
            header=False,
            strip_bom=False,
        )


def test_string_node_schemas_are_parameterized_and_legacy_nodes_remain() -> None:
    schemas = build_schemas(FOUNDATION_NODES)
    expected = {
        "dinkster.string.format",
        "dinkster.string.transform",
        "dinkster.string.test",
        "dinkster.string.length",
        "dinkster.string.regex",
        "dinkster.string.join",
        "dinkster.string.split",
        "dinkster.string.json",
        "dinkster.string.json_emit",
        "dinkster.string.csv_parse",
        "dinkster.string.csv_emit",
    }
    assert expected <= set(schemas)
    assert {"std.string.concat", "std.string.join", "std.string.split"} <= set(schemas)
    assert schemas["dinkster.string.format"].input_families[0].member_names == tuple(
        "abcdefghijklmnopqrstuvwxyz"
    )
    assert schemas["dinkster.string.join"].input_families[0].max_members == 100
    assert schemas["dinkster.string.split"].outputs[0].type == StringSplit.schema().outputs[0].type


def test_string_nodes_execute_through_engine() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        engine = Engine(
            schemas=build_schemas(FOUNDATION_NODES),
            registry=registry,
            worker=InProcessWorker(build_node_types(FOUNDATION_NODES), registry),
            cache=MemoryLRUCache(),
        )
        graph = Graph(
            nodes={
                "source": GraphNode("dinkster.string", {"value": "  Hello  "}),
                "transform": GraphNode(
                    "dinkster.string.transform",
                    {"text": Link("source", "value"), "operation": "trim"},
                ),
                "length": GraphNode("dinkster.string.length", {"text": Link("transform", "text")}),
            }
        )
        result = await engine.run(graph, ["transform", "length"])
        assert result.outputs["transform"]["text"].resolve() == "Hello"
        assert result.outputs["length"]["length"].resolve() == 5

    asyncio.run(scenario())
