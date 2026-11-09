"""Public behavior of native ComfyUI text nodes."""

from __future__ import annotations

import re
from typing import Any, cast

import pytest
from dinkster_api.v1 import schema_to_wire
from dinkster_nodes_comfy.text import (
    CaseConverter,
    ConvertArrayToString,
    ConvertDictionaryToString,
    JsonExtractString,
    RegexExtract,
    RegexMatch,
    RegexReplace,
    StringCompare,
    StringConcatenate,
    StringContains,
    StringFormat,
    StringLength,
    StringReplace,
    StringSubstring,
    StringTrim,
)


def test_format_wire_preserves_optional_named_values() -> None:
    interface = cast(
        list[dict[str, Any]], schema_to_wire(StringFormat.define_schema())["interface"]
    )
    assert [item["id"] for item in interface] == ["values", "f_string", "STRING"]
    family, text, output = interface
    assert family["role"] == "inputFamily"
    assert family["required"] is True
    assert family["minMembers"] == 0
    assert family["memberNames"] == list("abcdefghijklmnopqrstuvwxyz")
    assert family["template"] == [
        {"role": "input", "id": "value", "required": False, "type": {"kind": "wildcard"}}
    ]
    assert text["role"] == "input"
    assert text["required"] is True
    assert text["type"] == {"kind": "concrete", "types": ["STRING"]}
    assert text["default"] == "{a}"
    assert output["role"] == "output"
    assert output["type"] == {"kind": "concrete", "types": ["STRING"]}


def test_format_uses_python_formatting() -> None:
    assert StringFormat.execute(
        values={"a": {"name": "Ada"}, "b": 3.14159, "c": [7, 11]},
        f_string="{a[name]!r}: {b:.2f} / {c[1]:04d}",
    ) == {"STRING": "'Ada': 3.14 / 0011"}
    assert StringFormat.execute(values={}, f_string="{{empty}}") == {"STRING": "{empty}"}
    with pytest.raises(KeyError):
        StringFormat.execute(values={}, f_string="{a}")


def test_text_operations_preserve_order_and_python_slicing() -> None:
    assert StringConcatenate.execute(string_a="red", string_b="blue", delimiter=" | ") == {
        "STRING": "red | blue"
    }
    assert StringSubstring.execute(string="abcdefg", start=-5, end=-1) == {"STRING": "cdef"}
    assert StringSubstring.execute(string="abcdefg", start=5, end=2) == {"STRING": ""}
    assert StringLength.execute(string="A\U0001f600e\u0301") == {"length": 4}
    assert StringReplace.execute(string="one ONE one", find="one", replace="two") == {
        "STRING": "two ONE two"
    }


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("UPPERCASE", "HELLO WORLD"),
        ("lowercase", "hello world"),
        ("Capitalize", "Hello world"),
        ("Title Case", "Hello World"),
    ],
)
def test_case_converter(mode: str, expected: str) -> None:
    assert CaseConverter.execute(string="hELLO wORLD", mode=mode) == {"STRING": expected}


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("Both", "text"), ("Left", "text \n"), ("Right", "\t text")],
)
def test_trim(mode: str, expected: str) -> None:
    assert StringTrim.execute(string="\t text \n", mode=mode) == {"STRING": expected}


@pytest.mark.parametrize("case_sensitive", [True, False])
def test_contains_case_policy(case_sensitive: bool) -> None:
    assert StringContains.execute(
        string="My Alpha",
        substring="alpha",
        case_sensitive=case_sensitive,
    ) == {"contains": not case_sensitive}


@pytest.mark.parametrize(
    ("mode", "a", "b", "expected"),
    [
        ("Equal", "Alpha", "alpha", True),
        ("Equal", "Alphab", "alpha", False),
        ("Starts With", "AlphaBeta", "alpha", True),
        ("Starts With", "BetaAlpha", "alpha", False),
        ("Ends With", "BetaAlpha", "alpha", True),
        ("Ends With", "AlphaBeta", "alpha", False),
    ],
)
def test_compare(mode: str, a: str, b: str, expected: bool) -> None:
    assert StringCompare.execute(
        string_a=a,
        string_b=b,
        mode=mode,
        case_sensitive=False,
    ) == {"BOOLEAN": expected}
    assert StringCompare.execute(
        string_a=a,
        string_b=b,
        mode=mode,
        case_sensitive=True,
    ) == {"BOOLEAN": False}


@pytest.mark.parametrize(
    ("pattern", "case_insensitive", "multiline", "dotall", "expected"),
    [
        ("BETA", True, False, False, True),
        ("BETA", False, False, False, False),
        ("^beta", False, True, False, True),
        ("^beta", False, False, False, False),
        ("A.*beta", False, False, True, True),
        ("A.*beta", False, False, False, False),
        ("[", True, True, True, False),
    ],
)
def test_regex_search_flags_and_invalid_pattern(
    pattern: str,
    case_insensitive: bool,
    multiline: bool,
    dotall: bool,
    expected: bool,
) -> None:
    assert RegexMatch.execute(
        string="A\nbeta",
        regex_pattern=pattern,
        case_insensitive=case_insensitive,
        multiline=multiline,
        dotall=dotall,
    ) == {"matches": expected}


@pytest.mark.parametrize(
    ("mode", "group_index", "expected"),
    [
        ("First Match", 1, "A=12"),
        ("All Matches", 1, "A\nB"),
        ("First Group", 2, "12"),
        ("First Group", 0, "A=12"),
        ("All Groups", 2, "12\n34"),
        ("All Groups", 0, "A=12\nB=34"),
        ("First Group", 3, ""),
        ("All Groups", 3, ""),
    ],
)
def test_regex_extract_groups(mode: str, group_index: int, expected: str) -> None:
    assert RegexExtract.execute(
        string="A=12 B=34",
        regex_pattern=r"([A-Z])=(\d+)",
        mode=mode,
        case_insensitive=False,
        multiline=False,
        dotall=False,
        group_index=group_index,
    ) == {"STRING": expected}


@pytest.mark.parametrize("pattern", ["no match", "["])
def test_regex_extract_empty_or_invalid(pattern: str) -> None:
    assert RegexExtract.execute(
        string="abc",
        regex_pattern=pattern,
        mode="First Match",
        case_insensitive=True,
        multiline=False,
        dotall=False,
        group_index=1,
    ) == {"STRING": ""}


def test_regex_replace_optional_defaults_count_and_errors() -> None:
    assert RegexReplace.execute(string="A1 a2 a3", regex_pattern=r"a(\d)", replace=r"b\1") == {
        "STRING": "b1 b2 b3"
    }
    assert RegexReplace.execute(
        string="A1 a2 a3",
        regex_pattern=r"a(\d)",
        replace=r"b\1",
        count=1,
        case_insensitive=False,
    ) == {"STRING": "A1 b2 a3"}
    with pytest.raises(re.error):
        RegexReplace.execute(string="abc", regex_pattern="[", replace="x")


@pytest.mark.parametrize(
    ("text", "key", "expected"),
    [
        ('prose {broken ```json\n{"name": "Ada"}\n```', "name", "Ada"),
        ('{"other": 1} {"name": "second"}', "name", ""),
        ('{"name": null}', "name", ""),
        ('{"name": false}', "name", "False"),
        ('{"name": 0}', "name", "0"),
        ("no object", "name", ""),
    ],
)
def test_json_extract_uses_first_valid_object(text: str, key: str, expected: str) -> None:
    assert JsonExtractString.execute(json_string=text, key=key) == {"STRING": expected}


def test_json_serialization_retains_unicode_order_and_indent() -> None:
    assert ConvertDictionaryToString.execute(dictionary={"z": "\u00e9", "a": 2}) == {
        "STRING": '{\n  "z": "\u00e9",\n  "a": 2\n}'
    }
    assert ConvertArrayToString.execute(array=["\u00e9", {"b": False}], indent=0) == {
        "STRING": '["\u00e9", {"b": false}]'
    }
