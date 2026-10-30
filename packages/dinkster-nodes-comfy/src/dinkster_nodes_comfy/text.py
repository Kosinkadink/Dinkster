"""Text nodes with the interfaces and behavior of ComfyUI core nodes."""

from __future__ import annotations

import json
import re
import string
from collections.abc import Mapping
from typing import cast

from dinkster_api.v1 import (
    ARRAY_TYPE,
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_INT,
    CORE_STRING,
    DICT_TYPE,
    ComboWidget,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
)

STRING = TypeExpr.concrete(CORE_STRING)
INT = TypeExpr.concrete(CORE_INT)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
COMBO = TypeExpr.concrete(CORE_COMBO)


def _text(name: str, *, multiline: bool = True, default: str | None = None) -> InputSpec:
    return InputSpec(name, STRING, default=default, widget=StringWidget(multiline=multiline))


def _combo(name: str, options: tuple[str, ...]) -> InputSpec:
    return InputSpec(name, COMBO, widget=ComboWidget(options=options))


def _regex_inputs(*, optional: bool = False) -> tuple[InputSpec, ...]:
    return tuple(
        InputSpec(name, BOOLEAN, required=not optional, default=default, advanced=True)
        for name, default in (("case_insensitive", True), ("multiline", False), ("dotall", False))
    )


def _regex_flags(case_insensitive: bool, multiline: bool, dotall: bool) -> int:
    return (
        (re.IGNORECASE if case_insensitive else 0)
        | (re.MULTILINE if multiline else 0)
        | (re.DOTALL if dotall else 0)
    )


class StringFormat(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="StringFormat",
            display_name="Format Text",
            category="text",
            inputs=(_text("f_string", default="{a}"),),
            input_order=("values", "f_string"),
            input_families=(
                InputFamilySpec(
                    "values",
                    (InputSpec("value", TypeExpr.wildcard(), required=False),),
                    min_members=0,
                    member_names=tuple(string.ascii_lowercase),
                ),
            ),
            outputs=(OutputSpec("STRING", STRING),),
        )

    @classmethod
    def execute(cls, *, values: Mapping[str, object], f_string: str) -> Mapping[str, object]:
        return cls.outputs(STRING=f_string.format(**values))


class StringConcatenate(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="StringConcatenate",
            display_name="Concatenate Text",
            category="text",
            inputs=(
                _text("string_a"),
                _text("string_b"),
                _text("delimiter", multiline=False, default=""),
            ),
            outputs=(OutputSpec("STRING", STRING),),
        )

    @classmethod
    def execute(cls, *, string_a: str, string_b: str, delimiter: str) -> Mapping[str, object]:
        return cls.outputs(STRING=delimiter.join((string_a, string_b)))


class StringSubstring(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="StringSubstring",
            display_name="Substring",
            category="text",
            inputs=(_text("string"), InputSpec("start", INT), InputSpec("end", INT)),
            outputs=(OutputSpec("STRING", STRING),),
        )

    @classmethod
    def execute(cls, *, string: str, start: int, end: int) -> Mapping[str, object]:
        return cls.outputs(STRING=string[start:end])


class StringLength(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="StringLength",
            display_name="Text Length",
            category="text",
            inputs=(_text("string"),),
            outputs=(OutputSpec("length", INT),),
        )

    @classmethod
    def execute(cls, *, string: str) -> Mapping[str, object]:
        return cls.outputs(length=len(string))


class CaseConverter(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="CaseConverter",
            display_name="Convert Text Case",
            category="text",
            inputs=(
                _text("string"),
                _combo("mode", ("UPPERCASE", "lowercase", "Capitalize", "Title Case")),
            ),
            outputs=(OutputSpec("STRING", STRING),),
        )

    @classmethod
    def execute(cls, *, string: str, mode: str) -> Mapping[str, object]:
        if mode == "UPPERCASE":
            result = string.upper()
        elif mode == "lowercase":
            result = string.lower()
        elif mode == "Capitalize":
            result = string.capitalize()
        elif mode == "Title Case":
            result = string.title()
        else:
            result = string
        return cls.outputs(STRING=result)


class StringTrim(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="StringTrim",
            display_name="Trim Text",
            category="text",
            inputs=(_text("string"), _combo("mode", ("Both", "Left", "Right"))),
            outputs=(OutputSpec("STRING", STRING),),
        )

    @classmethod
    def execute(cls, *, string: str, mode: str) -> Mapping[str, object]:
        if mode == "Both":
            result = string.strip()
        elif mode == "Left":
            result = string.lstrip()
        elif mode == "Right":
            result = string.rstrip()
        else:
            result = string
        return cls.outputs(STRING=result)


class StringReplace(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="StringReplace",
            display_name="Replace Text",
            category="text",
            inputs=(_text("string"), _text("find"), _text("replace")),
            outputs=(OutputSpec("STRING", STRING),),
        )

    @classmethod
    def execute(cls, *, string: str, find: str, replace: str) -> Mapping[str, object]:
        return cls.outputs(STRING=string.replace(find, replace))


class StringContains(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="StringContains",
            display_name="Contains Text",
            category="text",
            inputs=(
                _text("string"),
                _text("substring"),
                InputSpec("case_sensitive", BOOLEAN, default=True, advanced=True),
            ),
            outputs=(OutputSpec("contains", BOOLEAN),),
        )

    @classmethod
    def execute(cls, *, string: str, substring: str, case_sensitive: bool) -> Mapping[str, object]:
        if not case_sensitive:
            string, substring = string.lower(), substring.lower()
        return cls.outputs(contains=substring in string)


class StringCompare(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="StringCompare",
            display_name="Compare Text",
            category="text",
            inputs=(
                _text("string_a"),
                _text("string_b"),
                _combo("mode", ("Starts With", "Ends With", "Equal")),
                InputSpec("case_sensitive", BOOLEAN, default=True, advanced=True),
            ),
            outputs=(OutputSpec("BOOLEAN", BOOLEAN),),
        )

    @classmethod
    def execute(
        cls,
        *,
        string_a: str,
        string_b: str,
        mode: str,
        case_sensitive: bool,
    ) -> Mapping[str, object]:
        a, b = (string_a, string_b) if case_sensitive else (string_a.lower(), string_b.lower())
        if mode == "Equal":
            result = a == b
        elif mode == "Starts With":
            result = a.startswith(b)
        elif mode == "Ends With":
            result = a.endswith(b)
        else:
            raise ValueError(f"Unknown comparison mode: {mode}")
        return cls.outputs(BOOLEAN=result)


class RegexMatch(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="RegexMatch",
            display_name="Match Text",
            category="text",
            inputs=(_text("string"), _text("regex_pattern"), *_regex_inputs()),
            outputs=(OutputSpec("matches", BOOLEAN),),
        )

    @classmethod
    def execute(
        cls,
        *,
        string: str,
        regex_pattern: str,
        case_insensitive: bool,
        multiline: bool,
        dotall: bool,
    ) -> Mapping[str, object]:
        try:
            result = (
                re.search(
                    regex_pattern,
                    string,
                    _regex_flags(case_insensitive, multiline, dotall),
                )
                is not None
            )
        except re.error:
            result = False
        return cls.outputs(matches=result)


class RegexExtract(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="RegexExtract",
            display_name="Extract Text",
            category="text",
            inputs=(
                _text("string"),
                _text("regex_pattern"),
                _combo("mode", ("First Match", "All Matches", "First Group", "All Groups")),
                *_regex_inputs(),
                InputSpec(
                    "group_index",
                    INT,
                    default=1,
                    advanced=True,
                    widget=NumberWidget(min=0, max=100),
                ),
            ),
            outputs=(OutputSpec("STRING", STRING),),
        )

    @classmethod
    def execute(
        cls,
        *,
        string: str,
        regex_pattern: str,
        mode: str,
        case_insensitive: bool,
        multiline: bool,
        dotall: bool,
        group_index: int,
    ) -> Mapping[str, object]:
        flags = _regex_flags(case_insensitive, multiline, dotall)
        result = ""
        try:
            if mode == "First Match":
                match = re.search(regex_pattern, string, flags)
                result = match.group(0) if match else ""
            elif mode == "All Matches":
                matches = cast(
                    "list[str | tuple[str, ...]]", re.findall(regex_pattern, string, flags)
                )
                result = "\n".join(m[0] if isinstance(m, tuple) else m for m in matches)
            elif mode == "First Group":
                match = re.search(regex_pattern, string, flags)
                if match and len(match.groups()) >= group_index:
                    result = match.group(group_index)
            elif mode == "All Groups":
                result = "\n".join(
                    match.group(group_index)
                    for match in re.finditer(regex_pattern, string, flags)
                    if match.groups() and len(match.groups()) >= group_index
                )
        except re.error:
            result = ""
        return cls.outputs(STRING=result)


class RegexReplace(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="RegexReplace",
            display_name="Replace Text (Regex)",
            category="text",
            inputs=(
                _text("string"),
                _text("regex_pattern"),
                _text("replace"),
                *_regex_inputs(optional=True),
                InputSpec(
                    "count",
                    INT,
                    required=False,
                    default=0,
                    advanced=True,
                    widget=NumberWidget(min=0, max=100),
                ),
            ),
            outputs=(OutputSpec("STRING", STRING),),
        )

    @classmethod
    def execute(
        cls,
        *,
        string: str,
        regex_pattern: str,
        replace: str,
        case_insensitive: bool = True,
        multiline: bool = False,
        dotall: bool = False,
        count: int = 0,
    ) -> Mapping[str, object]:
        return cls.outputs(
            STRING=re.sub(
                regex_pattern,
                replace,
                string,
                count=count,
                flags=_regex_flags(case_insensitive, multiline, dotall),
            )
        )


class JsonExtractString(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="JsonExtractString",
            display_name="Extract Text from JSON",
            category="text",
            inputs=(_text("json_string"), _text("key", multiline=False)),
            outputs=(OutputSpec("STRING", STRING),),
        )

    @classmethod
    def execute(cls, *, json_string: str, key: str) -> Mapping[str, object]:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", json_string):
            try:
                data, _ = decoder.raw_decode(json_string, match.start())
            except json.JSONDecodeError:
                continue
            value = data.get(key)
            return cls.outputs(STRING="" if value is None else str(value))
        return cls.outputs(STRING="")


class ConvertDictionaryToString(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ConvertDictionaryToString",
            display_name="Convert Dictionary to String",
            category="text",
            inputs=(
                InputSpec("dictionary", TypeExpr.concrete(DICT_TYPE)),
                InputSpec("indent", INT, default=2, widget=NumberWidget(min=0, max=8)),
            ),
            outputs=(OutputSpec("STRING", STRING),),
        )

    @classmethod
    def execute(cls, *, dictionary: dict[str, object], indent: int = 2) -> Mapping[str, object]:
        return cls.outputs(STRING=json.dumps(dictionary, ensure_ascii=False, indent=indent or None))


class ConvertArrayToString(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ConvertArrayToString",
            display_name="Convert Array to String",
            category="text",
            inputs=(
                InputSpec("array", TypeExpr.concrete(ARRAY_TYPE)),
                InputSpec("indent", INT, default=2, widget=NumberWidget(min=0, max=8)),
            ),
            outputs=(OutputSpec("STRING", STRING),),
        )

    @classmethod
    def execute(cls, *, array: list[object], indent: int = 2) -> Mapping[str, object]:
        return cls.outputs(STRING=json.dumps(array, ensure_ascii=False, indent=indent or None))


TEXT_NODES = (
    StringFormat,
    StringConcatenate,
    StringSubstring,
    StringLength,
    CaseConverter,
    StringTrim,
    StringReplace,
    StringContains,
    StringCompare,
    RegexMatch,
    RegexExtract,
    RegexReplace,
    JsonExtractString,
    ConvertDictionaryToString,
    ConvertArrayToString,
)
