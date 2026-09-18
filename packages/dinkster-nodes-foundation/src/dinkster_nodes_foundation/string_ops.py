"""Deterministic string, regular-expression, JSON, and CSV operations."""

from __future__ import annotations

import csv
import io
import json
import re
import string
from collections.abc import Mapping, Sequence
from typing import cast

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    ComboWidget,
    ConditionalWidgetGroup,
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
LIST_STRING = TypeExpr.list_of(STRING)
LIST_LIST_STRING = TypeExpr.list_of(LIST_STRING)

MAX_TEXT_BYTES = 1_048_576
MAX_TEMPLATE_BYTES = 65_536
MAX_FORMAT_WIDTH = 65_536
MAX_FORMAT_PRECISION = 1_000
MAX_REGEX_PATTERN_BYTES = 4_096
MAX_REGEX_RESULTS = 10_000
MAX_JOIN_ITEMS = 10_000
MAX_SPLIT_PARTS = 10_000
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 100_000
MAX_JSON_MEMBERS = 10_000
MAX_JSON_STRING_BYTES = 65_536
MAX_CSV_ROWS = 10_000
MAX_CSV_COLUMNS = 256
MAX_CSV_CELLS = 1_000_000
MAX_CSV_CELL_BYTES = 65_536


def _require_text_size(value: str, *, limit: int = MAX_TEXT_BYTES, subject: str = "text") -> str:
    if len(value.encode("utf-8")) > limit:
        raise ValueError(f"{subject} exceeds {limit} UTF-8 bytes")
    return value


def _join_bounded(values: Sequence[str], separator: str, *, subject: str) -> str:
    encoded_separator = len(separator.encode("utf-8"))
    encoded_size = sum(len(value.encode("utf-8")) for value in values)
    if values:
        encoded_size += encoded_separator * (len(values) - 1)
    if encoded_size > MAX_TEXT_BYTES:
        raise ValueError(f"{subject} exceeds {MAX_TEXT_BYTES} UTF-8 bytes")
    return separator.join(values)


def _enum_input(
    identifier: str, options: tuple[str, ...], default: str, *, advanced: bool = False
) -> InputSpec:
    return InputSpec(
        identifier,
        COMBO,
        default=default,
        widget=ComboWidget(options=options),
        advanced=advanced,
    )


_FORMAT_SPEC = re.compile(
    r"(?:(?P<fill>.)(?P<align>[<^>])|(?P<align_only>[<^>]))?"
    r"(?P<sign>[+ -])?(?P<zero>0)?(?P<width>\d+)?"
    r"(?:\.(?P<precision>\d+))?(?P<type>[sdboxXfeEgG%])?"
)


def _validate_format_spec(spec: str) -> None:
    match = _FORMAT_SPEC.fullmatch(spec)
    if match is None:
        raise ValueError(f"unsupported format specifier {spec!r}")
    width = match.group("width")
    precision = match.group("precision")
    if width is not None and int(width) > MAX_FORMAT_WIDTH:
        raise ValueError(f"format width exceeds {MAX_FORMAT_WIDTH}")
    if precision is not None and int(precision) > MAX_FORMAT_PRECISION:
        raise ValueError(f"format precision exceeds {MAX_FORMAT_PRECISION}")
    fill = match.group("fill")
    if fill in ("{", "}"):
        raise ValueError("format fill cannot be a brace")


def format_string(template: str, slots: Mapping[str, object]) -> str:
    """Apply the bounded formatting grammar used by ``dinkster.string.format``."""
    _require_text_size(template, limit=MAX_TEMPLATE_BYTES, subject="template")
    formatter = string.Formatter()
    pieces: list[str] = []
    encoded_size = 0

    def append(piece: str) -> None:
        nonlocal encoded_size
        encoded_size += len(piece.encode("utf-8"))
        if encoded_size > MAX_TEXT_BYTES:
            raise ValueError(f"formatted text exceeds {MAX_TEXT_BYTES} UTF-8 bytes")
        pieces.append(piece)

    try:
        parsed = formatter.parse(template)
        for literal, field_name, format_spec, conversion in parsed:
            append(literal)
            if field_name is None:
                continue
            format_spec = format_spec or ""
            if len(field_name) != 1 or field_name not in string.ascii_lowercase:
                raise ValueError("format fields must be one lowercase letter")
            if conversion is not None:
                raise ValueError("format conversions are not supported")
            if "{" in format_spec or "}" in format_spec:
                raise ValueError("nested format fields are not supported")
            _validate_format_spec(format_spec)
            if field_name not in slots:
                raise ValueError(f"missing format slot {field_name!r}")
            value = slots[field_name]
            if value is not None and type(value) not in (str, int, float, bool):
                raise TypeError("format slots must be JSON scalar values")
            if type(value) is float and not _finite_float(value):
                raise ValueError("format floats must be finite")
            append(format(value, format_spec))
    except (KeyError, IndexError) as exc:
        raise ValueError(f"invalid format field: {exc}") from None
    return "".join(pieces)


class StringFormat(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.string.format",
            display_name="Format Text",
            category="string",
            inputs=(
                InputSpec(
                    "template",
                    STRING,
                    default="{a}",
                    widget=StringWidget(multiline=True),
                ),
            ),
            input_families=(
                InputFamilySpec(
                    "slots",
                    TypeExpr.union(CORE_STRING, CORE_INT, CORE_FLOAT, CORE_BOOLEAN),
                    min_members=0,
                    member_names=tuple(string.ascii_lowercase),
                ),
            ),
            outputs=(OutputSpec("text", STRING),),
        )

    @classmethod
    def execute(cls, *, template: str, slots: Mapping[str, object]) -> Mapping[str, object]:
        return cls.outputs(text=format_string(template, slots))


TRANSFORM_OPERATIONS = (
    "identity",
    "upper",
    "lower",
    "capitalize",
    "title",
    "trim",
    "trim_left",
    "trim_right",
    "prefix",
    "suffix",
    "replace_literal",
    "slice",
    "take",
    "pad_left",
    "pad_right",
    "pad_center",
    "join_nonempty_lines",
    "select_line",
    "select_hash_section",
)


def transform_string(
    *,
    text: str,
    operation: str,
    value: str,
    replacement: str,
    start: int,
    end: int,
    count: int,
    fill: str,
    unit: str,
    side: str,
) -> str:
    _require_text_size(text)
    _require_text_size(value, subject="value")
    _require_text_size(replacement, subject="replacement")
    if operation == "identity":
        result = text
    elif operation == "upper":
        result = text.upper()
    elif operation == "lower":
        result = text.lower()
    elif operation == "capitalize":
        result = text.capitalize()
    elif operation == "title":
        result = text.title()
    elif operation == "trim":
        result = text.strip()
    elif operation == "trim_left":
        result = text.lstrip()
    elif operation == "trim_right":
        result = text.rstrip()
    elif operation == "prefix":
        result = _join_bounded((value, text), "", subject="transformed text")
    elif operation == "suffix":
        result = _join_bounded((text, value), "", subject="transformed text")
    elif operation == "replace_literal":
        occurrences = len(text) + 1 if value == "" else text.count(value)
        result_size = len(text.encode("utf-8")) + occurrences * (
            len(replacement.encode("utf-8")) - len(value.encode("utf-8"))
        )
        if result_size > MAX_TEXT_BYTES:
            raise ValueError(f"transformed text exceeds {MAX_TEXT_BYTES} UTF-8 bytes")
        result = text.replace(value, replacement)
    elif operation == "slice":
        result = text[start:end]
    elif operation == "take":
        if unit == "characters":
            sequence: Sequence[str] = text
            joiner = ""
        elif unit == "words":
            sequence = text.split()
            joiner = " "
        else:
            raise ValueError(f"unknown take unit {unit!r}")
        if side == "start":
            selected = sequence[:count] if count >= 0 else sequence[count:]
        elif side == "end":
            selected = sequence[-count:] if count >= 0 else sequence[:count]
        else:
            raise ValueError(f"unknown take side {side!r}")
        result = joiner.join(selected)
    elif operation in ("pad_left", "pad_right", "pad_center"):
        if len(fill) != 1:
            raise ValueError("padding fill must be one Unicode code point")
        if not 0 <= count <= MAX_FORMAT_WIDTH:
            raise ValueError(f"padding width must be between 0 and {MAX_FORMAT_WIDTH}")
        if operation == "pad_left":
            result = text.rjust(count, fill)
        elif operation == "pad_right":
            result = text.ljust(count, fill)
        else:
            result = text.center(count, fill)
    elif operation == "join_nonempty_lines":
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        result = _join_bounded(lines, value, subject="transformed text")
    elif operation == "select_line":
        lines = text.split("\n")
        result = lines[count % len(lines)]
    elif operation == "select_hash_section":
        sections: list[str] = []
        current: list[str] = []
        for line in text.split("\n"):
            if line.startswith("#"):
                if current:
                    sections.append("\n".join(current).strip())
                    current = []
            current.append(line)
        if current:
            sections.append("\n".join(current).strip())
        result = sections[count % len(sections)]
        if result.startswith("#"):
            result = result[1:]
    else:
        raise ValueError(f"unknown string transform operation {operation!r}")
    return _require_text_size(result, subject="transformed text")


class StringTransform(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.string.transform",
            display_name="Transform Text",
            category="string",
            inputs=(
                InputSpec("text", STRING, widget=StringWidget(multiline=True)),
                _enum_input("operation", TRANSFORM_OPERATIONS, "trim"),
                InputSpec("value", STRING, default=""),
                InputSpec("replacement", STRING, default=""),
                InputSpec("start", INT, default=0),
                InputSpec("end", INT, default=(1 << 53) - 1),
                InputSpec("count", INT, default=0),
                InputSpec("fill", STRING, default=" "),
                _enum_input("unit", ("characters", "words"), "characters"),
                _enum_input("side", ("start", "end"), "start"),
            ),
            widget_groups=(ConditionalWidgetGroup("operation", ("take",), ("unit", "side")),),
            outputs=(OutputSpec("text", STRING),),
        )

    @classmethod
    def execute(
        cls,
        *,
        text: str,
        operation: str,
        value: str,
        replacement: str,
        start: int,
        end: int,
        count: int,
        fill: str,
        unit: str,
        side: str,
    ) -> Mapping[str, object]:
        return cls.outputs(
            text=transform_string(
                text=text,
                operation=operation,
                value=value,
                replacement=replacement,
                start=start,
                end=end,
                count=count,
                fill=fill,
                unit=unit,
                side=side,
            )
        )


class StringTest(Node):
    OPERATIONS = ("equals", "not_equals", "contains", "starts_with", "ends_with")

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.string.test",
            display_name="Test Text",
            category="string",
            inputs=(
                InputSpec("text", STRING),
                InputSpec("query", STRING),
                _enum_input("operation", cls.OPERATIONS, "contains"),
                _enum_input("case_mode", ("sensitive", "lower_both"), "sensitive"),
            ),
            outputs=(OutputSpec("result", BOOLEAN),),
        )

    @classmethod
    def execute(
        cls, *, text: str, query: str, operation: str, case_mode: str
    ) -> Mapping[str, object]:
        _require_text_size(text)
        _require_text_size(query, subject="query")
        if case_mode == "lower_both":
            text, query = text.lower(), query.lower()
        elif case_mode != "sensitive":
            raise ValueError(f"unknown case mode {case_mode!r}")
        if operation == "equals":
            result = text == query
        elif operation == "not_equals":
            result = text != query
        elif operation == "contains":
            result = query in text
        elif operation == "starts_with":
            result = text.startswith(query)
        elif operation == "ends_with":
            result = text.endswith(query)
        else:
            raise ValueError(f"unknown string test operation {operation!r}")
        return cls.outputs(result=result)


class StringLength(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.string.length",
            display_name="Text Length",
            category="string",
            inputs=(InputSpec("text", STRING, widget=StringWidget(multiline=True)),),
            outputs=(OutputSpec("length", INT),),
        )

    @classmethod
    def execute(cls, *, text: str) -> Mapping[str, object]:
        _require_text_size(text)
        return cls.outputs(length=len(text))


REGEX_OPERATIONS = (
    "search",
    "match",
    "extract_first",
    "extract_all_legacy",
    "extract_group_first",
    "extract_group_all",
    "replace",
)


def _regex_flags(case_mode: str, multiline: bool, dotall: bool) -> int:
    flags = 0
    if case_mode == "unicode_ignorecase":
        flags |= re.IGNORECASE
    elif case_mode not in ("sensitive", "lower_both"):
        raise ValueError(f"unknown regex case mode {case_mode!r}")
    if multiline:
        flags |= re.MULTILINE
    if dotall:
        flags |= re.DOTALL
    return flags


def _regex_replacement_upper_bound(match: re.Match[str], replacement: str) -> int:
    def group_size(identifier: str) -> int:
        try:
            value = match.group(int(identifier) if identifier.isdecimal() else identifier)
        except (IndexError, KeyError):
            return MAX_TEXT_BYTES
        return len((value or "").encode("utf-8"))

    encoded_size = 0
    index = 0
    while index < len(replacement):
        if encoded_size > MAX_TEXT_BYTES:
            return encoded_size
        if replacement[index] != "\\":
            encoded_size += len(replacement[index].encode("utf-8"))
            index += 1
            continue
        if index + 1 >= len(replacement):
            encoded_size += 1
            break
        escaped = replacement[index + 1]
        if escaped == "g" and index + 2 < len(replacement) and replacement[index + 2] == "<":
            end = replacement.find(">", index + 3)
            if end < 0:
                encoded_size += MAX_TEXT_BYTES
                break
            encoded_size += group_size(replacement[index + 3 : end])
            index = end + 1
        elif (
            escaped in "01234567"
            and index + 3 < len(replacement)
            and all(character in "01234567" for character in replacement[index + 2 : index + 4])
        ):
            encoded_size += len(replacement[index : index + 4].encode("utf-8"))
            index += 4
        elif escaped in "123456789":
            identifier = escaped
            if index + 2 < len(replacement) and replacement[index + 2].isdecimal():
                identifier += replacement[index + 2]
                index += 1
            encoded_size += group_size(identifier)
            index += 2
        elif escaped == "0" or escaped in "\\abfnrtv":
            encoded_size += len(replacement[index : index + 2].encode("utf-8"))
            index += 2
        else:
            encoded_size += len(replacement[index : index + 2].encode("utf-8"))
            index += 2
    return encoded_size


class StringRegex(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.string.regex",
            display_name="Regular Expression",
            category="string",
            inputs=(
                InputSpec("text", STRING, widget=StringWidget(multiline=True)),
                InputSpec("pattern", STRING),
                _enum_input("operation", REGEX_OPERATIONS, "search"),
                InputSpec("replacement", STRING, default=""),
                InputSpec(
                    "group_index", INT, default=1, widget=NumberWidget(min=0, max=100, step=1)
                ),
                InputSpec("count", INT, default=0, widget=NumberWidget(min=0, max=100, step=1)),
                _enum_input(
                    "case_mode",
                    ("sensitive", "unicode_ignorecase", "lower_both"),
                    "sensitive",
                ),
                InputSpec("multiline", BOOLEAN, default=False),
                InputSpec("dotall", BOOLEAN, default=False),
            ),
            outputs=(
                OutputSpec("matched", BOOLEAN),
                OutputSpec("text", STRING),
                OutputSpec("count", INT),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        text: str,
        pattern: str,
        operation: str,
        replacement: str,
        group_index: int,
        count: int,
        case_mode: str,
        multiline: bool,
        dotall: bool,
    ) -> Mapping[str, object]:
        _require_text_size(text)
        _require_text_size(pattern, limit=MAX_REGEX_PATTERN_BYTES, subject="regex pattern")
        _require_text_size(replacement, subject="regex replacement")
        if not 0 <= group_index <= 100 or not 0 <= count <= 100:
            raise ValueError("regex group_index and count must be between 0 and 100")
        if case_mode == "lower_both":
            text, pattern = text.lower(), pattern.lower()
        flags = _regex_flags(case_mode, multiline, dotall)
        try:
            compiled: re.Pattern[str] = re.compile(pattern, flags)
        except re.error:
            if operation == "replace":
                raise
            return cls.outputs(matched=False, text="", count=0)

        if operation in ("search", "match"):
            match = compiled.search(text) if operation == "search" else compiled.match(text)
            return cls.outputs(matched=match is not None, text="", count=int(match is not None))
        if operation == "replace":
            encoded_size = 0
            previous_end = 0
            replacement_matches = 0

            def replace(match: re.Match[str]) -> str:
                nonlocal encoded_size, previous_end, replacement_matches
                replacement_matches += 1
                if replacement_matches > MAX_REGEX_RESULTS:
                    raise ValueError(f"regex results exceed {MAX_REGEX_RESULTS}")
                unmatched_size = len(text[previous_end : match.start()].encode("utf-8"))
                upper_bound = _regex_replacement_upper_bound(match, replacement)
                if encoded_size + unmatched_size + upper_bound > MAX_TEXT_BYTES:
                    raise ValueError(
                        f"regex replacement result exceeds {MAX_TEXT_BYTES} UTF-8 bytes"
                    )
                expanded = match.expand(replacement)
                encoded_size += unmatched_size
                encoded_size += len(expanded.encode("utf-8"))
                previous_end = match.end()
                return expanded

            result, replacements = compiled.subn(replace, text, count=count)
            return cls.outputs(
                matched=replacements > 0,
                text=_require_text_size(result, subject="regex replacement result"),
                count=replacements,
            )

        if operation not in (
            "extract_first",
            "extract_all_legacy",
            "extract_group_first",
            "extract_group_all",
        ):
            raise ValueError(f"unknown regex operation {operation!r}")
        values: list[str] = []
        matched = False
        encoded_size = 0
        for index, match in enumerate(compiled.finditer(text)):
            if index >= MAX_REGEX_RESULTS:
                raise ValueError(f"regex results exceed {MAX_REGEX_RESULTS}")
            matched = True
            selected: str | None = None
            if operation == "extract_first":
                selected = match.group(0)
            elif operation == "extract_all_legacy":
                group = 1 if compiled.groups else 0
                selected = match.group(group) or ""
            elif operation == "extract_group_first":
                if group_index <= compiled.groups:
                    selected = match.group(group_index) or ""
            elif operation == "extract_group_all":
                if compiled.groups and group_index <= compiled.groups:
                    selected = match.group(group_index) or ""
            if selected is not None:
                encoded_size += len(selected.encode("utf-8")) + int(bool(values))
                if encoded_size > MAX_TEXT_BYTES:
                    raise ValueError(
                        f"regex extraction result exceeds {MAX_TEXT_BYTES} UTF-8 bytes"
                    )
                values.append(selected)
            if operation in ("extract_first", "extract_group_first"):
                break
        result = "\n".join(values)
        return cls.outputs(
            matched=matched,
            text=result,
            count=len(values),
        )


class StringJoin(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.string.join",
            display_name="Join Text",
            category="string",
            inputs=(
                InputSpec("items", LIST_STRING, required=False),
                InputSpec("separator", STRING, default=", "),
                InputSpec("trim", BOOLEAN, default=False),
                InputSpec("skip_empty", BOOLEAN, default=False),
                _enum_input("delimiter_escape", ("raw", "newline"), "raw", advanced=True),
            ),
            input_families=(InputFamilySpec("pieces", STRING, min_members=0, max_members=100),),
            outputs=(OutputSpec("text", STRING),),
        )

    @classmethod
    def execute(
        cls,
        *,
        separator: str,
        trim: bool,
        skip_empty: bool,
        delimiter_escape: str,
        pieces: Mapping[str, str],
        items: list[str] | None = None,
    ) -> Mapping[str, object]:
        values = [*(items or ()), *pieces.values()]
        if len(values) > MAX_JOIN_ITEMS:
            raise ValueError(f"join items exceed {MAX_JOIN_ITEMS}")
        _require_text_size(separator, subject="join separator")
        for value in values:
            _require_text_size(value, subject="join item")
        if trim:
            values = [value.strip() for value in values]
        if skip_empty:
            values = [value for value in values if value]
        if delimiter_escape == "newline":
            if separator == r"\n":
                separator = "\n"
        elif delimiter_escape != "raw":
            raise ValueError(f"unknown delimiter escape mode {delimiter_escape!r}")
        return cls.outputs(text=_join_bounded(values, separator, subject="joined text"))


def _split_literal_any(text: str, delimiters: list[str], max_parts: int) -> list[str]:
    ordered = sorted(enumerate(delimiters), key=lambda item: (-len(item[1]), item[0]))
    pattern = re.compile("|".join(re.escape(delimiter) for _, delimiter in ordered))
    max_splits = MAX_SPLIT_PARTS if max_parts == 0 else max_parts - 1
    return pattern.split(text, maxsplit=max_splits)


class StringSplit(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.string.split",
            display_name="Split Text",
            category="string",
            inputs=(
                InputSpec("text", STRING, widget=StringWidget(multiline=True)),
                _enum_input(
                    "mode", ("literal_any", "whitespace", "lines", "characters"), "literal_any"
                ),
                InputSpec("delimiter", STRING, default=" "),
                InputSpec("keep_empty", BOOLEAN, default=False),
                InputSpec(
                    "max_parts",
                    INT,
                    default=0,
                    widget=NumberWidget(min=0, max=MAX_SPLIT_PARTS, step=1),
                ),
            ),
            input_families=(
                InputFamilySpec("additional_delimiters", STRING, min_members=0, max_members=15),
            ),
            outputs=(OutputSpec("parts", LIST_STRING),),
        )

    @classmethod
    def execute(
        cls,
        *,
        text: str,
        mode: str,
        delimiter: str,
        keep_empty: bool,
        max_parts: int,
        additional_delimiters: Mapping[str, str],
    ) -> Mapping[str, object]:
        _require_text_size(text)
        delimiters = [delimiter, *additional_delimiters.values()]
        for candidate in delimiters:
            _require_text_size(
                candidate,
                limit=MAX_REGEX_PATTERN_BYTES,
                subject="split delimiter",
            )
        if not 0 <= max_parts <= MAX_SPLIT_PARTS:
            raise ValueError(f"max_parts must be between 0 and {MAX_SPLIT_PARTS}")
        if mode not in ("literal_any", "whitespace", "lines", "characters"):
            raise ValueError(f"unknown split mode {mode!r}")
        if mode == "literal_any" and any(not candidate for candidate in delimiters):
            raise ValueError("literal delimiters must be non-empty")
        if mode == "whitespace" and keep_empty:
            raise ValueError("whitespace splitting does not support keep_empty")
        if mode == "characters" and (delimiter != " " or additional_delimiters):
            raise ValueError("character splitting does not use delimiters")
        if max_parts == 1:
            parts = [text]
        elif mode == "literal_any":
            parts = _split_literal_any(text, delimiters, max_parts)
        elif mode == "whitespace":
            max_splits = MAX_SPLIT_PARTS if max_parts == 0 else max_parts - 1
            parts = text.split(maxsplit=max_splits)
        elif mode == "lines":
            max_splits = MAX_SPLIT_PARTS if max_parts == 0 else max_parts - 1
            parts = re.split(r"\r\n|\n|\r", text, maxsplit=max_splits)
        elif mode == "characters":
            if max_parts == 0 and len(text) > MAX_SPLIT_PARTS:
                raise ValueError(f"split results exceed {MAX_SPLIT_PARTS}")
            parts = (
                list(text)
                if max_parts == 0 or len(text) <= max_parts
                else [*text[: max_parts - 1], text[max_parts - 1 :]]
            )
        if not keep_empty and mode != "whitespace":
            parts = [part for part in parts if part]
        if len(parts) > MAX_SPLIT_PARTS:
            raise ValueError(f"split results exceed {MAX_SPLIT_PARTS}")
        return cls.outputs(parts=parts)


def _finite_float(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _validate_json_value(
    value: object, *, depth: int = 0, counter: list[int] | None = None
) -> None:
    if counter is None:
        counter = [0]
    counter[0] += 1
    if counter[0] > MAX_JSON_NODES:
        raise ValueError(f"JSON value exceeds {MAX_JSON_NODES} nodes")
    if depth > MAX_JSON_DEPTH:
        raise ValueError(f"JSON value exceeds depth {MAX_JSON_DEPTH}")
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not _finite_float(value):
            raise ValueError("JSON numbers must be finite")
        return
    if type(value) is str:
        _require_text_size(value, limit=MAX_JSON_STRING_BYTES, subject="JSON string")
        return
    if type(value) is list:
        items = cast("list[object]", value)
        if len(items) > MAX_JSON_MEMBERS:
            raise ValueError(f"JSON arrays are limited to {MAX_JSON_MEMBERS} items")
        for item in items:
            _validate_json_value(item, depth=depth + 1, counter=counter)
        return
    if type(value) is dict:
        items = cast("dict[object, object]", value)
        if len(items) > MAX_JSON_MEMBERS:
            raise ValueError(f"JSON objects are limited to {MAX_JSON_MEMBERS} members")
        for key, item in items.items():
            if type(key) is not str:
                raise TypeError("JSON object keys must be strings")
            _require_text_size(key, limit=MAX_JSON_STRING_BYTES, subject="JSON key")
            _validate_json_value(item, depth=depth + 1, counter=counter)
        return
    raise TypeError(f"value of type {type(value).__name__} is not JSON-compatible")


def _strict_json_loads(text: str) -> object:
    _require_text_size(text, subject="JSON text")

    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {constant}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from None
    except RecursionError:
        raise ValueError(f"JSON value exceeds depth {MAX_JSON_DEPTH}") from None
    _validate_json_value(value)
    return value


def _bounded_json_encode(value: object, encoder: json.JSONEncoder) -> str:
    chunks: list[str] = []
    encoded_size = 0
    for chunk in encoder.iterencode(value):
        encoded_size += len(chunk.encode("utf-8"))
        if encoded_size > MAX_TEXT_BYTES:
            raise ValueError(f"JSON output exceeds {MAX_TEXT_BYTES} UTF-8 bytes")
        chunks.append(chunk)
    return "".join(chunks)


def _json_dump(value: object, *, indent: int, key_order: str) -> str:
    _validate_json_value(value)
    if not 0 <= indent <= 8:
        raise ValueError("JSON indentation must be between 0 and 8")
    if key_order not in ("preserve", "sorted"):
        raise ValueError(f"unknown JSON key order {key_order!r}")
    if indent == 0:
        encoder = json.JSONEncoder(
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=key_order == "sorted",
            separators=(",", ":"),
        )
    else:
        encoder = json.JSONEncoder(
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=key_order == "sorted",
            indent=indent,
        )
    return _bounded_json_encode(value, encoder)


def _json_pointer(value: object, pointer: str) -> object:
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must be empty or start with '/'")
    current = value
    for raw_token in pointer[1:].split("/"):
        if re.search(r"~(?![01])", raw_token):
            raise ValueError("JSON pointer contains an invalid escape")
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if type(current) is dict:
            mapping = cast("dict[str, object]", current)
            if token not in mapping:
                raise ValueError(f"JSON pointer member {token!r} does not exist")
            current = mapping[token]
        elif type(current) is list:
            if not re.fullmatch(r"0|[1-9]\d*", token):
                raise ValueError(f"invalid JSON array index {token!r}")
            index = int(token)
            sequence = cast("list[object]", current)
            if index >= len(sequence):
                raise ValueError(f"JSON array index {index} is out of range")
            current = sequence[index]
        else:
            raise ValueError("JSON pointer traverses through a scalar")
    return current


class StringJson(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.string.json",
            display_name="JSON Text",
            category="string/structured",
            inputs=(
                _enum_input(
                    "operation",
                    ("minify", "pretty", "select", "extract_string_legacy"),
                    "minify",
                ),
                InputSpec("text", STRING, widget=StringWidget(multiline=True)),
                InputSpec("selector", STRING, default=""),
                InputSpec("indent", INT, default=2, widget=NumberWidget(min=0, max=8, step=1)),
                _enum_input("key_order", ("preserve", "sorted"), "preserve"),
            ),
            outputs=(OutputSpec("text", STRING),),
        )

    @classmethod
    def execute(
        cls, *, operation: str, text: str, selector: str, indent: int, key_order: str
    ) -> Mapping[str, object]:
        if operation == "extract_string_legacy":
            _require_text_size(text, subject="JSON text")
            try:
                value: object = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                result = ""
            else:
                selected = (
                    cast("dict[str, object]", value).get(selector) if type(value) is dict else None
                )
                result = "" if selected is None else str(selected)
            return cls.outputs(text=_require_text_size(result, subject="JSON extraction result"))
        value = _strict_json_loads(text)
        if operation == "minify":
            result = _json_dump(value, indent=0, key_order=key_order)
        elif operation == "pretty":
            result = _json_dump(value, indent=indent, key_order=key_order)
        elif operation == "select":
            result = _json_dump(_json_pointer(value, selector), indent=indent, key_order=key_order)
        else:
            raise ValueError(f"unknown JSON text operation {operation!r}")
        return cls.outputs(text=result)


class StringJsonEmit(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.string.json_emit",
            display_name="Value to JSON Text",
            category="string/structured",
            inputs=(
                InputSpec("value", TypeExpr.wildcard()),
                _enum_input("operation", ("strict", "legacy"), "strict"),
                InputSpec("indent", INT, default=0, widget=NumberWidget(min=0, max=8, step=1)),
                _enum_input("key_order", ("preserve", "sorted"), "preserve"),
            ),
            outputs=(OutputSpec("text", STRING),),
        )

    @classmethod
    def execute(
        cls, *, value: object, operation: str, indent: int, key_order: str
    ) -> Mapping[str, object]:
        if operation == "strict":
            result = _json_dump(value, indent=indent, key_order=key_order)
        elif operation == "legacy":
            if not 0 <= indent <= 8:
                raise ValueError("JSON indentation must be between 0 and 8")
            result = _bounded_json_encode(
                value,
                json.JSONEncoder(ensure_ascii=False, indent=indent or None),
            )
        else:
            raise ValueError(f"unknown JSON emit operation {operation!r}")
        return cls.outputs(text=result)


def _validate_csv_dialect(delimiter: str, quote: str) -> None:
    if len(delimiter) != 1 or len(quote) != 1:
        raise ValueError("CSV delimiter and quote must each be one Unicode code point")
    if delimiter in "\r\n" or quote in "\r\n" or delimiter == quote:
        raise ValueError("CSV delimiter and quote must be distinct non-newline characters")


def _validate_csv_header(header: Sequence[str]) -> int:
    if len(header) > MAX_CSV_COLUMNS:
        raise ValueError(f"CSV columns exceed {MAX_CSV_COLUMNS}")
    if len(set(header)) != len(header):
        raise ValueError("CSV header names must be unique")
    for cell in header:
        if type(cell) is not str:
            raise TypeError("CSV header cells must be strings")
        _require_text_size(cell, limit=MAX_CSV_CELL_BYTES, subject="CSV header cell")
    return len(header)


def _validate_csv_row(row: Sequence[str], expected: int) -> None:
    if len(row) != expected:
        raise ValueError("CSV rows must all have the same number of columns")
    for cell in row:
        if type(cell) is not str:
            raise TypeError("CSV cells must be strings")
        _require_text_size(cell, limit=MAX_CSV_CELL_BYTES, subject="CSV cell")


def _validate_csv_rows(rows: Sequence[Sequence[str]], header: Sequence[str] | None) -> int:
    if len(rows) > MAX_CSV_ROWS:
        raise ValueError(f"CSV rows exceed {MAX_CSV_ROWS}")
    expected = _validate_csv_header(header) if header is not None else (len(rows[0]) if rows else 0)
    if rows and expected == 0:
        raise ValueError("CSV rows must contain at least one column")
    if expected > MAX_CSV_COLUMNS:
        raise ValueError(f"CSV columns exceed {MAX_CSV_COLUMNS}")
    cells = len(header or ())
    for row in rows:
        _validate_csv_row(row, expected)
        cells += len(row)
        if cells > MAX_CSV_CELLS:
            raise ValueError(f"CSV cells exceed {MAX_CSV_CELLS}")
    return expected


class _BoundedTextBuffer:
    def __init__(self, subject: str) -> None:
        self._subject = subject
        self._chunks: list[str] = []
        self._encoded_size = 0

    def write(self, value: str) -> int:
        self._encoded_size += len(value.encode("utf-8"))
        if self._encoded_size > MAX_TEXT_BYTES:
            raise ValueError(f"{self._subject} exceeds {MAX_TEXT_BYTES} UTF-8 bytes")
        self._chunks.append(value)
        return len(value)

    def getvalue(self) -> str:
        return "".join(self._chunks)


class StringCsvParse(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.string.csv_parse",
            display_name="Parse CSV Text",
            category="string/structured",
            inputs=(
                InputSpec("text", STRING, widget=StringWidget(multiline=True)),
                InputSpec("delimiter", STRING, default=","),
                InputSpec("quote", STRING, default='"'),
                InputSpec("header", BOOLEAN, default=False),
                InputSpec("strip_bom", BOOLEAN, default=True),
            ),
            outputs=(OutputSpec("header", LIST_STRING), OutputSpec("rows", LIST_LIST_STRING)),
        )

    @classmethod
    def execute(
        cls, *, text: str, delimiter: str, quote: str, header: bool, strip_bom: bool
    ) -> Mapping[str, object]:
        _require_text_size(text, subject="CSV text")
        if strip_bom and text.startswith("\ufeff"):
            text = text[1:]
        _validate_csv_dialect(delimiter, quote)
        parsed: list[list[str]] = []
        header_row: list[str] | None = None
        expected: int | None = None
        cells = 0
        try:
            reader = csv.reader(
                io.StringIO(text, newline=""),
                delimiter=delimiter,
                quotechar=quote,
                doublequote=True,
                quoting=csv.QUOTE_MINIMAL,
                strict=True,
            )
            for row in reader:
                if not row:
                    raise ValueError("CSV blank records are not supported")
                if header and header_row is None:
                    header_row = row
                    expected = _validate_csv_header(row)
                    cells = len(row)
                    continue
                if len(parsed) >= MAX_CSV_ROWS:
                    raise ValueError(f"CSV rows exceed {MAX_CSV_ROWS}")
                if expected is None:
                    expected = len(row)
                    if expected == 0:
                        raise ValueError("CSV rows must contain at least one column")
                    if expected > MAX_CSV_COLUMNS:
                        raise ValueError(f"CSV columns exceed {MAX_CSV_COLUMNS}")
                _validate_csv_row(row, expected)
                cells += len(row)
                if cells > MAX_CSV_CELLS:
                    raise ValueError(f"CSV cells exceed {MAX_CSV_CELLS}")
                parsed.append(row)
        except csv.Error as exc:
            raise ValueError(f"invalid CSV: {exc}") from None
        return cls.outputs(header=header_row or [], rows=parsed)


class StringCsvEmit(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.string.csv_emit",
            display_name="Emit CSV Text",
            category="string/structured",
            inputs=(
                InputSpec("rows", LIST_LIST_STRING),
                InputSpec("header", LIST_STRING, required=False),
                InputSpec("delimiter", STRING, default=","),
                InputSpec("quote", STRING, default='"'),
            ),
            outputs=(OutputSpec("text", STRING),),
        )

    @classmethod
    def execute(
        cls,
        *,
        rows: list[list[str]],
        delimiter: str,
        quote: str,
        header: list[str] | None = None,
    ) -> Mapping[str, object]:
        _validate_csv_rows(rows, header)
        _validate_csv_dialect(delimiter, quote)
        output = _BoundedTextBuffer("CSV output")
        writer = csv.writer(
            output,
            delimiter=delimiter,
            quotechar=quote,
            doublequote=True,
            quoting=csv.QUOTE_MINIMAL,
            strict=True,
            lineterminator="\n",
        )
        if header:
            writer.writerow(header)
        writer.writerows(rows)
        return cls.outputs(text=output.getvalue())


STRING_OPERATION_NODES: tuple[type[Node], ...] = (
    StringFormat,
    StringTransform,
    StringTest,
    StringLength,
    StringRegex,
    StringJoin,
    StringSplit,
    StringJson,
    StringJsonEmit,
    StringCsvParse,
    StringCsvEmit,
)
