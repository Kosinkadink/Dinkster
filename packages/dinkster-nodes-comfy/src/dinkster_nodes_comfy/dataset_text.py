"""Dataset text nodes with the fixed core reference's list wrapping."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    CORE_INT,
    CORE_STRING,
    Deprecation,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
)

_STRING = TypeExpr.concrete(CORE_STRING)


def _parameter(value: Any) -> Any:
    if isinstance(value, list):
        items = cast(list[Any], value)
        return items[0] if len(items) == 1 else items
    return value


def _schema(
    node_type: str,
    display_name: str,
    description: str,
    *,
    extra_inputs: tuple[InputSpec, ...] = (),
    search_terms: tuple[str, ...] = (),
    deprecated: bool = True,
) -> NodeSchema:
    return NodeSchema(
        node_type=node_type,
        display_name=display_name,
        category="text",
        description=description,
        search_terms=search_terms,
        deprecation=Deprecation("Deprecated in ComfyUI.") if deprecated else None,
        search_visibility="deprecated" if deprecated else "normal",
        inputs=(
            InputSpec(
                "texts",
                _STRING,
                doc="Text to process.",
                widget=StringWidget(multiline=False),
            ),
            *extra_inputs,
        ),
        outputs=(OutputSpec("texts", _STRING, doc="Processed texts"),),
    )


class TextToLowercase(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "TextToLowercase",
            "Convert Text to Lowercase (DEPRECATED)",
            "Convert text to lowercase.",
            search_terms=("lowercase",),
        )

    @classmethod
    def execute(cls, *, texts: str) -> Mapping[str, object]:
        return cls.outputs(texts=[texts.lower()])


class TextToUppercase(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "TextToUppercase",
            "Convert Text to Uppercase (DEPRECATED)",
            "Convert text to uppercase.",
            search_terms=("uppercase",),
        )

    @classmethod
    def execute(cls, *, texts: str) -> Mapping[str, object]:
        return cls.outputs(texts=[texts.upper()])


class TruncateText(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "TruncateText",
            "Truncate Text",
            "Truncate text to a maximum length.",
            extra_inputs=(
                InputSpec(
                    "max_length",
                    TypeExpr.concrete(CORE_INT),
                    default=77,
                    doc="Maximum text length.",
                    widget=NumberWidget(min=1, max=10000),
                ),
            ),
            search_terms=("truncate", "cut", "shorten"),
            deprecated=False,
        )

    @classmethod
    def execute(cls, *, texts: str, max_length: Any) -> Mapping[str, object]:
        return cls.outputs(texts=[texts[: _parameter(max_length)]])


class AddTextPrefix(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "AddTextPrefix",
            "Add Text Prefix (DEPRECATED)",
            "Add a prefix to all texts.",
            extra_inputs=(
                InputSpec(
                    "prefix",
                    _STRING,
                    default="",
                    doc="Prefix to add.",
                    widget=StringWidget(multiline=False),
                ),
            ),
        )

    @classmethod
    def execute(cls, *, texts: str, prefix: Any) -> Mapping[str, object]:
        return cls.outputs(texts=[_parameter(prefix) + texts])


class AddTextSuffix(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "AddTextSuffix",
            "Add Text Suffix (DEPRECATED)",
            "Add a suffix to all texts.",
            extra_inputs=(
                InputSpec(
                    "suffix",
                    _STRING,
                    default="",
                    doc="Suffix to add.",
                    widget=StringWidget(multiline=False),
                ),
            ),
        )

    @classmethod
    def execute(cls, *, texts: str, suffix: Any) -> Mapping[str, object]:
        return cls.outputs(texts=[texts + _parameter(suffix)])


class ReplaceText(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ReplaceText",
            "Replace Text (DEPRECATED)",
            "Replace text in all texts.",
            extra_inputs=(
                InputSpec(
                    "find",
                    _STRING,
                    default="",
                    doc="Text to find.",
                    widget=StringWidget(multiline=False),
                ),
                InputSpec(
                    "replace",
                    _STRING,
                    default="",
                    doc="Text to replace with.",
                    widget=StringWidget(multiline=False),
                ),
            ),
        )

    @classmethod
    def execute(cls, *, texts: str, find: Any, replace: Any) -> Mapping[str, object]:
        return cls.outputs(texts=[texts.replace(_parameter(find), _parameter(replace))])


class StripWhitespace(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "StripWhitespace",
            "Strip Whitespace (DEPRECATED)",
            "Strip leading and trailing whitespace from all texts.",
        )

    @classmethod
    def execute(cls, *, texts: str) -> Mapping[str, object]:
        return cls.outputs(texts=[texts.strip()])


DATASET_TEXT_NODES = (
    TextToLowercase,
    TextToUppercase,
    TruncateText,
    AddTextPrefix,
    AddTextSuffix,
    ReplaceText,
    StripWhitespace,
)
