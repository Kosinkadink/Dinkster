"""Bounded contracts for calling the lookups a node declares.

A lookup is a pure ``@lookup`` function in a pack. The editor asks the host
for its answer with the node type, the function's wire name and the typed
arguments; the host finds the function among the declarations the pack
loaded (never by importing a name it was sent), checks the arguments and the
answer against the function's annotations, and keeps every call inside one
time limit and one answer size.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
import types
import typing
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

from dinkster_schema import ComboOption, Lookup, NodeSchema
from dinkster_schema.body import ASSET_REF_CLASS, LookupMark, function_name, lookup_mark

LOOKUPS_PATH: Final = "/api/lookups"
LOOKUP_TIME_LIMIT: Final = 10.0
"""Seconds one call may take once the pack's worker is running."""
LOOKUP_ANSWER_MAX_BYTES: Final = 1024 * 1024
LOOKUP_REQUEST_MAX_BYTES: Final = 64 * 1024
LOOKUP_CALLS_UNDER_WAY: Final = 8
"""Calls one user may have under way on one backend at once."""
LOOKUP_CALLS_PER_MINUTE: Final = 60
"""New calls one user may start on one backend in any 60 seconds."""
LOOKUP_SNAPSHOT_HEADER: Final = "X-Dinkster-Lookup-Snapshot"

JSON_SAFE_INTEGER_MAX: Final = 2**53 - 1

LOOKUP_ERRORS: Final = {
    "lookup-unknown": 404,
    "lookup-invalid-arguments": 400,
    "lookup-file-unchecked": 409,
    "lookup-paused": 429,
    "lookup-wrong-shape": 502,
    "lookup-failed": 502,
    "lookup-timeout": 504,
}
"""Each answer the editor can tell apart, with its HTTP status. The message a
user sees is generic; the author's detail goes to the host log."""


class LookupCallError(Exception):
    """A call that ended without an answer, with the code the editor reads."""

    def __init__(self, code: str, detail: str = "") -> None:
        if code not in LOOKUP_ERRORS:
            raise ValueError(f"unknown lookup error code {code!r}")
        super().__init__(detail or code)
        self.code = code
        self.detail = detail


# --- finding declared lookups ---------------------------------------------------


def declared_lookups(value: object) -> Iterator[Lookup]:
    """Every ``Lookup`` a node schema (or any part of one) declares."""
    if isinstance(value, Lookup):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in cast("Sequence[object]", value):
            yield from declared_lookups(item)
    elif isinstance(value, Mapping):
        for item in cast("Mapping[object, object]", value).values():
            yield from declared_lookups(item)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for item in dataclasses.fields(value):
            yield from declared_lookups(getattr(value, item.name))


def lookup_names(schema: NodeSchema) -> frozenset[str]:
    """The wire names of the lookups one node type declares."""
    return frozenset(item.name for item in declared_lookups(schema))


def lookup_functions(schemas: Mapping[str, NodeSchema]) -> dict[str, Callable[..., object]]:
    """The marked functions a pack's loaded schemas name, by wire name.

    Only a function a schema declares can be called: a name the editor sends
    is looked up here and never imported."""
    found: dict[str, Callable[..., object]] = {}
    for schema in schemas.values():
        for item in declared_lookups(schema):
            function = item.function
            if isinstance(function, str) or lookup_mark(function) is None:
                continue
            found.setdefault(item.name, function)
    return found


def lookup_marks(functions: Mapping[str, Callable[..., object]]) -> dict[str, LookupMark]:
    return {name: lookup_mark(function) or LookupMark() for name, function in functions.items()}


def lookup_marks_to_wire(marks: Mapping[str, LookupMark]) -> dict[str, dict[str, object]]:
    return {
        name: {"watches": list(mark.watches), "afterRun": mark.after_run}
        for name, mark in sorted(marks.items())
    }


def lookup_marks_from_wire(raw: object) -> dict[str, LookupMark]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("lookup marks must be an object")
    marks: dict[str, LookupMark] = {}
    for name, item in cast("Mapping[object, object]", raw).items():
        if type(name) is not str or not isinstance(item, Mapping):
            raise ValueError("lookup marks are objects by function name")
        data = cast("Mapping[str, object]", item)
        watches = data.get("watches", [])
        after_run = data.get("afterRun", False)
        if (
            not isinstance(watches, list)
            or not all(type(kind) is str and kind for kind in cast("list[object]", watches))
            or type(after_run) is not bool
        ):
            raise ValueError(f"lookup marks of {name!r} are malformed")
        marks[name] = LookupMark(tuple(cast("list[str]", watches)), after_run)
    return marks


# --- the request ---------------------------------------------------------------


@dataclass(frozen=True)
class LookupRequest:
    node_type: str
    function: str
    args: tuple[object, ...]

    def key(self) -> str:
        """Equal requests have equal keys: the typed arguments, canonically encoded."""
        return json.dumps(
            [self.function, list(self.args)],
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def _plain_value(value: object, *, top: bool) -> bool:
    if type(value) in (bool, str):
        return True
    if type(value) is int:
        return True
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, list):
        return top and all(_plain_value(item, top=False) for item in cast("list[object]", value))
    if isinstance(value, dict):
        # An asset argument travels as the asset's wire object.
        return top and all(type(key) is str for key in cast("dict[object, object]", value))
    return False


def lookup_request_from_wire(raw: object) -> LookupRequest:
    if not isinstance(raw, Mapping):
        raise ValueError("a lookup request is a JSON object")
    data = cast("Mapping[str, object]", raw)
    if set(data) != {"nodeType", "function", "args"}:
        raise ValueError("a lookup request has nodeType, function and args")
    node_type, function, args = data["nodeType"], data["function"], data["args"]
    if type(node_type) is not str or not node_type or type(function) is not str or not function:
        raise ValueError("nodeType and function are non-empty strings")
    if not isinstance(args, list) or not all(
        _plain_value(item, top=True) for item in cast("list[object]", args)
    ):
        raise ValueError("args are values: no null, no nested lists, finite numbers")
    return LookupRequest(node_type, function, tuple(cast("list[object]", args)))


# --- arguments and answers against the annotations -------------------------------


def _is_asset_ref(annotation: object) -> bool:
    return (
        isinstance(annotation, type)
        and (annotation.__module__, annotation.__qualname__) == ASSET_REF_CLASS
    )


def _parameters(function: Callable[..., object]) -> tuple[list[object], int]:
    """Positional parameter annotations, and how many of them have no default."""
    signature = inspect.signature(function, eval_str=True)
    annotations: list[object] = []
    required = 0
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            continue
        annotations.append(parameter.annotation)
        if parameter.default is inspect.Parameter.empty:
            required += 1
    return annotations, required


def _argument(value: object, annotation: object, asset: Callable[[object], object]) -> object:
    if _is_asset_ref(annotation):
        if not isinstance(value, dict):
            raise ValueError("an asset argument is an asset")
        return asset(cast("dict[str, object]", value))
    if annotation is bool:
        if type(value) is not bool:
            raise ValueError("expected a bool")
        return value
    if annotation is int:
        if type(value) is not int:
            raise ValueError("expected a whole number")
        return value
    if annotation is float:
        if type(value) not in (int, float):
            raise ValueError("expected a number")
        return float(cast("float", value))
    if annotation is str:
        if type(value) is not str:
            raise ValueError("expected text")
        return value
    if typing.get_origin(annotation) is list:
        (item,) = typing.get_args(annotation)
        if not isinstance(value, list):
            raise ValueError("expected a list")
        return [_argument(entry, item, asset) for entry in cast("list[object]", value)]
    raise ValueError(f"parameter annotation {annotation!r} is not supported")


def lookup_arguments(
    function: Callable[..., object],
    args: Sequence[object],
    *,
    asset: Callable[[object], object],
) -> list[object]:
    """The Python values a lookup is called with, checked against its parameters.

    ``asset`` turns an asset's wire object into the ``AssetRef`` the function
    receives, or raises ``LookupCallError("lookup-file-unchecked")``."""
    annotations, required = _parameters(function)
    if not required <= len(args) <= len(annotations):
        raise LookupCallError(
            "lookup-invalid-arguments",
            f"{function_name(function)} takes {required} to {len(annotations)} "
            f"arguments, given {len(args)}",
        )
    values: list[object] = []
    for position, (value, annotation) in enumerate(zip(args, annotations, strict=False)):
        try:
            values.append(_argument(value, annotation, asset))
        except ValueError as exc:
            raise LookupCallError(
                "lookup-invalid-arguments",
                f"{function_name(function)} argument {position + 1}: {exc}",
            ) from None
    return values


def _fits_answer(value: object, annotation: object) -> bool:
    if annotation is None or annotation is type(None):
        return value is None
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        return any(_fits_answer(value, item) for item in typing.get_args(annotation))
    if origin is list:
        (item,) = typing.get_args(annotation)
        return isinstance(value, (list, tuple)) and all(
            _fits_answer(entry, item) for entry in cast("Sequence[object]", value)
        )
    if annotation is bool:
        return type(value) is bool
    if annotation is int:
        return type(value) is int and abs(value) <= JSON_SAFE_INTEGER_MAX
    if annotation is float:
        return (type(value) is int and abs(value) <= JSON_SAFE_INTEGER_MAX) or (
            type(value) is float and math.isfinite(value)
        )
    if annotation is str:
        return type(value) is str
    if annotation is ComboOption:
        return isinstance(value, ComboOption)
    return False


def _answer_to_wire(value: object) -> object:
    if isinstance(value, ComboOption):
        wire: dict[str, object] = {"value": value.value}
        for name in ("label", "info", "folder"):
            item = getattr(value, name)
            if item is not None:
                wire[name] = item
        return wire
    if isinstance(value, (list, tuple)):
        return [_answer_to_wire(item) for item in cast("Sequence[object]", value)]
    return value


def lookup_answer_to_wire(function: Callable[..., object], answer: object) -> object:
    """The answer as JSON, refused when it does not fit the function's return
    annotation or is larger than the answer bound."""
    expected = inspect.signature(function, eval_str=True).return_annotation
    if not _fits_answer(answer, expected):
        raise LookupCallError(
            "lookup-wrong-shape",
            f"{function_name(function)} answered {type(answer).__name__} "
            f"where its annotation says {expected!r}",
        )
    wire = _answer_to_wire(answer)
    try:
        encoded = json.dumps(wire, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (UnicodeEncodeError, ValueError):
        raise LookupCallError(
            "lookup-wrong-shape", f"{function_name(function)} answered text that is not valid"
        ) from None
    if len(encoded) > LOOKUP_ANSWER_MAX_BYTES:
        raise LookupCallError(
            "lookup-wrong-shape",
            f"{function_name(function)} answered {len(encoded)} bytes; the limit is "
            f"{LOOKUP_ANSWER_MAX_BYTES}",
        )
    return wire


def lookup_answer_size(wire: object) -> int:
    return len(json.dumps(wire, ensure_ascii=False, allow_nan=False).encode("utf-8"))
