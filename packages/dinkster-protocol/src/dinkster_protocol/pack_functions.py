"""The contract of a button's ``Call`` step: what the editor asks and what the pack answers.

A press names the node type it was pressed on, the declared function, its
arguments already worked out by the editor, and either the fields the answer
may write or the name of the file it delivers. The host refuses a function the
node type does not declare in the current pack.

Numbers keep their kind across JSON: a whole number outside JavaScript's safe
range travels as ``{"$int": "digits"}`` and a decimal number with no fraction
as ``{"$float": n}``, so an integer field and a decimal field never trade
values by accident.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

from dinkster_schema import NodeSchema
from dinkster_schema.body import Call, declared_calls

PACK_FUNCTION_PATH: Final = "/api/node-functions/call"
PACK_FUNCTION_REQUEST_MAX_BYTES: Final = 1024 * 1024
PACK_FUNCTION_ANSWER_MAX_BYTES: Final = 1024 * 1024
PACK_FUNCTION_FILE_MAX_BYTES: Final = 64 * 1024 * 1024
PACK_FUNCTION_NOTE_MAX_CHARS: Final = 2000
PACK_FUNCTION_MESSAGE_MAX_CHARS: Final = 500
PACK_FUNCTION_PROGRESS_INTERVAL: Final = 0.1
"""Seconds between progress lines sent to the presser; the last one always goes."""

JSON_SAFE_INTEGER_MAX: Final = 2**53 - 1

EXTRA_KEY_MESSAGE: Final = "The pack answered something this button may not set"
PAUSED_MESSAGE: Final = "Paused. Try again in a moment."
TIMEOUT_MESSAGE: Final = "The pack took too long to answer."
FAILED_MESSAGE: Final = "The pack could not do this."
FILE_TOO_LARGE_MESSAGE: Final = "The file is larger than 64 MiB, so it was not delivered."


class PackFunctionRefused(Exception):
    """A call the host refuses or a pack answer it cannot use.

    ``code`` is stable for the editor; ``message`` is shown at the button;
    ``detail`` is for the pack author and goes to the log, never to the editor."""

    def __init__(self, code: str, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail


def value_to_wire(value: object) -> object:
    """A pack value as JSON that keeps whole and decimal numbers apart."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return (
            value
            if -JSON_SAFE_INTEGER_MAX <= value <= JSON_SAFE_INTEGER_MAX
            else {"$int": str(value)}
        )
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("a number must be finite")
        return (
            {"$float": int(value)}
            if value.is_integer() and abs(value) <= JSON_SAFE_INTEGER_MAX
            else value
        )
    if isinstance(value, (list, tuple)):
        return [value_to_wire(item) for item in cast("Sequence[object]", value)]
    if isinstance(value, Mapping):
        items = cast("Mapping[object, object]", value)
        if any(not isinstance(key, str) for key in items):
            raise ValueError("a mapping's keys must be text")
        if set(items) in ({"$int"}, {"$float"}):
            raise ValueError("a mapping cannot have only the key $int or $float")
        return {cast("str", key): value_to_wire(item) for key, item in items.items()}
    raise ValueError(f"{type(value).__name__} is not text, a number, a bool, a list or a mapping")


def value_from_wire(value: object) -> object:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("a number must be finite")
        return value
    if isinstance(value, list):
        return [value_from_wire(item) for item in cast("list[object]", value)]
    if isinstance(value, dict):
        data = cast("dict[str, object]", value)
        if set(data) == {"$int"}:
            digits = data["$int"]
            if not isinstance(digits, str) or not digits.lstrip("-").isdigit():
                raise ValueError("$int holds decimal digits")
            return int(digits)
        if set(data) == {"$float"}:
            number = data["$float"]
            if type(number) not in (int, float) or not math.isfinite(cast("float", number)):
                raise ValueError("$float holds a finite number")
            return float(cast("float", number))
        return {key: value_from_wire(item) for key, item in data.items()}
    raise ValueError("not a JSON value")


@dataclass(frozen=True)
class PackFunctionRequest:
    """One press's ``Call``: the node type it was declared on and what the editor worked out."""

    node_type: str
    function: str
    args: tuple[object, ...]
    writes: tuple[str, ...] = ()
    download: str | None = None

    def __post_init__(self) -> None:
        if not self.node_type or not self.function:
            raise ValueError("a call names its node type and function")
        if (self.download is None) == (not self.writes):
            raise ValueError("a call either writes fields or downloads a file")

    def to_wire(self) -> dict[str, object]:
        wire: dict[str, object] = {
            "nodeType": self.node_type,
            "function": self.function,
            "args": [value_to_wire(arg) for arg in self.args],
        }
        if self.download is not None:
            wire["download"] = self.download
        else:
            wire["writes"] = list(self.writes)
        return wire

    @classmethod
    def from_wire(cls, raw: object) -> PackFunctionRequest:
        if not isinstance(raw, dict):
            raise ValueError("a call is an object")
        data = cast("dict[str, object]", raw)
        if not set(data) <= {"nodeType", "function", "args", "writes", "download"}:
            raise ValueError("a call has unknown fields")
        node_type, function = data.get("nodeType"), data.get("function")
        args, writes = data.get("args", []), data.get("writes", [])
        download = data.get("download")
        if not isinstance(node_type, str) or not isinstance(function, str):
            raise ValueError("a call names its node type and function")
        if not isinstance(args, list) or not isinstance(writes, list):
            raise ValueError("a call's args and writes are lists")
        if any(not isinstance(item, str) for item in cast("list[object]", writes)):
            raise ValueError("a call writes field ids")
        if download is not None and not isinstance(download, str):
            raise ValueError("a call's download is a file name")
        return cls(
            node_type,
            function,
            tuple(value_from_wire(arg) for arg in cast("list[object]", args)),
            tuple(cast("list[str]", writes)),
            download,
        )


def declared_call(schema: NodeSchema | None, request: PackFunctionRequest) -> Call | None:
    """The node type's ``Call`` step this request stands for: same function, same writes or file."""
    if schema is None:
        return None
    return next(
        (
            call
            for call in declared_calls(schema.body, schema.menu)
            if call.name == request.function
            and call.writes == request.writes
            and call.download == request.download
        ),
        None,
    )


@dataclass(frozen=True)
class PackFunctionAnswer:
    """What a call gave back: the fields it sets (as wire JSON) or a file, and its notes."""

    fields: Mapping[str, object] | None
    file: bytes | None
    notes: tuple[str, ...] = ()

    @classmethod
    def from_reply(
        cls, reply: Mapping[str, object], blobs: Sequence[bytes], request: PackFunctionRequest
    ) -> PackFunctionAnswer:
        """Read a worker's ``packFunctionResult``; a refusal raises ``PackFunctionRefused``."""
        code, message = reply.get("code"), reply.get("message")
        if code is not None:
            if not isinstance(code, str) or not isinstance(message, str):
                raise PackFunctionRefused("failed", FAILED_MESSAGE, "malformed refusal")
            raise PackFunctionRefused(code, message)
        raw_notes = reply.get("notes", [])
        if not isinstance(raw_notes, list) or any(
            not isinstance(note, str) for note in cast("list[object]", raw_notes)
        ):
            raise PackFunctionRefused("failed", FAILED_MESSAGE, "malformed notes")
        notes = tuple(cast("list[str]", raw_notes))
        if request.download is not None:
            if len(blobs) != 1:
                raise PackFunctionRefused("failed", FAILED_MESSAGE, "the file is missing")
            return cls(None, check_file(blobs[0]), notes)
        fields = reply.get("answer")
        if not isinstance(fields, dict) or any(
            key not in request.writes for key in cast("dict[str, object]", fields)
        ):
            raise PackFunctionRefused("failed", FAILED_MESSAGE, "malformed answer")
        return cls(cast("dict[str, object]", fields), None, notes)


def check_answer(answer: object, writes: Sequence[str]) -> dict[str, object]:
    """The answer of a call that writes fields, as wire JSON.

    The keys must be some of ``writes``: one key more refuses the whole answer,
    a key left out keeps that field as it is."""
    if not isinstance(answer, Mapping):
        raise PackFunctionRefused(
            "wrong-shape", FAILED_MESSAGE, f"answered {type(answer).__name__}, not a mapping"
        )
    items = cast("Mapping[object, object]", answer)
    extra = sorted(repr(key) for key in items if key not in writes)
    if extra:
        raise PackFunctionRefused(
            "extra-key",
            EXTRA_KEY_MESSAGE,
            f"answered {', '.join(extra)}, which the button's writes do not name",
        )
    try:
        wire = {cast("str", key): value_to_wire(value) for key, value in items.items()}
    except ValueError as exc:
        raise PackFunctionRefused("wrong-shape", FAILED_MESSAGE, str(exc)) from None
    size = len(json.dumps(wire, separators=(",", ":")).encode())
    if size > PACK_FUNCTION_ANSWER_MAX_BYTES:
        raise PackFunctionRefused(
            "too-large",
            FAILED_MESSAGE,
            f"the answer is {size} bytes; at most {PACK_FUNCTION_ANSWER_MAX_BYTES}",
        )
    return wire


def check_file(answer: object) -> bytes:
    """The bytes of a call that delivers a file."""
    if isinstance(answer, str):
        answer = answer.encode()
    if not isinstance(answer, (bytes, bytearray)):
        raise PackFunctionRefused(
            "wrong-shape", FAILED_MESSAGE, f"answered {type(answer).__name__}, not bytes or text"
        )
    data = bytes(answer)
    if len(data) > PACK_FUNCTION_FILE_MAX_BYTES:
        raise PackFunctionRefused(
            "too-large", FILE_TOO_LARGE_MESSAGE, f"the file is {len(data)} bytes"
        )
    return data


def user_message(exc: BaseException) -> str:
    """The first line of what a pack function raised, shortened, for the button."""
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    if not text:
        return FAILED_MESSAGE
    if len(text) > PACK_FUNCTION_MESSAGE_MAX_CHARS:
        text = text[: PACK_FUNCTION_MESSAGE_MAX_CHARS - 3] + "..."
    return text
