"""Typed scalar conversion and piecewise-linear schedule operations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import cast

from dinkster_api.v1 import (
    ABSENT,
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    ComboWidget,
    CurveWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    report_event,
)

from .expression import MAX_INTEGER_BITS
from .types import CURVE_TYPE, MAX_CURVE_POINTS, Curve

INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
STRING = TypeExpr.concrete(CORE_STRING)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
SCALAR = TypeExpr.union(CORE_INT, CORE_FLOAT, CORE_STRING, CORE_BOOLEAN)
LIST_FLOAT = TypeExpr.list_of(FLOAT)
LIST_INT = TypeExpr.list_of(INT)
CURVE = TypeExpr.concrete(CURVE_TYPE)

MAX_CONVERSION_TEXT_BYTES = 1 << 20
MAX_SCHEDULE_TEXT_BYTES = 1 << 16


def _finite_float(value: object, subject: str) -> float:
    if type(value) not in (int, float, str):
        raise ValueError(f"{subject} is not a finite number")
    try:
        result = float(cast("int | float | str", value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{subject} is not a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{subject} is not a finite number")
    return result


def _text(value: str) -> str:
    if len(value.encode("utf-8")) > MAX_CONVERSION_TEXT_BYTES:
        raise ValueError(f"conversion text exceeds {MAX_CONVERSION_TEXT_BYTES} UTF-8 bytes")
    return value.strip()


def _lossy(target: str) -> ValueError:
    return ValueError(f"conversion to {target} would be lossy; enable force_lossy to continue")


def _bounded_int(value: int) -> int:
    if value.bit_length() > MAX_INTEGER_BITS:
        raise ValueError(f"integer magnitude exceeds the limit of {MAX_INTEGER_BITS} bits")
    return value


def _to_int(value: object, force_lossy: bool) -> int:
    if type(value) is bool:
        return 1 if value else 0
    if type(value) is int:
        return _bounded_int(value)
    if type(value) is float:
        number = _finite_float(value, "float input")
        result = int(number)
        if not force_lossy and result != number:
            raise _lossy("int")
        return _bounded_int(result)
    if type(value) is str:
        text = _text(value)
        if not text:
            raise ValueError("cannot convert empty text to int")
        if force_lossy:
            number = _finite_float(text, "text input")
            try:
                return _bounded_int(int(text))
            except ValueError:
                return _bounded_int(int(number))
        try:
            decimal = Decimal(text)
        except InvalidOperation as exc:
            raise ValueError(f"cannot convert text to int: {value!r}") from exc
        if not decimal.is_finite():
            raise ValueError("text input is not a finite number")
        if decimal != decimal.to_integral_value():
            raise _lossy("int")
        if decimal.adjusted() > MAX_INTEGER_BITS:
            raise ValueError(f"integer magnitude exceeds the limit of {MAX_INTEGER_BITS} bits")
        return _bounded_int(int(decimal))
    raise TypeError(f"unsupported conversion input type {type(value).__name__}")


def _to_float(value: object, force_lossy: bool) -> float:
    if type(value) is bool:
        return 1.0 if value else 0.0
    if type(value) is int:
        result = _finite_float(value, "int input")
        if not force_lossy and int(result) != value:
            raise _lossy("float")
        return result
    if type(value) is float:
        return _finite_float(value, "float input")
    if type(value) is str:
        text = _text(value)
        if not text:
            raise ValueError("cannot convert empty text to float")
        return _finite_float(text, "text input")
    raise TypeError(f"unsupported conversion input type {type(value).__name__}")


def _to_string(value: object) -> str:
    if type(value) is float:
        _finite_float(value, "float input")
    if type(value) not in (bool, int, float, str):
        raise TypeError(f"unsupported conversion input type {type(value).__name__}")
    if type(value) is str:
        if len(value.encode("utf-8")) > MAX_CONVERSION_TEXT_BYTES:
            raise ValueError(f"conversion text exceeds {MAX_CONVERSION_TEXT_BYTES} UTF-8 bytes")
        return value
    try:
        result = str(value)
    except ValueError as exc:
        raise ValueError("conversion result exceeds Python's integer text limit") from exc
    if len(result.encode("utf-8")) > MAX_CONVERSION_TEXT_BYTES:
        raise ValueError(f"conversion text exceeds {MAX_CONVERSION_TEXT_BYTES} UTF-8 bytes")
    return result


def _to_boolean(value: object, force_lossy: bool) -> bool:
    if type(value) is bool:
        return value
    if type(value) is int:
        if not force_lossy and value not in (0, 1):
            raise _lossy("boolean")
        return bool(value)
    if type(value) is float:
        number = _finite_float(value, "float input")
        if not force_lossy and number not in (0.0, 1.0):
            raise _lossy("boolean")
        return bool(number)
    if type(value) is str:
        text = _text(value).casefold()
        if text in ("true", "1", "yes", "on"):
            return True
        if text in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"cannot convert text to boolean: {value!r}")
    raise TypeError(f"unsupported conversion input type {type(value).__name__}")


class ValueConvert(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.value.convert",
            display_name="Convert Value",
            category="utilities/conversion",
            description="Converts one scalar to a selected type or to both numeric outputs.",
            inputs=(
                InputSpec("value", SCALAR, force_input=True),
                InputSpec(
                    "target",
                    TypeExpr.concrete("core.combo"),
                    default="float",
                    widget=ComboWidget(options=("int", "float", "number", "string", "boolean")),
                ),
                InputSpec("force_lossy", BOOLEAN, default=False, advanced=True),
            ),
            outputs=(
                OutputSpec("int", INT, optional=True),
                OutputSpec("float", FLOAT, optional=True),
                OutputSpec("string", STRING, optional=True),
                OutputSpec("boolean", BOOLEAN, optional=True),
            ),
        )

    @classmethod
    def execute(cls, *, value: object, target: str, force_lossy: bool) -> Mapping[str, object]:
        if target == "number":
            float_result = _to_float(value, force_lossy)
            int_result = _to_int(value, force_lossy)
            return cls.outputs(
                int=int_result,
                float=float_result,
                string=ABSENT,
                boolean=ABSENT,
            )
        if target == "int":
            result = _to_int(value, force_lossy)
        elif target == "float":
            result = _to_float(value, force_lossy)
        elif target == "string":
            result = _to_string(value)
        elif target == "boolean":
            result = _to_boolean(value, force_lossy)
        else:
            raise ValueError(f"unknown conversion target {target!r}")
        outputs: dict[str, object] = {
            "int": ABSENT,
            "float": ABSENT,
            "string": ABSENT,
            "boolean": ABSENT,
        }
        outputs[target] = result
        return cls.outputs(**outputs)


def _curve_from_values(values: Sequence[float], start: float, step: float) -> Curve:
    if isinstance(values, (str, bytes)):
        raise TypeError("curve values must be a sequence of numbers")
    if not 1 <= len(values) <= MAX_CURVE_POINTS:
        raise ValueError(f"curve values must contain between 1 and {MAX_CURVE_POINTS} items")
    start = _finite_float(start, "curve start")
    step = _finite_float(step, "curve step")
    if step <= 0:
        raise ValueError("curve step must be greater than zero")
    points: list[tuple[float, float]] = []
    for index, value in enumerate(values):
        position = start + step * index
        if not math.isfinite(position):
            raise ValueError("curve positions must be finite")
        points.append((position, _finite_float(value, f"curve value {index}")))
    return Curve(tuple(points))


class CurveFromValues(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.curve.from_values",
            display_name="Curve from Values",
            category="utilities/curve",
            inputs=(
                InputSpec("values", LIST_FLOAT, force_input=True),
                InputSpec("start", FLOAT, default=0.0),
                InputSpec("step", FLOAT, default=1.0, widget=NumberWidget(min=0.0, step=0.1)),
            ),
            outputs=(OutputSpec("curve", CURVE),),
        )

    @classmethod
    def execute(cls, *, values: Sequence[float], start: float, step: float) -> Mapping[str, object]:
        return cls.outputs(curve=_curve_from_values(values, start, step))


class CurveEvaluate(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.curve.evaluate",
            display_name="Evaluate Curve",
            category="utilities/curve",
            inputs=(InputSpec("curve", CURVE), InputSpec("position", FLOAT, default=0.0)),
            outputs=(OutputSpec("value", FLOAT),),
        )

    @classmethod
    def execute(cls, *, curve: Curve, position: float) -> Mapping[str, object]:
        return cls.outputs(value=curve.evaluate(position))


class CurveSample(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.curve.sample",
            display_name="Sample Curve",
            category="utilities/curve",
            inputs=(
                InputSpec("curve", CURVE),
                InputSpec(
                    "count",
                    INT,
                    default=10,
                    widget=NumberWidget(min=1, max=MAX_CURVE_POINTS, step=1),
                ),
            ),
            outputs=(OutputSpec("values", LIST_FLOAT),),
        )

    @classmethod
    def execute(cls, *, curve: Curve, count: int) -> Mapping[str, object]:
        if type(count) is not int or not 1 <= count <= MAX_CURVE_POINTS:
            raise ValueError(f"sample count must be between 1 and {MAX_CURVE_POINTS}")
        if count == 1:
            positions = (curve.start,)
        else:
            positions = tuple(
                (1.0 - index / (count - 1)) * curve.start + (index / (count - 1)) * curve.end
                for index in range(count)
            )
        return cls.outputs(values=[curve.evaluate(position) for position in positions])


CURVE_EDITOR_DEFAULT = Curve(((0.0, 0.0), (1.0, 1.0)), "monotone_cubic")


class CurveEditor(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.curve.editor",
            display_name="Curve Editor",
            category="utilities/curve",
            inputs=(
                InputSpec(
                    "curve", CURVE, default=CURVE_EDITOR_DEFAULT.to_record(), widget=CurveWidget()
                ),
                InputSpec("histogram", LIST_INT, required=False),
            ),
            outputs=(OutputSpec("curve", CURVE),),
            emits_previews=True,
        )

    @classmethod
    def execute(
        cls, *, curve: Curve, histogram: Sequence[int] | None = None
    ) -> Mapping[str, object]:
        if histogram is not None:
            if len(histogram) != 256 or any(
                type(value) is not int or value < 0 for value in histogram
            ):
                raise ValueError("curve histogram must contain exactly 256 nonnegative integers")
            report_event("dinkster.curve.histogram", {"histogram": list(histogram)})
        return cls.outputs(curve=curve)


def parse_schedule(text: str) -> Curve:
    if len(text.encode("utf-8")) > MAX_SCHEDULE_TEXT_BYTES:
        raise ValueError(f"schedule text exceeds {MAX_SCHEDULE_TEXT_BYTES} UTF-8 bytes")
    entries = [entry.strip() for entry in text.split(",")]
    if entries and not entries[-1]:
        entries.pop()
    if not entries or any(not entry for entry in entries):
        raise ValueError("schedule must contain comma-separated position:value entries")
    if len(entries) > MAX_CURVE_POINTS:
        raise ValueError(f"schedule exceeds {MAX_CURVE_POINTS} points")
    points: list[tuple[float, float]] = []
    for index, entry in enumerate(entries):
        parts = entry.split(":")
        if len(parts) != 2:
            raise ValueError(f"schedule entry {index} must contain one ':' separator")
        position_text, value_text = (part.strip() for part in parts)
        if value_text.startswith("(") or value_text.endswith(")"):
            if not (value_text.startswith("(") and value_text.endswith(")")):
                raise ValueError(f"schedule entry {index} has unbalanced value parentheses")
            value_text = value_text[1:-1].strip()
        if not position_text or not value_text:
            raise ValueError(f"schedule entry {index} requires a position and value")
        points.append(
            (
                _finite_float(position_text, f"schedule position {index}"),
                _finite_float(value_text, f"schedule value {index}"),
            )
        )
    points.sort(key=lambda point: point[0])
    return Curve(tuple(points))


class StringScheduleParse(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.string.schedule_parse",
            display_name="Parse Value Schedule",
            category="utilities/curve",
            inputs=(InputSpec("text", STRING, widget=StringWidget(multiline=True)),),
            outputs=(OutputSpec("curve", CURVE),),
        )

    @classmethod
    def execute(cls, *, text: str) -> Mapping[str, object]:
        return cls.outputs(curve=parse_schedule(text))


CONVERSION_NODES: tuple[type[Node], ...] = (
    ValueConvert,
    CurveFromValues,
    CurveEditor,
    CurveEvaluate,
    CurveSample,
    StringScheduleParse,
)


__all__ = [
    "CONVERSION_NODES",
    "CurveEvaluate",
    "CurveEditor",
    "CurveFromValues",
    "CurveSample",
    "StringScheduleParse",
    "ValueConvert",
    "parse_schedule",
]
