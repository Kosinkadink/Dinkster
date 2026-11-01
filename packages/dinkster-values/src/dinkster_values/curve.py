"""Shared immutable curve value and its canonical wire codec."""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from typing import Literal, cast

from .builtin_types import CURVE_TYPE
from .registry import RenditionUnavailable, TypeRegistry

MAX_CURVE_POINTS = 4096
_CURVE_POINTS_RENDITION = "curve-points"


def _finite_number(value: object, subject: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"curve {subject} must be a number")
    try:
        result = float(cast("int | float", value))
    except OverflowError as exc:
        raise ValueError(f"curve {subject} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"curve {subject} must be finite")
    return result


@dataclass(frozen=True)
class Curve:
    """An immutable, ordered float curve."""

    points: tuple[tuple[float, float], ...]
    interpolation: Literal["linear", "monotone_cubic"] = "monotone_cubic"
    _tangents: tuple[tuple[float, float], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        raw_points = cast("object", self.points)
        if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
            raise TypeError("curve points must be a sequence")
        points = cast("Sequence[object]", raw_points)
        normalized: list[tuple[float, float]] = []
        for index, raw_point in enumerate(points):
            if (
                not isinstance(raw_point, Sequence)
                or isinstance(raw_point, (str, bytes))
                or len(cast("Sequence[object]", raw_point)) != 2
            ):
                raise TypeError(f"curve point {index} must contain position and value")
            point = cast("Sequence[object]", raw_point)
            position = _finite_number(point[0], f"point {index} position")
            value = _finite_number(point[1], f"point {index} value")
            normalized.append((position, value))
        normalized.sort(key=lambda point: point[0])
        object.__setattr__(self, "points", tuple(normalized))
        if type(self.interpolation) is not str or self.interpolation not in {
            "linear",
            "monotone_cubic",
        }:
            raise ValueError("curve interpolation must be 'linear' or 'monotone_cubic'")
        object.__setattr__(
            self,
            "_tangents",
            self._monotone_tangents() if self.interpolation == "monotone_cubic" else (),
        )

    @property
    def start(self) -> float:
        return self.points[0][0] if self.points else 0.0

    @property
    def end(self) -> float:
        return self.points[-1][0] if self.points else 0.0

    def evaluate(self, position: float) -> float:
        position = _finite_number(position, "sample position")
        if not self.points:
            return 0.0
        if position < self.start or (
            self.interpolation == "monotone_cubic" and position == self.start
        ):
            return self.points[0][1]
        if position >= self.end:
            return self.points[-1][1]
        right = bisect_right(self.points, position, key=lambda point: point[0])
        right_position, right_value = self.points[right]
        left_position, left_value = self.points[right - 1]
        span = right_position - left_position
        if math.isfinite(span):
            amount = (position - left_position) / span
        else:
            scale = max(abs(left_position), abs(right_position), abs(position), 1.0)
            amount = (position / scale - left_position / scale) / (
                right_position / scale - left_position / scale
            )
        if self.interpolation == "monotone_cubic":
            amount2 = amount * amount
            amount3 = amount2 * amount
            left_tangent, right_tangent = self._tangents[right - 1]
            value_scale = max(abs(left_value), abs(right_value))
            if value_scale == 0:
                return 0.0
            scaled_left = left_value / value_scale
            scaled_right = right_value / value_scale
            scaled_delta = scaled_right - scaled_left
            scaled_value = (
                (2 * amount3 - 3 * amount2 + 1) * scaled_left
                + (amount3 - 2 * amount2 + amount) * left_tangent * scaled_delta
                + (-2 * amount3 + 3 * amount2) * scaled_right
                + (amount3 - amount2) * right_tangent * scaled_delta
            )
            value = scaled_value * value_scale
            if not math.isfinite(value):
                raise ValueError("curve interpolation produced a non-finite value")
            return value
        value = (1.0 - amount) * left_value + amount * right_value
        if not math.isfinite(value):
            raise ValueError("curve interpolation produced a non-finite value")
        return value

    def _monotone_tangents(self) -> tuple[tuple[float, float], ...]:
        if len(self.points) < 2:
            return ()
        with localcontext() as context:
            context.prec = 50
            points = tuple(
                (Decimal.from_float(position), Decimal.from_float(value))
                for position, value in self.points
            )
            secants = [
                (right[1] - left[1]) / (right[0] - left[0]) if right[0] != left[0] else Decimal(0)
                for left, right in zip(points, points[1:], strict=False)
            ]
            slopes = [secants[0]]
            slopes.extend(
                Decimal(0) if left * right <= 0 else (left + right) / 2
                for left, right in zip(secants, secants[1:], strict=False)
            )
            slopes.append(secants[-1])
            for index, secant in enumerate(secants):
                if secant == 0:
                    slopes[index] = slopes[index + 1] = Decimal(0)
                    continue
                alpha = slopes[index] / secant
                beta = slopes[index + 1] / secant
                magnitude = alpha * alpha + beta * beta
                if magnitude > 9:
                    scale = Decimal(3) / magnitude.sqrt()
                    slopes[index] = scale * alpha * secant
                    slopes[index + 1] = scale * beta * secant
            return tuple(
                (float(slopes[index] / secant), float(slopes[index + 1] / secant))
                if secant != 0
                else (0.0, 0.0)
                for index, secant in enumerate(secants)
            )

    def to_record(self) -> dict[str, object]:
        return {
            "interpolation": self.interpolation,
            "points": [{"position": position, "value": value} for position, value in self.points],
        }


def coerce_curve(obj: object) -> Curve:
    if isinstance(obj, Curve):
        return obj
    interpolation: Literal["linear", "monotone_cubic"] = "monotone_cubic"
    if isinstance(obj, Mapping):
        record = cast("Mapping[str, object]", obj)
        if set(record) not in ({"points"}, {"points", "interpolation"}):
            raise ValueError(f"{CURVE_TYPE} requires points and optional interpolation fields")
        raw_points = record["points"]
        interpolation = cast(
            "Literal['linear', 'monotone_cubic']", record.get("interpolation", "monotone_cubic")
        )
    else:
        raw_points = obj
    if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
        raise TypeError(f"{CURVE_TYPE} points must be an array")
    points: list[tuple[float, float]] = []
    for index, raw_point in enumerate(cast("Sequence[object]", raw_points)):
        if isinstance(raw_point, Mapping):
            point = cast("Mapping[str, object]", raw_point)
            if set(point) != {"position", "value"}:
                raise ValueError(f"{CURVE_TYPE} point {index} requires exactly position and value")
            points.append(
                (
                    cast("float", point["position"]),
                    cast("float", point["value"]),
                )
            )
        elif isinstance(raw_point, Sequence) and not isinstance(raw_point, (str, bytes)):
            points.append(cast("tuple[float, float]", tuple(cast("Sequence[object]", raw_point))))
        else:
            raise TypeError(f"{CURVE_TYPE} point {index} must be an object or position/value pair")
    return Curve(tuple(points), interpolation)


def _encode_curve(obj: object) -> bytes:
    curve = coerce_curve(obj)
    return json.dumps(
        curve.to_record(),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _decode_curve(data: bytes) -> object:
    try:
        record = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {CURVE_TYPE} payload") from exc
    return coerce_curve(record)


def _validate_curve_encoded(data: bytes, _metadata: Mapping[str, object]) -> None:
    _decode_curve(data)


def _curve_meta(obj: object) -> Mapping[str, object]:
    curve = coerce_curve(obj)
    return {"pointCount": len(curve.points), "start": curve.start, "end": curve.end}


def _render_curve_points(obj: object) -> bytes:
    try:
        curve = coerce_curve(obj)
        if len(curve.points) > MAX_CURVE_POINTS:
            raise ValueError(f"curve preview exceeds {MAX_CURVE_POINTS} points")
        return _encode_curve(curve)
    except (TypeError, ValueError) as error:
        raise RenditionUnavailable(str(error)) from error


def register_curve_type(registry: TypeRegistry) -> None:
    if CURVE_TYPE not in registry:
        registry.register(
            CURVE_TYPE,
            encode=_encode_curve,
            decode=_decode_curve,
            coerce=coerce_curve,
            meta=_curve_meta,
            validate_encoded=_validate_curve_encoded,
            asset_references_complete=True,
        )
        registry.register_rendition(
            CURVE_TYPE,
            _CURVE_POINTS_RENDITION,
            mime="application/json",
            render=_render_curve_points,
            limits={"points": MAX_CURVE_POINTS},
        )


__all__ = [
    "CURVE_TYPE",
    "MAX_CURVE_POINTS",
    "Curve",
    "coerce_curve",
    "register_curve_type",
]
