"""Curve value, codec and rendition contracts."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest
from dinkster_api.v1 import CURVE_TYPE, Curve, register_curve_type
from dinkster_values import TypeRegistry


def test_curve_is_frozen_validated_and_interpolates() -> None:
    curve = Curve(((-2, 10), (0, 0), (4, 8)), "linear")
    assert curve.points == ((-2.0, 10.0), (0.0, 0.0), (4.0, 8.0))
    assert curve.start == -2.0
    assert curve.end == 4.0
    assert curve.evaluate(-10) == 10.0
    assert curve.evaluate(-1) == 5.0
    assert curve.evaluate(2) == 4.0
    assert curve.evaluate(10) == 8.0
    assert Curve(((-1e308, -1e308), (1e308, 1e308))).evaluate(0.0) == 0.0
    high = 1e308
    high_end = math.nextafter(math.nextafter(high, math.inf), math.inf)
    high_middle = math.nextafter(high, math.inf)
    assert Curve(((high, 0), (high_end, 2))).evaluate(high_middle) == 1.0
    with pytest.raises(FrozenInstanceError):
        curve.points = ()  # type: ignore[misc]


def test_curve_defaults_and_source_forms_share_monotone_interpolation() -> None:
    registry = TypeRegistry()
    register_curve_type(registry)
    spec = registry.spec(CURVE_TYPE)
    # ComfyUI CurveInput.from_raw, F bc5f880d: points [(1,1),(0,0),(.25,.8)],
    # interpolation omitted, interp(.1). The native object form has identical meaning.
    expected = 0.3778232634051082
    points = [(1.0, 1.0), (0.0, 0.0), (0.25, 0.8)]
    for raw in (
        points,
        {"points": points},
        {"points": [{"position": x, "value": y} for x, y in points]},
        Curve(tuple(points)),
    ):
        curve = registry.wrap(CURVE_TYPE, raw).resolve()
        assert isinstance(curve, Curve)
        assert curve.interpolation == "monotone_cubic"
        assert curve.points == ((0.0, 0.0), (0.25, 0.8), (1.0, 1.0))
        assert curve.evaluate(0.1) == pytest.approx(expected, rel=0, abs=1e-15)
        assert spec.decode(spec.encode(curve)) == curve
    # ComfyUI CurveInput.from_raw, F bc5f880d: empty points, omitted mode,
    # interp(-1), interp(0), interp(.1), interp(2) all return exactly zero.
    for raw in ([], {"points": []}, Curve(())):
        empty = registry.wrap(CURVE_TYPE, raw).resolve()
        assert isinstance(empty, Curve)
        assert [empty.evaluate(x) for x in (-1.0, 0.0, 0.1, 2.0)] == [0.0] * 4
        assert spec.decode(spec.encode(empty)) == empty


def test_curve_duplicate_positions_preserve_stable_order_and_interpolation() -> None:
    # ComfyUI CurveInput.from_raw, F bc5f880d: these unsorted points with duplicate
    # x=.5, both explicit modes, interp(0), interp(.5), interp(.75), interp(1).
    points = ((1.0, 0.0), (0.5, 0.25), (0.0, 0.0), (0.5, 1.0))
    smooth = Curve(points)
    linear = Curve(points, "linear")
    assert smooth.points == linear.points == ((0.0, 0.0), (0.5, 0.25), (0.5, 1.0), (1.0, 0.0))
    assert [smooth.evaluate(x) for x in (0.0, 0.5, 0.75, 1.0)] == [0.0, 1.0, 0.625, 0.0]
    assert [linear.evaluate(x) for x in (0.0, 0.5, 0.75, 1.0)] == [0.0, 1.0, 0.5, 0.0]


def test_curve_interpolation_is_explicit_and_shape_preserving() -> None:
    with pytest.raises(ValueError, match="interpolation"):
        Curve(((0, 0),), "spline")  # type: ignore[arg-type]
    linear = Curve(((0, 0), (1, 2), (3, 3)), "linear")
    smooth = Curve(linear.points, "monotone_cubic")
    assert linear.evaluate(0.5) == 1.0
    assert smooth.evaluate(0.5) == pytest.approx(1.09375)
    samples = [smooth.evaluate(index / 20 * 3) for index in range(21)]
    assert samples == sorted(samples)
    assert min(samples) == 0.0 and max(samples) == 3.0
    assert Curve(((0, 1), (1, 1), (2, 3)), "monotone_cubic").evaluate(0.5) == 1.0
    assert Curve(((0, 1),), "monotone_cubic").evaluate(100) == 1.0
    assert Curve(((-1e308, 0), (1e308, 1)), "monotone_cubic").evaluate(0) == 0.5
    assert Curve(((0, -1e308), (1, 1e308)), "monotone_cubic").evaluate(0.5) == 0.0


@pytest.mark.parametrize(
    ("points", "position", "expected"),
    [
        (((0, 0), (1, 2), (3, 3)), 0.5, 1.09375),
        (((0, 0), (1, 2), (3, 3)), 2.0, 2.6875),
        (((0, 0), (1, 2), (2, 0)), 0.5, 1.25),
        (((0, 1), (1, 1), (2, 3)), 1.5, 1.75),
        (((-2, -4), (-1, -1), (2, 2)), 0.0, 0.4444444444444444),
    ],
)
def test_monotone_curve_matches_the_maintained_editor_reference(
    points: tuple[tuple[int, int], ...], position: float, expected: float
) -> None:
    assert Curve(points, "monotone_cubic").evaluate(position) == pytest.approx(expected)
