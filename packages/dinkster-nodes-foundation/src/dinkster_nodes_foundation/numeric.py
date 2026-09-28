"""Typed comparison, selection, logic, range, and random value nodes."""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Mapping
from typing import cast

from dinkster_api.v1 import (
    ABSENT,
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    ComboWidget,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    SelectorSpec,
    TypeExpr,
)

INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
COMBO = TypeExpr.concrete(CORE_COMBO)
T = TypeExpr.variable("T")
NUMBER = TypeExpr.variable("N", (CORE_INT, CORE_FLOAT))

_COMPARISONS = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
}


def _combo(options: tuple[str, ...], default: str) -> InputSpec:
    return InputSpec("operation", COMBO, default=default, widget=ComboWidget(options=options))


class ValueCompare(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.value.compare",
            display_name="Compare Values",
            category="logic",
            inputs=(
                InputSpec("a", TypeExpr.wildcard()),
                InputSpec("b", TypeExpr.wildcard()),
                _combo(("eq", "ne", "lt", "le", "gt", "ge"), "eq"),
                InputSpec("epsilon", FLOAT, default=0.0, widget=NumberWidget(min=0.0)),
            ),
            outputs=(OutputSpec("result", BOOLEAN),),
        )

    @classmethod
    def execute(
        cls, *, a: object, b: object, operation: str, epsilon: float
    ) -> Mapping[str, object]:
        raw_comparison = _COMPARISONS.get(operation)
        if raw_comparison is None:
            raise ValueError(f"unknown comparison operation {operation!r}")
        comparison = cast("Callable[[object, object], bool]", raw_comparison)
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be finite and >= 0")
        numeric = type(a) in (int, float) and type(b) in (int, float)
        if numeric:
            left_number = cast("int | float", a)
            right_number = cast("int | float", b)
            if (type(a) is float and not math.isfinite(left_number)) or (
                type(b) is float and not math.isfinite(right_number)
            ):
                raise ValueError("numeric comparisons require finite values")
            if epsilon == 0:
                return cls.outputs(result=comparison(left_number, right_number))
            try:
                left = float(left_number)
                right = float(right_number)
            except OverflowError:
                raise ValueError("epsilon comparison inputs must fit float64") from None
        else:
            if epsilon != 0:
                raise ValueError("epsilon is only valid for numeric comparisons")
            if type(a) is not type(b):
                raise TypeError("non-numeric comparisons require matching input types")
            if operation not in ("eq", "ne") and type(a) not in (str, bool):
                raise TypeError(f"{operation} is not supported for {type(a).__name__}")
            return cls.outputs(result=comparison(a, b))

        if operation == "eq":
            result = abs(left - right) <= epsilon
        elif operation == "ne":
            result = abs(left - right) > epsilon
        else:
            boundary = right + epsilon if operation in ("le", "gt") else right - epsilon
            result = comparison(left, boundary)
        return cls.outputs(result=result)


class ValueSelect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.value.select",
            aliases=("ComfySwitchNode",),
            display_name="Select Value",
            category="logic",
            inputs=(
                InputSpec("condition", BOOLEAN),
                InputSpec("on_false", T, lazy=True),
                InputSpec("on_true", T, lazy=True),
            ),
            outputs=(OutputSpec("value", T),),
            selector=SelectorSpec("condition", {"false": "on_false", "true": "on_true"}),
        )

    @classmethod
    def check_lazy_status(
        cls, *, condition: bool, on_false: object | None, on_true: object | None
    ) -> tuple[str, ...]:
        if condition and on_true is None:
            return ("on_true",)
        if not condition and on_false is None:
            return ("on_false",)
        return ()

    @classmethod
    def execute(
        cls, *, condition: bool, on_false: object | None, on_true: object | None
    ) -> Mapping[str, object]:
        return cls.outputs(value=on_true if condition else on_false)


class BoolLogic(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.bool.logic",
            display_name="Boolean Logic",
            category="logic",
            inputs=(_combo(("and", "or", "not", "xor"), "and"),),
            input_families=(InputFamilySpec("values", BOOLEAN, min_members=1),),
            outputs=(OutputSpec("result", BOOLEAN),),
        )

    @classmethod
    def execute(cls, *, operation: str, values: Mapping[str, bool]) -> Mapping[str, object]:
        items = list(values.values())
        if operation == "and":
            result = all(items)
        elif operation == "or":
            result = any(items)
        elif operation == "not":
            if len(items) != 1:
                raise ValueError("not requires exactly one input")
            result = not items[0]
        elif operation == "xor":
            result = sum(bool(item) for item in items) % 2 == 1
        else:
            raise ValueError(f"unknown boolean operation {operation!r}")
        return cls.outputs(result=result)


class ValueClamp(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.value.clamp",
            display_name="Clamp Value",
            category="math",
            inputs=(
                InputSpec("value", NUMBER),
                InputSpec("minimum", NUMBER),
                InputSpec("maximum", NUMBER),
            ),
            outputs=(OutputSpec("value", NUMBER),),
        )

    @classmethod
    def execute(
        cls, *, value: int | float, minimum: int | float, maximum: int | float
    ) -> Mapping[str, object]:
        if any(
            type(item) is float and not math.isfinite(item) for item in (value, minimum, maximum)
        ):
            raise ValueError("clamp inputs must be finite")
        if minimum > maximum:
            raise ValueError("minimum must be <= maximum")
        return cls.outputs(value=min(maximum, max(minimum, value)))


class ValueRemap(Node):
    CURVES = ("linear", "smoothstep", "smootherstep", "ease_in", "ease_out", "ease_in_out")

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.value.remap",
            display_name="Remap Value",
            category="math",
            inputs=(
                InputSpec("value", FLOAT, default=0.0),
                InputSpec("input_minimum", FLOAT, default=0.0),
                InputSpec("input_maximum", FLOAT, default=1.0),
                InputSpec("output_minimum", FLOAT, default=0.0),
                InputSpec("output_maximum", FLOAT, default=1.0),
                InputSpec("clamp", BOOLEAN, default=True),
                InputSpec("curve", COMBO, default="linear", widget=ComboWidget(options=cls.CURVES)),
            ),
            outputs=(OutputSpec("value", FLOAT),),
        )

    @classmethod
    def execute(
        cls,
        *,
        value: float,
        input_minimum: float,
        input_maximum: float,
        output_minimum: float,
        output_maximum: float,
        clamp: bool,
        curve: str,
    ) -> Mapping[str, object]:
        inputs = (value, input_minimum, input_maximum, output_minimum, output_maximum)
        if not all(math.isfinite(item) for item in inputs):
            raise ValueError("remap inputs must be finite")
        if input_minimum == input_maximum:
            raise ValueError("input range must not be empty")
        amount = (value - input_minimum) / (input_maximum - input_minimum)
        if not math.isfinite(amount):
            raise ValueError("remap normalization produced a non-finite value")
        if clamp:
            amount = min(1.0, max(0.0, amount))
        if curve == "smoothstep":
            amount = amount * amount * (3.0 - 2.0 * amount)
        elif curve == "smootherstep":
            amount = amount**3 * (amount * (amount * 6.0 - 15.0) + 10.0)
        elif curve == "ease_in":
            amount = amount * amount
        elif curve == "ease_out":
            amount = 1.0 - (1.0 - amount) ** 2
        elif curve == "ease_in_out":
            amount = (
                2.0 * amount * amount if amount < 0.5 else 1.0 - (-2.0 * amount + 2.0) ** 2 / 2.0
            )
        elif curve != "linear":
            raise ValueError(f"unknown remap curve {curve!r}")
        result = (1.0 - amount) * output_minimum + amount * output_maximum
        if not math.isfinite(result):
            raise ValueError("remap produced a non-finite value")
        return cls.outputs(value=result)


_MASK_64 = (1 << 64) - 1
_FLOAT_UNIT = float(1 << 53)


class _SplitMix64:
    def __init__(self, seed: int) -> None:
        self.state = seed & _MASK_64

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK_64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK_64
        return value ^ (value >> 31)

    def below(self, bound: int) -> int:
        limit = (1 << 64) - ((1 << 64) % bound)
        value = self.next()
        while value >= limit:
            value = self.next()
        return value % bound


class ValueRandom(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.value.random",
            display_name="Random Value",
            category="math",
            description="Deterministic SplitMix64 with unsigned 64-bit seed normalization.",
            inputs=(
                InputSpec(
                    "seed",
                    INT,
                    default=0,
                    widget=NumberWidget(
                        min=0,
                        max=(1 << 53) - 1,
                        step=1,
                        control_after_generate="randomize",
                    ),
                ),
                InputSpec("minimum", FLOAT, default=0.0),
                InputSpec("maximum", FLOAT, default=1.0),
                InputSpec(
                    "number_type",
                    COMBO,
                    default="float",
                    widget=ComboWidget(options=("float", "int")),
                ),
            ),
            outputs=(
                OutputSpec("int", INT, optional=True),
                OutputSpec("float", FLOAT, optional=True),
            ),
        )

    @classmethod
    def execute(
        cls, *, seed: int, minimum: float, maximum: float, number_type: str
    ) -> Mapping[str, object]:
        if type(seed) is not int:
            raise TypeError("seed must be an int")
        if number_type not in ("float", "int"):
            raise ValueError(f"unknown random number type {number_type!r}")
        if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum > maximum:
            raise ValueError("random range must be finite and minimum must be <= maximum")
        generator = _SplitMix64(seed)
        if number_type == "int":
            if not float(minimum).is_integer() or not float(maximum).is_integer():
                raise ValueError("integer random ranges require integral endpoints")
            lower = int(minimum)
            upper = int(maximum)
            width = upper - lower + 1
            if width <= 0 or width > 1 << 64:
                raise ValueError("integer random range is too wide")
            return cls.outputs(int=lower + generator.below(width), float=ABSENT)
        unit = (generator.next() >> 11) / _FLOAT_UNIT
        if minimum == maximum:
            value = minimum
        else:
            value = (1.0 - unit) * minimum + unit * maximum
            value = min(math.nextafter(maximum, minimum), max(minimum, value))
        return cls.outputs(int=ABSENT, float=value)


NUMERIC_NODES: tuple[type[Node], ...] = (
    ValueCompare,
    ValueSelect,
    BoolLogic,
    ValueClamp,
    ValueRemap,
    ValueRandom,
)
