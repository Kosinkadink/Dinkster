"""Math expressions evaluated with the core node's bounded function table."""

from __future__ import annotations

import importlib
import math
import string
from collections.abc import Callable, Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    StringWidget,
    TypeExpr,
)

simple_eval = cast("Callable[..., object]", importlib.import_module("simpleeval").simple_eval)

MAX_EXPONENT = 4000


def _variadic_sum(*args: Any) -> Any:
    if len(args) == 1 and hasattr(args[0], "__iter__"):
        return sum(args[0])
    return sum(args)


def _safe_pow(base: Any, exp: Any) -> Any:
    if abs(exp) > MAX_EXPONENT:
        raise ValueError(f"Exponent {exp} exceeds maximum allowed ({MAX_EXPONENT})")
    return cast("Any", pow(base, exp))


MATH_FUNCTIONS = {
    "sum": _variadic_sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "pow": _safe_pow,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "int": int,
    "float": float,
}


class ComfyMathExpression(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ComfyMathExpression",
            display_name="Math Expression",
            category="utilities",
            inputs=(
                InputSpec(
                    "expression",
                    TypeExpr.concrete(CORE_STRING),
                    default="a + b",
                    widget=StringWidget(multiline=True),
                ),
            ),
            input_families=(
                InputFamilySpec(
                    "values",
                    TypeExpr.union(CORE_FLOAT, CORE_INT, CORE_BOOLEAN),
                    min_members=1,
                    member_names=tuple(string.ascii_lowercase),
                ),
            ),
            outputs=(
                OutputSpec("FLOAT", TypeExpr.concrete(CORE_FLOAT)),
                OutputSpec("INT", TypeExpr.concrete(CORE_INT)),
                OutputSpec("BOOL", TypeExpr.concrete(CORE_BOOLEAN)),
            ),
        )

    @classmethod
    def execute(cls, *, expression: str, values: Mapping[str, object]) -> Mapping[str, object]:
        if not expression.strip():
            raise ValueError("Expression cannot be empty.")
        context = dict(values)
        context["values"] = list(values.values())
        result = simple_eval(expression, names=context, functions=MATH_FUNCTIONS)
        if not isinstance(result, (int, float)):
            raise ValueError(
                f"Math Expression '{expression}' must evaluate to a numeric result, "
                f"got {type(result).__name__}: {result!r}"
            )
        try:
            float_result = float(result)
        except OverflowError:
            raise ValueError(
                f"Math Expression '{expression}' produced a result too large to "
                f"represent as a float: {result}"
            ) from None
        if not math.isfinite(float_result):
            raise ValueError(
                f"Math Expression '{expression}' produced a non-finite result: {result}"
            )
        return cls.outputs(FLOAT=float_result, INT=int(result), BOOL=bool(result))
