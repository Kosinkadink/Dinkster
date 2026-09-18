"""Deterministic scalar math expressions."""

from __future__ import annotations

import ast
import math
import operator
import string
from collections.abc import Callable, Mapping, Sequence
from typing import cast

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    InputFamilySpec,
    InputSpec,
    MirrorSpec,
    MirrorTolerance,
    Node,
    NodeSchema,
    OutputSpec,
    StringWidget,
    TextCompletionItem,
    TextCompletions,
    TypeExpr,
)

INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
STRING = TypeExpr.concrete(CORE_STRING)

EXPRESSION_GRAMMAR_VERSION = 1
MAX_EXPRESSION_LENGTH = 4096
MAX_EXPRESSION_NODES = 256
MAX_EXPONENT = 4000
MAX_SHIFT = 1000
MAX_INTEGER_BITS = 16384

Scalar = int | float | bool


def _safe_pow(base: Scalar, exponent: Scalar) -> Scalar:
    if abs(exponent) > MAX_EXPONENT:
        raise ValueError(f"exponent {exponent} exceeds the limit of {MAX_EXPONENT}")
    if (
        type(base) is int
        and type(exponent) is int
        and exponent > 0
        and abs(base) > 1
        and base.bit_length() * exponent > MAX_INTEGER_BITS
    ):
        raise ValueError(f"integer magnitude exceeds the limit of {MAX_INTEGER_BITS} bits")
    return _require_scalar(pow(base, exponent))


def _safe_shift(left: Scalar, right: Scalar, operation: Callable[[int, int], int]) -> int:
    if type(left) is not int or type(right) is not int:
        raise TypeError("bit shifts require integer operands")
    if right < 0 or right > MAX_SHIFT:
        raise ValueError(f"shift amount must be between 0 and {MAX_SHIFT}")
    return operation(left, right)


def _safe_left_shift(left: Scalar, right: Scalar) -> int:
    return _safe_shift(left, right, lambda a, b: a << b)


def _safe_right_shift(left: Scalar, right: Scalar) -> int:
    return _safe_shift(left, right, lambda a, b: a >> b)


def _add(left: Scalar, right: Scalar) -> Scalar:
    return _require_scalar(left + right)


def _subtract(left: Scalar, right: Scalar) -> Scalar:
    return _require_scalar(left - right)


def _multiply(left: Scalar, right: Scalar) -> Scalar:
    return _require_scalar(left * right)


def _divide(left: Scalar, right: Scalar) -> Scalar:
    return _require_scalar(left / right)


def _floor_divide(left: Scalar, right: Scalar) -> Scalar:
    return _require_scalar(left // right)


def _modulo(left: Scalar, right: Scalar) -> Scalar:
    return _require_scalar(left % right)


def _bitwise(left: Scalar, right: Scalar, operation: Callable[[int, int], int]) -> int:
    if type(left) is not int or type(right) is not int:
        raise TypeError("bitwise operators require integer operands")
    return operation(left, right)


def _bitwise_or(left: Scalar, right: Scalar) -> int:
    return _bitwise(left, right, lambda a, b: a | b)


def _bitwise_and(left: Scalar, right: Scalar) -> int:
    return _bitwise(left, right, lambda a, b: a & b)


def _bitwise_xor(left: Scalar, right: Scalar) -> int:
    return _bitwise(left, right, lambda a, b: a ^ b)


def _clamp(value: Scalar, minimum: Scalar, maximum: Scalar) -> Scalar:
    if minimum > maximum:
        raise ValueError("clamp minimum must be <= maximum")
    return min(maximum, max(minimum, value))


def _sign(value: Scalar) -> int:
    return (value > 0) - (value < 0)


def _lerp(start: Scalar, end: Scalar, amount: Scalar) -> Scalar:
    return start + (end - start) * amount


def _log(value: Scalar, base: Scalar | None = None) -> float:
    return math.log(value) if base is None else math.log(value, base)


def _variadic_sum(*values: object) -> Scalar:
    if len(values) == 1 and isinstance(values[0], Sequence):
        values = tuple(cast("Sequence[object]", values[0]))
    return sum(_require_scalar(value) for value in values)


def _minimum(*values: object) -> Scalar:
    if len(values) == 1 and isinstance(values[0], Sequence):
        values = tuple(cast("Sequence[object]", values[0]))
    return min(_require_scalar(value) for value in values)


def _maximum(*values: object) -> Scalar:
    if len(values) == 1 and isinstance(values[0], Sequence):
        values = tuple(cast("Sequence[object]", values[0]))
    return max(_require_scalar(value) for value in values)


FUNCTIONS: Mapping[str, Callable[..., object]] = {
    "sum": _variadic_sum,
    "min": _minimum,
    "max": _maximum,
    "clamp": _clamp,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "abs": abs,
    "sign": _sign,
    "pow": _safe_pow,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": _log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "lerp": _lerp,
    "mod": operator.mod,
    "int": int,
    "float": float,
}

BINARY_OPERATOR_SPECS: Mapping[
    type[ast.operator], tuple[str, Callable[[Scalar, Scalar], object]]
] = {
    ast.Add: ("+", _add),
    ast.Sub: ("-", _subtract),
    ast.Mult: ("*", _multiply),
    ast.Div: ("/", _divide),
    ast.FloorDiv: ("//", _floor_divide),
    ast.Mod: ("%", _modulo),
    ast.Pow: ("**", _safe_pow),
    ast.LShift: ("<<", _safe_left_shift),
    ast.RShift: (">>", _safe_right_shift),
    ast.BitOr: ("|", _bitwise_or),
    ast.BitAnd: ("&", _bitwise_and),
    ast.BitXor: ("^", _bitwise_xor),
}

COMPARE_OPERATOR_SPECS: Mapping[type[ast.cmpop], tuple[str, Callable[[Scalar, Scalar], bool]]] = {
    ast.Eq: ("==", operator.eq),
    ast.NotEq: ("!=", operator.ne),
    ast.Lt: ("<", operator.lt),
    ast.LtE: ("<=", operator.le),
    ast.Gt: (">", operator.gt),
    ast.GtE: (">=", operator.ge),
}

UNARY_OPERATOR_SYMBOLS: Mapping[type[ast.unaryop], str] = {
    ast.UAdd: "+",
    ast.USub: "-",
    ast.Not: "not",
    ast.Invert: "~",
}

BOOLEAN_OPERATOR_SYMBOLS: Mapping[type[ast.boolop], str] = {
    ast.And: "and",
    ast.Or: "or",
}

EXPRESSION_OPERATOR_SYMBOLS = tuple(
    dict.fromkeys(
        (
            *(symbol for symbol, _ in BINARY_OPERATOR_SPECS.values()),
            *UNARY_OPERATOR_SYMBOLS.values(),
            *BOOLEAN_OPERATOR_SYMBOLS.values(),
            *(symbol for symbol, _ in COMPARE_OPERATOR_SPECS.values()),
        )
    )
)

EXPRESSION_COMPLETIONS = TextCompletions(
    items=(
        *(
            TextCompletionItem(
                name,
                label=f"{name}()",
                insert_text=f"{name}(",
                detail="Function",
            )
            for name in FUNCTIONS
        ),
        TextCompletionItem("True", detail="Constant"),
        TextCompletionItem("False", detail="Constant"),
        TextCompletionItem("values", detail="All operands"),
        *(
            TextCompletionItem(
                symbol,
                detail="Operator",
                kind="identifier" if symbol.isalpha() else "operator",
            )
            for symbol in EXPRESSION_OPERATOR_SYMBOLS
        ),
    ),
    input_families=("values",),
)


def _require_scalar(value: object) -> Scalar:
    if type(value) not in (int, float, bool):
        raise TypeError(
            f"expression values must be int, float, or bool, got {type(value).__name__}"
        )
    if type(value) is int and value.bit_length() > MAX_INTEGER_BITS:
        raise ValueError(f"integer magnitude exceeds the limit of {MAX_INTEGER_BITS} bits")
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"expression value is non-finite: {value}")
    return cast("Scalar", value)


def _evaluate(node: ast.AST, names: Mapping[str, object]) -> object:
    if isinstance(node, ast.Constant):
        return _require_scalar(node.value)
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise NameError(f"unknown expression name {node.id!r}")
        return names[node.id]
    if isinstance(node, ast.BinOp):
        operation_spec = BINARY_OPERATOR_SPECS.get(type(node.op))
        if operation_spec is None:
            raise ValueError(f"operator {type(node.op).__name__} is not allowed")
        return operation_spec[1](
            _require_scalar(_evaluate(node.left, names)),
            _require_scalar(_evaluate(node.right, names)),
        )
    if isinstance(node, ast.UnaryOp):
        value = _require_scalar(_evaluate(node.operand, names))
        if isinstance(node.op, ast.UAdd):
            return +value
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.Not):
            return not value
        if isinstance(node.op, ast.Invert):
            if type(value) is not int:
                raise TypeError("bitwise inversion requires an integer operand")
            return ~value
        raise ValueError(f"operator {type(node.op).__name__} is not allowed")
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result: object = True
            for item in node.values:
                result = _evaluate(item, names)
                if not result:
                    break
            return result
        if isinstance(node.op, ast.Or):
            result = False
            for item in node.values:
                result = _evaluate(item, names)
                if result:
                    break
            return result
        raise ValueError(f"operator {type(node.op).__name__} is not allowed")
    if isinstance(node, ast.Compare):
        left = _require_scalar(_evaluate(node.left, names))
        for raw_operation, raw_right in zip(node.ops, node.comparators, strict=True):
            operation_spec = COMPARE_OPERATOR_SPECS.get(type(raw_operation))
            if operation_spec is None:
                raise ValueError(f"operator {type(raw_operation).__name__} is not allowed")
            right = _require_scalar(_evaluate(raw_right, names))
            if not operation_spec[1](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        branch = node.body if _evaluate(node.test, names) else node.orelse
        return _evaluate(branch, names)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ValueError("only direct function calls without keyword arguments are allowed")
        function = FUNCTIONS.get(node.func.id)
        if function is None:
            raise NameError(f"unknown expression function {node.func.id!r}")
        return function(*(_evaluate(argument, names) for argument in node.args))
    if isinstance(node, ast.Subscript):
        container = _evaluate(node.value, names)
        index = _evaluate(node.slice, names)
        if not isinstance(container, Sequence) or type(index) is not int:
            raise TypeError("subscripts require a sequence and an integer index")
        return cast("Sequence[object]", container)[index]
    raise ValueError(f"expression syntax {type(node).__name__} is not allowed")


def _parse(expression: str) -> ast.Expression:
    if not expression.strip():
        raise ValueError("expression cannot be empty")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"expression exceeds {MAX_EXPRESSION_LENGTH} characters")
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression: {exc.msg}") from None
    if sum(1 for _ in ast.walk(parsed)) > MAX_EXPRESSION_NODES:
        raise ValueError(f"expression exceeds {MAX_EXPRESSION_NODES} syntax nodes")
    return parsed


def _scalar_outputs(result: object) -> tuple[float, int, bool]:
    value = _require_scalar(result)
    try:
        float_result = float(value)
    except OverflowError:
        raise ValueError("expression result is too large to represent as a float") from None
    if not math.isfinite(float_result):
        raise ValueError(f"expression produced a non-finite result: {value}")
    return float_result, int(value), bool(value)


def evaluate_expression(expression: str, values: Mapping[str, object]) -> tuple[float, int, bool]:
    """Evaluate grammar v1 over scalar operands."""
    parsed = _parse(expression)
    for name, value in values.items():
        if len(name) != 1 or name not in string.ascii_lowercase:
            raise ValueError(f"expression input name must be one lowercase letter, got {name!r}")
        _require_scalar(value)
    context = dict(values)
    context["values"] = list(values.values())
    return _scalar_outputs(_evaluate(parsed.body, context))


class MathExpression(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.math.expression",
            display_name="Math Expression",
            category="math",
            description=f"Deterministic scalar expression grammar v{EXPRESSION_GRAMMAR_VERSION}.",
            inputs=(
                InputSpec(
                    "expression",
                    STRING,
                    default="a + b",
                    widget=StringWidget(multiline=True, completions=EXPRESSION_COMPLETIONS),
                ),
            ),
            input_families=(
                InputFamilySpec(
                    "values",
                    (
                        InputSpec(
                            "value",
                            TypeExpr.union(CORE_FLOAT, CORE_INT, CORE_BOOLEAN),
                            force_input=True,
                        ),
                    ),
                    min_members=1,
                    member_names=tuple(string.ascii_lowercase),
                ),
            ),
            outputs=(
                OutputSpec("float", FLOAT),
                OutputSpec("int", INT),
                OutputSpec("boolean", BOOLEAN),
            ),
            search_terms=("formula", "calculate", "calculator", "eval"),
            # Bounded because the grammar admits transcendentals, whose libm
            # and JS-engine implementations differ by ulps across platforms.
            # Expressions restricted to correctly-rounded IEEE-754 operations
            # are bit-identical in practice; the parity corpus bit-compares
            # those cases (tools/generate_math_expression_vector.py).
            mirror=MirrorSpec(
                kind="expression",
                precision="bounded",
                tolerance=MirrorTolerance(relative=1e-12),
                grammar_version=EXPRESSION_GRAMMAR_VERSION,
            ),
        )

    @classmethod
    def execute(cls, *, expression: str, values: Mapping[str, object]) -> Mapping[str, object]:
        float_result, int_result, bool_result = evaluate_expression(expression, values)
        return cls.outputs(
            float=float_result,
            int=int_result,
            boolean=bool_result,
        )
