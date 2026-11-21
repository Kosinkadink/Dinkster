"""The formula language that the engine and the editor both evaluate.

A formula means what Python says it means, on a closed subset where a browser
can give exactly the same answer. The editor's twin lives in Dinkster-Frontend
``packages/core/src/mirror/formula.ts``; both load the vector file written
by ``tools/generate_formula_vectors.py`` and must agree on every case.

A formula has three outcomes besides a value. *Refused* is decided when the
pack loads (:class:`FormulaRefused`). *Unknown* means a name it needs has no
value yet. *Failed* means the values make it impossible. Unknown and failed
both stop evaluation at the first operand that produces them, in Python's
evaluation order, except where ``and``, ``or`` and ``x if c else y`` never
look at an operand and where ``known(x)`` asks about one.
"""

from __future__ import annotations

import ast
import decimal
import io
import keyword
import math
import re
import tokenize
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, cast

FORMULA_LANGUAGE_VERSION: Final = 1

MAX_FORMULA_LENGTH: Final = 4096
MAX_FORMULA_PARTS: Final = 256
MAX_BRACKET_DEPTH: Final = 200
MAX_INTEGER_BITS: Final = 16384
MAX_SHIFT: Final = 1000
MAX_EXPONENT: Final = 4000
MAX_LIST_LENGTH: Final = 4096
MAX_TEXT_LITERAL: Final = 4096
MAX_TEXT_VALUE: Final = 65536
MAX_TABLE_ITEMS: Final = 4096
MAX_ROUND_DIGITS: Final = 32
MAX_FIXED_DIGITS: Final = 12

VALUE_KINDS: Final = frozenset({"int", "float", "bool", "text", "choice"})
"""Kinds of a name that has a value. ``<kind>[]`` is a group row across members."""
NAME_KINDS: Final = VALUE_KINDS | {f"{kind}[]" for kind in VALUE_KINDS} | {"opaque", "group"}

ASSET_FACTS: Final = ("width", "height", "frames", "fps", "duration", "kind", "filename")


class FormulaRefused(ValueError):
    """The formula is outside the language; decided when the pack loads."""


class _UnknownType:
    _instance: _UnknownType | None = None

    def __new__(cls) -> _UnknownType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNKNOWN"


UNKNOWN: Final = _UnknownType()


@dataclass(frozen=True, slots=True)
class Failed:
    """The values make the formula impossible. ``reason`` is for the pack author."""

    reason: str


@dataclass(frozen=True, slots=True)
class Table:
    """A constant from ``NodeSchema.tables``, or a dictionary inside one."""

    items: Mapping[str | int, object]

    def __bool__(self) -> bool:
        return bool(self.items)


type Value = int | float | bool | str | None | tuple[Value, ...]
type Outcome = Value | _UnknownType | Failed


@dataclass(frozen=True, slots=True)
class FormulaScope:
    """What a formula may name, known when the pack loads.

    ``names`` maps each visible name (a dotted name is one name) to its kind,
    one of :data:`NAME_KINDS`. ``inputs`` are the names ``linked`` and
    ``linked_type`` may ask about; ``assets`` the names the asset facts may.
    ``derived`` holds the node's named formulas; ``in_member`` allows
    ``index()`` and ``key()``.
    """

    names: Mapping[str, str] = field(default_factory=dict[str, str])
    inputs: frozenset[str] = frozenset()
    assets: frozenset[str] = frozenset()
    tables: Mapping[str, object] = field(default_factory=dict[str, object])
    derived: Mapping[str, str] = field(default_factory=dict[str, str])
    in_member: bool = False


@dataclass(frozen=True, slots=True)
class FormulaValues:
    """The state a formula is evaluated against.

    A name missing from ``values``, or bound to :data:`UNKNOWN`, is unknown.
    ``facts`` is keyed by ``(fact, name)``, such as ``("fps", "video")``.
    """

    values: Mapping[str, object] = field(default_factory=dict[str, object])
    linked: Mapping[str, bool] = field(default_factory=dict[str, bool])
    linked_types: Mapping[str, str | None] = field(default_factory=dict[str, str | None])
    group_sizes: Mapping[str, int] = field(default_factory=dict[str, int])
    member_index: int | None = None
    member_key: str | None = None
    facts: Mapping[tuple[str, str], object] = field(default_factory=dict[tuple[str, str], object])


class _Stop(Exception):
    def __init__(self, outcome: _UnknownType | Failed) -> None:
        super().__init__()
        self.outcome = outcome


def _fail(reason: str) -> _Stop:
    return _Stop(Failed(reason))


def _refuse(reason: str) -> FormulaRefused:
    return FormulaRefused(reason)


# --- text-level check -------------------------------------------------------

_NUMBER = re.compile(
    r"(?:0[xX](?:_?[0-9a-fA-F])+"
    r"|(?:[1-9](?:_?[0-9])*|0)"
    r"|(?:(?:[0-9](?:_?[0-9])*)?\.[0-9](?:_?[0-9])*|[0-9](?:_?[0-9])*\.)(?:[eE][+-]?[0-9](?:_?[0-9])*)?"
    r"|[0-9](?:_?[0-9])*[eE][+-]?[0-9](?:_?[0-9])*)\Z"
)
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_ESCAPES = {"\\": "\\", '"': '"', "'": "'", "n": "\n", "t": "\t"}


def _check_text_literal(token: str) -> None:
    quote = token[0]
    if quote not in "\"'":
        raise _refuse("text literals take no prefix")
    if token.startswith(quote * 3):
        raise _refuse("text literals cannot be triple-quoted")
    body = token[1:-1]
    index = 0
    length = 0
    while index < len(body):
        char = body[index]
        if char == "\n":
            raise _refuse("a text literal is one line")
        if char == "\\":
            escaped = body[index + 1 : index + 2]
            if escaped not in _ESCAPES:
                raise _refuse(f"escape \\{escaped} is not allowed")
            index += 2
        else:
            index += 1
        length += 1
    if length > MAX_TEXT_LITERAL:
        raise _refuse(f"a text literal exceeds {MAX_TEXT_LITERAL} characters")


_WHITESPACE: Final = " \t\n\r\f"
_OUTSIDE_TEXT = re.compile(r"[^\x20-\x7e\t\n\r\f]")
_TEXT_LITERAL = re.compile(r"'(?:[^'\\\n]|\\.)*'|\"(?:[^\"\\\n]|\\.)*\"")


def _blank_text_literals(text: str) -> str:
    """``text`` with every one-line text literal replaced by spaces."""
    return _TEXT_LITERAL.sub(lambda match: " " * len(match.group(0)), text)


def _check_tokens(text: str) -> None:
    depth = 0
    previous: tokenize.TokenInfo | None = None
    if _OUTSIDE_TEXT.search(text) is not None and _OUTSIDE_TEXT.search(_blank_text_literals(text)):
        raise _refuse("outside text literals a formula holds printable ASCII only")
    try:
        # An unknown escape is refused below; Python would also print a SyntaxWarning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError, ValueError) as exc:
        raise _refuse(f"invalid formula: {exc}") from None
    for token in tokens:
        kind = token.type
        if kind in (
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.ENDMARKER,
            tokenize.INDENT,
            tokenize.DEDENT,
        ):
            continue
        if kind == tokenize.COMMENT:
            raise _refuse("comments are not allowed")
        if kind == tokenize.ERRORTOKEN:
            raise _refuse(f"invalid character {token.string!r}")
        if kind == tokenize.STRING:
            if previous is not None and previous.type == tokenize.STRING:
                raise _refuse("two text literals cannot stand side by side")
            _check_text_literal(token.string)
        elif kind in (tokenize.FSTRING_START, tokenize.FSTRING_MIDDLE, tokenize.FSTRING_END):
            raise _refuse("text literals take no prefix")
        elif kind == tokenize.NUMBER:
            if not _NUMBER.match(token.string):
                raise _refuse(f"number {token.string!r} is not allowed")
        elif kind == tokenize.NAME:
            if not _NAME.match(token.string):
                raise _refuse(f"name {token.string!r} is not ASCII")
            if (
                previous is not None
                and previous.type == tokenize.NUMBER
                and previous.end == token.start
            ):
                raise _refuse("a name cannot follow a number directly")
        elif kind == tokenize.OP:
            if token.string in ("(", "[", "{"):
                depth += 1
                if depth > MAX_BRACKET_DEPTH:
                    raise _refuse(f"brackets nest deeper than {MAX_BRACKET_DEPTH}")
            elif token.string in (")", "]", "}"):
                depth -= 1
        previous = token
    if re.search(r"\\\s*\n", text) or text.rstrip().endswith("\\"):
        raise _refuse("line continuations are not allowed")


# --- structure check --------------------------------------------------------

_BINARY: Final[Mapping[type[ast.operator], str]] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.BitAnd: "&",
}
_COMPARE: Final[Mapping[type[ast.cmpop], str]] = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.In: "in",
    ast.NotIn: "not in",
}
_UNARY: Final[Mapping[type[ast.unaryop], str]] = {
    ast.UAdd: "+",
    ast.USub: "-",
    ast.Invert: "~",
    ast.Not: "not",
}

# Function name -> (fewest arguments, most arguments or None for no limit).
_ARITY: Final[Mapping[str, tuple[int, int | None]]] = {
    "abs": (1, 1),
    "sign": (1, 1),
    "floor": (1, 1),
    "ceil": (1, 1),
    "round": (1, 2),
    "min": (1, None),
    "max": (1, None),
    "clamp": (3, 3),
    "sum": (1, 1),
    "sqrt": (1, 1),
    "mod": (2, 2),
    "lerp": (3, 3),
    "int": (1, 1),
    "float": (1, 1),
    "len": (1, 1),
    "lower": (1, 1),
    "upper": (1, 1),
    "strip": (1, 1),
    "startswith": (2, 2),
    "endswith": (2, 2),
    "replace": (3, 3),
    "split": (2, 2),
    "lines": (1, 1),
    "join": (2, 2),
    "str": (1, 1),
    "fixed": (2, 2),
    "number": (1, 1),
    "any": (1, 1),
    "all": (1, 1),
    "count": (2, 2),
    "position": (2, 2),
    "range": (1, 1),
    "get": (3, 3),
    "linked": (1, 1),
    "linked_type": (1, 1),
    "index": (0, 0),
    "key": (0, 0),
    "known": (1, 1),
    **{fact: (1, 1) for fact in ASSET_FACTS},
}
_NAME_ARGUMENT_FACTS: Final = frozenset({"linked", "linked_type", *ASSET_FACTS})


def _dotted(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


@dataclass(slots=True)
class _Checker:
    scope: FormulaScope
    derived_stack: tuple[str, ...]
    locals_: frozenset[str] = frozenset()
    parts: int = 0
    in_comprehension: bool = False
    derived_used: set[str] = field(default_factory=set[str])

    def count(self, n: int = 1) -> None:
        self.parts += n
        if self.parts > MAX_FORMULA_PARTS:
            raise _refuse(f"formula exceeds {MAX_FORMULA_PARTS} parts")

    def name(self, name: str, *, as_value: bool = True) -> None:
        if name in self.locals_:
            return
        if name in self.scope.derived:
            if name in self.derived_stack:
                raise _refuse(f"derived formula {name!r} refers to itself")
            self.derived_used.add(name)
            return
        if name in self.scope.tables:
            return
        kind = self.scope.names.get(name)
        if kind is None:
            raise _refuse(f"unknown name {name!r}")
        if as_value and kind == "opaque":
            raise _refuse(f"{name!r} has no value in a formula; ask about it with a fact")
        if as_value and kind == "group":
            raise _refuse(f"group {name!r} has no value in a formula; use len({name})")

    def visit(self, node: ast.expr) -> None:
        dotted = _dotted(node)
        if dotted is not None:
            self.count()
            if isinstance(node, ast.Name) or dotted in self.scope.names:
                self.name(dotted)
                return
            raise _refuse(f"unknown name {dotted!r}")
        if isinstance(node, ast.Constant):
            self.count()
            if type(node.value) not in (int, float, bool, str) and node.value is not None:
                raise _refuse(f"literal {node.value!r} is not allowed")
            return
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _BINARY:
                raise _refuse(f"operator {type(node.op).__name__} is not allowed")
            self.count()
            self.visit(node.left)
            self.visit(node.right)
            return
        if isinstance(node, ast.UnaryOp):
            self.count()
            self.visit(node.operand)
            return
        if isinstance(node, ast.BoolOp):
            self.count(len(node.values) - 1)
            for value in node.values:
                self.visit(value)
            return
        if isinstance(node, ast.Compare):
            for operation in node.ops:
                if type(operation) not in _COMPARE:
                    raise _refuse(f"comparison {type(operation).__name__} is not allowed")
            self.count(len(node.ops))
            self.visit(node.left)
            for comparator in node.comparators:
                self.visit(comparator)
            return
        if isinstance(node, ast.IfExp):
            self.count()
            self.visit(node.test)
            self.visit(node.body)
            self.visit(node.orelse)
            return
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Slice):
                raise _refuse("slices are not allowed")
            self.count()
            self.visit(node.value)
            self.visit(node.slice)
            return
        if isinstance(node, (ast.List, ast.Tuple)):
            self.count()
            for item in node.elts:
                if isinstance(item, ast.Starred):
                    raise _refuse("unpacking is not allowed")
                self.visit(item)
            return
        if isinstance(node, ast.ListComp):
            self.comprehension(node)
            return
        if isinstance(node, ast.Call):
            self.call(node)
            return
        raise _refuse(f"syntax {type(node).__name__} is not allowed")

    def comprehension(self, node: ast.ListComp) -> None:
        if self.in_comprehension:
            raise _refuse("comprehensions cannot nest")
        if len(node.generators) != 1:
            raise _refuse("a comprehension has one for")
        generator = node.generators[0]
        if generator.is_async or len(generator.ifs) > 1:
            raise _refuse("a comprehension has one for and at most one if")
        if not isinstance(generator.target, ast.Name):
            raise _refuse("a comprehension binds one name")
        target = generator.target.id
        if (
            target in self.scope.names
            or target in self.scope.tables
            or target in self.scope.derived
        ):
            raise _refuse(f"comprehension name {target!r} is already a row, table or formula")
        self.count()
        self.in_comprehension = True
        self.visit(generator.iter)
        self.in_comprehension = False
        inner = _Checker(
            self.scope,
            self.derived_stack,
            self.locals_ | {target},
            self.parts + 1,
            True,
            self.derived_used,
        )
        inner.visit(node.elt)
        for condition in generator.ifs:
            inner.visit(condition)
        self.parts = inner.parts

    def call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id in self.locals_:
            raise _refuse("only the language's functions can be called")
        name = node.func.id
        if node.keywords:
            raise _refuse("keyword arguments are not allowed")
        arity = _ARITY.get(name)
        if arity is None:
            raise _refuse(f"unknown function {name!r}")
        fewest, most = arity
        count = len(node.args)
        if count < fewest or (most is not None and count > most):
            raise _refuse(f"{name}() takes the wrong number of arguments")
        if any(isinstance(argument, ast.Starred) for argument in node.args):
            raise _refuse("unpacking is not allowed")
        self.count()
        if name in ("index", "key") and not self.scope.in_member:
            raise _refuse(f"{name}() is only allowed inside a group member's rows")
        if name in _NAME_ARGUMENT_FACTS:
            target = _dotted(node.args[0])
            allowed = (
                self.scope.inputs
                if name in ("linked", "linked_type")
                else (self.scope.assets | self.scope.inputs)
            )
            if target is None or target not in allowed:
                raise _refuse(f"{name}() takes the name of an input")
            self.count()
            return
        if name == "len" and _dotted(node.args[0]) is not None:
            target = cast(str, _dotted(node.args[0]))
            if self.scope.names.get(target) == "group":
                self.count()
                return
        if name == "known":
            argument = node.args[0]
            target = _dotted(argument)
            if target is not None:
                self.count()
                self.name(target, as_value=self.scope.names.get(target) != "opaque")
                return
            if not (
                isinstance(argument, ast.Call)
                and isinstance(argument.func, ast.Name)
                and argument.func.id in ASSET_FACTS
            ):
                raise _refuse("known() takes a name or a fact")
        for argument in node.args:
            self.visit(argument)


@dataclass(frozen=True, slots=True)
class Formula:
    """A formula checked against a scope, ready to evaluate."""

    text: str
    scope: FormulaScope
    tree: ast.expr
    derived: Mapping[str, Formula]

    def evaluate(self, values: FormulaValues) -> Outcome:
        return _Evaluator(self.scope, values, {}).outcome(self)


def parse_formula(text: str, scope: FormulaScope) -> Formula:
    """Check ``text`` against ``scope``; raise :class:`FormulaRefused` if outside the language."""
    return _compile(text, scope, (), {})


def _compile(
    text: str, scope: FormulaScope, stack: tuple[str, ...], done: dict[str, Formula]
) -> Formula:
    if not text.strip(_WHITESPACE):
        raise _refuse("a formula cannot be empty")
    if len(text) > MAX_FORMULA_LENGTH:
        raise _refuse(f"formula exceeds {MAX_FORMULA_LENGTH} characters")
    for name in scope.names:
        if not all(_NAME.match(part) and not keyword.iskeyword(part) for part in name.split(".")):
            raise _refuse(f"row id {name!r} cannot be named in a formula")
    if re.search(r"[\x00\ud800-\udfff]", text):
        raise _refuse("a formula cannot hold NUL or a lone surrogate")
    if re.search(r"\r(?!\n)", text):
        raise _refuse("a carriage return must be followed by a line feed")
    _check_tokens(text)
    try:
        parsed = ast.parse(text.strip(_WHITESPACE), mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        raise _refuse(f"invalid formula: {exc}") from None
    checker = _Checker(scope, stack)
    checker.visit(parsed.body)
    derived: dict[str, Formula] = {}
    for name in sorted(checker.derived_used):
        if name not in done:
            done[name] = _compile(scope.derived[name], scope, (*stack, name), done)
        derived[name] = done[name]
    return Formula(text, scope, parsed.body, derived)


def check_scope(scope: FormulaScope) -> None:
    """Refuse a scope whose tables or derived formulas are outside the language."""
    for name, kind in scope.names.items():
        if kind not in NAME_KINDS:
            raise _refuse(f"{name!r} has unknown kind {kind!r}")
    for name, table in scope.tables.items():
        if _table_items(table) > MAX_TABLE_ITEMS:
            raise _refuse(f"table {name!r} exceeds {MAX_TABLE_ITEMS} items")
    for name, text in scope.derived.items():
        _compile(text, scope, (name,), {})


def _table_items(value: object) -> int:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        for key in mapping:
            if type(key) not in (str, int):
                raise _refuse("table keys are text or whole numbers")
        return len(mapping) + sum(_table_items(item) for item in mapping.values())
    if isinstance(value, (list, tuple)):
        items = cast(Sequence[object], value)
        for item in items:
            if item is not None and type(item) not in (int, float, bool, str):
                raise _refuse("a list in a table holds plain values only")
        return len(items)
    if value is None or type(value) in (int, float, bool, str):
        return 0
    raise _refuse(f"table value {value!r} is not plain data")


# --- values -----------------------------------------------------------------


def _check(value: object) -> object:
    kind = type(value)
    if kind is int:
        if cast(int, value).bit_length() > MAX_INTEGER_BITS:
            raise _fail(f"whole number exceeds {MAX_INTEGER_BITS} bits")
    elif kind is float:
        if not math.isfinite(cast(float, value)):
            raise _fail("result is not a finite number")
    elif kind is str:
        if len(cast(str, value)) > MAX_TEXT_VALUE:
            raise _fail(f"text exceeds {MAX_TEXT_VALUE} characters")
    elif kind is tuple:
        if len(cast(tuple[object, ...], value)) > MAX_LIST_LENGTH:
            raise _fail(f"list exceeds {MAX_LIST_LENGTH} items")
    return value


def _is_number(value: object) -> bool:
    return type(value) in (int, float, bool)


def _is_whole(value: object) -> bool:
    return type(value) in (int, bool)


def _number(value: object, what: str) -> int | float | bool:
    if not _is_number(value):
        raise _fail(f"{what} needs a number, got {_type_name(value)}")
    return cast(int | float | bool, value)


def _text(value: object, what: str) -> str:
    if not isinstance(value, str):
        raise _fail(f"{what} needs text, got {_type_name(value)}")
    return value


def _list(value: object, what: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise _fail(f"{what} needs a list, got {_type_name(value)}")
    return cast(tuple[object, ...], value)


def _type_name(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, Table):
        return "table"
    names: dict[type, str] = {
        bool: "boolean",
        int: "whole number",
        float: "float",
        str: "text",
        tuple: "list",
    }
    return names.get(type(value), type(value).__name__)


def _list_item(value: object) -> object:
    if value is None or type(value) in (int, float, bool, str):
        return value
    raise _fail(f"a list holds numbers, booleans and text, not {_type_name(value)}")


def _concatenated(left: object, right: object) -> object | None:
    if isinstance(left, str) and isinstance(right, str):
        return left + right
    if type(left) is tuple and type(right) is tuple:
        return cast(tuple[object, ...], left) + cast(tuple[object, ...], right)
    return None


def _arith(symbol: str, left: object, right: object) -> object:
    if symbol == "+" and (joined := _concatenated(left, right)) is not None:
        return joined
    if symbol in ("<<", ">>", "|", "^", "&"):
        if not (_is_whole(left) and _is_whole(right)):
            raise _fail(f"{symbol} needs whole numbers")
        a, b = cast(int, left), cast(int, right)
        if symbol in ("<<", ">>"):
            if b < 0 or b > MAX_SHIFT:
                raise _fail(f"shift amount must be between 0 and {MAX_SHIFT}")
            if symbol == "<<" and a.bit_length() + b > MAX_INTEGER_BITS + 1:
                raise _fail(f"whole number exceeds {MAX_INTEGER_BITS} bits")
            return a << b if symbol == "<<" else a >> b
        return {"|": a | b, "^": a ^ b, "&": a & b}[symbol]
    a, b = _number(left, symbol), _number(right, symbol)
    try:
        if symbol == "+":
            return a + b
        if symbol == "-":
            return a - b
        if symbol == "*":
            return a * b
        if symbol == "/":
            return a / b
        if symbol == "//":
            return a // b
        if symbol == "%":
            return a % b
        if symbol == "**":
            if not (_is_whole(a) and _is_whole(b)):
                raise _fail("** needs whole numbers on both sides")
            base, exponent = int(a), int(b)
            if exponent < 0 or exponent > MAX_EXPONENT:
                raise _fail(f"exponent must be between 0 and {MAX_EXPONENT}")
            if abs(base) > 1 and (base.bit_length() - 1) * exponent > MAX_INTEGER_BITS:
                raise _fail(f"whole number exceeds {MAX_INTEGER_BITS} bits")
            return base**exponent
    except ZeroDivisionError:
        raise _fail("division by zero") from None
    except OverflowError:
        raise _fail("number too large") from None
    raise AssertionError(symbol)


def _ordered(left: object, right: object, symbol: str) -> None:
    if not (_is_number(left) and _is_number(right)):
        raise _fail(f"{symbol} compares numbers only")


def _contains(container: object, item: object) -> bool:
    if type(container) is str:
        return _text(item, "in") in container
    if type(container) is tuple:
        return item in cast(tuple[object, ...], container)
    if isinstance(container, Table):
        if item is None or type(item) not in (int, float, bool, str):
            raise _fail("a table key is text or a whole number")
        return item in container.items
    raise _fail(f"in needs text, a list or a table, got {_type_name(container)}")


def _from_table(value: object) -> object:
    if isinstance(value, Mapping):
        return Table(cast(Mapping[str | int, object], value))
    if isinstance(value, (list, tuple)):
        return tuple(_from_table(item) for item in cast(Sequence[object], value))
    return value


def _subscript(container: object, index: object) -> object:
    if isinstance(container, Table):
        if index is None or type(index) not in (int, float, bool, str):
            raise _fail("a table key is text or a whole number")
        try:
            return _from_table(container.items[cast(str | int, index)])
        except KeyError:
            raise _fail(f"table has no key {index!r}") from None
    if type(container) not in (str, tuple):
        raise _fail(f"x[i] needs text, a list or a table, got {_type_name(container)}")
    if not _is_whole(index):
        raise _fail("an index is a whole number")
    sequence = cast(str | tuple[object, ...], container)
    try:
        return sequence[cast(int, index)]
    except IndexError:
        raise _fail("index out of range") from None


def _bind(kind: str, value: object) -> object:
    if value is UNKNOWN:
        return UNKNOWN
    if kind.endswith("[]"):
        item_kind = kind[:-2]
        items = cast(Sequence[object], value)
        bound = tuple(_bind(item_kind, item) for item in items)
        return UNKNOWN if any(item is UNKNOWN for item in bound) else bound
    if kind == "float" and type(value) in (int, bool):
        return float(cast(int, value))
    if kind == "int" and type(value) is bool:
        return int(value)
    return value


# --- functions --------------------------------------------------------------


def _round(x: object, digits: object = None) -> object:
    number = _number(x, "round")
    if digits is None:
        return round(number)
    if not _is_whole(digits):
        raise _fail("round() digits is a whole number")
    if not -MAX_ROUND_DIGITS <= cast(int, digits) <= MAX_ROUND_DIGITS:
        raise _fail(f"round() digits must be between -{MAX_ROUND_DIGITS} and {MAX_ROUND_DIGITS}")
    return round(number, cast(int, digits))


def _min_max(name: str, arguments: Sequence[object]) -> object:
    items: Sequence[object]
    if len(arguments) == 1:
        items = _list(arguments[0], name)
    else:
        items = arguments
    if not items:
        raise _fail(f"{name}() of an empty list")
    best = _number(items[0], name)
    for item in items[1:]:
        value = _number(item, name)
        if (value < best) if name == "min" else (value > best):
            best = value
    return best


def _sum(items: object) -> object:
    total: object = 0
    for item in _list(items, "sum"):
        total = _check(_arith("+", _number(total, "sum"), _number(item, "sum")))
    return total


def _clamp(x: object, low: object, high: object) -> object:
    value, lo, hi = _number(x, "clamp"), _number(low, "clamp"), _number(high, "clamp")
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _sqrt(x: object) -> float:
    number = _number(x, "sqrt")
    try:
        return math.sqrt(number)
    except ValueError:
        raise _fail("sqrt() of a negative number") from None
    except OverflowError:
        raise _fail("number too large") from None


def _to_int(x: object) -> int:
    return int(_number(x, "int"))


def _to_float(x: object) -> float:
    try:
        return float(_number(x, "float"))
    except OverflowError:
        raise _fail("number too large") from None


def _len(x: object) -> int:
    if type(x) in (str, tuple):
        return len(cast(str | tuple[object, ...], x))
    raise _fail(f"len() needs text, a list or a group, got {_type_name(x)}")


_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
_ASCII_UPPER = str.maketrans("abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_STRIPPED = " \t\n\r"


def _lines(t: object) -> tuple[str, ...]:
    text = _text(t, "lines").replace("\r\n", "\n").replace("\r", "\n")
    return tuple(line.strip(_STRIPPED) for line in text.split("\n") if line.strip(_STRIPPED))


def _split(t: object, sep: object) -> tuple[str, ...]:
    separator = _text(sep, "split")
    if not separator:
        raise _fail("split() with an empty separator")
    return tuple(_text(t, "split").split(separator))


def _join(sep: object, items: object) -> str:
    separator = _text(sep, "join")
    return separator.join(_text(item, "join") for item in _list(items, "join"))


def _str(x: object) -> str:
    if type(x) is int:
        # str() refuses whole numbers above 4300 digits; Decimal converts exactly.
        return format(decimal.Decimal(x), "f")
    if type(x) in (bool, str):
        return str(x)
    raise _fail(f"str() takes a whole number, boolean or text, got {_type_name(x)}")


def _fixed(x: object, digits: object) -> str:
    number = _number(x, "fixed")
    if not _is_whole(digits) or not 0 <= cast(int, digits) <= MAX_FIXED_DIGITS:
        raise _fail(f"fixed() digits must be between 0 and {MAX_FIXED_DIGITS}")
    try:
        return format(float(number), f".{int(cast(int, digits))}f")
    except OverflowError:
        raise _fail("number too large") from None


_NUMBER_TEXT = re.compile(r"[+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?\Z")


def _parse_number(t: object) -> object:
    text = _text(t, "number")
    match = _NUMBER_TEXT.match(text)
    if match is None:
        raise _fail(f"number() cannot read {text!r}")
    if match.group(1) is None and match.group(2) is None:
        if len(text) > MAX_INTEGER_BITS:
            raise _fail(f"whole number exceeds {MAX_INTEGER_BITS} bits")
        return int(decimal.Decimal(text))
    return float(text)


def _count(items: object, x: object) -> int:
    return _list(items, "count").count(x)


def _position(items: object, x: object) -> int:
    try:
        return _list(items, "position").index(x)
    except ValueError:
        raise _fail("position() did not find the item") from None


def _range(n: object) -> tuple[int, ...]:
    if not _is_whole(n):
        raise _fail("range() takes a whole number")
    if cast(int, n) > MAX_LIST_LENGTH:
        raise _fail(f"range() is at most {MAX_LIST_LENGTH}")
    return tuple(range(cast(int, n)))


def _get(table: object, key: object, default: object) -> object:
    if not isinstance(table, Table):
        raise _fail(f"get() needs a table, got {_type_name(table)}")
    if key is None or type(key) not in (int, float, bool, str):
        raise _fail("a table key is text or a whole number")
    if key in table.items:
        return _from_table(table.items[cast(str | int, key)])
    return default


def _any_all(name: str, items: object) -> bool:
    values = _list(items, name)
    return any(values) if name == "any" else all(values)


def _unary(name: str, function: Callable[[int | float | bool], object]) -> Callable[..., object]:
    def call(x: object) -> object:
        return function(_number(x, name))

    return call


def _sign(x: int | float | bool) -> int:
    return (x > 0) - (x < 0)


def _min(*arguments: object) -> object:
    return _min_max("min", arguments)


def _max(*arguments: object) -> object:
    return _min_max("max", arguments)


def _mod(a: object, b: object) -> object:
    return _arith("%", a, b)


def _lerp(a: object, b: object, t: object) -> object:
    return _arith("+", a, _check(_arith("*", _check(_arith("-", b, a)), t)))


def _lower(t: object) -> str:
    return _text(t, "lower").translate(_ASCII_LOWER)


def _upper(t: object) -> str:
    return _text(t, "upper").translate(_ASCII_UPPER)


def _strip(t: object) -> str:
    return _text(t, "strip").strip(_STRIPPED)


def _startswith(t: object, prefix: object) -> bool:
    return _text(t, "startswith").startswith(_text(prefix, "startswith"))


def _endswith(t: object, suffix: object) -> bool:
    return _text(t, "endswith").endswith(_text(suffix, "endswith"))


def _replace(t: object, old: object, new: object) -> str:
    return _text(t, "replace").replace(_text(old, "replace"), _text(new, "replace"))


def _any(items: object) -> bool:
    return _any_all("any", items)


def _all(items: object) -> bool:
    return _any_all("all", items)


_FUNCTIONS: Final[Mapping[str, Callable[..., object]]] = {
    "abs": _unary("abs", abs),
    "sign": _unary("sign", _sign),
    "floor": _unary("floor", math.floor),
    "ceil": _unary("ceil", math.ceil),
    "round": _round,
    "min": _min,
    "max": _max,
    "clamp": _clamp,
    "sum": _sum,
    "sqrt": _sqrt,
    "mod": _mod,
    "lerp": _lerp,
    "int": _to_int,
    "float": _to_float,
    "len": _len,
    "lower": _lower,
    "upper": _upper,
    "strip": _strip,
    "startswith": _startswith,
    "endswith": _endswith,
    "replace": _replace,
    "split": _split,
    "lines": _lines,
    "join": _join,
    "str": _str,
    "fixed": _fixed,
    "number": _parse_number,
    "any": _any,
    "all": _all,
    "count": _count,
    "position": _position,
    "range": _range,
    "get": _get,
}


@dataclass(slots=True)
class _Evaluator:
    scope: FormulaScope
    values: FormulaValues
    derived_results: dict[str, Outcome]
    locals_: dict[str, object] = field(default_factory=dict[str, object])

    def outcome(self, formula: Formula) -> Outcome:
        try:
            result = self.eval(formula.tree, formula)
            if isinstance(result, Table):
                raise _fail("a formula's result cannot be a table")
            return cast(Value, result)
        except _Stop as stop:
            return stop.outcome
        except RecursionError:
            return Failed("formula too deep")

    def lookup(self, name: str, formula: Formula) -> object:
        if name in self.locals_:
            return self.locals_[name]
        if name in formula.derived:
            if name not in self.derived_results:
                self.derived_results[name] = _Evaluator(
                    self.scope, self.values, self.derived_results
                ).outcome(formula.derived[name])
            result = self.derived_results[name]
            if result is UNKNOWN or isinstance(result, Failed):
                raise _Stop(cast(_UnknownType | Failed, result))
            return result
        if name in self.scope.tables:
            return _from_table(self.scope.tables[name])
        kind = self.scope.names[name]
        value = _bind(kind, self.values.values.get(name, UNKNOWN))
        if value is UNKNOWN:
            raise _Stop(UNKNOWN)
        return _check(value)

    def eval(self, node: ast.expr, formula: Formula) -> object:
        dotted = _dotted(node)
        if dotted is not None:
            return self.lookup(dotted, formula)
        if isinstance(node, ast.Constant):
            return _check(node.value)
        if isinstance(node, ast.BinOp):
            left = self.eval(node.left, formula)
            right = self.eval(node.right, formula)
            return _check(_arith(_BINARY[type(node.op)], left, right))
        if isinstance(node, ast.UnaryOp):
            operand = self.eval(node.operand, formula)
            symbol = _UNARY[type(node.op)]
            if symbol == "not":
                return not operand
            if symbol == "~":
                if not _is_whole(operand):
                    raise _fail("~ needs a whole number")
                return ~int(cast(int, operand))
            number = _number(operand, symbol)
            return _check(+number if symbol == "+" else -number)
        if isinstance(node, ast.BoolOp):
            is_and = isinstance(node.op, ast.And)
            result: object = None
            for item in node.values:
                result = self.eval(item, formula)
                if bool(result) != is_and:
                    return result
            return result
        if isinstance(node, ast.Compare):
            left = self.eval(node.left, formula)
            for operation, comparator in zip(node.ops, node.comparators, strict=True):
                right = self.eval(comparator, formula)
                symbol = _COMPARE[type(operation)]
                if symbol in ("<", "<=", ">", ">="):
                    _ordered(left, right, symbol)
                    a, b = cast(float, left), cast(float, right)
                    holds = {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[symbol]
                elif symbol in ("==", "!="):
                    if isinstance(left, Table) or isinstance(right, Table):
                        raise _fail("tables cannot be compared")
                    holds = (left == right) == (symbol == "==")
                else:
                    holds = _contains(right, left) == (symbol == "in")
                if not holds:
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            test = self.eval(node.test, formula)
            return self.eval(node.body if test else node.orelse, formula)
        if isinstance(node, ast.Subscript):
            container = self.eval(node.value, formula)
            index = self.eval(node.slice, formula)
            return _check(_subscript(container, index))
        if isinstance(node, (ast.List, ast.Tuple)):
            return _check(tuple(_list_item(self.eval(item, formula)) for item in node.elts))
        if isinstance(node, ast.ListComp):
            return self.comprehension(node, formula)
        if isinstance(node, ast.Call):
            return self.call(node, formula)
        raise AssertionError(type(node).__name__)

    def comprehension(self, node: ast.ListComp, formula: Formula) -> object:
        generator = node.generators[0]
        target = cast(ast.Name, generator.target).id
        items = _list(self.eval(generator.iter, formula), "a comprehension")
        results: list[object] = []
        for item in items:
            self.locals_[target] = item
            try:
                if all(self.eval(condition, formula) for condition in generator.ifs):
                    results.append(_list_item(self.eval(node.elt, formula)))
            finally:
                del self.locals_[target]
        return _check(tuple(results))

    def call(self, node: ast.Call, formula: Formula) -> object:
        name = cast(ast.Name, node.func).id
        if name in ("linked", "linked_type") or name in ASSET_FACTS:
            target = cast(str, _dotted(node.args[0]))
            if name == "linked":
                return bool(self.values.linked.get(target, False))
            if name == "linked_type":
                return self.values.linked_types.get(target)
            fact = self.values.facts.get((name, target), UNKNOWN)
            if fact is UNKNOWN:
                raise _Stop(UNKNOWN)
            return _check(fact)
        if name == "index" or name == "key":
            value = self.values.member_index if name == "index" else self.values.member_key
            if value is None:
                raise _Stop(UNKNOWN)
            return value
        if name == "known":
            try:
                self.eval(node.args[0], formula)
            except _Stop:
                return False
            return True
        if name == "len":
            target = _dotted(node.args[0])
            if target is not None and self.scope.names.get(target) == "group":
                size = self.values.group_sizes.get(target)
                if size is None:
                    raise _Stop(UNKNOWN)
                return size
        arguments = [self.eval(argument, formula) for argument in node.args]
        return _check(_FUNCTIONS[name](*arguments))
