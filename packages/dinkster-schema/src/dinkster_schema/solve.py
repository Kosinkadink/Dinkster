"""Runtime type-variable solving for generic nodes (DESIGN 3.13).

Generic combinators (MakeList, ConcatLists, ...) declare template variables
(``TypeExpr.variable("T")``) in their interfaces. Elaboration is pure over
document state (see elaborate.py). Graph validation runs the same solver over
the input types it can already see: two of them that bind one variable
differently reject the graph, and region interfaces use it to prove a generic
body output concrete from its linked, runtime-resolvable input declarations.

At execution, variables resolve *per invocation, at the worker boundary*,
from the authoritative runtime type ids on the input envelopes:

    bind_type_variables(elaborated inputs, input value type ids) -> bindings
    resolved_type_id(output TypeExpr, bindings) -> runtime type id

Unification is structural through the ``list``, ``asset``, and ``stream``
constructors: an input declared ``list<T>`` receiving a ``list<INT>``
value binds T=INT, and the same rule applies to typed assets and streams.
All mentions of a variable must agree, and a non-empty allowlist must contain
the exact bound atom. Registered type equivalences are applied by engine input
admission before this worker-side solver runs.
Violations raise TypeSolveError,
which worker shims surface as contract errors on the node, never engine
crashes.

Determinism: bindings depend only on the elaborated interface and the input
envelopes of one invocation, both fixed before execute() runs. Cache identity
needs no extra bookkeeping - the engine's cache key covers each input's
runtime type id alongside its fingerprint, so the same generic node over
different element types (or the same asset bytes under different stamps)
never collides.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from dinkster_values import parse_asset_type_id, parse_list_type_id
from dinkster_values.streams import parse_stream_type_id
from dinkster_values.tuples import parse_tuple_type_id

from .model import Input, TypeExpr


class TypeSolveError(ValueError):
    """Input envelope types cannot bind this interface's template variables."""


class TypeVariableConflict(TypeSolveError):
    """One template variable is fed two different types."""

    def __init__(self, input_id: str, variable: str, bound: str, got: str) -> None:
        super().__init__(
            f"input '{input_id}': variable '{variable}' already bound to {bound}, got {got}"
        )
        self.input_id = input_id
        self.variable = variable
        self.bound = bound
        self.got = got


def bind_type_variables(
    specs: Iterable[Input], input_types: Mapping[str, str | TypeExpr]
) -> dict[str, TypeExpr]:
    """Solve template bindings, retaining declared record fields when known.

    specs is the *elaborated* input interface (family members flattened);
    input_types maps input id to a runtime type id or its full declaration.
    Callers must exclude absent inputs: ``core.absent`` stands for a missing
    value and can never bind a variable. Inputs with no received value (an
    unfed optional) bind nothing.
    """
    bindings: dict[str, TypeExpr] = {}
    for spec in specs:
        received = input_types.get(spec.id)
        if received is not None:
            expression = (
                _type_expr_from_runtime_id(received) if isinstance(received, str) else received
            )
            if spec.accepts_list and spec.type.kind != "list":
                if expression.kind == "list":
                    assert expression.element is not None
                    expression = expression.element
            _unify(spec.id, spec.type, expression, bindings)
    return bindings


def resolved_type_id(expr: TypeExpr, bindings: Mapping[str, TypeExpr]) -> str | None:
    """The runtime type id of an expression under bindings, else None.

    The bindings-free case is ``TypeExpr.runtime_type_id()``; this is the
    same bridge with solved variables substituted.
    """
    resolved = resolved_type_expr(expr, bindings)
    return None if resolved is None else resolved.runtime_type_id()


def resolved_type_expr(expr: TypeExpr, bindings: Mapping[str, TypeExpr]) -> TypeExpr | None:
    """A fully concrete expression under bindings, else None."""
    if expr.kind in ("concrete", "record"):
        return expr
    if expr.kind == "variable":
        return bindings.get(expr.template_id)
    if expr.kind == "tuple":
        positions = tuple(resolved_type_expr(item, bindings) for item in expr.elements)
        if any(item is None for item in positions):
            return None
        return TypeExpr.tuple_of(*(item for item in positions if item is not None))
    if expr.kind == "tuple_element":
        if expr.element is None or type(expr.index) is not int:
            raise TypeSolveError("tuple element must be elaborated before type solving")
        source = resolved_type_expr(expr.element, bindings)
        if source is None:
            return None
        if source.kind != "tuple":
            raise TypeSolveError("tuple element requires a tuple input")
        if not -len(source.elements) <= expr.index < len(source.elements):
            raise TypeSolveError(f"tuple index {expr.index} is out of range")
        return source.elements[expr.index]
    if expr.kind == "stream":
        assert expr.element is not None
        inner = resolved_type_expr(expr.element, bindings)
        return None if inner is None else TypeExpr.stream_of(inner)
    if expr.kind == "list":
        assert expr.element is not None  # post_init invariant
        if expr.element.kind == "wildcard":
            return expr
        inner = resolved_type_expr(expr.element, bindings)
        return None if inner is None else TypeExpr.list_of(inner)
    if expr.kind == "asset":
        assert expr.element is not None  # post_init invariant
        inner = resolved_type_expr(expr.element, bindings)
        return None if inner is None else TypeExpr.asset_of(inner)
    return None


def _type_expr_from_runtime_id(type_id: str) -> TypeExpr:
    if (positions := parse_tuple_type_id(type_id)) is not None:
        return TypeExpr.tuple_of(*(_type_expr_from_runtime_id(item) for item in positions))
    if (inner := parse_stream_type_id(type_id)) is not None:
        return TypeExpr.stream_of(_type_expr_from_runtime_id(inner))
    if (inner := parse_list_type_id(type_id)) is not None:
        return TypeExpr.list_of(
            TypeExpr.wildcard() if inner == "*" else _type_expr_from_runtime_id(inner)
        )
    if (inner := parse_asset_type_id(type_id)) is not None:
        return TypeExpr.asset_of(_type_expr_from_runtime_id(inner))
    return TypeExpr.concrete(type_id)


def _unify(
    input_id: str, expr: TypeExpr, received: TypeExpr, bindings: dict[str, TypeExpr]
) -> None:
    if expr.kind == "record" and expr.template_id:
        problem = expr.record_connection_problem(received)
        if problem is not None:
            raise TypeSolveError(f"input '{input_id}': {problem}")
        _unify(input_id, TypeExpr.variable(expr.template_id), received, bindings)
        return
    if expr.kind not in ("variable", "tuple", "stream", "list", "asset"):
        return
    type_id = received.runtime_type_id()
    if type_id is None:
        raise TypeSolveError(f"input '{input_id}': received type is not runtime-resolvable")
    if expr.kind == "tuple":
        if received.kind != "tuple" or len(received.elements) != len(expr.elements):
            raise TypeSolveError(
                f"input '{input_id}': expected a tuple of {len(expr.elements)} positions, "
                f"got {type_id}"
            )
        for item, position in zip(expr.elements, received.elements, strict=True):
            _unify(input_id, item, position, bindings)
        return
    if expr.kind == "variable":
        if expr.types and type_id not in expr.types:
            raise TypeSolveError(
                f"input '{input_id}': {type_id} is outside variable "
                f"'{expr.template_id}' allowlist {list(expr.types)}"
            )
        bound = bindings.get(expr.template_id)
        if bound is None:
            bindings[expr.template_id] = received
        elif bound.runtime_type_id() != type_id:
            raise TypeVariableConflict(
                input_id, expr.template_id, str(bound.runtime_type_id()), type_id
            )
        return
    if expr.kind == "stream":
        if received.kind != "stream" or received.element is None:
            raise TypeSolveError(f"input '{input_id}': expected a stream value, got {type_id}")
        assert expr.element is not None
        _unify(input_id, expr.element, received.element, bindings)
        return
    if expr.kind == "list":
        if received.kind != "list" or received.element is None:
            raise TypeSolveError(f"input '{input_id}': expected a list value, got {type_id}")
        assert expr.element is not None  # post_init invariant
        _unify(input_id, expr.element, received.element, bindings)
        return
    if expr.kind == "asset":
        if received.kind != "asset" or received.element is None:
            raise TypeSolveError(f"input '{input_id}': expected an asset value, got {type_id}")
        assert expr.element is not None  # post_init invariant
        _unify(input_id, expr.element, received.element, bindings)
    # concrete/union/wildcard bind nothing; edge compatibility is the
    # advisory document-time check, not this solver's job.
