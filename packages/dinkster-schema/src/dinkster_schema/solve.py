"""Runtime type-variable solving for generic nodes (DESIGN 3.13).

Generic combinators (MakeList, ConcatLists, ...) declare template variables
(``TypeExpr.variable("T")``) in their interfaces. Elaboration is pure over
document state (see elaborate.py). Graph validation otherwise stays per-edge
advisory, but region interfaces use the same solver to prove a generic body
output concrete from its linked, runtime-resolvable input declarations.

At execution, variables resolve *per invocation, at the worker boundary*,
from the authoritative runtime type ids on the input envelopes:

    bind_type_variables(elaborated inputs, input value type ids) -> bindings
    resolved_type_id(output TypeExpr, bindings) -> runtime type id

Unification is structural through the ``list``, ``asset``, and ``stream``
constructors: an input declared ``list<T>`` receiving a ``list<core.int>``
value binds T=core.int, and the same rule applies to typed assets and streams.
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

from .model import InputSpec, TypeExpr


class TypeSolveError(ValueError):
    """Input envelope types cannot bind this interface's template variables."""


def bind_type_variables(
    specs: Iterable[InputSpec], input_types: Mapping[str, str]
) -> dict[str, str]:
    """Solve template-variable bindings from runtime input type ids.

    specs is the *elaborated* input interface (family members flattened);
    input_types maps input id -> the received value's runtime type id.
    Callers must exclude absent inputs: ``core.absent`` stands for a missing
    value and can never bind a variable. Inputs with no received value (an
    unfed optional) bind nothing.
    """
    bindings: dict[str, str] = {}
    for spec in specs:
        type_id = input_types.get(spec.id)
        if type_id is not None:
            _unify(spec.id, spec.type, type_id, bindings)
    return bindings


def resolved_type_id(expr: TypeExpr, bindings: Mapping[str, str]) -> str | None:
    """The runtime type id of an expression under bindings, else None.

    The bindings-free case is ``TypeExpr.runtime_type_id()``; this is the
    same bridge with solved variables substituted.
    """
    resolved = resolved_type_expr(expr, bindings)
    return None if resolved is None else resolved.runtime_type_id()


def resolved_type_expr(expr: TypeExpr, bindings: Mapping[str, str]) -> TypeExpr | None:
    """A fully concrete expression under bindings, else None."""
    if expr.kind == "concrete":
        return expr
    if expr.kind == "variable":
        type_id = bindings.get(expr.template_id)
        return None if type_id is None else _type_expr_from_runtime_id(type_id)
    if expr.kind == "stream":
        assert expr.element is not None
        inner = resolved_type_expr(expr.element, bindings)
        return None if inner is None else TypeExpr.stream_of(inner)
    if expr.kind == "list":
        assert expr.element is not None  # post_init invariant
        inner = resolved_type_expr(expr.element, bindings)
        return None if inner is None else TypeExpr.list_of(inner)
    if expr.kind == "asset":
        assert expr.element is not None  # post_init invariant
        inner = resolved_type_expr(expr.element, bindings)
        return None if inner is None else TypeExpr.asset_of(inner)
    return None


def _type_expr_from_runtime_id(type_id: str) -> TypeExpr:
    if (inner := parse_stream_type_id(type_id)) is not None:
        return TypeExpr.stream_of(_type_expr_from_runtime_id(inner))
    if (inner := parse_list_type_id(type_id)) is not None:
        return TypeExpr.list_of(_type_expr_from_runtime_id(inner))
    if (inner := parse_asset_type_id(type_id)) is not None:
        return TypeExpr.asset_of(_type_expr_from_runtime_id(inner))
    return TypeExpr.concrete(type_id)


def _unify(input_id: str, expr: TypeExpr, type_id: str, bindings: dict[str, str]) -> None:
    if expr.kind == "variable":
        if expr.types and type_id not in expr.types:
            raise TypeSolveError(
                f"input '{input_id}': {type_id} is outside variable "
                f"'{expr.template_id}' allowlist {list(expr.types)}"
            )
        bound = bindings.get(expr.template_id)
        if bound is None:
            bindings[expr.template_id] = type_id
        elif bound != type_id:
            raise TypeSolveError(
                f"input '{input_id}': variable '{expr.template_id}' already "
                f"bound to {bound}, got {type_id}"
            )
        return
    if expr.kind == "stream":
        inner = parse_stream_type_id(type_id)
        if inner is None:
            raise TypeSolveError(f"input '{input_id}': expected a stream value, got {type_id}")
        assert expr.element is not None
        _unify(input_id, expr.element, inner, bindings)
        return
    if expr.kind == "list":
        inner = parse_list_type_id(type_id)
        if inner is None:
            raise TypeSolveError(f"input '{input_id}': expected a list value, got {type_id}")
        assert expr.element is not None  # post_init invariant
        _unify(input_id, expr.element, inner, bindings)
        return
    if expr.kind == "asset":
        inner = parse_asset_type_id(type_id)
        if inner is None:
            raise TypeSolveError(f"input '{input_id}': expected an asset value, got {type_id}")
        assert expr.element is not None  # post_init invariant
        _unify(input_id, expr.element, inner, bindings)
    # concrete/union/wildcard bind nothing; edge compatibility is the
    # advisory document-time check, not this solver's job.
