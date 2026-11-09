"""Runtime type-variable solving at the worker boundary."""

import pytest
from dinkster_schema import (
    InputSpec,
    TypeExpr,
    TypeSolveError,
    bind_type_variables,
    resolved_type_id,
)

INT = TypeExpr.concrete("INT")
T = TypeExpr.variable("T")


def test_bind_flat_variable() -> None:
    specs = (InputSpec("a", T), InputSpec("b", T))
    bindings = bind_type_variables(specs, {"a": "INT", "b": "INT"})
    assert bindings == {"T": "INT"}


def test_bind_through_list_constructor() -> None:
    specs = (InputSpec("list", TypeExpr.list_of(T)),)
    bindings = bind_type_variables(specs, {"list": "list<list<INT>>"})
    assert bindings == {"T": "list<INT>"}


def test_bind_conflict_is_loud() -> None:
    specs = (InputSpec("a", T), InputSpec("b", T))
    with pytest.raises(TypeSolveError, match="already bound"):
        bind_type_variables(specs, {"a": "INT", "b": "STRING"})


def test_bind_respects_allowlist() -> None:
    constrained = TypeExpr.variable("N", allowed=("INT", "FLOAT"))
    with pytest.raises(TypeSolveError, match="allowlist"):
        bind_type_variables((InputSpec("a", constrained),), {"a": "STRING"})


def test_bind_allowlist_requires_exact_atom() -> None:
    constrained = TypeExpr.variable("T", allowed=("IMAGE", "MASK"))
    assert bind_type_variables((InputSpec("a", constrained),), {"a": "IMAGE"}) == {"T": "IMAGE"}
    with pytest.raises(TypeSolveError, match="allowlist"):
        bind_type_variables((InputSpec("a", constrained),), {"a": "CUSTOM_IMAGE"})


def test_bind_agreement_requires_exact_atom() -> None:
    specs = (InputSpec("a", T), InputSpec("b", T))
    with pytest.raises(TypeSolveError, match="already bound"):
        bind_type_variables(specs, {"a": "IMAGE", "b": "CUSTOM_IMAGE"})


def test_bind_does_not_conflate_image_and_mask() -> None:
    constrained = TypeExpr.variable("N", allowed=("MASK",))
    with pytest.raises(TypeSolveError, match="allowlist"):
        bind_type_variables((InputSpec("a", constrained),), {"a": "IMAGE"})
    specs = (InputSpec("a", T), InputSpec("b", T))
    with pytest.raises(TypeSolveError, match="already bound"):
        bind_type_variables(specs, {"a": "IMAGE", "b": "MASK"})


def test_bind_scalar_into_list_expression_is_loud() -> None:
    specs = (InputSpec("list", TypeExpr.list_of(T)),)
    with pytest.raises(TypeSolveError, match="expected a list"):
        bind_type_variables(specs, {"list": "INT"})


def test_bind_ignores_unfed_and_nonvariable_inputs() -> None:
    specs = (InputSpec("a", T), InputSpec("b", INT))
    assert bind_type_variables(specs, {"b": "INT"}) == {}


def test_resolved_type_id_substitutes_recursively() -> None:
    bindings = {"T": "INT"}
    assert resolved_type_id(T, bindings) == "INT"
    assert resolved_type_id(TypeExpr.list_of(T), bindings) == "list<INT>"
    assert resolved_type_id(TypeExpr.list_of(TypeExpr.list_of(T)), bindings) == "list<list<INT>>"
    assert resolved_type_id(TypeExpr.variable("U"), bindings) is None
    assert resolved_type_id(TypeExpr.wildcard(), bindings) is None
