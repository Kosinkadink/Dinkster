"""Input type-admission and asset-coercion policy.

The ONE resolution of the exceptional hard type boundary and of "does this
runtime value fit that input, and how?" Graph validation, the engine, and
workers consume this policy so diagnostics and execution cannot drift.

Most Dinkster type compatibility remains deliberately advisory. ``core.combo``
is the scoped exception: once ordinary TypeExpr acceptance and a valid asset
coercion are exhausted, an incompatible runtime type involving core.combo
(including inside list/asset shapes) is a hard admission error. Wildcards
and explicit unions containing the arriving type accept normally before
this predicate is consulted.

The closed plan vocabulary, exactly the pinned contract:

- ``decode``: ``asset<T> -> T`` (or ``asset<list<T>> -> list<T>``) via the
  decode provider registered for the target.
- ``lift``:   ``list<asset<T>> -> list<T>``, the one generic elementwise
  rule, using the same per-element decoder.
- ``merge``:  ``list<asset<T>> -> T`` for a scalar-T destination, via the
  per-element decoder plus T's registered batch-merge provider - the
  explicit mechanism for internally-batched types (user amendment
  2026-07-26); merge fires ONLY as the terminal step of an asset
  coercion, never for plain list values.

EXACTLY ONE coercion step: targets are never themselves asset-typed
(TypeRegistry.register_asset_decoder enforces it at registration), and no
other list/scalar bridging exists (DESIGN 3.13 - structural cardinality
stays authoritative). A destination that accepts the asset type AS-IS
(declared ``TypeExpr.asset_of``, wildcard, matching variable) never
coerces: the ref itself is the value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from dinkster_values import (
    CORE_COMBO,
    list_type_id,
    parse_asset_type_id,
    parse_list_type_id,
    runtime_type_atom,
)

from .model import TypeExpr

__all__ = [
    "AssetCoercionPlan",
    "CoercionProviders",
    "TypeEquivalenceProvider",
    "combo_type_mismatch_is_error",
    "plan_asset_coercion",
    "plan_type_equivalence",
]


def _expr_mentions_combo(expr: TypeExpr) -> bool:
    if expr.kind in ("concrete", "union", "variable"):
        return CORE_COMBO in expr.types
    if expr.kind in ("list", "asset", "stream"):
        assert expr.element is not None
        return _expr_mentions_combo(expr.element)
    return False


def combo_type_mismatch_is_error(source_type_id: str, expected: TypeExpr) -> bool:
    """Whether an incompatible source/destination pair crosses the hard
    core.combo boundary.

    Callers first honor ``expected.accepts_concrete(source_type_id)`` and
    ``plan_asset_coercion``. This function only classifies a mismatch after
    those normal acceptance paths fail. Peeling the runtime atom makes the
    exception apply recursively to list<core.combo>, asset<core.combo>, and
    stream<core.combo>.
    """
    return runtime_type_atom(source_type_id) == CORE_COMBO or _expr_mentions_combo(expected)


@runtime_checkable
class CoercionProviders(Protocol):
    """What a plan consumer needs from a type registry to check that a
    plan's providers are actually registered (TypeRegistry satisfies it).
    Pure document tooling without a registry validates plans structurally
    only."""

    def asset_decoder_for(self, target_type_id: str) -> object | None: ...

    def batch_merge_for(self, type_id: str) -> object | None: ...


@runtime_checkable
class TypeEquivalenceProvider(Protocol):
    """Registry surface needed to plan an explicit atom equivalence."""

    def equivalent_type(self, type_id: str) -> str | None: ...


def plan_type_equivalence(
    source_type_id: str,
    expected: TypeExpr,
    provider: TypeEquivalenceProvider,
) -> str | None:
    """The equivalent atom accepted by ``expected``, or None.

    Exact acceptance always wins and keeps the producer's spelling. The
    provider is pairwise, so an unaccepted source has at most one candidate.
    """
    if expected.accepts_concrete(source_type_id):
        return None
    counterpart = provider.equivalent_type(source_type_id)
    if counterpart is not None and expected.accepts_concrete(counterpart):
        return counterpart
    return None


@dataclass(frozen=True)
class AssetCoercionPlan:
    """One planned coercion step from an asset-typed value into an input.

    ``target_type_id`` keys the decode-provider lookup (``T`` or
    ``list<T>``); ``merge_type_id`` is set only for merge plans and keys
    the batch-merge lookup (always an atom). ``result_type_id`` is what
    the coerced value will be stamped as."""

    kind: Literal["decode", "lift", "merge"]
    target_type_id: str
    result_type_id: str
    merge_type_id: str | None = None

    def missing_providers(self, providers: CoercionProviders) -> tuple[str, ...]:
        """Human-readable descriptions of every provider this plan needs
        that ``providers`` does not have - empty means executable."""
        missing: list[str] = []
        if providers.asset_decoder_for(self.target_type_id) is None:
            missing.append(f"asset decoder for target '{self.target_type_id}'")
        if self.merge_type_id is not None:
            if providers.batch_merge_for(self.merge_type_id) is None:
                missing.append(f"batch merge for type '{self.merge_type_id}'")
        return tuple(missing)


def plan_asset_coercion(source_type_id: str, expected: TypeExpr) -> AssetCoercionPlan | None:
    """The coercion step that fits ``source_type_id`` into ``expected``,
    or None when none exists (which for an asset-typed source the caller
    reports; for anything else it just means "not an asset question").

    Callers must check ``expected.accepts_concrete(source_type_id)``
    FIRST: a destination that takes the asset as-is wins over any decode
    (declared asset inputs receive refs, wildcards preview refs)."""
    target = parse_asset_type_id(source_type_id)
    if target is not None:
        # asset<T> -> T, including asset<list<T>> -> list<T> (the single
        # asset's own decoder targets the list type).
        if expected.accepts_concrete(target):
            return AssetCoercionPlan(kind="decode", target_type_id=target, result_type_id=target)
        return None
    element = parse_list_type_id(source_type_id)
    if element is None:
        return None
    target = parse_asset_type_id(element)
    if target is None:
        return None
    # list<asset<T>>: the destination's declared type chooses the outcome.
    lifted = list_type_id(target)
    if expected.accepts_concrete(lifted):
        return AssetCoercionPlan(kind="lift", target_type_id=target, result_type_id=lifted)
    if (
        expected.cardinality() == "scalar"
        and parse_list_type_id(target) is None
        and expected.accepts_concrete(target)
    ):
        # Merge-to-one-batch: scalar-T destination, atoms only (a list
        # decode target never merges - list-of-lists does not batch).
        return AssetCoercionPlan(
            kind="merge",
            target_type_id=target,
            result_type_id=target,
            merge_type_id=target,
        )
    return None
