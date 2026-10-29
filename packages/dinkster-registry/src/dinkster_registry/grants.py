"""The namespace grant table: manifests claim, the registry grants.

The single authority for who owns which namespace in the public corpus.
A manifest's ``[pack] namespaces`` is publisher-controlled input - a
claim - and nothing in it has corpus-wide consequence until this table
says so. Pack names go through the SAME table (claiming a pack name IS a
namespace claim), so there is no second, weaker path to squatting a name.

Exclusivity invariant: no two grants held by different publishers may
conflict under ``dinkster_schema.claims_conflict`` (atom-sequence prefix,
separator-blind). That is deliberately coarser than node-type coverage:
``img`` and ``img-extra`` cannot belong to different owners because
separator equivalence makes ``img-extra`` the same claim as ``img.extra``,
which nests inside ``img``. Nested grants for ONE publisher are legal -
the Impact-Pack/Impact-Subpack suite shape.

Reserved roots (``std``, ``comfy``, ``core``, ``dinkster``) are never
grantable here; core packs hold them through host trust locally, not
through the public registry (hazard H5).

Audit (who granted/transferred, when, why) is the server layer's job,
riding the same actor-attributed record shape as ``ReviewLog``; this
model enforces only what must never be representable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from dinkster_schema import canonical_name, claims_conflict, reserved_root, validate_name

from .model import RegistryError

ClaimStatus = Literal["granted", "grantable", "free", "denied-reserved", "denied-taken"]
"""Admission-facing answer for one claim by one publisher:

- ``granted``: this exact canonical claim is already this publisher's.
- ``grantable``: every conflicting grant belongs to this publisher (the
  claim nests in - or encloses - their own territory), so granting it
  needs no review: exclusivity already points at them.
- ``free``: conflicts with nothing; first claim of free namespace is the
  explicit-review exception.
- ``denied-reserved`` / ``denied-taken``: never grantable here / another
  publisher's territory.
"""


@dataclass(frozen=True)
class Grant:
    """One namespace grant: a canonical claim owned by a publisher."""

    claim: str
    publisher: str


class GrantTable:
    """All namespace grants, upholding cross-publisher exclusivity."""

    def __init__(self, grants: Iterable[Grant] = ()) -> None:
        self._grants: list[Grant] = []
        for grant in grants:
            self.grant(grant.claim, grant.publisher)

    def grants(self) -> tuple[Grant, ...]:
        return tuple(self._grants)

    def owner_of(self, claim: str) -> str | None:
        """The publisher holding exactly this claim (canonical match)."""
        canonical = canonical_name(claim)
        for grant in self._grants:
            if grant.claim == canonical:
                return grant.publisher
        return None

    def evaluate(self, claim: str, publisher: str) -> ClaimStatus:
        """What granting ``claim`` to ``publisher`` would mean.

        Assumes grammar-valid inputs (admission validates grammar first);
        grammar violations here are caller bugs and raise.
        """
        problem = validate_name(claim)
        if problem is not None:
            raise RegistryError(f"claim {claim!r} {problem}")
        if reserved_root(claim) is not None:
            return "denied-reserved"
        canonical = canonical_name(claim)
        publisher_canonical = canonical_name(publisher)
        conflicting = [g for g in self._grants if claims_conflict(g.claim, canonical)]
        for grant in conflicting:
            if grant.publisher != publisher_canonical:
                return "denied-taken"
        for grant in conflicting:
            if grant.claim == canonical:
                return "granted"
        return "grantable" if conflicting else "free"

    def grant(self, claim: str, publisher: str) -> Grant:
        """Record a grant; refuses reserved roots and other publishers'
        territory; idempotent for an already-held exact claim."""
        status = self.evaluate(claim, publisher)
        canonical = canonical_name(claim)
        publisher_canonical = canonical_name(publisher)
        if status == "denied-reserved":
            raise RegistryError(
                f"namespace {claim!r} is reserved ({reserved_root(claim)!r}) and "
                f"never granted by the public registry"
            )
        if status == "denied-taken":
            owners = sorted(
                {
                    g.publisher
                    for g in self._grants
                    if claims_conflict(g.claim, canonical) and g.publisher != publisher_canonical
                }
            )
            raise RegistryError(
                f"namespace {claim!r} conflicts with grants held by {', '.join(owners)}"
            )
        if status == "granted":
            for grant in self._grants:
                if grant.claim == canonical:
                    return grant
        grant = Grant(canonical, publisher_canonical)
        self._grants.append(grant)
        return grant

    def transfer(self, claim: str, to_publisher: str) -> Grant:
        """Reassign an exact grant to another publisher - a registry
        operation, never a manifest edit. Refused when the transfer would
        split a nested suite across owners (the exclusivity invariant
        must survive every mutation)."""
        canonical = canonical_name(claim)
        to_canonical = canonical_name(to_publisher)
        index = next(
            (i for i, g in enumerate(self._grants) if g.claim == canonical),
            None,
        )
        if index is None:
            raise RegistryError(f"no grant exists for namespace {claim!r}")
        replacement = Grant(canonical, to_canonical)
        proposed = [*self._grants]
        proposed[index] = replacement
        for i, a in enumerate(proposed):
            for b in proposed[i + 1 :]:
                if a.publisher != b.publisher and claims_conflict(a.claim, b.claim):
                    raise RegistryError(
                        f"transferring {claim!r} to {to_publisher!r} would split "
                        f"nested grants {a.claim!r} and {b.claim!r} across "
                        f"publishers; use transfer_suite on the suite root"
                    )
        self._grants = proposed
        return replacement

    def transfer_suite(self, root_claim: str, to_publisher: str) -> tuple[Grant, ...]:
        """Atomically reassign a grant and every grant of the same owner
        nested inside it - the only way a nested suite changes hands,
        since any partial transfer would split it across publishers."""
        canonical = canonical_name(root_claim)
        to_canonical = canonical_name(to_publisher)
        owner = self.owner_of(canonical)
        if owner is None:
            raise RegistryError(f"no grant exists for namespace {root_claim!r}")
        moved: list[Grant] = []
        proposed: list[Grant] = []
        for grant in self._grants:
            # Nested inside (or equal to) the root: the root's atoms prefix
            # the grant's atoms. Canonical forms join atoms with '-', so a
            # '-'-bounded string prefix is exactly the atom-prefix test.
            # Deliberately NOT claims_conflict, which is symmetric and would
            # silently move an ENCLOSING grant when given a leaf as root.
            nested = grant.claim == canonical or grant.claim.startswith(canonical + "-")
            if grant.publisher == owner and nested:
                replacement = Grant(grant.claim, to_canonical)
                moved.append(replacement)
                proposed.append(replacement)
            else:
                proposed.append(grant)
        for i, a in enumerate(proposed):
            for b in proposed[i + 1 :]:
                if a.publisher != b.publisher and claims_conflict(a.claim, b.claim):
                    raise RegistryError(
                        f"transferring the {root_claim!r} suite to {to_publisher!r} "
                        f"would still split {a.claim!r} and {b.claim!r} across "
                        f"publishers; transfer from the enclosing root instead"
                    )
        self._grants = proposed
        return tuple(moved)


__all__ = ["ClaimStatus", "Grant", "GrantTable"]
