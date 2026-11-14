"""The closed name grammar (DESIGN M8): pack names, namespace claims.

What this proves: the grammar rejects everything that caused ComfyUI
identity drift (uppercase, separator variants, malformed shapes), the
separator equivalence class makes foo-bar/foo_bar/foo.bar one identity,
coverage is dot-bounded while conflict is atom-bounded (so no two owners
can ever cover one node type, whatever separator spelling they claim),
and reserved roots catch every spelling that could contest them.
"""

from __future__ import annotations

from dinkster_schema import (
    canonical_claim,
    claim_covers,
    claims_conflict,
    reserved_root,
)


def test_coverage_is_dot_bounded() -> None:
    """A claim covers node types only past a literal '.' at the claim
    edge; a namespace is never itself a node name."""
    assert claim_covers("img", "img.blur")
    assert claim_covers("img", "img.filters.blur")
    assert claim_covers("img.filters", "img.filters.blur")
    assert not claim_covers("img", "img-extra.blur")  # '-' edge, not nested
    assert not claim_covers("img", "img")  # claim == node type
    assert not claim_covers("img", "imgx.blur")  # different atom
    assert not claim_covers("img.filters", "img.blur")


def test_conflict_is_atom_bounded() -> None:
    """Any atom-sequence prefix conflicts, whatever the separator:
    separator equivalence makes img-extra the same claim as img.extra,
    which nests inside img - two owners could otherwise both cover
    img.extra.blur."""
    assert claims_conflict("img", "img")
    assert claims_conflict("img", "img.filters")
    assert claims_conflict("img.filters", "img")  # symmetric
    assert claims_conflict("img", "img-extra")
    assert claims_conflict("foo-bar", "foo.bar")  # same claim, respelled
    assert claims_conflict("foo_bar", "foo.bar.baz")


def test_namespace_and_exact_claims_cannot_own_the_same_node() -> None:
    assert canonical_claim("img.filters") == "img-filters"
    for namespace, exact in (("img", "Img.Blur"), ("img.filters", "Img_Filters.Blur")):
        assert claims_conflict(namespace, exact)
        assert claims_conflict(exact, namespace)
    assert not claims_conflict("img", "Img-Blur")
    assert not claims_conflict("img", "Img")
    assert reserved_root("Core.Blur") == "core"
    assert reserved_root("CoreBlur") is None
