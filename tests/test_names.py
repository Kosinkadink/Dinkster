"""The closed name grammar (DESIGN M8): pack names, namespace claims.

What this proves: the grammar rejects everything that caused ComfyUI
identity drift (uppercase, separator variants, malformed shapes), the
separator equivalence class makes foo-bar/foo_bar/foo.bar one identity,
coverage is dot-bounded while conflict is atom-bounded (so no two owners
can ever cover one node type, whatever separator spelling they claim),
and reserved roots catch every spelling that could contest them.
"""

from __future__ import annotations

import pytest
from dinkster_schema import (
    NAME_MAX,
    RESERVED_NAMESPACES,
    canonical_name,
    claim_covers,
    claims_conflict,
    reserved_root,
    validate_name,
)


@pytest.mark.parametrize(
    "name",
    ["a", "img", "my-pack", "img.filters", "foo_bar", "a1", "v2.x", "a-b_c.d9"],
)
def test_valid_names(name: str) -> None:
    assert validate_name(name) is None


@pytest.mark.parametrize(
    "name",
    [
        "",  # empty
        "Img",  # uppercase is rejected, never normalized
        "IMG",
        "-img",  # leading separator
        "img-",  # trailing separator
        "img--x",  # repeated separator
        "img._x",  # adjacent separators
        "1img",  # must start with a letter
        "img extra",  # whitespace
        "img/extra",  # foreign separator
        "im\u00e4ge",  # non-ASCII
        "a" * (NAME_MAX + 1),  # too long
    ],
)
def test_invalid_names(name: str) -> None:
    assert validate_name(name) is not None


def test_max_length_boundary() -> None:
    assert validate_name("a" * NAME_MAX) is None


def test_canonical_collapses_separators() -> None:
    """-, _ and . are one equivalence class: one claim, one pack."""
    assert canonical_name("foo-bar") == "foo-bar"
    assert canonical_name("foo_bar") == "foo-bar"
    assert canonical_name("foo.bar") == "foo-bar"
    assert canonical_name("a.b_c-d") == "a-b-c-d"
    assert canonical_name("img") == "img"


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


def test_coverage_handles_legacy_casing_and_separators() -> None:
    """Node type ids are not grammar-bound: compat preserves v1 casing
    (comfy.<pack>.<V1Name>), and coverage compares atoms case-insensitively."""
    assert claim_covers("comfy", "comfy.My-Pack.LoadImage")
    assert claim_covers("comfy", "comfy.rgthree.Fast Groups")  # verbatim tail
    assert not claim_covers("comfy", "comfyui.LoadImage")


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


def test_siblings_do_not_conflict() -> None:
    assert not claims_conflict("img.filters", "img.masks")
    assert not claims_conflict("img", "image")  # atom identity, not string prefix
    assert not claims_conflict("img2", "img")
    assert not claims_conflict("impact-pack", "impact-subpack")


def test_reserved_roots() -> None:
    assert RESERVED_NAMESPACES == {"std", "comfy", "core", "dinkster"}
    assert reserved_root("std") == "std"
    assert reserved_root("std.magic") == "std"
    assert reserved_root("std-extra") == "std"  # every contesting spelling
    assert reserved_root("comfy.rgthree") == "comfy"
    assert reserved_root("dinkster-nodes-foundation") == "dinkster"
    assert reserved_root("standard") is None
    assert reserved_root("my-pack") is None
