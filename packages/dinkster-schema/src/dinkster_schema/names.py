"""The closed name grammar: pack names, publisher ids, namespace claims.

One grammar for every registry-adjacent identifier, closed
before the registry exists so no corpus of nonconforming names can
accumulate (H7). The rules answer two ComfyUI failures - capitalization
and separator drift making one pack look like two, and node-name
collisions arbitrated by load order on user machines:

- **Lowercase only.** Uppercase is rejected by the grammar, never
  normalized: "first claimant's casing becomes canonical" drift cannot
  exist because uppercase never exists.
- **Separators are one equivalence class.** ``-``, ``_``, and ``.`` split
  a name into *atoms*, and identity is the atom sequence: ``foo-bar``,
  ``foo_bar``, and ``foo.bar`` are one name (one pack, one claim).
  :func:`canonical_name` is the uniqueness key.
- **Coverage is dot-bounded; conflict is atom-bounded.** A namespace
  claim covers a node type when the claim's atoms prefix the node type's
  atoms *and* the node type has a literal ``.`` at the claim edge: claim
  ``img`` covers ``img.blur`` but never ``img-extra.blur``. Claim-vs-claim
  conflict is deliberately coarser - any atom-sequence prefix conflicts,
  whatever the separator - because separator equivalence makes
  ``img-extra`` and ``img.extra`` ONE claim, and ``img.extra`` nests
  inside ``img``; letting ``img`` and ``img-extra`` coexist across owners
  would let two owners cover the node type ``img.extra.blur``. Conflict
  conservative, coverage precise: ``img``'s owner is the only one who
  *may* claim ``img-extra``, but must claim it to cover its nodes.

Node *type* ids themselves are not forced into this grammar - compat
translation legitimately preserves v1 casing in later segments
(``comfy.<pack>.<V1Name>``). Only the claim prefix is grammar-bound;
coverage compares atoms case-insensitively so a lowercase claim governs
regardless of what a legacy tail looks like.

Reserved namespaces (``std``, ``comfy``, ``core``, ``dinkster``) are data
here, policy elsewhere: the public registry never grants them, and local
composition requires explicit host trust to compose a pack claiming one
(the host's spec is the local analog of the registry's grant table).
Core packs claim them through the ordinary manifest field - the same
declaration as everyone, no privileged shortcut (H5).
"""

from __future__ import annotations

import re

NAME_MAX = 64

RESERVED_NAMESPACES: frozenset[str] = frozenset({"std", "comfy", "core", "dinkster"})

_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*$")
_SEP_SPLIT_RE = re.compile(r"([-_.])")


def validate_name(name: str) -> str | None:
    """The problem with ``name`` under the closed grammar, or None.

    Shared by manifest loading, doctor, and (eventually) the registry so
    "loads locally" and "registrable" can never disagree on grammar.
    """
    if not name:
        return "must not be empty"
    if len(name) > NAME_MAX:
        return f"must be at most {NAME_MAX} characters, got {len(name)}"
    if not _NAME_RE.fullmatch(name):
        return (
            "must be lowercase ASCII letters/digits in segments separated "
            "by single '-', '_' or '.', starting with a letter "
            "(e.g. 'my-pack' or 'img.filters')"
        )
    return None


def _atoms(name: str) -> tuple[list[str], list[str]]:
    """Split into (atoms, separators); ``separators[i]`` follows
    ``atoms[i]``. Works on any id, not just grammar-valid names, so
    coverage checks can run against legacy node types verbatim."""
    tokens = _SEP_SPLIT_RE.split(name)
    return tokens[0::2], tokens[1::2]


def canonical_name(name: str) -> str:
    """The uniqueness key: atoms joined with ``-``. Two names with the
    same canonical form are one identity everywhere (pack table, claim
    ownership, publisher ids)."""
    atoms, _ = _atoms(name)
    return "-".join(atom.casefold() for atom in atoms)


def claim_covers(claim: str, node_type: str) -> bool:
    """Whether a namespace claim covers a node type id.

    The claim's atoms must prefix the node type's atoms (case-insensitive
    - legacy tails keep v1 casing), the node type must continue past the
    claim (a namespace is not a node name), and the separator in the node
    type at the claim edge must be a literal ``.`` - so ``img`` covers
    ``img.blur`` and ``img`` covers ``img-x`` never.
    """
    claim_atoms, _ = _atoms(claim)
    node_atoms, node_seps = _atoms(node_type)
    edge = len(claim_atoms)
    if len(node_atoms) <= edge:
        return False
    for claim_atom, node_atom in zip(claim_atoms, node_atoms[:edge], strict=True):
        if claim_atom.casefold() != node_atom.casefold():
            return False
    return node_seps[edge - 1] == "."


def claims_conflict(a: str, b: str) -> bool:
    """Whether two claims cannot be held by different owners: one atom
    sequence prefixes the other (equal counts as prefix), regardless of
    separators. ``img`` vs ``img.filters`` conflict, and so do ``img`` vs
    ``img-extra`` - separator equivalence makes ``img-extra`` the same
    claim as ``img.extra``, which nests inside ``img``. Coarser than
    coverage on purpose: exclusivity must hold under every separator
    spelling, or two owners could cover one node type."""
    a_atoms = [atom.casefold() for atom in _atoms(a)[0]]
    b_atoms = [atom.casefold() for atom in _atoms(b)[0]]
    if len(a_atoms) > len(b_atoms):
        a_atoms, b_atoms = b_atoms, a_atoms
    return b_atoms[: len(a_atoms)] == a_atoms


def reserved_root(claim: str) -> str | None:
    """The reserved namespace this claim is or nests inside, else None.
    ``std``, ``std.magic``, and ``std-extra`` all answer ``std`` (conflict
    is atom-bounded, so every spelling that could contest a reserved root
    is caught); ``standard`` answers None."""
    for root in RESERVED_NAMESPACES:
        if claim == root or claims_conflict(root, claim):
            return root
    return None


__all__ = [
    "NAME_MAX",
    "RESERVED_NAMESPACES",
    "canonical_name",
    "claim_covers",
    "claims_conflict",
    "reserved_root",
    "validate_name",
]
