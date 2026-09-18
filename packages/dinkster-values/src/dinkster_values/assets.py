"""The ``asset<T>`` type-id constructor (typed assets, joint contract
2026-07-26).

The closed runtime type-id grammar is
``atom | list<id> | asset<id> | stream<id>``. An ``asset<T>`` value is the SAME
envelope shape as the untyped ``dinkster.asset`` atom - an AssetRef wrapped
by the base asset codec, fingerprinted by its content digest - stamped
with the parametric id so the type system knows what the asset decodes
to. Bare ``dinkster.asset`` remains the untyped atom for assets with no
single decode target (checkpoint-style multi-output loads).

The canonical grammar lives here (values own the type-id namespace),
using the constructor parsers owned by lists.py, assets.py, and streams.py;
dinkster-schema imports them so expression-level types and runtime type ids can
never drift apart.
"""

from __future__ import annotations

from .lists import parse_list_type_id
from .streams import parse_stream_type_id

ASSET_TYPE_PREFIX = "asset<"
ASSET_TYPE_SUFFIX = ">"

ASSET_BASE_TYPE = "dinkster.asset"
"""The registered atom whose TypeSpec backs every ``asset<...>`` id.

Defined here rather than in dinkster-assets (which depends on this package)
so TypeRegistry can resolve parametric asset ids without importing
upward; dinkster-assets imports this constant as its ASSET_TYPE."""


def asset_type_id(element_type_id: str) -> str:
    """The canonical type id of an asset that decodes to ``element_type_id``."""
    return f"{ASSET_TYPE_PREFIX}{element_type_id}{ASSET_TYPE_SUFFIX}"


def parse_asset_type_id(type_id: str) -> str | None:
    """The decode-target type id if ``type_id`` is a canonical asset id,
    else None. The single canonical parser - nothing else string-matches
    asset ids."""
    if (
        type_id.startswith(ASSET_TYPE_PREFIX)
        and type_id.endswith(ASSET_TYPE_SUFFIX)
        and len(type_id) > len(ASSET_TYPE_PREFIX) + len(ASSET_TYPE_SUFFIX)
    ):
        return type_id[len(ASSET_TYPE_PREFIX) : -len(ASSET_TYPE_SUFFIX)]
    return None


def runtime_type_atom(type_id: str) -> str | None:
    """The concrete atom inside a runtime type id, or None when malformed.

    THE strict validator of the whole closed grammar
    (``atom | list<id> | asset<id> | stream<id>``): peels ``list<...>``,
    ``asset<...>`` and ``stream<...>`` layers recursively and returns the
    innermost atom only when it is a real atom - non-empty, no angle brackets.
    Every malformed spelling (unbalanced brackets, empty payloads, trailing
    garbage like ``list<a>>``) fails the final atom check, because the
    prefix/suffix parsers strip exactly one balanced constructor layer.
    Consumers with a registry at hand then check the atom is registered;
    this function alone checks shape."""
    atom = type_id
    while True:
        inner = parse_list_type_id(atom)
        if inner is None:
            inner = parse_asset_type_id(atom)
        if inner is None:
            inner = parse_stream_type_id(atom)
        if inner is None:
            break
        atom = inner
    if not atom or "<" in atom or ">" in atom:
        return None
    return atom
