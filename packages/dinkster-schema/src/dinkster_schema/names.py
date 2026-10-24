"""Shared name grammar, re-exported for schema and pack consumers."""

from dinkster_values.names import (
    NAME_MAX,
    RESERVED_NAMESPACES,
    canonical_name,
    claim_covers,
    claims_conflict,
    reserved_root,
    validate_name,
)

__all__ = [
    "NAME_MAX",
    "RESERVED_NAMESPACES",
    "canonical_name",
    "claim_covers",
    "claims_conflict",
    "reserved_root",
    "validate_name",
]
