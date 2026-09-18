"""List values: the one structured envelope (DESIGN 3.13).

A list is itself a ``Value`` - type id ``list<element_type_id>``, a
fingerprint derived from the ordered children's identity, and a payload
holding the child *envelopes*. Children stay full Values inside the
parent, so everything the engine/memory/cache layers learn from a scalar
envelope - resource references, residency, cost - remains discoverable
inside a list via recursive traversal (``iter_value_tree``,
``value_resource_ids``). A list must never hide a resource-bearing child:
pin safety, invalidate-then-release, placement, and accounting all read
through it.

The canonical list type-id grammar lives here (values own the type-id
namespace); dinkster-schema imports it so ``TypeExpr.list_of`` and runtime
type ids can never drift apart.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import cast

from .model import (
    RESOURCE_ID_META_KEY,
    RESOURCE_OWNER_META_KEY,
    RESOURCE_PRODUCER_ARM_META_KEY,
    RESOURCE_REFS_META_KEY,
    Value,
    ValueMeta,
    stable_hash,
)

LIST_TYPE_PREFIX = "list<"
LIST_TYPE_SUFFIX = ">"


def list_type_id(element_type_id: str) -> str:
    """The canonical type id of a list of ``element_type_id``."""
    return f"{LIST_TYPE_PREFIX}{element_type_id}{LIST_TYPE_SUFFIX}"


def parse_list_type_id(type_id: str) -> str | None:
    """The element type id if ``type_id`` is a canonical list id, else None.
    The single canonical parser - nothing else string-matches list ids."""
    if (
        type_id.startswith(LIST_TYPE_PREFIX)
        and type_id.endswith(LIST_TYPE_SUFFIX)
        and len(type_id) > len(LIST_TYPE_PREFIX) + len(LIST_TYPE_SUFFIX)
    ):
        return type_id[len(LIST_TYPE_PREFIX) : -len(LIST_TYPE_SUFFIX)]
    return None


@dataclass(frozen=True)
class ListPayload:
    """The payload of a list value: the child envelopes themselves.

    ``load()`` produces the plain Python list node code sees - children
    resolve lazily, so a list of encoded/cross-boundary values decodes
    element-by-element only when actually loaded."""

    children: tuple[Value, ...]
    transport: str = "list"

    def load(self) -> object:
        return [child.resolve() for child in self.children]


def list_fingerprint(children: Sequence[Value]) -> str:
    """Deterministic identity of an ordered list: domain-tagged hash of each
    child's (type id, fingerprint). Order matters; equal children in equal
    order key the same cache entries everywhere (hazard H4)."""
    parts: list[bytes] = [b"list"]
    for child in children:
        parts.append(child.type_id.encode("utf-8"))
        parts.append(child.fingerprint.encode("utf-8"))
    return stable_hash(parts)


LENGTH_META_KEY = "length"
"""Well-known ValueMeta entry on list values: element count, interrogable
without touching any payload."""


def make_list_value(element_type_id: str, children: Sequence[Value]) -> Value:
    """Build a list envelope from already-wrapped children.

    Wrapping machinery only (worker shims, boundary decode, cache
    rehydration) - node authors never construct envelopes (hazard H9).
    Children must already carry the element type: a mismatch here is a bug
    in wrapping code, not user input."""
    for child in children:
        if child.type_id != element_type_id:
            raise ValueError(f"list<{element_type_id}> child has type '{child.type_id}'")
    ordered = tuple(children)
    return Value(
        type_id=list_type_id(element_type_id),
        fingerprint=list_fingerprint(ordered),
        meta=ValueMeta({LENGTH_META_KEY: len(ordered)}),
        payload=ListPayload(ordered),
    )


def list_children(value: Value) -> tuple[Value, ...] | None:
    """The child envelopes if ``value`` is a list, else None."""
    payload = value.payload
    return payload.children if isinstance(payload, ListPayload) else None


def iter_value_tree(value: Value) -> Iterator[Value]:
    """The value and every descendant envelope, depth-first in list order.

    THE way to read envelope metadata that may live on children: residency
    (RESOURCES_META_KEY), cost (COST_META_KEY), resource references. Any
    consumer that reads only the top envelope will silently miss children."""
    yield value
    children = list_children(value)
    if children is not None:
        for child in children:
            yield from iter_value_tree(child)


def value_resource_ids(value: Value) -> tuple[str, ...]:
    """Every resource identity referenced anywhere in the value tree,
    deduplicated, in first-seen order. The pin/release/invalidation layers
    use this so a resource stub inside a list is held exactly like a
    top-level one."""
    seen: dict[str, None] = {}
    for node in iter_value_tree(value):
        for resource_id in _local_resource_ids(node):
            seen.setdefault(resource_id)
    return tuple(seen)


def _local_resource_ids(value: Value) -> tuple[str, ...]:
    resource_ids: list[str] = []
    rid = value.meta.get(RESOURCE_ID_META_KEY)
    if isinstance(rid, str):
        resource_ids.append(rid)
    refs = value.meta.get(RESOURCE_REFS_META_KEY)
    if isinstance(refs, (list, tuple)):
        resource_ids.extend(
            reference for reference in cast("Sequence[object]", refs) if isinstance(reference, str)
        )
    return tuple(resource_ids)


def value_resource_refs(value: Value) -> tuple[tuple[str, str | None], ...]:
    """Every (resource id, owner token) pair referenced anywhere in the
    value tree, deduplicated, in first-seen order. Owner is the envelope's
    RESOURCE_OWNER_META_KEY when the producer stamped one (a non-string
    entry is treated as unstamped, never coerced). Dispatch and cache
    admission read pairs, not bare ids: the owner token is what makes a
    resident reference routable and its liveness checkable."""
    seen: dict[tuple[str, str | None], None] = {}
    for node in iter_value_tree(value):
        owner = node.meta.get(RESOURCE_OWNER_META_KEY)
        owner_token = owner if isinstance(owner, str) else None
        for resource_id in _local_resource_ids(node):
            seen.setdefault((resource_id, owner_token))
    return tuple(seen)


def value_resource_provenance_refs(
    value: Value,
) -> tuple[tuple[str, object, object, bool], ...]:
    """Every resource reference with raw owner and producer-arm metadata.

    Raw objects are intentional: dispatch must distinguish an absent stamp
    from a present malformed one instead of normalizing both to None.
    """
    refs: list[tuple[str, object, object, bool]] = []
    for node in iter_value_tree(value):
        for resource_id in _local_resource_ids(node):
            refs.append(
                (
                    resource_id,
                    node.meta.get(RESOURCE_OWNER_META_KEY),
                    node.meta.get(RESOURCE_PRODUCER_ARM_META_KEY),
                    RESOURCE_PRODUCER_ARM_META_KEY in node.meta.entries,
                )
            )
    return tuple(refs)


def stamp_resource_producer_arm(value: Value, owner: str, arm: str) -> Value:
    """Stamp this session's unstamped resident envelopes, recursively.

    Foreign-owner envelopes and already stamped provenance are preserved
    exactly. Fingerprints do not change because both stamps are ephemeral
    routing metadata, not computation identity.
    """
    children = list_children(value)
    if children is not None:
        stamped = tuple(stamp_resource_producer_arm(child, owner, arm) for child in children)
        if stamped != children:
            payload = value.payload
            assert isinstance(payload, ListPayload)
            value = replace(value, payload=replace(payload, children=stamped))
    if (
        _local_resource_ids(value)
        and value.meta.get(RESOURCE_OWNER_META_KEY) == owner
        and RESOURCE_PRODUCER_ARM_META_KEY not in value.meta.entries
    ):
        entries = dict(value.meta.entries)
        entries[RESOURCE_PRODUCER_ARM_META_KEY] = arm
        value = replace(value, meta=ValueMeta(entries))
    return value
