"""One registry shape for every inference vocabulary.

Samplers, schedulers, codecs, and model families all need the same
thing ComfyUI never built: registration with namespaced ids, loud
collisions, and alias resolution - instead of hardcoded module lists
extended by monkey-patching (comfy/samplers.py KSAMPLER_NAMES,
supported_models.models @ b78cec87).

Ids follow the closed name grammar (dinkster_schema.names) and must be
namespaced (contain at least one ``.``), matching node-type ids -
``dinkster.euler``, ``res4lyf.res_2m``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Generic, Protocol, TypeVar

from dinkster_schema.names import canonical_name, validate_name


class Registrable(Protocol):
    """Anything with a namespaced id and optional legacy aliases."""

    @property
    def id(self) -> str: ...

    @property
    def aliases(self) -> tuple[str, ...]: ...


D = TypeVar("D", bound=Registrable)


class RegistryError(ValueError):
    """Invalid id or a registration collision - always a loud failure."""


def validate_registry_id(id_: str) -> None:
    """Raise RegistryError unless ``id_`` is grammar-valid and namespaced."""
    problem = validate_name(id_)
    if problem is not None:
        raise RegistryError(f"invalid id {id_!r}: {problem}")
    if "." not in id_:
        raise RegistryError(f"invalid id {id_!r}: must be namespaced (contain '.')")


class Registry(Generic[D]):
    """Insertion-ordered descriptor registry.

    Collisions raise instead of shadowing: two owners claiming one id
    (or one alias) is a composition error, never a silent override.
    Collision and lookup keys use :func:`dinkster_schema.names.canonical_name`,
    so ``dinkster.foo-bar`` and ``dinkster.foo_bar`` are ONE name here exactly
    as they are one identity everywhere else in the workspace grammar.
    Aliases resolve on lookup but never appear in ``ids()`` - they are
    legacy vocabulary, not identity (same stance as node-name aliases
    in dinkster-schema). ``ids()`` returns ids as registered.
    """

    def __init__(self) -> None:
        # canonical key -> descriptor / registered-id; _ids preserves
        # the registered spelling in insertion order.
        self._by_key: dict[str, D] = {}
        self._alias_to_key: dict[str, str] = {}
        self._ids: dict[str, str] = {}

    def register(self, descriptor: D) -> None:
        validate_registry_id(descriptor.id)
        for alias in descriptor.aliases:
            problem = validate_name(alias)
            if problem is not None:
                raise RegistryError(f"invalid alias {alias!r}: {problem}")
        # Check the whole (id, *aliases) set for canonical duplicates -
        # including against itself - before mutating anything.
        claims: dict[str, str] = {}
        for name in (descriptor.id, *descriptor.aliases):
            key = canonical_name(name)
            if key in claims:
                raise RegistryError(f"{name!r} duplicates {claims[key]!r} within one descriptor")
            taken = self._owner_of(key)
            if taken is not None:
                raise RegistryError(f"{name!r} already registered by {taken!r}")
            claims[key] = name
        id_key = canonical_name(descriptor.id)
        self._by_key[id_key] = descriptor
        self._ids[id_key] = descriptor.id
        for alias in descriptor.aliases:
            self._alias_to_key[canonical_name(alias)] = id_key

    def get(self, id_or_alias: str) -> D | None:
        key = canonical_name(id_or_alias)
        return self._by_key.get(self._alias_to_key.get(key, key))

    def ids(self) -> tuple[str, ...]:
        return tuple(self._ids.values())

    def __iter__(self) -> Iterator[D]:
        return iter(self._by_key.values())

    def __len__(self) -> int:
        return len(self._by_key)

    def _owner_of(self, key: str) -> str | None:
        if key in self._ids:
            return self._ids[key]
        alias_owner = self._alias_to_key.get(key)
        return self._ids[alias_owner] if alias_owner is not None else None


__all__ = [
    "Registrable",
    "Registry",
    "RegistryError",
    "validate_registry_id",
]
