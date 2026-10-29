"""Derive the registry declaration owned by a pack's inference entry."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import cast

from dinkster_schema import canonical_name
from dinkster_workers import PackProvides, load_manifest

_REGISTRY_BY_SURFACE = {
    "inference-families": "dinkster.model-families",
    "inference-samplers": "dinkster.samplers",
    "inference-schedulers": "dinkster.schedulers",
}
_REGISTRY_TABLE = re.compile(r"(?ms)^\[pack\.provides\.registry\][ \t]*\n.*?(?=^\[|\Z)")


class RegistryDeclarationError(Exception):
    """A pack's inference declaration could not be derived."""


def registry_providers(
    contributions: Iterable[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    """Return manifest registry ids for materialized inference contributions."""
    providers = {
        (registry, descriptor_id)
        for surface_id, descriptor_id in contributions
        if (registry := _REGISTRY_BY_SURFACE.get(canonical_name(surface_id))) is not None
    }
    return tuple(sorted(providers))


def undeclared_registry_providers(
    provides: PackProvides,
    contributions: Iterable[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    """Return materialized registry providers absent from the manifest."""
    declared = {
        (canonical_name(provider.registry), canonical_name(provider.id))
        for provider in provides.registry
    }
    return tuple(
        provider
        for provider in registry_providers(contributions)
        if (canonical_name(provider[0]), canonical_name(provider[1])) not in declared
    )


def registry_declaration_toml(contributions: Iterable[tuple[str, str]]) -> str:
    """Render the exact manifest table for materialized registry providers."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for registry, descriptor_id in registry_providers(contributions):
        grouped[registry].append(descriptor_id)
    lines = ["[pack.provides.registry]"]
    for registry, descriptor_ids in sorted(grouped.items()):
        values = ", ".join(f'"{descriptor_id}"' for descriptor_id in descriptor_ids)
        lines.append(f'"{registry}" = [{values}]')
    return "\n".join(lines)


def load_registry_contributions(pack_root: Path | str) -> tuple[Path, tuple[tuple[str, str], ...]]:
    """Load one pack's inference entry and return its registry contributions."""
    from dinkster_workers.catalog import read_catalog
    from dinkster_workers.doctor import prepare_catalog, render_text

    root = Path(pack_root).resolve()
    manifest_path = root / "dinkster-pack.toml" if root.is_dir() else root
    manifest = load_manifest(manifest_path)
    report = prepare_catalog(manifest_path)
    if not report.ok:
        raise RegistryDeclarationError(render_text(report))
    catalog = read_catalog(manifest)
    if catalog is None:
        raise RegistryDeclarationError("pack probe produced no runtime catalog")
    raw = cast("object", catalog.declarations.get("inferenceContributions", ()))
    if not isinstance(raw, list):
        raise RegistryDeclarationError("pack probe returned invalid inference contributions")
    contributions: list[tuple[str, str]] = []
    for item in cast("list[object]", raw):
        if not isinstance(item, dict):
            raise RegistryDeclarationError("pack probe returned invalid inference contributions")
        contribution = cast("dict[str, object]", item)
        surface_id = contribution.get("surface_id")
        descriptor_id = contribution.get("id")
        if not isinstance(surface_id, str) or not isinstance(descriptor_id, str):
            raise RegistryDeclarationError("pack probe returned invalid inference contributions")
        contributions.append((surface_id, descriptor_id))
    return manifest.path, tuple(contributions)


def write_registry_declaration(manifest_path: Path, declaration: str) -> None:
    """Replace or append the manifest's registry provider table."""
    source = manifest_path.read_text(encoding="utf-8")
    replacement = declaration + "\n\n"
    if _REGISTRY_TABLE.search(source):
        updated = _REGISTRY_TABLE.sub(replacement, source, count=1)
    else:
        updated = source.rstrip() + "\n\n" + replacement
    manifest_path.write_text(updated.rstrip() + "\n", encoding="utf-8")


__all__ = [
    "RegistryDeclarationError",
    "load_registry_contributions",
    "registry_declaration_toml",
    "registry_providers",
    "undeclared_registry_providers",
    "write_registry_declaration",
]
