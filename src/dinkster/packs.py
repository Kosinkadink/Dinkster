"""Pack provenance glue: manifest presentation -> server pack table.

Umbrella-owned like the rest of the wiring: dinkster-workers knows manifests,
dinkster-server knows its /api/nodes packs table, and neither imports the
other. Composition code building a multi-pack server collects, per loaded
worker, ``(worker.pack, pack_info_from_manifest(manifest))`` for the packs
table and ``{node_type: worker.pack for node_type in worker.schemas}`` for
per-node attribution - attribution always comes from the host-side loading
record, never from a schema's own claims.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from dinkster_server import (
    PackBlueprintAsset,
    PackDocAsset,
    PackDocPageAsset,
    PackDocsAsset,
    PackIconAsset,
    PackInfo,
    PackLocaleCatalogAsset,
    PackTemplateAsset,
)
from dinkster_workers import (
    PackBlueprint,
    PackManifest,
    PackPresentation,
    PackTemplate,
)

__all__ = [
    "blueprint_assets",
    "pack_info_from_manifest",
    "pack_info_from_presentation",
    "template_assets",
]


def _docs_asset(manifest: PackManifest) -> PackDocsAsset | None:
    """Manifest docs -> immutable server records, preserving digested bytes."""
    if manifest.docs is None:
        return None
    assets = {
        asset.source: PackDocAsset(
            source=asset.source,
            digest=asset.digest,
            media_type=asset.media_type,
            data=asset.data,
        )
        for asset in manifest.docs.assets
    }
    return PackDocsAsset(
        default_locale=manifest.docs.default_locale,
        pages=tuple(
            PackDocPageAsset(
                kind=page.kind,
                id=page.id,
                locale=page.locale,
                title=page.title,
                summary=page.summary,
                schema_version=page.schema_version,
                digest=page.digest,
                assets=tuple(assets[asset.source] for asset in page.assets),
                order=page.order,
                tags=page.tags,
                guide_kind=page.guide_kind,
                data=page.data,
            )
            for page in manifest.docs.pages
        ),
    )


def blueprint_assets(
    blueprints: Sequence[PackBlueprint],
) -> tuple[PackBlueprintAsset, ...]:
    """Manifest blueprints -> server assets, bytes carried verbatim: the
    manifest validator already read, parsed, and digested the files, so
    whoever serves a blueprint serves exactly what was digested."""
    return tuple(
        PackBlueprintAsset(
            id=blueprint.id,
            name=blueprint.name,
            digest=blueprint.digest,
            description=blueprint.description,
            tags=blueprint.tags,
            boundary_inputs=blueprint.boundary_inputs,
            boundary_outputs=blueprint.boundary_outputs,
            data=blueprint.data,
        )
        for blueprint in blueprints
    )


def template_assets(
    templates: Sequence[PackTemplate],
) -> tuple[PackTemplateAsset, ...]:
    """Manifest templates -> server assets, bytes carried verbatim: the
    manifest validator already read, parsed, and digested the files (and
    checked asset references against the pack's surviving declarations),
    so whoever serves a template serves exactly what was digested."""
    return tuple(
        PackTemplateAsset(
            id=template.id,
            name=template.name,
            digest=template.digest,
            description=template.description,
            tags=template.tags,
            family=template.family,
            models=template.models,
            assets=template.assets,
            thumbnail=(
                PackIconAsset(
                    digest=template.thumbnail.digest,
                    media_type=template.thumbnail.media_type,
                    data=template.thumbnail.data,
                )
                if template.thumbnail is not None
                else None
            ),
            data=template.data,
        )
        for template in templates
    )


def pack_info_from_presentation(
    presentation: PackPresentation | None, fallback_name: str
) -> PackInfo:
    """The canonical presentation -> packs-table conversion: declared
    fields when present, otherwise just the fallback display name."""
    if presentation is None:
        return PackInfo(display_name=fallback_name)
    icon = None
    if presentation.icon is not None:
        # The manifest validator already read, sniffed, and digested the
        # bytes; the server asset carries them verbatim so what is served
        # is exactly what was digested.
        icon = PackIconAsset(
            digest=presentation.icon.digest,
            media_type=presentation.icon.media_type,
            data=presentation.icon.data,
        )
    return PackInfo(
        display_name=presentation.display_name or fallback_name,
        abbr=presentation.abbr,
        mark=presentation.mark,
        color=presentation.color,
        icon=icon,
    )


def pack_info_from_manifest(manifest: PackManifest) -> PackInfo:
    """Manifest -> packs-table conversion: the pack's declared presentation
    (falling back to the pack name) plus its shipped blueprints, templates,
    and declared assets."""
    info = pack_info_from_presentation(manifest.presentation, manifest.name)
    if manifest.blueprints:
        info = replace(info, blueprints=blueprint_assets(manifest.blueprints))
    if manifest.templates:
        info = replace(info, templates=template_assets(manifest.templates))
    if manifest.docs is not None:
        info = replace(info, docs=_docs_asset(manifest))
    if manifest.locale_catalogs:
        info = replace(
            info,
            locale_catalogs=tuple(
                PackLocaleCatalogAsset(
                    locale=catalog.locale,
                    digest=catalog.digest,
                    node_references=catalog.node_references,
                    data=catalog.data,
                )
                for catalog in manifest.locale_catalogs
            ),
        )
    if manifest.assets:
        # Already validated DeclaredAssets - the server carries them
        # verbatim as descriptors; bytes never ride the packs table.
        info = replace(info, assets=manifest.assets)
    if manifest.comfy_aliases is not None:
        info = replace(info, comfy_aliases=manifest.comfy_aliases)
    if manifest.comfy_groups is not None:
        info = replace(info, comfy_groups=manifest.comfy_groups)
    return info
