"""Compose LTX fixtures with their installed, versioned helper providers."""

from dataclasses import replace
from pathlib import Path

from dinkster.compose import (
    PackSpec,
    ServingComposer,
    default_pack_spec,
)


async def compose_ltx_packs(
    composer: ServingComposer,
    *manifests: Path,
    in_process: bool,
    worker_group: str | None = None,
) -> None:
    providers = tuple(
        replace(default_pack_spec(identity), in_process=in_process)
        for identity in (
            "dinkster-latent",
            "dinkster-conditioning",
            "dinkster-models",
            "dinkster-llm",
            "dinkster-video",
        )
    )
    specs = (*providers, *(PackSpec(manifest, trust_reserved=True) for manifest in manifests))
    group = None if in_process else worker_group
    for spec in specs:
        await composer.add_pack(
            PackSpec(
                spec.manifest,
                trust_reserved=True,
                packs=spec.packs,
                in_process=in_process,
                worker_group=group,
                group_manifests=tuple(Path(item.manifest) for item in specs) if group else (),
            )
        )
