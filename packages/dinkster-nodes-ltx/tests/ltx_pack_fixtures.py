"""Compose LTX fixtures with their installed, versioned helper providers."""

from pathlib import Path

from dinkster.compose import (
    PackSpec,
    ServingComposer,
    _installed_pack_spec,  # pyright: ignore[reportPrivateUsage]
)


async def compose_ltx_packs(
    composer: ServingComposer,
    *manifests: Path,
    in_process: bool,
    worker_group: str | None = None,
) -> None:
    providers = tuple(
        _installed_pack_spec(identity, module, in_process=in_process)
        for identity, module in (
            ("dinkster-latent", "dinkster_nodes_latent"),
            ("dinkster-conditioning", "dinkster_nodes_conditioning"),
            ("dinkster-models", "dinkster_nodes_models"),
            ("dinkster-llm", "dinkster_nodes_llm"),
            ("dinkster-video", "dinkster_nodes_video"),
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
