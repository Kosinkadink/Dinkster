"""Compose Wan fixtures with their installed, versioned helper providers."""

from pathlib import Path

from dinkster_api.v1 import TypeRegistry
from dinkster_nodes_wan import register_types

from dinkster.compose import (
    PackInfo,
    PackSpec,
    ServingComposer,
    _installed_pack_spec,  # pyright: ignore[reportPrivateUsage]
)


def register_wan_types(registry: TypeRegistry) -> None:
    with registry.registering_pack(
        "dinkster-wan", ("WAN_CAMERA_EMBEDDING", "TRACKS"), allow_bare=True
    ):
        register_types(registry)


async def compose_wan_packs(
    composer: ServingComposer,
    native: Path,
    *manifests: Path,
    in_process: bool,
    worker_group: str,
) -> None:
    providers = tuple(
        _installed_pack_spec(identity, module, in_process=in_process)
        for identity, module in (
            ("dinkster-latent", "dinkster_nodes_latent"),
            ("dinkster-conditioning", "dinkster_nodes_conditioning"),
            ("dinkster-models", "dinkster_nodes_models"),
            ("dinkster-vision-sam3", "dinkster_nodes_vision_sam3"),
        )
    )
    specs = (
        *providers,
        PackSpec(
            native, trust_reserved=True, packs={"dinkster-wan": PackInfo("Wan", version="0.0.1")}
        ),
        *(PackSpec(manifest, trust_reserved=True) for manifest in manifests),
    )
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
