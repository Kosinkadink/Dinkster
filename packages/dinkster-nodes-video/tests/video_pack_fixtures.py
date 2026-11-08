"""Install the video provider's codecs without transferring type ownership."""

from pathlib import Path

from dinkster_api.v1 import TypeRegistry
from dinkster_video import register_types

from dinkster.compose import PackInfo, PackSpec, ServingComposer


def register_video_types(registry: TypeRegistry) -> None:
    with registry.registering_pack(
        "dinkster-video", ("INTERP_MODEL", "OPTICAL_FLOW"), allow_bare=True
    ):
        register_types(registry)


async def compose_video_packs(
    composer: ServingComposer,
    native: Path,
    fixture: Path,
    *,
    in_process: bool,
    worker_group: str,
) -> None:
    for manifest in (native, fixture):
        await composer.add_pack(
            PackSpec(
                manifest,
                in_process=in_process,
                trust_reserved=True,
                packs={"dinkster-video": PackInfo("Video", version="0.0.1")}
                if manifest == native
                else None,
                worker_group=None if in_process else worker_group,
                group_manifests=() if in_process else (native, fixture),
            )
        )
