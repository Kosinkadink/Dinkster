"""Compose geometry with its real image provider in ordinary host tests."""

import tomllib
from pathlib import Path

import dinkster_image
from dinkster_server import PackInfo

from dinkster.compose import PackSpec


def geometry_pack_specs(
    *fixtures: Path, in_process: bool, worker_group: str | None = None
) -> tuple[PackSpec, ...]:
    image = Path(dinkster_image.__file__).resolve().parents[2] / "dinkster-pack.toml"
    geometry = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
    manifests = (image, geometry, *fixtures)
    specs: list[PackSpec] = []
    for manifest in manifests:
        pack = tomllib.loads(manifest.read_text(encoding="utf-8"))["pack"]
        packs = (
            {pack["name"]: PackInfo(display_name=pack["name"], version=pack["version"])}
            if "version" in pack
            else None
        )
        specs.append(
            PackSpec(
                manifest,
                in_process=in_process,
                packs=packs,
                trust_reserved=True,
                worker_group=None if in_process else worker_group,
                group_manifests=manifests if worker_group and not in_process else (),
            )
        )
    return tuple(specs)
