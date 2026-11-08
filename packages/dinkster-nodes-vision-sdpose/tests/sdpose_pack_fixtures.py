"""Compose SDPose with the real providers in its declared dependency closure."""

import importlib
import tomllib
from pathlib import Path
from typing import cast

from dinkster_server import PackInfo

from dinkster.compose import PackSpec


def sdpose_pack_specs(*fixtures: Path, in_process: bool) -> tuple[PackSpec, ...]:
    providers = (
        "dinkster_latent",
        "dinkster_conditioning",
        "dinkster_qwen",
        "dinkster_misc_models",
        "dinkster_image",
    )
    manifests = tuple(
        Path(cast(str, importlib.import_module(name).__file__)).resolve().parents[2]
        / "dinkster-pack.toml"
        for name in providers
    ) + (Path(__file__).resolve().parents[1] / "dinkster-pack.toml", *fixtures)
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
                worker_group=None if in_process else "sdpose",
                group_manifests=() if in_process else manifests,
            )
        )
    return tuple(specs)
