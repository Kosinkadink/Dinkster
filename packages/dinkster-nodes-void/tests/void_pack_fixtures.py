"""Installed provider identities for ordinary VOID composition tests."""

from dataclasses import replace
from pathlib import Path

from dinkster.compose import PackSpec, default_pack_spec


def void_providers(in_process: bool, *, sampling: bool = False) -> dict[Path, PackSpec]:
    packages = [
        "dinkster-latent",
        "dinkster-conditioning",
        "dinkster-video",
    ]
    if sampling:
        packages.append("dinkster-sampling")
    specs = (replace(default_pack_spec(identity), in_process=in_process) for identity in packages)
    return {Path(spec.manifest): spec for spec in specs}
