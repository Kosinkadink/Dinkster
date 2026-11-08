"""Installed provider identities for ordinary MiniMax composition tests."""

from dataclasses import replace
from pathlib import Path

from dinkster.compose import PackSpec, default_pack_spec


def conditioning_providers(in_process: bool) -> dict[Path, PackSpec]:
    return {
        Path(spec.manifest): spec
        for spec in (
            replace(default_pack_spec("dinkster-latent"), in_process=in_process),
            replace(default_pack_spec("dinkster-conditioning"), in_process=in_process),
        )
    }
