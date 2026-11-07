"""Installed provider identities for ordinary MiniMax composition tests."""

from pathlib import Path

from dinkster.compose import (
    PackSpec,
    _installed_pack_spec,  # pyright: ignore[reportPrivateUsage]
)


def conditioning_providers(in_process: bool) -> dict[Path, PackSpec]:
    return {
        Path(spec.manifest): spec
        for spec in (
            _installed_pack_spec("dinkster-latent", "dinkster_nodes_latent", in_process=in_process),
            _installed_pack_spec(
                "dinkster-conditioning", "dinkster_nodes_conditioning", in_process=in_process
            ),
        )
    }
