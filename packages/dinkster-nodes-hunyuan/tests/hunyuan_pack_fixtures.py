"""Installed provider identities for ordinary Hunyuan composition tests."""

from dinkster.compose import (
    PackSpec,
    _installed_pack_spec,  # pyright: ignore[reportPrivateUsage]
)


def hunyuan_providers(in_process: bool) -> tuple[PackSpec, ...]:
    return tuple(
        _installed_pack_spec(identity, module, in_process=in_process)
        for identity, module in (
            ("dinkster-latent", "dinkster_nodes_latent"),
            ("dinkster-conditioning", "dinkster_nodes_conditioning"),
            ("dinkster-video", "dinkster_nodes_video"),
        )
    )
