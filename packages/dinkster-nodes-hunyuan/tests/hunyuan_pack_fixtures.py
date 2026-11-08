"""Installed provider identities for ordinary Hunyuan composition tests."""

from dataclasses import replace

from dinkster.compose import PackSpec, default_pack_spec


def hunyuan_providers(in_process: bool) -> tuple[PackSpec, ...]:
    return tuple(
        replace(default_pack_spec(identity), in_process=in_process)
        for identity in (
            "dinkster-latent",
            "dinkster-conditioning",
            "dinkster-video",
        )
    )
