"""Installed provider identities for ordinary VOID composition tests."""

from pathlib import Path

from dinkster.compose import PackSpec, _installed_pack_spec


def void_providers(in_process: bool, *, sampling: bool = False) -> dict[Path, PackSpec]:
    packages = [
        ("dinkster-latent", "dinkster_nodes_latent"),
        ("dinkster-conditioning", "dinkster_nodes_conditioning"),
    ]
    if sampling:
        packages.append(("dinkster-sampling", "dinkster_nodes_sampling"))
    specs = (
        _installed_pack_spec(identity, module, in_process=in_process)
        for identity, module in packages
    )
    return {Path(spec.manifest): spec for spec in specs}
