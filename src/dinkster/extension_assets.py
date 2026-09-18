"""Read declared installed-pack modules as data, never Python imports."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from dinkster_protocol.frontend_modules import FrontendModule, validate_frontend_modules
from dinkster_workers.manifest import PackManifest

MODULE_MAX_BYTES = 1024 * 1024


def read_module(manifest: PackManifest, module: FrontendModule) -> bytes:
    root = manifest.root.resolve(strict=True)
    path = (root / module.module).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("frontend module escapes its installed pack")
    with path.open("rb") as stream:
        data = stream.read(MODULE_MAX_BYTES + 1)
    if len(data) > MODULE_MAX_BYTES:
        raise ValueError("frontend module exceeds 1 MiB")
    data.decode("utf-8")
    if module.module_digest and module_digest(data) != module.module_digest:
        raise ValueError("frontend module bytes do not match the selected snapshot")
    return data


def module_digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def resolve_frontend_modules(manifest: PackManifest) -> tuple[FrontendModule, ...]:
    modules = manifest.extension.frontend_modules
    validate_frontend_modules(modules, pack=manifest.name)
    events = {event.name for event in manifest.extension.events}
    for module in modules:
        for contribution in module.contributions:
            if contribution.event is not None and contribution.event not in events:
                raise ValueError("frontend consumer must reference an event declared by its pack")
    return tuple(
        replace(module, module_digest=module_digest(read_module(manifest, module)))
        for module in modules
    )
