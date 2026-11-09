from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from dinkster_workers.launch import LaunchSpec, SubprocessLauncher


class SitecustomizeLauncher(SubprocessLauncher):
    def __init__(self, patch_root: Path) -> None:
        self._patch_root = patch_root

    async def launch(self, spec: LaunchSpec):
        environment = dict(spec.env)
        pythonpath = environment.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(self._patch_root), pythonpath) if part
        )
        return await super().launch(replace(spec, env=environment))


def worker_patch_launcher(tmp_path: Path, patch: str) -> SitecustomizeLauncher:
    patch_root = tmp_path / "worker-patch"
    patch_root.mkdir()
    (patch_root / "sitecustomize.py").write_text(patch, encoding="utf-8")
    return SitecustomizeLauncher(patch_root)
