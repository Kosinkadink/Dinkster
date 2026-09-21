"""Create the local state consumed by the default launcher."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dinkster_assets import MountDef, dump_mounts, load_mounts, load_output_mount

from .installer import Installer


def default_roots() -> tuple[Path, Path]:
    home = Path(os.environ.get("DINKSTER_HOME", Path.home() / ".dinkster")).expanduser()
    return home / "library", home / "packs"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dinkster setup",
        description="Prepare the local library and managed pack installation",
    )
    parser.parse_args(argv)
    library_root, install_root = default_roots()
    library_root.mkdir(parents=True, exist_ok=True)
    output_root = library_root / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    mounts_path = library_root / "mounts.toml"
    mounts = list(load_mounts(mounts_path))
    default_output_index = next(
        (
            index
            for index, mount in enumerate(mounts)
            if mount.id == "output" or mount.path == output_root
        ),
        None,
    )
    if default_output_index is None:
        mounts.append(MountDef(id="output", path=output_root, mode="readwrite"))
        default_output_index = len(mounts) - 1
    elif mounts[default_output_index].mode != "readwrite":
        mount = mounts[default_output_index]
        mounts[default_output_index] = MountDef(
            id=mount.id,
            path=mount.path,
            mode="readwrite",
            priority=mount.priority,
        )
    default_output_mount = mounts[default_output_index].id
    output_mount = load_output_mount(mounts_path) or default_output_mount
    mounts_path.write_text(dump_mounts(mounts, output_mount=output_mount), "utf-8")
    Installer(install_root)
    print(f"Library: {library_root}")
    print(f"Packs: {install_root}")
    print("Setup complete. Run `dinkster` to open the editor.")
    return 0
