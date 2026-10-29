"""Create the local state consumed by the default launcher."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dinkster_assets import MountDef, dump_mounts, load_mounts, load_output_mount

from .installer import Installer
from .project_folders import ProjectFolders


def default_roots() -> tuple[Path, Path]:
    home = Path(os.environ.get("DINKSTER_HOME", Path.home() / ".dinkster")).expanduser()
    return home / "library", home / "packs"


def ensure_default_mounts(library_root: Path) -> None:
    """Initialize visible folders once without restoring or widening existing grants."""
    library_root.mkdir(parents=True, exist_ok=True)
    for name in ("output", "imports"):
        _ensure_default_mount(library_root, name)


def _ensure_default_mount(library_root: Path, name: str) -> None:
    initialized = library_root / f".{name}-initialized"
    if initialized.exists():
        return
    mounts_path = library_root / "mounts.toml"
    mounts = load_mounts(mounts_path)
    root = library_root / name
    output_mount = load_output_mount(mounts_path)
    existing = next((mount for mount in mounts if mount.id == name), None)
    if existing is None:
        existing = next((mount for mount in mounts if mount.path == root), None)
    if name == "output" and output_mount is not None and existing is None:
        initialized.touch()
        return
    project_path = library_root / "project-folders.json"
    projects = ProjectFolders(project_path) if project_path.exists() else None
    if existing is not None:
        if projects is not None:
            projects.initialize(tuple(mount.id for mount in mounts), shared_defaults=(existing.id,))
        initialized.touch()
        return
    root.mkdir(parents=True, exist_ok=True)
    text = dump_mounts(
        (*mounts, MountDef(id=name, path=root, mode="readwrite")),
        output_mount=name if name == "output" else output_mount,
    )
    temporary = mounts_path.with_name(f"{mounts_path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, mounts_path)
    finally:
        temporary.unlink(missing_ok=True)
    if projects is not None:
        projects.initialize(tuple(mount.id for mount in mounts) + (name,), shared_defaults=(name,))
    initialized.touch()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dinkster setup",
        description="Prepare the local library and managed pack installation",
    )
    parser.parse_args(argv)
    library_root, install_root = default_roots()
    ensure_default_mounts(library_root)
    Installer(install_root)
    print(f"Library: {library_root}")
    print(f"Packs: {install_root}")
    print("Setup complete. Run `dinkster` to open the editor.")
    return 0
