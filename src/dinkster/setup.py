"""Create the local state consumed by the default launcher."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

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
    Installer(install_root)
    print(f"Library: {library_root}")
    print(f"Packs: {install_root}")
    print("Setup complete. Run `dinkster` to open the editor.")
    return 0
