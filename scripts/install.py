"""Install the locked backend into the extracted release's own environment."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

TRAINING_REVISION = "5d332635c6bde7c4540783c1b01ec46f10108101"
TRAINING_REPOSITORY = "git+https://github.com/Kosinkadink/dinkster-training.git"
TRAINING_PACKAGES = tuple(
    f"{name} @ {TRAINING_REPOSITORY}@{TRAINING_REVISION}#subdirectory=packages/{name}"
    for name in ("dinkster-nodes-training", "dinkster-training-worker")
)


def install(root: Path, uv: str) -> Path:
    root = root.resolve()
    if not (root / "uv.lock").is_file() or not (root / "packages").is_dir():
        raise ValueError("extract the complete Dinkster backend archive before installing")
    environment = os.environ.copy()
    environment["UV_PROJECT_ENVIRONMENT"] = str(root / ".venv")
    subprocess.run(
        [uv, "sync", "--project", str(root), "--locked", "--no-dev", "--all-packages"],
        env=environment,
        check=True,
    )
    scripts = root / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    subprocess.run(
        [uv, "pip", "install", "--python", str(python), *TRAINING_PACKAGES],
        env=environment,
        check=True,
    )
    return scripts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", default="uv", help="uv executable (default: uv on PATH)")
    args = parser.parse_args()
    uv = shutil.which(args.uv)
    if uv is None:
        parser.error("install uv first: https://docs.astral.sh/uv/getting-started/installation/")
    scripts = install(Path(__file__).resolve().parent.parent, uv)
    print(f"Installed backend commands in {scripts}")
    print("Next: follow docs/install.md to install a pack and start the backend.")


if __name__ == "__main__":
    main()
