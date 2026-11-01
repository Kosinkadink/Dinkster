"""Acquire the pinned Windows base interpreter with the SQLite WAL fix."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

# This source pin moves with the Windows base's SQLite override, not with uv's catalog.
ENVIRONMENTS_COMMIT = "e80a7240f6a6e835ffa724f43c36552f58e73b5e"
INPUTS = {
    "windows_sqlite.py": "9f681989e1928d979868f2acbbb5be88d587c1e6f7c97216b3e777a963208b95",
    "variants.toml": "6c13daa5b35c65c787c859d85bae44a2d463678269c54fcf4ac4141c3ba64db8",
}
MINIMUM_UV_VERSION = (0, 13, 6)
PINNED_PYTHON_VERSION = "3.14.10"


def require_uv_version(uv: str = "uv") -> None:
    executable = shutil.which(uv) or uv
    installed = subprocess.check_output([executable, "--version"], text=True, timeout=10).strip()
    match = re.match(r"uv (\d+)\.(\d+)\.(\d+)(?:\s|$)", installed)
    if match is None or tuple(map(int, match.groups())) < MINIMUM_UV_VERSION:
        minimum = ".".join(map(str, MINIMUM_UV_VERSION))
        raise RuntimeError(
            f"Installed {installed!r}; uv >= {minimum} is required for pinned "
            f"CPython {PINNED_PYTHON_VERSION}. Install a supported uv version before setup."
        )


def prepare(output: Path, uv: str) -> Path:
    if sys.platform != "win32":
        raise RuntimeError("Windows runtime preparation requires native Windows")
    require_uv_version(uv)
    with tempfile.TemporaryDirectory(prefix="dinkster-windows-python-") as temporary:
        work = Path(temporary)
        for filename, digest in INPUTS.items():
            url = (
                "https://raw.githubusercontent.com/Kosinkadink/Dinkster-Environments/"
                f"{ENVIRONMENTS_COMMIT}/{filename}"
            )
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError(f"Windows runtime input hash mismatch: {filename}")
            (work / filename).write_bytes(payload)
        subprocess.run(
            [
                uv,
                "run",
                "--no-project",
                "--python",
                sys.executable,
                "--with",
                "ziglang==0.14.1",
                "--",
                "python",
                str(work / "windows_sqlite.py"),
                "acquire",
                "--config",
                str(work / "variants.toml"),
                "--output",
                str(output),
            ],
            check=True,
        )
    python = output / "python/python.exe"
    subprocess.run(
        [
            str(python),
            "-I",
            str(
                Path(__file__).resolve().parents[1]
                / "packages/dinkster-supervisor/src/dinkster_supervisor/sqlite_runtime.py"
            ),
        ],
        check=True,
        timeout=60,
    )
    return python


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--github-env", action="store_true")
    args = parser.parse_args()
    python = prepare(args.output.resolve(), args.uv)
    if args.github_env:
        with Path(os.environ["GITHUB_ENV"]).open("a", encoding="utf-8") as environment:
            environment.write(f"UV_PYTHON={python}\nUV_PYTHON_DOWNLOADS=never\n")
    print(python)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        raise SystemExit(str(error)) from None
