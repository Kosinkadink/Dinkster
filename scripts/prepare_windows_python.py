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
import time
import urllib.error
import urllib.request
from pathlib import Path

# This source pin moves with the Windows base's SQLite override, not with uv's catalog.
ENVIRONMENTS_COMMIT = "f0d10799dff79d92fab2d6b45da9c91e2f35b4e9"
INPUTS = {
    "windows_sqlite.py": "8b8b02d2a86f22f06bbb10424076ad06aac5c3f27f7caf3916a42cfc55f5aa6d",
    "variants.toml": "884515be492964f2a1c5bcde70fc0e7e6ee95ad77c205ea8917ed6f9a121e02e",
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
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=60) as response:
                        payload = response.read()
                    break
                except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
                    if isinstance(error, urllib.error.HTTPError):
                        transient = 500 <= error.code < 600
                    elif isinstance(error, urllib.error.URLError):
                        transient = isinstance(error.reason, (ConnectionError, TimeoutError))
                    else:
                        transient = True
                    if not transient or attempt == 2:
                        raise
                    time.sleep(attempt + 1)
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
