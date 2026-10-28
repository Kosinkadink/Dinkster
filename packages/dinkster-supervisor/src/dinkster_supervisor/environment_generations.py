"""Construct complete generations exclusively from verified local store inputs."""

from __future__ import annotations

import errno
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from .environment_state import (
    EnvironmentError,
    Generation,
    content_digest,
    digest,
    generation_number,
    interpreter,
)


def _link_file(source: str, destination: str) -> str:
    # Dereference interpreter-internal aliases so every generation is self-contained.
    original = Path(source).resolve(strict=True)
    if sys.platform.startswith("linux"):
        import fcntl

        try:
            with original.open("rb") as source_file, open(destination, "xb") as target_file:
                fcntl.ioctl(target_file.fileno(), 0x40049409, source_file.fileno())  # FICLONE
            shutil.copymode(original, destination)
            return destination
        except OSError as exc:
            if exc.errno not in {errno.EOPNOTSUPP, errno.ENOTTY, errno.EINVAL, errno.EXDEV}:
                raise
            Path(destination).unlink(missing_ok=True)
    os.link(original, destination)
    return destination


def build_generation(
    *,
    root: Path,
    number: int,
    store: Path,
    base: str,
    engine_version: str,
    engine_release: str,
    constraints: Path,
    variant: str,
    worker_protocol: int,
    in_process_packs: tuple[str, ...] = (),
    prepare_packs: Callable[[Path], None] | None = None,
    uv: str = "uv",
) -> Generation:
    """Build a target without activating it or touching another generation.

    The caller has already verified/extracted the base and fetched the engine
    artifacts, and holds the project/store writer locks. ``constraints`` is
    the effective hashed requirements assembled for the selected base variant.
    Pack preflight and staging run through ``prepare_packs`` before inventory;
    an exception removes the incomplete target just like a failed sync.
    """
    generation_number(number)
    digest(base)
    digest(engine_release)
    source = store / "bases" / base.removeprefix("sha256:") / "python"
    destination = root / "generations" / str(number)
    if destination.exists():
        raise EnvironmentError(f"generation {number} already exists")
    if not source.is_dir():
        raise EnvironmentError(f"base {base} is not in the local store")
    if source.stat().st_dev != root.stat().st_dev:
        raise EnvironmentError("project and wheel store must be on the same volume")
    if not constraints.is_file():
        raise EnvironmentError("effective constraints.txt is missing")
    destination.mkdir(parents=True)
    try:
        shutil.copytree(source, destination / "python", copy_function=_link_file)
        shutil.copyfile(constraints, destination / "constraints.txt")
        python = interpreter(destination)
        if not python.is_file():
            raise EnvironmentError(f"base has no generation interpreter: {python}")
        base_python = source / python.relative_to(destination / "python")
        link_mode = "hardlink" if python.samefile(base_python) else "clone"
        wheels = store / "wheels"
        links = [wheels, *sorted({path.parent for path in wheels.glob("*/*.whl")})]
        command = [
            uv,
            "pip",
            "sync",
            "--python",
            str(python),
            # Standalone bases may carry a manager marker; only this owned target is writable.
            "--break-system-packages",
            "--offline",
            "--no-index",
            "--require-hashes",
            "--only-binary",
            ":all:",
            "--link-mode",
            link_mode,
        ]
        for directory in links:
            command.extend(["--find-links", str(directory)])
        command.append(str(destination / "constraints.txt"))
        environment = {
            **os.environ,
            "UV_CACHE_DIR": str(store / "uv-cache"),
            "UV_OFFLINE": "1",
            "UV_NO_INDEX": "1",
            "UV_PYTHON_DOWNLOADS": "never",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        result = subprocess.run(command, env=environment, capture_output=True, text=True)
        if result.returncode:
            raise EnvironmentError(f"offline generation sync failed: {result.stderr.strip()}")
        if prepare_packs is not None:
            prepare_packs(destination)
        files = tuple(
            (path.relative_to(destination).as_posix(), content_digest(path))
            for path in sorted(destination.rglob("*"))
            if path.is_file()
            and path.relative_to(destination).parts[0] != "packs"
            and "__pycache__" not in path.relative_to(destination).parts
        )
        generation = Generation(
            number,
            base,
            engine_version,
            engine_release,
            content_digest(destination / "constraints.txt"),
            variant,
            worker_protocol,
            files,
            in_process_packs,
        )
        generation.write(destination / "generation.json")
        return generation
    except BaseException:
        shutil.rmtree(destination)
        raise
