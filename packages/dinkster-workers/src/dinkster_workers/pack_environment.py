"""Install reviewed dependency locks without resolving or mutating a generation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from packaging.utils import canonicalize_name

from .pack_lock import GenerationPackages, LockedWheel, PackLock, PackPlan, plan_pack
from .provision import ProvisionError


def venv_python(directory: Path) -> Path:
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _run(command: list[str], cache: Path) -> str:
    environment = {**os.environ, "UV_CACHE_DIR": str(cache)}
    result = subprocess.run(command, capture_output=True, text=True, env=environment)
    if result.returncode:
        raise ProvisionError((result.stderr or result.stdout).strip())
    return result.stdout


def _verify(path: Path, wheel: LockedWheel) -> bool:
    if not path.is_file() or path.stat().st_size != wheel.size:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == wheel.sha256


def fetch_wheel(wheel: LockedWheel, store: Path) -> Path:
    """Admit bytes only after size and digest verification; prefer stored bytes."""
    target = store / wheel.sha256 / wheel.filename
    for candidate in (target, store / wheel.filename):
        if _verify(candidate, wheel):
            return candidate
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=target.parent, suffix=".partial")
    staging = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(wheel.url) as source:
            remaining = wheel.size
            while block := source.read(min(1024 * 1024, remaining + 1)):
                remaining -= len(block)
                if remaining < 0:
                    raise ProvisionError(f"wheel exceeds locked size: {wheel.filename}")
                output.write(block)
        if not _verify(staging, wheel):
            raise ProvisionError(f"wheel hash or size mismatch: {wheel.filename}")
        os.replace(staging, target)
    finally:
        staging.unlink(missing_ok=True)
    return target


def provision_pack(
    plan: PackPlan,
    *,
    generation_python: Path,
    packs: Path,
    wheel_store: Path,
    uv_cache: Path,
    variant: str,
    uv: str = "uv",
) -> Path:
    """Recheck the reviewed plan, then create packs/<pack> from hashed wheels only."""
    plan.require_layering()
    current_plan = plan_pack(
        plan.lock,
        GenerationPackages.inspect(generation_python),
        variant=variant,
        own_environment_size=plan.own_environment_size,
    )
    current_plan.require_layering()
    if current_plan != plan:
        raise ProvisionError(f"pack {plan.lock.pack}: generation changed; review the plan again")
    destination = packs / plan.lock.pack
    if destination.exists():
        raise ProvisionError(f"pack venv already exists: {destination}; rebuild from its lock")
    wheels = [fetch_wheel(wheel, wheel_store) for wheel in plan.wheels]
    try:
        _run(
            [
                uv,
                "venv",
                "--python",
                str(generation_python),
                "--system-site-packages",
                str(destination),
            ],
            uv_cache,
        )
        python = venv_python(destination)
        if wheels:
            # Direct wheel paths with --no-deps cannot pull a second torch or consult an index.
            _run(
                [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    "--offline",
                    "--no-index",
                    "--no-deps",
                    *(str(path) for path in wheels),
                ],
                uv_cache,
            )
        plan.lock.write(destination / "pack-lock.json")
        return python
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def pack_drift(python: Path, lock: PackLock, generation: GenerationPackages) -> tuple[str, ...]:
    """Report extra, missing, or changed local distributions and offer a lock rebuild."""
    script = """
import importlib.metadata as m, json, sysconfig
print(json.dumps({d.metadata['Name']: d.version for d in m.distributions(
    path=[sysconfig.get_path('purelib'), sysconfig.get_path('platlib')]) if d.metadata['Name']}))
"""
    result = subprocess.run([str(python), "-c", script], capture_output=True, text=True)
    if result.returncode:
        return (f"{lock.pack}: interpreter unavailable; rebuild from pack-lock.json",)
    actual: dict[str, str] = {canonicalize_name(k): v for k, v in json.loads(result.stdout).items()}
    expected = {
        package.name: package.version
        for package in lock.packages
        if package.name not in generation.versions
    }
    return tuple(
        f"{lock.pack}: {name}: expected {expected.get(name, 'absent')}, "
        f"found {actual.get(name, 'absent')}; rebuild from pack-lock.json"
        for name in sorted(actual.keys() | expected.keys())
        if actual.get(name) != expected.get(name)
    )


def scratch_venv(
    *, generation_python: Path, directory: Path, uv_cache: Path, uv: str = "uv"
) -> Path:
    """Return a disposable terminal interpreter, never the immutable generation."""
    if not venv_python(directory).is_file():
        _run(
            [
                uv,
                "venv",
                "--python",
                str(generation_python),
                "--system-site-packages",
                str(directory),
            ],
            uv_cache,
        )
    return venv_python(directory)
