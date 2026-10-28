"""Install reviewed dependency locks without resolving or mutating a generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from packaging.utils import canonicalize_name

from .boundary import PROTOCOL_VERSION
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


def _create_layered_venv(generation_python: Path, directory: Path, uv_cache: Path, uv: str) -> Path:
    _run(
        [
            uv,
            "venv",
            "--python",
            str(generation_python),
            "--system-site-packages",
            "--relocatable",
            str(directory),
        ],
        uv_cache,
    )
    python = venv_python(directory)
    if python.is_symlink():
        python.unlink()
        python.symlink_to(os.path.relpath(generation_python.absolute(), python.parent.absolute()))
    return python


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
        python = _create_layered_venv(generation_python, destination, uv_cache, uv)
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
        return _create_layered_venv(generation_python, directory, uv_cache, uv)
    return venv_python(directory)


@dataclass(frozen=True)
class GenerationContext:
    generation: Path
    generation_python: Path
    constraints: Path
    variant: str
    worker_protocol: int
    wheel_store: Path
    uv_cache: Path
    own_environment_size: int
    uv: str = "uv"

    def validate(self) -> None:
        if self.worker_protocol != PROTOCOL_VERSION:
            raise ProvisionError(
                f"workerProtocol mismatch: generation declares {self.worker_protocol}, "
                f"provisioner speaks {PROTOCOL_VERSION}"
            )
        if not self.constraints.is_file():
            raise ProvisionError(f"generation constraints missing: {self.constraints}")
        if self.own_environment_size < 0:
            raise ProvisionError("own-environment size must not be negative")


def _generation_locks(
    source_generation: Path | None, overrides: Sequence[Path]
) -> tuple[PackLock, ...]:
    locks: dict[str, PackLock] = {}
    replacements = {lock.pack: lock for path in overrides for lock in (PackLock.read(path),)}
    if len(replacements) != len(overrides):
        raise ProvisionError("more than one supplied dependency lock for the same pack")
    if source_generation is not None:
        if not source_generation.is_dir():
            raise ProvisionError(f"source generation missing: {source_generation}")
        packs = source_generation / "packs"
        if packs.is_dir():
            for directory in sorted(packs.iterdir()):
                if not directory.is_dir():
                    continue
                path = directory / "pack-lock.json"
                if directory.name in replacements:
                    locks[directory.name] = replacements.pop(directory.name)
                else:
                    if not path.is_file():
                        raise ProvisionError(
                            f"pack {directory.name}: recorded dependency lock missing"
                        )
                    lock = PackLock.read(path)
                    if lock.pack != directory.name:
                        raise ProvisionError(
                            f"pack {directory.name}: recorded lock names {lock.pack}"
                        )
                    locks[lock.pack] = lock
    locks.update(replacements)
    return tuple(locks[name] for name in sorted(locks))


def preflight_generation(
    context: GenerationContext,
    *,
    source_generation: Path | None = None,
    locks: Sequence[Path] = (),
) -> tuple[PackPlan, ...]:
    """Plan all recorded packs without fetching wheels or changing either generation."""
    context.validate()
    recorded = _generation_locks(source_generation, locks)
    if not recorded:
        return ()
    facts = GenerationPackages.inspect(context.generation_python)
    return tuple(
        plan_pack(
            lock, facts, variant=context.variant, own_environment_size=context.own_environment_size
        )
        for lock in recorded
    )


def rebuild_generation(context: GenerationContext, plans: Sequence[PackPlan]) -> tuple[str, ...]:
    """Build every target pack before its owner starts or activates the generation."""
    context.validate()
    for plan in plans:
        plan.require_layering()
    packs = context.generation / "packs"
    if any((packs / plan.lock.pack).exists() for plan in plans):
        raise ProvisionError("target already has pack venvs; rebuild into a fresh generation")
    created: list[Path] = []
    try:
        for plan in plans:
            provision_pack(
                plan,
                generation_python=context.generation_python,
                packs=packs,
                wheel_store=context.wheel_store,
                uv_cache=context.uv_cache,
                variant=context.variant,
                uv=context.uv,
            )
            created.append(packs / plan.lock.pack)
    except BaseException:
        for directory in created:
            shutil.rmtree(directory)
        raise
    return tuple(f"rebuilt pack {plan.lock.pack} from its dependency lock" for plan in plans)


def doctor_generation(context: GenerationContext) -> tuple[str, ...]:
    """Read-only inspection; repair is an explicitly confirmed generation rebuild."""
    plans = preflight_generation(context, source_generation=context.generation)
    facts = GenerationPackages.inspect(context.generation_python) if plans else None
    changes: list[str] = []
    for plan in plans:
        changes.extend(f"{plan.lock.pack}: {message}" for message in plan.conflicts)
        assert facts is not None
        changes.extend(
            pack_drift(
                venv_python(context.generation / "packs" / plan.lock.pack),
                plan.lock,
                facts,
            )
        )
    return tuple(changes)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("preflight", "rebuild", "doctor", "terminal"))
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--generation-python", type=Path, required=True)
    parser.add_argument("--constraints", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--worker-protocol", type=int, required=True)
    parser.add_argument("--wheel-store", type=Path, required=True)
    parser.add_argument("--uv-cache", type=Path, required=True)
    parser.add_argument("--own-environment-size", type=int, required=True)
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--source-generation", type=Path)
    parser.add_argument("--lock", type=Path, action="append", default=[])
    parser.add_argument("--pack")
    args = parser.parse_args(argv)
    context = GenerationContext(
        args.generation,
        args.generation_python,
        args.constraints,
        args.variant,
        args.worker_protocol,
        args.wheel_store,
        args.uv_cache,
        args.own_environment_size,
        args.uv,
    )
    plans: tuple[PackPlan, ...] = ()
    changes: tuple[str, ...] = ()
    code = 0
    try:
        context.validate()
        if args.operation == "doctor":
            plans = preflight_generation(context, source_generation=context.generation)
            changes = doctor_generation(context)
            code = int(bool(changes))
        elif args.operation == "terminal":
            if args.pack:
                available = _generation_locks(context.generation, ())
                if args.pack not in {lock.pack for lock in available}:
                    raise ProvisionError(f"pack {args.pack}: no installed dependency lock")
                python = venv_python(context.generation / "packs" / args.pack)
                if not python.is_file():
                    raise ProvisionError(f"pack {args.pack}: venv missing; rebuild from lock")
            else:
                python = scratch_venv(
                    generation_python=context.generation_python,
                    directory=context.generation.parent.parent
                    / "scratch"
                    / context.generation.name,
                    uv_cache=context.uv_cache,
                    uv=context.uv,
                )
            changes = (str(python),)
        else:
            plans = preflight_generation(
                context,
                source_generation=args.source_generation,
                locks=args.lock,
            )
            code = int(any(plan.conflicts for plan in plans))
            if not code and args.operation == "rebuild":
                changes = rebuild_generation(context, plans)
    except (ProvisionError, OSError, ValueError) as exc:
        changes = (str(exc),)
        print(str(exc), file=sys.stderr)
        code = 1
    print(
        json.dumps(
            {
                "packs": [
                    {
                        "pack": plan.lock.pack,
                        "conflicts": list(plan.conflicts),
                        "downloadBytes": sum(
                            wheel.size
                            for wheel in plan.wheels
                            if not any(
                                _verify(path, wheel)
                                for path in (
                                    context.wheel_store / wheel.sha256 / wheel.filename,
                                    context.wheel_store / wheel.filename,
                                )
                            )
                        ),
                        "ownEnvironmentBytes": plan.own_environment_size,
                    }
                    for plan in plans
                ],
                "changes": list(changes),
            }
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
