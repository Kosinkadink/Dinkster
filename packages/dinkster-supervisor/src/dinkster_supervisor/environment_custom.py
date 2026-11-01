"""User-defined base variants, independent of checkout-local engine resolution."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import tarfile
import tempfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from platform import machine
from typing import Any, cast
from urllib.parse import urlsplit

import zstandard
from packaging.version import InvalidVersion, Version

from .environment_artifacts import acquire_base, validate_manifest
from .environment_constraints import requirement_hashes
from .environment_local import retain_wheels, run_local_command
from .environment_state import (
    EnvironmentError,
    content_digest,
    interpreter,
    read_document,
    write_document,
)


def variant_file(root: Path, store: Path, explicit: Path | None) -> Path | None:
    """An explicit file wins; project definitions take precedence over store defaults."""
    if explicit is not None:
        path = explicit.resolve(strict=True)
        if not path.is_file():
            raise EnvironmentError("--variants must name a TOML file")
        return path
    return next(
        (path for path in (root / "variants.toml", store / "variants.toml") if path.is_file()), None
    )


def custom_selection(path: Path, variant: str, template: dict[str, Any]) -> dict[str, Any] | None:
    """Read a supported custom row without creating store or project files."""
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    rows = config.get("variants", {})
    if not isinstance(rows, dict):
        raise EnvironmentError("variants.toml must contain a variants table")
    value = cast("dict[str, Any]", rows).get(variant)
    if value is None:
        return None
    if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", variant) is None or not isinstance(value, dict):
        raise EnvironmentError("custom variant must be a named TOML table")
    row: dict[str, Any] = {**config["variants"][variant]}
    row.setdefault("extras-index", "https://pypi.org/simple")
    for key in ("torch", "torch-index", "extras-index"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise EnvironmentError(f"variants.{variant} requires {key}")
    for key in ("torch", "torchvision"):
        if key in row:
            try:
                Version(row[key])
            except (InvalidVersion, TypeError) as exc:
                raise EnvironmentError(
                    f"variants.{variant}.{key} must be an exact version"
                ) from exc
    for key in ("torch-index", "extras-index"):
        parsed = urlsplit(cast("str", row[key]))
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.fragment
            or parsed.query
        ):
            raise EnvironmentError(
                f"variants.{variant}.{key} must be a credential-free HTTPS index"
            )
    platform = template["platform"]
    if (
        row.get("os", platform["os"]) != platform["os"]
        or row.get("arch", platform["arch"]) != platform["arch"]
    ):
        raise EnvironmentError(
            f"variant {variant} does not match host {platform['os']}/{platform['arch']}"
        )
    return {
        **row,
        "variant": variant,
        "sourceFile": str(path.resolve(strict=True)),
        "definitionDigest": content_digest(path),
    }


def base_details(base: dict[str, Any]) -> dict[str, object]:
    """Describe the active base, never infer provenance from a variant's name."""
    torch = next((wheel for wheel in base["wheels"] if wheel["name"] == "torch"), None)
    return {
        "variant": base["variant"],
        "origin": "user-defined" if base.get("source") == "user-defined" else "published",
        "torchVersion": torch["version"] if torch else None,
        "torchIndex": torch["index"] if torch else None,
    }


def native_template(bases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Select an interpreter donor by OS/architecture, not accelerator support."""
    host = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    arch = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine().lower(), machine().lower())
    candidates = [base for base in bases.values() if base["platform"] == {"os": host, "arch": arch}]
    if not candidates:
        raise EnvironmentError(f"channel has no native interpreter donor for {host}/{arch}")
    return min(candidates, key=lambda base: (base["nvidia"] is not None, base["variant"]))


def build_custom_base(
    selection: dict[str, Any], template: dict[str, Any], sources: tuple[str, ...], store: Path
) -> dict[str, Any]:
    """Build a verified producer-format base; engine installation remains offline."""
    path = Path(selection["sourceFile"])
    if content_digest(path) != selection["definitionDigest"]:
        raise EnvironmentError("custom variants file changed after planning; request a new plan")
    # Reuse a verified native interpreter, including Windows' SQLite repair.
    native = acquire_base(template, sources, store)
    with tempfile.TemporaryDirectory(prefix="dinkster-custom-", dir=store) as directory:
        work = Path(directory)
        python = native / template["interpreter"]["executable"]
        roots = {
            wheel["name"]: wheel["version"]
            for wheel in template["wheels"]
            if wheel["name"] in {"torch", "numpy", "opencv-contrib-python-headless"}
        }
        roots["torch"] = selection["torch"]
        if "torchvision" in selection:
            roots["torchvision"] = selection["torchvision"]
        requirements = [f"{name}=={version}" for name, version in roots.items()]
        if "torchvision" not in selection:
            requirements.append("torchvision")
        host_platform = template["platform"]
        host_os = {"windows": "win32", "macos": "darwin"}.get(host_platform["os"], "linux")
        host_arch = "AMD64" if host_platform["os"] == "windows" else host_platform["arch"]
        environment = f"sys_platform == '{host_os}' and platform_machine == '{host_arch}'"
        project = work / "pyproject.toml"
        project.write_text(
            '[project]\nname = "dinkster-custom-base"\nversion = "0"\n'
            + f"requires-python = {json.dumps('==' + template['python'])}\n"
            + f"dependencies = {json.dumps(requirements)}\n"
            + "[tool.uv]\n"
            + f"environments = [{json.dumps(environment)}]\n"
            + '[[tool.uv.index]]\nname = "torch"\n'
            + f"url = {json.dumps(selection['torch-index'])}\n"
            + (
                '[[tool.uv.index]]\nname = "extras"\n'
                + f"url = {json.dumps(selection['extras-index'])}\n"
                if selection["extras-index"].rstrip("/") != selection["torch-index"].rstrip("/")
                else ""
            )
            + "[tool.uv.sources]\n"
            + 'torch = { index = "torch" }\n',
            encoding="ascii",
        )
        output = work / "pylock.toml"
        run_local_command(
            [
                "uv",
                "lock",
                "--python",
                str(python),
                "--no-build",
                "--no-python-downloads",
                "--default-index",
                "https://pypi.org/simple",
            ],
            cwd=work,
        )
        run_local_command(
            [
                "uv",
                "export",
                "--frozen",
                "--no-emit-project",
                "--no-dev",
                "--format",
                "pylock.toml",
                "--output-file",
                str(output),
                "--no-header",
            ],
            cwd=work,
        )
        locked = tomllib.loads((work / "uv.lock").read_text(encoding="utf-8"))
        resolution = tomllib.loads(output.read_text(encoding="utf-8"))
        wheels = retain_wheels(resolution, python, store, work)
        for wheel in wheels:
            package = next(
                package
                for package in locked["package"]
                if package["name"] == wheel["name"] and package["version"] == wheel["version"]
            )
            wheel.update(
                index=package["source"]["registry"],
                verification="index",
                lockedVersion=None,
            )
        payload = work / "payload"
        shutil.copytree(native / "python", payload / "python", symlinks=True)
        (payload / "wheels").mkdir()
        for wheel in wheels:
            shutil.copyfile(
                store / "wheels" / wheel["sha256"] / wheel["filename"],
                payload / "wheels" / wheel["filename"],
            )
        archive = work / "base.tar.zst"
        with (
            archive.open("wb") as target,
            zstandard.ZstdCompressor().stream_writer(target) as compressed,
            tarfile.open(fileobj=compressed, mode="w|") as tar,
        ):
            tar.add(payload / "python", arcname="python")
            tar.add(payload / "wheels", arcname="wheels")
        base: dict[str, Any] = {
            "format": "dinkster.base/1",
            "variant": selection["variant"],
            "platform": template["platform"],
            "python": template["python"],
            "interpreter": template["interpreter"],
            "wheels": sorted(wheels, key=lambda wheel: wheel["filename"]),
            "replaces": sorted(set(template["replaces"]) | {wheel["name"] for wheel in wheels}),
            "workerProtocol": template["workerProtocol"],
            "nvidia": selection.get("nvidia"),
            "archive": {"format": "tar+zstd"},
        }
        identity = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(base, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
                    "ascii"
                )
            ).hexdigest()
        )
        base.update(
            id=identity,
            signature=None,
            source="user-defined",
            localBuild={
                "sourceFile": selection["sourceFile"],
                "definitionDigest": selection["definitionDigest"],
                "builtAt": datetime.now(UTC).isoformat(),
                "torchIndex": selection["torch-index"],
                "extrasIndex": selection["extras-index"],
            },
        )
        parts: list[dict[str, Any]] = []
        with archive.open("rb") as compressed:
            while compressed.tell() < archive.stat().st_size:
                part = work / f"base.tar.zst.{len(parts) + 1:03d}"
                remaining = 1_900_000_000
                with part.open("wb") as target:
                    while remaining and (chunk := compressed.read(min(remaining, 1024 * 1024))):
                        target.write(chunk)
                        remaining -= len(chunk)
                parts.append(
                    {
                        "filename": part.name,
                        "size": part.stat().st_size,
                        "sha256": content_digest(part)[7:],
                    }
                )
        base["archive"]["parts"] = parts
        validate_manifest(json.dumps(base).encode("ascii"), identity, selection["variant"])
        retained = store / "downloads" / identity[7:]
        retained.mkdir(parents=True, exist_ok=True)
        for part in parts:
            shutil.copyfile(work / part["filename"], retained / part["filename"])
        write_document(retained / "manifest.json", base)
        installed = acquire_base(base, (str(retained),), store)
        existing = installed / "manifest.json"
        if existing.exists():
            return read_document(existing)
        write_document(existing, base)
        write_document(
            installed / "extras.json",
            {
                "format": "dinkster.extras/1",
                "variant": selection["variant"],
                "base": identity,
                "index": selection["extras-index"],
                "wheels": [],
            },
        )
        return base


def prepare_custom_pack_locks(
    generation: Path,
    *,
    store: Path,
    base: dict[str, Any],
    catalog_lock: Path,
    artifacts: Path,
    output: Path,
) -> tuple[Path, ...]:
    """Resolve standard locks in the acquired target interpreter before activation."""
    hashes = requirement_hashes((generation / "constraints.txt").read_text(encoding="utf-8"))
    wheels = store / "wheels"
    links = [
        wheels,
        *sorted({path.parent for path in wheels.glob("*/*.whl") if path.parent.name in hashes}),
    ]
    command = [
        str(interpreter(generation)),
        "-B",
        "-m",
        "dinkster.generation_packs",
        "--generation",
        str(generation),
        "--catalog-lock",
        str(catalog_lock),
        "--artifacts",
        str(artifacts),
        "--prepare-locks",
        str(output),
        "--variant",
        base["variant"],
        "--accelerator",
        "auto",
        "--index",
        "https://pypi.org/simple",
        "--json",
    ]
    extras = base["localBuild"]["extrasIndex"]
    if extras.rstrip("/") != "https://pypi.org/simple":
        command.extend(["--index", extras])
    for path in links:
        command.extend(["--find-links", str(path)])
    response: object = json.loads(run_local_command(command, cwd=generation))
    if not isinstance(response, dict):
        raise EnvironmentError("custom pack preparation did not return an object")
    document = cast("dict[str, object]", response)
    paths = document.get("locks")
    if (
        document.get("errors") != []
        or not isinstance(paths, list)
        or any(not isinstance(path, str) for path in cast("list[object]", paths))
    ):
        raise EnvironmentError("custom pack preparation did not return dependency lock paths")
    locks = tuple(Path(path) for path in cast("list[str]", paths))
    if len(set(locks)) != len(locks) or any(
        not path.is_absolute() or path.parent != output or not path.is_file() for path in locks
    ):
        raise EnvironmentError("custom dependency lock output escapes its scratch directory")
    return locks
