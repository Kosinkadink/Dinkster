"""User-defined base variants, independent of checkout-local engine resolution."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
import tempfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

import zstandard
from packaging.version import InvalidVersion, Version

from .environment_artifacts import acquire_base, validate_manifest
from .environment_local import retain_wheels, run_local_command
from .environment_state import EnvironmentError, content_digest, read_document, write_document


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
    row: dict[str, Any] = config["variants"][variant]
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
        project = work / "pyproject.toml"
        project.write_text(
            '[project]\nname = "dinkster-custom-base"\nversion = "0"\n'
            + f"requires-python = {json.dumps('==' + template['python'])}\n"
            + f"dependencies = {json.dumps(requirements)}\n"
            + '[[tool.uv.index]]\nname = "torch"\nexplicit = true\n'
            + f"url = {json.dumps(selection['torch-index'])}\n"
            + "[tool.uv.sources]\n"
            + "\n".join(f'{name} = {{ index = "torch" }}' for name in ("torch", "torchvision"))
            + "\n",
            encoding="ascii",
        )
        output = work / "pylock.toml"
        run_local_command(
            [
                "uv",
                "pip",
                "compile",
                str(project),
                "--python",
                str(python),
                "--only-binary",
                ":all:",
                "--format",
                "pylock.toml",
                "--generate-hashes",
                "--output-file",
                str(output),
                "--index",
                selection["extras-index"],
                "--default-index",
                "https://pypi.org/simple",
                "--no-header",
            ],
            cwd=work,
        )
        resolution = tomllib.loads(output.read_text(encoding="utf-8"))
        wheels = retain_wheels(resolution, python, store, work)
        for wheel in wheels:
            package = next(
                package for package in resolution["packages"] if package["name"] == wheel["name"]
            )
            wheel.update(
                index=package.get(
                    "index",
                    selection["torch-index"]
                    if wheel["name"] in {"torch", "torchvision"}
                    else selection["extras-index"],
                ),
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
            "nvidia": template["nvidia"],
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
