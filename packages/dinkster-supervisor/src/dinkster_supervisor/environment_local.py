"""Explicit development-only resolution from a checkout, never a channel fallback."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

from packaging.tags import compatible_tags, cpython_tags, platform_tags
from packaging.utils import parse_wheel_filename

from .environment_constraints import effective_constraints
from .environment_downloads import fetch_artifact
from .environment_state import EnvironmentError, content_digest, write_document

DEFAULT_VARIANTS = Path(__file__).with_name("local_variants.toml")


def _run(command: list[str], *, cwd: Path) -> str:
    environment = os.environ.copy()
    for name in ("UV_PROJECT", "UV_PROJECT_ENVIRONMENT", "PYTHONPATH", "PYTHONHOME"):
        environment.pop(name, None)
    result = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True)
    if result.returncode:
        raise EnvironmentError(f"local environment command failed: {result.stderr.strip()}")
    return result.stdout.strip()


def local_selection(
    source: Path, variant: str, variants: Path, python: Path | None
) -> dict[str, Any]:
    """Validate local inputs without resolving or modifying the prospective project."""
    config = tomllib.loads(variants.read_text(encoding="utf-8"))
    lock = tomllib.loads((source / "uv.lock").read_text(encoding="utf-8"))
    project = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))
    row = config["variants"].get(variant)
    if row is None and python is not None:
        row = tomllib.loads(DEFAULT_VARIANTS.read_text(encoding="utf-8"))["variants"].get(variant)
    if row is None:
        raise EnvironmentError(
            f"missing variants.toml row: variants.{variant}; "
            "use --from-scratch --python PATH for an explicit host interpreter override"
        )
    host = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}[platform.system()]
    arch = {"AMD64": "x86_64", "aarch64": "arm64"}.get(platform.machine(), platform.machine())
    matches = row["os"] == host and row.get("arch", "x86_64") == arch
    if python is None and (not matches or host not in config.get("interpreters", {})):
        raise EnvironmentError(
            f"missing variants.toml interpreter row for {host}/{arch} ({variant}); "
            "use --python PATH with --from-scratch to select a host interpreter explicitly"
        )
    if python is not None and not python.is_file():
        raise EnvironmentError(f"explicit interpreter does not exist: {python}")
    boundary = (source / "packages/dinkster-workers/src/dinkster_workers/boundary.py").read_text()
    protocol = re.search(r"(?m)^PROTOCOL_VERSION = ([0-9]+)$", boundary)
    if protocol is None:
        raise EnvironmentError("source worker protocol is missing")
    return {
        "variant": variant,
        "platform": {"os": host, "arch": arch},
        "python": str(python) if python is not None else config["base"]["python"],
        "override": python is not None,
        "index": row["torch-index"],
        "packages": config["base"]["packages"],
        "lock": lock,
        "lockDigest": content_digest(source / "uv.lock"),
        "version": project["project"]["version"],
        "workerProtocol": int(protocol[1]),
    }


def _resolve(
    requirements: Path, constraints: Path, output: Path, *, python: Path, index: str, source: Path
) -> dict[str, Any]:
    _run(
        [
            "uv",
            "pip",
            "compile",
            str(requirements),
            "--constraints",
            str(constraints),
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
            index,
            "--index-strategy",
            "unsafe-best-match",
            "--no-header",
        ],
        cwd=source,
    )
    return tomllib.loads(output.read_text(encoding="utf-8"))


def _wheels(
    resolution: dict[str, Any], python: Path, store: Path, source: Path
) -> list[dict[str, Any]]:
    version = json.loads(
        _run(
            [
                str(python),
                "-I",
                "-c",
                "import sys,json; print(json.dumps(list(sys.version_info[:2])))",
            ],
            cwd=source,
        )
    )
    platforms = list(platform_tags())
    tags = [
        *cpython_tags(tuple(version), platforms=platforms),
        *compatible_tags(
            tuple(version), interpreter=f"cp{version[0]}{version[1]}", platforms=platforms
        ),
    ]
    rank = {tag: index for index, tag in enumerate(tags)}
    rows: list[dict[str, Any]] = []
    for package in resolution.get("packages", []):
        candidates: list[tuple[int, str, str, dict[str, Any]]] = []
        for wheel in package.get("wheels", []):
            url = wheel.get("url")
            if url is None:
                url = Path(wheel["path"]).resolve().as_uri()
            filename = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
            _, _, _, wheel_tags = parse_wheel_filename(filename)
            compatible = [rank[tag] for tag in wheel_tags if tag in rank]
            if compatible:
                candidates.append((min(compatible), filename, url, wheel))
        if not candidates:
            raise EnvironmentError(f"no compatible locked wheel for {package['name']}")
        _, filename, url, wheel = min(candidates, key=lambda row: (row[0], row[1]))
        sha = wheel.get("hashes", {}).get("sha256")
        size = wheel.get("size")
        if sha is None and urlsplit(url).scheme == "https":
            with tempfile.TemporaryDirectory(prefix="dinkster-local-wheel-") as directory:
                downloaded = Path(directory) / filename
                with urlopen(url, timeout=60) as response, downloaded.open("wb") as output:
                    shutil.copyfileobj(response, output)
                sha, size = content_digest(downloaded)[7:], downloaded.stat().st_size
                fetch_artifact((directory,), filename, store / "wheels" / sha / filename, sha, size)
        if sha is not None and size is None and urlsplit(url).scheme == "https":
            with urlopen(Request(url, method="HEAD"), timeout=30) as response:
                size = int(response.headers["Content-Length"])
        if sha is None or size is None:
            local = Path(unquote(urlsplit(url).path))
            if urlsplit(url).scheme != "file":
                raise EnvironmentError(f"resolver omitted wheel integrity: {filename}")
            sha, size = content_digest(local)[7:], local.stat().st_size
        destination = store / "wheels" / sha / filename
        mirror = (
            str(Path(unquote(urlsplit(url).path)).parent)
            if urlsplit(url).scheme == "file"
            else url.rsplit("/", 1)[0]
        )
        fetch_artifact((mirror,), filename, destination, sha, size)
        rows.append(
            {
                "name": package["name"],
                "version": package["version"],
                "filename": filename,
                "sha256": sha,
                "size": size,
                "url": url,
            }
        )
    return rows


def prepare_local(source: Path, selection: dict[str, Any], store: Path) -> dict[str, Any]:
    """Resolve once, retain exact wheels, and return ordinary generation build inputs."""
    with tempfile.TemporaryDirectory(prefix="dinkster-local-") as directory:
        work = Path(directory)
        python = Path(
            _run(["uv", "python", "find", selection["python"], "--no-python-downloads"], cwd=source)
        )
        exported = work / "export.txt"
        _run(
            [
                "uv",
                "export",
                "--locked",
                "--all-packages",
                "--no-dev",
                "--group",
                "engine-base",
                "--no-emit-workspace",
                "--no-header",
                "--no-annotate",
                "--output-file",
                str(exported),
            ],
            cwd=source,
        )
        lock = selection["lock"]
        packages: dict[str, list[dict[str, Any]]] = {}
        for package in lock["package"]:
            packages.setdefault(package["name"], []).append(package)
        # All backend-specific descendants must be replaced, including remove-only CUDA rows.
        replaces: set[str] = set()
        pending = list(selection["packages"])
        while pending:
            name = pending.pop()
            if name in replaces:
                continue
            replaces.add(name)
            for package in packages.get(name, []):
                pending.extend(dep["name"] for dep in package.get("dependencies", []))
        pins = work / "pins.txt"
        pins.write_text(
            "\n".join(
                re.sub(r"\s+--hash=sha256:[0-9a-f]{64}", "", line)
                for line in exported.read_text().replace("\\\n", "").splitlines()
                if line.strip()
            )
            + "\n"
        )
        base_requirements = work / "base.in"
        base_requirements.write_text(
            "\n".join(f"{name}=={packages[name][0]['version']}" for name in selection["packages"])
            + "\n"
        )
        resolved = _resolve(
            base_requirements,
            pins,
            work / "pylock.base.toml",
            python=python,
            index=selection["index"],
            source=source,
        )
        base_wheels = _wheels(resolved, python, store, source)
        base: dict[str, Any] = {
            "format": "dinkster.base/1",
            "source": "local",
            "variant": selection["variant"],
            "platform": selection["platform"],
            "lock": selection["lockDigest"],
            "workerProtocol": selection["workerProtocol"],
            "wheels": base_wheels,
            "replaces": sorted(replaces | {row["name"] for row in base_wheels}),
            "interpreter": {
                "executable": "python/python.exe" if os.name == "nt" else "python/bin/python3"
            },
        }
        effective = work / "effective.txt"
        effective.write_text(effective_constraints(exported.read_text(), base))
        built = work / "built"
        for package in ("dinkster", "dinkster-api", "dinkster-supervisor"):
            _run(
                ["uv", "build", "--package", package, "--wheel", "--out-dir", str(built)],
                cwd=source,
            )
        engine_requirements = work / "engine.in"
        engine_requirements.write_text("\n".join(str(path) for path in sorted(built.glob("*.whl"))))
        engine = _resolve(
            engine_requirements,
            effective,
            work / "pylock.engine.toml",
            python=python,
            index=selection["index"],
            source=source,
        )
        wheels = _wheels(engine, python, store, source)
        prefix = Path(
            _run([str(python), "-I", "-c", "import sys; print(sys.base_prefix)"], cwd=source)
        )
        identity = "sha256:" + hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()
        base["id"] = identity
        base_root = store / "bases" / identity[7:]
        if not base_root.exists():
            staging = work / "base"
            shutil.copytree(
                prefix,
                staging / "python",
                symlinks=False,
                ignore=shutil.ignore_patterns("site-packages", "__pycache__"),
            )
            write_document(staging / "manifest.json", base)
            base_root.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging), base_root)
        release: dict[str, Any] = {
            "format": "dinkster.engine/1",
            "source": "local",
            "version": selection["version"],
            "workerProtocol": selection["workerProtocol"],
            "lock": selection["lockDigest"],
            "wheels": wheels,
        }
        release_id = (
            "sha256:" + hashlib.sha256(json.dumps(release, sort_keys=True).encode()).hexdigest()
        )
        engine_root = store / "engines" / release_id[7:]
        engine_root.mkdir(parents=True, exist_ok=True)
        write_document(engine_root / "release-manifest.json", release)
        constraints = engine_root / f"constraints-{selection['variant']}.txt"
        constraints.write_text(
            "\n".join(
                f"{row['name']}=={row['version']} --hash=sha256:{row['sha256']}" for row in wheels
            )
            + "\n"
        )
        return {
            "base": identity,
            "engine_release": release_id,
            "constraints": constraints,
            "engine_version": selection["version"],
            "variant": selection["variant"],
            "worker_protocol": selection["workerProtocol"],
            "source_kind": "local",
        }
