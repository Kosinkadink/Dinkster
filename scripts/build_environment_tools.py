"""Build a native, extraction-only environment client from locked wheel inputs."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import ssl
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import time
import tomllib
import zipfile
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen

from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

from scripts.build_release import sha256, wheel_metadata

_REMOTE_ATTEMPTS = 3


def acquire(source: str, destination: Path, size: int, digest: str) -> None:
    """Admit a local input or HTTPS payload only by its complete size and hash."""
    if size <= 0 or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("tools inputs require a positive size and lowercase SHA256")
    parsed = urlsplit(source)
    if Path(source).is_file():
        shutil.copyfile(source, destination)
    elif parsed.scheme:
        if parsed.scheme != "https" or parsed.username or parsed.password:
            raise ValueError("tools acquisition requires credential-free HTTPS")
        for attempt in range(1, _REMOTE_ATTEMPTS + 1):
            try:
                with urlopen(source, timeout=60) as response, destination.open("wb") as target:
                    if urlsplit(response.url).scheme != "https":
                        raise ValueError("tools acquisition redirected outside HTTPS")
                    shutil.copyfileobj(response, target)
                break
            except (HTTPException, URLError, OSError) as exc:
                destination.unlink(missing_ok=True)
                transient = (
                    500 <= exc.code < 600
                    if isinstance(exc, HTTPError)
                    else isinstance(
                        exc, (HTTPException, URLError, ssl.SSLError, ConnectionError, TimeoutError)
                    )
                )
                if isinstance(exc, HTTPError):
                    exc.close()
                if not transient or attempt == _REMOTE_ATTEMPTS:
                    raise
                time.sleep(2 ** (attempt - 1))
    else:
        shutil.copyfile(source, destination)
    if destination.stat().st_size != size or sha256(destination) != digest:
        destination.unlink()
        raise ValueError("tools input size or SHA256 mismatch")


def locked_wheels(root: Path, directory: Path, uv: str) -> Path:
    """Select native wheels from the exported lock without querying an index."""
    exported = subprocess.check_output(
        [
            uv,
            "export",
            "--locked",
            "--package",
            "dinkster-supervisor",
            "--group",
            "environment-tools",
            "--no-dev",
            "--no-default-groups",
            "--no-emit-workspace",
            "--no-header",
            "--no-annotate",
        ],
        cwd=root,
        text=True,
    )
    lock = tomllib.loads((root / "uv.lock").read_text())
    priority = {tag: index for index, tag in enumerate(sys_tags())}
    constraints = []
    for line in exported.replace("\\\n", "").splitlines():
        if not line.strip():
            continue
        requirement = Requirement(line.split(" --hash=", 1)[0].strip())
        if requirement.marker and not requirement.marker.evaluate():
            continue
        if requirement.url or not re.fullmatch(r"==[^*,]+", str(requirement.specifier)):
            raise ValueError("tools requirements must be exact locked registry pins")
        admitted = set(re.findall(r"--hash=sha256:([0-9a-f]{64})", line))
        candidates = []
        for package in lock["package"]:
            if canonicalize_name(package["name"]) != canonicalize_name(requirement.name):
                continue
            if not requirement.specifier.contains(package["version"], prereleases=True):
                continue
            for row in package.get("wheels", []):
                name = unquote(urlsplit(row["url"]).path).rsplit("/", 1)[-1]
                distribution, version, _, tags = parse_wheel_filename(name)
                ranks = [priority[tag] for tag in tags if tag in priority]
                digest = row["hash"].removeprefix("sha256:")
                if (
                    distribution == canonicalize_name(requirement.name)
                    and requirement.specifier.contains(version, prereleases=True)
                    and ranks
                    and digest in admitted
                ):
                    candidates.append((min(ranks), name, row, digest))
        if not candidates:
            raise ValueError(f"no locked native tools wheel for {requirement.name}")
        _, name, row, digest = min(candidates, key=lambda item: (item[0], item[1]))
        acquire(row["url"], directory / name, row["size"], digest)
        constraints.append(f"{requirement.name}{requirement.specifier} --hash=sha256:{digest}")
    path = directory / "constraints.txt"
    path.write_text("\n".join(constraints) + "\n", encoding="ascii")
    return path


def build(
    root: Path,
    output: Path,
    interpreter: str,
    interpreter_size: int,
    interpreter_sha256: str,
    supervisor_wheel: Path,
    *,
    uv: str = "uv",
) -> dict[str, object]:
    """Return the ordinary engine artifact row for a verified native tools ZIP."""
    if output.exists():
        raise ValueError("tools output must be a new directory")
    name, version = wheel_metadata(supervisor_wheel)
    if canonicalize_name(name) != "dinkster-supervisor":
        raise ValueError("tools require the release's dinkster-supervisor wheel")
    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    system = {"win32": "windows", "darwin": "macos", "linux": "linux"}.get(sys.platform)
    if system is None or arch not in {"x86_64", "arm64"}:
        raise ValueError("unsupported native tools platform")
    executable = "python/python.exe" if os.name == "nt" else "python/bin/python3"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dinkster-tools-", dir=output.parent) as temporary:
        work = Path(temporary)
        source = work / "interpreter.tar.gz"
        acquire(interpreter, source, interpreter_size, interpreter_sha256)
        payload = work / "payload"
        payload.mkdir()
        with tarfile.open(source, "r:gz") as archive:
            for member in archive:
                if not member.name.startswith("python/") and member.name != "python":
                    raise ValueError("standalone interpreter must contain only python/")
                archive.extract(member, payload, filter="data")
        for path in payload.rglob("*"):
            if not path.resolve(strict=True).is_relative_to(payload):
                raise ValueError("standalone interpreter contains an escaping link")
        python = payload / executable
        probe = json.loads(
            subprocess.check_output(
                [
                    str(python),
                    "-I",
                    "-c",
                    "import json,sys,sysconfig,platform; print(json.dumps([sys.version_info[:3], "
                    "sys.platform,platform.machine().lower(),sys.implementation.name, "
                    "sysconfig.get_config_var('SOABI'), "
                    "bool(sysconfig.get_config_var('Py_GIL_DISABLED'))]))",
                ],
                text=True,
            )
        )
        if probe != [
            list(sys.version_info[:3]),
            sys.platform,
            machine,
            sys.implementation.name,
            sysconfig.get_config_var("SOABI"),
            bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        ]:
            raise ValueError(
                "run the tools builder with its target Python version, architecture and ABI"
            )
        wheels = work / "wheels"
        wheels.mkdir()
        constraints = locked_wheels(root, wheels, uv)
        shutil.copyfile(supervisor_wheel, wheels / supervisor_wheel.name)
        with constraints.open("a", encoding="ascii") as target:
            target.write(
                f"dinkster-supervisor=={version} --hash=sha256:{sha256(supervisor_wheel)}\n"
            )
        subprocess.run(
            [
                uv,
                "pip",
                "sync",
                "--python",
                str(python),
                "--break-system-packages",
                "--offline",
                "--no-index",
                "--require-hashes",
                "--only-binary",
                ":all:",
                "--no-compile-bytecode",
                "--link-mode",
                "copy",
                "--find-links",
                str(wheels),
                str(constraints),
            ],
            env={**os.environ, "UV_PYTHON_DOWNLOADS": "never"},
            check=True,
        )
        checked = subprocess.run(
            [uv, "pip", "check", "--python", str(python), "--offline", "--no-python-downloads"],
            capture_output=True,
            text=True,
        )
        if checked.returncode:
            raise ValueError(
                "tools dependency closure is inconsistent: " + checked.stdout + checked.stderr
            )
        scripts = Path(
            subprocess.check_output(
                [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('scripts'))"],
                text=True,
            ).strip()
        )
        if not scripts.resolve().is_relative_to((payload / "python").resolve()):
            raise ValueError("tools scripts directory must be inside the standalone interpreter")
        bundled_uv = scripts / ("uv.exe" if os.name == "nt" else "uv")
        if not bundled_uv.is_file():
            raise ValueError("locked tools installation is missing uv")
        command_directory = python.parent if os.name != "nt" else python.parent / "Scripts"
        if scripts != command_directory:
            # Distribution Python builds can install scripts under python/local/bin.
            # Keep the bootstrap's native bin/Scripts PATH contract independent of that scheme.
            for command in ("uv", "uvx"):
                name = command + (".exe" if os.name == "nt" else "")
                shutil.copy2(scripts / name, command_directory / name)
        # ZIP dereferences safe internal aliases, requiring no symlink privilege at bootstrap.
        # Relocation cannot rely on installer-generated console-script shebangs.
        for path in scripts.glob("dinkster-*"):
            if path.is_file():
                path.unlink()
        subprocess.run(
            [str(python), "-I", "-m", "dinkster_supervisor.environment", "--help"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        staged = work / "result"
        staged.mkdir()
        portable = work / "portable"
        shutil.copytree(payload, portable)
        artifact = staged / f"dinkster-tools-{system}-{arch}.zip"
        with zipfile.ZipFile(artifact, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(portable.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(portable).as_posix())
        row: dict[str, object] = {
            "name": artifact.name,
            "size": artifact.stat().st_size,
            "sha256": sha256(artifact),
        }
        (staged / "SHA256SUMS").write_text(f"{row['sha256']}  {artifact.name}\n", encoding="ascii")
        staged.rename(output)
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interpreter", required=True, help="Standalone tar.gz path or HTTPS URL")
    parser.add_argument("--interpreter-size", required=True, type=int)
    parser.add_argument("--interpreter-sha256", required=True)
    parser.add_argument("--supervisor-wheel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--uv", default="uv")
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                Path(__file__).resolve().parents[1],
                args.output.resolve(),
                args.interpreter,
                args.interpreter_size,
                args.interpreter_sha256,
                args.supervisor_wheel.resolve(),
                uv=args.uv,
            ),
            sort_keys=True,
        )
    )
