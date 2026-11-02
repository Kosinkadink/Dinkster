"""Build offline publisher artifacts without importing a test runner or test helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import sysconfig
import tarfile
import zipfile
from pathlib import Path

import zstandard


def canonical_json(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def artifact(name: str, data: bytes, *, key: str = "name") -> dict:
    return {key: name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def wheel(directory: Path, name: str, version: str, value: int, *, engine: bool = False) -> Path:
    path = directory / f"{name}-{version}-py3-none-any.whl"
    metadata = f"{name}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{name}.py", f"VALUE = {value}\n")
        archive.writestr(
            f"{metadata}/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        )
        archive.writestr(
            f"{metadata}/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{metadata}/RECORD", "")
        if engine:
            archive.writestr("dinkster/__init__.py", "")
            archive.writestr("dinkster/serve.py", Path(__file__).with_name("serve.py").read_bytes())
            archive.writestr(
                "dinkster/manager.py", Path(__file__).with_name("manager.py").read_bytes()
            )
            for module in ("project_requirements", "workflow_requirements"):
                archive.writestr(
                    f"dinkster/{module}.py", Path(__file__).with_name(f"{module}.py").read_bytes()
                )
            archive.writestr(
                "dinkster/generation_packs.py",
                "import argparse,json,shutil\n"
                "from pathlib import Path\n"
                "p=argparse.ArgumentParser()\n"
                "p.add_argument('--generation',type=Path)\n"
                "p.add_argument('--catalog-lock',type=Path)\n"
                "p.add_argument('--artifacts',type=Path)\n"
                "p.add_argument('--json',action='store_true')\n"
                "a=p.parse_args()\n"
                "assert json.loads(a.catalog_lock.read_text())['packs']==[]\n"
                "shutil.copyfile(a.catalog_lock,a.generation/'dinkster.lock')\n"
                "print(json.dumps({'packs':[],'errors':[]}))\n",
            )
    return path


def standalone_tree(root: Path) -> None:
    source = Path(sys.base_prefix)
    if os.name == "nt":
        root.mkdir(parents=True)
        for path in source.iterdir():
            if path.suffix in {".exe", ".dll"}:
                shutil.copy2(path, root / path.name)
        shutil.copytree(source / "DLLs", root / "DLLs")
        shutil.copytree(
            source / "Lib", root / "Lib", ignore=shutil.ignore_patterns("site-packages")
        )
    else:
        (root / "bin").mkdir(parents=True)
        shutil.copy2(Path(sys.executable).resolve(), root / "bin/python3")
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        shutil.copytree(
            Path(sysconfig.get_path("stdlib")),
            root / "lib" / version,
            ignore=shutil.ignore_patterns("site-packages", "__pycache__"),
        )
        library = sysconfig.get_config_var("LDLIBRARY")
        if library:
            path = Path(sysconfig.get_config_var("LIBDIR")) / library
            if path.is_file():
                shutil.copy2(path.resolve(), root / "lib" / path.name)
        (root / "bin/python-alias").symlink_to("python3")
        (root / "resources").mkdir()
        (root / "resources/value").write_bytes(b"internal directory alias")
        (root / "resource-alias").symlink_to("resources", target_is_directory=True)
    python = root / ("python.exe" if os.name == "nt" else "bin/python3")
    subprocess.run([str(python), "-I", "-B", "-c", "import colorsys, http.server, ssl"], check=True)


def build_base(mirror: Path, scratch: Path, variant: str) -> dict:
    tree = scratch / "tree"
    standalone_tree(tree / "python")
    (tree / "wheels").mkdir()
    base_wheel = wheel(tree / "wheels", "fixture_base", "1.0", 37)
    executable = "python/python.exe" if os.name == "nt" else "python/bin/python3"
    architecture = platform.machine().lower()
    if architecture == "amd64":
        architecture = "x86_64"
    base = {
        "format": "dinkster.base/1",
        "variant": variant,
        "platform": {
            "os": "macos" if sys.platform == "darwin" else platform.system().lower(),
            "arch": architecture,
        },
        "python": platform.python_version(),
        "interpreter": {
            **artifact("interpreter.tar.gz", b"fixture interpreter source", key="filename"),
            "release": "fixture",
            "url": "https://example.invalid/interpreter.tar.gz",
            "executable": executable,
        },
        "wheels": [
            {
                **artifact(base_wheel.name, base_wheel.read_bytes(), key="filename"),
                "name": "fixture-base",
                "version": "1.0",
                "lockedVersion": "1.0",
                "index": "https://pypi.org/simple",
                "url": "https://example.invalid/fixture.whl",
                "verification": "lock",
            }
        ],
        "replaces": ["fixture-base"],
        "workerProtocol": 11,
        "nvidia": None,
        "archive": {"format": "tar+zstd"},
    }
    identity = hashlib.sha256(canonical_json(base)).hexdigest()
    directory = mirror / "base" / variant / identity
    directory.mkdir(parents=True)
    part = directory / "base.tar.zst.001"
    with part.open("wb") as output, zstandard.ZstdCompressor().stream_writer(output) as encoded:
        with tarfile.open(fileobj=encoded, mode="w|", dereference=False) as archive:
            archive.add(tree / "python", "python")
            archive.add(tree / "wheels", "wheels")
    base.update(
        id="sha256:" + identity,
        archive={
            "format": "tar+zstd",
            "parts": [artifact(part.name, part.read_bytes(), key="filename")],
        },
        signature=None,
    )
    (directory / "manifest.json").write_bytes(canonical_json(base))
    return base


def executable_release(mirror: Path, base: dict, version: str, value: int) -> None:
    """Publish one engine and select it on main using the production release format."""
    scratch = mirror / "build"
    scratch.mkdir(exist_ok=True)
    try:
        path = wheel(scratch, "fixture_engine", version, value, engine=True)
        payload = path.read_bytes()
        constraints = (
            f"fixture-base==1.0 --hash=sha256:{base['wheels'][0]['sha256']}\n"
            f"fixture-engine=={version} --hash=sha256:{hashlib.sha256(payload).hexdigest()}\n"
        ).encode("ascii")
        engine = {
            "format": "dinkster.engine/1",
            "repository": "Kosinkadink/Dinkster",
            "tag": "main-" + "d" * 40,
            "commit": "d" * 40,
            "version": version,
            "workerProtocol": 11,
            "artifacts": [artifact("constraints.txt", constraints), artifact(path.name, payload)],
        }
        data = canonical_json(engine) + b"\n"
        release = hashlib.sha256(data).hexdigest()
        directory = mirror / "engine" / release
        directory.mkdir(parents=True)
        (directory / "release-manifest.json").write_bytes(data)
        (directory / "constraints.txt").write_bytes(constraints)
        (directory / path.name).write_bytes(payload)
        channel_path = mirror / "channels/main.json"
        channel = json.loads(channel_path.read_bytes())
        channel["engine"] = {
            "version": version,
            "release": "sha256:" + release,
            "locations": [{"type": "mirror", "path": f"engine/{release}/release-manifest.json"}],
        }
        channel_path.write_bytes(canonical_json(channel))
    finally:
        shutil.rmtree(scratch)


def build(output: Path, variant: str, port: int) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    mirror = output / "mirror"
    scratch = output / "build"
    scratch.mkdir()
    try:
        base = build_base(mirror, scratch, variant)
    finally:
        shutil.rmtree(scratch)
    channels = mirror / "channels"
    channels.mkdir()
    channel = {
        "format": "dinkster.channel/1",
        "base": {
            variant: {
                "id": base["id"],
                "locations": [
                    {
                        "type": "mirror",
                        "path": f"base/{variant}/{base['id'][7:]}/manifest.json",
                    }
                ],
            }
        },
        "minimumDesktopVersion": "0.2.0",
        "signature": None,
    }
    (channels / "main.json").write_bytes(canonical_json(channel))
    executable_release(mirror, base, "1.0", 41)
    initial = (channels / "main.json").read_bytes()
    (output / "initial-channel.json").write_bytes(initial)
    executable_release(mirror, base, "2.0", 59)
    (output / "updated-channel.json").write_bytes((channels / "main.json").read_bytes())
    for name in ("main", "stable"):
        (channels / f"{name}.json").write_bytes(initial)
    root, store = output / "project", output / "store"
    command = [sys.executable, "-m", "dinkster_supervisor.environment", "--root", str(root)]
    document = {
        "mirror": str(mirror),
        "root": str(root),
        "store": str(store),
        "variant": variant,
        "channel": "main",
        "port": port,
        "baseManifest": str(mirror / "base" / variant / base["id"][7:] / "manifest.json"),
        "initialChannel": str(output / "initial-channel.json"),
        "updatedChannel": str(output / "updated-channel.json"),
        "commands": {
            "setup": command
            + [
                "setup",
                "--mirror",
                str(mirror),
                "--channel",
                "main",
                "--variant",
                variant,
                "--data",
                str(store),
                "--port",
                str(port),
                "--yes",
                "--json",
            ],
            "advance": [sys.executable, str(Path(__file__).resolve()), "--advance", str(output)],
            **{
                name: command + [name, "--yes", "--json"] for name in ("update", "rollback", "stop")
            },
            **{name: command + [name, "--json"] for name in ("status", "doctor")},
        },
        "expected": {
            "initial": {"version": "1.0", "value": 41},
            "updated": {"version": "2.0", "value": 59},
        },
    }
    (output / "fixture.json").write_bytes(canonical_json(document) + b"\n")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path, help="New directory; existing paths are refused")
    group.add_argument(
        "--advance", type=Path, help="Select the updated release in an existing fixture"
    )
    parser.add_argument("--variant", default="win-cpu" if os.name == "nt" else "linux-cpu")
    parser.add_argument(
        "--port", type=int, help="Engine port; defaults to an available loopback port"
    )
    args = parser.parse_args()
    if args.advance:
        output = args.advance.resolve()
        data = (output / "updated-channel.json").read_bytes()
        for name in ("main", "stable"):
            destination = output / "mirror/channels" / f"{name}.json"
            temporary = destination.with_suffix(".tmp")
            temporary.write_bytes(data)
            temporary.replace(destination)
        document = json.loads((output / "fixture.json").read_bytes())
    else:
        port = args.port
        if port is None:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = int(sock.getsockname()[1])
        if not 1 <= port <= 65535:
            parser.error("--port must be in 1..65535")
        document = build(args.output.resolve(), args.variant, port)
    print(json.dumps(document, sort_keys=True))


if __name__ == "__main__":
    main()
