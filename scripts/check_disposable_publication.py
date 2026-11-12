"""Run the real publisher and scanner with staged inputs and no external network."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path


def verify_active(inputs: Path, rows: list[dict], timeout: float) -> None:
    import urllib.request

    from dinkster_registry import Lockfile
    from dinkster_workers.manifest import PackDependency

    from dinkster.registries import NamedRegistry
    from scripts.publish_official_packs import _wait_active

    registry = NamedRegistry("", "http://registry:8000", token_env="DINKSTER_REGISTRY_TOKEN")
    catalog = Lockfile.from_record_json((inputs / "sources/official-packs.lock").read_text())
    candidates = {row["pack"]: row.get("candidateId") for row in rows}
    deadline = time.monotonic() + timeout
    for pack in catalog.packs:
        _wait_active(
            registry,
            PackDependency(pack.pack, "==" + pack.version),
            pack.version,
            candidates[pack.pack],
            deadline,
        )
        url = registry.endpoint + f"/v1/packs/{pack.pack}/versions/{pack.version}"
        with urllib.request.urlopen(url, timeout=5) as response:
            row = json.load(response)
        if row["artifact"]["digest"] != pack.artifact_digest:
            raise RuntimeError(f"{pack.pack}: active artifact differs from selected bytes")


def check_publication(
    root: Path,
    inputs: Path,
    package_index: Path,
    service_image: str,
    publisher_image: str,
    *,
    timeout: float = 600,
) -> dict:
    """Images and the complete dependency index must already exist locally."""
    start = time.monotonic()
    deadline = start + timeout
    name = "dinkster-publication-" + uuid.uuid4().hex
    containers: list[str] = []
    network_created = volume_created = False
    token: str | None = None
    environment = {key: os.environ[key] for key in ("PATH", "HOME") if key in os.environ}

    def docker(*arguments: str, token: str | None = None) -> str:
        env = environment | ({"DINKSTER_REGISTRY_TOKEN": token} if token else {})
        result = subprocess.run(
            ["docker", *arguments],
            env=env,
            capture_output=True,
            text=True,
            timeout=max(1, deadline - time.monotonic()),
            check=False,
        )
        if result.returncode:
            message = result.stderr.replace(token, "<redacted>") if token else result.stderr
            raise RuntimeError(f"disposable command failed: {message}")
        return result.stdout.strip()

    service_id = docker("image", "inspect", service_image, "--format", "{{.Id}}")
    publisher_id = docker("image", "inspect", publisher_image, "--format", "{{.Id}}")
    bases = sorted(inputs.glob("bases/*/manifest.json"))
    locks = sorted(inputs.glob("locks/*.pack-lock.json"))
    if not bases or not locks or not (inputs / "sources/official-packs.lock").is_file():
        raise ValueError("disposable publication requires sources, bases and native locks")
    if not package_index.is_dir():
        raise ValueError("a staged package index is required")

    def run_container(role: str, image: str, *arguments: str) -> str:
        container = name + "-" + role
        # Register before starting so a failed start cannot leave an owned container behind.
        containers.append(container)
        docker(
            "run",
            "--detach",
            "--pull=never",
            "--name",
            container,
            "--network",
            name,
            "--network-alias",
            role,
            "--dns",
            "127.0.0.1",
            *arguments,
            image,
        )
        return container

    try:
        docker("network", "create", "--internal", name)
        network_created = True
        docker("volume", "create", name)
        volume_created = True
        common = (
            "--mount",
            f"type=volume,src={name},dst=/data",
            "--env",
            "DINKSTER_DATABASE_URL=sqlite:////data/registry.sqlite",
            "--env",
            "DINKSTER_OBJECT_STORE_ROOT=/data/objects",
            "--env",
            "DINKSTER_PUBLIC_BASE_URL=http://registry:8000",
            "--env",
            "DINKSTER_PROBE_PACKAGE_INDEX_URL=http://wheels:8000/simple/",
            "--env",
            "DINKSTER_PROBE_DEPENDENCY_SOURCE=index",
        )
        if (inputs / "engine/release-manifest.json").is_file():
            common += (
                "--mount",
                f"type=bind,src={(inputs / 'engine').resolve()},dst=/engine,readonly",
                "--env",
                "DINKSTER_ENV=development",
                "--env",
                "DINKSTER_PROBE_ENGINE_RELEASE=/engine",
            )
        registry = run_container(
            "registry",
            service_id,
            *common,
            "--user",
            "root",
        )
        wheels = name + "-wheels"
        containers.append(wheels)
        docker(
            "run",
            "--detach",
            "--pull=never",
            "--name",
            wheels,
            "--network",
            name,
            "--network-alias",
            "wheels",
            "--dns",
            "127.0.0.1",
            "--mount",
            f"type=bind,src={package_index.resolve()},dst=/index,readonly",
            "--entrypoint",
            "python",
            service_id,
            "-m",
            "http.server",
            "8000",
            "--directory",
            "/index",
        )
        ready = (
            "import time, urllib.request; end=time.monotonic()+60\n"
            "while True:\n"
            " try: urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=2); break\n"
            " except OSError:\n"
            "  if time.monotonic() >= end: raise\n"
            "  time.sleep(.2)\n"
        )
        docker("exec", registry, "python", "-c", ready)
        docker("exec", registry, "dinkster-registry", "admin", "add-user", "dinkster")
        docker(
            "exec",
            registry,
            "dinkster-registry",
            "admin",
            "add-publisher",
            "dinkster",
            "--owner",
            "dinkster",
        )
        token = docker(
            "exec",
            registry,
            "dinkster-registry",
            "admin",
            "mint-token",
            "dinkster",
            "--user",
            "dinkster",
            "--expires",
            (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        )
        scanner = name + "-scanner"
        containers.append(scanner)
        docker(
            "run",
            "--detach",
            "--pull=never",
            "--name",
            scanner,
            "--network",
            name,
            "--dns",
            "127.0.0.1",
            "--user",
            "root",
            "--privileged",
            *common,
            service_id,
            "dinkster-registry",
            "scanner",
        )
        publisher = name + "-publisher"
        containers.append(publisher)
        pythonpath = ":".join(
            [
                "/source",
                "/source/src",
                "/source/packages/dinkster-supervisor/shared",
                *[
                    "/source/" + str(path.relative_to(root))
                    for path in sorted((root / "packages").glob("*/src"))
                ],
            ]
        )
        docker(
            "run",
            "--detach",
            "--pull=never",
            "--name",
            publisher,
            "--network",
            name,
            "--dns",
            "127.0.0.1",
            "--user",
            "root",
            "--mount",
            f"type=bind,src={root.resolve()},dst=/source,readonly",
            "--mount",
            f"type=bind,src={inputs.resolve()},dst=/inputs,readonly",
            "--env",
            "PYTHONPATH=" + pythonpath,
            "--env",
            "DINKSTER_REGISTRY_TOKEN",
            "--entrypoint",
            "sleep",
            publisher_id,
            "infinity",
            token=token,
        )
        denied = (
            "import socket\n"
            "try: socket.create_connection(('1.1.1.1',443),timeout=2)\n"
            "except OSError: pass\n"
            "else: raise RuntimeError('external network is reachable')\n"
        )
        for container in (registry, scanner, publisher, wheels):
            docker("exec", container, "python", "-c", denied)
        arguments = [
            "--sources",
            "/inputs/sources",
            "--registry",
            "http://registry:8000",
            "--publish",
        ]
        if (inputs / "engine/release-manifest.json").is_file():
            arguments.extend(["--engine-manifest", "/inputs/engine/release-manifest.json"])
        for path in bases:
            arguments.extend(["--base-manifest", "/inputs/" + str(path.relative_to(inputs))])
        for path in locks:
            arguments.extend(["--pack-lock", "/inputs/" + str(path.relative_to(inputs))])
        submissions = []
        for attempt in ("submitted", "already-active"):
            submissions.append(
                json.loads(
                    docker(
                        "exec",
                        publisher,
                        "python",
                        "-m",
                        "scripts.publish_official_packs",
                        *arguments,
                        "--output",
                        "/tmp/" + attempt,
                        token=token,
                    )
                )
            )
            verify = (
                "import json,sys; from pathlib import Path; "
                "from scripts.check_disposable_publication import verify_active; "
                "verify_active(Path('/inputs'),json.loads(sys.argv[1]),float(sys.argv[2]))"
            )
            docker(
                "exec",
                publisher,
                "python",
                "-c",
                verify,
                json.dumps(submissions[-1]),
                str(max(1, deadline - time.monotonic())),
                token=token,
            )
        return {
            "serviceImage": service_id,
            "publisherImage": publisher_id,
            "externalNetworkDenied": True,
            "submissions": submissions,
            "elapsedSeconds": time.monotonic() - start,
        }
    except Exception:
        # Retain scanner diagnostics before deleting the disposable database.
        report = (
            "import json,sqlite3; db=sqlite3.connect('/data/registry.sqlite'); "
            "print(json.dumps(db.execute('SELECT stage,operator_details,doctor "
            "FROM release_scan_runs ORDER BY created_at DESC LIMIT 10').fetchall()))"
        )
        try:
            result = subprocess.run(
                ["docker", "exec", name + "-registry", "python", "-c", report],
                env=environment,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if result.returncode == 0:
                diagnostic = result.stdout[-65536:]
                print(
                    diagnostic.replace(token, "<redacted>") if token else diagnostic,
                    file=sys.stderr,
                )
        except (OSError, subprocess.TimeoutExpired):
            pass
        raise
    finally:
        # Cleanup has its own bound even when the publication deadline expired.
        deadline = time.monotonic() + 60
        for container in reversed(containers):
            subprocess.run(["docker", "rm", "--force", container], capture_output=True, timeout=15)
        if volume_created:
            docker("volume", "rm", name)
        if network_created:
            docker("network", "rm", name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--package-index", required=True, type=Path)
    parser.add_argument("--service-image", required=True)
    parser.add_argument("--publisher-image", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            check_publication(
                Path(__file__).resolve().parents[1],
                args.inputs.resolve(),
                args.package_index.resolve(),
                args.service_image,
                args.publisher_image,
            )
        )
    )
