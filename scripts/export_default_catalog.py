"""Export a sealed checkout snapshot's default pack catalog for frontend decoding."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
from importlib.metadata import version
from pathlib import Path

from verify_release_launch import available_port, stop


def export_catalog(output: Path, pack_sources: Path | None = None) -> None:
    sources: list[Path] = []
    if pack_sources is not None:
        from dinkster.registries import default_pack_minimums

        available = {
            tomllib.loads(manifest.read_text(encoding="utf-8"))["pack"]["name"]: manifest.parent
            for manifest in pack_sources.resolve().glob("packages/*/dinkster-pack.toml")
        }
        for name in sorted(default_pack_minimums()):
            if name not in available:
                raise ValueError(f"default pack source missing: {name}")
            sources.append(available[name])
    with tempfile.TemporaryDirectory(prefix="dinkster-catalog-") as directory:
        root = Path(directory)
        environment = {
            **os.environ,
            "DINKSTER_HOME": str(root / "home"),
            "XDG_DATA_HOME": str(root / "user-data"),
            "DINKSTER_REMOTE_CATALOG_BASE": "",
            "DINKSTER_REMOTE_GATEWAY_BASE": "",
        }
        # Published packs are the default. Breaking API development selects the
        # matching pack sources explicitly, using the same snapshot provisioner.
        management = [sys.executable, "-m", "dinkster_supervisor.environment"]
        management_environment = {
            key: value
            for key, value in environment.items()
            if key
            not in {
                "VIRTUAL_ENV",
                "PYTHONPATH",
                "PYTHONHOME",
                "UV_PROJECT",
                "UV_PROJECT_ENVIRONMENT",
            }
        }
        project = root / "project"
        port = available_port()
        setup = subprocess.run(
            [
                *management,
                "--root",
                str(project),
                *(["develop"] if pack_sources is not None else ["setup", "--from-scratch"]),
                "--source",
                str(Path(__file__).resolve().parents[1]),
                *[argument for source in sources for argument in ("--pack-source", str(source))],
                *(
                    [
                        "--library-source",
                        str(pack_sources.resolve() / "packages/dinkster-pack-helpers"),
                    ]
                    if pack_sources is not None
                    else []
                ),
                "--variant",
                "linux-cpu",
                "--python",
                sys.executable,
                "--inference",
                version("dinkster-inference"),
                "--data",
                str(root / "store"),
                "--port",
                str(port),
                "--yes",
                "--json",
            ],
            env=management_environment,
            stdout=subprocess.PIPE,
            text=True,
            check=False,
        )
        if setup.returncode:
            raise RuntimeError(f"default snapshot setup failed: {setup.stdout}")
        print(setup.stdout, file=sys.stderr, end="")
        result = json.loads(setup.stdout)
        if result["status"] == "applied":
            subprocess.run(
                [
                    *management,
                    "--root",
                    str(project),
                    "stop",
                    "--yes",
                    "--json",
                ],
                env=management_environment,
                check=True,
            )
        if result["status"] != "applied" or result.get("unavailablePacks"):
            raise RuntimeError(f"default snapshot setup incomplete: {result}")
        generation = project / "builds" / str(result["current"])
        url = f"http://127.0.0.1:{port}/api/nodes"
        with tempfile.TemporaryFile(mode="w+") as log:
            backend = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "dinkster.serve",
                    "--generation",
                    str(generation),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--library-root",
                    "",
                    "--strict-packs",
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                deadline = time.monotonic() + 300
                while time.monotonic() < deadline:
                    if backend.poll() is not None:
                        raise RuntimeError(f"backend exited with {backend.returncode}")
                    try:
                        with urllib.request.urlopen(url, timeout=1) as response:
                            payload = response.read()
                        catalog = json.loads(payload)
                        if "composing" not in catalog:
                            if not catalog["nodes"] or not set(catalog["packs"]) - {"core"}:
                                raise RuntimeError("default snapshot catalog has no pack nodes")
                            output.write_bytes(payload)
                            return
                    except (urllib.error.URLError, TimeoutError):
                        pass
                    time.sleep(0.05)
                raise RuntimeError("default catalog did not finish composing within 300 seconds")
            except Exception:
                log.seek(0)
                print(log.read(), file=sys.stderr)
                raise
            finally:
                stop(backend)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pack-sources", type=Path, help="checkout containing every default pack")
    args = parser.parse_args()
    export_catalog(args.output, args.pack_sources)
