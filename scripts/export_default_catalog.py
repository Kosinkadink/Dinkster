"""Export a sealed default generation's live catalog for frontend decoding."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from verify_release_launch import available_port, stop


def export_catalog(output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="dinkster-catalog-") as directory:
        root = Path(directory)
        environment = {
            **os.environ,
            "XDG_DATA_HOME": str(root / "user-data"),
            "DINKSTER_REMOTE_CATALOG_BASE": "",
            "DINKSTER_REMOTE_GATEWAY_BASE": "",
        }
        project = root / "project"
        port = available_port()
        setup = subprocess.run(
            [
                sys.executable,
                "-m",
                "dinkster_supervisor.environment",
                "--root",
                str(project),
                "setup",
                "--mirror",
                "https://releases.dinkster.org",
                "--channel",
                "main",
                "--variant",
                "linux-cpu",
                "--data",
                str(root / "store"),
                "--port",
                str(port),
                "--yes",
                "--json",
            ],
            env=environment,
            stdout=subprocess.PIPE,
            text=True,
            check=False,
        )
        if setup.returncode:
            raise RuntimeError(f"default generation setup failed: {setup.stdout}")
        print(setup.stdout, file=sys.stderr, end="")
        result = json.loads(setup.stdout)
        if result["status"] == "applied":
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "dinkster_supervisor.environment",
                    "--root",
                    str(project),
                    "stop",
                    "--yes",
                    "--json",
                ],
                env=environment,
                check=True,
            )
        if result["status"] != "applied" or result.get("unavailablePacks"):
            raise RuntimeError(f"default generation setup incomplete: {result}")
        generation = project / "generations" / str(result["current"])
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
                                raise RuntimeError("default generation catalog has no pack nodes")
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
    export_catalog(parser.parse_args().output)
