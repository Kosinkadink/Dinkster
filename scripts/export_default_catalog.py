"""Export the composed default catalog from a running backend for frontend decoding."""

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
    environment = {
        **os.environ,
        "DINKSTER_SERVING_PYTHON": sys.executable,
        "DINKSTER_REMOTE_CATALOG_BASE": "",
        "DINKSTER_REMOTE_GATEWAY_BASE": "",
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dinkster.manager",
            "prepare-catalogs",
            "--defaults",
            "--library-root",
            "",
        ],
        env=environment,
        check=True,
    )
    port = available_port()
    url = f"http://127.0.0.1:{port}/api/nodes"
    with tempfile.TemporaryFile(mode="w+") as log:
        backend = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "dinkster.serve",
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
                    if "composing" not in json.loads(payload):
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
