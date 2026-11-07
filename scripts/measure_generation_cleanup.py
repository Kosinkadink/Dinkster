"""Measure allocated Linux fixture blocks across automatic and explicit collection.

Run with the repository's development interpreter as
python -m scripts.measure_generation_cleanup --output NEW_DIRECTORY.
Only newly created disposable installations are operated on; they remain for inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from threading import Event, Thread
from typing import Any

from tests.fixtures.environments.build import build, build_base, executable_release


def allocated(root: Path, store: Path) -> int:
    result = subprocess.check_output(
        ["du", "-s", "--block-size=1", "--", str(root), str(store)],
        text=True,
        stderr=subprocess.PIPE,
    )
    return sum(int(line.split()[0]) for line in result.splitlines())


def scenario(output: Path, changed_base: bool) -> dict[str, Any]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    fixture = build(output, "linux-cpu", port)
    root, store, mirror = (Path(fixture[key]) for key in ("root", "store", "mirror"))
    environment = {**os.environ, "XDG_DATA_HOME": str(output / "user-data")}
    command = [sys.executable, "-m", "dinkster_supervisor.environment", "--root", str(root)]

    def run(*arguments: str) -> dict[str, Any]:
        completed = subprocess.run(
            [*command, *arguments, "--json"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        return json.loads(completed.stdout)

    setup = fixture["commands"]["setup"]
    try:
        run(*setup[setup.index("setup") : -1])
        (mirror / "channels/main.json").write_bytes(Path(fixture["updatedChannel"]).read_bytes())
        run("update", "--yes")
        protected: dict[Path, bytes] = {}
        for name in ("workflows", "models", "outputs", "mounts"):
            directory = root / name
            directory.mkdir(exist_ok=True)
            path = directory / "sentinel"
            path.write_bytes(name.encode("ascii") + b" private data")
            protected[path] = path.read_bytes()
        protected.update({path: path.read_bytes() for path in (root / "history").glob("*.json")})
        base = json.loads(Path(fixture["baseManifest"]).read_bytes())
        if changed_base:
            with tempfile.TemporaryDirectory(dir=output, prefix="new-base-") as scratch:
                base = build_base(mirror, Path(scratch), "linux-cpu", version="2.0")
            channel = json.loads((mirror / "channels/main.json").read_bytes())
            channel["base"]["linux-cpu"] = {
                "id": base["id"],
                "locations": [
                    {"type": "mirror", "path": f"base/linux-cpu/{base['id'][7:]}/manifest.json"}
                ],
            }
            (mirror / "channels/main.json").write_text(json.dumps(channel))
        executable_release(mirror, base, "3.0", 73)
        before = allocated(root, store)
        samples = [before]
        rejected = 0
        stopped = Event()

        def sample() -> None:
            nonlocal rejected
            while not stopped.wait(0.05):
                try:
                    samples.append(allocated(root, store))
                except subprocess.CalledProcessError:
                    rejected += 1

        sampler = Thread(target=sample)
        sampler.start()
        try:
            updated = run("update", "--yes")
        finally:
            stopped.set()
            sampler.join()
        assert updated["current"] == 3, updated
        assert sorted(path.name for path in (root / "generations").iterdir()) == ["2", "3"]
        after = allocated(root, store)
        samples.append(after)
        assert all(path.read_bytes() == data for path, data in protected.items())
        caches = run("gc", "--yes")
        assert all(path.read_bytes() == data for path, data in protected.items())
        return {
            "changedBase": changed_base,
            "beforeUpdateAllocatedBytes": before,
            "observedPeakAllocatedBytes": max(samples),
            "afterAutomaticCollectionAllocatedBytes": after,
            "afterExplicitCacheGCAllocatedBytes": allocated(root, store),
            "samples": len(samples),
            "rejectedConcurrentScans": rejected,
            "cleanupAttempt": updated.get("cleanupAttempt"),
            "explicitCacheGC": caches,
        }
    finally:
        if (root / "current").exists():
            run("stop", "--yes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("this allocation measurement requires Linux and GNU du")
    output = args.output.absolute()
    output.mkdir()
    report = {
        "sourceCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "filesystem": subprocess.check_output(
            ["findmnt", "-n", "-o", "FSTYPE", "-T", str(output)], text=True
        ).strip(),
        "method": (
            "GNU du allocated blocks over project+store in one scan; hardlinks counted once. "
            "Reflink extents are not deduplicated. Peak samples are non-atomic observations, "
            "not a continuous exact peak or filesystem free-space measurement."
        ),
        "scenarios": [
            scenario(output / name, changed)
            for name, changed in (("engine-only", False), ("changed-base", True))
        ],
    }
    (output / "measurements.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
