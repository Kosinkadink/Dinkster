"""CPU VIDEO value/save memory gate: python tools/video_conformance.py --rss."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

RSS_LIMIT = 200 * 1024 * 1024


def _peak_rss() -> int:
    if sys.platform == "win32":
        import psutil

        return cast(Any, psutil.Process().memory_info()).peak_wset
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _rss_usage(baseline: int) -> dict[str, int]:
    peak = _peak_rss()
    usage = {
        "baseline_rss": baseline,
        "peak_rss": peak,
        "rss_growth": peak - baseline,
        "limit": RSS_LIMIT,
    }
    if usage["rss_growth"] >= RSS_LIMIT:
        raise RuntimeError(json.dumps(usage))
    return usage


def _source(path: Path) -> None:
    import av
    import numpy as np

    with av.open(str(path), "w", format="mp4") as opened:
        output = cast(Any, opened)
        stream = output.add_stream("libx264", rate=30)
        stream.width, stream.height = 1920, 1080
        stream.pix_fmt = "yuv420p"
        stream.codec_context.thread_count = 1
        stream.options = {"preset": "fast", "crf": "18"}
        for index in range(300):
            pixels = np.zeros((1080, 1920, 3), np.uint8)
            pixels[:, :, 0] = index % 256
            pixels[:, :, 1] = np.arange(1920, dtype=np.uint16) % 256
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 30)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def _measure(path: Path, self_test_allocation_mib: int = 0) -> dict[str, object]:
    import av
    from dinkster_assets import AssetRef, MountSnapshotResolver, digest_file
    from dinkster_nodes_media_io import LoadVideoValue, SaveVideoValue, TrimVideo
    from dinkster_values import video_meta

    class Resolver:
        def resolve(self, digest: str) -> Path:
            return path

    baseline = _peak_rss()
    if self_test_allocation_mib:
        allocation = bytearray(self_test_allocation_mib * 1024 * 1024)
        allocation[::4096] = b"x" * len(allocation[::4096])
        return dict(_rss_usage(baseline))
    ref = AssetRef(digest_file(path), path.name, path.stat().st_size, resolver=Resolver())
    value = LoadVideoValue.execute(video=ref)["video"]
    value = TrimVideo.execute(video=value, start_time=1.01, duration=7.5)["video"]
    with tempfile.TemporaryDirectory(prefix="dinkster-video-output-") as directory:
        root = Path(directory)
        index, snapshot = root / "index.json", root / "mounts.json"
        index.write_text("{}", encoding="utf-8")
        snapshot.write_text(
            json.dumps(
                {
                    "mounts": [
                        {
                            "id": "comfy-output",
                            "root": str(root),
                            "index": str(index),
                            "mode": "readwrite",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        previous = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT")
        os.environ["DINKSTER_MOUNTS_SNAPSHOT"] = str(snapshot)
        try:
            saved = cast(AssetRef, SaveVideoValue.execute(video=value)["video"])
            resolved = AssetRef.from_wire(saved.to_wire(), MountSnapshotResolver(snapshot))
            with resolved.open() as output, av.open(output, mode="r") as opened:
                stream = cast(Any, opened).streams.video[0]
                frames = stream.frames
            size, suffix, mime = saved.size, Path(saved.name).suffix, saved.media_type
        finally:
            if previous is None:
                os.environ.pop("DINKSTER_MOUNTS_SNAPSHOT", None)
            else:
                os.environ["DINKSTER_MOUNTS_SNAPSHOT"] = previous
    result = {
        **_rss_usage(baseline),
        "encoded_bytes": size,
        "frames": frames,
        "suffix": suffix,
        "mime": mime,
        "meta": dict(video_meta(value)),
    }
    if frames != 225:
        raise RuntimeError(f"expected 225 selected frames, received {frames}")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rss", action="store_true", required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--generate-source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--self-test-allocation-mib", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.generate_source:
        _source(args.generate_source)
        return
    if args.source:
        print(json.dumps(_measure(args.source, args.self_test_allocation_mib), sort_keys=True))
        return
    with tempfile.TemporaryDirectory(prefix="dinkster-video-rss-") as directory:
        path = Path(directory) / "source.mp4"
        generated = subprocess.run(
            [sys.executable, __file__, "--rss", "--generate-source", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if generated.returncode:
            raise RuntimeError(generated.stderr)
        command = [sys.executable, __file__, "--rss", "--source", str(path)]
        if args.self_test_allocation_mib:
            command.extend(["--self-test-allocation-mib", str(args.self_test_allocation_mib)])
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stderr)
        measured = json.loads(result.stdout)
        measured["corpus"] = {
            "generator": "PyAV 16.0.1 libx264 1080p 30fps 10s RGB ramp",
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(json.dumps(measured, sort_keys=True))


if __name__ == "__main__":
    main()
