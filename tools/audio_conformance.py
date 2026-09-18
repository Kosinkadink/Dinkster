"""CPU AUDIO memory gate: a real two-hour stereo FLAC, lazy trim, and streaming save."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch


def generate(path: Path) -> None:
    import av
    import numpy as np

    rate = 48_000
    samples = np.arange(rate, dtype=np.float32)
    tone = np.sin(samples * (2 * np.pi * 440 / rate)) * 0.125
    pcm = np.ascontiguousarray(np.stack((tone, tone * 0.5)).T.reshape(1, -1))
    with av.open(str(path), "w", format="flac") as container:
        stream = cast(Any, container.add_stream("flac", rate=rate, layout="stereo"))
        for second in range(7200):
            frame = av.AudioFrame.from_ndarray(pcm, format="flt", layout="stereo")
            frame.sample_rate, frame.pts = rate, second * rate
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))


def measure(path: Path) -> dict[str, object]:
    import numpy as np
    import psutil
    from dinkster_assets import AssetRef, MountSnapshotResolver, digest_file
    from dinkster_nodes_media_io.audio import LoadAudio, SaveAudio
    from dinkster_nodes_media_io.audio_ops import TrimAudio
    from dinkster_values.audio_codec import audio_window, effective_audio_facts

    class Resolver:
        def resolve(self, digest: str) -> Path:
            return path

    root = path.parent / "output"
    root.mkdir()
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
    os.environ["DINKSTER_MOUNTS_SNAPSHOT"] = str(snapshot)
    ref = AssetRef(digest_file(path), path.name, path.stat().st_size, resolver=Resolver())
    baseline = psutil.Process().memory_info().rss
    largest_allocation = 0

    def guard(function):
        def checked(shape, *args, **kwargs):
            nonlocal largest_allocation
            size = math.prod(shape) if isinstance(shape, (tuple, list)) else int(shape)
            dtype = np.dtype(kwargs.get("dtype", args[0] if args else np.float64))
            largest_allocation = max(largest_allocation, size * dtype.itemsize)
            if size * dtype.itemsize > 8 * 1024 * 1024:
                raise AssertionError("consumer attempted an unbounded PCM allocation")
            return function(shape, *args, **kwargs)

        return checked

    with patch.object(np, "zeros", guard(np.zeros)), patch.object(np, "empty", guard(np.empty)):
        value = LoadAudio.execute(audio=ref)["audio"]
        source_facts = effective_audio_facts(value)
        trimmed = TrimAudio.execute(audio=value, start=7190.0, duration=10.0)["audio"]
        saved = cast(list[AssetRef], SaveAudio.execute(audio=trimmed)["audios"])[0]
        resolved = AssetRef.from_wire(saved.to_wire(), MountSnapshotResolver(snapshot))
        output = LoadAudio.execute(audio=resolved)["audio"]
        facts = effective_audio_facts(output)
        # The source and result use the same lossless FLAC PCM encoding.
        np.testing.assert_array_equal(
            audio_window(trimmed, 47, 1000)["waveform"], audio_window(output, 47, 1000)["waveform"]
        )
    if sys.platform == "win32":
        peak = cast(Any, psutil.Process().memory_info()).peak_wset
    else:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = int(peak if sys.platform == "darwin" else peak * 1024)
    report = {
        "baseline_rss_bytes": baseline,
        "peak_rss_bytes": peak,
        "rss_growth_bytes": max(0, peak - baseline),
        "limit_bytes": 64 * 1024 * 1024,
        "largest_guarded_allocation_bytes": largest_allocation,
        "source_frames": source_facts["frames"],
        "saved_frames": facts["frames"],
        "source_bytes": path.stat().st_size,
        "saved_bytes": saved.size,
    }
    if report["rss_growth_bytes"] >= 64 * 1024 * 1024 or facts["frames"] != 480000:
        raise AssertionError(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rss", action="store_true", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--measure", action="store_true")
    args = parser.parse_args()
    source = args.directory / "two-hour.flac"
    if args.measure:
        print(json.dumps(measure(source), sort_keys=True))
    else:
        generate(source)
        result = subprocess.run(
            [sys.executable, __file__, "--rss", "--directory", str(args.directory), "--measure"],
            check=True,
            text=True,
            capture_output=True,
        )
        print(result.stdout, end="")


if __name__ == "__main__":
    main()
