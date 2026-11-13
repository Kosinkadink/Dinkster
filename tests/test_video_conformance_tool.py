from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

TOOL_PATH = Path(__file__).resolve().parent.parent / "tools" / "video_conformance.py"


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("video_conformance", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rss_gate_saves_video_in_a_fresh_process(tmp_path: Path) -> None:
    import av
    import numpy as np

    source = tmp_path / "source.mp4"
    with av.open(str(source), "w", format="mp4") as container:
        stream = cast(av.VideoStream, container.add_stream("libx264", rate=30))
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for _ in range(300):
            frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), np.uint8))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import subprocess, sys; sys.exit(subprocess.run(sys.argv[1:]).returncode)",
            sys.executable,
            str(TOOL_PATH),
            "--rss",
            "--source",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    measured = json.loads(result.stdout)
    assert measured["frames"] == 225
    assert measured["mime"] == "video/mp4"
    assert measured["encoded_bytes"] > 0
    assert 0 <= measured["inference_init_rss"] <= measured["baseline_rss"]
    assert measured["rss_growth"] == measured["peak_rss"] - measured["baseline_rss"]
    assert measured["rss_growth"] < measured["limit"] == 200 * 1024 * 1024
