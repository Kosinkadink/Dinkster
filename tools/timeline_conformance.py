"""Synthetic 60s/720p CPU timeline RSS gate; not an editor-export corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

LIMIT = 2 * 1024**3


def source(path: Path, offset: int) -> None:
    import av
    import numpy as np

    with av.open(str(path), "w", format="mp4") as container:
        output = cast(Any, container)
        stream = output.add_stream("libx264", rate=24)
        stream.width, stream.height = 1280, 720
        stream.pix_fmt = "yuv420p"
        stream.codec_context.thread_count = 1
        stream.options = {"preset": "ultrafast", "crf": "18"}
        for index in range(31 * 24):
            pixels = np.empty((720, 1280, 3), np.uint8)
            pixels[..., 0] = (index + offset) % 256
            pixels[..., 1] = np.arange(1280, dtype=np.uint16) % 256
            pixels[..., 2] = offset
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 24)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def measure(root: Path) -> dict[str, object]:
    import resource

    import av
    from dinkster_assets import AssetRef, digest_file
    from dinkster_values import video_from_source
    from dinkster_values.timeline_video import TimelineVideo
    from dinkster_values.video_document import encode_document, video_reference
    from dinkster_video import save_video_stream
    from dinkster_video.document import clip, make, mutate

    paths = {digest_file(p): p for p in (root / "a.mp4", root / "b.mp4")}

    class Resolver:
        def resolve(self, digest: str) -> Path | None:
            return paths.get(digest)

    resolver = Resolver()
    doc = make(clips=[clip("a", start=0.5, duration=30), clip("b", start=0.5, duration=30)])
    for label, (digest, path) in zip(("a", "b"), paths.items(), strict=True):
        asset = AssetRef(digest, path.name, path.stat().st_size, resolver=resolver)
        doc = mutate(
            doc,
            "bind_source",
            {"source": label, "reference": video_reference(video_from_source(asset))},
        )
    doc = mutate(doc, "transition", {"in_offset": 0.5, "out_offset": 0.5})
    for index in (0, 2):
        doc = mutate(
            doc,
            "set_effect",
            {
                "clip": index,
                "effect": {
                    "node_type": "dinkster.image.draw_text",
                    "parameters": {
                        "text": "Bounded CPU timeline",
                        "font_size": 48,
                        "x": 32,
                        "y": 32,
                    },
                },
            },
        )
    encoded = encode_document(doc)
    output_path = root / "timeline.mp4"
    with output_path.open("wb") as output:
        save_video_stream(
            TimelineVideo(doc, lambda wire: AssetRef.from_wire(wire, resolver)), output
        )
    with av.open(str(output_path)) as opened:
        frames = cast(Any, opened).streams.video[0].frames
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
    result = {
        "platform": platform.platform(),
        "peak_rss_bytes": peak_bytes,
        "limit_bytes": LIMIT,
        "frames": frames,
        "document_bytes": len(encoded),
        "encoded_bytes": output_path.stat().st_size,
        "corpus": {
            p.name: {
                "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in paths.values()
        },
    }
    if peak_bytes >= LIMIT or frames != 1440 or len(encoded) >= 1024**2:
        raise RuntimeError(json.dumps(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.measure:
        print(json.dumps(measure(args.measure), sort_keys=True))
        return
    with tempfile.TemporaryDirectory(prefix="dinkster-timeline-") as directory:
        root = Path(directory)
        source(root / "a.mp4", 0)
        source(root / "b.mp4", 100)
        subprocess.run([sys.executable, __file__, "--measure", directory], check=True)


if __name__ == "__main__":
    main()
