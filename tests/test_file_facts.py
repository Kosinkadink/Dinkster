"""Decoded formula facts and their verified, digest-keyed retention."""

from __future__ import annotations

import io
import struct
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import pytest
from dinkster_assets import digest_bytes
from dinkster_assets.integrity import AssetIntegrityError
from dinkster_assets.probe import probe_cached, probe_file
from dinkster_values import file_facts
from dinkster_values.media_containers import bmff_boxes
from PIL import Image


def test_upright_image_and_animation(tmp_path: Path) -> None:
    image = Image.new("RGB", (13, 7), "red")
    exif = Image.Exif()
    exif[274] = 6
    path = tmp_path / "not-an-image.bin"
    image.save(path, format="JPEG", exif=exif)
    assert probe_file(path)["facts"] == {"width": 7, "height": 13, "frames": 1}
    frames = [Image.new("RGB", (13, 7), color) for color in ("red", "green", "blue")]
    frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=[100, 200, 300],
    )
    assert probe_file(path)["facts"] == {
        "width": 13,
        "height": 7,
        "frames": 3,
        "duration": pytest.approx(0.6),
        "fps": pytest.approx(5),
    }


def test_video_exact_frames_and_audio_duration(tmp_path: Path) -> None:
    path = tmp_path / "incorrect-extension.txt"
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp4") as opened:
        output = cast(Any, opened)
        stream = output.add_stream("libx264", rate=Fraction(25))
        stream.width, stream.height = 32, 16
        stream.pix_fmt = "yuv420p"
        for index in range(7):
            frame = av.VideoFrame.from_ndarray(
                np.full((16, 32, 3), index * 20, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    source = buffer.getvalue()
    data = bytearray(source)
    matrix_start = None
    for kind, start, end in bmff_boxes(source, 0, len(source)):
        if kind != b"moov":
            continue
        for kind, track_start, track_end in bmff_boxes(source, start, end):
            if kind != b"trak":
                continue
            for kind, header_start, _ in bmff_boxes(source, track_start, track_end):
                if kind == b"tkhd":
                    matrix_start = header_start + (52 if data[header_start] else 40)
                    struct.pack_into(
                        ">9i",
                        data,
                        matrix_start,
                        0,
                        -65536,
                        0,
                        65536,
                        0,
                        0,
                        0,
                        0,
                        1 << 30,
                    )
    path.write_bytes(data)
    assert probe_file(path)["facts"] == {
        "width": 16,
        "height": 32,
        "frames": 7,
        "fps": 25.0,
        "duration": pytest.approx(7 / 25),
    }
    assert matrix_start is not None
    for a, b, c, d in ((0, 65536, -65536, 0), (-65536, 0, 0, -65536)):
        struct.pack_into(">9i", data, matrix_start, a, b, 0, c, d, 0, 0, 0, 1 << 30)
        path.write_bytes(data)
        expected_size = (32, 16) if b == 0 else (16, 32)
        assert probe_file(path)["facts"] == {
            "width": expected_size[0],
            "height": expected_size[1],
            "frames": 7,
            "fps": 25.0,
            "duration": pytest.approx(7 / 25),
        }
    struct.pack_into(">9i", data, matrix_start, 46341, -46341, 0, 46341, 46341, 0, 0, 0, 1 << 30)
    path.write_bytes(data)
    digest = digest_bytes(bytes(data))
    metadata = probe_cached(path, digest, tmp_path / "metadata")
    assert metadata["facts"] == {
        "frames": 7,
        "fps": 25.0,
        "duration": pytest.approx(7 / 25),
    }
    assert "45 degrees" in str(metadata["factsError"])
    assert "width and height are unknown" in str(metadata["factsError"])
    assert probe_cached(path, digest, tmp_path / "metadata") == metadata
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0\0" * 2400)
    assert probe_file(path)["facts"] == {"duration": pytest.approx(0.3)}
    path.write_text("not media", encoding="ascii")
    assert probe_file(path) == {"format": "unknown"}


def test_concurrent_and_later_reads_share_decoded_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "source"
    Image.new("RGB", (19, 11), "blue").save(path, format="PNG")
    digest = digest_bytes(path.read_bytes())
    cache = tmp_path / "metadata"
    original = file_facts.decoded_file_facts
    entered, release = threading.Event(), threading.Event()
    count = 0

    def decode(handle: Any, media_type: str) -> tuple[dict[str, int | float], str | None]:
        nonlocal count
        count += 1
        entered.set()
        assert release.wait(10)
        return original(handle, media_type)

    monkeypatch.setattr(file_facts, "decoded_file_facts", decode)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(probe_cached, path, digest, cache)
        assert entered.wait(10)
        second = executor.submit(probe_cached, path, digest, cache)
        release.set()
        expected = {"width": 19, "height": 11, "frames": 1}
        assert first.result()["facts"] == expected
        assert second.result()["facts"] == expected
    assert probe_cached(path, digest, cache)["facts"] == expected
    assert count == 1
    moved = tmp_path / "moved-source"
    path.rename(moved)
    with pytest.raises(FileNotFoundError):
        probe_cached(path, digest, cache)
    assert probe_cached(moved, digest, cache)["facts"] == expected
    path.write_bytes(b"not the retained image")
    with pytest.raises(AssetIntegrityError):
        probe_cached(path, digest, cache)
    assert probe_cached(moved, digest, cache)["facts"] == expected
    assert count == 1
