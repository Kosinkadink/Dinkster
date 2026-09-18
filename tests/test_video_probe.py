"""Source facts retain precision, container identity, and display rotation."""

from __future__ import annotations

import io
import struct
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import pytest
from dinkster_assets import AssetRef, AssetVault, digest_bytes
from dinkster_values.media_containers import bmff_boxes
from dinkster_values.video_probe import probe_video


def _encode(container: str, codec: str, pixel_format: str, *, hdr: bool = False) -> bytes:
    buffer = io.BytesIO()
    with av.open(buffer, "w", format=container) as opened:
        output = cast(Any, opened)
        stream = output.add_stream(codec, rate=Fraction(30000, 1001))
        stream.width, stream.height = 64, 32
        stream.pix_fmt = pixel_format
        stream.codec_context.thread_count = 1
        if hdr:
            stream.codec_context.color_primaries = 9
            stream.codec_context.color_trc = 18
            stream.codec_context.colorspace = 9
            stream.codec_context.color_range = 1
        for i in range(3):
            array = np.full((32, 64, 4), 64 + i * 32, dtype=np.uint8)
            array[:, :, 3] = 128
            frame = av.VideoFrame.from_ndarray(array, format="rgba")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("container", "codec", "pixel_format", "expected"),
    [
        ("mp4", "libx264", "yuv420p", "mp4"),
        ("mov", "libx264", "yuv420p", "mov"),
        ("matroska", "ffv1", "bgra", "mkv"),
        ("webm", "libvpx-vp9", "yuv420p", "webm"),
        ("avi", "mpeg4", "yuv420p", "avi"),
        ("gif", "gif", "rgb8", "gif"),
    ],
)
def test_container_and_asset_source(
    tmp_path: Path, container: str, codec: str, pixel_format: str, expected: str
) -> None:
    data = _encode(container, codec, pixel_format)
    facts = probe_video(data)
    assert facts["container"] == expected
    assert (facts["width"], facts["height"]) == (64, 32)
    assert facts["duration"] is not None
    assert facts["audio"] == []
    assert facts["rotation"] == 0
    assert isinstance(facts["fps"], Fraction)
    vault = AssetVault(tmp_path / "vault")
    digest = digest_bytes(data)
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()
    ref = AssetRef(digest, "incorrect-name.png", len(data), resolver=vault)
    assert probe_video(ref) == facts


def test_hdr_depth_color_and_rational_rate() -> None:
    facts = probe_video(_encode("mp4", "libx264", "yuv420p10le", hdr=True))
    assert facts["pix_fmt"] == "yuv420p10le"
    assert facts["bit_depth"] == 10
    assert facts["color_space"] == "HDR"
    assert [facts[k] for k in ("primaries", "transfer", "matrix", "range")] == [9, 18, 9, 1]
    assert facts["fps"] == Fraction(30000, 1001)
    assert facts["frame_count"] == 3
    assert facts["frame_count_kind"] == "header"
    assert facts["duration"] == Fraction(3003, 30000)


def test_alpha_uses_source_pixel_format() -> None:
    facts = probe_video(_encode("matroska", "ffv1", "bgra"))
    assert facts["alpha"] is True
    assert facts["bit_depth"] == 8


def _rotate_mp4(data: bytes) -> bytes:
    result = bytearray(data)
    matrix = (0, -65536, 0, 65536, 0, 0, 0, 0, 1 << 30)
    for k, s, e in bmff_boxes(data, 0, len(data)):
        if k != b"moov":
            continue
        for k, ts, te in bmff_boxes(data, s, e):
            if k != b"trak":
                continue
            for k, hs, he in bmff_boxes(data, ts, te):
                if k == b"tkhd":
                    assert he - hs >= 84
                    struct.pack_into(">9i", result, hs + (52 if data[hs] else 40), *matrix)
                    return bytes(result)
    raise AssertionError("generated MP4 has no track header")


def test_rotation_requires_no_frame_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _rotate_mp4(_encode("mp4", "libx264", "yuv420p"))
    original_open = av.open

    class ProbeOnly:
        def __init__(self, opened: object) -> None:
            self.inner = cast(Any, opened)

        def __getattr__(self, name: str) -> Any:
            if name in ("decode", "demux"):
                raise AssertionError("probe must not decode or iterate media packets")
            return getattr(self.inner, name)

    def guarded_open(*args: Any, **kwargs: Any) -> ProbeOnly:
        return ProbeOnly(original_open(*args, **kwargs))

    monkeypatch.setattr(av, "open", guarded_open)
    assert probe_video(data)["rotation"] == 90


def test_non_video_and_unsupported_container_fail() -> None:
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="wav") as output:
        stream = cast(Any, output).add_stream("pcm_s16le", rate=8000)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(
            np.zeros((1, 800), np.int16), format="s16", layout="mono"
        )
        frame.sample_rate = 8000
        for packet in stream.encode(frame):
            output.mux(packet)
    with pytest.raises(ValueError, match="no video stream"):
        probe_video(buffer.getvalue())
    with pytest.raises(ValueError, match="unsupported video container"):
        probe_video(_encode("flv", "flv1", "yuv420p"))
