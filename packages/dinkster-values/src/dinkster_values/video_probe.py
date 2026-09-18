"""Source-only VIDEO facts without decoding frames or reading a full file into RAM."""

from __future__ import annotations

import io
import math
import mmap
import struct
from collections.abc import Generator
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO, Protocol, cast

from .media_containers import MediaBuffer, bmff_boxes, ebml_elements


class VideoSource(Protocol):
    def open(self) -> BinaryIO: ...


@contextmanager
def open_video_source(source: bytes | VideoSource | Path) -> Generator[BinaryIO]:
    opened = (
        io.BytesIO(source)
        if isinstance(source, bytes)
        else source.open("rb")
        if isinstance(source, Path)
        else source.open()
    )
    with opened as handle:
        yield handle


def _matrix_rotation(matrix: tuple[int, ...]) -> float:
    a, b, _, c, d, *_ = matrix
    sx, sy = math.hypot(a, c), math.hypot(b, d)
    if not sx or not sy:
        raise ValueError("singular video display matrix")
    rotation = -math.degrees(math.atan2(b / sy, a / sx))
    return 0.0 if rotation == 0 else rotation


def _bmff_rotation(data: MediaBuffer, track_id: int) -> float:
    for kind, movie_start, movie_end in bmff_boxes(data, 0, len(data)):
        if kind != b"moov":
            continue
        for kind, track_start, track_end in bmff_boxes(data, movie_start, movie_end):
            if kind != b"trak":
                continue
            for kind, start, end in bmff_boxes(data, track_start, track_end):
                if kind != b"tkhd":
                    continue
                if end - start < 84 or data[start] not in (0, 1):
                    raise ValueError("invalid video track header")
                extra = 12 if data[start] == 1 else 0
                if end - start < 84 + extra:
                    raise ValueError("truncated video track header")
                if struct.unpack_from(">I", data, start + 12 + (8 if extra else 0))[0] == track_id:
                    return _matrix_rotation(struct.unpack_from(">9i", data, start + 40 + extra))
    raise ValueError("video track has no display matrix header")


def _ebml_float(data: MediaBuffer, start: int, end: int) -> float:
    if end - start not in (4, 8):
        raise ValueError("invalid video projection float")
    result: float = struct.unpack_from(">f" if end - start == 4 else ">d", data, start)[0]
    if not math.isfinite(result):
        raise ValueError("video projection angles must be finite")
    return result


def _matroska_rotation(data: MediaBuffer, video_index: int) -> float:
    """Select accepted video TrackEntries in demuxer order, not AVStream.id."""
    ordinal = 0
    for kind, segment_start, segment_end in ebml_elements(data, 0, len(data)):
        if kind != 0x18538067:
            continue
        for kind, tracks_start, tracks_end in ebml_elements(data, segment_start, segment_end):
            if kind != 0x1654AE6B:
                continue
            for kind, start, end in ebml_elements(data, tracks_start, tracks_end):
                if kind != 0xAE:
                    continue
                fields = ebml_elements(data, start, end)
                types = [int.from_bytes(data[s:e], "big") for k, s, e in fields if k == 0x83]
                codecs = [data[s:e] for k, s, e in fields if k == 0x86 and e - s <= 64]
                if types != [1] or len(codecs) != 1 or not codecs[0].startswith(b"V_"):
                    continue
                if ordinal != video_index:
                    ordinal += 1
                    continue
                for kind, video_start, video_end in fields:
                    if kind != 0xE0:
                        continue
                    for kind, start, end in ebml_elements(data, video_start, video_end):
                        if kind != 0x7670:
                            continue
                        projection = dict[int, float]()
                        for k, s, e in ebml_elements(data, start, end):
                            if k == 0x7671:
                                projection[k] = int.from_bytes(data[s:e], "big")
                            elif k in (0x7673, 0x7674, 0x7675):
                                projection[k] = _ebml_float(data, s, e)
                        if projection.get(0x7671, 0) != 0 or projection.get(0x7674, 0) != 0:
                            return 0.0
                        yaw = projection.get(0x7673, 0)
                        if yaw not in (0, 180, -180):
                            return 0.0
                        return projection.get(0x7675, 0)
                return 0.0
    raise ValueError("video stream has no matching Matroska track")


def _container_facts(data: MediaBuffer, format_name: str, track_id: int) -> tuple[str, float]:
    if "mov" in format_name.split(","):
        brands = [
            data[s : s + 4]
            for k, s, e in bmff_boxes(data, 0, len(data))
            if k == b"ftyp" and e - s >= 8
        ]
        container = "mov" if not brands or brands[0] == b"qt  " else "mp4"
        return container, _bmff_rotation(data, track_id)
    if "matroska" in format_name.split(","):
        header = ebml_elements(data, 0, len(data))[0]
        if header[0] != 0x1A45DFA3:
            raise ValueError("Matroska source has no EBML header")
        types = [
            data[s:e]
            for k, s, e in ebml_elements(data, header[1], header[2])
            if k == 0x4282 and e - s <= 32
        ]
        if types not in ([b"webm"], [b"matroska"]):
            raise ValueError("unsupported EBML document type")
        return "webm" if types == [b"webm"] else "mkv", _matroska_rotation(data, 0)
    if format_name == "avi" and data[:4] == b"RIFF" and data[8:12] == b"AVI ":
        return "avi", 0.0
    if format_name == "gif" and data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif", 0.0
    raise ValueError("unsupported video container; expected mp4, mkv, mov, webm, avi, or gif")


def _stream_seconds(stream: Any, field: str) -> Fraction | None:
    value = getattr(stream, field)
    return Fraction(value) * stream.time_base if value is not None and stream.time_base else None


def probe_video(source: bytes | VideoSource | Path) -> dict[str, object]:
    """Probe an admitted source; edits never call this function."""
    import av

    with open_video_source(source) as handle:
        opened = av.open(handle, mode="r")
        try:
            container = cast(Any, opened)
            if not container.streams.video:
                raise ValueError("source has no video stream")
            stream = container.streams.video[0]
            codec = stream.codec_context
            if isinstance(source, bytes):
                kind, rotation = _container_facts(source, container.format.name, stream.id)
            else:
                with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
                    kind, rotation = _container_facts(data, container.format.name, stream.id)
            rate = Fraction(stream.average_rate) if stream.average_rate else None
            frame_count = int(stream.frames) if stream.frames else None
            duration = _stream_seconds(stream, "duration")
            duration_kind = "stream" if duration is not None else "unknown"
            if duration is None and frame_count is not None and rate:
                duration, duration_kind = Fraction(frame_count) / rate, "frames"
            if duration is None and container.duration is not None:
                duration, duration_kind = Fraction(container.duration, av.time_base), "container"
            count_kind = "header" if frame_count is not None else "unknown"
            if frame_count is None and duration is not None and rate:
                frame_count, count_kind = math.ceil(duration * rate), "estimated"
            fmt = codec.format
            alpha = bool(fmt and any(c.is_alpha for c in fmt.components))
            alpha = alpha or stream.metadata.get("alpha_mode") == "1"
            depth = max((c.bits for c in fmt.components), default=8) if fmt else None
            transfer = int(codec.color_trc)
            return {
                "container": kind,
                "video_codec": codec.codec.canonical_name,
                "pix_fmt": fmt.name if fmt else None,
                "bit_depth": depth,
                "alpha": alpha,
                "color_space": {1: "sRGB", 13: "sRGB", 18: "HDR", 16: "HDR PQ"}.get(
                    transfer, "unknown"
                ),
                "primaries": int(codec.color_primaries),
                "transfer": transfer,
                "matrix": int(codec.colorspace),
                "range": int(codec.color_range),
                "width": int(codec.width),
                "height": int(codec.height),
                "rotation": rotation,
                "fps": rate,
                "time_base": Fraction(stream.time_base) if stream.time_base else None,
                "start_time": _stream_seconds(stream, "start_time") or Fraction(0),
                "frame_count": frame_count,
                "frame_count_kind": count_kind,
                "duration": duration,
                "duration_kind": duration_kind,
                "audio": [
                    {
                        "index": int(s.index),
                        "codec": s.codec_context.codec.canonical_name,
                        "sample_rate": int(s.codec_context.sample_rate),
                        "channels": int(s.codec_context.channels),
                        "layout": s.codec_context.layout.name,
                        "time_base": Fraction(s.time_base) if s.time_base else None,
                        "start_time": _stream_seconds(s, "start_time") or Fraction(0),
                        "duration": _stream_seconds(s, "duration"),
                    }
                    for s in container.streams.audio
                ],
            }
        finally:
            opened.close()
