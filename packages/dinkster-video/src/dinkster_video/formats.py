"""Fixed CPU animated-image and PNG-sequence writers using the shared VIDEO iterator."""

from __future__ import annotations

import inspect
import io
import json
import logging
import struct
import zlib
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from contextlib import ExitStack, closing
from fractions import Fraction
from functools import wraps
from itertools import chain
from math import ceil
from typing import Any, BinaryIO, ParamSpec, cast
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from dinkster_values import coerce_video, effective_video_facts, open_video_source, video_source
from dinkster_values.image_codec import image_color
from dinkster_values.video_edits import mapping

DITHERS = (
    "bayer",
    "heckbert",
    "floyd_steinberg",
    "sierra2",
    "sierra2_4a",
    "sierra3",
    "burkes",
    "atkinson",
    "none",
)
FRAME_FORMATS = ("gif_pillow", "gif_ffmpeg", "webp", "png8", "png16")
_METADATA_KEY = "dinkster_metadata"
_P = ParamSpec("_P")


def _encoded_format_facts(destination: BinaryIO, suffix: str) -> dict[str, object]:
    import av

    position = destination.tell()
    try:
        destination.seek(0)
        kind, codec, pixel_format = suffix.removeprefix("."), "", ""
        layouts: list[str] = []
        audio: list[dict[str, object]] = []
        if kind in ("png", "zip"):
            if kind == "zip":
                with ZipFile(destination) as archive, archive.open("000001.png") as first:
                    header = first.read(33)
            else:
                header = destination.read(33)
            codec = "png" if kind == "zip" else "apng"
            alpha, depth = header[25] == 6, header[24]
            pixel_format = (
                ("rgba64be" if alpha else "rgb48be")
                if depth == 16
                else "rgba"
                if alpha
                else "rgb24"
            )
        elif kind == "gif":
            codec, pixel_format = "gif", "pal8"
        elif kind == "webp":
            destination.seek(12)
            alpha, lossless = False, False
            while header := destination.read(8):
                tag, size = struct.unpack("<4sI", header)
                start = destination.tell()
                if tag == b"VP8X":
                    alpha = bool(destination.read(1)[0] & 16)
                elif tag == b"ANMF":
                    destination.seek(start + 16)
                    lossless = destination.read(4) == b"VP8L"
                    break
                destination.seek(start + size + size % 2)
            codec = "webp"
            pixel_format = (
                ("rgba" if alpha else "rgb24") if lossless else "yuva420p" if alpha else "yuv420p"
            )
        else:
            with av.open(destination, "r") as opened:
                stream = opened.streams.video[0]
                codec = stream.codec_context.codec.canonical_name
                video_format = stream.codec_context.format
                assert video_format is not None
                pixel_format = video_format.name
                if codec == "vp9" and stream.metadata.get("alpha_mode") == "1":
                    pixel_format = "yuva420p"
                for track in opened.streams.audio:
                    layouts.append(track.layout.name)
                    audio.append(
                        {"codec": track.codec_context.name, "channelLayout": track.layout.name}
                    )
        return {
            "container": kind,
            "codec": codec,
            "pixelFormat": pixel_format,
            "channelLayout": layouts[0] if layouts else None,
            "audioStreams": audio,
        }
    finally:
        destination.seek(position)


def encoded_diagnostics(
    encode: Callable[_P, tuple[str, str]],
) -> Callable[_P, tuple[str, str]]:
    """Report substitutions only after success, using the final encoded headers, not attempts."""
    signature = inspect.signature(encode)

    @wraps(encode)
    def run(*args: _P.args, **kwargs: _P.kwargs) -> tuple[str, str]:
        bound = signature.bind(*args, **kwargs)
        report = cast(
            Callable[[Mapping[str, object]], None] | None, bound.arguments.get("on_diagnostic")
        )
        if report is None:
            return encode(*args, **kwargs)
        container, codec = bound.arguments.get("container"), bound.arguments.get("codec")
        format_name = bound.arguments.get("format")
        if format_name is not None:
            container, codec = {
                "gif_pillow": ("gif", "gif"),
                "gif_ffmpeg": ("gif", "gif"),
                "webp": ("webp", "webp"),
                "png8": ("zip", "png"),
                "png16": ("zip", "png"),
                "apng": ("png", "apng"),
            }.get(format_name, (format_name, None))
        layout = bound.arguments.get("audio_layout")
        requested = {
            "container": None if container == "auto" else container,
            "codec": None if codec == "auto" else codec,
            "pixelFormat": None,
            "channelLayout": None if layout == "preserve" else layout,
        }
        notices: list[Mapping[str, object]] = []
        bound.arguments["on_diagnostic"] = notices.append
        result = encode(*bound.args, **bound.kwargs)
        if notices:
            facts = _encoded_format_facts(bound.arguments["destination"], result[0])
            audio = facts.pop("audioStreams")
            report(
                {
                    "code": "media_format_fallback",
                    "requested": requested,
                    "effective": facts,
                    "reason": next(
                        (notice["reason"] for notice in notices if "reason" in notice),
                        "preserving_cpu_default",
                    ),
                    "audioStreams": audio,
                    "substitutions": notices,
                }
            )
        return result

    return run


def format_diagnostic(
    requested: Mapping[str, object],
    effective: Mapping[str, object],
    report: Callable[[Mapping[str, object]], None] | None,
    *,
    reason: str | None = None,
) -> None:
    diagnostic = {
        "code": "media_format_fallback",
        "requested": dict(requested),
        "effective": dict(effective),
    }
    if reason is not None:
        diagnostic["reason"] = reason
    if report is not None:
        report(diagnostic)
    else:
        logging.getLogger(__name__).warning("Using preserving media defaults: %s", diagnostic)


def metadata_json(metadata: Mapping[str, object] | None) -> str:
    data = mapping(metadata if metadata is not None else {}, "video metadata")
    result = json.dumps(data, allow_nan=False, separators=(",", ":"), ensure_ascii=True)
    if len(result) > 1024 * 1024:
        raise ValueError("video metadata exceeds 1 MiB")
    return result


def metadata_tags(metadata: Mapping[str, object] | None) -> dict[str, str]:
    serialized = metadata_json(metadata)
    if not metadata:
        return {}
    tags = {
        k: v if isinstance(v, str) else json.dumps(v, allow_nan=False, separators=(",", ":"))
        for k, v in metadata.items()
    }
    # A typed envelope preserves JSON-looking strings and case-sensitive keys across muxers.
    tags[_METADATA_KEY] = serialized
    tags["comment"] = _METADATA_KEY + ":" + serialized
    return tags


def read_video_metadata(source: object) -> dict[str, object]:
    """Read the exact custom metadata object from a container, animation, or PNG archive."""
    import av
    from PIL import Image

    if isinstance(source, Mapping) and ("components" in source or "source" in source):
        value = coerce_video(cast(Mapping[str, object], source))
        if "source" not in value:
            return {}
        source = video_source(value)
    with open_video_source(cast(Any, source)) as handle:
        magic = handle.read(12)
        handle.seek(0)
        if magic.startswith(b"PK\x03\x04"):
            with ZipFile(handle) as archive, archive.open("000001.png") as first:
                with Image.open(first) as image:
                    raw = image.info.get(_METADATA_KEY, "{}")
        elif magic.startswith((b"GIF8", b"\x89PNG")) or magic[8:12] == b"WEBP":
            with Image.open(handle) as image:
                raw = image.info.get(
                    _METADATA_KEY, image.info.get("xmp", image.info.get("comment", "{}"))
                )
        else:
            with av.open(handle) as opened:
                tags = {k.lower(): v for k, v in opened.metadata.items()}
                raw = tags.get(_METADATA_KEY, tags.get("comment", "{}"))
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if isinstance(raw, str) and raw.startswith(_METADATA_KEY + ":"):
            raw = raw[len(_METADATA_KEY) + 1 :]
        if not isinstance(raw, str) or len(raw) > 1024 * 1024:
            raise ValueError("invalid video metadata envelope")
        return dict(mapping(json.loads(raw), "video metadata"))


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _png_metadata(metadata: str, probe: Mapping[str, object]) -> bytes:
    extra = (
        _png_chunk(b"tEXt", _METADATA_KEY.encode() + b"\0" + metadata.encode()) if metadata else b""
    )
    for key, value in json.loads(metadata or "{}").items():
        try:
            keyword = key.encode("latin-1")
        except UnicodeEncodeError:
            continue
        # Non-PNG keywords remain lossless in the typed envelope.
        if (
            key == _METADATA_KEY
            or not 1 <= len(keyword) <= 79
            or any(byte < 32 or 126 < byte < 161 for byte in keyword)
            or key != key.strip(" ")
            or "  " in key
        ):
            continue
        text = (
            value
            if isinstance(value, str)
            else json.dumps(
                value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
            )
        )
        try:
            encoded = text.encode("utf-8")
        except UnicodeEncodeError:
            continue
        extra += _png_chunk(b"iTXt", keyword + bytes(5) + encoded)
    color = image_color(
        {"primaries": probe["primaries"], "transfer": probe["transfer"], "range": 2}
    )
    primaries, transfer = cast(int, color["primaries"]), cast(int, color["transfer"])
    if primaries <= 255 and transfer <= 255:
        extra += _png_chunk(b"cICP", bytes((primaries, transfer, 0, 1)))
    return extra


def _png(
    frame: Any,
    depth: int,
    alpha: bool,
    metadata: str,
    probe: Mapping[str, object],
    compression: int = 6,
) -> bytes:
    import av

    encoder = cast(Any, av.CodecContext.create("png", "w"))
    encoder.width, encoder.height = frame.width, frame.height
    encoder.pix_fmt = (
        ("rgba64be" if alpha else "rgb48be") if depth == 16 else "rgba" if alpha else "rgb24"
    )
    encoder.thread_count = 1
    encoder.options = {"compression_level": str(compression)}
    frame = frame.reformat(format=encoder.pix_fmt, src_color_range=2, dst_color_range=2)
    data = b"".join(bytes(p) for p in (*encoder.encode(frame), *encoder.encode(None)))
    return data[:33] + _png_metadata(metadata, probe) + data[33:]


def _riff_chunk(kind: bytes, data: bytes) -> bytes:
    return kind + struct.pack("<I", len(data)) + data + (b"\0" if len(data) % 2 else b"")


def _insert_bytes(destination: BinaryIO, start: int, data: bytes) -> None:
    end = destination.seek(0, 2)
    cursor = end
    while cursor > start:
        first = max(start, cursor - 1024 * 1024)
        destination.seek(first)
        chunk = destination.read(cursor - first)
        destination.seek(first + len(data))
        destination.write(chunk)
        cursor = first
    destination.seek(start)
    destination.write(data)
    destination.seek(end + len(data))


def _apng_delay(duration: Fraction) -> bytes:
    delay = duration.limit_denominator(min(65535, max(1, 65535 // ceil(duration))))
    if not 0 < delay.numerator <= 65535:
        raise ValueError("APNG frame duration cannot be represented")
    return struct.pack(">HH", delay.numerator, delay.denominator)


def _gif_metadata(destination: BinaryIO, metadata: str) -> None:
    """Insert a GIF comment after its global palette without holding the encoded file in RAM."""
    destination.seek(10)
    packed = destination.read(1)[0]
    start = 13 + (3 * 2 ** ((packed & 7) + 1) if packed & 128 else 0)
    data = metadata.encode()
    comment = (
        b"!\xfe"
        + b"".join(
            bytes((len(data[i : i + 255]),)) + data[i : i + 255] for i in range(0, len(data), 255)
        )
        + b"\0"
    )
    _insert_bytes(destination, start, comment)


def _uint24(value: int) -> bytes:
    if not 0 <= value < 2**24:
        raise ValueError("animated WebP dimension or duration exceeds 24 bits")
    return value.to_bytes(3, "little")


def _palette(frame: Any, dither: str) -> Any:
    from av.filter import Graph

    graph = cast(Any, Graph())
    source = graph.add_buffer(template=frame)
    split = graph.add("split")
    generate = graph.add("palettegen", "reserve_transparent=1:stats_mode=single")
    apply = graph.add("paletteuse", f"dither={dither}:new=1")
    sink = graph.add("buffersink")
    source.link_to(split)
    split.link_to(generate, 0, 0)
    split.link_to(apply, 1, 0)
    generate.link_to(apply, 0, 1)
    apply.link_to(sink)
    graph.configure()
    source.push(frame)
    source.push(None)
    return sink.pull()


def _frame_durations(
    frames: Iterator[tuple[Fraction, Any]],
    rate: Fraction,
    end: Fraction | None,
) -> Iterator[tuple[Fraction, Fraction, Any]]:
    current = next(frames, None)
    while current is not None:
        following = next(frames, None)
        timestamp, frame = current
        stop = following[0] if following is not None else end or timestamp + 1 / rate
        yield timestamp, stop - timestamp, frame
        current = following


@encoded_diagnostics
def save_video_frames(
    obj: object,
    destination: BinaryIO,
    *,
    format: str = "gif_pillow",
    loop: int = 0,
    dither: str = "sierra2_4a",
    lossless: bool = True,
    quality: int = 80,
    metadata: Mapping[str, object] | None = None,
    on_diagnostic: Callable[[Mapping[str, object]], None] | None = None,
) -> tuple[str, str]:
    """Write an animation or a ZIP of numbered PNGs, never a new VIDEO source type."""
    from .runtime import iter_export_frames, save_video_stream

    if any(token in format for token in ("{", "[", "\n")):
        raise ValueError("custom FFmpeg JSON is refused")
    value = coerce_video(obj)
    probe = mapping(value["probe"], "probe")
    facts = effective_video_facts(value)
    rate = cast(Fraction | None, facts["fps"])
    if rate is None:
        raise ValueError("frame export requires a known frame rate")
    depth = int(cast(int, probe["bit_depth"]) or 8)
    audio = (
        mapping(value["components"], "components").get("audio")
        if "components" in value
        else probe["audio"]
    )
    if (
        format not in FRAME_FORMATS
        or audio
        or (depth > 8 or probe["transfer"] in (16, 18))
        and not format.startswith("png")
    ):
        format_diagnostic({"format": format}, {"container": "mkv", "codec": "ffv1"}, on_diagnostic)
        return save_video_stream(
            value,
            destination,
            container="mkv",
            codec="ffv1",
            metadata=metadata,
            on_diagnostic=on_diagnostic,
        )
    target_depth = 16 if format == "png16" or depth > 8 else 8
    with closing(iter_export_frames(value, target_depth)) as frames:
        return save_frame_records(
            _frame_durations(frames, rate, cast(Fraction | None, facts["duration"])),
            destination,
            format=format,
            bit_depth=depth,
            color={key: probe[key] for key in ("primaries", "transfer", "range", "matrix")},
            loop=loop,
            dither=dither,
            lossless=lossless,
            quality=quality,
            metadata=metadata,
            on_diagnostic=on_diagnostic,
        )


def _checked_records(
    records: Iterator[tuple[Fraction, Fraction, object]], bit_depth: int, frame_count: int | None
) -> Generator[tuple[Fraction, Fraction, Any]]:
    import av
    import numpy as np
    from numpy.typing import NDArray

    from .runtime import _encoder_frame  # pyright: ignore[reportPrivateUsage]

    count = 0
    shape: tuple[int, int, bool] | None = None
    previous_end = Fraction(0)
    for timestamp, duration, item in records:
        if timestamp != previous_end or duration <= 0:
            raise ValueError(
                "frame records require contiguous timestamps from zero and positive durations"
            )
        count += 1
        if frame_count is not None and count > frame_count:
            raise ValueError("frame records exceed declared frame_count")
        if isinstance(item, np.ndarray):
            pixels = cast("NDArray[Any]", item)
            if pixels.dtype != np.float32 or pixels.ndim != 3 or pixels.shape[-1] not in (3, 4):
                raise ValueError("frame records require normalized float32 HWC RGB/RGBA images")
            if not np.isfinite(pixels).all():
                raise ValueError("frame records contain non-finite pixels")
            frame = _encoder_frame(pixels, bit_depth)
        elif isinstance(item, av.VideoFrame) and item.format.is_rgb:
            frame = item
        else:
            raise ValueError("frame records require an RGB/RGBA image or PyAV RGB frame")
        current = (frame.width, frame.height, any(c.is_alpha for c in frame.format.components))
        if shape is not None and current != shape:
            raise ValueError("frame records changed dimensions or channels")
        shape = current
        previous_end = timestamp + duration
        yield timestamp, duration, frame
    if frame_count is not None and count != frame_count:
        raise ValueError("frame records ended before declared frame_count")


@encoded_diagnostics
def save_frame_records(
    records: Iterable[tuple[Fraction, Fraction, object]],
    destination: BinaryIO,
    *,
    format: str,
    bit_depth: int,
    color: Mapping[str, object],
    frame_count: int | None = None,
    loop: int = 0,
    dither: str = "sierra2_4a",
    lossless: bool = True,
    quality: int = 80,
    method: int = 4,
    compression: int = 6,
    metadata: Mapping[str, object] | None = None,
    on_diagnostic: Callable[[Mapping[str, object]], None] | None = None,
) -> tuple[str, str]:
    """Encode bounded (start, duration, RGB frame) records; the caller owns spool/publication."""
    import av
    import numpy as np
    from PIL import GifImagePlugin, Image

    with ExitStack() as scope:
        close = getattr(records, "close", None)
        if callable(close):
            scope.callback(close)
        source = iter(records)
        if source is not records:
            close = getattr(source, "close", None)
            if callable(close):
                scope.callback(close)
        if any(token in format for token in ("{", "[", "\n")):
            raise ValueError("custom FFmpeg JSON is refused")
        if type(bit_depth) is not int or not 0 < bit_depth <= 16:
            raise ValueError("animation/PNG encoders require source precision of 1 through 16 bits")
        if frame_count is not None and (type(frame_count) is not int or frame_count < 1):
            raise ValueError("frame_count must be positive when supplied")
        if type(loop) is not int or not 0 <= loop <= 65535:
            raise ValueError("loop must be an integer in 0..65535 (0 means infinite)")
        if type(lossless) is not bool or type(quality) is not int or not 0 <= quality <= 100:
            raise ValueError("lossless must be boolean and quality an integer in 0..100")
        if type(method) is not int or not 0 <= method <= 6:
            raise ValueError("WebP method must be an integer in 0..6")
        if type(compression) is not int or not 0 <= compression <= 9:
            raise ValueError("PNG compression must be an integer in 0..9")
        color = image_color(color)
        if dither not in DITHERS:
            format_diagnostic({"dither": dither}, {"dither": "sierra2_4a"}, on_diagnostic)
            dither = "sierra2_4a"
        tags = metadata_json(metadata)
        packing_depth = 16 if format == "png16" or bit_depth > 8 else 8
        checked = scope.enter_context(closing(_checked_records(source, packing_depth, frame_count)))
        first = next(checked, None)
        if first is None:
            raise ValueError("frame records contain no frames")
        width, height = first[2].width, first[2].height
        alpha = any(c.is_alpha for c in first[2].format.components)
        if format not in (*FRAME_FORMATS, "apng") or (
            (bit_depth > 8 or color["transfer"] in (16, 18))
            and format not in ("png8", "png16", "apng")
        ):
            format_diagnostic({"format": format}, {"format": "apng"}, on_diagnostic)
            format = "apng"
        if format.startswith("gif") and alpha:
            format_diagnostic(
                {"format": format}, {"format": "webp", "lossless": True}, on_diagnostic
            )
            format, lossless = "webp", True
        if format == "png8" and bit_depth > 8:
            format_diagnostic({"format": format}, {"format": "png16"}, on_diagnostic)
            format = "png16"
        target_depth = 16 if format == "png16" or bit_depth > 8 else 8
        duration = first[1]
        rate = 1 / duration
        frames = chain((first,), checked)
        del first
        if format in ("png8", "png16"):
            with ZipFile(destination, "w", compression=ZIP_STORED) as archive:
                for count, (_, _, frame) in enumerate(frames, 1):
                    archive.writestr(
                        ZipInfo(f"{count:06d}.png"),
                        _png(
                            frame,
                            target_depth,
                            alpha,
                            tags if count == 1 else "",
                            color,
                            compression,
                        ),
                    )
        elif format == "gif_ffmpeg":
            with av.open(destination, "w", format="gif", options={"loop": str(loop)}) as opened:
                output = cast(Any, opened)
                stream = output.add_stream("gif", rate=rate)
                stream.width, stream.height, stream.pix_fmt = width, height, "pal8"
                stream.codec_context.thread_count = 1
                stream.codec_context.time_base = Fraction(1, 100)
                output.metadata["comment"] = tags
                for timestamp, duration, frame in frames:
                    frame.time_base, frame.pts = Fraction(1, 100), round(timestamp * 100)
                    frame.duration = max(1, round((timestamp + duration) * 100) - frame.pts)
                    paletted = _palette(frame, dither)
                    for packet in stream.encode(paletted):
                        packet.duration = frame.duration
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)
            _gif_metadata(destination, tags)
        elif format == "apng":
            sequence, control_offset, count = 0, 0, 0
            for count, (_, span, frame) in enumerate(frames, 1):
                png = _png(frame, target_depth, alpha, "", color, compression)
                if count == 1:
                    destination.write(png[:33] + _png_metadata(tags, color))
                    control_offset = destination.tell()
                    destination.write(_png_chunk(b"acTL", struct.pack(">II", 0, loop)))
                # Full-frame source replacement avoids unsupported 16-bit alpha blending.
                control = (
                    struct.pack(">IIIII", sequence, width, height, 0, 0)
                    + _apng_delay(span)
                    + bytes(2)
                )
                destination.write(_png_chunk(b"fcTL", control))
                sequence += 1
                position = 33
                while position < len(png):
                    size, kind = struct.unpack(">I4s", png[position : position + 8])
                    if kind == b"IDAT":
                        if count == 1:
                            destination.write(png[position : position + size + 12])
                        else:
                            destination.write(
                                _png_chunk(
                                    b"fdAT",
                                    struct.pack(">I", sequence)
                                    + png[position + 8 : position + 8 + size],
                                )
                            )
                            sequence += 1
                    position += size + 12
            destination.write(_png_chunk(b"IEND", b""))
            end = destination.tell()
            destination.seek(control_offset)
            destination.write(_png_chunk(b"acTL", struct.pack(">II", count, loop)))
            destination.seek(end)
        else:
            if format == "webp":
                destination.write(b"RIFF\0\0\0\0WEBP")
                destination.write(
                    _riff_chunk(
                        b"VP8X",
                        bytes((0x16 if alpha else 0x06, 0, 0, 0))
                        + _uint24(width - 1)
                        + _uint24(height - 1),
                    )
                )
                destination.write(_riff_chunk(b"ANIM", bytes(4) + struct.pack("<H", loop)))
            for count, (timestamp, span, frame) in enumerate(frames, 1):
                pixels = cast(np.ndarray, frame.to_ndarray(format="rgba" if alpha else "rgb24"))
                image = Image.fromarray(pixels)
                duration = round((timestamp + span) * 1000) - round(timestamp * 1000)
                if format == "webp":
                    encoded = io.BytesIO()
                    image.save(
                        encoded,
                        format="WEBP",
                        lossless=lossless,
                        quality=quality,
                        method=method,
                        exact=True,
                    )
                    payload = encoded.getvalue()[12:]
                    # VP8X is a file header, not a frame payload; keep ALPH/VP8/VP8L chunks only.
                    offset = 0
                    chunks = bytearray()
                    while offset < len(payload):
                        size = struct.unpack_from("<I", payload, offset + 4)[0]
                        end = offset + 8 + size + size % 2
                        if payload[offset : offset + 4] in (b"ALPH", b"VP8 ", b"VP8L"):
                            chunks.extend(payload[offset:end])
                        offset = end
                    header = (
                        bytes(6)
                        + _uint24(width - 1)
                        + _uint24(height - 1)
                        + _uint24(duration)
                        + b"\x02"
                    )
                    destination.write(_riff_chunk(b"ANMF", header + chunks))
                else:
                    duration = max(1, round((timestamp + span) * 100) - round(timestamp * 100)) * 10
                    palette_image = image.convert("RGB").quantize(colors=255)
                    palette = palette_image.getpalette() or []
                    palette_image.putpalette(palette[:765] + [0, 0, 0])
                    if count == 1:
                        blocks, _ = GifImagePlugin.getheader(
                            palette_image, info={"loop": loop, "comment": tags.encode()}
                        )
                        for block in blocks:
                            destination.write(block)
                    for block in GifImagePlugin.getdata(
                        palette_image,
                        include_color_table=True,
                        duration=duration,
                        disposal=2,
                        transparency=255,
                    ):
                        destination.write(block)
            if format == "webp":
                destination.write(_riff_chunk(b"XMP ", tags.encode()))
                end = destination.tell()
                destination.seek(4)
                destination.write(struct.pack("<I", end - 8))
                destination.seek(end)
            else:
                destination.write(b";")
    return (
        (".zip", "application/zip")
        if format.startswith("png")
        else (".png", "image/png")
        if format == "apng"
        else (".webp", "image/webp")
        if format == "webp"
        else (".gif", "image/gif")
    )
