"""Shared record sink contracts for chunk adapters and VIDEO exporters."""

from __future__ import annotations

import asyncio
import io
import json
from fractions import Fraction
from typing import Any, cast
from zipfile import ZipFile

import av
import numpy as np
import pytest
from dinkster_api.v1 import annotate_image, save_frame_records
from dinkster_nodes_media_io.image_metadata import metadata_document, png_metadata
from dinkster_video import formats, read_video_metadata
from PIL import Image

COLOR = {"primaries": 1, "transfer": 13, "range": 2}
METADATA = {"workflow": {"nodes": []}, "prompt": "sample", "comment": "true"}


class Records:
    def __init__(
        self, count: int = 3, *, alpha: bool = False, duration: Fraction = Fraction(1, 10)
    ):
        self.count, self.alpha, self.duration = count, alpha, duration
        self.position = 0
        self.closed = 0

    def __iter__(self) -> Records:
        return self

    def __next__(self) -> tuple[Fraction, Fraction, object]:
        if self.position >= self.count:
            raise StopIteration
        index = self.position
        self.position += 1
        pixels = np.full((8, 8, 4 if self.alpha else 3), (index + 1) / 5, np.float32)
        if self.alpha:
            pixels[..., 3] = np.linspace(0, 1, 8, dtype=np.float32)
        return index * self.duration, self.duration, pixels

    def close(self) -> None:
        self.closed += 1


@pytest.mark.parametrize("format", [*formats.FRAME_FORMATS, "apng"])
@pytest.mark.parametrize("declared", [False, True])
def test_record_sink_counts_metadata_and_closes_without_building_video(
    format: str, declared: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(formats, "coerce_video", lambda *args: pytest.fail("materialized VIDEO"))
    source = Records()
    output = io.BytesIO()
    suffix, mime = save_frame_records(
        source,
        output,
        format=format,
        bit_depth=8,
        color=COLOR,
        frame_count=3 if declared else None,
        loop=3,
        metadata=METADATA,
    )
    assert source.closed == 1
    assert not output.closed
    data = output.getvalue()
    assert read_video_metadata(data) == METADATA
    if format.startswith("png"):
        assert (suffix, mime) == (".zip", "application/zip")
        with ZipFile(io.BytesIO(data)) as archive:
            assert len(archive.namelist()) == 3
    else:
        assert (suffix, mime) == (
            (".png", "image/png")
            if format == "apng"
            else (".webp", "image/webp")
            if format == "webp"
            else (".gif", "image/gif")
        )
        with Image.open(io.BytesIO(data)) as image:
            assert cast(Any, image).n_frames == 3
            assert image.info["loop"] == 3
            for index in range(3):
                image.seek(index)
                np.testing.assert_allclose(
                    np.asarray(image.convert("RGB")) / 255, (index + 1) / 5, atol=1 / 255
                )


@pytest.mark.parametrize("format", ["gif_pillow", "gif_ffmpeg", "webp", "apng"])
def test_long_frame_durations_are_encoded_not_truncated(format: str) -> None:
    output = io.BytesIO()
    save_frame_records(
        Records(2, duration=Fraction(100)), output, format=format, bit_depth=8, color=COLOR
    )
    with Image.open(io.BytesIO(output.getvalue())) as image:
        for index in range(2):
            image.seek(index)
            image.load()
            assert image.info["duration"] == 100000


@pytest.mark.parametrize("format", ["apng", "gif_pillow", "gif_ffmpeg", "webp"])
def test_variable_duration_and_last_frame_duration(format: str) -> None:
    records = [
        (t, d, np.full((8, 8, 3), value, np.float32))
        for t, d, value in (
            (Fraction(0), Fraction(13, 100), 0.2),
            (Fraction(13, 100), Fraction(27, 100), 0.4),
            (Fraction(2, 5), Fraction(2, 25), 0.6),
        )
    ]
    output = io.BytesIO()
    save_frame_records(records, output, format=format, bit_depth=8, color=COLOR)
    with Image.open(io.BytesIO(output.getvalue())) as image:
        actual = []
        for index in range(3):
            image.seek(index)
            image.load()
            actual.append(image.info["duration"])
        assert actual == [130, 270, 80]


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("depth", [8, 16])
def test_apng_identical_frames_retain_count_duration_and_loop(count: int, depth: int) -> None:
    pixels = np.full((8, 8, 4), 0.4, np.float32)
    records = ((index * Fraction(1, 4), Fraction(1, 4), pixels) for index in range(count))
    output = io.BytesIO()
    save_frame_records(records, output, format="apng", bit_depth=depth, color=COLOR, loop=7)
    with Image.open(io.BytesIO(output.getvalue())) as image:
        assert cast(Any, image).n_frames == count
        assert image.info["loop"] == 7
        for index in range(count):
            image.seek(index)
            assert image.info["duration"] == 250
            np.testing.assert_array_equal(np.asarray(image), np.full((8, 8, 4), 102))
    with av.open(io.BytesIO(output.getvalue()), "r") as opened:
        decoded = list(opened.decode(video=0))
        assert len(decoded) == count
        for frame in decoded:
            actual = frame.to_ndarray(format="rgba64le" if depth == 16 else "rgba")
            np.testing.assert_array_equal(actual, np.full((8, 8, 4), round(0.4 * (2**depth - 1))))


def test_png16_does_not_pack_normalized_input_through_uint8() -> None:
    pixels = np.full((8, 8, 3), 12345 / 65535, np.float32)
    output = io.BytesIO()
    save_frame_records(
        [(Fraction(0), Fraction(1), pixels)], output, format="png16", bit_depth=8, color=COLOR
    )
    with ZipFile(io.BytesIO(output.getvalue())) as archive:
        with av.open(io.BytesIO(archive.read("000001.png")), "r") as opened:
            actual = next(opened.decode(video=0)).to_ndarray(format="rgb48le")
            np.testing.assert_array_equal(actual, np.full((8, 8, 3), 12345))


def test_hdr_record_fallback_retains_precision_alpha_color_and_reports_effective_format() -> None:
    output = io.BytesIO()
    diagnostics: list[Any] = []
    pixels = np.full((8, 8, 4), 12345 / 65535, np.float32)
    result = save_frame_records(
        [(Fraction(0), Fraction(1), pixels)],
        output,
        format="gif_pillow",
        bit_depth=16,
        color={"primaries": 9, "transfer": 16, "range": 2},
        on_diagnostic=diagnostics.append,
    )
    assert result == (".png", "image/png")
    assert len(diagnostics) == 1
    assert diagnostics[0]["code"] == "media_format_fallback"
    assert diagnostics[0]["requested"] == {
        "container": "gif",
        "codec": "gif",
        "pixelFormat": None,
        "channelLayout": None,
    }
    assert diagnostics[0]["effective"] == {
        "container": "png",
        "codec": "apng",
        "pixelFormat": "rgba64be",
        "channelLayout": None,
    }
    assert diagnostics[0]["reason"] == "preserving_cpu_default"
    data = output.getvalue()
    assert data[24] == 16
    color_offset = data.index(b"cICP") + 4
    assert data[color_offset : color_offset + 4] == bytes((9, 16, 0, 1))
    with av.open(io.BytesIO(data), "r") as opened:
        actual = next(opened.decode(video=0)).to_ndarray(format="rgba64le")
        np.testing.assert_array_equal(actual, np.full((8, 8, 4), 12345))


@pytest.mark.parametrize("depth", [8, 16])
def test_apng_preserves_alpha_precision_and_compression(depth: int) -> None:
    encoded: list[bytes] = []
    for compression in (0, 9):
        output = io.BytesIO()
        save_frame_records(
            Records(alpha=True),
            output,
            format="apng",
            bit_depth=depth,
            color=COLOR,
            compression=compression,
        )
        encoded.append(output.getvalue())
    assert len(encoded[1]) < len(encoded[0])
    for data in encoded:
        assert data[24] == depth
        with av.open(io.BytesIO(data), "r") as opened:
            for index, frame in enumerate(opened.decode(video=0)):
                actual = frame.to_ndarray(format="rgba64le" if depth == 16 else "rgba")
                alpha = np.rint(np.linspace(0, 1, 8) * (2**depth - 1))
                np.testing.assert_array_equal(actual[..., 3], np.broadcast_to(alpha, (8, 8)))
                np.testing.assert_allclose(
                    actual[..., :3] / (2**depth - 1), (index + 1) / 5, atol=1 / (2**depth - 1)
                )


@pytest.mark.parametrize("format", ["png8", "png16", "webp", "apng"])
def test_record_sink_encodes_premultiplied_images_as_straight_alpha(format: str) -> None:
    pixels = np.full((8, 8, 4), 0.25, np.float32)
    pixels[..., 3] = 0.5
    premultiplied = annotate_image(pixels, alpha="premultiplied")
    output = io.BytesIO()
    save_frame_records(
        [(Fraction(0), Fraction(1), premultiplied)],
        output,
        format=format,
        bit_depth=16 if format == "png16" else 8,
        color=COLOR,
    )
    data = output.getvalue()
    if format.startswith("png"):
        with ZipFile(io.BytesIO(data)) as archive:
            data = archive.read("000001.png")
    actual: np.ndarray | None = None
    if format == "webp":
        with Image.open(io.BytesIO(data)) as image:
            actual = np.asarray(image.convert("RGBA"))
    else:
        with av.open(io.BytesIO(data), "r") as opened:
            actual = next(opened.decode(video=0)).to_ndarray(
                format="rgba64le" if format == "png16" else "rgba"
            )
    assert actual is not None
    scale = 65535 if format == "png16" else 255
    np.testing.assert_allclose(actual[..., :3] / scale, 0.5, atol=2 / scale)
    np.testing.assert_allclose(actual[..., 3] / scale, 0.5, atol=2 / scale)


@pytest.mark.parametrize("declared", [2, 4])
def test_count_mismatch_closes_source_without_closing_destination(declared: int) -> None:
    source = Records()
    output = io.BytesIO()
    with pytest.raises(ValueError, match="declared frame_count"):
        save_frame_records(
            source, output, format="webp", bit_depth=8, color=COLOR, frame_count=declared
        )
    assert source.closed == 1 and not output.closed


def test_schema_error_closes_input_even_before_iteration() -> None:
    source = Records()
    with pytest.raises(ValueError, match="quality"):
        save_frame_records(
            source, io.BytesIO(), format="webp", bit_depth=8, color=COLOR, quality=101
        )
    assert source.closed == 1 and source.position == 0


def test_destination_failure_closes_source() -> None:
    class Broken(io.BytesIO):
        def write(self, data: Any) -> int:
            raise OSError("disk full")

    source = Records()
    with pytest.raises(OSError, match="disk full"):
        save_frame_records(source, Broken(), format="webp", bit_depth=8, color=COLOR)
    assert source.closed == 1 and source.position == 1


@pytest.mark.parametrize("fault", ["gap", "overlap", "duration", "shape", "dtype", "nan"])
def test_malformed_records_close_the_generator(fault: str) -> None:
    closed = []

    def records():
        try:
            yield Fraction(0), Fraction(1), np.zeros((8, 8, 3), np.float32)
            timestamp = (
                Fraction(2)
                if fault == "gap"
                else Fraction(0)
                if fault == "overlap"
                else Fraction(1)
            )
            duration = Fraction(0) if fault == "duration" else Fraction(1)
            pixels = np.zeros((9 if fault == "shape" else 8, 8, 3), np.float32)
            if fault == "dtype":
                pixels = pixels.astype(np.float64)
            if fault == "nan":
                pixels[0, 0, 0] = np.nan
            yield timestamp, duration, pixels
        finally:
            closed.append(True)

    with pytest.raises(ValueError, match="frame records"):
        save_frame_records(records(), io.BytesIO(), format="webp", bit_depth=8, color=COLOR)
    assert closed == [True]


def test_pillow_webp_receives_method(monkeypatch: pytest.MonkeyPatch) -> None:
    methods = []
    original = Image.Image.save

    def save(self: Image.Image, *args: Any, **kwargs: Any) -> None:
        methods.append(kwargs.get("method"))
        original(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", save)
    save_frame_records(Records(), io.BytesIO(), format="webp", bit_depth=8, color=COLOR, method=6)
    assert methods == [6, 6, 6]


@pytest.mark.parametrize("format", ["apng", "png8", "png16"])
def test_png_metadata_matches_ordinary_still_reader(format: str) -> None:
    metadata = {
        "prompt": {"1": {"inputs": {"text": "snow \u2603"}}},
        "workflow": {"nodes": [], "extra": {}},
        "caption": "snow \u2603",
        "comment": "true",
    }
    reference = io.BytesIO()
    Image.new("RGB", (8, 8)).save(
        reference, format="PNG", pnginfo=png_metadata(json.dumps(metadata))
    )
    output = io.BytesIO()
    save_frame_records(
        Records(), output, format=format, bit_depth=8, color=COLOR, metadata=metadata
    )
    data = output.getvalue()
    assert read_video_metadata(data) == metadata
    if format != "apng":
        with ZipFile(io.BytesIO(data)) as archive:
            data = archive.read("000001.png")
            with Image.open(io.BytesIO(archive.read("000002.png"))) as following:
                assert "prompt" not in following.info
    with Image.open(io.BytesIO(data)) as image, Image.open(reference) as ordinary:
        expected = metadata_document(
            cast(dict[str, object], ordinary.info),
            name="image.png",
            digest="",
            media_type="image/png",
            size=len(data),
        )
        actual = metadata_document(
            cast(dict[str, object], image.info),
            name="image.png",
            digest="",
            media_type="image/png",
            size=len(data),
        )
        assert actual["comfy"] == expected["comfy"]
        for key in metadata:
            assert image.info[key] == ordinary.info[key]


def test_png_nonstandard_metadata_keys_remain_in_typed_envelope() -> None:
    metadata = {key: "value" for key in ("", "bad\0key", "\u2603", "x" * 80, "dinkster_metadata")}
    metadata["unpaired_surrogate"] = "\ud800"
    output = io.BytesIO()
    save_frame_records(
        Records(), output, format="apng", bit_depth=8, color=COLOR, metadata=metadata
    )
    assert read_video_metadata(output.getvalue()) == metadata


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_source_failure_or_cancellation_closes_input_and_keeps_destination_open(error) -> None:
    class Broken(Records):
        def __next__(self):
            if self.position == 1:
                raise error("stop")
            return super().__next__()

    source = Broken()
    output = io.BytesIO()
    with pytest.raises(error):
        save_frame_records(source, output, format="apng", bit_depth=8, color=COLOR)
    assert source.closed == 1 and not output.closed


def test_iter_creation_failure_closes_source() -> None:
    class Broken(Records):
        def __iter__(self):
            raise RuntimeError("iteration unavailable")

    source = Broken()
    with pytest.raises(RuntimeError, match="iteration unavailable"):
        save_frame_records(source, io.BytesIO(), format="apng", bit_depth=8, color=COLOR)
    assert source.closed == 1


def test_fallback_header_reads_are_bounded_and_reporting_follows_source_close() -> None:
    class BoundedReads(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            assert size is not None and 0 <= size <= 4096
            return super().read(size)

    source = Records(count=100, alpha=True)
    output = BoundedReads()
    diagnostics = []

    def report(record):
        assert source.closed == 1
        assert output.tell() == len(output.getvalue())
        diagnostics.append(record)

    save_frame_records(
        source, output, format="gif_pillow", bit_depth=8, color=COLOR, on_diagnostic=report
    )
    assert len(diagnostics) == 1
    assert diagnostics[0]["effective"] == {
        "container": "webp",
        "codec": "webp",
        "pixelFormat": "rgba",
        "channelLayout": None,
    }
    assert not output.closed


def test_failed_fallback_does_not_report_successful_substitution() -> None:
    source = Records(alpha=True)
    diagnostics = []
    with pytest.raises(ValueError, match="declared frame_count"):
        save_frame_records(
            source,
            io.BytesIO(),
            format="gif_pillow",
            bit_depth=8,
            color=COLOR,
            frame_count=1,
            on_diagnostic=diagnostics.append,
        )
    assert source.closed == 1 and diagnostics == []
