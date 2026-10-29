"""Executable VIDEO v2 source, edit, serialization, and streaming contracts."""

from __future__ import annotations

import asyncio
import io
import json
import struct
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO, cast

import av
import numpy as np
import pytest
from av.error import FFmpegError, InvalidDataError
from dinkster_api.v1 import VIDEO_TYPE
from dinkster_assets import AssetRef, AssetVault, digest_bytes, register_video_value_type
from dinkster_nodes_media_io.video_ops import CropVideo, TrimVideo, VideoInfo
from dinkster_values import (
    EncodedPayload,
    TypeRegistry,
    decode_video,
    edit_video,
    effective_video_facts,
    encode_video,
    video_fingerprint,
    video_from_source,
    video_meta,
)
from dinkster_values.storage import image_input
from dinkster_video import assemble_video, disassemble_video, save_video_stream


def _source(*, depth: int = 8, b_frames: int = 0, container: str = "mp4", offset: int = 0) -> bytes:
    output = io.BytesIO()
    with av.open(output, "w", format=container) as opened:
        mux = cast(Any, opened)
        stream = mux.add_stream("libx264", rate=10)
        stream.width, stream.height = 64, 32
        stream.pix_fmt = "yuv420p10le" if depth == 10 else "yuv420p"
        stream.codec_context.thread_count = 1
        stream.codec_context.max_b_frames = b_frames
        stream.gop_size = 5
        stream.options = {"crf": "0", "x264-params": "scenecut=0"}
        if depth == 10:
            stream.codec_context.color_primaries = 9
            stream.codec_context.color_trc = 18
            stream.codec_context.colorspace = 9
            stream.codec_context.color_range = 1
        for i in range(20):
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 64, 3), (i * 10 + offset) % 256, np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = i, Fraction(1, 10)
            for packet in stream.encode(frame):
                mux.mux(packet)
        for packet in stream.encode():
            mux.mux(packet)
    return output.getvalue()


def _packets(data: bytes) -> list[bytes]:
    packets: list[bytes] = []
    with av.open(io.BytesIO(data), mode="r") as source:
        packets = [bytes(p) for p in source.demux(video=0) if p.size]
    return packets


def _x264_options(data: bytes) -> bytes:
    payload = b"".join(_packets(data))
    start = payload.index(b"x264 -")
    return payload[start : payload.index(b"\x00", start)]


def _source_style_video(
    images: np.ndarray,
    *,
    fps: Fraction,
    depth: int,
    color: str,
    crf: float | None = None,
    preset: str | None = None,
    codec: str = "h264",
) -> bytes:
    """Encode source components independently of the public VIDEO writer."""
    output = io.BytesIO()
    rate = Fraction(round(fps * 1000), 1000)
    pix_fmt = "yuv420p10le" if depth > 8 else "yuv420p"
    color_values = {"sRGB": (1, 13, 1, 1), "HDR PQ": (9, 16, 9, 1)}[color]
    with av.open(output, "w", format="mp4", options={"movflags": "use_metadata_tags"}) as opened:
        mux = cast(Any, opened)
        stream = mux.add_stream("libsvtav1" if codec == "av1" else "libx264", rate=rate)
        stream.width, stream.height = images.shape[2], images.shape[1]
        stream.pix_fmt = pix_fmt
        stream.options = {"crf": str((24 if codec == "av1" else 18) if crf is None else crf)}
        if preset is not None:
            stream.options["preset"] = preset
        (
            stream.codec_context.color_primaries,
            stream.codec_context.color_trc,
            stream.codec_context.colorspace,
            stream.codec_context.color_range,
        ) = color_values
        for samples in images:
            if depth > 8:
                pixels = np.clip(samples.astype(np.float32) * 65535, 0, 65535).astype("<u2")
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb48le")
            else:
                pixels = np.clip(samples * 255, 0, 255).astype(np.uint8)
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame = frame.reformat(format=pix_fmt, dst_colorspace=color_values[2])
            (
                frame.color_primaries,
                frame.color_trc,
                frame.colorspace,
                frame.color_range,
            ) = color_values
            for packet in stream.encode(frame):
                mux.mux(packet)
        for packet in stream.encode():
            mux.mux(packet)
    return output.getvalue()


@pytest.mark.parametrize(
    ("depth", "color", "crf", "preset"),
    [(8, "sRGB", None, None), (10, "HDR PQ", 18.5, "slow")],
)
def test_source_component_h264_matches_approved_source_encoding(
    depth: int, color: str, crf: float | None, preset: str | None
) -> None:
    fps = Fraction(30000, 1001)
    images = np.linspace(-0.01, 1.01, 4 * 64 * 64 * 3, dtype=np.float64).reshape(4, 64, 64, 3)
    value = assemble_video(images, fps=fps, bit_depth=str(depth), color_space=color)
    expected = _source_style_video(
        images, fps=fps, depth=depth, color=color, crf=crf, preset=preset
    )
    actual = io.BytesIO()
    save_video_stream(
        value,
        actual,
        codec="h264",
        crf=crf,
        preset=preset,
        source_components=True,
    )
    assert cast(dict[str, object], value["probe"])["fps"] == fps
    with av.open(io.BytesIO(actual.getvalue()), mode="r") as container:
        stream = container.streams.video[0]
        assert stream.average_rate == Fraction(2997, 100)
        assert stream.pix_fmt == ("yuv420p10le" if depth > 8 else "yuv420p")
        assert (
            stream.codec_context.color_primaries,
            stream.codec_context.color_trc,
            stream.codec_context.colorspace,
            stream.codec_context.color_range,
        ) == {"sRGB": (1, 13, 1, 1), "HDR PQ": (9, 16, 9, 1)}[color]
    assert len(_packets(actual.getvalue())) == len(_packets(expected))
    assert _x264_options(actual.getvalue()) == _x264_options(expected)
    np.testing.assert_array_equal(
        disassemble_video(video_from_source(actual.getvalue()))["images"],
        disassemble_video(video_from_source(expected))["images"],
    )


def test_default_component_writer_remains_separate_from_source_component_encoding() -> None:
    fps = Fraction(30000, 1001)
    images = np.linspace(0, 1, 4 * 16 * 24 * 3, dtype=np.float32).reshape(4, 16, 24, 3)
    value = assemble_video(images, fps=fps)
    implicit, explicit, source_style = io.BytesIO(), io.BytesIO(), io.BytesIO()
    save_video_stream(value, implicit, codec="h264")
    save_video_stream(value, explicit, codec="h264", source_components=False)
    save_video_stream(value, source_style, codec="h264", source_components=True)
    assert cast(dict[str, object], value["probe"])["fps"] == fps
    assert _packets(implicit.getvalue()) == _packets(explicit.getvalue())
    options = _x264_options(implicit.getvalue())
    assert b"crf=18.0" in options
    assert b"bframes=3" in options
    assert b"rc_lookahead=40" in options and b"mbtree=1" in options
    with av.open(io.BytesIO(implicit.getvalue())) as container:
        # Four frames span 133467 ticks under the default microsecond timestamp policy.
        assert container.streams.video[0].average_rate == Fraction(4_000_000, 133_467)
    assert _x264_options(source_style.getvalue()) == _x264_options(
        _source_style_video(images, fps=fps, depth=8, color="sRGB")
    )


def test_saved_source_file_retains_timestamps_without_streaming_encoder_options() -> None:
    source = _source(b_frames=2)
    selected = edit_video(
        video_from_source(source), {"trim": {"start_time": 0.25, "duration": 0.5}}
    )
    output = io.BytesIO()
    save_video_stream(selected, output, codec="h264", crf=18)
    options = _x264_options(output.getvalue())
    assert b"bframes=0" in options
    assert b"rc_lookahead=40" in options and b"mbtree=1" in options
    with av.open(io.BytesIO(output.getvalue()), mode="r") as container:
        frames = list(container.decode(video=0))
        assert len(frames) == 5
        for i, frame in enumerate(frames):
            assert frame.pts is not None and frame.time_base is not None
            assert frame.pts * frame.time_base == Fraction(i, 10) + Fraction(1, 20)


@pytest.mark.parametrize("shape", [(2, 16, 25, 3), (2, 16, 24, 4)])
def test_source_component_h264_preserves_source_shape_failures(
    shape: tuple[int, int, int, int],
) -> None:
    images = np.zeros(shape, dtype=np.float32)
    with pytest.raises((ValueError, FFmpegError)):
        save_video_stream(
            assemble_video(images, fps=30),
            io.BytesIO(),
            container="mkv",
            codec="h264",
            source_components=True,
        )


@pytest.mark.parametrize(("height", "width"), [(8, 12), (64, 64)])
def test_source_component_av1_preserves_source_encode_contract(height: int, width: int) -> None:
    images = np.linspace(0, 1, 2 * height * width * 3, dtype=np.float32).reshape(
        2, height, width, 3
    )
    value = assemble_video(images, fps=30)
    encoded = io.BytesIO()
    try:
        expected = _source_style_video(images, fps=Fraction(30), depth=8, color="sRGB", codec="av1")
    except FFmpegError as expected_error:
        if height >= 64 and width >= 64:
            raise
        # Small-frame support belongs to the source encoder, including its failures.
        with pytest.raises(type(expected_error)) as actual_error:
            save_video_stream(value, encoded, container="mkv", codec="av1", source_components=True)
        assert type(actual_error.value) is type(expected_error)
        assert actual_error.value.errno == expected_error.errno
        assert encoded.getvalue() == b""
        return
    save_video_stream(
        value,
        encoded,
        container="mkv",
        codec="av1",
        source_components=True,
    )
    with av.open(io.BytesIO(encoded.getvalue()), mode="r") as container:
        stream = container.streams.video[0]
        assert stream.codec_context.codec.canonical_name == "av1"
        assert (stream.width, stream.height) == (width, height)
        assert len(list(container.decode(video=0))) == 2
    if height >= 64 and width >= 64:
        # Unaligned small inputs do not have bit-stable source encoder output.
        np.testing.assert_array_equal(
            disassemble_video(video_from_source(encoded.getvalue()))["images"],
            disassemble_video(video_from_source(expected))["images"],
        )


@pytest.mark.parametrize("encoded", (False, True))
def test_public_pixel_iterator_applies_edits_to_components_and_encoded_video(encoded: bool) -> None:
    from dinkster_api.v1 import iter_video_pixels

    pixels = np.arange(4 * 4 * 6 * 3, dtype=np.uint16).reshape(4, 4, 6, 3).astype(np.uint8)
    video = assemble_video(pixels, fps=Fraction(5))
    if encoded:
        output = io.BytesIO()
        save_video_stream(video, output, codec="ffv1", container="mkv")
        video = video_from_source(output.getvalue())
    video = edit_video(video, {"trim": {"start_time": Fraction(1, 5), "duration": Fraction(2, 5)}})
    video = edit_video(video, {"crop": {"x": 2, "y": 2, "width": 4, "height": 2}})
    frames = iter_video_pixels(video)
    try:
        actual = list(frames)
    finally:
        frames.close()
    assert [time for time, _pixels in actual] == [Fraction(0), Fraction(1, 5)]
    expected = pixels[1:3, 2:4, 2:6].astype(np.float32) / 255
    for (_time, frame), wanted in zip(actual, expected, strict=True):
        assert frame.dtype == np.float32
        np.testing.assert_array_equal(frame, wanted)


def test_public_pixel_iterator_early_close_releases_encoded_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_api.v1 import iter_video_pixels

    data = _source()
    vault = AssetVault(tmp_path)
    digest = digest_bytes(data)
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()
    video = video_from_source(AssetRef(digest, "source.mp4", len(data), resolver=vault))
    opened: list[BinaryIO] = []
    original = AssetRef.open

    def track_open(ref: AssetRef) -> BinaryIO:
        handle = original(ref)
        opened.append(handle)
        return handle

    monkeypatch.setattr(AssetRef, "open", track_open)
    frames = iter_video_pixels(video)
    try:
        timestamp, pixels = next(frames)
        assert timestamp == 0
        assert pixels.shape == (32, 64, 3)
        assert opened and any(not handle.closed for handle in opened)
    finally:
        frames.close()
    assert all(handle.closed for handle in opened)


def test_inline_v2_roundtrip_and_v1_migration() -> None:
    data = _source()
    value = video_from_source(data)
    encoded = encode_video(value)
    assert encoded.startswith(b"DINKSTER-VIDEO\x02")
    assert decode_video(encoded) == value
    assert decode_video(data) == value
    assert video_meta(value)["codec_version"] == 2
    assert video_fingerprint(VIDEO_TYPE)(value) == video_fingerprint(VIDEO_TYPE)(decode_video(data))


def test_asset_codec_does_not_open_and_rebinds_at_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _source()
    vault = AssetVault(tmp_path)
    digest = digest_bytes(data)
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()
    ref = AssetRef(digest, "source.mp4", len(data), resolver=vault)
    value = video_from_source(ref)
    with monkeypatch.context() as patch:
        patch.setattr(
            AssetRef, "open", lambda self: pytest.fail("serialization must not open source")
        )
        encoded = encode_video(value)
        assert len(encoded) < 2000
    registry = TypeRegistry()
    register_video_value_type(registry, VIDEO_TYPE, vault)
    decoded = registry.spec(VIDEO_TYPE).decode(encoded)
    with monkeypatch.context() as patch:
        patch.setattr(AssetRef, "open", lambda self: pytest.fail("admitted edits must not probe"))
        assert video_meta(decoded)["asset_refs"] == [ref.to_wire()]
        edit_video(decoded, {"trim": {"start_time": 0.5, "duration": 1}})
    result = io.BytesIO()
    save_video_stream(decoded, result)
    assert result.getvalue() == data


@pytest.mark.parametrize("depth", [8, 10])
def test_exact_closed_trim_copies_packet_payloads(
    depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_video import runtime

    data = _source(depth=depth)
    value = edit_video(video_from_source(data), {"trim": {"start_time": 0.5, "duration": 1}})
    monkeypatch.setattr(runtime, "_decoded_video", lambda *a: pytest.fail("safe trim must copy"))
    output = io.BytesIO()
    save_video_stream(value, output)
    assert _packets(output.getvalue()) == _packets(data)[5:15]
    facts = cast(dict[str, Any], video_from_source(output.getvalue())["probe"])
    assert facts["bit_depth"] == depth
    assert facts["duration"] == 1
    if depth == 10:
        assert [facts[k] for k in ("primaries", "transfer", "matrix", "range")] == [9, 18, 9, 1]


def test_non_packet_aligned_trim_transcodes_without_snapping() -> None:
    value = edit_video(
        video_from_source(_source()), {"trim": {"start_time": 0.11, "duration": 0.4}}
    )
    output = io.BytesIO()
    save_video_stream(value, output)
    decoded = disassemble_video(video_from_source(output.getvalue()))
    assert decoded["frame_count"] == 4
    images = cast(np.ndarray, image_input(decoded["images"]))
    assert np.max(np.abs(images[:, 0, 0, 0] - np.arange(2, 6) * 10 / 255)) < 0.02


@pytest.mark.parametrize("first", [0, 4])
@pytest.mark.parametrize("origin", [0, 2])
def test_fractional_encoded_trim_uses_half_open_stream_ticks(first: int, origin: int) -> None:
    from dinkster_video import iter_video_pixels

    rate = Fraction(30000, 1001)
    pixels = (np.arange(7 * 4 * 32 * 3).reshape(7, 4, 32, 3) % 251).astype(np.uint8)
    encoded = io.BytesIO()
    save_video_stream(assemble_video(pixels, fps=rate), encoded, codec="ffv1", container="mkv")
    if origin:
        shifted = io.BytesIO()
        with (
            av.open(io.BytesIO(encoded.getvalue()), "r") as source,
            av.open(shifted, "w", format="matroska") as destination,
        ):
            target = cast(Any, destination).add_stream_from_template(source.streams.video[0])
            for packet in source.demux(video=0):
                if packet.dts is not None:
                    assert packet.pts is not None
                    shift = int(Fraction(origin) / packet.time_base)
                    packet.pts += shift
                    packet.dts += shift
                    packet.stream = target
                    destination.mux(packet)
        encoded = shifted
    value = video_from_source(encoded.getvalue())
    assert cast(dict[str, Any], value["probe"])["start_time"] == origin
    assert cast(dict[str, Any], value["probe"])["time_base"] == Fraction(1, 1000)
    assert cast(dict[str, Any], value["probe"])["duration"] == origin + Fraction(233, 1000)
    start, duration = first / rate, 3 / rate
    selected_duration = min(duration, origin + Fraction(233, 1000) - start)
    trimmed = edit_video(value, {"trim": {"start_time": start, "duration": duration}})
    # Fixed F truncates these bounds to ticks: head [0,100), tail [133,233).
    expected = pixels[first : first + 3].astype(np.float32) / 255
    frames = list(iter_video_pixels(trimmed))
    np.testing.assert_array_equal(np.stack([array for _, array in frames]), expected)
    source_ticks = [0, 33, 67] if first == 0 else [133, 167, 200]
    assert [time for time, _ in frames] == [Fraction(tick, 1000) - start for tick in source_ticks]
    decoded = disassemble_video(trimmed)
    np.testing.assert_array_equal(image_input(decoded["images"]), expected)
    assert decoded["duration"] == float(selected_duration)
    assert decoded["fps"] == float(rate)
    assert effective_video_facts(trimmed)["duration"] == selected_duration
    chained = edit_video(
        edit_video(value, {"trim": {"start_time": 0, "duration": 7 / rate}}),
        {"trim": {"start_time": start, "duration": duration}},
    )
    np.testing.assert_array_equal(disassemble_video(chained)["images"], decoded["images"])
    saved = io.BytesIO()
    save_video_stream(trimmed, saved, codec="ffv1", container="mkv")
    np.testing.assert_array_equal(
        image_input(disassemble_video(video_from_source(saved.getvalue()))["images"]), expected
    )
    assert value["edits"] == []


@pytest.mark.parametrize("encoded", [False, True])
def test_fractional_trim_keeps_rational_timeline_and_audio_samples(encoded: bool) -> None:
    from dinkster_video import iter_video_pixels
    from dinkster_video.runtime import iter_video_audio

    rate = Fraction(30000, 1001)
    pixels = (np.arange(7 * 4 * 32 * 3).reshape(7, 4, 32, 3) % 251).astype(np.uint8)
    pcm = np.arange(2 * 12000, dtype=np.float32).reshape(1, 2, 12000) / 32768
    value = assemble_video(pixels, fps=rate, audio={"waveform": pcm, "sample_rate": 48000})
    if encoded:
        output = io.BytesIO()
        save_video_stream(value, output, codec="ffv1", container="mkv")
        value = video_from_source(output.getvalue())
    start, duration = 4 / rate, 3 / rate
    trimmed = edit_video(value, {"trim": {"start_time": start, "duration": duration}})
    frames = list(iter_video_pixels(trimmed))
    expected_times = (
        [Fraction(tick, 1000) - start for tick in (133, 167, 200)]
        if encoded
        else [0, 1 / rate, 2 / rate]
    )
    assert [time for time, _ in frames] == expected_times
    np.testing.assert_array_equal(
        np.stack([array for _, array in frames]), pixels[4:].astype(np.float32) / 255
    )
    audio = list(iter_video_audio(trimmed))
    assert audio[0][0] == Fraction(6407, 48000) - start
    np.testing.assert_array_equal(
        np.concatenate([frame.to_ndarray() for _, frame in audio], axis=1), pcm[0, :, 6407:11212]
    )
    assert effective_video_facts(trimmed)["duration"] == duration
    assert value["edits"] == []


def test_mkv_unchanged_is_exact_source_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _source(container="matroska")
    value = video_from_source(data)
    monkeypatch.setattr(av, "open", lambda *a, **kw: pytest.fail("unchanged save must not decode"))
    output = io.BytesIO()
    assert save_video_stream(value, output) == (".mkv", "video/x-matroska")
    assert output.getvalue() == data


@pytest.mark.parametrize("edit", [{"crop": {}}, {"scale": {"width": 64, "height": 32}}])
def test_full_frame_edits_keep_source_copy(
    edit: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _source(depth=10)
    value = edit_video(video_from_source(data), edit)
    monkeypatch.setattr(av, "open", lambda *a, **kw: pytest.fail("no-op edits must not decode"))
    output = io.BytesIO()
    save_video_stream(value, output)
    assert output.getvalue() == data


@pytest.mark.parametrize(
    "widget", [{}, {"crop": {}}, {"trim": {}}, {"trim": {"start_time": -0.5, "duration": 0}}]
)
def test_video_edit_is_read_only_and_sections_are_independent(
    widget: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    value = video_from_source(_source())
    snapshot = json.dumps(widget, sort_keys=True)
    monkeypatch.setattr(av, "open", lambda *a, **kw: pytest.fail("edits/info must not probe"))
    trimmed = TrimVideo.execute(video=value, video_edit=widget)["video"]
    cropped = CropVideo.execute(video=trimmed, video_edit=widget)["video"]
    assert json.dumps(widget, sort_keys=True) == snapshot
    info = VideoInfo.execute(video=cropped)
    assert (info["width"], info["height"]) == (64, 32)
    assert info["duration"] == (0.5 if widget.get("trim") else 2)


def test_deferred_components_are_exact_across_codec_and_cropped() -> None:
    images = np.random.default_rng(5).random((8, 32, 64, 4), dtype=np.float32)
    value = assemble_video(images, fps=8, bit_depth="10", color_space="HDR PQ")
    value = edit_video(value, {"crop": {"x": 11, "y": 3, "width": 24, "height": 20}})
    result = disassemble_video(decode_video(encode_video(value)))
    assert np.array_equal(cast(np.ndarray, result["images"]), images[:, 2:22, 10:34])
    assert result["bit_depth"] == 10
    assert result["color_space"] == "HDR PQ"


@pytest.mark.parametrize("channels,layout", [(3, "3.0"), (6, "5.1")])
def test_component_audio_encoder_frames_retain_samples_layout_and_timing(channels, layout) -> None:
    from dinkster_video.runtime import _audio_frames, _plan

    pcm = np.arange(channels * 48000, dtype=np.float32).reshape(1, channels, 48000) / 524288
    value = assemble_video(
        np.zeros((2, 2, 2, 3), np.float32),
        fps=2,
        audio={"waveform": pcm, "sample_rate": 48000, "layout": layout},
    )
    value = edit_video(value, {"trim": {"start_time": 0.1, "duration": 0.5}})
    pieces = []
    samples = 0
    for timestamp, frame in _audio_frames(_plan(value), 0):
        assert isinstance(frame, av.AudioFrame)
        assert frame.sample_rate == 48000
        assert frame.layout.name == layout
        assert frame.format.name == "fltp"
        assert timestamp == Fraction(samples, 48000)
        assert 0 < frame.samples <= 1024
        pieces.append(frame.to_ndarray())
        samples += frame.samples
    assert samples == 24000
    np.testing.assert_array_equal(np.concatenate(pieces, axis=1)[None], pcm[..., 4800:28800])


def test_concat_matches_templates_and_spatial_dimensions() -> None:
    value = video_from_source(_source())
    joined = edit_video(value, {"concat": [value]})
    output = io.BytesIO()
    save_video_stream(joined, output)
    assert _packets(output.getvalue()) == _packets(_source()) * 2
    different = video_from_source(_source(depth=10))
    incompatible = edit_video(value, {"concat": [different]})
    with pytest.raises(ValueError, match="identical"):
        save_video_stream(incompatible, io.BytesIO())


def test_saved_hdr_samples_do_not_quantize_to_eight_bits() -> None:
    images = np.full((3, 32, 64, 3), 0.5, np.float32)
    images[:, :, 32:] += 1 / 1023
    value = assemble_video(images, fps=10, color_space="HDR PQ")
    output = io.BytesIO()
    save_video_stream(value, output, crf=0)
    reloaded = video_from_source(output.getvalue())
    decoded = cast(np.ndarray, disassemble_video(reloaded)["images"])
    assert decoded[:, :, 32:].mean() > decoded[:, :, :32].mean()
    assert cast(dict[str, Any], reloaded["probe"])["bit_depth"] == 10


def test_packet_copy_applies_metadata_before_first_mux() -> None:
    value = edit_video(video_from_source(_source()), {"trim": {"start_time": 0.5, "duration": 1}})
    output = io.BytesIO()
    save_video_stream(value, output, metadata={"title": "retained title"})
    with av.open(io.BytesIO(output.getvalue())) as decoded:
        assert decoded.metadata["title"] == "retained title"


def test_concat_validates_templates_for_disassembly_and_accepts_matching_edits() -> None:
    value = video_from_source(_source())
    crop = {"crop": {"x": 4, "y": 2, "width": 32, "height": 16}}
    cropped = edit_video(value, crop)
    with pytest.raises(ValueError, match="effective dimensions"):
        edit_video(cropped, {"concat": [value]})
    joined = edit_video(cropped, {"concat": [cropped]})
    assert cast(np.ndarray, disassemble_video(joined)["images"]).shape == (40, 16, 32, 3)
    output = io.BytesIO()
    save_video_stream(joined, output)
    assert cast(
        np.ndarray, disassemble_video(video_from_source(output.getvalue()))["images"]
    ).shape == (40, 16, 32, 3)
    incompatible = edit_video(value, {"concat": [video_from_source(_source(depth=10))]})
    with pytest.raises(ValueError, match="identical"):
        disassemble_video(incompatible)


def test_edit_parameters_are_snapshots_not_caller_owned_mappings() -> None:
    crop = {"x": 4, "y": 2, "width": 32, "height": 16}
    value = edit_video(video_from_source(_source()), {"crop": crop})
    crop["width"] = 64
    assert cast(dict[str, Any], video_meta(value)["effective"])["width"] == 32


def test_component_probe_cannot_disagree_with_carried_storage() -> None:
    value = assemble_video(np.zeros((2, 16, 32, 3), np.float32))
    cast(dict[str, object], value["probe"])["width"] = 64
    with pytest.raises(ValueError, match="probe does not match"):
        encode_video(value)


def test_disassembly_measures_missing_header_timing_without_mutating_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_values import video_codec

    data = _source()
    probe = dict(cast(dict[str, object], video_from_source(data)["probe"]))
    probe.update(fps=None, duration=None, duration_kind="unknown")
    monkeypatch.setattr(video_codec, "probe_video", lambda source: probe)
    value = video_from_source(data)
    result = disassemble_video(value)
    assert result["fps"] == 10
    assert result["duration"] == 2
    assert probe["fps"] is None and probe["duration"] is None


def test_alpha_encoding_cannot_silently_reduce_ten_bit_components() -> None:
    value = assemble_video(np.zeros((2, 16, 32, 4), np.float32), bit_depth="10")
    output = io.BytesIO()
    diagnostics: list[object] = []
    save_video_stream(
        value, output, container="webm", codec="vp9", on_diagnostic=diagnostics.append
    )
    probe = cast(dict[str, Any], video_from_source(output.getvalue())["probe"])
    assert probe["alpha"] and probe["bit_depth"] >= 10
    assert probe["video_codec"] == "ffv1"
    assert diagnostics


def test_compat_path_save_publishes_atomically(tmp_path: Path) -> None:
    from dinkster_compat_comfy.video import _VideoValue

    source = _source()
    value = _VideoValue(video_from_source(source))
    path = tmp_path / "video.mp4"
    path.write_bytes(b"existing file")
    with pytest.raises(ValueError, match="custom FFmpeg"):
        value.save_to(str(path), format="webm", codec='{"command":"unsafe"}')
    assert path.read_bytes() == b"existing file"
    assert list(tmp_path.iterdir()) == [path]
    value.save_to(str(path))
    assert path.read_bytes() == source
    assert list(tmp_path.iterdir()) == [path]


def test_native_video_chain_matches_across_worker_shared_memory() -> None:
    from dinkster_nodes_media_io import MEDIA_IO_NODES, register_media_types
    from dinkster_protocol import Invocation
    from dinkster_schema import build_node_types
    from dinkster_values import register_core_types
    from dinkster_workers import BoundaryDiagnostic, InProcessWorker, IsolatedWorker

    source = video_from_source(_source(depth=10))

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        register_media_types(registry)

        async def run(worker: InProcessWorker | IsolatedWorker) -> tuple[bytes, np.ndarray]:
            trimmed = await worker.invoke(
                Invocation(
                    invocation_id="trim",
                    node_id="trim",
                    node_type="dinkster.video.trim",
                    inputs={
                        "video": registry.wrap(VIDEO_TYPE, source),
                        "start_time": registry.wrap("FLOAT", 0.5),
                        "duration": registry.wrap("FLOAT", 1.0),
                    },
                    effective_schema=worker.schemas["dinkster.video.trim"],
                )
            )
            assert trimmed.error is None
            assert trimmed.outputs is not None
            frames = await worker.invoke(
                Invocation(
                    invocation_id="frames",
                    node_id="frames",
                    node_type="dinkster.video.disassemble",
                    inputs={"video": trimmed.outputs["video"]},
                    effective_schema=worker.schemas["dinkster.video.disassemble"],
                )
            )
            assert frames.error is None
            assert frames.outputs is not None
            assert frames.outputs["bit_depth"].resolve() == 10
            return (
                encode_video(trimmed.outputs["video"].resolve()),
                cast(np.ndarray, frames.outputs["images"].resolve()),
            )

        expected = await run(InProcessWorker(build_node_types(MEDIA_IO_NODES), registry))
        diagnostics: list[BoundaryDiagnostic] = []
        manifest = (
            Path(__file__).resolve().parents[1]
            / "packages/dinkster-nodes-media-io/dinkster-pack.toml"
        )
        worker = IsolatedWorker(
            manifest, registry, shm_threshold=64, on_diagnostic=diagnostics.append
        )
        await worker.start()
        try:
            actual = await run(worker)
            assert actual[0] == expected[0]
            np.testing.assert_array_equal(actual[1], expected[1])
            assert [
                crossing.transport
                for diagnostic in diagnostics
                for crossing in diagnostic.inputs
                if crossing.type_id == VIDEO_TYPE
            ] == ["shm", "shm"]
        finally:
            await worker.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("in_process", (True, False))
def test_native_video_values_cross_ordinary_hosts_twice_with_layered_cache(
    tmp_path: Path, in_process: bool
) -> None:
    from dinkster_graph import Graph, GraphNode, Link
    from dinkster_nodes_comfy import register_inference_types
    from dinkster_values import VIDEO_EDIT_TYPE, effective_video_facts, video_source

    from dinkster.compose import PackSpec, ServingComposer

    edit = {"trim": {"duration": 1.0, "start_time": 0.5}, "unused": {"preserved": True}}
    trim_edit = {"trim": edit["trim"]}
    source = video_from_source(_source(depth=10))
    expected = edit_video(source, trim_edit)
    expected_facts = effective_video_facts(expected)
    expected_fingerprint = video_fingerprint(VIDEO_TYPE)(expected)

    registry = TypeRegistry()
    register_inference_types(registry)
    video_spec = registry.spec(VIDEO_TYPE)
    edit_spec = registry.spec(VIDEO_EDIT_TYPE)
    register_inference_types(registry)
    assert registry.spec(VIDEO_TYPE) is video_spec
    assert registry.spec(VIDEO_EDIT_TYPE) is edit_spec
    encoded_edit = edit_spec.encode(edit)
    assert (
        encoded_edit
        == b'{"trim": {"duration": 1.0, "start_time": 0.5}, "unused": {"preserved": true}}'
    )
    assert edit_spec.decode(encoded_edit) == edit

    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        native_manifest = root / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
        fixture_manifest = Path(__file__).with_name("video-native-value-fixture.toml")
        manifests = (native_manifest, fixture_manifest)
        composer = ServingComposer(
            cache_mode="layered",
            cache_dir=tmp_path / "cache",
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": str(Path(__file__).parent),
            },
        )
        try:
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        host_types=register_inference_types
                        if manifest == native_manifest
                        else None,
                        in_process=in_process,
                        worker_group=None if in_process else "video-native-value-fixture",
                        group_manifests=() if in_process else manifests,
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("video-native-value-fixture.source", {}),
                    "consumer": GraphNode(
                        "video-native-value-fixture.consumer",
                        {
                            "video": Link("source", "video"),
                            "video_edit": Link("source", "video_edit"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for repeat in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["video_edit"].resolve() == edit
                assert result.outputs["consumer"]["facts"].resolve() == json.dumps(
                    expected_facts, default=str, sort_keys=True
                )
                assert result.outputs["consumer"]["fingerprint"].resolve() == expected_fingerprint
                assert video_source(result.outputs["consumer"]["source"].resolve()) == _source(
                    depth=10
                )
                assert encode_video(result.outputs["consumer"]["edited"].resolve()) == encode_video(
                    expected
                )
                if repeat:
                    assert set(result.cached) == {"source", "consumer"}
                else:
                    assert set(result.executed) == {"source", "consumer"}
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_concat_trim_to_second_source_never_exports_first_source() -> None:
    first, second = _source(), _source(offset=60)
    joined = edit_video(video_from_source(first), {"concat": [video_from_source(second)]})
    selected = edit_video(joined, {"trim": {"start_time": 2, "duration": 2}})
    output = io.BytesIO()
    save_video_stream(selected, output)
    assert _packets(output.getvalue()) == _packets(second)
    assert _packets(output.getvalue()) != _packets(first)
    np.testing.assert_array_equal(
        disassemble_video(video_from_source(output.getvalue()))["images"],
        disassemble_video(selected)["images"],
    )


@pytest.mark.parametrize(
    "color,depth,kr", [("sRGB", 8, 0.2126), ("HDR", 10, 0.2627), ("HDR PQ", 10, 0.2627)]
)
def test_rgb_encode_uses_declared_yuv_matrix(color: str, depth: int, kr: float) -> None:
    images = np.zeros((1, 32, 64, 3), np.float32)
    images[..., 0] = 1
    output = io.BytesIO()
    save_video_stream(
        assemble_video(images, bit_depth=str(depth), color_space=color), output, crf=0
    )
    with av.open(io.BytesIO(output.getvalue()), mode="r") as container:
        frame = next(container.decode(video=0))
        plane = frame.planes[0]
        y = np.frombuffer(plane, dtype=np.uint8 if depth == 8 else "<u2")
        y = y.reshape(plane.height, plane.line_size // (1 if depth == 8 else 2))[:, :64]
        expected = round((16 + 219 * kr) * 2 ** (depth - 8))
        assert np.max(np.abs(y.astype(int) - expected)) <= 1


def test_fractional_crf_reaches_h264_encoder_without_rounding() -> None:
    pixels = np.random.default_rng(174).integers(0, 256, (3, 32, 64, 3), dtype=np.uint8)
    value = assemble_video(pixels, fps=3)
    encoded: dict[float, bytes] = {}
    for crf in (18, 18.5, 19):
        output = io.BytesIO()
        save_video_stream(value, output, codec="h264", crf=crf)
        encoded[crf] = output.getvalue()
        # libx264 records its effective rate-control settings in the encoded SEI.
        assert f"crf={crf:.1f}".encode() in b"".join(_packets(encoded[crf]))
        with av.open(io.BytesIO(encoded[crf]), mode="r") as container:
            assert len(list(container.decode(video=0))) == 3
    assert _packets(encoded[18.5]) != _packets(encoded[18])
    assert _packets(encoded[18.5]) != _packets(encoded[19])


@pytest.mark.parametrize("codec,limit", [("h264", 51), ("vp9", 63)])
@pytest.mark.parametrize(
    "crf", [True, False, float("nan"), float("inf"), float("-inf"), -0.01, 10**1000]
)
def test_video_crf_rejects_nonfinite_boolean_and_out_of_range(
    codec: str, limit: int, crf: float
) -> None:
    value = assemble_video(np.zeros((1, 32, 64, 3), np.uint8))
    output = io.BytesIO()
    with pytest.raises(ValueError, match=f"crf must be a finite number in 0..{limit}"):
        save_video_stream(value, output, codec=codec, crf=crf)
    assert output.getvalue() == b""


@pytest.mark.parametrize("codec,limit", [("h264", 51), ("vp9", 63)])
def test_video_crf_accepts_integer_and_float_codec_bounds(codec: str, limit: int) -> None:
    value = assemble_video(np.zeros((1, 32, 64, 3), np.uint8))
    for crf in (0, 0.0, limit, float(limit)):
        output = io.BytesIO()
        save_video_stream(value, output, codec=codec, crf=crf)
        with av.open(io.BytesIO(output.getvalue()), mode="r") as container:
            assert len(list(container.decode(video=0))) == 1
    with pytest.raises(ValueError, match=f"crf must be a finite number in 0..{limit}"):
        save_video_stream(value, io.BytesIO(), codec=codec, crf=limit + 0.01)


def test_video_crf_unsupported_codec_policy_is_unchanged() -> None:
    value = assemble_video(np.zeros((1, 32, 64, 3), np.uint8))
    output = io.BytesIO()
    diagnostics: list[object] = []
    save_video_stream(value, output, codec="ffv1", crf=18.5, on_diagnostic=diagnostics.append)
    assert any(
        item["requested"] == {"crf": 18.5} and item["effective"] == {"crf": None, "codec": "ffv1"}
        for report in cast(list[dict[str, object]], diagnostics)
        for item in cast(list[dict[str, object]], report["substitutions"])
    )
    assert (
        cast(dict[str, object], video_from_source(output.getvalue())["probe"])["video_codec"]
        == "ffv1"
    )


@pytest.mark.parametrize("depth", [8, 10])
def test_av1_trim_uses_codec_identity_not_decoder_name(depth: int) -> None:
    value = assemble_video(
        np.zeros((3, 64, 64, 3), np.float32),
        fps=3,
        bit_depth=str(depth),
        color_space="HDR PQ" if depth == 10 else "sRGB",
    )
    encoded = io.BytesIO()
    save_video_stream(value, encoded, codec="av1", crf=20)
    source = video_from_source(encoded.getvalue())
    assert cast(dict[str, object], source["probe"])["video_codec"] == "av1"
    trimmed = edit_video(source, {"trim": {"start_time": 0.4, "duration": 0.4}})
    output = io.BytesIO()
    save_video_stream(trimmed, output)
    probe = cast(dict[str, object], video_from_source(output.getvalue())["probe"])
    assert probe["video_codec"] == "av1"
    assert probe["bit_depth"] == depth
    assert probe["frame_count"] == 1
    assert probe["color_space"] == ("HDR PQ" if depth == 10 else "sRGB")


@pytest.mark.parametrize(
    "matrix,depth,transfer,kr,kb",
    [(1, 8, 13, 0.2126, 0.0722), (9, 10, 18, 0.2627, 0.0593), (9, 10, 16, 0.2627, 0.0593)],
)
@pytest.mark.parametrize("full", [False, True])
def test_source_decode_and_spatial_encode_use_source_matrix_and_range(
    matrix: int, depth: int, transfer: int, kr: float, kb: float, full: bool
) -> None:
    # Independent YCbCr red samples, derived from the matrix coefficients, not an RGB roundtrip.
    unit = 2 ** (depth - 8)
    y_scale, c_scale = (2**depth - 1, 2**depth - 1) if full else (219 * unit, 224 * unit)
    levels = [
        round((0 if full else 16 * unit) + y_scale * kr),
        round(128 * unit - c_scale * kr / (2 * (1 - kb))),
        round(128 * unit + c_scale / 2),
    ]
    levels = [min(2**depth - 1, n) for n in levels]
    output = io.BytesIO()
    with av.open(output, "w", format="mp4") as opened:
        mux = cast(Any, opened)
        stream = mux.add_stream("libx264", rate=1)
        stream.width, stream.height = 64, 32
        stream.pix_fmt = "yuv420p" if depth == 8 else "yuv420p10le"
        stream.codec_context.thread_count = 1
        stream.codec_context.colorspace = matrix
        stream.codec_context.color_primaries = matrix
        stream.codec_context.color_trc = transfer
        stream.codec_context.color_range = 2 if full else 1
        stream.options = {"crf": "0"}
        frame = av.VideoFrame(64, 32, stream.pix_fmt)
        for plane, level in zip(frame.planes, levels, strict=True):
            plane.update(
                np.full(
                    plane.buffer_size // (1 if depth == 8 else 2),
                    level,
                    dtype=np.uint8 if depth == 8 else "<u2",
                ).tobytes()
            )
        frame.pts, frame.time_base = 0, Fraction(1)
        for packet in (*stream.encode(frame), *stream.encode()):
            mux.mux(packet)
    source = video_from_source(output.getvalue())
    pixels = cast(np.ndarray, image_input(disassemble_video(source)["images"]))
    np.testing.assert_allclose(pixels.mean(axis=(0, 1, 2)), [1, 0, 0], atol=0.01)
    cropped = edit_video(source, {"crop": {"width": 32, "height": 16}})
    encoded = io.BytesIO()
    save_video_stream(cropped, encoded, crf=0)
    with av.open(io.BytesIO(encoded.getvalue()), mode="r") as container:
        frame = next(container.decode(video=0))
        plane = frame.planes[0]
        y = np.frombuffer(plane, dtype=np.uint8 if depth == 8 else "<u2")
        y = y.reshape(plane.height, plane.line_size // (1 if depth == 8 else 2))[:, :32]
        assert np.max(np.abs(y.astype(int) - levels[0])) <= 1


def test_spatial_decode_preserves_known_frame_local_matrix_and_range(monkeypatch) -> None:
    import dinkster_video.runtime as runtime

    source = video_from_source(_source())
    cropped = edit_video(source, {"crop": {"width": 32, "height": 16}})
    decode = runtime._decoded_video
    seen = []

    def frame_local(container, stream):
        for frame in decode(container, stream):
            frame.colorspace = 9
            frame.color_range = 2
            yield frame

    pixels = runtime._pixels

    def capture(frame, depth, alpha):
        seen.append((frame.colorspace, frame.color_range))
        return pixels(frame, depth, alpha)

    monkeypatch.setattr(runtime, "_decoded_video", frame_local)
    monkeypatch.setattr(runtime, "_pixels", capture)
    save_video_stream(cropped, io.BytesIO(), crf=0)
    assert seen and set(seen) == {(9, 2)}


@pytest.mark.parametrize("kind", ["mp4", "mov", "mkv"])
@pytest.mark.parametrize("path", ["remux", "cut", "transcode", "components"])
def test_video_custom_metadata_is_json_on_every_mux_path(kind: str, path: str) -> None:
    value = video_from_source(_source())
    if path in ("cut", "transcode"):
        value = edit_video(
            value, {"trim": {"start_time": 0.5 if path == "cut" else 0.11, "duration": 1}}
        )
    elif path == "components":
        value = assemble_video(np.zeros((2, 32, 64, 3), np.float32))
    output = io.BytesIO()
    workflow = {"nodes": [{"id": 1}], "enabled": True, "extra": None}
    save_video_stream(
        value, output, container=kind, metadata={"workflow": workflow, "custom": "retained"}
    )
    with av.open(io.BytesIO(output.getvalue())) as container:
        tags = {k.lower(): v for k, v in container.metadata.items()}
        assert tags["custom"] == "retained"
        assert json.loads(tags["workflow"]) == workflow


@pytest.mark.parametrize("field,replacement", [("width", 128), ("bit_depth", 10), ("alpha", True)])
@pytest.mark.parametrize("asset", [False, True])
def test_probe_claims_must_match_authorized_source(
    tmp_path: Path, field: str, replacement: object, asset: bool
) -> None:
    from dinkster_assets.value import bind_video_value

    data = _source()
    value = video_from_source(data)
    cast(dict[str, object], value["probe"])[field] = replacement
    vault = AssetVault(tmp_path)
    if asset:
        digest = digest_bytes(data)
        with vault.writer(digest) as writer:
            writer.write(data)
            writer.commit()
        value["source"] = AssetRef(digest, "source.mp4", len(data)).to_wire()
    with pytest.raises(ValueError, match="probe does not match source"):
        bind_video_value(value, vault)


def test_invalid_inline_video_cannot_be_admitted_or_exported() -> None:
    value = video_from_source(_source())
    value["source"] = b"not a video"
    with pytest.raises(InvalidDataError):
        encode_video(value)
    output = io.BytesIO()
    with pytest.raises(InvalidDataError):
        save_video_stream(value, output)
    assert output.getvalue() == b""


def test_compat_trim_getter_overrides_pinned_file_getter(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import ModuleType

    from dinkster_compat_comfy.video import _upstream_video

    class VideoFromFile:
        __start_time: float
        __duration: float

        def _get_raw_duration(self) -> float:
            raise AssertionError("upstream private state must not be used")

        # ComfyUI 15eb748b3ec5 public getter, without the ComfyUI runtime bootstrap.
        def get_active_trim_window(self) -> tuple[float, float]:
            start_time = self.__start_time
            if start_time < 0:
                start_time = max(self._get_raw_duration() + start_time, 0.0)
            return float(start_time), float(self.__duration)

    module = ModuleType("comfy_api.input_impl")
    cast(Any, module).VideoFromFile = VideoFromFile
    monkeypatch.setitem(sys.modules, "comfy_api.input_impl", module)
    source = video_from_source(_source())
    value = cast(Any, _upstream_video(source))
    assert value.get_active_trim_window() == (0.0, 2.0)
    trimmed = value.as_trimmed(0.5, 1).as_cropped(0, 0, 32, 16).as_trimmed(-0.5, 0)
    assert trimmed.get_active_trim_window() == (1.0, 0.5)
    assert trimmed.get_dimensions() == (32, 16)
    assert trimmed.get_stream_source().getvalue() == source["source"]
    joined = edit_video(source, {"concat": [source]})
    with pytest.raises(ValueError, match="one original-source trim window"):
        cast(Any, _upstream_video(joined)).get_active_trim_window()
    selected = edit_video(joined, {"trim": {"start_time": 0, "duration": 1}})
    assert cast(Any, _upstream_video(selected)).get_active_trim_window() == (0.0, 1.0)
    components = assemble_video(np.zeros((1, 32, 64, 3), np.float32))
    with pytest.raises(ValueError, match="one original-source trim window"):
        cast(Any, _upstream_video(components)).get_active_trim_window()


@pytest.mark.parametrize(
    "invalid", ["unknown", "nested", "duplicate", "probe", "array", "dtype", "total"]
)
def test_invalid_video_tree_never_calls_component_decoder(
    invalid: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_values import video_codec

    value = assemble_video(np.zeros((2, 32, 64, 3), np.float32))
    data = encode_video(value)
    prefix = len(b"DINKSTER-VIDEO\x02")
    length = struct.unpack_from("<Q", data, prefix)[0]
    header = json.loads(data[prefix + 8 : prefix + 8 + length])
    body = data[prefix + 8 + length :]
    wire = header["video"]
    if invalid == "unknown":
        wire["edits"] = [{"unknown_op": {}}]
    elif invalid == "nested":
        child = json.loads(json.dumps(wire))
        child["components"]["images"]["chunk"] = 1
        child["edits"] = [{"unknown_op": {}}]
        wire["edits"] = [{"concat": [child]}]
        header["chunks"] *= 2
        body *= 2
    elif invalid == "duplicate":
        wire["edits"] = [{"concat": [json.loads(json.dumps(wire))]}]
    elif invalid == "probe":
        wire["probe"]["width"] = 128
    elif invalid == "array":
        wire["components"]["images"]["meta"]["shape"][0] = 1000
    elif invalid == "dtype":
        wire["components"]["images"]["meta"]["dtype"] = ">f4"
        body = body.replace(b"'<f4'", b"'>f4'", 1)
    else:
        body += b"extra"
    header = json.dumps(header).encode()
    damaged = data[:prefix] + struct.pack("<Q", len(header)) + header + body
    monkeypatch.setattr(
        video_codec, "decode_image_array", lambda *a: pytest.fail("decoder ran before preflight")
    )
    monkeypatch.setattr(
        video_codec, "decode_audio", lambda *a: pytest.fail("decoder ran before preflight")
    )
    with pytest.raises(ValueError):
        decode_video(damaged)


def test_nested_large_sources_publish_without_reprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _source() + struct.pack(">I4s", 300_008, b"free") + bytes(300_000)
    clip = video_from_source(data)
    joined = edit_video(clip, {"concat": [clip]})
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))
    monkeypatch.setattr(
        av, "open", lambda *a, **kw: pytest.fail("admitted source must not reprobe")
    )
    registry = TypeRegistry()
    register_video_value_type(registry, VIDEO_TYPE)
    value = registry.wrap(VIDEO_TYPE, joined)
    assert len(cast(list[object], value.meta.get("asset_refs"))) == 1
    assert value.meta.get("cost") == {"ram": 0}
    assert AssetVault(tmp_path / "vault").resolve(digest_bytes(data)) is not None


def test_compat_migrates_large_legacy_sources_without_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_compat_comfy.video import _VideoValue

    data = _source() + struct.pack(">I4s", 300_008, b"free") + bytes(300_000)
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))
    value = _VideoValue(decode_video(data))
    assert len(encode_video(value.value)) < 2000
    output = io.BytesIO()
    value.save_to(output)
    assert output.getvalue() == data


def test_alpha_source_codec_and_samples_survive_crop() -> None:
    data = io.BytesIO()
    pixels = np.zeros((32, 64, 4), np.uint8)
    pixels[..., :3] = (25, 50, 100)
    pixels[..., 3] = np.arange(64, dtype=np.uint8) * 4
    with av.open(data, "w", format="matroska") as opened:
        output = cast(Any, opened)
        stream = output.add_stream("ffv1", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 32, "bgra"
        for i in range(3):
            frame = av.VideoFrame.from_ndarray(pixels, format="rgba")
            frame.pts, frame.time_base = i, Fraction(1, 10)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    value = edit_video(
        video_from_source(data.getvalue()), {"crop": {"x": 4, "y": 2, "width": 32, "height": 16}}
    )
    saved = io.BytesIO()
    save_video_stream(value, saved)
    result = video_from_source(saved.getvalue())
    probe = cast(dict[str, Any], result["probe"])
    assert (probe["video_codec"], probe["alpha"]) == ("ffv1", True)
    actual = cast(np.ndarray, image_input(disassemble_video(result)["images"]))
    np.testing.assert_array_equal(np.rint(actual[0] * 255).astype(np.uint8), pixels[2:18, 4:36])


def test_nested_video_source_preflight_and_remote_staging_fetch_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aiohttp import web
    from dinkster_graph import Graph, GraphNode, TypedLiteral
    from dinkster_protocol import Invocation
    from dinkster_server.preflight import graph_asset_names
    from test_remote import start_service, stop_service
    from test_remote_asset_staging import ENDPOINT_TOKEN, AssetHost
    from test_remote_job_assets import job_worker, write_assetpack_manifest

    data = _source()
    digest = digest_bytes(data)
    wire = AssetRef(digest, "source.mp4", len(data)).to_wire()
    clip = {**video_from_source(data), "source": wire}
    joined = edit_video(clip, {"concat": [clip]})
    graph = Graph(nodes={"n": GraphNode("apack.read", {"data": TypedLiteral(VIDEO_TYPE, joined)})})
    assert graph_asset_names(graph) == {digest: "source.mp4"}

    class VideoHost(AssetHost):
        async def _handle(self, request: web.Request) -> web.StreamResponse:
            self.requests += 1
            assert request.headers.get("Authorization") == f"Bearer {ENDPOINT_TOKEN}"
            assert request.match_info["digest"] == digest
            return web.Response(body=data)

    async def scenario() -> None:
        root = tmp_path / "daemon-vault"
        proc, host, port = await start_service(
            write_assetpack_manifest(tmp_path), tmp_path, "--asset-vault", str(root)
        )
        try:
            async with VideoHost() as endpoint:
                worker = job_worker(host, port, endpoint.endpoint)
                await worker.start()
                try:
                    registry = TypeRegistry()
                    register_video_value_type(registry, VIDEO_TYPE)
                    value = registry.wrap(VIDEO_TYPE, joined)
                    invocation = Invocation(
                        invocation_id="video",
                        node_id="n",
                        node_type="apack.read",
                        inputs={"data": value},
                        effective_schema=worker.schemas["apack.read"],
                    )
                    monkeypatch.setattr(
                        av,
                        "open",
                        lambda *a, **kw: pytest.fail("staging must not decode on producer"),
                    )
                    await worker._stage_job_assets(invocation)
                    await worker._stage_job_assets(invocation)
                    assert endpoint.requests == 1
                    stored = AssetVault(root).resolve(digest)
                    assert stored is not None and stored.read_bytes() == data
                finally:
                    await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_remote_load_then_trim_preserves_qualified_cost_and_reuses_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aiohttp import web
    from dinkster_caches import MemoryLRUCache
    from dinkster_engine import Engine
    from dinkster_graph import Graph, GraphNode, Link
    from dinkster_nodes_media_io import register_media_types
    from dinkster_values import register_core_types
    from dinkster_workers import RemoteWorker
    from test_remote import TLS_ARGS, TLS_CERT, TOKEN, start_service, stop_service
    from test_remote_asset_staging import ENDPOINT_TOKEN, AssetHost

    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "engine-vault"))
    data = _source()
    digest = digest_bytes(data)
    wire = AssetRef(digest, "source.mp4", len(data)).to_wire()
    graph = Graph(
        nodes={
            "load": GraphNode("dinkster.load_video_value", {"video": wire}),
            "trim": GraphNode(
                "dinkster.video.trim",
                {"video": Link("load", "video"), "start_time": 0.5, "duration": 0.5},
            ),
        }
    )
    loaded = {**video_from_source(data), "source": wire}
    expected = edit_video(
        loaded,
        {"trim": {"start_time": 0.5, "duration": 0.5}},
    )
    encoded_sizes = {
        "load": len(encode_video(loaded)),
        "trim": len(encode_video(expected)),
    }

    class VideoHost(AssetHost):
        async def _handle(self, request: web.Request) -> web.StreamResponse:
            self.requests += 1
            assert request.headers.get("Authorization") == f"Bearer {ENDPOINT_TOKEN}"
            assert request.match_info["digest"] == digest
            return web.Response(body=data)

    async def scenario() -> None:
        manifest = Path(__file__).parents[1] / "packages/dinkster-nodes-media-io/dinkster-pack.toml"
        proc, host, port = await start_service(
            manifest, tmp_path, *TLS_ARGS, "--asset-vault", str(tmp_path / "daemon-vault")
        )
        try:
            async with VideoHost() as endpoint:
                registry = TypeRegistry()
                register_core_types(registry)
                register_media_types(registry)
                worker = RemoteWorker(
                    host,
                    port,
                    TOKEN,
                    registry,
                    name="remote-video",
                    tls_ca_file=TLS_CERT,
                    asset_endpoint=endpoint.endpoint,
                    asset_endpoint_token=ENDPOINT_TOKEN,
                )
                await worker.start()
                try:
                    monkeypatch.setattr(
                        av, "open", lambda *a, **kw: pytest.fail("engine must not decode VIDEO")
                    )
                    for _ in range(2):
                        result = await Engine(
                            schemas=dict(worker.schemas),
                            registry=registry,
                            worker=worker,
                            cache=MemoryLRUCache(),
                        ).run(graph, ["load", "trim"])
                        assert result.executed == ("load", "trim")
                        assert result.cached == ()
                        assert result.skipped == ()
                        for node in ("load", "trim"):
                            assert result.outputs[node]["video"].meta.get("cost") == {
                                "ram@remote-video": encoded_sizes[node]
                            }
                        payload = result.outputs["trim"]["video"].payload
                        assert isinstance(payload, EncodedPayload)
                        assert payload.data == encode_video(expected)
                        assert endpoint.requests == 1
                finally:
                    await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())
