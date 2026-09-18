"""CPU format facts and independent alpha/precision goldens, not stream-copy parity."""

from __future__ import annotations

import io
from collections.abc import Mapping
from fractions import Fraction
from typing import Any, cast
from zipfile import ZipFile

import av
import numpy as np
import pytest
from dinkster_api.v1 import annotate_image
from dinkster_values import edit_video, video_from_source
from dinkster_video import (
    DITHERS,
    FRAME_FORMATS,
    assemble_video,
    disassemble_video,
    read_video_metadata,
    save_video_frames,
    save_video_stream,
)
from PIL import Image

METADATA = {
    "workflow": {"nodes": [{"id": 1}]},
    "Custom": "true",
    "custom": [1, None],
    "comment": "a=b;#\\\n",
}


def pixels(alpha: bool = False) -> np.ndarray:
    result = np.zeros((3, 64, 64, 4 if alpha else 3), np.float32)
    result[..., :3] = (0.2, 0.4, 0.6)
    result[1, ..., :3] = (0.4, 0.2, 0.1)
    result[2, ..., :3] = (0.1, 0.3, 0.7)
    if alpha:
        result[..., 3] = np.linspace(0, 1, 64, dtype=np.float32)
    return result


@pytest.mark.parametrize(
    "codec,container",
    [
        ("h264", "mp4"),
        ("hevc", "mp4"),
        ("av1", "webm"),
        ("vp9", "webm"),
        ("vp8", "webm"),
        ("prores", "mov"),
        ("ffv1", "mkv"),
    ],
)
def test_cpu_codec_facts_pixels_metadata(codec: str, container: str) -> None:
    expected = pixels()
    output = io.BytesIO()
    save_video_stream(
        assemble_video(expected, fps=10),
        output,
        codec=codec,
        container=container,
        metadata=METADATA,
    )
    data = output.getvalue()
    value = video_from_source(data)
    probe = cast(dict[str, Any], value["probe"])
    assert probe["video_codec"] == codec
    assert probe["container"] == container
    result = disassemble_video(value)
    assert result["frame_count"] == 3
    np.testing.assert_allclose(cast(np.ndarray, result["images"]), expected, atol=0.025)
    assert read_video_metadata(data) == METADATA


@pytest.mark.parametrize("codec,container", [("h264", "mp4"), ("hevc", "mp4"), ("av1", "webm")])
@pytest.mark.parametrize("color,transfer", [("HDR", 18), ("HDR PQ", 16)])
def test_cpu_hdr_encoded_precision_and_color(
    codec: str, container: str, color: str, transfer: int
) -> None:
    expected = np.full((3, 64, 64, 3), 0.5, np.float32)
    expected[:, :, 32:] += 2 / 1023
    output = io.BytesIO()
    save_video_stream(
        assemble_video(expected, fps=10, color_space=color),
        output,
        codec=codec,
        container=container,
        crf=0,
    )
    value = video_from_source(output.getvalue())
    probe = cast(dict[str, Any], value["probe"])
    assert [probe[k] for k in ("bit_depth", "primaries", "transfer", "matrix", "range")] == [
        10,
        9,
        transfer,
        9,
        1,
    ]
    actual = cast(np.ndarray, disassemble_video(value)["images"])
    assert actual[:, :, 32:].mean() - actual[:, :, :32].mean() > 1 / 1023
    np.testing.assert_allclose(actual, expected, atol=0.003)


@pytest.mark.parametrize("profile", ["lt", "standard", "hq", "4444", "4444xq"])
def test_prores_profiles(profile: str) -> None:
    output = io.BytesIO()
    save_video_stream(
        assemble_video(pixels(), fps=10), output, codec="prores", container="mov", profile=profile
    )
    with av.open(io.BytesIO(output.getvalue()), mode="r") as source:
        stream = source.streams.video[0]
        assert (
            stream.codec_context.profile
            == {"lt": "LT", "standard": "Standard", "hq": "HQ", "4444": "4444", "4444xq": "XQ"}[
                profile
            ]
        )
        assert len(list(source.decode(video=0))) == 3


@pytest.mark.parametrize(
    "codec,container,depth",
    [("vp9", "webm", 8), ("prores", "mov", 10), ("ffv1", "mkv", 8), ("ffv1", "mkv", 10)],
)
def test_independent_alpha_gradient(codec: str, container: str, depth: int) -> None:
    expected = pixels(True)
    output = io.BytesIO()
    save_video_stream(
        assemble_video(expected, fps=10, bit_depth=str(depth)),
        output,
        codec=codec,
        container=container,
    )
    value = video_from_source(output.getvalue())
    probe = cast(dict[str, Any], value["probe"])
    assert probe["alpha"] is True
    assert probe["bit_depth"] >= depth
    actual = cast(np.ndarray, disassemble_video(value)["images"])
    np.testing.assert_allclose(
        actual[..., 3], expected[..., 3], atol=1 / 255 if depth == 8 else 1 / 1023
    )
    np.testing.assert_allclose(actual[..., :3], expected[..., :3], atol=0.025)


@pytest.mark.parametrize("codec,container", [("ffv1", "mkv"), ("prores", "mov")])
def test_stream_encoder_converts_premultiplied_components_to_straight_alpha(
    codec: str, container: str
) -> None:
    expected = np.full((2, 64, 64, 4), 0.5, np.float32)
    premultiplied = expected.copy()
    premultiplied[..., :3] *= premultiplied[..., 3:]
    output = io.BytesIO()
    save_video_stream(
        assemble_video(annotate_image(premultiplied, alpha="premultiplied"), fps=10),
        output,
        codec=codec,
        container=container,
    )
    actual = cast(np.ndarray, disassemble_video(video_from_source(output.getvalue()))["images"])
    np.testing.assert_allclose(actual, expected, atol=0.025)


@pytest.mark.parametrize(
    "codec,container,expects_fallback",
    [("ffv1", "mkv", False), ("h264", "mp4", True)],
)
def test_premultiplied_spatial_padding_encodes_straight_pad_color(
    codec: str, container: str, expects_fallback: bool
) -> None:
    pixels = np.full((1, 32, 64, 4), 0.25, np.float32)
    pixels[..., 3] = 0.5
    video = assemble_video(annotate_image(pixels, alpha="premultiplied"), fps=1)
    video = edit_video(
        video,
        {
            "scale": {
                "width": 64,
                "height": 64,
                "fit": "pad",
                "pad_color": [0.25, 0, 0, 0.5],
            }
        },
    )
    output = io.BytesIO()
    diagnostics: list[Mapping[str, object]] = []
    save_video_stream(
        video,
        output,
        codec=codec,
        container=container,
        on_diagnostic=diagnostics.append,
    )
    actual = cast(np.ndarray, disassemble_video(video_from_source(output.getvalue()))["images"])
    np.testing.assert_allclose(actual[0, 0, 0], [0.25, 0, 0, 0.5], atol=2 / 255)
    assert bool(diagnostics) is expects_fallback
    if expects_fallback:
        effective = cast(Mapping[str, object], diagnostics[0]["effective"])
        assert effective["codec"] == "ffv1"


@pytest.mark.parametrize("format", FRAME_FORMATS)
@pytest.mark.parametrize("alpha", [False, True])
def test_frame_exports_pixels_loop_depth_and_metadata(format: str, alpha: bool) -> None:
    expected = pixels(alpha)
    if format.startswith("gif") and alpha:
        expected[..., 3] = (expected[..., 3] > 0.5).astype(np.float32)
    output = io.BytesIO()
    save_video_frames(
        assemble_video(expected, fps=10), output, format=format, loop=3, metadata=METADATA
    )
    data = output.getvalue()
    assert read_video_metadata(data) == METADATA
    if format.startswith("png"):
        depth = 16 if format == "png16" else 8
        with ZipFile(io.BytesIO(data)) as archive:
            assert archive.namelist() == ["000001.png", "000002.png", "000003.png"]
            for index, name in enumerate(archive.namelist()):
                encoded = archive.read(name)
                assert encoded[24] == depth
                assert read_video_metadata(encoded) == (METADATA if index == 0 else {})
                with av.open(io.BytesIO(encoded), mode="r") as png:
                    frame = next(png.decode(video=0))
                    fmt = (
                        ("rgba64le" if alpha else "rgb48le")
                        if depth == 16
                        else "rgba"
                        if alpha
                        else "rgb24"
                    )
                    actual = frame.to_ndarray(format=fmt).astype(np.float64) / (2**depth - 1)
                    np.testing.assert_allclose(actual, expected[index], atol=1 / (2**depth - 1))
    else:
        with Image.open(io.BytesIO(data)) as image:
            assert cast(Any, image).n_frames == 3
            assert image.info["loop"] == 3
            for index in range(3):
                image.seek(index)
                actual = np.asarray(image.convert("RGBA" if alpha else "RGB")) / 255
                if alpha:
                    np.testing.assert_allclose(
                        actual[..., 3], expected[index, ..., 3], atol=1 / 255
                    )
                    visible = expected[index, ..., 3] > 0
                    np.testing.assert_allclose(
                        actual[visible, :3], expected[index][visible, :3], atol=0.02
                    )
                else:
                    np.testing.assert_allclose(actual, expected[index], atol=0.02)


@pytest.mark.parametrize("dither", DITHERS)
def test_every_ffmpeg_gif_dither(dither: str) -> None:
    expected = pixels()
    expected[..., 0] = np.linspace(0, 1, 64)[None, None, :]
    expected[..., 1] = np.linspace(0, 1, 64)[None, :, None]
    output = io.BytesIO()
    save_video_frames(assemble_video(expected, fps=10), output, format="gif_ffmpeg", dither=dither)
    with Image.open(io.BytesIO(output.getvalue())) as image:
        assert cast(Any, image).n_frames == 3
        for index in range(3):
            image.seek(index)
            actual = np.asarray(image.convert("RGB")) / 255
            assert np.mean(np.abs(actual - expected[index])) < 0.025


@pytest.mark.parametrize("trim", [False, True])
def test_short_audio_trim_versus_silence_padding(trim: bool) -> None:
    audio = {"waveform": np.full((1, 2, 24000), 0.1, np.float32), "sample_rate": 48000}
    value = assemble_video(np.tile(pixels(), (4, 1, 1, 1)), fps=10, audio=audio)
    output = io.BytesIO()
    save_video_stream(value, output, codec="ffv1", container="mkv", trim_to_audio=trim)
    with av.open(io.BytesIO(output.getvalue()), mode="r") as source:
        assert len(list(source.decode(video=0))) == (5 if trim else 12)
    with av.open(io.BytesIO(output.getvalue()), mode="r") as source:
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=48000)
        frames = [
            f.to_ndarray()
            for decoded in source.decode(audio=0)
            for f in resampler.resample(decoded)
        ]
        actual = np.concatenate(frames, axis=1)
        assert actual.shape[1] == (24000 if trim else 57600)
        if not trim:
            assert np.count_nonzero(actual[:, 24000:]) == 0


def test_webp_lossy_quality_and_infinite_loop() -> None:
    output = io.BytesIO()
    save_video_frames(
        assemble_video(pixels(True), fps=10), output, format="webp", lossless=False, quality=95
    )
    with Image.open(io.BytesIO(output.getvalue())) as image:
        assert image.info["loop"] == 0
        assert cast(Any, image).n_frames == 3
        np.testing.assert_allclose(
            np.asarray(image.convert("RGBA"))[..., 3] / 255, pixels(True)[0, ..., 3], atol=1 / 255
        )


def test_png16_keeps_sub_eight_bit_precision() -> None:
    expected = pixels(True)
    expected[:, :, 32:, 0] += 1 / 65535
    output = io.BytesIO()
    save_video_frames(assemble_video(expected, bit_depth="10"), output, format="png16")
    with ZipFile(io.BytesIO(output.getvalue())) as archive:
        with av.open(io.BytesIO(archive.read("000001.png")), mode="r") as source:
            actual = next(source.decode(video=0)).to_ndarray(format="rgba64le")
            assert actual[0, 32, 0] - int(actual[0, 0, 0]) == 1


def test_arbitrary_commands_are_never_executed() -> None:
    with pytest.raises(ValueError, match="custom FFmpeg"):
        save_video_frames(
            assemble_video(pixels()), io.BytesIO(), format='{"main_pass":["-f","evil"]}'
        )


@pytest.mark.parametrize("codec", ["h264_nvenc", "hevc_nvenc", "av1_nvenc"])
def test_nvenc_is_refused_without_hardware_encoder_policy(codec: str) -> None:
    destination = io.BytesIO()
    with pytest.raises(ValueError, match="NVENC is refused"):
        save_video_stream(assemble_video(pixels()), destination, codec=codec)
    assert destination.getvalue() == b""


def surround_source(audio_samples: int = 14400) -> bytes:
    output = io.BytesIO()
    with av.open(output, "w", format="matroska") as opened:
        mux = cast(Any, opened)
        video = mux.add_stream("ffv1", rate=10)
        video.width, video.height, video.pix_fmt = 64, 64, "bgr0"
        audio = mux.add_stream("flac", rate=48000)
        audio.layout = "5.1"
        for index, array in enumerate(pixels()):
            frame = av.VideoFrame.from_ndarray(
                np.rint(array * 255).astype(np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = index, Fraction(1, 10)
            for packet in video.encode(frame):
                mux.mux(packet)
        for packet in video.encode():
            mux.mux(packet)
        signal = np.zeros((6, 14400), np.int16)
        for channel in range(6):
            # In-band tones measure channel identity without relying on codec DC response.
            signal[channel, channel * 2000 : channel * 2000 + 1000] = np.rint(
                12000 * np.sin(2 * np.pi * (96 if channel == 3 else 960) * np.arange(1000) / 48000)
            ).astype(np.int16)
        frame = av.AudioFrame.from_ndarray(
            signal[:, :audio_samples].T.reshape(1, -1), format="s16", layout="5.1"
        )
        frame.sample_rate, frame.pts = 48000, 0
        for packet in (*audio.encode(frame), *audio.encode()):
            mux.mux(packet)
    return output.getvalue()


@pytest.mark.parametrize(
    "codec,container",
    [
        ("h264", "mp4"),
        ("hevc", "mp4"),
        ("vp9", "webm"),
        ("av1", "webm"),
        ("prores", "mov"),
        ("ffv1", "mkv"),
    ],
)
@pytest.mark.parametrize("layout,channels", [("preserve", 6), ("stereo", 2), ("mono", 1)])
def test_declared_surround_and_explicit_downmix(
    codec: str, container: str, layout: str, channels: int
) -> None:
    value = video_from_source(surround_source())
    value = edit_video(value, {"crop": {"x": 0, "y": 0, "width": 64, "height": 64}})
    output = io.BytesIO()
    save_video_stream(
        value,
        output,
        codec=codec,
        container=container,
        audio_layout=layout,
        crf=20 if codec in ("h264", "hevc", "av1", "vp9") else 0,
    )
    with av.open(io.BytesIO(output.getvalue()), mode="r") as source:
        assert source.streams.video[0].codec_context.codec.canonical_name == codec
        stream = source.streams.audio[0]
        assert stream.codec_context.channels == channels
        assert stream.codec_context.layout.name == {6: "5.1", 2: "stereo", 1: "mono"}[channels]
        resampler = av.AudioResampler(
            format="fltp", layout=stream.codec_context.layout.name, rate=48000
        )
        chunks = [
            f.to_ndarray()
            for decoded in source.decode(audio=0)
            for f in resampler.resample(decoded)
        ]
        actual = np.concatenate(chunks, axis=1)
        assert actual.shape[1] >= 14400
        if channels == 6:
            for channel in range(6):
                section = actual[channel, channel * 2000 + 200 : channel * 2000 + 800]
                assert 0.18 < float(np.sqrt(np.mean(section**2))) < 0.32
                others = np.delete(
                    actual[:, channel * 2000 + 200 : channel * 2000 + 800], channel, 0
                )
                assert np.max(np.abs(others)) < 0.025
        else:
            # Center and rear channels contribute to explicit downmix; LFE exclusion is intentional.
            for channel in (0, 1, 2, 4, 5):
                assert np.max(np.abs(actual[:, channel * 2000 + 200 : channel * 2000 + 800])) > 0.01


@pytest.mark.parametrize(
    "codec,container,alpha,depth,effective",
    [
        ("h264", "mp4", True, 8, "ffv1"),
        ("av1", "webm", True, 8, "ffv1"),
        ("vp9", "webm", True, 10, "ffv1"),
        ("unknown", "unknown", True, 10, "ffv1"),
    ],
)
def test_preserving_defaults_report_requested_and_actual(
    codec: str, container: str, alpha: bool, depth: int, effective: str
) -> None:
    diagnostics: list[Any] = []
    expected = pixels(alpha)
    output = io.BytesIO()
    save_video_stream(
        assemble_video(expected, bit_depth=str(depth)),
        output,
        codec=codec,
        container=container,
        on_diagnostic=diagnostics.append,
    )
    assert diagnostics[0]["code"] == "media_format_fallback"
    assert diagnostics[0]["requested"]["codec"] == codec
    assert diagnostics[0]["effective"]["codec"] == effective
    value = video_from_source(output.getvalue())
    probe = cast(dict[str, Any], value["probe"])
    assert probe["video_codec"] == effective
    assert probe["alpha"] == alpha and probe["bit_depth"] >= depth
    np.testing.assert_allclose(
        cast(np.ndarray, disassemble_video(value)["images"]), expected, atol=0.025
    )


@pytest.mark.parametrize("format", FRAME_FORMATS)
def test_export_never_calls_full_batch_materializer(
    format: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_video import runtime

    monkeypatch.setattr(runtime, "disassemble_video", lambda *a: pytest.fail("full frame batch"))
    encoded = []
    for _ in range(2):
        output = io.BytesIO()
        save_video_frames(assemble_video(pixels(), fps=10), output, format=format)
        encoded.append(output.getvalue())
    assert encoded[0] == encoded[1]


@pytest.mark.parametrize("format", ["gif_pillow", "gif_ffmpeg", "webp"])
def test_animation_uses_actual_variable_frame_timestamps(format: str) -> None:
    original = io.BytesIO()
    with av.open(original, "w", format="matroska") as opened:
        mux = cast(Any, opened)
        stream = mux.add_stream("ffv1", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "bgr0"
        for timestamp, array in zip((0, 1, 4), pixels(), strict=True):
            frame = av.VideoFrame.from_ndarray(
                np.rint(array * 255).astype(np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = timestamp, Fraction(1, 10)
            for packet in stream.encode(frame):
                mux.mux(packet)
        for packet in stream.encode():
            mux.mux(packet)
    output = io.BytesIO()
    save_video_frames(video_from_source(original.getvalue()), output, format=format)
    with Image.open(io.BytesIO(output.getvalue())) as image:
        durations = []
        for index in range(cast(Any, image).n_frames):
            image.seek(index)
            image.load()
            durations.append(image.info["duration"])
        assert durations == [100, 300, 100]


@pytest.mark.parametrize("format", ["gif_pillow", "gif_ffmpeg"])
def test_gif_quantization_does_not_accumulate_frame_duration_error(format: str) -> None:
    output = io.BytesIO()
    save_video_frames(
        assemble_video(np.tile(pixels(), (8, 1, 1, 1)), fps=24), output, format=format
    )
    with Image.open(io.BytesIO(output.getvalue())) as image:
        duration = 0
        for index in range(cast(Any, image).n_frames):
            image.seek(index)
            duration += image.info["duration"]
        assert duration == 1000


@pytest.mark.parametrize("codec", ["h264", "hevc", "av1", "vp9"])
def test_preserving_default_keeps_odd_geometry(codec: str) -> None:
    expected = pixels()[:, :7, :9]
    output = io.BytesIO()
    diagnostics: list[Any] = []
    save_video_stream(
        assemble_video(expected), output, codec=codec, on_diagnostic=diagnostics.append
    )
    assert diagnostics
    actual = cast(np.ndarray, disassemble_video(video_from_source(output.getvalue()))["images"])
    np.testing.assert_allclose(actual, expected, atol=1 / 255)


@pytest.mark.parametrize("format", ["ffv1", *FRAME_FORMATS])
def test_hdr_alpha_defaults_retain_sixteen_bit_samples(format: str) -> None:
    expected = pixels(True)
    expected[:, :, 32:, 0] += 1 / 65535
    value = assemble_video(expected, bit_depth="10", color_space="HDR PQ")
    output = io.BytesIO()
    if format == "ffv1":
        save_video_stream(value, output, codec="ffv1")
    else:
        save_video_frames(value, output, format=format)
    data = output.getvalue()
    if format.startswith("png"):
        with ZipFile(io.BytesIO(data)) as archive:
            data = archive.read("000001.png")
        assert data[24] == 16
        assert b"cICP" + bytes((9, 16, 0, 1)) in data
    else:
        probe = cast(dict[str, Any], video_from_source(data)["probe"])
        assert (probe["bit_depth"], probe["transfer"], probe["alpha"]) == (16, 16, True)
    with av.open(io.BytesIO(data), "r") as opened:
        frame = next(opened.decode(video=0))
        actual = frame.to_ndarray(format="rgba64le")
        np.testing.assert_array_equal(actual, np.rint(expected[0] * 65535).astype(np.uint16))


def test_source_copy_does_not_bypass_short_audio_padding() -> None:
    output = io.BytesIO()
    save_video_stream(video_from_source(surround_source(4800)), output)
    with av.open(io.BytesIO(output.getvalue()), "r") as opened:
        frames = [frame.to_ndarray() for frame in opened.decode(audio=0)]
        actual = np.concatenate(frames, axis=1).reshape(-1, 6)
        assert actual.shape == (14400, 6)
        assert np.count_nonzero(actual[4800:]) == 0


def test_native_encoded_audio_endpoint_still_trims_video() -> None:
    output, diagnostics = io.BytesIO(), []
    save_video_stream(
        video_from_source(surround_source(4800)),
        output,
        codec="ffv1",
        container="mkv",
        trim_to_audio=True,
        on_diagnostic=diagnostics.append,
    )
    assert diagnostics == []
    with av.open(io.BytesIO(output.getvalue()), "r") as opened:
        assert len(list(opened.decode(video=0))) == 1
    with av.open(io.BytesIO(output.getvalue()), "r") as opened:
        assert sum(frame.samples for frame in opened.decode(audio=0)) == 4800


@pytest.mark.parametrize("codec,container", [("h264", "mp4"), ("vp9", "webm"), ("ffv1", "mkv")])
def test_discrete_channels_use_preserving_pcm_default(codec: str, container: str) -> None:
    original = io.BytesIO()
    expected = np.arange(7 * 14400, dtype=np.float32).reshape(14400, 7) / (7 * 14400)
    with av.open(original, "w", format="matroska") as opened:
        mux = cast(Any, opened)
        video = mux.add_stream("ffv1", rate=10)
        video.width, video.height, video.pix_fmt = 64, 64, "bgr0"
        audio = mux.add_stream("pcm_f32le", rate=48000)
        audio.layout = "7 channels"
        for array in pixels():
            frame = av.VideoFrame.from_ndarray(
                np.rint(array * 255).astype(np.uint8), format="rgb24"
            )
            for packet in video.encode(frame):
                mux.mux(packet)
        for packet in video.encode():
            mux.mux(packet)
        frame = av.AudioFrame.from_ndarray(
            expected.reshape(1, -1), format="flt", layout="7 channels"
        )
        frame.sample_rate = 48000
        for packet in (*audio.encode(frame), *audio.encode()):
            mux.mux(packet)
    value = edit_video(
        video_from_source(original.getvalue()),
        {"crop": {"x": 0, "y": 0, "width": 64, "height": 64}},
    )
    output = io.BytesIO()
    diagnostics: list[Any] = []
    save_video_stream(
        value, output, codec=codec, container=container, crf=0, on_diagnostic=diagnostics.append
    )
    assert diagnostics[0]["audioStreams"] == [{"codec": "pcm_f32le", "channelLayout": "7 channels"}]
    with av.open(io.BytesIO(output.getvalue()), "r") as opened:
        stream = opened.streams.audio[0]
        assert stream.codec_context.name == "pcm_f32le"
        assert stream.layout.name == "7 channels"
        actual = np.concatenate([f.to_ndarray() for f in opened.decode(audio=0)], axis=1)
        np.testing.assert_array_equal(actual.reshape(-1, 7), expected)
    with pytest.raises(ValueError, match="explicit canonical AUDIO channel_map matrix"):
        save_video_stream(
            value, io.BytesIO(), codec=codec, container=container, audio_layout="stereo"
        )


@pytest.mark.parametrize("enum", [0, 2, 128, 255])
def test_png_preserves_reserved_color_enums_through_shared_authority(enum: int) -> None:
    from dinkster_video import formats

    frame = av.VideoFrame.from_ndarray(np.full((8, 8, 3), 102, np.uint8), format="rgb24")
    data = formats._png(frame, 8, False, "", {"primaries": enum, "transfer": enum, "range": 1})
    assert b"cICP" + bytes((enum, enum, 0, 1)) in data
    with Image.open(io.BytesIO(data)) as image:
        np.testing.assert_array_equal(np.asarray(image), np.full((8, 8, 3), 102, np.uint8))


@pytest.mark.parametrize("invalid", [-1, 1.5, "sRGB", True])
def test_png_uses_shared_color_validation_without_coercing_enums(invalid: object) -> None:
    from dinkster_video import formats

    frame = av.VideoFrame.from_ndarray(np.zeros((8, 8, 3), np.uint8), format="rgb24")
    with pytest.raises(ValueError, match="invalid image color primaries"):
        formats._png(frame, 8, False, "", {"primaries": invalid, "transfer": 13, "range": 2})
