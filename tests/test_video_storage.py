"""Compact VIDEO pixels and PCM components normalize only at consumer boundaries."""

from __future__ import annotations

import io
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import pytest
from dinkster_assets.value import bind_video_value
from dinkster_nodes_media_io import video as media_video
from dinkster_values import (
    annotate_image,
    decode_image_array,
    decode_video,
    edit_video,
    encode_image_array,
    encode_video,
    image_array_meta,
    video_from_source,
)
from dinkster_values.storage import BFLOAT16_FIELD, image_input
from dinkster_video import assemble_video, disassemble_video, runtime, save_video_stream


class _Source:
    def __init__(self, data: bytes):
        self.data = data

    def open(self) -> io.BytesIO:
        return io.BytesIO(self.data)


def _gray_source(depth: int, transfer: int = 16) -> bytes:
    data = io.BytesIO()
    with av.open(data, "w", format="mp4") as opened:
        mux = cast(Any, opened)
        stream = mux.add_stream("libx264", rate=2)
        stream.width, stream.height = 64, 32
        stream.pix_fmt = "yuv420p" if depth == 8 else "yuv420p10le"
        stream.codec_context.thread_count = 1
        stream.codec_context.colorspace = 9 if depth == 10 else 1
        stream.codec_context.color_primaries = 9 if depth == 10 else 1
        stream.codec_context.color_trc = transfer if depth == 10 else 13
        stream.codec_context.color_range = 1
        stream.options = {"crf": "0"}
        for index in range(2):
            frame = av.VideoFrame(64, 32, stream.pix_fmt)
            for plane_index, plane in enumerate(frame.planes):
                level = 2 ** (depth - 1) + (index if plane_index == 0 else 0)
                plane.update(
                    np.full(
                        plane.buffer_size // (1 if depth == 8 else 2),
                        level,
                        np.uint8 if depth == 8 else np.uint16,
                    ).tobytes()
                )
            frame.pts, frame.time_base = index, Fraction(1, 2)
            for packet in stream.encode(frame):
                mux.mux(packet)
        for packet in stream.encode():
            mux.mux(packet)
    return data.getvalue()


def _decode_legacy(data: bytes) -> np.ndarray:
    return media_video.decode_video_frames(
        _Source(data),
        force_rate=0,
        custom_width=0,
        custom_height=0,
        frame_load_cap=0,
        start_time=0,
        select_every_nth=1,
    )[0]


@pytest.mark.parametrize("depth", [8, 10])
def test_decoded_storage_retains_source_precision_and_color_math(depth: int) -> None:
    data = _gray_source(depth)
    result = disassemble_video(video_from_source(data))
    images = cast(np.ndarray, result["images"])
    dtype = np.uint8 if depth == 8 else np.uint16
    maximum = np.iinfo(dtype).max
    assert images.dtype == dtype
    assert images.nbytes == images.size * np.dtype(dtype).itemsize
    assert result["color_space"] == ("HDR PQ" if depth == 10 else "sRGB")
    assert result["bit_depth"] == depth
    with av.open(io.BytesIO(data), mode="r") as container:
        for actual, decoded in zip(images, container.decode(video=0), strict=True):
            frame = cast(Any, decoded)
            reference = frame.reformat(
                format="gbrpf32le",
                src_colorspace=frame.colorspace,
                dst_colorspace=0,
                src_color_range=1,
                dst_color_range=2,
            ).to_ndarray()
            expected = np.rint(np.clip(reference, 0, 1) * maximum).astype(dtype)
            np.testing.assert_array_equal(actual, expected)
    legacy = _decode_legacy(data)
    assert legacy.dtype == dtype
    assert legacy.shape == images.shape
    if depth == 10:
        assert images[1].mean() > images[0].mean()
        assert legacy[1].mean() > legacy[0].mean()
        assert np.unique(np.rint(images / 257)).size == 1


@pytest.mark.parametrize("transfer", [16, 18])
def test_hdr_color_and_source_depth_survive_image_hop_and_reencode(transfer: int) -> None:
    source = video_from_source(_gray_source(10, transfer))
    images = disassemble_video(source)["images"]
    color = {"primaries": 9, "transfer": transfer, "matrix": 9, "range": 2, "bit_depth": 10}
    assert image_array_meta(images)["color"] == color
    carried = cast(np.ndarray, decode_image_array(encode_image_array(images)))
    video = assemble_video(carried, fps=2)
    probe = cast(dict[str, Any], video["probe"])
    assert [probe[key] for key in ("primaries", "transfer", "matrix", "range")] == [
        9,
        transfer,
        9,
        1,
    ]
    assert probe["bit_depth"] == 10
    target = io.BytesIO()
    save_video_stream(video, target, container="mp4", codec="h264", crf=0)
    saved = video_from_source(target.getvalue())
    saved_probe = cast(dict[str, Any], saved["probe"])
    for key in ("primaries", "transfer", "matrix", "range", "bit_depth"):
        assert saved_probe[key] == probe[key]
    decoded = cast(np.ndarray, disassemble_video(saved)["images"])
    # One ten-bit quantization step, expressed in the full uint16 storage scale.
    np.testing.assert_allclose(decoded.astype(float), carried.astype(float), rtol=0, atol=65)
    explicit = cast(
        dict[str, Any], assemble_video(carried, color_space="sRGB", bit_depth="8")["probe"]
    )
    assert [explicit[key] for key in ("primaries", "transfer", "matrix", "range")] == [1, 13, 1, 1]
    assert explicit["bit_depth"] == 8


def test_unspecified_color_is_not_replaced_with_srgb() -> None:
    images = annotate_image(
        np.zeros((2, 32, 64, 3), dtype=np.uint8),
        color={"primaries": 2, "transfer": 2, "range": 2},
    )
    video = assemble_video(decode_image_array(encode_image_array(images)))
    probe = cast(dict[str, Any], video["probe"])
    assert probe["color_space"] == "unknown"
    assert [probe[key] for key in ("primaries", "transfer", "matrix")] == [2, 2, 2]
    target = io.BytesIO()
    save_video_stream(video, target, container="mp4", codec="h264", crf=0)
    saved = cast(dict[str, Any], video_from_source(target.getvalue())["probe"])
    assert saved["color_space"] == "unknown"
    assert [saved[key] for key in ("primaries", "transfer", "matrix")] == [2, 2, 2]


def test_source_depth_is_not_inferred_from_storage_width() -> None:
    images = np.zeros((1, 4, 4, 3), dtype=np.uint16)
    assert cast(dict[str, Any], assemble_video(images)["probe"])["bit_depth"] == 8
    tagged = annotate_image(
        images, color={"primaries": 1, "transfer": 13, "range": 2, "bit_depth": 12}
    )
    assert cast(dict[str, Any], assemble_video(tagged)["probe"])["bit_depth"] == 12


def test_rgb_source_transcode_tags_match_yuv_conversion() -> None:
    source = io.BytesIO()
    pixels = np.full((32, 64, 3), [128, 64, 32], dtype=np.uint8)
    with av.open(source, "w", format="matroska") as opened:
        mux = cast(Any, opened)
        stream = mux.add_stream("ffv1", rate=2)
        stream.width, stream.height, stream.pix_fmt = 64, 32, "bgr0"
        stream.codec_context.color_primaries = 1
        stream.codec_context.color_trc = 13
        stream.codec_context.colorspace = 0
        stream.codec_context.color_range = 2
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        for packet in stream.encode(frame):
            mux.mux(packet)
        for packet in stream.encode():
            mux.mux(packet)
    video = video_from_source(source.getvalue())
    assert cast(dict[str, Any], video["probe"])["matrix"] == 0
    target = io.BytesIO()
    save_video_stream(video, target, container="mp4", codec="h264", crf=0)
    saved = video_from_source(target.getvalue())
    assert cast(dict[str, Any], saved["probe"])["matrix"] == 1
    actual = cast(np.ndarray, disassemble_video(saved)["images"])
    np.testing.assert_allclose(actual[0].astype(float), pixels.astype(float), rtol=0, atol=2)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float16])
def test_component_wire_and_crop_keep_storage(dtype: Any) -> None:
    images = np.arange(2 * 8 * 16 * 3).reshape(2, 8, 16, 3).astype(dtype)
    value = assemble_video(images)
    decoded = decode_video(encode_video(value))
    actual = cast(np.ndarray, disassemble_video(decoded)["images"])
    assert actual.dtype == images.dtype
    np.testing.assert_array_equal(actual, images)
    cropped = edit_video(value, {"crop": {"x": 2, "y": 2, "width": 8, "height": 4}})
    actual = cast(np.ndarray, disassemble_video(cropped)["images"])
    assert actual.dtype == images.dtype
    np.testing.assert_array_equal(actual, images[:, 2:6, 2:10])


@pytest.mark.parametrize("kind", ["uint8", "uint16", "fp16", "bf16"])
def test_save_normalizes_frames_not_batches(kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    normalized = np.full((4, 32, 64, 3), 0.5, np.float32)
    if kind == "bf16":
        images = (
            (normalized.view(np.uint32) >> 16).astype(np.uint16).view([(BFLOAT16_FIELD, "<u2")])
        )
    elif kind == "fp16":
        images = normalized.astype(np.float16)
    else:
        dtype = np.dtype(kind)
        images = np.rint(normalized * np.iinfo(dtype).max).astype(dtype)
    before = images.copy()
    original = image_input
    calls: list[tuple[int, ...]] = []

    def per_frame(obj: object) -> object:
        array = cast(np.ndarray, obj)
        assert array.ndim == 3, "save expanded an entire frame batch"
        calls.append(array.shape)
        return original(obj)

    monkeypatch.setattr(runtime, "image_input", per_frame)
    output = io.BytesIO()
    save_video_stream(assemble_video(images, fps=4), output, crf=0)
    data = output.getvalue()
    if kind in ("fp16", "bf16"):
        assert len(calls) >= images.shape[0]
    pixels = cast(np.ndarray, image_input(disassemble_video(video_from_source(data))["images"]))
    np.testing.assert_allclose(pixels, 0.5, atol=2 / 255)
    np.testing.assert_array_equal(images, before)


def test_uint16_component_save_preserves_ten_bit_steps() -> None:
    images = np.full((2, 32, 64, 3), 32768, np.uint16)
    images[:, :, 32:] += 64
    output = io.BytesIO()
    save_video_stream(assemble_video(images, fps=2, bit_depth="10"), output, crf=0)
    data = output.getvalue()
    decoded = disassemble_video(video_from_source(data))
    pixels = cast(np.ndarray, decoded["images"])
    assert decoded["bit_depth"] == 10
    assert pixels.dtype == np.uint16
    assert pixels[:, :, 32:].mean() > pixels[:, :, :32].mean()
    assert np.unique(np.rint(pixels / 257)).size == 1


@pytest.mark.parametrize("depth", [8, 10])
def test_compact_alpha_and_padding_retain_levels(depth: int) -> None:
    dtype = np.uint8 if depth == 8 else np.uint16
    maximum = np.iinfo(dtype).max
    images = np.empty((2, 16, 32, 4), dtype)
    images[..., :3] = np.rint(np.array([0.25, 0.5, 0.75]) * maximum)
    images[..., 3] = np.rint(np.linspace(0, 1, 32) * maximum)
    for source in images:
        actual = runtime._pixels(runtime._frame(source, depth), depth, True)
        np.testing.assert_array_equal(actual, source)
    value = edit_video(
        assemble_video(images, bit_depth=str(depth)),
        {"scale": {"width": 32, "height": 32, "fit": "pad", "pad_color": [0.25, 0.5, 0.75, 1]}},
    )
    result = cast(np.ndarray, disassemble_video(value)["images"])
    assert result.dtype == dtype
    np.testing.assert_array_equal(result[:, 8:24], images)
    np.testing.assert_array_equal(
        result[0, 0, 0], np.rint(np.array([0.25, 0.5, 0.75, 1]) * maximum)
    )


@pytest.mark.parametrize("depth", [8, 10])
def test_decode_limits_charge_compact_array_bytes(
    depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _gray_source(depth)
    size = 2 * 32 * 64 * 3 * (1 if depth == 8 else 2)
    monkeypatch.setattr(media_video, "MAX_DECODED_FRAME_BYTES", size)
    assert _decode_legacy(data).nbytes == size
    monkeypatch.setattr(media_video, "MAX_DECODED_FRAME_BYTES", size - 1)
    with pytest.raises(ValueError, match="decoded video frames exceed"):
        _decode_legacy(data)


def test_pcm16_components_encode_like_normalized_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DINKSTER_PACK_SCRATCH", str(tmp_path))
    rate = 8000
    waveform = np.rint(np.sin(np.arange(rate) * 2 * np.pi * 440 / rate) * 8192).astype(np.int16)
    waveform = waveform[None, None, :]
    audio = {"waveform": waveform, "sample_rate": rate}
    images = np.full((2, 32, 64, 3), 128, np.uint8)
    actual = cast(
        dict[str, Any], disassemble_video(assemble_video(images, fps=2, audio=audio))["audio"]
    )
    pcm = cast(dict[str, Any], actual["source"])["pcm"]
    np.testing.assert_array_equal(pcm, waveform)
    assert pcm.dtype == np.int16
    results: list[np.ndarray] = []
    for samples in (waveform, waveform.astype(np.float32) / 32768):
        selected_audio = {"waveform": samples, "sample_rate": rate}
        output = io.BytesIO()
        save_video_stream(assemble_video(images, fps=2, audio=selected_audio), output, crf=0)
        data = output.getvalue()
        decoded = cast(
            dict[str, Any],
            disassemble_video(bind_video_value(video_from_source(data), for_audio_extraction=True))[
                "audio"
            ],
        )
        results.append(decoded["waveform"])
    np.testing.assert_array_equal(results[0], results[1])
