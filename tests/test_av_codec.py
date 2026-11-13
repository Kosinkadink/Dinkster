"""Torch-free proofs for the AUDIO and VIDEO byte contracts."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from functools import cache

import numpy as np
import pytest
from dinkster_values import (
    decode_audio,
    decode_video,
    encode_audio,
    encode_video,
    render_audio_wav,
    render_video_original,
    validate_video_encoded,
    video_meta,
    video_rendition_mime,
)

from tests.test_video_probe import _encode


@dataclass
class FakeTensor:
    array: np.ndarray

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.array


@cache
def mp4_bytes() -> bytes:
    return _encode("mp4", "libx264", "yuv420p")


@cache
def webm_bytes() -> bytes:
    return _encode("webm", "libvpx-vp9", "yuv420p")


def test_audio_refuses_invalid_shape_rate_and_payload() -> None:
    with pytest.raises(ValueError, match="B, C, T"):
        encode_audio({"waveform": np.zeros((2, 3)), "sample_rate": 48_000})
    with pytest.raises(ValueError, match="positive uint64"):
        encode_audio({"waveform": np.zeros((1, 1, 1)), "sample_rate": 0})
    with pytest.raises(ValueError, match="shorter"):
        decode_audio(b"short")


def test_audio_wav_is_exact_pcm16_little_endian_and_sample_interleaved() -> None:
    waveform = np.array(
        [[[-2.0, -1.0, -0.5, 0.0], [0.5, 1.0, 2.0, 0.25]]],
        dtype=np.float32,
    )
    wav = render_audio_wav({"waveform": waveform, "sample_rate": 8_000})
    assert struct.unpack("<4sI4s4sIHHIIHH4sI", wav[:44]) == (
        b"RIFF",
        52,
        b"WAVE",
        b"fmt ",
        16,
        1,
        2,
        8_000,
        32_000,
        4,
        16,
        b"data",
        16,
    )
    assert struct.unpack("<8h", wav[44:]) == (
        -32768,
        16384,
        -32768,
        32767,
        -16384,
        32767,
        0,
        8192,
    )


@pytest.mark.parametrize(
    ("waveform", "sample_rate", "message"),
    [
        (np.zeros((0, 1, 1), dtype=np.float32), 1, "nonempty batch"),
        (np.zeros((1, 0, 1), dtype=np.float32), 1, "nonempty channels"),
        (np.zeros((1, 1, 0), dtype=np.float32), 1, "nonempty channels"),
        (np.array([[[np.nan]]], dtype=np.float32), 1, "finite samples"),
        (np.array([[[np.inf]]], dtype=np.float32), 1, "finite samples"),
        (np.zeros((1, 1, 1), dtype=np.float32), 1 << 32, "exceeds uint32"),
        (np.zeros((1, 2, 1), dtype=np.float32), 1 << 30, "byte rate exceeds uint32"),
    ],
)
def test_audio_wav_refuses_invalid_cardinality_samples_and_rate(
    waveform: np.ndarray, sample_rate: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        render_audio_wav({"waveform": waveform, "sample_rate": sample_rate})


@pytest.mark.parametrize(
    "container,data", [("mp4", mp4_bytes()), ("webm", webm_bytes())], ids=["mp4", "webm"]
)
def test_video_round_trip_preserves_raw_container_bytes(container: str, data: bytes) -> None:
    value = {"container": container, "bytes": data}
    encoded = encode_video(value)
    assert encoded.startswith(b"DINKSTER-VIDEO\x02")
    assert decode_video(encoded)["source"] == data
    assert decode_video(data)["source"] == data
    assert video_meta(value)["container"] == container
    assert video_meta(value)["byte_size"] == len(data)


def test_video_refuses_unknown_mismatched_and_unrecognized_containers() -> None:
    with pytest.raises(ValueError, match="does not match"):
        encode_video({"container": "mov", "bytes": mp4_bytes()})
    with pytest.raises(ValueError, match="does not match"):
        encode_video({"container": "webm", "bytes": mp4_bytes()})
    with pytest.raises(ValueError):
        decode_video(b"not a container")
    with pytest.raises(ValueError):
        decode_video(b"\x00\x00\x00\x18ftypqt  quicktime")


@pytest.mark.parametrize(
    ("container", "data", "mime"),
    [("mp4", mp4_bytes(), "video/mp4"), ("webm", webm_bytes(), "video/webm")],
    ids=["mp4", "webm"],
)
def test_video_original_rendition_preserves_validated_bytes_and_metadata_mime(
    container: str, data: bytes, mime: str
) -> None:
    value = {"container": container, "bytes": data}
    assert render_video_original(value) == data
    metadata = video_meta(value)
    assert video_rendition_mime(metadata) == mime
    validate_video_encoded(data, metadata)
    with pytest.raises(ValueError, match="metadata container"):
        video_rendition_mime({"container": "unsupported"})
