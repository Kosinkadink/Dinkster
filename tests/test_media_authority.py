"""Byte-derived upload media authority and immutable scoped grants."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import struct
import threading
import wave
import zlib
from collections.abc import AsyncIterator, Mapping
from contextlib import closing, suppress
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import av
import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import (
    AssetError,
    AssetRef,
    AssetVault,
    LibraryStore,
    MediaClassification,
    MediaGrant,
    classify_media,
    classify_media_file,
    classify_media_handle,
    digest_bytes,
)
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EventListener
from dinkster_nodes_media_io import LoadAudio
from dinkster_schema import Node, NodeSchema, build_node_types, build_schemas
from dinkster_server import Principal, ServerLibrary, create_app
from dinkster_server.auth import LOCAL_PRINCIPAL, PRINCIPAL_KEY
from dinkster_server.library import LIBRARY_KEY, handle_latent_upload, handle_media_upload
from dinkster_values import (
    TypeRegistry,
    audio_meta,
    register_core_types,
    render_audio_wav,
)
from dinkster_values.audio_codec import normalize_audio_waveform_request, render_audio_waveform
from dinkster_workers import InProcessWorker
from multidict import MultiDict, MultiDictProxy
from PIL import Image


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))


def _png() -> bytes:
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", b"")
        + _png_chunk(b"IEND", b"")
    )


def _jpeg() -> bytes:
    sof = b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    sos = b"\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00"
    return b"\xff\xd8" + sof + sos + b"\x00\xff\xd9"


def _riff(form: bytes, chunks: list[tuple[bytes, bytes]]) -> bytes:
    body = form
    for kind, payload in chunks:
        body += kind + struct.pack("<I", len(payload)) + payload
        if len(payload) % 2:
            body += b"\x00"
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _webp() -> bytes:
    frame = b"\x10\x00\x00\x9d\x01\x2a\x01\x00\x01\x00"
    return _riff(b"WEBP", [(b"VP8 ", frame)])


def _animated_webp() -> bytes:
    extended = b"\x02" + bytes(3) + bytes(6)
    frame = bytes(16) + _webp()[12:]
    return _riff(b"WEBP", [(b"VP8X", extended), (b"ANIM", bytes(6)), (b"ANMF", frame)])


def _wav() -> bytes:
    fmt = struct.pack("<HHIIHH", 1, 1, 8_000, 16_000, 2, 16)
    return _riff(b"WAVE", [(b"fmt ", fmt), (b"data", b"\x00\x00")])


def _flac_streaminfo() -> bytes:
    packed = (44_100 << 44) | (1 << 41) | (15 << 36) | 1
    return struct.pack(">HH", 4_096, 4_096) + bytes(6) + packed.to_bytes(8, "big") + bytes(16)


def _flac() -> bytes:
    streaminfo = _flac_streaminfo()
    frame_header = b"\xff\xf8\x89\x18\x00"
    crc = 0
    for byte in frame_header:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return b"fLaC" + b"\x80\x00\x00\x22" + streaminfo + frame_header + bytes([crc, 0])


def _mp3() -> bytes:
    # MPEG-1 Layer III, 128 kbps, 44.1 kHz, no padding: 417-byte frame.
    return b"\xff\xfb\x90\x00" + bytes(413)


def _ogg_page(payload: bytes, *, flags: int = 0x06) -> bytes:
    assert len(payload) < 255
    page = bytearray(
        b"OggS"
        + bytes([0, flags])
        + bytes(8)
        + struct.pack("<I", 1)
        + struct.pack("<I", 0)
        + bytes(4)
        + b"\x01"
        + bytes([len(payload)])
        + payload
    )
    crc = 0
    for byte in page:
        crc ^= byte << 24
        for _ in range(8):
            crc = (
                ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF
                if crc & 0x80000000
                else (crc << 1) & 0xFFFFFFFF
            )
    page[22:26] = struct.pack("<I", crc)
    return bytes(page)


def _ogg() -> bytes:
    return _ogg_page(b"OpusHead" + b"\x01\x01" + bytes(9))


def _ogg_speex() -> bytes:
    header = bytearray(80)
    header[:8] = b"Speex   "
    header[32:36] = (80).to_bytes(4, "little")
    header[36:40] = (16_000).to_bytes(4, "little")
    header[40:44] = (1).to_bytes(4, "little")
    header[48:52] = (1).to_bytes(4, "little")
    header[56:60] = (320).to_bytes(4, "little")
    header[64:68] = (1).to_bytes(4, "little")
    return _ogg_page(bytes(header))


def _ogg_flac() -> bytes:
    packet = b"\x7fFLAC\x01\x00\x00\x01fLaC" + b"\x00\x00\x00\x22" + _flac_streaminfo()
    return _ogg_page(packet)


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload) + 8) + kind + payload


def _mp4_track(handler: bytes, sample: bytes, *, minf: bytes | None = None) -> bytes:
    if minf is None:
        stsd = _box(b"stsd", bytes(4) + struct.pack(">I", 1) + sample)
        minf = _box(b"stbl", stsd)
    hdlr = _box(b"hdlr", bytes(8) + handler + bytes(12))
    mdia = _box(b"mdia", _box(b"mdhd", bytes(24)) + hdlr + _box(b"minf", minf))
    return _box(b"trak", _box(b"tkhd", bytes(84)) + mdia)


def _video_mp4_track() -> bytes:
    sample_payload = bytearray(78)
    sample_payload[6:8] = b"\x00\x01"
    sample_payload[24:28] = b"\x00\x01\x00\x01"
    sample = _box(b"avc1", bytes(sample_payload))
    return _mp4_track(b"vide", sample)


def _descriptor(kind: int, payload: bytes) -> bytes:
    assert len(payload) < 128
    return bytes([kind, len(payload)]) + payload


def _aac_esds() -> bytes:
    audio_specific_config = _descriptor(0x05, b"\x11\x88")
    decoder_config = _descriptor(0x04, b"\x40\x15" + bytes(11) + audio_specific_config)
    elementary_stream = _descriptor(0x03, b"\x00\x01\x00" + decoder_config)
    return _box(b"esds", bytes(4) + elementary_stream)


def _audio_mp4_track(sample_type: bytes = b"mp4a", *, descriptor: bytes | None = None) -> bytes:
    sample_payload = bytearray(28)
    sample_payload[6:8] = b"\x00\x01"
    sample_payload[16:20] = b"\x00\x01\x00\x10"
    sample_payload[24:28] = (48_000 << 16).to_bytes(4, "big")
    if descriptor is None:
        descriptor = _aac_esds()
    return _mp4_track(b"soun", _box(sample_type, bytes(sample_payload) + descriptor))


def _mp4(*tracks: bytes) -> bytes:
    ftyp = _box(b"ftyp", b"isom" + bytes(4) + b"isom")
    moov = _box(b"moov", _box(b"mvhd", bytes(100)) + b"".join(tracks or (_video_mp4_track(),)))
    return ftyp + moov + _box(b"mdat", b"")


def _vint(value: int) -> bytes:
    assert 0 <= value < 127
    return bytes([0x80 | value])


def _element(element_id: bytes, payload: bytes) -> bytes:
    return element_id + _vint(len(payload)) + payload


def _webm_container(track: bytes) -> bytes:
    ebml = _element(b"\x1aE\xdf\xa3", _element(b"B\x82", b"webm"))
    info = _element(
        b"\x15I\xa9f",
        _element(b"M\x80", b"dinkster") + _element(b"WA", b"dinkster"),
    )
    tracks = _element(b"\x16T\xaek", track)
    cluster = _element(b"\x1fC\xb6u", _element(b"\xe7", b"\x00"))
    segment = _element(b"\x18S\x80g", info + tracks + cluster)
    return ebml + segment


def _video_webm_track() -> bytes:
    video = _element(b"\xe0", _element(b"\xb0", b"\x01") + _element(b"\xba", b"\x01"))
    return _element(
        b"\xae",
        _element(b"\xd7", b"\x01")
        + _element(b"s\xc5", b"\x01")
        + _element(b"\x83", b"\x01")
        + _element(b"\x86", b"V_VP9")
        + video,
    )


def _audio_webm_track(*, number: int = 1) -> bytes:
    audio = _element(
        b"\xe1",
        _element(b"\xb5", struct.pack(">d", 48_000.0)) + _element(b"\x9f", b"\x01"),
    )
    return _element(
        b"\xae",
        _element(b"\xd7", bytes([number]))
        + _element(b"s\xc5", bytes([number]))
        + _element(b"\x83", b"\x02")
        + _element(b"\x86", b"A_OPUS")
        + audio,
    )


def _webm() -> bytes:
    return _webm_container(_video_webm_track())


def _audio_webm() -> bytes:
    return _webm_container(_audio_webm_track())


def _real_audio_webm() -> bytes:
    output = BytesIO()
    with av.open(output, "w", format="webm") as container:
        stream = container.add_stream("libopus", rate=48_000)
        stream.layout = "mono"
        stream.codec_context.thread_count = 1
        samples = np.arange(960, dtype=np.int16).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = 48_000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


def _media_recorder_audio_webm(codec: str) -> bytes:
    output = BytesIO()
    with av.open(output, "w", format="webm", options={"live": "1"}) as container:
        options = {"strict": "experimental"} if codec == "vorbis" else None
        stream = cast(av.AudioStream, container.add_stream(codec, rate=48_000, options=options))
        stream.layout = "stereo"
        stream.codec_context.thread_count = 1
        for index in range(5):
            samples = np.arange(index * 1920, (index + 1) * 1920, dtype=np.int16).reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="stereo")
            frame.sample_rate = 48_000
            frame.pts = index * 960
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


def _real_audio_m4a() -> bytes:
    output = BytesIO()
    with av.open(output, "w", format="ipod") as container:
        stream = container.add_stream("aac", rate=48_000)
        stream.layout = "mono"
        stream.codec_context.thread_count = 1
        samples = np.arange(1024, dtype=np.int16).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = 48_000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


def _glb_chunks(*chunks: tuple[bytes, bytes]) -> bytes:
    body = b"".join(struct.pack("<I4s", len(payload), kind) + payload for kind, payload in chunks)
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def _glb_json(payload: bytes) -> bytes:
    return _glb_chunks((b"JSON", payload + b" " * (-len(payload) % 4)))


def _glb(
    document: dict[str, object] | None = None,
    bin_payload: bytes | None = b"\x00\x00\x00\x00",
) -> bytes:
    if document is None:
        document = {"asset": {"version": "2.0"}}
        if bin_payload is not None:
            document["buffers"] = [{"byteLength": len(bin_payload)}]
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    chunks = [(b"JSON", encoded)]
    if bin_payload is not None:
        chunks.append((b"BIN\x00", bin_payload))
    return _glb_chunks(*chunks)


def _ply(
    header: bytes | None = None,
    vertices: int = 2,
    properties: tuple[tuple[bytes, bytes], ...] = (
        (b"float", b"x"),
        (b"float", b"y"),
        (b"float", b"z"),
    ),
    body: bytes | None = None,
) -> bytes:
    if header is None:
        header = (
            b"ply\nformat binary_little_endian 1.0\ncomment gaussian splat\n"
            + b"element vertex %d\n" % vertices
            + b"".join(b"property %s %s\n" % entry for entry in properties)
            + b"end_header\n"
        )
    if body is None:
        sizes = {b"float": 4, b"double": 8, b"uchar": 1, b"short": 2, b"int": 4}
        body = bytes(vertices * sum(sizes[kind] for kind, _ in properties))
    return header + body


def _latent(metadata: dict[str, str] | None = None) -> bytes:
    body = struct.pack("<f", 1.25)
    table: dict[str, object] = {
        "latent_tensor": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [0, len(body)],
        }
    }
    if metadata is not None:
        table["__metadata__"] = metadata
    header = json.dumps(table, separators=(",", ":")).encode("utf-8")
    header += b" " * (-len(header) % 8)
    return struct.pack("<Q", len(header)) + header + body


class _LatentContent:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.consumed = False

    def at_eof(self) -> bool:
        return self.consumed

    async def read(self, _size: int) -> bytes:
        self.consumed = True
        return self.body


class _LatentRequest:
    def __init__(self, library: ServerLibrary, name: str = "sample.latent") -> None:
        body = _latent()
        self.app = {LIBRARY_KEY: library}
        self.content_length = len(body)
        self.content = _LatentContent(body)
        self.query = MultiDictProxy(MultiDict({"scope": "local", "name": name}))
        self.headers = {"Content-Type": "application/x-comfy-latent"}

    def __getitem__(self, key: web.RequestKey[Principal]) -> Principal:
        assert key is PRINCIPAL_KEY
        return LOCAL_PRINCIPAL


MEDIA_CASES = (
    (_png(), MediaClassification("media/image", "image/png", "png")),
    (_jpeg(), MediaClassification("media/image", "image/jpeg", "jpg")),
    (_webp(), MediaClassification("media/image", "image/webp", "webp")),
    (_wav(), MediaClassification("media/audio", "audio/wav", "wav")),
    (_flac(), MediaClassification("media/audio", "audio/flac", "flac")),
    (_mp3(), MediaClassification("media/audio", "audio/mpeg", "mp3")),
    (_ogg(), MediaClassification("media/audio", "audio/ogg", "ogg")),
    (_audio_webm(), MediaClassification("media/audio", "audio/webm", "webm")),
    (_mp4(_audio_mp4_track()), MediaClassification("media/audio", "audio/mp4", "m4a")),
    (_mp4(), MediaClassification("media/video", "video/mp4", "mp4")),
    (_webm(), MediaClassification("media/video", "video/webm", "webm")),
    (_glb(), MediaClassification("media/model3d", "model/gltf-binary", "glb")),
    (_ply(), MediaClassification("media/model3d", "model/ply", "ply")),
)


@pytest.mark.parametrize(("data", "expected"), MEDIA_CASES)
def test_classify_media_accepts_closed_byte_derived_set(
    data: bytes, expected: MediaClassification
) -> None:
    assert classify_media(data) == expected


@pytest.mark.parametrize(
    ("image_format", "expected"),
    [
        ("PNG", MediaClassification("media/image", "image/png", "png")),
        ("JPEG", MediaClassification("media/image", "image/jpeg", "jpg")),
        ("WEBP", MediaClassification("media/image", "image/webp", "webp")),
    ],
)
def test_classify_media_accepts_real_image_encoder_output(
    image_format: str, expected: MediaClassification
) -> None:
    output = BytesIO()
    Image.new("RGB", (2, 2), (10, 20, 30)).save(output, format=image_format)
    assert classify_media(output.getvalue()) == expected


def test_classify_media_accepts_real_wave_writer_output() -> None:
    output = BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8_000)
        writer.writeframes(b"\x00\x00")
    assert classify_media(output.getvalue()) == MediaClassification(
        "media/audio", "audio/wav", "wav"
    )


def test_classify_media_accepts_structural_animated_webp() -> None:
    assert classify_media(_animated_webp()) == MediaClassification(
        "media/image", "image/webp", "webp"
    )


def test_classify_media_rejects_layer_two_and_handles_id3_versions() -> None:
    layer_two = b"\xff\xfd\x80\x00" + bytes(413)
    with pytest.raises(AssetError):
        classify_media(layer_two)
    id3v23_experimental = b"ID3\x03\x00\x20" + bytes(4) + _mp3()
    assert classify_media(id3v23_experimental).media_type == "audio/mpeg"
    id3v24_header = b"ID3\x04\x00\x10" + bytes(4)
    id3v24_footer = id3v24_header + b"3DI" + id3v24_header[3:10] + _mp3()
    assert classify_media(id3v24_footer).media_type == "audio/mpeg"


def test_classify_media_rejects_invalid_internal_framing() -> None:
    flac = bytearray(_flac())
    flac[-2] ^= 1
    malformed_sos = _jpeg().replace(b"\xff\xda\x00\x08\x01", b"\xff\xda\x00\x08\x02")
    unknown_jpeg_component = _jpeg().replace(
        b"\xff\xda\x00\x08\x01\x01", b"\xff\xda\x00\x08\x01\x02"
    )
    wav = bytearray(_wav())
    struct.pack_into("<I", wav, 28, 123)
    bad_footer = b"ID3\x04\x00\x10" + bytes(14) + _mp3()
    for data in (
        bytes(flac),
        malformed_sos,
        unknown_jpeg_component,
        bytes(wav),
        bad_footer,
    ):
        with pytest.raises(AssetError):
            classify_media(data)


def test_classify_media_rejects_truncated_ogg_codec_id_and_unsupported_mp4_audio() -> None:
    for data in (_ogg_page(b"OpusHead"), _mp4(_audio_mp4_track(b"alac"))):
        with pytest.raises(AssetError):
            classify_media(data)


def test_classify_media_uses_mp4_track_types_and_prefers_video() -> None:
    assert classify_media(_mp4(_audio_mp4_track())) == MediaClassification(
        "media/audio", "audio/mp4", "m4a"
    )
    assert classify_media(_mp4(_audio_mp4_track(), _video_mp4_track())).media_type == "video/mp4"


@pytest.mark.parametrize(("start", "end"), [(6, 8), (16, 18), (18, 20), (24, 28)])
def test_classify_media_rejects_invalid_mp4_audio_sample_fields(start: int, end: int) -> None:
    track = _audio_mp4_track()
    sample_offset = track.index(b"mp4a") + 4
    damaged = track[: sample_offset + start] + bytes(end - start) + track[sample_offset + end :]
    with pytest.raises(AssetError):
        classify_media(_mp4(damaged))


@pytest.mark.parametrize(
    "descriptor",
    [
        b"",
        _aac_esds()[:-1],
        _aac_esds().replace(b"\x40\x15", b"\x20\x15"),
        _aac_esds().replace(b"\x11\x88", b"\x00\x00"),
    ],
)
def test_classify_media_requires_valid_mp4_aac_descriptor(descriptor: bytes) -> None:
    with pytest.raises(AssetError):
        classify_media(_mp4(_audio_mp4_track(descriptor=descriptor)))


@pytest.mark.parametrize(("count", "entry_count"), [(0, 0), (0, 1), (2, 1)])
def test_classify_media_rejects_mp4_stsd_entry_count_mismatch(count: int, entry_count: int) -> None:
    sample = _audio_mp4_track()
    sample_start = sample.index(b"mp4a") - 4
    sample_size = struct.unpack_from(">I", sample, sample_start)[0]
    entry = sample[sample_start : sample_start + sample_size]
    stsd = _box(b"stsd", bytes(4) + struct.pack(">I", count) + entry * entry_count)
    with pytest.raises(AssetError):
        classify_media(_mp4(_mp4_track(b"soun", b"", minf=_box(b"stbl", stsd))))


def test_classify_media_rejects_ambiguous_and_excessive_mp4_audio_tables() -> None:
    sample = _audio_mp4_track()
    stbl_start = sample.index(b"stbl") - 4
    stbl_size = struct.unpack_from(">I", sample, stbl_start)[0]
    stbl = sample[stbl_start : stbl_start + stbl_size]
    duplicate_stbl = _mp4_track(b"soun", b"", minf=stbl + stbl)
    excessive_minf = _mp4_track(b"soun", b"", minf=_box(b"free", b"") * 100_001)
    for track in (duplicate_stbl, excessive_minf):
        with pytest.raises(AssetError):
            classify_media(_mp4(track))


def test_classify_media_uses_webm_track_type_and_rejects_trackless_container() -> None:
    assert classify_media(_audio_webm()) == MediaClassification("media/audio", "audio/webm", "webm")
    assert classify_media(_webm()).media_type == "video/webm"
    assert (
        classify_media(
            _webm_container(_video_webm_track() + _audio_webm_track(number=2))
        ).media_type
        == "video/webm"
    )
    with pytest.raises(AssetError):
        classify_media(_webm_container(b""))


def test_classify_media_accepts_speex_and_flac_ogg_identification() -> None:
    for data in (_ogg_speex(), _ogg_flac()):
        assert classify_media(data) == MediaClassification("media/audio", "audio/ogg", "ogg")


def test_classify_media_rejects_truncated_mp4_boxes_and_zero_visual_dimensions() -> None:
    mp4 = _mp4()
    truncated_mvhd = mp4.replace(_box(b"mvhd", bytes(100)), _box(b"mvhd", bytes(20)))
    valid_sample = bytearray(78)
    valid_sample[6:8] = b"\x00\x01"
    valid_sample[24:28] = b"\x00\x01\x00\x01"
    zero_dimensions = mp4.replace(_box(b"avc1", bytes(valid_sample)), _box(b"avc1", bytes(78)))
    for data in (truncated_mvhd, zero_dimensions):
        with pytest.raises(AssetError):
            classify_media(data)


def test_classify_media_rejects_webp_order_and_webm_codec_near_misses() -> None:
    animated = _animated_webp()
    out_of_order = _riff(
        b"WEBP",
        [
            (b"VP8X", b"\x02" + bytes(9)),
            (b"ANMF", bytes(16) + _webp()[12:]),
            (b"ANIM", bytes(6)),
        ],
    )
    top_level_image = animated + b"ignored"
    matroska_codec = _webm().replace(b"V_VP9", b"V_AVC")
    for data in (out_of_order, top_level_image, matroska_codec):
        with pytest.raises(AssetError):
            classify_media(data)


def test_classify_media_rejects_extensible_wave_overstated_extension() -> None:
    fmt = bytearray(40)
    struct.pack_into("<HHIIHHH", fmt, 0, 0xFFFE, 1, 8_000, 16_000, 2, 16, 65_535)
    fmt[18:20] = (16).to_bytes(2, "little")
    fmt[24:40] = b"\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x008\x9bq"
    with pytest.raises(AssetError):
        classify_media(_riff(b"WAVE", [(b"fmt ", bytes(fmt)), (b"data", b"\x00\x00")]))


@pytest.mark.parametrize(("data", "expected"), MEDIA_CASES)
def test_classify_media_rejects_every_truncated_prefix(
    data: bytes, expected: MediaClassification
) -> None:
    del expected
    for length in range(len(data)):
        with pytest.raises(AssetError):
            classify_media(data[:length])


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"not media",
        _png() + _jpeg(),
        _webp().replace(b"\x9d\x01\x2a", b"\x9c\x01\x2a"),
        _wav()[:4] + struct.pack("<I", 1) + _wav()[8:],
        b"fLaC" + b"\x80\x00\x00\x21" + bytes(33) + b"\xff\xf8",
        _mp3() + b"foreign trailer",
        _ogg_page(b"\x80theora" + bytes(8)),
        _ogg_page(b"OpusHead" + b"\x10\x01" + bytes(9)),
        _ogg_page(b"OpusHead" + b"\x01\x01" + bytes(8) + b"\x01\x00\x00\x00"),
        _mp4().replace(b"vide", b"soun"),
        _audio_webm().replace(b"A_OPUS", b"A_FAKE"),
    ],
)
def test_classify_media_rejects_unknown_near_miss_and_ambiguous_bytes(
    data: bytes,
) -> None:
    with pytest.raises(AssetError):
        classify_media(data)


@pytest.mark.parametrize(
    "data",
    [
        struct.pack("<4sII", b"glTF", 1, 12),
        struct.pack("<4sII", b"glTF", 2, 12),
        _glb()[:8] + struct.pack("<I", len(_glb()) + 4) + _glb()[12:],
        struct.pack("<4sII", b"glTF", 2, 16) + bytes(4),
        struct.pack("<4sII", b"glTF", 2, 24) + struct.pack("<I4s", 16, b"JSON") + b"    ",
        _glb_chunks((b"BIN\x00", b"\x00\x00\x00\x00")),
        _glb_chunks((b"JSON", b'{"asset":{"version":"2.0"}}')),
        _glb_chunks((b"JSON", b"{}  "), (b"JSON", b"    ")),
        _glb_chunks((b"JSON", b"{}  "), (b"BIN\x00", b""), (b"BIN\x00", b"")),
        _glb_chunks((b"JSON", b"{}  "), (b"ABCD", b""), (b"BIN\x00", b"")),
        _glb_chunks((b"JSON", b"{}  ")),
        _glb_chunks((b"JSON", b"")),
        _glb_chunks((b"JSON", b"{{{ ")),
        _glb_chunks((b"JSON", b"\xff\xff\xff\xff")),
        _glb_chunks((b"JSON", b"[]  ")),
        _glb_json(b'{"asset":{"version":"2.0"},"x":NaN}'),
        _glb_json('{"asset":{"version":"2.\u0660"}}'.encode()),
        _glb_json('{"asset":{"version":"2.0"}}'.encode("utf-16-le")),
        _glb({"asset": {}}, bin_payload=None),
        _glb({"asset": {"version": "3.0"}}, bin_payload=None),
        _glb({"asset": {"version": "2.0"}}),
        _glb({"asset": {"version": "2.0"}, "buffers": [{"byteLength": 16}]}),
        _glb({"asset": {"version": "2.0"}, "buffers": [{"byteLength": 0}]}),
        _glb({"asset": {"version": "2.0"}, "buffers": [{"byteLength": 0}]}, bin_payload=b""),
        _glb({"asset": {"version": "2.0"}, "buffers": [{"byteLength": -1}]}, bin_payload=b""),
        _glb({"asset": {"version": "2.0"}, "buffers": [{"byteLength": True}]}),
        _glb({"asset": {"version": "2.0"}, "buffers": [{"byteLength": 3.5}]}),
        _glb({"asset": {"version": "2.0"}, "buffers": "no"}, bin_payload=None),
        _glb({"asset": {"version": "2.0"}, "buffers": [{"byteLength": 4}]}, bin_payload=None),
        _glb(
            {"asset": {"version": "2.0"}, "buffers": [{"byteLength": 1}]},
            bin_payload=b"\x00\x01\x01\x01",
        ),
    ],
)
def test_classify_media_rejects_malformed_glb(data: bytes) -> None:
    with pytest.raises(AssetError):
        classify_media(data)


@pytest.mark.parametrize(
    "data",
    [
        b"ply\n",
        _ply(header=b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n"),
        _ply()[:-1],
        _ply() + b"\x00",
        _ply(body=b""),
        _ply(properties=((b"float", b"x"), (b"float", b"y"))),
        _ply(
            header=b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
            b"property list uchar int vertex_indices\nend_header\n",
            body=b"",
        ),
        _ply(
            header=b"ply\nformat binary_little_endian 1.0\nelement face 1\n"
            b"property float x\nproperty float y\nproperty float z\nend_header\n",
            body=bytes(12),
        ),
        _ply(header=b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nend_header\n"),
        _ply(
            header=b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
            b"property float x\nproperty float y\nproperty float z\nend_header"
        ),
        _ply(
            header=b"ply\r\nformat binary_little_endian 1.0\r\nelement vertex 1\r\n"
            b"property double x\nproperty float y\nproperty float z\nend_header\r\n"
        ),
    ],
)
def test_classify_media_rejects_malformed_ply(data: bytes) -> None:
    with pytest.raises(AssetError):
        classify_media(data)


def test_classify_media_rejects_glb_json_depth_and_size_bombs() -> None:
    # Built inside the test: these payloads are too large for parametrize IDs.
    deeply_nested = _glb_json(b"[" * 60_000 + b"]" * 60_000)
    oversized = _glb_chunks((b"JSON", b" " * (64 * 1024 * 1024 + 4)))
    for data in (deeply_nested, oversized):
        with pytest.raises(AssetError):
            classify_media(data)


def test_classify_media_accepts_spec_conforming_glb_variants() -> None:
    expected = MediaClassification("media/model3d", "model/gltf-binary", "glb")
    # A BIN chunk may be padded with up to three zero bytes past the buffer.
    for declared in (1, 2, 3, 4):
        assert (
            classify_media(
                _glb({"asset": {"version": "2.0"}, "buffers": [{"byteLength": declared}]})
            )
            == expected
        )
    # An integral JSON number is a valid buffer byteLength.
    assert (
        classify_media(_glb({"asset": {"version": "2.0"}, "buffers": [{"byteLength": 4.0}]}))
        == expected
    )
    # Chunks with unknown types after JSON and BIN are ignored per the spec.
    trailing = _glb_chunks(
        (b"JSON", b'{"asset":{"version":"2.0"}} '),
        (b"ABCD", b"\xff\xff\xff\xff"),
    )
    assert classify_media(trailing) == expected
    document = json.dumps(
        {"asset": {"version": "2.0"}, "buffers": [{"byteLength": 4}]},
        separators=(",", ":"),
    ).encode()
    document += b" " * (-len(document) % 4)
    with_unknown = _glb_chunks((b"JSON", document), (b"BIN\x00", bytes(4)), (b"ABCD", b""))
    assert classify_media(with_unknown) == expected
    assert classify_media(_glb(bin_payload=None)) == expected


def test_classify_media_file_failure_releases_mapping_for_deletion(tmp_path: Path) -> None:
    corrupted = bytearray(_png())
    corrupted[29] ^= 0x01  # flip one IHDR CRC byte
    path = tmp_path / "corrupted.png"
    path.write_bytes(bytes(corrupted))
    with pytest.raises(AssetError) as file_error:
        classify_media_file(path)
    assert file_error.value.__context__ is None
    with path.open("rb") as handle:
        with pytest.raises(AssetError) as handle_error:
            classify_media_handle(handle)
    assert handle_error.value.__context__ is None
    path.unlink()


def test_classification_is_immutable_and_has_no_metadata_authority() -> None:
    result = classify_media(_png())
    with pytest.raises(AttributeError):
        result.media_type = "video/mp4"  # type: ignore[misc]
    with pytest.raises(TypeError):
        classify_media(_png(), media_type="video/mp4")  # type: ignore[call-arg]
    with pytest.raises((AssetError, TypeError)):
        classify_media({"bytes": _png(), "name": "lie.mp4"})  # type: ignore[arg-type]


def _grant(scope: str = "alice", data: bytes = _png()) -> MediaGrant:
    facts = classify_media(data)
    return MediaGrant(
        scope=scope,
        digest=digest_bytes(data),
        kind=facts.kind,
        media_type=facts.media_type,
        extension=facts.extension,
        byte_size=len(data),
    )


def test_media_grant_is_immutable_validated_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "library.sqlite"
    grant = _grant()
    with closing(LibraryStore(path)) as store:
        assert store.grant_media(grant, _png()) == grant
        assert store.media_grant(grant.scope, grant.digest, grant.kind) == grant
        assert store.media_grant("bob", grant.digest, grant.kind) is None
        with pytest.raises(TypeError):
            store.media_grant(grant.scope, grant.digest)  # type: ignore[call-arg]
    with closing(LibraryStore(path)) as reopened:
        assert reopened.media_grant(grant.scope, grant.digest, grant.kind) == grant
    with pytest.raises(AttributeError):
        grant.byte_size = 0  # type: ignore[misc]


def test_media_grant_exact_regrant_is_idempotent_and_conflicts_refuse(
    tmp_path: Path,
) -> None:
    grant = _grant()
    with closing(LibraryStore(tmp_path / "library.sqlite")) as store:
        assert store.grant_media(grant, _png()) == grant
        assert store.grant_media(grant, _png()) == grant
        for replacement in (
            MediaGrant(
                **{
                    **grant.__dict__,
                    "media_type": "image/webp",
                    "extension": "webp",
                }
            ),
            MediaGrant(**{**grant.__dict__, "byte_size": grant.byte_size + 1}),
        ):
            with pytest.raises(AssetError, match="do not match supplied bytes"):
                store.grant_media(replacement, _png())
        assert store.media_grant(grant.scope, grant.digest, grant.kind) == grant


def test_media_grant_constructor_closes_facts_and_scope() -> None:
    grant = _grant()
    invalid = (
        {**grant.__dict__, "scope": " "},
        {**grant.__dict__, "digest": "not-a-digest"},
        {**grant.__dict__, "kind": "media/video"},
        {**grant.__dict__, "media_type": "text/plain"},
        {**grant.__dict__, "extension": "jpeg"},
        {**grant.__dict__, "byte_size": -1},
        {**grant.__dict__, "byte_size": True},
    )
    for fields in invalid:
        with pytest.raises(AssetError):
            MediaGrant(**fields)


def test_media_grant_concurrent_exact_and_conflicting_writes_are_atomic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "library.sqlite"
    grant = _grant()
    conflict = MediaGrant(**{**grant.__dict__, "byte_size": grant.byte_size + 1})
    barrier = threading.Barrier(8)
    results: list[MediaGrant] = []
    errors: list[Exception] = []
    stores = [LibraryStore(path) for _ in range(8)]

    def write(store: LibraryStore, candidate: MediaGrant) -> None:
        try:
            barrier.wait()
            results.append(store.grant_media(candidate, _png()))
        except Exception as error:  # noqa: BLE001 - evidence captures competing result
            errors.append(error)
        finally:
            store.close()

    threads = [
        threading.Thread(
            target=write,
            args=(store, grant if index < 4 else conflict),
        )
        for index, store in enumerate(stores)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    with closing(LibraryStore(path)) as reopened:
        stored = reopened.media_grant(grant.scope, grant.digest, grant.kind)
    assert stored == grant
    assert results and all(result == grant for result in results)
    assert errors and all(isinstance(error, AssetError) for error in errors)


class _Noop(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(node_type="test.media_noop", inputs=(), outputs=())

    @classmethod
    async def execute(cls) -> Mapping[str, object]:
        return cls.outputs()


_MEDIA_SCHEMAS = build_schemas((_Noop,))


def _media_engine(on_event: EventListener | None = None) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    return Engine(
        schemas=_MEDIA_SCHEMAS,
        registry=registry,
        worker=InProcessWorker(build_node_types((_Noop,)), registry),
        cache=MemoryLRUCache(),
        on_event=on_event,
    )


class _Tokens:
    def __init__(self, principals: Mapping[str, Principal]) -> None:
        self.principals = dict(principals)

    async def authenticate(self, token: str) -> Principal | None:
        return self.principals.get(token)


def _uniform_media_limits(limit: int) -> dict[str, int]:
    return dict.fromkeys(("media/image", "media/audio", "media/video", "media/model3d"), limit)


async def _media_client(
    tmp_path: Path,
    *,
    upload_limit: int = 1024 * 1024,
    media_upload_limits: Mapping[str, int] | None = None,
    authenticator: _Tokens | None = None,
) -> tuple[TestClient, ServerLibrary]:
    library = ServerLibrary(
        vault=AssetVault(tmp_path / "vault"),
        store=LibraryStore(tmp_path / "library.sqlite"),
        upload_limit=upload_limit,
        media_upload_limits=(
            _uniform_media_limits(upload_limit)
            if media_upload_limits is None
            else media_upload_limits
        ),
    )
    app = create_app(
        _media_engine,
        _MEDIA_SCHEMAS,
        library=library,
        authenticator=authenticator,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client, library


def _large_wav(byte_count: int) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(44_100)
        writer.writeframes(bytes(byte_count // 2 * 2))
    return output.getvalue()


def test_streaming_media_bypasses_aiohttp_buffer_limit_while_buffered_routes_remain_bounded(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        generic_limit = 16 * 1024 * 1024
        client, _library = await _media_client(
            tmp_path,
            upload_limit=generic_limit,
            media_upload_limits=_uniform_media_limits(32 * 1024 * 1024),
        )
        try:
            media = _large_wav(2 * 1024 * 1024)
            uploaded = await client.post(
                _media_path("local", "media/audio", "large.wav"),
                data=media,
                headers=_media_headers("audio/wav"),
            )
            assert uploaded.status == 201, await uploaded.text()
            generic = await client.post("/api/assets", data=bytes(generic_limit + 1))
            assert generic.status == 413
            oversized_json = await client.post(
                "/api/library",
                data=b'{"padding":"' + bytes(2 * 1024 * 1024) + b'"}',
                headers={"Content-Type": "application/json"},
            )
            assert oversized_json.status == 413
        finally:
            await client.close()

    asyncio.run(scenario())


def test_large_media_classification_does_not_block_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_server import library as library_module

    async def scenario() -> None:
        media = _large_wav(2 * 1024 * 1024)
        client, _library = await _media_client(tmp_path, upload_limit=16 * 1024 * 1024)
        entered = threading.Event()
        release = threading.Event()
        real_classify = classify_media_file

        def blocked_classify(path: Path) -> MediaClassification:
            entered.set()
            assert release.wait(timeout=5)
            return real_classify(path)

        monkeypatch.setattr(library_module, "classify_media_file", blocked_classify)
        upload = asyncio.create_task(
            client.post(
                _media_path("local", "media/audio", "large.wav"),
                data=media,
                headers=_media_headers("audio/wav"),
            )
        )
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            health = await asyncio.wait_for(client.get("/api/health"), timeout=1)
            assert health.status == 200
            release.set()
            response = await asyncio.wait_for(upload, timeout=5)
            assert response.status == 201
        finally:
            release.set()
            if not upload.done():
                upload.cancel()
            await client.close()

    asyncio.run(scenario())


def test_media_upload_spool_write_does_not_block_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_server import library as library_module

    async def scenario() -> None:
        client, _library = await _media_client(tmp_path)
        entered = threading.Event()
        release = threading.Event()
        real_spool = library_module._spool_media_upload_chunk

        def blocked_spool(temporary: object, hasher: object, chunk: bytes) -> None:
            entered.set()
            assert release.wait(timeout=5)
            real_spool(temporary, hasher, chunk)

        monkeypatch.setattr(library_module, "_spool_media_upload_chunk", blocked_spool)
        upload = asyncio.create_task(
            client.post(
                _media_path("local", "media/image", "concurrent.png"),
                data=_png(),
                headers=_media_headers("image/png"),
            )
        )
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            health = await asyncio.wait_for(client.get("/api/health"), timeout=1)
            assert health.status == 200
            release.set()
            response = await asyncio.wait_for(upload, timeout=5)
            assert response.status == 201
        finally:
            release.set()
            if not upload.done():
                upload.cancel()
                with suppress(asyncio.CancelledError):
                    await upload
            await client.close()

    asyncio.run(scenario())


def test_media_upload_repeated_cancellation_waits_for_spool_write_before_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_server import library as library_module

    class Content:
        async def iter_chunked(self, _size: int) -> AsyncIterator[bytes]:
            yield _png()

    class Request:
        def __init__(self, library: ServerLibrary) -> None:
            self.app = {LIBRARY_KEY: library}
            self.content_length = len(_png())
            self.content = Content()
            self.query = MultiDictProxy(
                MultiDict({"scope": "local", "kind": "media/image", "name": "cancel.png"})
            )
            self.headers = {"Content-Type": "image/png"}

        def __getitem__(self, key: web.RequestKey[Principal]) -> Principal:
            assert key is PRINCIPAL_KEY
            return LOCAL_PRINCIPAL

    async def scenario() -> None:
        library = ServerLibrary(
            vault=AssetVault(tmp_path / "vault"),
            store=LibraryStore(tmp_path / "library.sqlite"),
        )
        entered = threading.Event()
        release = threading.Event()
        real_spool = library_module._spool_media_upload_chunk

        def blocked_spool(temporary: object, hasher: object, chunk: bytes) -> None:
            entered.set()
            assert release.wait(timeout=5)
            real_spool(temporary, hasher, chunk)

        monkeypatch.setattr(library_module, "_spool_media_upload_chunk", blocked_spool)
        task = asyncio.create_task(handle_media_upload(cast(web.Request, Request(library))))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert len(list(library.vault.root.glob(".media-upload-*"))) == 1
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert list(library.vault.root.glob(".media-upload-*")) == []
            assert library.vault.digests() == []
        finally:
            release.set()
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            library.store.close()

    asyncio.run(scenario())


def _media_path(scope: str, kind: str, name: str) -> str:
    return f"/api/assets/media?scope={scope}&kind={kind}&name={name}"


def _media_headers(media_type: str, token: str | None = None) -> dict[str, str]:
    headers = {"Content-Type": media_type}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def test_media_ingest_auth_scope_idempotence_and_canonical_response(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        auth = _Tokens(
            {
                "none": Principal("none", {"a": frozenset()}),
                "a": Principal("alice", {"a": frozenset({"assets:write"})}),
                "both": Principal(
                    "alice",
                    {
                        "a": frozenset({"assets:write"}),
                        "b": frozenset({"assets:write"}),
                    },
                ),
            }
        )
        client, library = await _media_client(tmp_path, authenticator=auth)
        path = _media_path("a", "media/image", "portrait.png")
        try:
            missing = await client.post(path, data=_png(), headers=_media_headers("image/png"))
            assert missing.status == 401
            assert await missing.json() == {
                "error": {
                    "code": "asset.media.authentication_required",
                    "message": "a valid Bearer credential is required",
                }
            }
            denied = await client.post(
                path, data=_png(), headers=_media_headers("image/png", "none")
            )
            assert denied.status == 403
            assert (await denied.json())["error"]["code"] == "asset.media.capability_required"
            wrong_scope = await client.post(
                _media_path("b", "media/image", "portrait.png"),
                data=_png(),
                headers=_media_headers("image/png", "a"),
            )
            assert wrong_scope.status == 403
            assert await wrong_scope.json() == {
                "error": {
                    "code": "asset.media.scope_forbidden",
                    "message": "scope b does not grant assets:write",
                }
            }

            generic_get = await client.get("/api/assets/media")
            assert generic_get.status == 401
            assert await generic_get.json() == {
                "error": "authentication-required",
                "message": "a valid Bearer credential is required",
            }
            generic_denied = await client.get(
                "/api/assets/media", headers={"Authorization": "Bearer none"}
            )
            assert generic_denied.status == 403
            assert await generic_denied.json() == {
                "error": "capability-required",
                "capability": "assets:read",
                "message": "this route requires assets:read",
            }

            response = await client.post(
                path, data=_png(), headers=_media_headers("IMAGE/PNG", "a")
            )
            digest = digest_bytes(_png())
            expected = {
                "asset": {
                    "digest": digest,
                    "name": "portrait.png",
                    "size": len(_png()),
                    "mediaType": "image/png",
                    "virtualPath": "",
                },
                "kind": "media/image",
            }
            assert response.status == 201
            assert await response.json() == expected
            repeated = await client.post(
                path, data=_png(), headers=_media_headers("image/png", "a")
            )
            assert repeated.status == 200
            assert await repeated.json() == expected
            cross_scope = await client.post(
                _media_path("b", "media/image", "other.png"),
                data=_png(),
                headers=_media_headers("image/png", "both"),
            )
            assert cross_scope.status == 201
            assert library.store.media_grant("a", digest, "media/image") is not None
            assert library.store.media_grant("b", digest, "media/image") is not None
        finally:
            await client.close()

    asyncio.run(scenario())


def test_latent_ingest_is_classified_scoped_and_idempotent(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, library = await _media_client(tmp_path)
        path = "/api/assets/latent?scope=local&name=sample.png"
        try:
            first = await client.post(
                path,
                data=_latent(),
                headers={"Content-Type": "application/octet-stream"},
            )
            digest = digest_bytes(_latent())
            assert first.status == 201, await first.text()
            assert await first.json() == {
                "asset": {
                    "digest": digest,
                    "name": "sample.png",
                    "size": len(_latent()),
                    "mediaType": "application/x-comfy-latent",
                    "virtualPath": "",
                },
                "kind": "data/latent",
            }
            repeated = await client.post(
                path,
                data=_latent(),
                headers={"Content-Type": "application/x-comfy-latent"},
            )
            assert repeated.status == 200
            assert library.vault.resolve(digest) is not None
            assert library.store.media_grant("local", digest, "data/latent") is not None
            metadata = await client.get(f"/api/assets/{digest}/metadata")
            assert metadata.status == 200
            assert (await metadata.json())["metadata"]["latent"] == {
                "profile": "comfyui-single",
                "streams": [{"name": "latent_tensor", "dtype": "F32", "shape": [1]}],
                "vaeHint": "",
            }
        finally:
            await client.close()

    asyncio.run(scenario())


def test_latent_ingest_refuses_malformed_and_cleans_startup_orphan(tmp_path: Path) -> None:
    async def scenario() -> None:
        vault = tmp_path / "vault"
        vault.mkdir()
        orphan = vault / ".latent-upload-orphan"
        orphan.write_bytes(b"partial")
        client, library = await _media_client(tmp_path)
        try:
            assert not orphan.exists()
            response = await client.post(
                "/api/assets/latent?scope=local&name=bad.latent",
                data=b"not safetensors",
                headers={"Content-Type": "application/x-comfy-latent"},
            )
            assert response.status == 415
            assert library.vault.digests() == []
            assert not list(vault.glob(".latent-upload-*"))
        finally:
            await client.close()

    asyncio.run(scenario())


def test_latent_ingest_normalizes_integer_digit_limit_to_invalid_input(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, library = await _media_client(tmp_path)
        raw = b'{"value":' + b"9" * 5000 + b"}"
        raw += b" " * (-len(raw) % 8)
        payload = struct.pack("<Q", len(raw)) + raw
        try:
            response = await client.post(
                "/api/assets/latent?scope=local&name=bad-number.latent",
                data=payload,
                headers={"Content-Type": "application/x-comfy-latent"},
            )
            assert response.status == 415
            assert (await response.json())["error"]["code"] == "asset.media.invalid_latent"
            assert library.vault.digests() == []
            assert library.latent_reserved_bytes == [0]
            assert not list(library.vault.root.glob(".latent-upload-*"))
        finally:
            await client.close()

    asyncio.run(scenario())


def test_latent_ingest_normalizes_bounded_nesting_to_invalid_input(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, library = await _media_client(tmp_path)
        raw = b"[" * 10000 + b"0" + b"]" * 10000
        raw += b" " * (-len(raw) % 8)
        payload = struct.pack("<Q", len(raw)) + raw
        try:
            response = await client.post(
                "/api/assets/latent?scope=local&name=bad-nesting.latent",
                data=payload,
                headers={"Content-Type": "application/x-comfy-latent"},
            )
            assert response.status == 415
            assert (await response.json())["error"]["code"] == "asset.media.invalid_latent"
            assert library.vault.digests() == []
            assert library.latent_reserved_bytes == [0]
            assert not list(library.vault.root.glob(".latent-upload-*"))
        finally:
            await client.close()

    asyncio.run(scenario())


def test_latent_metadata_probe_ignores_integer_limit_vae_hint(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, _library = await _media_client(tmp_path)
        hint = '{"version":' + "9" * 5000 + "}"
        payload = _latent({"dinkster_vae_hint": hint})
        digest = digest_bytes(payload)
        try:
            uploaded = await client.post(
                "/api/assets/latent?scope=local&name=hint.latent",
                data=payload,
                headers={"Content-Type": "application/x-comfy-latent"},
            )
            assert uploaded.status == 201
            metadata = await client.get(f"/api/assets/{digest}/metadata")
            assert metadata.status == 200
            assert (await metadata.json())["metadata"]["latent"]["vaeHint"] == ""
        finally:
            await client.close()

    asyncio.run(scenario())


def test_latent_metadata_probe_ignores_bounded_nesting_vae_hint(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, _library = await _media_client(tmp_path)
        hint = '{"version":' + "[" * 3900 + "1" + "]" * 3900 + "}"
        payload = _latent({"dinkster_vae_hint": hint})
        digest = digest_bytes(payload)
        try:
            uploaded = await client.post(
                "/api/assets/latent?scope=local&name=nested-hint.latent",
                data=payload,
                headers={"Content-Type": "application/x-comfy-latent"},
            )
            assert uploaded.status == 201
            metadata = await client.get(f"/api/assets/{digest}/metadata")
            assert metadata.status == 200
            assert (await metadata.json())["metadata"]["latent"]["vaeHint"] == ""
        finally:
            await client.close()

    asyncio.run(scenario())


def test_latent_ingest_requires_auth_scope_and_admission_capacity(tmp_path: Path) -> None:
    async def scenario() -> None:
        auth = _Tokens(
            {
                "none": Principal("none", {"a": frozenset()}),
                "a": Principal("alice", {"a": frozenset({"assets:write"})}),
            }
        )
        client, library = await _media_client(tmp_path, authenticator=auth)
        path = "/api/assets/latent?scope=a&name=sample.latent"
        try:
            missing = await client.post(path, data=_latent())
            assert missing.status == 401
            denied = await client.post(
                path,
                data=_latent(),
                headers={"Authorization": "Bearer none"},
            )
            assert denied.status == 403
            wrong_scope = await client.post(
                "/api/assets/latent?scope=b&name=sample.latent",
                data=_latent(),
                headers={"Authorization": "Bearer a"},
            )
            assert wrong_scope.status == 403

            await library.latent_ingest_slots.acquire()
            await library.latent_ingest_slots.acquire()
            try:
                busy = await client.post(
                    path,
                    data=_latent(),
                    headers={"Authorization": "Bearer a"},
                )
                assert busy.status == 429
                assert library.latent_reserved_bytes == [0]
            finally:
                library.latent_ingest_slots.release()
                library.latent_ingest_slots.release()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_latent_ingest_preserves_filesystem_headroom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "dinkster_server.library.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=2 * 1024 * 1024 * 1024 - 1),
    )

    async def scenario() -> None:
        client, library = await _media_client(tmp_path)
        try:
            response = await client.post(
                "/api/assets/latent?scope=local&name=sample.latent",
                data=_latent(),
            )
            assert response.status == 507
            assert library.latent_reserved_bytes == [0]
            assert library.vault.digests() == []
        finally:
            await client.close()

    asyncio.run(scenario())


def test_latent_spool_write_does_not_block_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_server import library as library_module

    async def scenario() -> None:
        client, _library = await _media_client(tmp_path)
        entered = threading.Event()
        release = threading.Event()
        real_spool = library_module._spool_media_upload_chunk

        def blocked_spool(temporary: object, hasher: object, chunk: bytes) -> None:
            entered.set()
            assert release.wait(timeout=5)
            real_spool(temporary, hasher, chunk)

        monkeypatch.setattr(library_module, "_spool_media_upload_chunk", blocked_spool)
        upload = asyncio.create_task(
            client.post(
                "/api/assets/latent?scope=local&name=responsive.latent",
                data=_latent(),
                headers={"Content-Type": "application/x-comfy-latent"},
            )
        )
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            health = await asyncio.wait_for(client.get("/api/health"), timeout=1)
            assert health.status == 200
            release.set()
            response = await asyncio.wait_for(upload, timeout=5)
            assert response.status == 201
        finally:
            release.set()
            if not upload.done():
                upload.cancel()
                with suppress(asyncio.CancelledError):
                    await upload
            await client.close()

    asyncio.run(scenario())


def test_latent_repeated_cancellation_waits_for_spool_owner_and_cleans_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_server import library as library_module

    async def scenario() -> None:
        library = ServerLibrary(
            vault=AssetVault(tmp_path / "vault"),
            store=LibraryStore(tmp_path / "library.sqlite"),
        )
        entered = threading.Event()
        release = threading.Event()
        real_spool = library_module._spool_media_upload_chunk

        def blocked_spool(temporary: object, hasher: object, chunk: bytes) -> None:
            entered.set()
            assert release.wait(timeout=5)
            real_spool(temporary, hasher, chunk)

        monkeypatch.setattr(library_module, "_spool_media_upload_chunk", blocked_spool)
        task = asyncio.create_task(handle_latent_upload(cast(web.Request, _LatentRequest(library))))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert library.latent_reserved_bytes == [1024 * 1024 * 1024]
            assert len(list(library.vault.root.glob(".latent-upload-*"))) == 1
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert library.latent_reserved_bytes == [0]
            assert list(library.vault.root.glob(".latent-upload-*")) == []
            assert library.vault.digests() == []
            assert (
                library.store.media_grant("local", digest_bytes(_latent()), "data/latent") is None
            )
            await asyncio.wait_for(library.latent_ingest_slots.acquire(), timeout=1)
            await asyncio.wait_for(library.latent_ingest_slots.acquire(), timeout=1)
            library.latent_ingest_slots.release()
            library.latent_ingest_slots.release()
        finally:
            release.set()
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            library.store.close()

    asyncio.run(scenario())


def test_latent_cancellation_before_publication_commit_leaves_nothing(tmp_path: Path) -> None:
    async def scenario() -> None:
        library = ServerLibrary(
            vault=AssetVault(tmp_path / "vault"),
            store=LibraryStore(tmp_path / "library.sqlite"),
        )
        await library.media_ingest_lock.acquire()
        task = asyncio.create_task(handle_latent_upload(cast(web.Request, _LatentRequest(library))))
        try:
            for _ in range(500):
                waiters = getattr(library.media_ingest_lock, "_waiters", None)
                if waiters:
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("latent upload did not wait for publication ownership")
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert library.latent_reserved_bytes == [0]
            assert list(library.vault.root.glob(".latent-upload-*")) == []
            assert library.vault.digests() == []
            assert (
                library.store.media_grant("local", digest_bytes(_latent()), "data/latent") is None
            )
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            library.media_ingest_lock.release()
            library.store.close()

    asyncio.run(scenario())


def test_latent_cancellation_after_commit_point_settles_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        library = ServerLibrary(
            vault=AssetVault(tmp_path / "vault"),
            store=LibraryStore(tmp_path / "library.sqlite"),
        )
        entered = threading.Event()
        release = threading.Event()
        original = library.store.grant_latent_result

        def blocked(grant: MediaGrant) -> tuple[MediaGrant, bool]:
            entered.set()
            assert release.wait(timeout=5)
            return original(grant)

        monkeypatch.setattr(library.store, "grant_latent_result", blocked)
        task = asyncio.create_task(handle_latent_upload(cast(web.Request, _LatentRequest(library))))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            digest = digest_bytes(_latent())
            assert library.vault.resolve(digest) is not None
            assert library.store.media_grant("local", digest, "data/latent") is not None
            assert library.latent_reserved_bytes == [0]
            assert list(library.vault.root.glob(".latent-upload-*")) == []
        finally:
            release.set()
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            library.store.close()

    asyncio.run(scenario())


def test_latent_grant_failure_rolls_back_new_bytes_and_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        library = ServerLibrary(
            vault=AssetVault(tmp_path / "vault"),
            store=LibraryStore(tmp_path / "library.sqlite"),
        )

        def fail(_grant: MediaGrant) -> tuple[MediaGrant, bool]:
            raise sqlite3.OperationalError("injected grant failure")

        monkeypatch.setattr(library.store, "grant_latent_result", fail)
        response = await handle_latent_upload(cast(web.Request, _LatentRequest(library)))
        assert response.status == 503
        digest = digest_bytes(_latent())
        assert library.vault.resolve(digest) is None
        assert library.store.media_grant("local", digest, "data/latent") is None
        assert library.latent_reserved_bytes == [0]
        assert list(library.vault.root.glob(".latent-upload-*")) == []
        await asyncio.wait_for(library.latent_ingest_slots.acquire(), timeout=1)
        await asyncio.wait_for(library.latent_ingest_slots.acquire(), timeout=1)
        library.latent_ingest_slots.release()
        library.latent_ingest_slots.release()
        library.store.close()

    asyncio.run(scenario())


def test_media_ingest_concurrent_same_scope_has_one_creator(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, _library = await _media_client(tmp_path)
        try:
            path = _media_path("local", "media/image", "same.png")

            async def upload() -> int:
                response = await client.post(path, data=_png(), headers=_media_headers("image/png"))
                return response.status

            assert sorted(await asyncio.gather(upload(), upload())) == [200, 201]
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(("data", "facts"), MEDIA_CASES)
def test_media_ingest_accepts_every_canonical_format(
    tmp_path: Path, data: bytes, facts: MediaClassification
) -> None:
    async def scenario() -> None:
        client, _library = await _media_client(tmp_path)
        try:
            response = await client.post(
                _media_path("local", facts.kind, f"sample.{facts.extension}"),
                data=data,
                headers=_media_headers(facts.media_type),
            )
            assert response.status == 201, await response.text()
            body = await response.json()
            assert body["kind"] == facts.kind
            assert body["asset"]["mediaType"] == facts.media_type
            assert body["asset"]["digest"] == digest_bytes(data)
            assert body["asset"]["size"] == len(data)
        finally:
            await client.close()

    asyncio.run(scenario())


def test_audio_webm_upload_load_and_rendition_use_canonical_asset(tmp_path: Path) -> None:
    async def scenario() -> None:
        payload = _real_audio_webm()
        client, library = await _media_client(tmp_path)
        try:
            mislabeled = await client.post(
                _media_path("local", "media/audio", "recording.webm"),
                data=payload,
                headers=_media_headers("video/webm"),
            )
            assert mislabeled.status == 409
            assert (await mislabeled.json())["error"]["code"] == "asset.media.content_type_mismatch"
            response = await client.post(
                _media_path("local", "media/audio", "recording.webm"),
                data=payload,
                headers=_media_headers("audio/webm;codecs=opus"),
            )
            assert response.status == 201, await response.text()
            body = await response.json()
            assert body["kind"] == "media/audio"
            assert body["asset"]["mediaType"] == "audio/webm"
            asset = AssetRef.from_wire(body["asset"], resolver=library.vault)
            audio = LoadAudio.execute(audio=asset)["audio"]
            assert audio_meta(audio)["frames"] == 1632
            assert audio_meta(audio)["duration"] == 0.034
            assert audio_meta(audio)["shape"] == (1, 1, 1632)
            rendition = render_audio_wav(audio)
            with wave.open(BytesIO(rendition), "rb") as decoded:
                assert decoded.getframerate() == 48_000
                assert decoded.getnchannels() == 1
                assert decoded.getnframes() > 0
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("codec", ["libopus", "vorbis"])
def test_media_recorder_audio_webm_publishes_length_and_waveform(
    tmp_path: Path, codec: str
) -> None:
    async def scenario() -> None:
        payload = _media_recorder_audio_webm(codec)
        expected_frames = 0
        with av.open(BytesIO(payload), mode="r") as container:
            stream = container.streams.audio[0]
            assert container.duration is None
            assert stream.duration is None
            expected_frames = sum(frame.samples for frame in container.decode(stream))

        client, library = await _media_client(tmp_path)
        try:
            response = await client.post(
                _media_path("local", "media/audio", "media-recorder.webm"),
                data=payload,
                headers=_media_headers("audio/webm"),
            )
            assert response.status == 201, await response.text()
            asset = AssetRef.from_wire((await response.json())["asset"], resolver=library.vault)
            audio = LoadAudio.execute(audio=asset)["audio"]
            metadata = audio_meta(audio)
            assert metadata["frames"] == expected_frames
            assert metadata["duration"] == expected_frames / 48_000
            assert metadata["shape"] == (1, 2, expected_frames)

            parameters = normalize_audio_waveform_request({"waveform": "32x16"}, metadata)
            with Image.open(BytesIO(render_audio_waveform(audio, parameters))) as image:
                assert image.format == "PNG"
                assert image.size == (32, 16)
        finally:
            await client.close()

    asyncio.run(scenario())


def test_audio_m4a_upload_load_and_rendition_use_canonical_asset(tmp_path: Path) -> None:
    async def scenario() -> None:
        payload = _real_audio_m4a()
        client, library = await _media_client(tmp_path)
        try:
            mislabeled = await client.post(
                _media_path("local", "media/audio", "recording.m4a"),
                data=payload,
                headers=_media_headers("video/mp4"),
            )
            assert mislabeled.status == 409
            assert (await mislabeled.json())["error"]["code"] == "asset.media.content_type_mismatch"
            response = await client.post(
                _media_path("local", "media/audio", "recording.m4a"),
                data=payload,
                headers=_media_headers("audio/mp4"),
            )
            assert response.status == 201, await response.text()
            body = await response.json()
            assert body["kind"] == "media/audio"
            assert body["asset"]["mediaType"] == "audio/mp4"
            asset = AssetRef.from_wire(body["asset"], resolver=library.vault)
            audio = LoadAudio.execute(audio=asset)["audio"]
            rendition = render_audio_wav(audio)
            with wave.open(BytesIO(rendition), "rb") as decoded:
                assert decoded.getframerate() == 48_000
                assert decoded.getnchannels() == 1
                assert decoded.getnframes() > 0
        finally:
            await client.close()

    asyncio.run(scenario())


def test_audio_m4a_upload_rejects_descriptor_free_track(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, library = await _media_client(tmp_path)
        try:
            response = await client.post(
                _media_path("local", "media/audio", "empty.m4a"),
                data=_mp4(_audio_mp4_track(descriptor=b"")),
                headers=_media_headers("audio/mp4"),
            )
            assert response.status == 415, await response.text()
            assert (await response.json())["error"]["code"] == "asset.media.unsupported_media"
            assert library.vault.digests() == []
        finally:
            await client.close()

    asyncio.run(scenario())


def test_media_ingest_validation_refuses_without_publication(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, library = await _media_client(tmp_path)
        good_path = _media_path("local", "media/image", "safe name.png")
        cases = (
            ("/api/assets/media?kind=media/image&name=x.png", _png(), "image/png", 400),
            (
                _media_path("local", "model/checkpoint", "x.png"),
                _png(),
                "image/png",
                400,
            ),
            (_media_path("local", "media/image", "../x.png"), _png(), "image/png", 400),
            (_media_path("local", "media/image", "x/y.png"), _png(), "image/png", 400),
            (_media_path("local", "media/image", "x\\y.png"), _png(), "image/png", 400),
            (_media_path("local", "media/image", "x" * 256), _png(), "image/png", 400),
            (good_path, b"", "image/png", 400),
            (good_path, b"not media", "image/png", 415),
            (good_path, _png(), "application/octet-stream", 415),
            (good_path, _png(), "video/mp4", 409),
            (_media_path("local", "media/video", "x.png"), _png(), "image/png", 409),
        )
        try:
            for path, data, media_type, status in cases:
                response = await client.post(path, data=data, headers=_media_headers(media_type))
                assert response.status == status, (path, await response.text())
                body = await response.json()
                assert set(body) == {"error"}
                assert body["error"]["code"].startswith("asset.media.")
            assert library.vault.digests() == []
            assert library.store.media_grant("local", digest_bytes(_png()), "media/image") is None
        finally:
            await client.close()

    asyncio.run(scenario())


def test_media_ingest_malformed_validator_failure_returns_415_and_cleans_spool(
    tmp_path: Path,
) -> None:
    corrupted = bytearray(_png())
    corrupted[29] ^= 0x01  # flip one IHDR CRC byte so the PNG validator itself fails

    async def scenario() -> None:
        client, library = await _media_client(tmp_path)
        try:
            response = await client.post(
                _media_path("local", "media/image", "corrupted.png"),
                data=bytes(corrupted),
                headers=_media_headers("image/png"),
            )
            assert response.status == 415, await response.text()
            body = await response.json()
            assert body["error"]["code"] == "asset.media.unsupported_media"
            assert list(library.vault.root.glob(".media-upload-*")) == []
            assert library.vault.digests() == []
        finally:
            await client.close()

    asyncio.run(scenario())


def test_media_ingest_digest_and_exact_byte_limits(tmp_path: Path) -> None:
    base = _jpeg()
    limit = len(base) + 2
    bodies = (
        base,
        base[:-2] + b"\x00" + base[-2:],
        base[:-2] + b"\x00\x00" + base[-2:],
        base[:-2] + b"\x00\x00\x00" + base[-2:],
    )
    assert [len(data) for data in bodies[1:]] == [limit - 1, limit, limit + 1]

    async def chunks(data: bytes) -> AsyncIterator[bytes]:
        yield data[:5]
        await asyncio.sleep(0)
        yield data[5:]

    async def scenario() -> None:
        for index, data in enumerate(bodies[1:]):
            case_root = tmp_path / str(index)
            client, library = await _media_client(case_root, upload_limit=limit)
            try:
                response = await client.post(
                    _media_path("local", "media/image", "x.jpg"),
                    data=data,
                    headers=_media_headers("image/jpeg"),
                )
                assert response.status == (201 if len(data) <= limit else 413)
                if len(data) > limit:
                    assert library.vault.digests() == []
            finally:
                await client.close()

        client, library = await _media_client(tmp_path / "chunked", upload_limit=limit)
        data = bodies[2]
        try:
            missing_length = await client.post(
                _media_path("local", "media/image", "x.jpg"),
                data=chunks(data),
                headers=_media_headers("image/jpeg"),
            )
            assert missing_length.status == 201
            malformed_digest = await client.post(
                _media_path("local", "media/image", "bad.jpg"),
                data=base,
                headers={**_media_headers("image/jpeg"), "X-Dinkster-Digest": "sha256:no"},
            )
            assert malformed_digest.status == 400
            mismatch = await client.post(
                _media_path("local", "media/image", "bad.jpg"),
                data=base,
                headers={
                    **_media_headers("image/jpeg"),
                    "X-Dinkster-Digest": digest_bytes(_png()),
                },
            )
            assert mismatch.status == 409
            assert library.vault.resolve(digest_bytes(base)) is None
        finally:
            await client.close()

    asyncio.run(scenario())


def test_default_media_limits_mirror_frontend_per_kind_bounds(tmp_path: Path) -> None:
    library = ServerLibrary(
        vault=AssetVault(tmp_path / "vault"),
        store=LibraryStore(tmp_path / "library.sqlite"),
    )
    try:
        assert dict(library.media_upload_limits) == {
            "media/image": 256 * 1024 * 1024,
            "media/audio": 1024 * 1024 * 1024,
            "media/video": 1024 * 1024 * 1024,
            "media/model3d": 1024 * 1024 * 1024,
        }
    finally:
        library.store.close()


def test_media_upload_limit_is_selected_by_claimed_kind(tmp_path: Path) -> None:
    async def scenario() -> None:
        image = _png()
        audio = _large_wav(4096)
        assert len(audio) > len(image)
        limits = {
            "media/image": len(image) - 1,
            "media/audio": 8 * 1024 * 1024,
            "media/video": 8 * 1024 * 1024,
        }
        client, library = await _media_client(tmp_path, media_upload_limits=limits)
        try:
            rejected = await client.post(
                _media_path("local", "media/image", "big.png"),
                data=image,
                headers=_media_headers("image/png"),
            )
            assert rejected.status == 413
            assert library.vault.digests() == []
            # A body larger than the image bound is accepted under the
            # audio bound: the claimed kind selects the limit.
            accepted = await client.post(
                _media_path("local", "media/audio", "long.wav"),
                data=audio,
                headers=_media_headers("audio/wav"),
            )
            assert accepted.status == 201, await accepted.text()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_default_image_bound_rejects_oversized_content_length_before_reading(
    tmp_path: Path,
) -> None:
    class UnreadBody:
        async def iter_chunked(self, _size: int) -> AsyncIterator[bytes]:
            raise AssertionError("the body must not be read past the length precheck")
            yield b""  # pragma: no cover - makes this an async generator

    class OversizedRequest:
        def __init__(self, library: ServerLibrary) -> None:
            self.app = {LIBRARY_KEY: library}
            self.content_length = 256 * 1024 * 1024 + 1
            self.content = UnreadBody()
            self.query = MultiDictProxy(
                MultiDict({"scope": "local", "kind": "media/image", "name": "big.png"})
            )
            self.headers = {"Content-Type": "image/png"}

        def __getitem__(self, key: web.RequestKey[Principal]) -> Principal:
            assert key is PRINCIPAL_KEY
            return LOCAL_PRINCIPAL

    async def scenario() -> None:
        library = ServerLibrary(
            vault=AssetVault(tmp_path / "vault"),
            store=LibraryStore(tmp_path / "library.sqlite"),
        )
        try:
            response = await handle_media_upload(cast(web.Request, OversizedRequest(library)))
            assert response.status == 413
            assert library.vault.digests() == []
        finally:
            library.store.close()

    asyncio.run(scenario())


def test_media_ingest_interruption_and_store_failure_are_atomic(tmp_path: Path) -> None:
    class BrokenContent:
        async def iter_chunked(self, _size: int) -> AsyncIterator[bytes]:
            yield _png()[:10]
            raise ConnectionResetError("client vanished from /private/host/path")

    class MalformedContent:
        async def iter_chunked(self, _size: int) -> AsyncIterator[bytes]:
            yield _png()[:10]
            raise web.RequestPayloadError("malformed body at /private/host/path")

    class BrokenRequest:
        def __init__(
            self,
            library: ServerLibrary,
            content: object,
            content_length: int | None,
        ) -> None:
            self.app = {LIBRARY_KEY: library}
            self.content_length = content_length
            self.content = content
            self.query = MultiDictProxy(
                MultiDict({"scope": "local", "kind": "media/image", "name": "x.png"})
            )
            self.headers = {"Content-Type": "image/png"}

        def __getitem__(self, key: web.RequestKey[Principal]) -> Principal:
            assert key is PRINCIPAL_KEY
            return LOCAL_PRINCIPAL

    class FixedContent:
        def __init__(self, data: bytes) -> None:
            self.data = data

        async def iter_chunked(self, _size: int) -> AsyncIterator[bytes]:
            yield self.data

    async def scenario() -> None:
        library = ServerLibrary(
            vault=AssetVault(tmp_path / "vault"),
            store=LibraryStore(tmp_path / "library.sqlite"),
        )
        request = cast(web.Request, BrokenRequest(library, BrokenContent(), 1))
        interrupted = await handle_media_upload(request)
        assert interrupted.status == 400
        assert isinstance(interrupted.body, bytes)
        assert b"private/host/path" not in interrupted.body
        assert library.vault.digests() == []

        malformed = await handle_media_upload(
            cast(web.Request, BrokenRequest(library, MalformedContent(), None))
        )
        assert malformed.status == 400
        assert malformed.body == interrupted.body
        assert library.vault.digests() == []

        limited = ServerLibrary(
            vault=AssetVault(tmp_path / "limited-vault"),
            store=LibraryStore(tmp_path / "limited.sqlite"),
            upload_limit=len(_png()) - 1,
            media_upload_limits=_uniform_media_limits(len(_png()) - 1),
        )
        misleading = cast(
            web.Request,
            BrokenRequest(limited, FixedContent(_png()), 1),
        )
        too_large = await handle_media_upload(misleading)
        assert too_large.status == 413
        assert limited.vault.digests() == []
        limited.store.close()

        original = library.store.grant_media_result

        def fail(_grant: MediaGrant, _data: bytes) -> tuple[MediaGrant, bool]:
            raise sqlite3.OperationalError("database at /private/host/path failed")

        library.store.grant_media_result = fail  # type: ignore[method-assign]
        client = TestClient(TestServer(create_app(_media_engine, _MEDIA_SCHEMAS, library=library)))
        await client.start_server()
        try:
            failed = await client.post(
                _media_path("local", "media/image", "x.png"),
                data=_png(),
                headers=_media_headers("image/png"),
            )
            assert failed.status == 503
            assert await failed.json() == {
                "error": {
                    "code": "asset.media.storage_unavailable",
                    "message": "media storage is unavailable",
                }
            }
            digest = digest_bytes(_png())
            assert library.vault.resolve(digest) is not None
            assert library.store.media_grant("local", digest, "media/image") is None
            generic = await client.post("/api/assets", data=_png())
            assert generic.status == 200
            assert await generic.json() == {"digest": digest}
        finally:
            library.store.grant_media_result = original  # type: ignore[method-assign]
            await client.close()

    asyncio.run(scenario())


def test_media_ingest_cancellation_retains_publication_serialization(
    tmp_path: Path,
) -> None:
    class Content:
        async def iter_chunked(self, _size: int) -> AsyncIterator[bytes]:
            yield _png()

    class Request:
        def __init__(self, library: ServerLibrary, name: str) -> None:
            self.app = {LIBRARY_KEY: library}
            self.content_length = len(_png())
            self.content = Content()
            self.query = MultiDictProxy(
                MultiDict({"scope": "local", "kind": "media/image", "name": name})
            )
            self.headers = {"Content-Type": "image/png"}

        def __getitem__(self, key: web.RequestKey[Principal]) -> Principal:
            assert key is PRINCIPAL_KEY
            return LOCAL_PRINCIPAL

    async def scenario() -> None:
        library = ServerLibrary(
            vault=AssetVault(tmp_path / "vault"),
            store=LibraryStore(tmp_path / "library.sqlite"),
        )
        entered = threading.Event()
        release = threading.Event()
        original = library.store.grant_media_result
        calls = 0

        def blocked(grant: MediaGrant, data: bytes) -> tuple[MediaGrant, bool]:
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                assert release.wait(timeout=5)
            return original(grant, data)

        library.store.grant_media_result = blocked  # type: ignore[method-assign]
        first = asyncio.create_task(
            handle_media_upload(cast(web.Request, Request(library, "first.png")))
        )
        await asyncio.to_thread(entered.wait)
        first.cancel()
        second = asyncio.create_task(
            handle_media_upload(cast(web.Request, Request(library, "second.png")))
        )
        await asyncio.sleep(0.05)
        assert not second.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        response = await asyncio.wait_for(second, timeout=5)
        assert response.status == 200
        assert calls == 2
        library.store.grant_media_result = original  # type: ignore[method-assign]
        library.store.close()

    asyncio.run(scenario())


def test_media_ingest_cancellation_before_publication_handoff_removes_spool(
    tmp_path: Path,
) -> None:
    class Content:
        async def iter_chunked(self, _size: int) -> AsyncIterator[bytes]:
            yield _png()

    class Request:
        def __init__(self, library: ServerLibrary) -> None:
            self.app = {LIBRARY_KEY: library}
            self.content_length = len(_png())
            self.content = Content()
            self.query = MultiDictProxy(
                MultiDict({"scope": "local", "kind": "media/image", "name": "waiting.png"})
            )
            self.headers = {"Content-Type": "image/png"}

        def __getitem__(self, key: web.RequestKey[Principal]) -> Principal:
            assert key is PRINCIPAL_KEY
            return LOCAL_PRINCIPAL

    async def scenario() -> None:
        library = ServerLibrary(
            vault=AssetVault(tmp_path / "vault"),
            store=LibraryStore(tmp_path / "library.sqlite"),
        )
        await library.media_ingest_lock.acquire()
        task = asyncio.create_task(handle_media_upload(cast(web.Request, Request(library))))
        try:
            for _ in range(500):
                waiters = getattr(library.media_ingest_lock, "_waiters", None)
                if waiters:
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("media upload did not wait for publication ownership")
            spools = list(library.vault.root.glob(".media-upload-*"))
            assert len(spools) == 1
            assert spools[0].read_bytes() == _png()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert list(library.vault.root.glob(".media-upload-*")) == []
            assert library.vault.digests() == []
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            library.media_ingest_lock.release()
            library.store.close()

    asyncio.run(scenario())


def test_generic_asset_upload_contract_is_unchanged(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, _library = await _media_client(tmp_path)
        try:
            data = b"opaque and intentionally not media"
            first = await client.post("/api/assets", data=data)
            assert first.status == 201
            assert await first.json() == {"digest": digest_bytes(data)}
            second = await client.post("/api/assets", data=data)
            assert second.status == 200
            assert await second.json() == {"digest": digest_bytes(data)}
        finally:
            await client.close()

    asyncio.run(scenario())
