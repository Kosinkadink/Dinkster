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

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import (
    AssetError,
    AssetVault,
    AssetWriter,
    LibraryStore,
    MediaClassification,
    MediaGrant,
    MountDef,
    MountTable,
    classify_media,
    classify_media_file,
    digest_bytes,
)
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EventListener
from dinkster_protocol import AssetFolderContext
from dinkster_schema import Node, NodeSchema, build_node_types, build_schemas
from dinkster_server import Principal, ServerLibrary, create_app
from dinkster_server.auth import LOCAL_PRINCIPAL, PRINCIPAL_KEY
from dinkster_server.library import LIBRARY_KEY, handle_latent_upload, handle_media_upload
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker
from multidict import MultiDict, MultiDictProxy
from PIL import Image

from tests.async_scenarios import run_scenario


@pytest.fixture(autouse=True)
def _sufficient_latent_disk_space(monkeypatch: pytest.MonkeyPatch) -> None:
    # Upload behavior must not depend on runner disk pressure. Reserve tests
    # override this probe to exercise insufficient headroom explicitly.
    monkeypatch.setattr(
        "dinkster_server.library.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=8 * 1024**3),
    )


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


def test_classify_media_uses_mp4_track_types_and_prefers_video() -> None:
    assert classify_media(_mp4(_audio_mp4_track())) == MediaClassification(
        "media/audio", "audio/mp4", "m4a"
    )
    assert classify_media(_mp4(_audio_mp4_track(), _video_mp4_track())).media_type == "video/mp4"


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
    mounts: MountTable | None = None,
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
        resolver=mounts,
        upload_writer=AssetWriter(mounts) if mounts is not None else None,
        project_folders_for=(
            (
                lambda project: AssetFolderContext(
                    ("imports", "project") if project == "film" else ()
                )
            )
            if mounts is not None
            else None
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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


def _media_path(scope: str, kind: str, name: str) -> str:
    return f"/api/assets/media?scope={scope}&kind={kind}&name={name}"


def _media_headers(media_type: str, token: str | None = None) -> dict[str, str]:
    headers = {"Content-Type": media_type}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return headers


@pytest.mark.parametrize(
    ("data", "kind", "media_type"),
    (
        (_png(), "media/image", "image/png"),
        (_mp4(_video_mp4_track()), "media/video", "video/mp4"),
        (_glb(), "media/model3d", "model/gltf-binary"),
    ),
)
def test_named_media_upload_uses_project_folder_without_vault_copy(
    tmp_path: Path, data: bytes, kind: str, media_type: str
) -> None:
    async def scenario() -> None:
        mounts = MountTable()
        for mount_id in ("imports", "project"):
            root = tmp_path / mount_id
            root.mkdir()
            mounts.add(MountDef(id=mount_id, path=root, mode="readwrite"))
            mounts.scan(mount_id)
        client, library = await _media_client(tmp_path, mounts=mounts)
        try:
            for destination, folder, expected in (
                (None, None, "mounts/imports/portrait.png"),
                (None, None, "mounts/imports/portrait_00001.png"),
                ("project", "plates", "mounts/project/plates/portrait.png"),
            ):
                params = {
                    "scope": "local",
                    "kind": kind,
                    "name": "portrait.png",
                    "project": "film",
                }
                if destination is not None:
                    params["mount"] = destination
                    assert folder is not None
                    params["folder"] = folder
                response = await client.post(
                    "/api/assets/media",
                    params=params,
                    data=data,
                    headers=_media_headers(media_type),
                )
                assert response.status == 201, await response.text()
                body = await response.json()
                assert body["asset"]["virtualPath"] == expected
                assert body["asset"]["digest"] == digest_bytes(data)
                assert body["asset"]["mediaType"] == media_type
                assert (tmp_path / expected.removeprefix("mounts/")).read_bytes() == data
                fetched = await client.get(f"/api/assets/{digest_bytes(data)}")
                assert fetched.status == 200
                assert await fetched.read() == data
            assert library.vault.digests() == []
            assert not list(library.vault.root.glob(".media-upload-*"))
            assert library.store.media_grant("local", digest_bytes(data), kind) is not None
            assert library.store.media_grant("other", digest_bytes(data), kind) is None
            response = await client.get(
                "/api/assets/catalog",
                params={"scope": "local", "project": "film", "kind": kind, "origin": "upload"},
            )
            assert response.status == 200
            catalog = await response.json()
            assert len(catalog["assets"]) == 1
            asset = catalog["assets"][0]
            assert asset["asset"]["digest"] == digest_bytes(data)
            assert asset["aliases"] == ["portrait.png", "portrait_00001.png"]
            assert [item["asset"]["virtualPath"] for item in asset["observations"]] == [
                "mounts/imports/portrait.png",
                "mounts/imports/portrait_00001.png",
                "mounts/project/plates/portrait.png",
            ]
            assert str(tmp_path) not in json.dumps(catalog)
        finally:
            await client.close()

    run_scenario(scenario())


@pytest.mark.parametrize(
    "condition", ("readonly", "offline", "revoked", "other-project", "traversal", "symlink")
)
@pytest.mark.parametrize("route", ("/api/assets/media", "/api/assets"))
def test_named_media_upload_refuses_destination_without_fallback(
    tmp_path: Path, condition: str, route: str
) -> None:
    async def scenario() -> None:
        root = tmp_path / "project"
        root.mkdir()
        mounts = MountTable()
        mounts.add(
            MountDef(
                id="project", path=root, mode="read" if condition == "readonly" else "readwrite"
            )
        )
        mounts.scan("project")
        if condition == "offline":
            root.rename(tmp_path / "disconnected")
        elif condition == "revoked":
            mounts.remove("project")
        elif condition == "symlink":
            from tests.platform_support import symlink_or_skip

            outside = tmp_path / "outside"
            outside.mkdir()
            symlink_or_skip(root / "plates", outside, target_is_directory=True)
        client, library = await _media_client(tmp_path, mounts=mounts)
        try:
            params = {
                "scope": "local",
                "name": "portrait.png",
                "project": "other" if condition == "other-project" else "film",
                "mount": "project",
                "folder": "../outside" if condition == "traversal" else "plates",
            }
            if route == "/api/assets/media":
                params["kind"] = "media/image"
            response = await client.post(
                route,
                params=params,
                data=_png(),
                headers=_media_headers("image/png"),
            )
            expected = {"other-project": 403, "traversal": 400}.get(condition, 503)
            assert response.status == expected, await response.text()
            assert str(tmp_path) not in await response.text()
            assert library.vault.digests() == []
            assert not list(library.vault.root.glob(".media-upload-*"))
            assert not (root / "plates" / "portrait.png").exists()
            assert not (tmp_path / "outside" / "portrait.png").exists()
            assert library.store.media_grant("local", digest_bytes(_png()), "media/image") is None
            if condition == "offline":
                assert not root.exists()
        finally:
            await client.close()

    run_scenario(scenario())


def test_media_ingest_auth_scope_idempotence_and_canonical_response(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        auth = _Tokens(
            {
                "none": Principal("none", {"a": frozenset()}),
                "a": Principal("alice", {"a": frozenset({"assets:read", "assets:write"})}),
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
            generic_wrong_scope = await client.post(
                "/api/assets",
                params={"scope": "b", "name": "model.safetensors"},
                data=b"unclassified model bytes",
                headers=_media_headers("application/octet-stream", "a"),
            )
            assert generic_wrong_scope.status == 403
            assert await generic_wrong_scope.json() == {
                "error": "scope b does not grant assets:write"
            }
            assert library.vault.digests() == []
            catalog_wrong_scope = await client.get(
                "/api/assets/catalog?scope=b", headers={"Authorization": "Bearer a"}
            )
            assert catalog_wrong_scope.status == 403
            availability_wrong_scope = await client.get(
                "/api/assets/availability",
                params={"scope": "b", "digest": digest_bytes(_png())},
                headers={"Authorization": "Bearer a"},
            )
            assert availability_wrong_scope.status == 403
            availability_read = await client.get(
                "/api/assets/availability",
                params={"scope": "a", "digest": digest_bytes(_png())},
                headers={"Authorization": "Bearer a"},
            )
            assert availability_read.status == 200
            assert await availability_read.json() == {
                "assets": [{"digest": digest_bytes(_png()), "status": "missing"}]
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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


@pytest.mark.parametrize(
    ("free_bytes", "expected_status"),
    [(2 * 1024**3 - 1, 507), (2 * 1024**3, 201)],
)
def test_latent_ingest_preserves_filesystem_headroom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, free_bytes: int, expected_status: int
) -> None:
    monkeypatch.setattr(
        "dinkster_server.library.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=free_bytes),
    )

    async def scenario() -> None:
        client, library = await _media_client(tmp_path)
        try:
            response = await client.post(
                "/api/assets/latent?scope=local&name=sample.latent",
                data=_latent(),
            )
            assert response.status == expected_status
            assert library.latent_reserved_bytes == [0]
            assert library.vault.digests() == (
                [] if expected_status == 507 else [digest_bytes(_latent())]
            )
        finally:
            await client.close()

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())
