"""The shared image-array codec (dinkster_values.image_codec): npy bytes as
the cross-interpreter contract, PNG as the browser form.

What this proves: the bytes round-trip losslessly through numpy alone (the
torchless-engine half of the comfy.IMAGE story), torch tensors are accepted
by duck type without this interpreter ever importing torch, fingerprints
are runtime-form-independent (hazard H4: a tensor fingerprinted in a worker
keys the same cache entry as its numpy twin here), and the host-side compat
registration (register_comfy_host_types) declares and serves the PNG
rendition from encoded payloads - the /api/values path in miniature."""

from __future__ import annotations

import io
import struct
import threading
import warnings
import zlib
from dataclasses import dataclass
from itertools import product
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import dinkster_values.image_codec as image_codec_module
import numpy as np
import pytest
from dinkster_values import (
    PNG_CONTAINER_VERSION,
    EncodedPayload,
    TypeRegistry,
    Value,
    ValueMeta,
    decode_image_array,
    encode_canonical_png,
    encode_image_array,
    image_array_fingerprint,
    image_array_meta,
    prepare_image_array_encoding,
    render_image_png,
    render_mask_png,
    stable_hash,
)
from PIL import Image

from dinkster.comfy_compose import COMFY_IMAGE_TYPE, COMFY_MASK_TYPE, register_comfy_host_types


@dataclass
class FakeTensor:
    """A torch tensor's boundary-relevant surface (detach/cpu/numpy) with
    no torch import: the codec must accept tensors by duck type."""

    array: np.ndarray

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.array


@dataclass
class FakeBFloat16Tensor(FakeTensor):
    dtype: str = "torch.bfloat16"

    def float(self) -> FakeTensor:
        return FakeTensor(self.array.astype(np.float32))

    def numpy(self) -> np.ndarray:
        raise TypeError("Got unsupported ScalarType BFloat16")


def png_header(data: bytes) -> tuple[int, int, int]:
    """(width, height, color_type) from the IHDR chunk. Full chunk-grammar
    decoding lives in test_renditions.py against the same renderer."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data[12:16] == b"IHDR"
    width, height, depth, color_type = struct.unpack(">IIBB", data[16:26])
    assert depth == 8
    return width, height, color_type


def test_npy_round_trip_preserves_dtype_shape_values() -> None:
    for dtype in (np.float32, np.float64, np.uint8):
        array = (np.arange(24).reshape(2, 3, 4) / 24).astype(dtype)
        loaded = np.asarray(decode_image_array(encode_image_array(array)))
        assert loaded.dtype == array.dtype
        assert loaded.shape == array.shape
        assert np.array_equal(loaded, array)


@pytest.mark.parametrize(
    "array",
    [
        np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        np.asfortranarray(np.arange(24, dtype=np.float64).reshape(2, 3, 4)),
        np.arange(48, dtype=np.int16).reshape(4, 6, 2)[:, ::2],
        np.arange(12, dtype=np.float32).astype(">f4").reshape(2, 2, 3),
        np.empty((0, 3), dtype=np.uint8),
    ],
    ids=("c-order", "fortran-order", "non-contiguous", "big-endian", "empty"),
)
def test_direct_buffer_encoding_exactly_matches_npy_bytes(array: np.ndarray) -> None:
    expected = encode_image_array(array)
    encoding = prepare_image_array_encoding(array)
    target = bytearray(encoding.size)

    assert encoding.write(memoryview(target)) == encoding.size
    assert bytes(target) == expected


def test_direct_buffer_encoding_preserves_structured_dtype_padding() -> None:
    dtype = np.dtype([("a", "u1"), ("b", "<f8")], align=True)
    backing = bytearray([0xA5]) * (dtype.itemsize * 2)
    array = np.ndarray((2,), dtype=dtype, buffer=backing)
    array["a"] = [1, 2]
    array["b"] = [3.0, 4.0]
    expected = encode_image_array(array)
    encoding = prepare_image_array_encoding(array)
    target = bytearray(encoding.size)

    assert encoding.write(memoryview(target)) == encoding.size
    assert bytes(target) == expected


def test_direct_buffer_encoding_matches_dtype_metadata_warning() -> None:
    array = np.arange(3, dtype=np.dtype("f4", metadata={"unit": "meters"}))
    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter("always")
        expected = encode_image_array(array)
    with warnings.catch_warnings(record=True) as direct_warnings:
        warnings.simplefilter("always")
        encoding = prepare_image_array_encoding(array)
        target = bytearray(encoding.size)
        assert encoding.write(memoryview(target)) == encoding.size

    assert bytes(target) == expected
    assert [(warning.category, str(warning.message)) for warning in direct_warnings] == [
        (warning.category, str(warning.message)) for warning in expected_warnings
    ]


@pytest.mark.parametrize("metadata", [None, {"unit": "meters"}])
def test_direct_buffer_encoding_matches_npy_v3_bytes_and_warnings(
    metadata: dict[str, str] | None,
) -> None:
    fields = [("漢", "i4")]
    dtype = np.dtype(fields) if metadata is None else np.dtype(fields, metadata=metadata)
    array = np.array([(1,)], dtype=dtype)
    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter("always")
        expected = encode_image_array(array)
    with warnings.catch_warnings(record=True) as direct_warnings:
        warnings.simplefilter("always")
        encoding = prepare_image_array_encoding(array)
        target = bytearray(encoding.size)
        assert encoding.write(memoryview(target)) == encoding.size

    assert expected[:8] == b"\x93NUMPY\x03\x00"
    assert bytes(target) == expected
    assert [(warning.category, str(warning.message)) for warning in direct_warnings] == [
        (warning.category, str(warning.message)) for warning in expected_warnings
    ]


def test_direct_buffer_preflight_does_not_change_other_thread_warning_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_workers.boundary as boundary_module
    from dinkster_workers import ValueCodec

    array = np.arange(3, dtype=np.dtype("f4", metadata={"unit": "meters"}))
    preflight_entered = threading.Event()
    release_preflight = threading.Event()
    worker_errors: list[BaseException] = []
    original_dtype_to_descr = np.lib.format.dtype_to_descr
    worker: threading.Thread

    def blocking_dtype_to_descr(dtype: np.dtype[Any]) -> object:
        if threading.current_thread() is worker:
            preflight_entered.set()
            if not release_preflight.wait(5):
                raise TimeoutError("preflight was not released")
        return original_dtype_to_descr(dtype)

    def run_preflight() -> None:
        try:
            prepare_image_array_encoding(array)
        except BaseException as exc:
            worker_errors.append(exc)

    monkeypatch.setattr(np.lib.format, "dtype_to_descr", blocking_dtype_to_descr)
    registry = make_host_registry()
    codec = ValueCodec(registry, shm_threshold=1)
    names: list[str] = []
    create_empty_segment = boundary_module._create_empty_segment

    def record_segment(size: int) -> SharedMemory:
        segment = create_empty_segment(size)
        names.append(segment.name)
        return segment

    monkeypatch.setattr(boundary_module, "_create_empty_segment", record_segment)
    blobs: list[bytes] = []
    segments: list[SharedMemory] = []
    worker = threading.Thread(target=run_preflight)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        worker.start()
        assert preflight_entered.wait(5)
        try:
            with pytest.raises(UserWarning, match="metadata on a dtype"):
                codec.encode(registry.wrap(COMFY_IMAGE_TYPE, array), blobs, segments)
        finally:
            release_preflight.set()
            worker.join(5)

    assert not worker.is_alive()
    assert worker_errors == []
    assert blobs == []
    assert segments == []
    assert len(names) == 1
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=names[0])


def test_non_contiguous_arrays_encode() -> None:
    array = np.arange(16, dtype=np.float32).reshape(4, 4)[:, ::2]
    loaded = np.asarray(decode_image_array(encode_image_array(array)))
    assert np.array_equal(loaded, array)


def test_torch_shaped_objects_encode_by_duck_type() -> None:
    array = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(1, 2, 2, 3)
    tensor_bytes = encode_image_array(FakeTensor(array))
    assert tensor_bytes == encode_image_array(array)


def test_bfloat16_torch_shaped_objects_encode_as_float32() -> None:
    array = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(1, 2, 2, 3)
    loaded = np.asarray(decode_image_array(encode_image_array(FakeBFloat16Tensor(array))))
    assert loaded.dtype == np.float32
    assert np.array_equal(loaded, array)


def test_fingerprint_is_runtime_form_independent() -> None:
    array = np.ones((2, 2, 3), dtype=np.float32)
    fingerprint = image_array_fingerprint(COMFY_IMAGE_TYPE)
    assert fingerprint(array) == fingerprint(FakeTensor(array))
    # ... and contiguity-independent: same elements, same identity.
    wide = np.ones((2, 4, 3), dtype=np.float32)[:, ::2, :]
    assert fingerprint(wide) == fingerprint(np.ascontiguousarray(wide))


def test_fingerprint_preserves_raw_byte_identity() -> None:
    array = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3)
    expected = stable_hash(
        [
            COMFY_IMAGE_TYPE.encode("utf-8"),
            str(array.dtype).encode("utf-8"),
            repr(array.shape).encode("utf-8"),
            array.tobytes(),
        ]
    )
    assert image_array_fingerprint(COMFY_IMAGE_TYPE)(array) == expected


def test_fingerprint_hashes_array_through_a_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks: list[bytes | memoryview] = []

    def capture(parts: list[bytes | memoryview]) -> str:
        chunks.extend(parts)
        return "fingerprint"

    monkeypatch.setattr(image_codec_module, "stable_hash", capture)
    array = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3)

    assert image_array_fingerprint(COMFY_IMAGE_TYPE)(array) == "fingerprint"
    assert isinstance(chunks[-1], memoryview)
    assert bytes(chunks[-1]) == array.tobytes()


def test_fingerprint_separates_type_dtype_shape_content() -> None:
    array = np.zeros((2, 2), dtype=np.float32)
    fingerprint = image_array_fingerprint("comfy.IMAGE")
    assert fingerprint(array) != image_array_fingerprint("dev.image")(array)
    assert fingerprint(array) != fingerprint(array.astype(np.float64))
    assert fingerprint(array) != fingerprint(array.reshape(1, 4))
    assert fingerprint(array) != fingerprint(array + 1.0)


def test_meta_reports_shape_and_dtype() -> None:
    meta = image_array_meta(np.zeros((1, 4, 6, 3), dtype=np.float32))
    assert meta == {
        "shape": (1, 4, 6, 3),
        "dtype": "float32",
        "channels": {"layout": "rgb", "alpha": "none"},
        "color": {"primaries": 1, "transfer": 13, "range": 2},
    }


def test_render_batched_image_draws_first_element() -> None:
    batch = np.zeros((2, 3, 5, 3), dtype=np.float32)
    batch[0, :, :] = [1.0, 0.0, 0.0]  # first frame red, second black
    width, height, color_type = png_header(render_image_png(batch))
    assert (width, height, color_type) == (5, 3, 2)
    assert render_image_png(batch) == render_image_png(batch[0])


def test_canonical_png_crosses_stored_deflate_block_boundary() -> None:
    pixels = np.arange(17_000 * 4, dtype=np.uint8).reshape(1, 17_000, 4)
    encoded = encode_canonical_png(pixels.tobytes(), width=17_000, height=1, color_type=6)
    length = struct.unpack(">I", encoded[33:37])[0]
    assert encoded[37:41] == b"IDAT"
    compressed = encoded[41 : 41 + length]
    scanline = b"\0" + pixels.tobytes()
    assert compressed[:2] == b"\x78\x01"
    cursor = 2
    blocks: list[tuple[bool, bytes]] = []
    while True:
        header = compressed[cursor]
        cursor += 1
        assert header in (0, 1)  # BTYPE=00, with only BFINAL possibly set.
        block_length, complement = struct.unpack("<HH", compressed[cursor : cursor + 4])
        cursor += 4
        assert complement == block_length ^ 0xFFFF
        blocks.append((header == 1, compressed[cursor : cursor + block_length]))
        cursor += block_length
        if header == 1:
            break
    assert [(final, len(block)) for final, block in blocks] == [
        (False, 65_535),
        (True, len(scanline) - 65_535),
    ]
    assert b"".join(block for _final, block in blocks) == scanline
    assert compressed[cursor:] == struct.pack(">I", zlib.adler32(scanline))
    assert zlib.decompress(compressed) == scanline
    with Image.open(io.BytesIO(encoded)) as image:
        np.testing.assert_array_equal(np.asarray(image), pixels)


def test_render_accepts_tensor_shaped_objects() -> None:
    array = np.ones((1, 2, 2, 3), dtype=np.float32)
    assert render_image_png(FakeTensor(array)) == render_image_png(array)


def test_render_mask_accepts_batched_tensor_shaped_objects() -> None:
    batch = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 24
    rendered = render_mask_png(FakeTensor(batch))

    assert png_header(rendered) == (4, 3, 0)
    assert rendered == render_mask_png(batch[0])


def test_render_mask_rejects_unrenderable_shapes() -> None:
    with pytest.raises(ValueError, match="empty mask batch"):
        render_mask_png(np.zeros((0, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="cannot render mask shape"):
        render_mask_png(np.zeros((1, 2, 2, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="cannot render mask shape"):
        render_mask_png(np.zeros((3,), dtype=np.float32))


def test_render_rejects_unrenderable_shapes() -> None:
    with pytest.raises(ValueError, match="empty image batch"):
        render_image_png(np.zeros((0, 2, 2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="cannot render"):
        render_image_png(np.zeros((2, 2, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="cannot render"):
        render_image_png(np.zeros((3,), dtype=np.float32))
    with pytest.raises(ValueError, match="cannot render"):
        render_image_png(np.zeros((0, 0), dtype=np.float32))


def make_host_registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_comfy_host_types(registry)
    return registry


def test_comfy_host_types_declare_png_rendition() -> None:
    registry = make_host_registry()
    assert COMFY_IMAGE_TYPE in registry
    assert registry.spec(COMFY_IMAGE_TYPE).prepare_buffer_encoding is prepare_image_array_encoding
    specs = registry.renditions_of(COMFY_IMAGE_TYPE)
    assert [(s.kind, s.mime, s.default) for s in specs] == [("png", "image/png", True)]
    assert specs[0].version == PNG_CONTAINER_VERSION


def test_comfy_host_mask_shares_image_bytes_with_mask_rendition() -> None:
    """comfy.MASK gets the same image-array byte contract as comfy.IMAGE
    (one value type with dinkster.mask) but renders through the mask PNG."""
    registry = make_host_registry()
    assert COMFY_MASK_TYPE in registry
    assert registry.spec(COMFY_MASK_TYPE).prepare_buffer_encoding is prepare_image_array_encoding
    specs = registry.renditions_of(COMFY_MASK_TYPE)
    assert [(s.kind, s.mime, s.default) for s in specs] == [("png", "image/png", True)]
    mask = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 24
    rendition = registry.render(registry.wrap(COMFY_MASK_TYPE, mask), "png")
    assert rendition.mime == "image/png"
    assert rendition.data == render_mask_png(mask)


def test_comfy_image_direct_shm_matches_npy_bytes() -> None:
    from dinkster_workers import ValueCodec
    from dinkster_workers.boundary import release_segment

    registry = make_host_registry()
    array = np.arange(60, dtype=np.float32).reshape(1, 4, 5, 3)
    value = registry.wrap(COMFY_IMAGE_TYPE, array)
    blobs: list[bytes] = []
    segments = []
    wire, stat = ValueCodec(registry, shm_threshold=1).encode(value, blobs, segments)
    try:
        segment = segments[0]
        assert segment.buf is not None
        assert bytes(segment.buf[: stat.size_bytes]) == encode_image_array(array)
        assert wire["fingerprint"] == value.fingerprint
        assert stat.transport == "shm"
        assert stat.reused is False
        assert len(blobs) == 1
    finally:
        for segment in segments:
            release_segment(segment)


def test_large_comfy_image_shm_does_not_materialize_encoded_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    import dinkster_workers.boundary as boundary_module
    from dinkster_workers import ValueCodec
    from dinkster_workers.boundary import release_segment

    registry = make_host_registry()
    array = np.zeros((1, 1280, 1280, 3), dtype=np.float32)
    value = registry.wrap(COMFY_IMAGE_TYPE, array)

    def forbid_bytes_io(*_args: object, **_kwargs: object) -> object:
        pytest.fail("full npy BytesIO encoding was used")

    monkeypatch.setattr(image_codec_module, "io", types.SimpleNamespace(BytesIO=forbid_bytes_io))
    monkeypatch.setattr(
        boundary_module,
        "_create_segment",
        lambda _data: pytest.fail("copy-based shared-memory creation was used"),
    )
    blobs: list[bytes] = []
    segments = []
    wire, stat = ValueCodec(registry, shm_threshold=1).encode(value, blobs, segments)
    try:
        payload = wire["payload"]
        assert isinstance(payload, dict)
        assert payload["transport"] == "shm"
        assert stat.size_bytes > array.nbytes
        assert len(blobs) == 1
        assert segments[0].buf is not None
        assert bytes(segments[0].buf[:6]) == b"\x93NUMPY"
    finally:
        for segment in segments:
            release_segment(segment)


def test_comfy_host_types_registration_is_idempotent() -> None:
    registry = make_host_registry()
    register_comfy_host_types(registry)  # second spec / reload: no raise
    assert "dinkster.asset" in registry
    assert len(registry.renditions_of(COMFY_IMAGE_TYPE)) == 1


def test_comfy_host_base_asset_registration_is_independent_of_image() -> None:
    registry = TypeRegistry()
    registry.register(COMFY_IMAGE_TYPE)
    register_comfy_host_types(registry)
    register_comfy_host_types(registry)
    assert "dinkster.asset" in registry


def test_comfy_host_types_wrap_typed_image_asset_without_manual_base_registration() -> None:
    from dinkster_assets import AssetRef

    registry = make_host_registry()
    ref = AssetRef(digest="blake3:" + "a" * 64, name="input.png", size=3)
    wrapped = registry.wrap("asset<comfy.IMAGE>", ref)
    assert wrapped.type_id == "asset<comfy.IMAGE>"
    assert wrapped.fingerprint == ref.digest
    resolved = wrapped.resolve()
    assert isinstance(resolved, AssetRef)
    assert resolved.resolver is None


def test_encoded_comfy_image_renders_without_torch() -> None:
    """The /api/values path in miniature: a worker-encoded comfy.IMAGE
    arrives as npy bytes (EncodedPayload, exactly what the boundary builds)
    and the host renders PNG from numpy alone."""
    registry = make_host_registry()
    array = np.zeros((1, 2, 3, 3), dtype=np.float32)
    array[0, 0, 0] = [0.0, 1.0, 0.0]
    spec = registry.spec(COMFY_IMAGE_TYPE)
    value = Value(
        type_id=COMFY_IMAGE_TYPE,
        fingerprint=image_array_fingerprint(COMFY_IMAGE_TYPE)(array),
        meta=ValueMeta(dict(image_array_meta(array))),
        payload=EncodedPayload(COMFY_IMAGE_TYPE, spec.encode(array), spec.decode),
    )
    rendition = registry.render(value, "png")
    assert (rendition.kind, rendition.mime) == ("png", "image/png")
    assert png_header(rendition.data) == (3, 2, 2)


def test_comfy_host_types_declare_asset_providers() -> None:
    """Typed assets: the host registers the asset<comfy.IMAGE> decoder and
    the comfy.IMAGE batch merge under their canonical provider ids."""
    from dinkster_values import (
        IMAGE_BATCH_MERGER_ID,
        IMAGE_FILE_DECODER_ID,
    )

    registry = make_host_registry()
    decoder = registry.asset_decoder_for(COMFY_IMAGE_TYPE)
    assert IMAGE_FILE_DECODER_ID == "dinkster.image-file@3"
    assert decoder is not None and decoder.provider_id == IMAGE_FILE_DECODER_ID
    merger = registry.batch_merge_for(COMFY_IMAGE_TYPE)
    assert merger is not None and merger.provider_id == IMAGE_BATCH_MERGER_ID


class BytesAsset:
    """decode_image_file's duck-typed asset surface: open() over bytes."""

    def __init__(self, data: bytes, name: str = "img.png") -> None:
        self.data = data
        self.name = name

    def open(self) -> Any:
        import io

        return io.BytesIO(self.data)


def png_bytes(array: np.ndarray) -> bytes:
    """Encode [H, W, 3] uint8 -> PNG via Pillow (the decode-side library)."""
    import PIL.Image

    buffer = io.BytesIO()
    PIL.Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def pillow_profile_bytes() -> bytes:
    import PIL.ImageCms

    return PIL.ImageCms.ImageCmsProfile(PIL.ImageCms.createProfile("sRGB")).tobytes()


def icc_tag_records(profile: bytes | bytearray) -> dict[bytes, tuple[int, int, int]]:
    count = struct.unpack_from(">I", profile, 128)[0]
    records: dict[bytes, tuple[int, int, int]] = {}
    for index in range(count):
        record_offset = 132 + index * 12
        signature, data_offset, size = struct.unpack_from(">4sII", profile, record_offset)
        records[signature] = (record_offset, data_offset, size)
    return records


def swapped_red_green_profile_bytes() -> bytes:
    """Build a non-sRGB RGB profile from Pillow's generated sRGB profile."""
    profile = bytearray(pillow_profile_bytes())
    records = icc_tag_records(profile)
    red_record = records[b"rXYZ"][0]
    green_record = records[b"gXYZ"][0]
    red_location = bytes(profile[red_record + 4 : red_record + 12])
    green_location = bytes(profile[green_record + 4 : green_record + 12])
    profile[red_record + 4 : red_record + 12] = green_location
    profile[green_record + 4 : green_record + 12] = red_location
    return bytes(profile)


def cmyk_profile_bytes() -> bytes:
    """Build a small CMYK input profile with an 8-bit conversion LUT."""
    base = pillow_profile_bytes()
    records = icc_tag_records(base)
    header = bytearray(base[:128])
    header[12:16] = b"scnr"
    header[16:20] = b"CMYK"
    header[20:24] = b"XYZ "
    header[84:100] = bytes(16)

    fixed_identity = b"".join(
        struct.pack(">i", round(value * 65536)) for value in (1, 0, 0, 0, 1, 0, 0, 0, 1)
    )
    identity_table = bytes(range(256))
    clut = bytearray()
    for cyan, magenta, yellow, black in product((0, 1), repeat=4):
        red = (1 - cyan) * (1 - black)
        green = (1 - magenta) * (1 - black)
        blue = (1 - yellow) * (1 - black)
        xyz = (
            0.4360747 * red + 0.3850649 * green + 0.1430804 * blue,
            0.2225045 * red + 0.7168786 * green + 0.0606169 * blue,
            0.0139322 * red + 0.0971045 * green + 0.7141733 * blue,
        )
        clut.extend(round(max(0.0, min(value, 1.0)) * 255) for value in xyz)
    a_to_b = (
        b"mft1"
        + bytes(4)
        + bytes((4, 3, 2, 0))
        + fixed_identity
        + identity_table * 4
        + bytes(clut)
        + identity_table * 3
    )

    tag_data = [
        (signature, base[offset : offset + size])
        for signature in (b"desc", b"cprt", b"wtpt")
        for _, offset, size in (records[signature],)
    ]
    tag_data.append((b"A2B0", a_to_b))
    table_size = 4 + len(tag_data) * 12
    payload_offset = 128 + table_size
    tag_table = bytearray(struct.pack(">I", len(tag_data)))
    payload = bytearray()
    for signature, data in tag_data:
        tag_table.extend(struct.pack(">4sII", signature, payload_offset + len(payload), len(data)))
        payload.extend(data)
        payload.extend(bytes(-len(data) % 4))
    profile = header + tag_table + payload
    struct.pack_into(">I", profile, 0, len(profile))
    return bytes(profile)


def test_decode_image_file_without_profile_preserves_rgb_bytes() -> None:
    from dinkster_values import decode_image_file

    pixels = np.array(
        [[[255, 0, 0], [0, 255, 0], [0, 0, 255]], [[0, 0, 0], [255, 255, 255], [128, 128, 128]]],
        dtype=np.uint8,
    )
    decoded = np.asarray(decode_image_file(BytesAsset(png_bytes(pixels))))
    expected = pixels.astype(np.float32) / 255.0
    assert decoded.shape == (1, 2, 3, 3)
    assert decoded.dtype == np.float32
    assert decoded.min() >= 0.0 and decoded.max() <= 1.0
    assert decoded[0].tobytes() == expected.tobytes()


def test_decode_image_file_applies_embedded_icc_profile() -> None:
    import PIL.Image
    import PIL.ImageCms
    from dinkster_values import decode_image_file

    pixels = np.array([[[255, 0, 0], [0, 255, 0], [100, 200, 25]]], dtype=np.uint8)
    profile = swapped_red_green_profile_bytes()
    image = PIL.Image.fromarray(pixels, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", icc_profile=profile)

    transformed = PIL.ImageCms.profileToProfile(
        image,
        PIL.ImageCms.ImageCmsProfile(io.BytesIO(profile)),
        PIL.ImageCms.createProfile("sRGB"),
        renderingIntent=PIL.ImageCms.Intent.PERCEPTUAL,
        outputMode="RGB",
    )
    assert transformed is not None
    expected = np.asarray(transformed, dtype=np.float32) / 255.0
    naive = pixels.astype(np.float32) / 255.0
    decoded = np.asarray(decode_image_file(BytesAsset(buffer.getvalue())))

    assert not np.array_equal(expected, naive)
    assert decoded[0, 0, 0, 1] > 0.9 and decoded[0, 0, 0, 0] < 0.1
    assert decoded[0, 0, 1, 0] > 0.9 and decoded[0, 0, 1, 1] < 0.1
    assert np.array_equal(decoded[0], expected)


def test_decode_image_file_malformed_icc_profile_falls_back_to_rgb() -> None:
    import PIL.Image
    from dinkster_values import decode_image_file

    pixels = np.array([[[20, 40, 60], [80, 100, 120]]], dtype=np.uint8)
    buffer = io.BytesIO()
    PIL.Image.fromarray(pixels, mode="RGB").save(
        buffer, format="PNG", icc_profile=b"not an ICC profile"
    )

    decoded = np.asarray(decode_image_file(BytesAsset(buffer.getvalue())))
    expected = pixels.astype(np.float32) / 255.0
    assert decoded[0].tobytes() == expected.tobytes()


def test_decode_image_file_cmyk_with_profile_returns_rgb_batch() -> None:
    import PIL.Image
    import PIL.ImageCms
    from dinkster_values import decode_image_file

    image = PIL.Image.new("CMYK", (2, 2))
    image.putdata([(0, 255, 255, 0), (255, 0, 255, 0), (255, 255, 0, 0), (10, 20, 30, 40)])
    profile = cmyk_profile_bytes()
    buffer = io.BytesIO()
    image.save(buffer, format="TIFF", icc_profile=profile)

    transformed = PIL.ImageCms.profileToProfile(
        image,
        PIL.ImageCms.ImageCmsProfile(io.BytesIO(profile)),
        PIL.ImageCms.createProfile("sRGB"),
        renderingIntent=PIL.ImageCms.Intent.PERCEPTUAL,
        outputMode="RGB",
    )
    assert transformed is not None
    expected = np.asarray(transformed, dtype=np.float32) / 255.0
    naive = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    decoded = np.asarray(decode_image_file(BytesAsset(buffer.getvalue(), name="cmyk.tiff")))
    assert decoded.shape == (1, 2, 2, 3)
    assert decoded.dtype == np.float32
    assert decoded.min() >= 0.0 and decoded.max() <= 1.0
    assert not np.array_equal(expected, naive)
    assert np.array_equal(decoded[0], expected)


def test_decode_image_file_preserves_rgba() -> None:
    import io

    import PIL.Image
    from dinkster_values import decode_image_file

    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[..., 0] = 200
    rgba[..., 3] = [[0, 64], [128, 255]]
    buffer = io.BytesIO()
    PIL.Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    decoded = np.asarray(decode_image_file(BytesAsset(buffer.getvalue())))
    assert decoded.shape == (1, 2, 2, 4)
    np.testing.assert_array_equal(decoded[0], rgba.astype(np.float32) / 255.0)


def test_decode_image_file_preserves_palette_transparency() -> None:
    import PIL.Image
    from dinkster_values import decode_image_file

    image = PIL.Image.new("P", (2, 1))
    image.putpalette([255, 0, 0, 0, 255, 0] + [0] * 762)
    image.putdata([0, 1])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", transparency=0)
    decoded = np.asarray(decode_image_file(BytesAsset(buffer.getvalue())))
    np.testing.assert_array_equal(decoded, [[[[1, 0, 0, 0], [0, 1, 0, 1]]]])


def test_decode_image_file_rejects_non_images() -> None:
    from dinkster_values import decode_image_file

    with pytest.raises(ValueError, match="cannot decode 'junk.bin'"):
        decode_image_file(BytesAsset(b"not an image", name="junk.bin"))
    with pytest.raises(TypeError, match="expects an asset with open"):
        decode_image_file(b"raw bytes")


def test_merge_image_batches_concatenates_and_reconciles() -> None:
    from dinkster_values import merge_image_batches

    first = np.zeros((2, 4, 4, 3), dtype=np.float32)
    same_shape = np.ones((1, 4, 4, 3), dtype=np.float32)
    merged = np.asarray(merge_image_batches([first, same_shape]))
    assert merged.shape == (3, 4, 4, 3)
    assert np.array_equal(merged[:2], first)
    assert np.array_equal(merged[2], same_shape[0])

    # Channel pad: RGB widens to RGBA with constant 1.0 (ComfyUI ImageBatch).
    rgba = np.zeros((1, 4, 4, 4), dtype=np.float32)
    padded = np.asarray(merge_image_batches([rgba, same_shape]))
    assert padded.shape == (2, 4, 4, 4)
    assert np.all(padded[1, :, :, 3] == 1.0)

    # Size reconcile: later images resize to the FIRST image's H/W.
    small = np.ones((1, 2, 2, 3), dtype=np.float32)
    resized = np.asarray(merge_image_batches([first, small]))
    assert resized.shape == (3, 4, 4, 3)
    assert np.allclose(resized[2], 1.0)  # constant image stays constant

    # Determinism, single passthrough, empty refusal.
    again = np.asarray(merge_image_batches([first, small]))
    assert np.array_equal(resized, again)
    assert np.array_equal(np.asarray(merge_image_batches([first])), first)
    with pytest.raises(ValueError, match="zero image batches"):
        merge_image_batches([])


def test_worker_image_type_decodes_to_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The worker half (dinkster_compat_comfy.image): same bytes, torch
    runtime form. torch is faked through sys.modules - the point is the
    codec calls torch.from_numpy on the decoded array, not torch itself."""
    import sys
    import types

    from dinkster_compat_comfy.image import register_image_type

    captured: list[np.ndarray] = []

    def from_numpy(array: Any) -> Any:
        captured.append(array)
        return FakeTensor(array)

    fake_torch = types.ModuleType("torch")
    fake_torch.from_numpy = from_numpy  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    registry = TypeRegistry()
    register_image_type(registry, COMFY_IMAGE_TYPE)
    array = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(1, 2, 2, 3)
    spec = registry.spec(COMFY_IMAGE_TYPE)
    decoded = spec.decode(spec.encode(FakeTensor(array)))
    assert isinstance(decoded, FakeTensor)
    assert np.array_equal(captured[0], array)
    assert spec.prepare_buffer_encoding is prepare_image_array_encoding
    # Both halves agree on identity: worker wrap and host fingerprint match.
    assert registry.wrap(COMFY_IMAGE_TYPE, array).fingerprint == (
        image_array_fingerprint(COMFY_IMAGE_TYPE)(array)
    )
