"""The shared image-array codec (dinkster_values.image_codec): npy bytes as
the cross-interpreter contract, PNG as the browser form.

What this proves: the bytes round-trip losslessly through numpy alone (the
torchless-engine half of the IMAGE story), torch tensors are accepted
by duck type without this interpreter ever importing torch, fingerprints
are runtime-form-independent (hazard H4: a tensor fingerprinted in a worker
keys the same cache entry as its numpy twin here), and the image pack
registration declares and serves the PNG
rendition from encoded payloads - the /api/values path in miniature."""

from __future__ import annotations

import io
import struct
import zlib
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pytest
from dinkster_api.v1 import IMAGE_TYPE as COMFY_IMAGE_TYPE
from dinkster_image import register_types as register_image_types
from dinkster_values import (
    TypeRegistry,
    annotate_mask,
    decode_image_array,
    encode_canonical_png,
    encode_image_array,
    image_array_fingerprint,
    prepare_image_array_encoding,
    render_image_png,
    render_mask_png,
)
from PIL import Image


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

    def view(self, dtype: object) -> FakeTensor:
        assert dtype == "uint16"
        return FakeTensor((self.array.view(np.uint32) >> 16).astype(np.uint16))

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


def test_fingerprint_separates_type_dtype_shape_content() -> None:
    array = np.zeros((2, 2), dtype=np.float32)
    fingerprint = image_array_fingerprint("IMAGE")
    assert fingerprint(array) != image_array_fingerprint("dev.image")(array)
    assert fingerprint(array) != fingerprint(array.astype(np.float64))
    assert fingerprint(array) != fingerprint(array.reshape(1, 4))
    assert fingerprint(array) != fingerprint(array + 1.0)


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


def test_render_mask_normalizes_compact_alpha_bytes() -> None:
    alpha = annotate_mask(
        np.array([[[0, 128, 255]]], dtype=np.uint8),
        polarity="transparency",
        semantic="alpha",
    )

    with Image.open(io.BytesIO(render_mask_png(alpha))) as image:
        np.testing.assert_array_equal(np.asarray(image), [[255, 127, 0]])


def make_host_registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_image_types(registry)
    return registry


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
    assert decoded.shape == (1, 2, 3, 3)
    assert decoded.dtype == np.uint8
    assert decoded[0].tobytes() == pixels.tobytes()


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
    expected = np.asarray(transformed, dtype=np.uint8)
    naive = pixels
    decoded = np.asarray(decode_image_file(BytesAsset(buffer.getvalue())))

    assert not np.array_equal(expected, naive)
    assert decoded[0, 0, 0, 1] > 0.9 * 255 and decoded[0, 0, 0, 0] < 0.1 * 255
    assert decoded[0, 0, 1, 0] > 0.9 * 255 and decoded[0, 0, 1, 1] < 0.1 * 255
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
    assert decoded[0].tobytes() == pixels.tobytes()


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
    expected = np.asarray(transformed, dtype=np.uint8)
    naive = np.asarray(image.convert("RGB"), dtype=np.uint8)

    decoded = np.asarray(decode_image_file(BytesAsset(buffer.getvalue(), name="cmyk.tiff")))
    assert decoded.shape == (1, 2, 2, 3)
    assert decoded.dtype == np.uint8
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
    np.testing.assert_array_equal(decoded[0], rgba)


def test_decode_image_file_preserves_palette_transparency() -> None:
    import PIL.Image
    from dinkster_values import decode_image_file

    image = PIL.Image.new("P", (2, 1))
    image.putpalette([255, 0, 0, 0, 255, 0] + [0] * 762)
    image.putdata([0, 1])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", transparency=0)
    decoded = np.asarray(decode_image_file(BytesAsset(buffer.getvalue())))
    np.testing.assert_array_equal(decoded, [[[[255, 0, 0, 0], [0, 255, 0, 255]]]])


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
