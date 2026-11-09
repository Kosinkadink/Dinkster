"""Retained shared image-file decoding and registered asset decoding."""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
import pytest
from dinkster_assets import AssetRef, digest_file
from dinkster_image import register_types
from dinkster_values import TypeRegistry
from dinkster_values.image_codec import decode_image_file
from dinkster_values.storage import image_input
from PIL import Image, ImageCms, ImageOps


class _Resolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path | None:
        return self.path if digest == digest_file(self.path) else None


def _asset(path: Path, media_type: str = "image/png") -> AssetRef:
    return AssetRef(
        digest=digest_file(path),
        name=path.name,
        size=path.stat().st_size,
        media_type=media_type,
        resolver=_Resolver(path),
    )


def test_load_sixteen_bit_still_preserves_source_samples(tmp_path: Path) -> None:
    samples = np.array([[0, 32768, 32832, 65535]], dtype=np.uint16)
    path = tmp_path / "sixteen-bit.png"
    Image.fromarray(samples).save(path)
    loaded = np.asarray(decode_image_file(_asset(path)))
    expected = np.repeat(samples[None, ..., None], 3, axis=-1)
    np.testing.assert_array_equal(loaded, expected)
    assert loaded.dtype == np.uint16


def _swapped_red_green_profile() -> bytes:
    profile = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    count = struct.unpack_from(">I", profile, 128)[0]
    records: dict[bytes, int] = {}
    for index in range(count):
        record_offset = 132 + index * 12
        signature = struct.unpack_from(">4s", profile, record_offset)[0]
        records[signature] = record_offset
    red_record = records[b"rXYZ"]
    green_record = records[b"gXYZ"]
    red_location = bytes(profile[red_record + 4 : red_record + 12])
    green_location = bytes(profile[green_record + 4 : green_record + 12])
    profile[red_record + 4 : red_record + 12] = green_location
    profile[green_record + 4 : green_record + 12] = red_location
    return bytes(profile)


def _decoded_image_reference(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        info = dict(source.info)
        source.load()
        oriented = ImageOps.exif_transpose(source)
        profile = info.get("icc_profile")
        if isinstance(profile, bytes):
            converted = ImageCms.profileToProfile(
                oriented,
                ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                ImageCms.createProfile("sRGB"),
                renderingIntent=ImageCms.Intent.PERCEPTUAL,
                outputMode="RGB",
            )
            assert converted is not None
        else:
            converted = oriented.convert("RGB")
        rgb = np.asarray(converted, dtype=np.float32)
    return np.ascontiguousarray(rgb / 255.0, dtype=np.float32)[None, :, :, :]


def test_decoded_image_rendition_matches_loader_corpus(tmp_path: Path) -> None:
    pixels = np.array(
        [
            [[255, 0, 0], [0, 255, 0], [100, 200, 25]],
            [[7, 31, 211], [90, 40, 10], [240, 128, 64]],
        ],
        dtype=np.uint8,
    )
    source = Image.fromarray(pixels, mode="RGB")
    plain = tmp_path / "plain.png"
    source.save(plain)
    profiled = tmp_path / "profiled.png"
    source.save(profiled, icc_profile=_swapped_red_green_profile())
    oriented = tmp_path / "oriented.jpg"
    exif = source.getexif()
    exif[274] = 6
    source.save(oriented, quality=100, subsampling=0, exif=exif)

    for path, media_type in (
        (plain, "image/png"),
        (profiled, "image/png"),
        (oriented, "image/jpeg"),
    ):
        decoded = np.asarray(decode_image_file(_asset(path, media_type)))
        expected = _decoded_image_reference(path)
        assert decoded.dtype == np.uint8 and decoded.flags.c_contiguous
        assert decoded.nbytes == expected.nbytes // 4
        assert np.array_equal(np.asarray(image_input(decoded)), expected)

    profiled_decoded = np.asarray(decode_image_file(_asset(profiled)))
    assert not np.array_equal(profiled_decoded[0], pixels)


def test_native_asset_decoder_preserves_alpha(tmp_path: Path) -> None:
    path = tmp_path / "alpha.png"
    Image.new("RGBA", (2, 3), (64, 128, 192, 128)).save(path)
    decoded = np.asarray(decode_image_file(_asset(path)))
    assert decoded.shape == (1, 3, 2, 4)
    np.testing.assert_array_equal(decoded[0, 0, 0], [64, 128, 192, 128])


def test_asset_decoder_rejects_decompression_bombs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "bomb.png"
    Image.new("RGB", (2, 2)).save(path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)
    with pytest.raises(ValueError, match="cannot decode.*decompression bomb"):
        decode_image_file(_asset(path))


def test_media_type_registration_provides_image_asset_decoder(tmp_path: Path) -> None:
    registry = TypeRegistry()
    register_types(registry)
    image = registry.asset_decoder_for("IMAGE")
    merge = registry.batch_merge_for("IMAGE")
    assert image is not None and image.provider_id == "dinkster.image-file@3"
    assert merge is not None and merge.provider_id == "dinkster.image-batch-merge@2"
    register_types(registry)
    assert registry.asset_decoder_for("IMAGE") is image
    assert registry.batch_merge_for("IMAGE") is merge
    path = tmp_path / "registered.png"
    pixels = np.array([[[7, 31, 211, 128], [240, 128, 64, 255]]], dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    np.testing.assert_array_equal(image.decode(_asset(path)), pixels[None])
