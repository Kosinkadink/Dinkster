from __future__ import annotations

import base64
import io
import json
import struct
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from dinkster_api.v1 import annotate_image, annotate_mask, media_semantics
from dinkster_assets import AssetError, AssetRef, digest_file
from dinkster_nodes_media_io import (
    LoadImage,
    LoadImageOutput,
    LoadMask,
    PaintMask,
    PreviewImage,
    ReadImageMetadata,
    SaveAnimatedImage,
    SaveImage,
    SaveMask,
    register_media_types,
)
from dinkster_schema import (
    AssetWidget,
    OutputRepresents,
    SaveTargetWidget,
    SourceFilenameSpec,
    StringWidget,
    TypeExpr,
)
from dinkster_values import TypeRegistry
from dinkster_values.storage import image_input
from PIL import Image, ImageCms, ImageOps
from PIL.PngImagePlugin import PngInfo


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


def test_load_sixteen_bit_still_and_mask_preserve_source_samples(tmp_path: Path) -> None:
    from dinkster_values.image_codec import decode_image_file

    samples = np.array([[0, 32768, 32832, 65535]], dtype=np.uint16)
    path = tmp_path / "sixteen-bit.png"
    Image.fromarray(samples).save(path)
    asset = _asset(path)
    loaded = np.asarray(LoadImage.execute(image=asset)["image"])
    expected = np.repeat(samples[None, ..., None], 3, axis=-1)
    np.testing.assert_array_equal(loaded, expected)
    assert loaded.dtype == np.uint16
    np.testing.assert_array_equal(decode_image_file(asset), expected)
    mask = np.asarray(LoadMask.execute(mask=asset, channel="red", mask_polarity="coverage")["mask"])
    np.testing.assert_array_equal(mask, samples[None])
    assert mask.dtype == np.uint16


def _mount(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "output"
    root.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "mounts": [{"id": "out", "root": str(root), "mode": "readwrite"}],
                "outputMount": "out",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    return root


def test_save_image_uses_the_configured_default_output_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _mount(tmp_path, monkeypatch)

    result = SaveImage.execute(images=np.zeros((1, 2, 3, 3), dtype=np.float32))

    (asset,) = cast("list[AssetRef]", result["assets"])
    assert asset.virtual_path == "mounts/out/ComfyUI_00001.png"
    assert (root / "ComfyUI_00001.png").is_file()


def test_image_io_schemas_use_typed_assets_and_mounted_targets() -> None:
    load = LoadImage.schema()
    output = LoadImageOutput.schema()
    mask = LoadMask.schema()
    save = SaveImage.schema()
    assert load.inputs[0].type == TypeExpr.asset_of(TypeExpr.concrete("dinkster.image"))
    assert load.inputs[0].widget == AssetWidget(
        ("image/png", "image/jpeg", "image/webp", "image/gif", "image/tiff"),
        kind="media/image",
        allow_upload=True,
    )
    assert load.inputs[0].source_filename == SourceFilenameSpec("media/image", "input")
    assert output.inputs[0].source_filename == SourceFilenameSpec("media/image", "output")
    assert mask.inputs[0].type == TypeExpr.asset_of(TypeExpr.concrete("dinkster.mask"))
    assert save.inputs[1].widget == SaveTargetWidget()
    assert [item.id for item in save.outputs] == ["images", "assets"]
    assert save.outputs[1].type == TypeExpr.list_of(
        TypeExpr.asset_of(TypeExpr.concrete("dinkster.image"))
    )
    assert load.aliases == ("LoadImage",)
    assert save.aliases == ("SaveImage",)
    assert PreviewImage.schema().outputs[0].preview is True
    representation = OutputRepresents(input="image", rendition="decoded-image")
    assert load.outputs[0].represents == representation
    assert output.outputs[0].represents == representation
    assert all(item.represents is None for item in (*load.outputs[1:], *output.outputs[1:]))
    assert all(item.represents is None for item in mask.outputs)

    paint = PaintMask.schema()
    assert paint.node_type == "dinkster.mask.paint"
    assert [item.id for item in paint.inputs] == ["source", "operations"]
    assert paint.inputs[0].type == TypeExpr.asset_of(TypeExpr.concrete("dinkster.image"))
    assert paint.inputs[0].widget == AssetWidget(
        ("image/png", "image/jpeg", "image/webp", "image/gif", "image/tiff"),
        kind="media/image",
        allow_upload=False,
    )
    assert paint.inputs[1].widget == StringWidget(multiline=True)
    assert [
        (item.id, item.type, item.mask_polarity, item.mask_semantic) for item in paint.outputs
    ] == [("mask", TypeExpr.concrete("dinkster.mask"), "transparency", "alpha")]


def _paint_operations(
    asset: AssetRef,
    width: int,
    height: int,
    commands: list[dict[str, object]],
) -> str:
    return json.dumps(
        {
            "version": 1,
            "sourceDigest": asset.digest,
            "width": width,
            "height": height,
            "commands": commands,
        },
        separators=(",", ":"),
    )


def test_paint_mask_replays_source_alpha_and_ordered_byte_commands(tmp_path: Path) -> None:
    pixels = np.zeros((5, 5, 4), dtype=np.uint8)
    pixels[..., :3] = 50
    pixels[..., 3] = 255
    pixels[0, 0, 3] = 15
    path = tmp_path / "source.png"
    Image.fromarray(pixels, mode="RGBA").save(path)
    asset = _asset(path)

    base = cast(
        np.ndarray,
        PaintMask.execute(source=asset, operations=_paint_operations(asset, 5, 5, []))["mask"],
    )
    expected = np.zeros((1, 5, 5), dtype=np.float32)
    expected[0, 0, 0] = 240 / 255
    np.testing.assert_array_equal(base, expected)
    assert media_semantics(base) == {"polarity": "transparency", "semantic": "alpha"}

    commands: list[dict[str, object]] = [
        {"op": "clear"},
        {
            "op": "stroke",
            "mode": "paint",
            "size": 2,
            "hardness": 1,
            "points": [{"x": 2.5, "y": 2.5, "pressure": 1}],
        },
        {
            "op": "stroke",
            "mode": "erase",
            "size": 2,
            "hardness": 1,
            "points": [{"x": 2.5, "y": 2.5, "pressure": 0.5}],
        },
        {"op": "invert"},
    ]
    painted = cast(
        np.ndarray,
        PaintMask.execute(
            source=asset,
            operations=_paint_operations(asset, 5, 5, commands),
        )["mask"],
    )
    expected_bytes = np.full((5, 5), 255, dtype=np.uint8)
    expected_bytes[2, 2] = 127
    for row, column in ((1, 2), (2, 1), (2, 3), (3, 2)):
        expected_bytes[row, column] = 0
    np.testing.assert_array_equal(painted, expected_bytes[None, ...].astype(np.float32) / 255)


def test_paint_mask_matches_pixel_centers_pressure_rounding_and_off_canvas_reach(
    tmp_path: Path,
) -> None:
    path = tmp_path / "opaque.png"
    Image.new("RGB", (3, 2), (20, 30, 40)).save(path)
    asset = _asset(path)
    commands: list[dict[str, object]] = [
        {
            "op": "stroke",
            "mode": "paint",
            "size": 2.0,
            "hardness": 1.0,
            "points": [
                {"x": 0.5, "y": 0.5, "pressure": 0.5},
                {"x": -0.5, "y": 1.5, "pressure": 1.0},
                {"x": 1.5, "y": 0.5, "pressure": 0.0},
            ],
        }
    ]
    mask = cast(
        np.ndarray,
        PaintMask.execute(
            source=asset,
            operations=_paint_operations(asset, 3, 2, commands),
        )["mask"],
    )
    expected = np.zeros((1, 2, 3), dtype=np.float32)
    expected[0, 0, 0] = 128 / 255
    expected[0, 1, 0] = 1
    np.testing.assert_array_equal(mask, expected)


def test_paint_mask_soft_edge_has_exact_byte_falloff_and_inclusive_radius(tmp_path: Path) -> None:
    path = tmp_path / "opaque.png"
    Image.new("RGB", (5, 1)).save(path)
    asset = _asset(path)
    stroke = {
        "op": "stroke",
        "mode": "paint",
        "size": 4,
        "hardness": 0.25,
        "points": [{"x": 2.5, "y": 0.5, "pressure": 1}],
    }
    mask = cast(
        np.ndarray,
        PaintMask.execute(
            source=asset,
            operations=_paint_operations(asset, 5, 1, [stroke]),
        )["mask"],
    )
    np.testing.assert_array_equal(
        mask,
        np.array([[[0, 170, 255, 170, 0]]], dtype=np.float32) / 255,
    )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda value: value.update(extra=True), "require version"),
        (lambda value: value.update(version=True), "version must be 1"),
        (lambda value: value.update(sourceDigest="sha256:" + "0" * 64), "BLAKE3"),
        (lambda value: value.update(width=0), "positive integers"),
        (lambda value: value.update(commands=[{"op": "clear", "extra": 1}]), "unknown fields"),
        (
            lambda value: value.update(
                commands=[
                    {
                        "op": "stroke",
                        "mode": "paint",
                        "size": True,
                        "hardness": 1,
                        "points": [{"x": 0.5, "y": 0.5, "pressure": 1}],
                    }
                ]
            ),
            "size must be a number",
        ),
        (
            lambda value: value.update(
                commands=[
                    {
                        "op": "stroke",
                        "mode": "paint",
                        "size": 1,
                        "hardness": 1,
                        "points": [{"x": -1, "y": 0.5, "pressure": 1}],
                    }
                ]
            ),
            "beyond the canvas brush reach",
        ),
    ],
)
def test_paint_mask_strictly_validates_operations(
    tmp_path: Path,
    mutate: Any,
    match: str,
) -> None:
    path = tmp_path / "source.png"
    Image.new("RGB", (2, 2)).save(path)
    asset = _asset(path)
    value: dict[str, object] = {
        "version": 1,
        "sourceDigest": asset.digest,
        "width": 2,
        "height": 2,
        "commands": [],
    }
    mutate(value)
    with pytest.raises(ValueError, match=match):
        PaintMask.execute(source=asset, operations=json.dumps(value))


def test_paint_mask_rejects_stale_dimensions_animation_and_bounded_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_nodes_media_io.image as image_io

    path = tmp_path / "source.png"
    Image.new("RGB", (3, 2)).save(path)
    asset = _asset(path)
    stale = json.loads(_paint_operations(asset, 3, 2, []))
    stale["sourceDigest"] = "blake3:" + "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        PaintMask.execute(source=asset, operations=json.dumps(stale))
    with pytest.raises(ValueError, match="dimensions do not match"):
        PaintMask.execute(source=asset, operations=_paint_operations(asset, 2, 2, []))

    animation = tmp_path / "animated.gif"
    Image.new("RGB", (2, 2), "red").save(
        animation,
        save_all=True,
        append_images=[Image.new("RGB", (2, 2), "blue")],
    )
    animated_asset = _asset(animation, "image/gif")
    with pytest.raises(ValueError, match="animated and multipage"):
        PaintMask.execute(
            source=animated_asset,
            operations=_paint_operations(animated_asset, 2, 2, []),
        )

    monkeypatch.setattr(image_io, "MAX_MASK_PAINT_JSON_BYTES", 10)
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        PaintMask.execute(source=asset, operations=_paint_operations(asset, 3, 2, []))
    monkeypatch.setattr(image_io, "MAX_MASK_PAINT_JSON_BYTES", 4 * 1024 * 1024)
    monkeypatch.setattr(image_io, "MAX_MASK_PAINT_COMMANDS", 1)
    with pytest.raises(ValueError, match="commands exceed 1"):
        PaintMask.execute(
            source=asset,
            operations=_paint_operations(asset, 3, 2, [{"op": "clear"}, {"op": "invert"}]),
        )
    monkeypatch.setattr(image_io, "MAX_MASK_PAINT_COMMANDS", 2048)
    monkeypatch.setattr(image_io, "MAX_MASK_PAINT_STROKE_POINTS", 1)
    points = [{"x": 0.5, "y": 0.5, "pressure": 1}] * 2
    stroke = {"op": "stroke", "mode": "paint", "size": 1, "hardness": 1, "points": points}
    with pytest.raises(ValueError, match="1 to 1 items"):
        PaintMask.execute(source=asset, operations=_paint_operations(asset, 3, 2, [stroke]))
    monkeypatch.setattr(image_io, "MAX_MASK_PAINT_STROKE_POINTS", 8192)
    monkeypatch.setattr(image_io, "MAX_MASK_PAINT_TOTAL_POINTS", 1)
    strokes = [
        {"op": "stroke", "mode": "paint", "size": 1, "hardness": 1, "points": [point]}
        for point in points
    ]
    with pytest.raises(ValueError, match="operations exceed 1 points"):
        PaintMask.execute(source=asset, operations=_paint_operations(asset, 3, 2, strokes))


def test_paint_mask_shared_bounds_are_inclusive(tmp_path: Path) -> None:
    import dinkster_nodes_media_io.image as image_io

    assert image_io.MAX_MASK_PAINT_JSON_BYTES == 4_194_304
    assert image_io.MAX_MASK_PAINT_COMMANDS == 2_048
    assert image_io.MAX_MASK_PAINT_STROKE_POINTS == 8_192
    assert image_io.MAX_MASK_PAINT_TOTAL_POINTS == 32_768

    path = tmp_path / "source.png"
    Image.new("RGB", (1, 1)).save(path)
    asset = _asset(path)
    minimal = _paint_operations(asset, 1, 1, [])
    exact_bytes = minimal + " " * (image_io.MAX_MASK_PAINT_JSON_BYTES - len(minimal))
    PaintMask.execute(source=asset, operations=exact_bytes)
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        PaintMask.execute(source=asset, operations=exact_bytes + " ")

    PaintMask.execute(
        source=asset,
        operations=_paint_operations(
            asset, 1, 1, [{"op": "clear"}] * image_io.MAX_MASK_PAINT_COMMANDS
        ),
    )
    point = {"x": 0.5, "y": 0.5, "pressure": 0}
    full_stroke = {
        "op": "stroke",
        "mode": "paint",
        "size": 1,
        "hardness": 1,
        "points": [point] * image_io.MAX_MASK_PAINT_STROKE_POINTS,
    }
    PaintMask.execute(
        source=asset,
        operations=_paint_operations(asset, 1, 1, [full_stroke] * 4),
    )


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
        decoded = cast(np.ndarray, LoadImage.execute(image=_asset(path, media_type))["image"])
        expected = _decoded_image_reference(path)
        assert decoded.dtype == np.uint8 and decoded.flags.c_contiguous
        assert decoded.nbytes == expected.nbytes // 4
        assert np.array_equal(np.asarray(image_input(decoded)), expected)

    profiled_decoded = cast(
        np.ndarray,
        LoadImage.execute(image=_asset(profiled))["image"],
    )
    assert not np.array_equal(profiled_decoded[0], pixels)


def test_load_image_applies_orientation_alpha_polarity_and_opaque_fallback(tmp_path: Path) -> None:
    rgba = np.zeros((2, 3, 4), dtype=np.uint8)
    rgba[..., 0] = np.array([[10, 20, 30], [40, 50, 60]])
    rgba[..., 3] = np.array([[0, 64, 128], [192, 255, 32]])
    source = Image.fromarray(rgba, mode="RGBA")
    exif = source.getexif()
    exif[274] = 6
    path = tmp_path / "oriented.png"
    source.save(path, exif=exif)

    loaded = LoadImage.execute(image=_asset(path))
    image = cast(np.ndarray, loaded["image"])
    mask = cast(np.ndarray, loaded["mask"])
    assert image.dtype == np.uint8 and image.flags.c_contiguous
    assert mask.dtype == np.uint8 and mask.flags.c_contiguous
    assert image.shape == (1, 3, 2, 3)
    expected_alpha = np.rot90(rgba[..., 3], k=3)
    np.testing.assert_array_equal(mask[0], expected_alpha)
    np.testing.assert_array_equal(
        image_input(mask)[0], np.float32(1.0) - expected_alpha.astype(np.float32) / 255
    )

    opaque_path = tmp_path / "opaque.png"
    Image.new("RGB", (7, 5), (1, 2, 3)).save(opaque_path)
    opaque = LoadImage.execute(image=_asset(opaque_path))
    opaque_mask = cast(np.ndarray, opaque["mask"])
    assert opaque_mask.shape == (1, 64, 64)
    assert np.all(opaque_mask == 255)
    assert np.count_nonzero(image_input(opaque_mask)) == 0


@pytest.mark.parametrize("mask_polarity", ["coverage", "transparency"])
@pytest.mark.parametrize("layout", ["rgb", "rgba"])
def test_load_image_matches_comfy_goldens(tmp_path: Path, mask_polarity: str, layout: str) -> None:
    fixture = json.loads((Path(__file__).parent / "goldens/image_load_e20d433a.json").read_text())
    assert fixture["baseline"] == "e20d433a4966dcc88fa5abbae6ace824cb78b263"
    case = fixture["cases"][layout]
    path = tmp_path / "source.png"
    path.write_bytes(base64.b64decode(case["png"]))
    result = LoadImage.execute(image=_asset(path), mask_polarity=mask_polarity)
    expected_semantics = {"semantic": "alpha"}
    if mask_polarity == "transparency":
        expected_semantics["polarity"] = "transparency"
    assert media_semantics(result["mask"]) == expected_semantics
    for output in ("image", "mask"):
        expected = np.asarray(case[output]["values"], dtype=np.float32).reshape(
            case[output]["shape"]
        )
        if output == "mask" and mask_polarity == "coverage":
            actual_storage = np.asarray(result[output])
            np.testing.assert_array_equal(
                actual_storage, np.rint((1.0 - expected) * 255).astype(np.uint8)
            )
            continue
        actual = np.asarray(image_input(result[output]))
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("format_name", ["WEBP", "PNG", "TIFF", "GIF"])
@pytest.mark.parametrize("with_alpha", [False, True])
def test_load_image_batches_animation_and_multipage_frames(
    tmp_path: Path, format_name: str, with_alpha: bool
) -> None:
    path = tmp_path / "animated"
    mode = "RGBA" if with_alpha else "RGB"
    colors = [(220, 40, 10, 128), (20, 100, 240, 255)]
    frames = [Image.new(mode, (3, 2), color if with_alpha else color[:3]) for color in colors]
    frames[0].save(
        path,
        format=format_name,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        lossless=True,
    )
    from PIL import ImageSequence

    with Image.open(path) as source:
        decoded = [frame.convert("RGBA") for frame in ImageSequence.Iterator(source)]
        expected = np.stack([np.asarray(frame) for frame in decoded]).astype(np.float32) / 255.0
    result = LoadImage.execute(image=_asset(path))
    np.testing.assert_array_equal(image_input(result["image"]), expected[..., :3])
    mask = np.asarray(result["mask"])
    if mask.shape[1:] == (64, 64):
        np.testing.assert_array_equal(mask, np.full((2, 64, 64), 255))
    else:
        expected_alpha = np.stack([np.asarray(frame) for frame in decoded])[..., 3]
        np.testing.assert_array_equal(mask, expected_alpha)
    assert mask.shape[0] == 2


def test_load_image_skips_different_sized_pages(tmp_path: Path) -> None:
    path = tmp_path / "pages.tiff"
    Image.new("RGB", (3, 2), "red").save(
        path,
        save_all=True,
        append_images=[Image.new("RGB", (1, 1)), Image.new("RGB", (3, 2), "blue")],
    )
    result = LoadImageOutput.execute(image=_asset(path))
    assert np.asarray(result["image"]).shape == (2, 2, 3, 3)
    np.testing.assert_array_equal(np.asarray(result["image"])[:, 0, 0], [[255, 0, 0], [0, 0, 255]])


def test_image_batches_replay_pinned_decode_contracts(tmp_path: Path) -> None:
    fixture = json.loads((Path(__file__).parent / "goldens/image_load_e20d433a.json").read_text())
    for name, case in fixture["cases"].items():
        if not name.startswith("batch_"):
            continue
        path = tmp_path / name
        path.write_bytes(base64.b64decode(case["file"]))
        result = LoadImage.execute(image=_asset(path))
        assert np.asarray(result["image"]).shape[0] == 2
        for output in ("image", "mask"):
            expected = np.array(case[output]["values"], np.float32).reshape(case[output]["shape"])
            actual = np.asarray(result[output])
            stored_values = 1.0 - expected if output == "mask" else expected
            expected_storage = np.rint(stored_values * 255).astype(np.uint8)
            if name.endswith("_tiff"):
                # The pinned PyAV loader decodes only the first TIFF page.
                assert expected.shape[0] == 1
                np.testing.assert_array_equal(actual[:1], expected_storage)
            elif name.endswith("_gif"):
                # Pillow preserves the palette instead of PyAV's lossy color conversion.
                exact = np.zeros((2, 2, 32, 3) if output == "image" else (2, 2, 32), np.uint8)
                if output == "image":
                    exact[0, ..., 0] = 255
                    exact[1, ..., 2] = 255
                else:
                    exact.fill(255)
                np.testing.assert_array_equal(actual, exact)
                assert expected.shape == exact.shape
                assert not np.array_equal(expected, exact)
            else:
                np.testing.assert_array_equal(actual, expected_storage)


def test_native_asset_decoder_preserves_alpha(tmp_path: Path) -> None:
    from dinkster_nodes_media_io.image import decode_image_file

    path = tmp_path / "alpha.png"
    Image.new("RGBA", (2, 3), (64, 128, 192, 128)).save(path)
    decoded = np.asarray(decode_image_file(_asset(path)))
    assert decoded.shape == (1, 3, 2, 4)
    np.testing.assert_array_equal(decoded[0, 0, 0], [64, 128, 192, 128])


@pytest.mark.parametrize("animated", [False, True])
def test_save_normalizes_premultiplied_alpha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, animated: bool
) -> None:
    root = _mount(tmp_path, monkeypatch)
    pixels = np.array([[[[0.25, 0.125, 0.0, 0.5], [0.0, 0.0, 0.0, 0.0]]]], np.float32)
    expected = pixels.copy()
    images = annotate_image(pixels, alpha="premultiplied")
    saver = SaveAnimatedImage if animated else SaveImage
    saver.execute(images=images, target={"mount": "out", "prefix": "alpha"}, format="png")
    with Image.open(next(root.glob("*.png"))) as image:
        np.testing.assert_array_equal(np.asarray(image), [[[127, 63, 0, 127], [0, 0, 0, 0]]])
    np.testing.assert_array_equal(images, expected)
    assert media_semantics(images)["alpha"] == "premultiplied"


def test_still_and_metadata_loaders_reject_decompression_bombs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "bomb.png"
    Image.new("RGB", (2, 2)).save(path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)
    with pytest.raises(ValueError, match="decompression safety limit"):
        LoadImage.execute(image=_asset(path))
    with pytest.raises(ValueError, match="cannot read metadata"):
        ReadImageMetadata.execute(image=_asset(path))


def test_load_mask_channels_and_polarity_match_core_contract(tmp_path: Path) -> None:
    rgba = np.array([[[10, 20, 30, 64], [40, 50, 60, 255]]], dtype=np.uint8)
    path = tmp_path / "channels.png"
    Image.fromarray(rgba, mode="RGBA").save(path)
    ref = _asset(path)
    alpha = cast(
        np.ndarray,
        LoadMask.execute(mask=ref, channel="alpha", mask_polarity="transparency")["mask"],
    )
    red = cast(
        np.ndarray, LoadMask.execute(mask=ref, channel="red", mask_polarity="coverage")["mask"]
    )
    np.testing.assert_array_equal(alpha, rgba[None, ..., 3])
    np.testing.assert_array_equal(red, rgba[None, ..., 0])
    assert media_semantics(alpha) == {"polarity": "transparency", "semantic": "alpha"}
    assert media_semantics(red) == {}

    opaque_path = tmp_path / "opaque.png"
    Image.new("RGB", (3, 2), "white").save(opaque_path)
    fallback = cast(
        np.ndarray,
        LoadMask.execute(mask=_asset(opaque_path), channel="alpha", mask_polarity="transparency")[
            "mask"
        ],
    )
    assert fallback.shape == (1, 64, 64)
    assert np.all(fallback == 255)
    assert np.count_nonzero(image_input(fallback)) == 0
    assert media_semantics(fallback) == {"polarity": "transparency", "semantic": "alpha"}


@pytest.mark.parametrize(
    ("format_name", "suffix", "media_type"),
    [("png", ".png", "image/png"), ("jpeg", ".jpg", "image/jpeg"), ("webp", ".webp", "image/webp")],
)
def test_save_image_writes_one_ordered_asset_per_batch_element(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    format_name: str,
    suffix: str,
    media_type: str,
) -> None:
    root = _mount(tmp_path, monkeypatch)
    images = np.zeros((2, 2, 3, 3), dtype=np.float32)
    images[0, ..., 0] = 1.0
    images[1, ..., 1] = 0.5
    result = SaveImage.execute(
        images=images,
        target={"mount": "out", "prefix": "renders/scene"},
        format=format_name,
        lossless=True,
    )
    assets = cast("list[AssetRef]", result["assets"])
    assert [asset.name for asset in assets] == [f"scene_00001{suffix}", f"scene_00002{suffix}"]
    assert [asset.media_type for asset in assets] == [media_type, media_type]
    for index, asset in enumerate(assets):
        with Image.open(root / "renders" / asset.name) as decoded:
            rgb = np.asarray(decoded.convert("RGB"))
        if index == 0:
            assert rgb[0, 0, 0] > 240 and rgb[0, 0, 1] < 15
        else:
            assert rgb[0, 0, 1] in range(120, 136) and rgb[0, 0, 0] < 15
    assert result["images"] is images


def test_save_image_rejects_alpha_jpeg_and_embeds_explicit_png_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _mount(tmp_path, monkeypatch)
    rgba = np.ones((1, 2, 2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="JPEG does not support alpha"):
        SaveImage.execute(images=rgba, target={"mount": "out", "prefix": "x"}, format="jpeg")

    rgb = rgba[..., :3]
    result = SaveImage.execute(
        images=rgb,
        target={"mount": "out", "prefix": "meta/image"},
        metadata_json=json.dumps({"prompt": {"3": {"class_type": "KSampler"}}}),
    )
    asset = cast("list[AssetRef]", result["assets"])[0]
    with Image.open(root / "meta" / asset.name) as decoded:
        assert json.loads(decoded.info["prompt"]) == {"3": {"class_type": "KSampler"}}
    with pytest.raises(ValueError, match="only for PNG"):
        SaveImage.execute(
            images=rgb,
            target={"mount": "out", "prefix": "meta/image"},
            format="webp",
            metadata_json="{}",
        )


def test_image_and_mask_io_preserve_forwarded_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mount(tmp_path, monkeypatch)
    image = annotate_image(np.full((1, 2, 3, 4), 0.5, np.float32), alpha="premultiplied")
    for node in (SaveImage, SaveAnimatedImage):
        result = node.execute(images=image, target={"mount": "out", "prefix": "annotated"})
        assert media_semantics(result["images"]) == media_semantics(image)
        np.testing.assert_array_equal(result["images"], image)
    preview = PreviewImage.execute(images=image)["images"]
    assert media_semantics(preview) == media_semantics(image)
    assert np.shares_memory(preview, image)
    mask = annotate_mask(
        np.full((2, 3), 0.5, np.float32), polarity="transparency", semantic="alpha"
    )
    result = SaveMask.execute(
        masks=mask, target={"mount": "out", "prefix": "mask"}, mask_polarity="coverage"
    )
    assert media_semantics(result["masks"]) == media_semantics(mask)
    np.testing.assert_array_equal(np.asarray(result["masks"])[0], mask)


def test_save_image_matches_core_uint8_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _mount(tmp_path, monkeypatch)
    image = np.full((1, 1, 1, 3), 0.5, dtype=np.float32)
    result = SaveImage.execute(
        images=image,
        target={"mount": "out", "prefix": "quantized"},
    )
    asset = cast("list[AssetRef]", result["assets"])[0]
    with Image.open(root / asset.name) as decoded:
        assert decoded.convert("RGB").getpixel((0, 0)) == (127, 127, 127)


@pytest.mark.parametrize(("bit_depth", "atol"), [("8", 0.5 / 255.0), ("16", 0.5 / 65535.0)])
@pytest.mark.parametrize("mask_polarity", ["coverage", "transparency"])
def test_mask_save_load_round_trip_quantization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bit_depth: str,
    atol: float,
    mask_polarity: str,
) -> None:
    root = _mount(tmp_path, monkeypatch)
    mask = np.linspace(0.0, 1.0, 35, dtype=np.float32).reshape(1, 5, 7)
    result = SaveMask.execute(
        masks=mask,
        target={"mount": "out", "prefix": "masks/test"},
        bit_depth=bit_depth,
        mask_polarity=mask_polarity,
    )
    asset = cast("list[AssetRef]", result["assets"])[0]
    loaded = cast(
        np.ndarray,
        LoadMask.execute(
            mask=_asset(root / "masks" / asset.name),
            channel="luminance",
            mask_polarity=mask_polarity,
        )["mask"],
    )
    maximum = 65535 if bit_depth == "16" else 255
    encoded = mask if mask_polarity == "coverage" else 1.0 - mask
    pixels = np.rint(encoded * maximum)
    with Image.open(root / "masks" / asset.name) as saved:
        np.testing.assert_array_equal(np.asarray(saved), pixels[0])
    assert loaded.dtype == (np.uint8 if bit_depth == "8" else np.uint16)
    np.testing.assert_array_equal(loaded, pixels)
    converted = np.asarray(image_input(loaded))
    np.testing.assert_allclose(converted, mask, atol=max(atol, 0.51 / maximum))


@pytest.mark.parametrize("format_name", ["png", "webp"])
def test_save_animated_image_preserves_frame_order_and_timing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    format_name: str,
) -> None:
    root = _mount(tmp_path, monkeypatch)
    images = np.zeros((3, 4, 5, 4), dtype=np.float32)
    images[0, ..., 0] = 1.0
    images[1, ..., 1] = 1.0
    images[2, ..., 2] = 1.0
    images[..., 3] = np.array([1.0, 0.5, 0.0], dtype=np.float32)[:, None, None]
    result = SaveAnimatedImage.execute(
        images=images,
        target={"mount": "out", "prefix": "animation/test"},
        format=format_name,
        fps=8.0,
        loop=2,
        lossless=True,
    )
    asset = cast(AssetRef, result["asset"])
    with Image.open(root / "animation" / asset.name) as opened:
        animation = cast(Any, opened)
        assert animation.n_frames == 3
        assert animation.info["loop"] == 2
        colors: list[tuple[int, int, int]] = []
        durations: list[int] = []
        alphas: list[int] = []
        for index in range(animation.n_frames):
            animation.seek(index)
            pixel = cast("tuple[int, int, int, int]", animation.convert("RGBA").getpixel((0, 0)))
            colors.append(pixel[:3])
            alphas.append(pixel[3])
            durations.append(cast(int, animation.info["duration"]))
    assert colors[:2] == [(255, 0, 0), (0, 255, 0)]
    assert alphas[0] == 255 and alphas[1] in range(126, 130) and alphas[2] == 0
    assert all(abs(duration - 125) <= 1 for duration in durations)


def test_animation_and_preview_reject_empty_or_nonfinite_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mount(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="nonempty"):
        SaveAnimatedImage.execute(images=np.empty((0, 1, 1, 3), np.float32))
    invalid = np.zeros((1, 1, 1, 3), np.float32)
    invalid[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        PreviewImage.execute(images=invalid)


def test_image_io_limits_and_invalid_targets_fail_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dinkster_nodes_media_io.image as image_io

    root = _mount(tmp_path, monkeypatch)
    monkeypatch.setattr(image_io, "MAX_ANIMATION_FRAMES", 1)
    with pytest.raises(ValueError, match="exceeds 1 frames"):
        SaveAnimatedImage.execute(images=np.zeros((2, 1, 1, 3), np.float32))

    monkeypatch.setattr(image_io, "MAX_IMAGE_DIMENSION", 1)
    with pytest.raises(ValueError, match="dimensions exceed 1"):
        PreviewImage.execute(images=np.zeros((1, 1, 2, 3), np.float32))
    with pytest.raises(AssetError, match="mount"):
        SaveImage.execute(images=np.zeros((1, 1, 1, 3), np.float32), target={})
    assert list(root.iterdir()) == []


def test_metadata_parser_normalizes_comfy_a1111_malformed_and_duplicates(tmp_path: Path) -> None:
    info = PngInfo()
    info.add_text("prompt", "not-json")
    info.add_text("prompt", json.dumps({"1": {"class_type": "LoadImage"}}))
    info.add_text("workflow", json.dumps({"nodes": [{"id": 1}]}))
    info.add_text(
        "parameters",
        "a castle\nNegative prompt: fog\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 42",
    )
    info.add_text("unknown", "preserved")
    path = tmp_path / "metadata.png"
    Image.new("RGB", (2, 2), "black").save(path, pnginfo=info)
    document = json.loads(cast(str, ReadImageMetadata.execute(image=_asset(path))["metadata"]))
    assert document["format"] == "dinkster.image-metadata/1"
    assert document["comfy"]["prompt"] == {"1": {"class_type": "LoadImage"}}
    assert document["comfy"]["workflow"] == {"nodes": [{"id": 1}]}
    assert document["a1111"] == {
        "negativePrompt": "fog",
        "prompt": "a castle",
        "settings": {"CFG scale": "7", "Sampler": "Euler a", "Seed": "42", "Steps": "20"},
    }
    assert document["raw"]["unknown"] == "preserved"
    assert document["provenance"]["digest"] == digest_file(path)

    malformed = PngInfo()
    malformed.add_text("workflow", "[")
    malformed_path = tmp_path / "malformed.png"
    Image.new("RGB", (1, 1)).save(malformed_path, pnginfo=malformed)
    parsed = json.loads(
        cast(str, ReadImageMetadata.execute(image=_asset(malformed_path))["metadata"])
    )
    assert parsed["raw"]["workflow"] == "["
    assert parsed["errors"] == ["workflow: invalid JSON"]

    nonfinite = PngInfo()
    nonfinite.add_text("prompt", '{"value":NaN}')
    nonfinite_path = tmp_path / "nonfinite.png"
    Image.new("RGB", (1, 1)).save(nonfinite_path, pnginfo=nonfinite)
    parsed = json.loads(
        cast(str, ReadImageMetadata.execute(image=_asset(nonfinite_path))["metadata"])
    )
    assert parsed["errors"] == ["prompt: invalid JSON"]


def test_metadata_parser_refuses_oversized_compressed_text(tmp_path: Path) -> None:
    info = PngInfo()
    info.add_text("parameters", "x" * (1024 * 1024 + 1), zip=True)
    path = tmp_path / "oversized.png"
    Image.new("RGB", (1, 1)).save(path, pnginfo=info)
    with pytest.raises(ValueError, match="metadata|Decompressed Data Too Large"):
        ReadImageMetadata.execute(image=_asset(path))


def test_metadata_parser_handles_absent_and_bounded_compressed_text(tmp_path: Path) -> None:
    absent_path = tmp_path / "absent.png"
    Image.new("RGB", (1, 1)).save(absent_path)
    absent = json.loads(cast(str, ReadImageMetadata.execute(image=_asset(absent_path))["metadata"]))
    assert absent["raw"] == {}
    assert "comfy" not in absent and "a1111" not in absent and "errors" not in absent

    compressed = PngInfo()
    compressed.add_text("parameters", "small compressed value", zip=True)
    compressed_path = tmp_path / "compressed.png"
    Image.new("RGB", (1, 1)).save(compressed_path, pnginfo=compressed)
    parsed = json.loads(
        cast(str, ReadImageMetadata.execute(image=_asset(compressed_path))["metadata"])
    )
    assert parsed["raw"]["parameters"] == "small compressed value"


def test_metadata_parser_refuses_oversized_source_before_open(tmp_path: Path) -> None:
    path = tmp_path / "claimed-large.png"
    Image.new("RGB", (1, 1)).save(path)
    source = _asset(path)
    oversized = AssetRef(
        digest=source.digest,
        name=source.name,
        size=512 * 1024 * 1024 + 1,
        media_type=source.media_type,
        resolver=source.resolver,
    )
    with pytest.raises(ValueError, match="metadata source exceeds"):
        ReadImageMetadata.execute(image=oversized)


def test_image_and_metadata_limits_use_verified_descriptor_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dinkster_nodes_media_io.image as image_io
    import dinkster_nodes_media_io.image_metadata as metadata_io

    path = tmp_path / "forged-size.png"
    Image.new("RGB", (1, 1)).save(path)
    source = _asset(path)
    forged = AssetRef(
        digest=source.digest,
        name=source.name,
        size=0,
        media_type=source.media_type,
        resolver=source.resolver,
    )
    monkeypatch.setattr(image_io, "MAX_IMAGE_FILE_BYTES", 1)
    with pytest.raises(ValueError, match="image asset exceeds"):
        LoadImage.execute(image=forged)
    monkeypatch.setattr(metadata_io, "MAX_METADATA_ASSET_BYTES", 1)
    with pytest.raises(ValueError, match="metadata source exceeds"):
        ReadImageMetadata.execute(image=forged)


def test_media_type_registration_provides_image_and_mask_asset_decoders() -> None:
    registry = TypeRegistry()
    register_media_types(registry)
    image = registry.asset_decoder_for("dinkster.image")
    mask = registry.asset_decoder_for("dinkster.mask")
    merge = registry.batch_merge_for("dinkster.image")
    assert image is not None and image.provider_id == "dinkster.media-image-file@2"
    assert mask is not None and mask.provider_id == "dinkster.mask-file@2"
    assert merge is not None and merge.provider_id == "dinkster.image-batch-merge@2"
    register_media_types(registry)


def test_registered_image_asset_decoder_uses_bounded_still_contract(tmp_path: Path) -> None:
    path = tmp_path / "animated.webp"
    Image.new("RGB", (2, 2), "red").save(
        path,
        format="WEBP",
        save_all=True,
        append_images=[Image.new("RGB", (2, 2), "blue")],
        duration=100,
    )
    registry = TypeRegistry()
    register_media_types(registry)
    decoder = registry.asset_decoder_for("dinkster.image")
    assert decoder is not None
    with pytest.raises(ValueError, match="animated and multipage"):
        decoder.decode(_asset(path, "image/webp"))


def test_metadata_loader_and_image_loader_emit_same_canonical_document(tmp_path: Path) -> None:
    info = PngInfo()
    info.add_text("prompt", "{}")
    path = tmp_path / "same.png"
    Image.new("RGB", (1, 1)).save(path, pnginfo=info)
    ref = _asset(path)
    loaded = cast(str, LoadImage.execute(image=ref)["metadata"])
    inspected = cast(str, ReadImageMetadata.execute(image=ref)["metadata"])
    assert loaded == inspected
    assert (
        json.dumps(json.loads(loaded), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        == loaded
    )
