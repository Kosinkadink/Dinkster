"""CPU image file input and output behavior."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dinkster_api.v1 import AssetError, AssetRef
from dinkster_assets import digest_file
from dinkster_nodes_comfy import image_file_tools
from dinkster_nodes_comfy.image_file_tools import (
    LoadImage,
    LoadImageMask,
    PreviewImage,
    SaveAnimatedPNG,
    SaveAnimatedWEBP,
    SaveImage,
)
from PIL import Image


@pytest.mark.parametrize("format_name", ("PNG", "GIF"))
def test_load_palette_image_preserves_transparency_mask(tmp_path: Path, format_name: str) -> None:
    path = tmp_path / f"palette.{format_name.lower()}"
    image = Image.fromarray(np.array([[0, 1, 2], [2, 0, 1]], dtype=np.uint8)).convert("P")
    image.putpalette([11, 43, 97, 191, 7, 53, 29, 227, 83] + [0] * (256 * 3 - 9))
    image.save(path, format=format_name, transparency=0)
    asset = AssetRef(
        digest_file(path),
        path.name,
        path.stat().st_size,
        resolver=SimpleNamespace(resolve=lambda _digest: path),
    )

    result = LoadImage.execute(image=asset)
    expected_mask = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    expected_image = (
        torch.tensor(
            [
                [
                    [[11, 43, 97], [191, 7, 53], [29, 227, 83]],
                    [[29, 227, 83], [11, 43, 97], [191, 7, 53]],
                ]
            ],
            dtype=torch.float32,
        )
        / 255.0
    )
    torch.testing.assert_close(result["IMAGE"], expected_image, rtol=0, atol=0)
    torch.testing.assert_close(result["MASK"], expected_mask, rtol=0, atol=0)
    torch.testing.assert_close(
        LoadImageMask.execute(image=asset, channel="alpha")["MASK"],
        expected_mask,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("node", (SaveImage, SaveAnimatedPNG, SaveAnimatedWEBP))
@pytest.mark.parametrize(
    ("shape", "reason"),
    (
        ((1, 1, 16_385, 3), "dimensions"),
        ((4097, 1, 1, 3), "frame limit"),
        ((9, 2048, 2048, 4), "working-set limit"),
    ),
)
def test_image_savers_reject_oversized_batches_before_materialization(node, shape, reason) -> None:
    images = torch.empty(shape, device="meta")
    options = (
        {"fps": 7.0, "lossless": True, "quality": 85, "method": "default"}
        if node is SaveAnimatedWEBP
        else {}
    )
    with pytest.raises(ValueError, match=reason):
        node.execute(images=images, filename_prefix="unused", **options)


def test_image_loader_checks_input_size_and_cumulative_decoded_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "frames.gif"
    first = Image.new("RGBA", (3, 2), (17, 83, 211, 255))
    second = Image.new("RGBA", (3, 2), (233, 29, 71, 255))
    first.save(path, save_all=True, append_images=[second], transparency=0)
    asset = AssetRef(
        digest_file(path),
        path.name,
        path.stat().st_size,
        resolver=SimpleNamespace(resolve=lambda _digest: path),
    )
    monkeypatch.setattr(image_file_tools, "IMAGE_WORKING_SET_LIMIT_BYTES", asset.size - 1)
    with pytest.raises(ValueError, match="input byte limit"):
        LoadImage.execute(image=asset)
    # Each RGBA frame yields six RGB pixels and six float32 mask values: 96 bytes.
    monkeypatch.setattr(image_file_tools, "IMAGE_WORKING_SET_LIMIT_BYTES", 192)
    assert LoadImage.execute(image=asset)["IMAGE"].shape == (2, 2, 3, 3)
    monkeypatch.setattr(image_file_tools, "IMAGE_WORKING_SET_LIMIT_BYTES", 191)
    with pytest.raises(ValueError, match="decoded image batch"):
        LoadImage.execute(image=asset)


@pytest.mark.parametrize("node", (SaveImage, SaveAnimatedPNG, SaveAnimatedWEBP))
def test_image_savers_enforce_encoded_output_limit_without_publishing_partial_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, node
) -> None:
    default, _ = _mount(tmp_path, monkeypatch)
    images = torch.tensor([[[[0.13, 0.71, 0.29], [0.89, 0.37, 0.61]]]])
    options = (
        {"fps": 7.0, "lossless": True, "quality": 85, "method": "default"}
        if node is SaveAnimatedWEBP
        else {}
    )
    node.execute(images=images, filename_prefix="first", **options)
    encoded_size = next(default.glob("first_*")).stat().st_size
    monkeypatch.setattr(image_file_tools, "ENCODED_MEDIA_LIMIT_BYTES", encoded_size)
    node.execute(images=images, filename_prefix="at-limit", **options)
    monkeypatch.setattr(image_file_tools, "ENCODED_MEDIA_LIMIT_BYTES", encoded_size - 1)
    with pytest.raises((ValueError, AssetError), match="limit"):
        node.execute(images=images, filename_prefix="over-limit", **options)
    assert not list(default.glob("over-limit*"))
    assert not list(default.glob(".dinkster-save-*"))


def _mount(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    default = tmp_path / "default"
    alternate = tmp_path / "alternate"
    default.mkdir()
    alternate.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "mounts": [
                    {"id": "default", "root": str(default), "mode": "readwrite"},
                    {"id": "alternate", "root": str(alternate), "mode": "readwrite"},
                ],
                "outputMount": "default",
            }
        ),
        encoding="ascii",
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    return default, alternate


def test_save_image_writes_each_cpu_batch_item_and_preserves_source(
    tmp_path: Path, monkeypatch
) -> None:
    default, alternate = _mount(tmp_path, monkeypatch)
    images = torch.tensor(
        [
            [[[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]]],
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        ],
        dtype=torch.float32,
    )

    default_result = SaveImage.execute(images=images, filename_prefix="default/%width%x%height%")
    assert default_result["images"] is images
    assert (default / "default" / "2x1_00001.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (default / "default" / "2x1_00002.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    chosen = images[:1]
    chosen_result = SaveImage.execute(images=chosen, filename_prefix="chosen", mount="alternate")
    assert chosen_result["images"] is chosen
    assert (alternate / "chosen_00001.png").is_file()
    assert torch.equal(images[0, 0, 0], torch.tensor([0.0, 0.5, 1.0]))


def test_save_image_uses_comfyui_truncation_and_batch_number_prefix(
    tmp_path: Path, monkeypatch
) -> None:
    default, _ = _mount(tmp_path, monkeypatch)
    images = torch.tensor(
        [
            [[[0.0, 0.5, 1.0], [0.501, 0.999, 0.001]]],
            [[[0.25, 0.75, 0.5], [0.1, 0.9, 0.4]]],
        ],
        dtype=torch.float32,
    )

    SaveImage.execute(images=images, filename_prefix="batch/item-%batch_num%")

    with Image.open(default / "batch" / "item-0_00001.png") as saved:
        assert list(saved.getdata()) == [(0, 127, 255), (127, 254, 0)]
    assert (default / "batch" / "item-1_00001.png").is_file()


def test_preview_image_preserves_cpu_source() -> None:
    images = torch.zeros((1, 2, 3, 3), dtype=torch.float32)

    assert PreviewImage.execute(images=images)["images"] is images


@pytest.mark.parametrize("format_name", ("png", "webp"))
@pytest.mark.parametrize("count", (1, 2))
def test_animated_savers_preserve_frame_order_alpha_timing_and_dimension_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, format_name: str, count: int
) -> None:
    default, alternate = _mount(tmp_path, monkeypatch)
    frames = torch.tensor(
        [
            [[[0.5, 0.25, 0.75, 1.0], [1.25, -0.5, 0.1, 0.5], [0.9, 0.2, 0.4, 0.75]]],
            [[[0.0, 1.0, 0.5, 0.25], [0.1, 0.3, 0.7, 1.0], [0.6, 0.8, 0.2, 0.5]]],
        ],
        dtype=torch.float32,
    )[:count]
    original = frames.clone()
    options = (
        {"compress_level": 3}
        if format_name == "png"
        else {"lossless": True, "quality": 93, "method": "slowest"}
    )
    node = SaveAnimatedPNG if format_name == "png" else SaveAnimatedWEBP
    assert (
        node.execute(
            images=frames,
            filename_prefix="animated/%width%x%height%",
            fps=7.0,
            mount="alternate",
            **options,
        )["images"]
        is frames
    )
    expected = np.array(
        [
            [[[127, 63, 191, 255], [255, 0, 25, 127], [229, 51, 102, 191]]],
            [[[0, 255, 127, 63], [25, 76, 178, 255], [153, 204, 51, 127]]],
        ],
        dtype=np.uint8,
    )[:count]
    with Image.open(alternate / f"animated/3x1_00001.{format_name}") as saved:
        assert saved.n_frames == count
        if format_name == "webp":
            assert saved.info["exif"]
            assert dict(saved.getexif()) == {}
        for index in range(count):
            saved.seek(index)
            np.testing.assert_array_equal(np.asarray(saved.convert("RGBA")), expected[index])
            if count > 1:
                assert saved.info["duration"] == 142
    assert not list(default.rglob(f"*.{format_name}"))
    torch.testing.assert_close(frames, original, rtol=0, atol=0)
