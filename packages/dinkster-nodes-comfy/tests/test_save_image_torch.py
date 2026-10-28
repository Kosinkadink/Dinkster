"""CPU image file input and output behavior."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from av.error import ArgumentError
from dinkster_api.v1 import (
    CORE_STRING,
    IMAGE_TYPE,
    MASK_TYPE,
    AssetError,
    AssetRef,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
)
from dinkster_assets import MountDef, MountTable, digest_file
from dinkster_graph import Graph, GraphNode, Link
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

from dinkster.compose import PackSpec, ServingComposer


def test_image_loader_preserves_sixteen_bit_samples(tmp_path: Path) -> None:
    samples = np.linspace(0, 65535, 64, dtype=np.uint16)[None]
    path = tmp_path / "sixteen-bit.png"
    Image.fromarray(samples).save(path)
    asset = AssetRef(
        digest_file(path),
        path.name,
        path.stat().st_size,
        resolver=SimpleNamespace(resolve=lambda _digest: path),
    )
    loaded = LoadImage.execute(image=asset)
    expected = np.repeat(samples[None, ..., None], 3, axis=-1).astype(np.float32) / 65535
    np.testing.assert_allclose(loaded["IMAGE"].numpy(), expected, rtol=0, atol=2 / 65535)
    assert torch.equal(loaded["MASK"], torch.zeros((1, 64, 64)))


@pytest.mark.parametrize("format_name", ("PNG", "WEBP"))
def test_image_loader_retains_animation_frames_and_alpha(tmp_path: Path, format_name: str) -> None:
    arrays = [
        np.full((2, 3, 4), color, np.uint8) for color in ((17, 83, 211, 61), (233, 29, 71, 197))
    ]
    frames = [Image.fromarray(array) for array in arrays]
    path = tmp_path / f"frames.{format_name.lower()}"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=70, lossless=True)
    asset = AssetRef(
        digest_file(path),
        path.name,
        path.stat().st_size,
        resolver=SimpleNamespace(resolve=lambda _digest: path),
    )
    loaded = LoadImage.execute(image=asset)
    expected = np.stack(arrays).astype(np.float32) / 255
    np.testing.assert_array_equal(loaded["IMAGE"].numpy(), expected[..., :3])
    np.testing.assert_array_equal(loaded["MASK"].numpy(), 1 - expected[..., 3])


@pytest.mark.parametrize("format_name", ("PNG", "GIF"))
def test_load_palette_image_preserves_source_result(tmp_path: Path, format_name: str) -> None:
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

    if format_name == "GIF":
        with pytest.raises(ArgumentError) as failure:
            LoadImage.execute(image=asset)
        assert failure.value.errno == 22
        return

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
    path = tmp_path / "frames.png"
    frames = [Image.new("RGBA", (32, 2), (index, 83, 211, 127)) for index in range(64)]
    frames[0].save(path, save_all=True, append_images=frames[1:])
    asset = AssetRef(
        digest_file(path),
        path.name,
        path.stat().st_size,
        resolver=SimpleNamespace(resolve=lambda _digest: path),
    )
    monkeypatch.setattr(image_file_tools, "IMAGE_WORKING_SET_LIMIT_BYTES", asset.size - 1)
    with pytest.raises(ValueError, match="input byte limit"):
        LoadImage.execute(image=asset)
    # RGBA backing arrays, inverted masks and stack copies, scratch and decoder allowance.
    monkeypatch.setattr(image_file_tools, "IMAGE_WORKING_SET_LIMIT_BYTES", 200_704)
    assert LoadImage.execute(image=asset)["IMAGE"].shape == (64, 2, 32, 3)
    monkeypatch.setattr(image_file_tools, "IMAGE_WORKING_SET_LIMIT_BYTES", 200_703)
    with pytest.raises(ValueError, match="decoded image batch"):
        LoadImage.execute(image=asset)


@pytest.mark.parametrize("node", (SaveImage, SaveAnimatedPNG, SaveAnimatedWEBP))
def test_image_savers_budget_live_conversion_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, node
) -> None:
    default, _ = _mount(tmp_path, monkeypatch)
    images = torch.linspace(0.1, 0.9, 48).reshape(2, 2, 3, 4)
    options = (
        {"fps": 7.0, "lossless": True, "quality": 85, "method": "default"}
        if node is SaveAnimatedWEBP
        else {}
    )
    # 192 source bytes plus float temporaries, byte frames and encoding scratch.
    monkeypatch.setattr(image_file_tools, "IMAGE_WORKING_SET_LIMIT_BYTES", 1152)
    node.execute(images=images, filename_prefix="at-limit", **options)
    assert list(default.glob("at-limit_*"))
    monkeypatch.setattr(image_file_tools, "IMAGE_WORKING_SET_LIMIT_BYTES", 1151)
    with pytest.raises(ValueError, match="working-set limit"):
        node.execute(images=images, filename_prefix="over-limit", **options)
    assert not list(default.glob("over-limit*"))


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


@pytest.mark.parametrize("prefix", ("item", "item-%batch_num%"))
def test_save_image_preserves_batch_counter_and_filename_tail(tmp_path: Path, monkeypatch, prefix):
    default, _ = _mount(tmp_path, monkeypatch)
    folder = default / "batch"
    folder.mkdir()
    existing = folder / f"{prefix}_00007_.png"
    existing.write_bytes(b"existing")
    images = torch.tensor([[[[0.1, 0.5, 0.9]]], [[[0.8, 0.3, 0.2]]]])
    before = images.clone()

    assert SaveImage.execute(images=images, filename_prefix=f"batch/{prefix}")["images"] is images
    assert SaveImage.execute(images=images, filename_prefix=f"batch/{prefix}")["images"] is images

    counters = (8, 9, 10, 11) if prefix == "item" else (8, 9, 9, 10)
    for position, counter in enumerate(counters):
        index = position % 2
        stem = prefix.replace("%batch_num%", str(index))
        with Image.open(folder / f"{stem}_{counter:05d}_.png") as saved:
            assert list(saved.getdata()) == [[(25, 127, 229)], [(204, 76, 51)]][index]
    assert existing.read_bytes() == b"existing"
    assert len(list(folder.glob("*.png"))) == 5
    assert torch.equal(images, before)


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
    assert (default / "default" / "2x1_00001_.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (default / "default" / "2x1_00002_.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    chosen = images[:1]
    chosen_result = SaveImage.execute(images=chosen, filename_prefix="chosen", mount="alternate")
    assert chosen_result["images"] is chosen
    assert (alternate / "chosen_00001_.png").is_file()
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

    with Image.open(default / "batch" / "item-0_00001_.png") as saved:
        assert list(saved.getdata()) == [(0, 127, 255), (127, 254, 0)]
    assert (default / "batch" / "item-1_00002_.png").is_file()


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


class _FileHostInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "image-file-fixture.inspect",
            inputs=(
                *(
                    InputSpec(name, TypeExpr.concrete(IMAGE_TYPE))
                    for name in ("loaded", "output_loaded", "saved", "preview", "png", "webp")
                ),
                InputSpec("mask", TypeExpr.concrete(MASK_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, mask, **images):
        expected = (
            torch.tensor(
                [
                    [
                        [[0, 51, 255], [102, 204, 51], [255, 0, 153]],
                        [[204, 102, 0], [51, 255, 204], [153, 51, 102]],
                    ]
                ],
                dtype=torch.float32,
            )
            / 255.0
        )
        for image in images.values():
            torch.testing.assert_close(materialize_image(image), expected, rtol=0, atol=0)
        torch.testing.assert_close(
            materialize_image(mask), torch.zeros((1, 64, 64)), rtol=0, atol=0
        )
        return cls.outputs(summary="image files crossed the ordinary host")


@pytest.mark.parametrize("in_process", (True, False))
@pytest.mark.parametrize("verified", (False, True))
def test_image_file_nodes_execute_twice_through_unchanged_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool, verified: bool
) -> None:
    source_pixels = np.array(
        [
            [[0, 51, 255], [102, 204, 51], [255, 0, 153]],
            [[204, 102, 0], [51, 255, 204], [153, 51, 102]],
        ],
        dtype=np.uint8,
    )
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    Image.fromarray(source_pixels).save(input_dir / "source.png")
    Image.fromarray(source_pixels).save(output_dir / "seed.png")
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot, index_root=tmp_path / "index")
    table.add(MountDef("input", input_dir), kind="media/image")
    table.add(MountDef("output", output_dir, mode="readwrite"), kind="media/image")
    table.select_output_mount("output")
    if verified:
        table.scan("input")
        table.scan("output")
    source = table.bind_name("source.png", "media/image")
    output_source = table.bind_name("seed.png", "media/image")
    if not verified:
        table.scan("output")
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))

    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        native_manifest = package / "dinkster-pack.toml"
        module = f"image_file_host_fixture_{tmp_path.name.replace('-', '_')}"
        (tmp_path / f"{module}.py").write_text(
            "from test_save_image_torch import _FileHostInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (_FileHostInspect,)\n",
            encoding="ascii",
        )
        fixture_manifest = tmp_path / "dinkster-pack.toml"
        fixture_manifest.write_text(
            '[pack]\nname = "image-file-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["image-file-fixture"]\n[pack.entry]\n'
            f'nodes = "{module}:NODES"\ntypes = "{module}:register_inference_types"\n',
            encoding="ascii",
        )
        paths = (str(tmp_path), str(Path(__file__).parent), *(p for p in sys.path if p))
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "DINKSTER_MOUNTS_SNAPSHOT": str(snapshot),
                "PYTHONPATH": os.pathsep.join(paths),
            }
        )
        try:
            manifests = (native_manifest, fixture_manifest)
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "image-file-fixture",
                        group_manifests=() if in_process else manifests,
                    )
                )
            nodes = {
                "load": GraphNode("LoadImage", {"image": source.to_wire()}),
                "load_output": GraphNode("LoadImageOutput", {"image": output_source.to_wire()}),
                "mask": GraphNode("LoadImageMask", {"image": source.to_wire(), "channel": "alpha"}),
                "save": GraphNode(
                    "SaveImage", {"images": Link("load", "IMAGE"), "filename_prefix": "saved"}
                ),
                "preview": GraphNode("PreviewImage", {"images": Link("load", "IMAGE")}),
                "png": GraphNode(
                    "SaveAnimatedPNG",
                    {
                        "images": Link("load", "IMAGE"),
                        "filename_prefix": "animated-png",
                        "fps": 7.0,
                        "compress_level": 3,
                    },
                ),
                "webp": GraphNode(
                    "SaveAnimatedWEBP",
                    {
                        "images": Link("load", "IMAGE"),
                        "filename_prefix": "animated-webp",
                        "fps": 7.0,
                        "lossless": True,
                        "quality": 93,
                        "method": "slowest",
                    },
                ),
            }
            nodes["inspect"] = GraphNode(
                "image-file-fixture.inspect",
                {
                    "loaded": Link("load", "IMAGE"),
                    "output_loaded": Link("load_output", "IMAGE"),
                    "mask": Link("mask", "MASK"),
                    "saved": Link("save", "images"),
                    "preview": Link("preview", "images"),
                    "png": Link("png", "images"),
                    "webp": Link("webp", "images"),
                },
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(Graph(nodes=nodes), ["inspect"])
                assert (
                    result.outputs["inspect"]["summary"].resolve()
                    == "image files crossed the ordinary host"
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
    for prefix, suffix in (("saved", "png"), ("animated-png", "png"), ("animated-webp", "webp")):
        saved = sorted(output_dir.glob(f"{prefix}_*.{suffix}"))
        assert len(saved) == 2
        for path in saved:
            with Image.open(path) as image:
                np.testing.assert_array_equal(np.asarray(image.convert("RGB")), source_pixels)
