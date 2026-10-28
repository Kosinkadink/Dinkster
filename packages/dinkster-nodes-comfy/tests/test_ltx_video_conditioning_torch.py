"""CPU contracts for native LTXV video conditioning nodes."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_STRING,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.ltx_video_conditioning import (
    LTXVConditioning,
    LTXVCropGuides,
    LTXVImgToVideo,
    LTXVImgToVideoInplace,
)

from dinkster.compose import PackSpec, ServingComposer

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)


class VAE:
    downscale_index_formula = (8, 32, 32)

    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        self.calls.append(pixels.clone())
        return pixels[:, 0, 0, 0].reshape(1, 1, -1, 1, 1).repeat(1, 128, 1, 1, 1)


def _image(frames: int = 3, height: int = 32, width: int = 32) -> torch.Tensor:
    return torch.stack(
        [
            torch.full((height, width, 4), float(frame), dtype=torch.float64)
            for frame in range(frames)
        ]
    )


def test_img_to_video_encodes_all_frames_and_preserves_conditioning() -> None:
    vae = VAE()
    positive, negative = [["positive", {"keep": 1}]], [["negative", {"keep": 2}]]
    output = LTXVImgToVideo.execute(
        positive=positive,
        negative=negative,
        image=_image(),
        vae=vae,
        width=32,
        height=32,
        length=17,
        batch_size=2,
        strength=0.25,
    )
    latent = output["latent"]
    assert output["positive"] is positive
    assert output["negative"] is negative
    assert vae.calls[0].shape == (3, 32, 32, 3)
    assert torch.equal(vae.calls[0][:, 0, 0, 0], torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64))
    assert latent["samples"].dtype is torch.get_default_dtype()
    assert latent["samples"].shape == (2, 128, 3, 1, 1)
    assert torch.equal(latent["samples"][0, 0, :, 0, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert latent["noise_mask"].dtype is torch.float32
    assert torch.equal(latent["noise_mask"][:, :, :, 0, 0], torch.full((2, 1, 3), 0.75))


@torch.no_grad()
def test_img_to_video_strength_extremes() -> None:
    for strength, expected in ((0.0, 1.0), (1.0, 0.0)):
        latent = LTXVImgToVideo.execute(
            positive=[],
            negative=[],
            image=_image(1),
            vae=VAE(),
            width=32,
            height=32,
            length=17,
            batch_size=1,
            strength=strength,
        )["latent"]
        assert torch.equal(latent["noise_mask"][:, :, :1], torch.full((1, 1, 1, 1, 1), expected))
        assert torch.equal(latent["noise_mask"][:, :, 1:], torch.ones((1, 1, 2, 1, 1)))


def test_inplace_bypass_returns_identity_without_touching_vae() -> None:
    vae = VAE()
    latent = {"samples": torch.ones((1, 128, 3, 1, 1)), "metadata": {"ignored": True}}
    output = LTXVImgToVideoInplace.execute(
        vae=vae, image=_image(), latent=latent, strength=0.5, bypass=True
    )
    assert output["latent"] is latent
    assert vae.calls == []


def test_inplace_clones_noise_mask_and_discards_unrelated_metadata() -> None:
    vae = VAE()
    original_mask = torch.full((1, 1, 3, 1, 1), 0.4, dtype=torch.float64)
    source = {
        "samples": torch.full((1, 128, 3, 1, 1), 9.0),
        "noise_mask": original_mask,
        "extra": 4,
    }
    output = LTXVImgToVideoInplace.execute(vae=vae, image=_image(2), latent=source, strength=0.5)[
        "latent"
    ]
    assert set(output) == {"samples", "noise_mask"}
    assert output["samples"] is not source["samples"]
    assert output["noise_mask"] is not original_mask
    assert torch.equal(original_mask, torch.full_like(original_mask, 0.4))
    assert torch.equal(output["samples"][0, 0, :, 0, 0], torch.tensor([0.0, 1.0, 9.0]))
    assert torch.equal(
        output["noise_mask"][0, 0, :, 0, 0], torch.tensor([0.5, 0.5, 0.4], dtype=torch.float64)
    )


def test_inplace_creates_float32_noise_mask_and_conditioning_preserves_inputs() -> None:
    source = {"samples": torch.zeros((1, 128, 3, 1, 1))}
    latent = LTXVImgToVideoInplace.execute(
        vae=VAE(), image=_image(1), latent=source, strength=0.25
    )["latent"]
    assert latent["noise_mask"].dtype is torch.float32
    assert torch.equal(latent["noise_mask"][0, 0, :, 0, 0], torch.tensor([0.75, 1.0, 1.0]))
    positive, negative = [["a", {"old": 1}]], [["b", {"old": 2}]]
    output = LTXVConditioning.execute(positive=positive, negative=negative, frame_rate=23.976)
    assert positive == [["a", {"old": 1}]]
    assert negative == [["b", {"old": 2}]]
    assert output["positive"] == [["a", {"old": 1, "frame_rate": 23.976}]]
    assert output["negative"] == [["b", {"old": 2, "frame_rate": 23.976}]]


class LTXSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ltx-video-conditioning-fixture.source",
            outputs=(
                OutputSpec("vae", _VAE),
                OutputSpec("image", _IMAGE),
                OutputSpec("positive", _CONDITIONING),
                OutputSpec("negative", _CONDITIONING),
                OutputSpec("latent", _LATENT),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            vae=VAE(),
            image=_image(),
            positive=[["positive", {"keep": "positive"}]],
            negative=[["negative", {"keep": "negative"}]],
            latent={"samples": torch.full((1, 128, 3, 1, 1), 7.0)},
        )


class LTXConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ltx-video-conditioning-fixture.consumer",
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("latent", _LATENT),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, positive: object, negative: object, latent: object) -> dict[str, object]:
        value = materialize_latent(latent)
        samples, noise_mask = value["samples"], value["noise_mask"]
        assert tuple(samples.shape) == (1, 128, 3, 1, 1)
        assert samples.dtype is torch.float32
        assert samples.device.type == "cpu"
        assert torch.equal(samples[0, 0, :, 0, 0], torch.tensor([0.0, 1.0, 2.0]))
        assert noise_mask.dtype is torch.float32
        assert torch.equal(noise_mask[0, 0, :, 0, 0], torch.full((3,), 0.5))
        assert positive == [["positive", {"keep": "positive", "frame_rate": 29.97}]]
        assert negative == [["negative", {"keep": "negative", "frame_rate": 29.97}]]
        return cls.outputs(summary="ltx-video-conditioning")


@pytest.mark.parametrize("in_process", (True,))
def test_ltx_video_conditioning_executes_twice_through_both_production_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(source_root))),
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "ltx_video_conditioning_fixture.py").write_text(
                "from test_ltx_video_conditioning_torch import LTXConsumer, LTXSource\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, LTXSource, LTXConsumer)\n",
                encoding="ascii",
            )
            fixture = tmp_path / "dinkster-pack.toml"
            fixture.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["ltx-video-conditioning-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "ltx_video_conditioning_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "ltx_video_conditioning_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=fixture, in_process=in_process))
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "source": GraphNode("ltx-video-conditioning-fixture.source"),
                    "create": GraphNode(
                        "LTXVImgToVideo",
                        {
                            "positive": Link("source", "positive"),
                            "negative": Link("source", "negative"),
                            "vae": Link("source", "vae"),
                            "image": Link("source", "image"),
                            "width": 32,
                            "height": 32,
                            "length": 17,
                            "batch_size": 1,
                            "strength": 0.25,
                        },
                    ),
                    "inplace": GraphNode(
                        "LTXVImgToVideoInplace",
                        {
                            "vae": Link("source", "vae"),
                            "image": Link("source", "image"),
                            "latent": Link("create", "latent"),
                            "strength": 0.5,
                            "bypass": False,
                        },
                    ),
                    "conditioning": GraphNode(
                        "LTXVConditioning",
                        {
                            "positive": Link("source", "positive"),
                            "negative": Link("source", "negative"),
                            "frame_rate": 29.97,
                        },
                    ),
                    "consumer": GraphNode(
                        "ltx-video-conditioning-fixture.consumer",
                        {
                            "positive": Link("conditioning", "positive"),
                            "negative": Link("conditioning", "negative"),
                            "latent": Link("inplace", "latent"),
                        },
                    ),
                }
            )
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == "ltx-video-conditioning"
        finally:
            await composer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("count", (0, 2, 5))
@pytest.mark.parametrize("existing_mask", (False, True))
def test_crop_guides_preserves_sources_and_clears_only_removed_guides(
    count: int, existing_mask: bool
) -> None:
    samples = torch.arange(2 * 3 * 5 * 2 * 3, dtype=torch.float64).reshape(2, 3, 5, 2, 3)
    mask = torch.linspace(0.1, 0.9, 2 * 5 * 2 * 3).reshape(2, 1, 5, 2, 3)
    positive = [[torch.ones(1, 2, 3), {"keep": "positive"}]]
    positive.append([torch.ones(1, 2, 3), {"keyframe_idxs": torch.zeros(1, 3, count * 6, 2)}])
    negative = [[torch.zeros(1, 2, 3), {"keep": "negative", "keyframe_idxs": "untouched"}]]
    latent = {"samples": samples, "discarded": True}
    if existing_mask:
        latent["noise_mask"] = mask
    before, before_mask = samples.clone(), mask.clone()
    result = LTXVCropGuides.execute(positive=positive, negative=negative, latent=latent)
    output = result["latent"]
    assert set(output) == {"samples", "noise_mask"}
    remaining = 5 - count
    torch.testing.assert_close(output["samples"], before[:, :, :remaining], rtol=0, atol=0)
    expected_mask = before_mask if existing_mask else torch.ones(2, 1, 5, 1, 1)
    torch.testing.assert_close(
        output["noise_mask"], expected_mask[:, :, :remaining], rtol=0, atol=0
    )
    assert torch.equal(samples, before) and torch.equal(mask, before_mask)
    assert "guide_attention_entries" not in positive[0][1]
    assert negative[0][1]["keyframe_idxs"] == "untouched"
    if count:
        for name in ("positive", "negative"):
            assert all(entry[1]["keyframe_idxs"] is None for entry in result[name])
            assert all(entry[1]["guide_attention_entries"] is None for entry in result[name])
        assert result["positive"][0][1]["keep"] == "positive"
        assert result["negative"][0][1]["keep"] == "negative"
    else:
        assert result["positive"] is positive and result["negative"] is negative
    output["samples"].zero_()
    output["noise_mask"].zero_()
    assert torch.equal(samples, before) and torch.equal(mask, before_mask)


def test_crop_guides_uses_first_present_key_and_non_video_fallbacks() -> None:
    samples = torch.arange(30.0).reshape(1, 3, 5, 2)
    mask = torch.ones(1, 1, 5, 1)
    indices = torch.tensor([0.0, 0.0, 8.0, 8.0]).reshape(1, 1, 4, 1)
    positive = [
        [torch.ones(1), {"keyframe_idxs": None}],
        [torch.ones(1), {"keyframe_idxs": indices}],
    ]
    result = LTXVCropGuides.execute(
        positive=positive, negative=[], latent={"samples": samples, "noise_mask": mask}
    )
    assert result["positive"] is positive
    torch.testing.assert_close(result["latent"]["samples"], samples, rtol=0, atol=0)
    for entries, remaining in (
        (None, 3),
        ([{"latent_shape": (1, 2, 3)}, {"latent_shape": (2, 1, 1)}], 2),
    ):
        positive = [[torch.ones(1), {"keyframe_idxs": indices, "guide_attention_entries": entries}]]
        result = LTXVCropGuides.execute(
            positive=positive, negative=[], latent={"samples": samples, "noise_mask": mask}
        )
        torch.testing.assert_close(
            result["latent"]["samples"], samples[:, :, :remaining], rtol=0, atol=0
        )


class CropSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ltx-crop-guides-fixture.source",
            outputs=(
                OutputSpec("positive", _CONDITIONING),
                OutputSpec("negative", _CONDITIONING),
                OutputSpec("latent", _LATENT),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            positive=[
                [torch.ones(1, 2, 3), {"keyframe_idxs": torch.zeros(1, 3, 12, 2), "keep": 7}]
            ],
            negative=[[torch.zeros(1, 2, 3), {"keep": 9}]],
            latent={
                "samples": torch.arange(2 * 3 * 5 * 2 * 3, dtype=torch.float64).reshape(
                    2, 3, 5, 2, 3
                )
            },
        )


class CropConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ltx-crop-guides-fixture.consumer",
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("latent", _LATENT),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, positive: object, negative: object, latent: object) -> dict[str, object]:
        value = materialize_latent(latent)
        expected = torch.arange(2 * 3 * 5 * 2 * 3, dtype=torch.float64).reshape(2, 3, 5, 2, 3)[
            :, :, :3
        ]
        torch.testing.assert_close(value["samples"], expected, rtol=0, atol=0)
        torch.testing.assert_close(value["noise_mask"], torch.ones(2, 1, 3, 1, 1), rtol=0, atol=0)
        assert positive[0][1] == {"keep": 7, "keyframe_idxs": None, "guide_attention_entries": None}
        assert negative[0][1] == {"keep": 9, "keyframe_idxs": None, "guide_attention_entries": None}
        return cls.outputs(summary="ltx-crop-guides")


@pytest.mark.parametrize("in_process", (True,))
def test_crop_guides_ordinary_native_and_separate_fixture_twice(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        module = "ltx_crop_guides_fixture"
        (tmp_path / f"{module}.py").write_text(
            "from test_ltx_video_conditioning_torch import CropSource, CropConsumer\n"
            "from dinkster_api.v1 import register_inference_types\n"
            "NODES=(CropSource,CropConsumer)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="ltx-crop-guides-fixture"\npublisher="dinkster"\nnamespaces=["ltx-crop-guides-fixture"]\n'
            f'[pack.entry]\nnodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
            encoding="ascii",
        )
        native = package / "dinkster-pack.toml"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(package / "src"))
                ),
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "ltx-crop-guides-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("ltx-crop-guides-fixture.source", {}),
                    "crop": GraphNode(
                        "LTXVCropGuides",
                        {name: Link("source", name) for name in ("positive", "negative", "latent")},
                    ),
                    "consumer": GraphNode(
                        "ltx-crop-guides-fixture.consumer",
                        {name: Link("crop", name) for name in ("positive", "negative", "latent")},
                    ),
                }
            )
            for _ in range(2):
                result = await composer.composition.make_engine(lambda _event: None).run(
                    graph, ["consumer"]
                )
                assert result.outputs["consumer"]["summary"].resolve() == "ltx-crop-guides"
        finally:
            await composer.close()

    asyncio.run(scenario())
