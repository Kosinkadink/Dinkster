"""MiniMax H3 AV frame boundaries and ordinary pack execution."""

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_STRING,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    MultiStreamLatent,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.minimax_h3_latent import EmptyMiniMaxH3LatentAV, MiniMaxH3ImageToVideo

from dinkster.compose import PackSpec, ServingComposer


def _assert_av(latent: Any, video_frames: int, audio_frames: int) -> None:
    assert set(latent) == {"samples"}
    samples = latent["samples"]
    assert type(samples) is MultiStreamLatent
    video = samples.by_role("video")
    audio = samples.by_role("audio")
    assert video.shape == (1, 24, video_frames, 4, 6)
    assert audio.shape == (1, 32, 2, audio_frames)
    for tensor in (video, audio):
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"
        assert torch.count_nonzero(tensor).item() == 0


@pytest.mark.parametrize(
    "length,video_frames,audio_frames",
    (
        (4, 2, 8),
        (5, 2, 8),
        (6, 7, 37),
        (21, 7, 37),
        (22, 7, 37),
        (23, 12, 65),
        (38, 12, 65),
        (39, 12, 65),
        (40, 17, 93),
        (124, 37, 207),
    ),
)
def test_empty_av_snaps_both_streams_at_frame_boundaries(
    length: int, video_frames: int, audio_frames: int
) -> None:
    latent = EmptyMiniMaxH3LatentAV.execute(width=96, height=64, length=length)["LATENT"]
    _assert_av(latent, video_frames, audio_frames)


class AVConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="minimax-empty-fixture.consumer",
            inputs=(InputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, latent: object) -> dict[str, object]:
        _assert_av(materialize_latent(latent), 12, 65)
        return cls.outputs(summary="video12-audio65-zero")


@pytest.mark.parametrize("in_process", (True,))
def test_empty_av_runs_twice_through_ordinary_pack_hosts(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(source_root))
                ),
            }
        )
        try:
            native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "minimax_empty_fixture.py").write_text(
                "from test_minimax_h3_latent_torch import AVConsumer\n"
                "from dinkster_nodes_comfy import register_inference_types\n"
                "NODES = (AVConsumer,)\n",
                encoding="ascii",
            )
            fixture = tmp_path / "dinkster-pack.toml"
            fixture.write_text(
                '[pack]\nname = "minimax-empty-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["minimax-empty-fixture"]\n\n[pack.entry]\n'
                'nodes = "minimax_empty_fixture:NODES"\n'
                'types = "minimax_empty_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "minimax-empty-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "empty": GraphNode(
                        "EmptyMiniMaxH3LatentAV", {"width": 96, "height": 64, "length": 23}
                    ),
                    "consumer": GraphNode(
                        "minimax-empty-fixture.consumer", {"latent": Link("empty", "LATENT")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == "video12-audio65-zero"
        finally:
            await composer.close()

    asyncio.run(scenario())


class MiniMaxClip:
    def __init__(self) -> None:
        self.hidden = torch.arange(6, dtype=torch.float64).reshape(1, 2, 3)
        self.conditioning: list[list[Any]] = [[self.hidden, {"schedule": (0.2, 0.7)}]]

    def tokenize(self, prompt: str, *, images: list[torch.Tensor]) -> dict[str, object]:
        return {"prompt": prompt, "images": images}

    def encode_from_tokens_scheduled(self, tokens: dict[str, Any]) -> list[list[Any]]:
        assert tokens["prompt"] == "camera pans left"
        self.conditioning[0][1]["images"] = tokens["images"]
        return self.conditioning


class MiniMaxEncoder:
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return pixels[:, ::16, ::16].permute(0, 3, 1, 2).unsqueeze(2) + 0.25


def _keyframe_image() -> torch.Tensor:
    image = torch.zeros((2, 64, 192, 4), dtype=torch.float64)
    image[0, :, :48, 0] = 1
    image[0, :, 48:144, 1] = 1
    image[0, :, 144:, 2] = 1
    image[0, :, :, 3] = 0.125
    image[1] = 0.5
    return image


def _check_image_conditioning(
    conditioning: list[list[Any]],
    latent: object,
    first: bool,
    last: bool,
    frame_count: int,
    image_dtype: torch.dtype = torch.float64,
) -> None:
    _assert_av(
        materialize_latent(latent), 7 if frame_count == 22 else 12, 37 if frame_count == 22 else 65
    )
    hidden, metadata = conditioning[0]
    assert torch.equal(hidden, torch.arange(6, dtype=torch.float64).reshape(1, 2, 3))
    assert metadata["schedule"] == (0.2, 0.7)
    images = metadata["images"]
    assert len(images) == int(first) + int(last)
    if not images:
        assert "minimax_keyframes" not in metadata
        return
    keyframes = metadata["minimax_keyframes"]
    assert [item["resolved_frame_index"] for item in keyframes] == (
        ([0] if first else []) + ([frame_count - 1] if last else [])
    )
    for index, (pixels, keyframe) in enumerate(zip(images, keyframes, strict=True)):
        pixels = materialize_image(pixels)
        assert pixels.shape == (1, 64, 96, 3)
        assert pixels.dtype == image_dtype
        assert set(keyframe) == {"resolved_frame_index", "latent"}
        encoded = keyframe["latent"]
        assert encoded.shape == (1, 3, 1, 4, 6)
        if first and index == 0:
            # Away from Lanczos transition bands the stretched thirds are exact colors.
            assert pixels[0, 16, 4].tolist() == [1, 0, 0]
            assert pixels[0, 16, 48].tolist() == [0, 1, 0]
            assert pixels[0, 16, 90].tolist() == [0, 0, 1]
            assert encoded[0, :, 0, 1, 0].tolist() == [1.25, 0.25, 0.25]
        else:
            expected = torch.zeros_like(pixels)
            expected[..., 1] = 1
            assert torch.equal(pixels, expected)
            assert encoded[0, :, 0, 1, 0].tolist() == [0.25, 1.25, 0.25]


@pytest.mark.parametrize(
    "first,last,length,frame_count",
    (
        (False, False, 6, 22),
        (True, False, 6, 22),
        (False, True, 23, 39),
        (True, True, 23, 39),
    ),
)
def test_minimax_image_keyframes_preserve_sources_and_temporal_alignment(
    first: bool, last: bool, length: int, frame_count: int
) -> None:
    clip = MiniMaxClip()
    image = _keyframe_image()
    before = image.clone()
    result = MiniMaxH3ImageToVideo.execute(
        clip=clip,
        vae=MiniMaxEncoder(),
        prompt="camera pans left",
        width=96,
        height=64,
        length=length,
        first_frame=image if first else None,
        last_frame=image if last else None,
    )
    conditioning: Any = result["positive"]
    _check_image_conditioning(conditioning, result["LATENT"], first, last, frame_count)
    assert conditioning[0][0] is clip.hidden
    if first or last:
        assert conditioning is not clip.conditioning
        assert "minimax_keyframes" not in clip.conditioning[0][1]
    else:
        assert conditioning is clip.conditioning
    assert torch.equal(image, before)


class MiniMaxImageSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="minimax-image-fixture.source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(kind))
                for name, kind in (
                    ("clip", CLIP_TYPE),
                    ("vae", VAE_TYPE),
                    ("image", IMAGE_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(clip=MiniMaxClip(), vae=MiniMaxEncoder(), image=_keyframe_image())


class MiniMaxImageConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="minimax-image-fixture.consumer",
            inputs=(
                InputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, conditioning: Any, latent: object) -> dict[str, object]:
        _check_image_conditioning(conditioning, latent, True, True, 39, torch.float32)
        return cls.outputs(summary="minimax-first-last-verified")


@pytest.mark.parametrize("in_process", (True,))
def test_minimax_image_runs_twice_through_ordinary_pack_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        (tmp_path / "minimax_image_fixture.py").write_text(
            "from test_minimax_h3_latent_torch import MiniMaxImageSource, MiniMaxImageConsumer\n"
            "from dinkster_api.v1 import register_inference_types\n"
            "NODES = (MiniMaxImageSource, MiniMaxImageConsumer)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="minimax-image-fixture"\npublisher="dinkster"\n'
            'namespaces=["minimax-image-fixture"]\n[pack.entry]\n'
            'nodes="minimax_image_fixture:NODES"\n'
            'types="minimax_image_fixture:register_inference_types"\n',
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
                        worker_group=None if in_process else "minimax-image-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("minimax-image-fixture.source", {}),
                    "condition": GraphNode(
                        "MiniMaxH3ImageToVideo",
                        {
                            "clip": Link("source", "clip"),
                            "vae": Link("source", "vae"),
                            "first_frame": Link("source", "image"),
                            "last_frame": Link("source", "image"),
                            "prompt": "camera pans left",
                            "width": 96,
                            "height": 64,
                            "length": 23,
                        },
                    ),
                    "consumer": GraphNode(
                        "minimax-image-fixture.consumer",
                        {
                            "conditioning": Link("condition", "positive"),
                            "latent": Link("condition", "LATENT"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert (
                    result.outputs["consumer"]["summary"].resolve() == "minimax-first-last-verified"
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
